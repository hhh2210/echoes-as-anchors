import argparse
import torch
import re
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple

# --- Model Definitions ---

class RepeatDetector(nn.Module):
    """
    A two-layer MLP for detecting repeats.
    This definition must match the one used for training in `train_repeat_mlp.py`.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Linear probe if hidden_dim is 0
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RepetitionPruningLogitsProcessor(LogitsProcessor):
    """Use an MLP probe to detect and remove repetitive content while preserving reasoning flow."""
    def __init__(
        self,
        question: str,
        tokenizer: AutoTokenizer,
        embedder: SentenceTransformer,
        mlp_probe: RepeatDetector,
        device: torch.device,
        prefix_tokens: int = 32,
        check_interval_tokens: int = 3,
        mlp_threshold: float = 0.5,
        min_think_tokens: int = 8,
        remove_strategy: str = "truncate_and_continue",  # "terminate" or "truncate_and_continue"
    ):
        self.question = question
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.mlp_probe = mlp_probe
        self.device = device
        self.prefix_tokens = prefix_tokens
        self.check_interval_tokens = check_interval_tokens
        self.mlp_threshold = mlp_threshold
        self.min_think_tokens = min_think_tokens
        self.remove_strategy = remove_strategy
        
        self.last_check_token_count = 0
        self.repetition_detected = False
        self.truncation_needed = False
        self.truncation_point = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        token_count = input_ids.shape[1]
        
        # Throttle the check for efficiency
        if token_count - self.last_check_token_count < self.check_interval_tokens:
            return scores
        self.last_check_token_count = token_count

        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract actual thinking content more precisely
        think_content = self._extract_thinking_content(decoded_text)
        if think_content is None:
            return scores  # <think> has not started yet or no valid thinking content

        # Only start checking after a minimum number of thinking tokens
        think_words = think_content.split()
        if len(think_words) < self.min_think_tokens:
            return scores
            
        # 1. Prepare features for the MLP probe
        q_embedding = self.embedder.encode(self.question, convert_to_tensor=True, device=self.device)
        
        # Use first prefix_tokens words of actual thinking content
        prefix_text = " ".join(think_words[:self.prefix_tokens])
        p_embedding = self.embedder.encode(prefix_text, convert_to_tensor=True, device=self.device)
        
        # Ensure embeddings are 2D for concatenation
        if len(q_embedding.shape) == 1: q_embedding = q_embedding.unsqueeze(0)
        if len(p_embedding.shape) == 1: p_embedding = p_embedding.unsqueeze(0)
            
        features = torch.cat([q_embedding, p_embedding], dim=1).to(self.device)

        # 2. Get prediction from MLP
        with torch.no_grad():
            logits = self.mlp_probe(features)
            # A sigmoid > 0.5 means it's likely a repeat (label 1)
            prob_is_repeat = torch.sigmoid(logits).item()

        print(f"\n[Probe Debug] Question: '{self.question[:50]}...'")
        print(f"[Probe Debug] Think content: '{prefix_text[:100]}...'")
        print(f"[Probe Debug] Prob(is_repeat)={prob_is_repeat:.2f}")

        # 3. Intervene if repetition is likely
        if prob_is_repeat > self.mlp_threshold and not self.repetition_detected:
            self.repetition_detected = True
            
            if self.remove_strategy == "terminate":
                print(f"[Probe] Detected repetition. Forcing EOS token to terminate generation.")
                # Create a score tensor filled with -inf
                new_scores = torch.full_like(scores, -float("inf"))
                # Set the score of the EOS token to a high value (e.g., 0) to force its selection
                new_scores[:, self.tokenizer.eos_token_id] = 0
                return new_scores
            
            elif self.remove_strategy == "truncate_and_continue":
                print(f"[Probe] Detected repetition. Marking for truncation and continuation.")
                print(f"[Probe] Full thinking content so far: '{think_content[:200]}...'")
                # Signal for truncation by setting a special attribute
                self.truncation_needed = True
                self.truncation_point = token_count
                # For now, continue generation normally - truncation will be handled externally
                return scores

        return scores
    
    def _extract_thinking_content(self, decoded_text: str) -> str:
        """Extract actual thinking content, filtering out prompt templates and questions."""
        
        # Find the <think> tag and everything after it
        think_match = re.search(r"<think>\s*(.*)", decoded_text, re.DOTALL)
        if not think_match:
            return None
            
        raw_think_content = think_match.group(1).strip()
        
        # If the thinking content is empty or too short, return None
        if len(raw_think_content.split()) < 3:
            return None
        
        # Remove common prompt artifacts and question repetitions
        # 1. Remove any leading "Question:" repetitions
        think_content = re.sub(r'^.*?Question:\s*.*?\?\s*', '', raw_think_content, flags=re.DOTALL)
        
        # 2. Remove any leading instruction repetitions
        think_content = re.sub(r'^.*?step by step.*?\s*', '', think_content, flags=re.IGNORECASE | re.DOTALL)
        
        # 3. Remove any leading tags or template artifacts
        think_content = re.sub(r'^.*?</think>.*?tags\.\s*', '', think_content, flags=re.DOTALL)
        think_content = re.sub(r'^.*?Answer:\s*', '', think_content, flags=re.DOTALL)
        
        # Strip and return the cleaned content
        think_content = think_content.strip()
        
        # If after cleaning there's not enough content, return None
        if len(think_content.split()) < 3:
            return None
            
        return think_content

# === Helper functions for reusable inference ===

def load_pruning_components(
    main_model_path: str,
    embedding_model_path: str,
    mlp_probe_path: str,
    mlp_hidden_dim: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, SentenceTransformer, RepeatDetector, torch.device]:
    """Convenience helper that loads *once* all objects required for pruned inference.

    Returns
    -------
    tokenizer, model, embedder, mlp_probe, device
        The fully-initialised components ready for generation.  These can be
        reused across many questions so that evaluation loops do not repeatedly
        incur loading overhead.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. HF causal LM
    tokenizer = AutoTokenizer.from_pretrained(main_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        main_model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    # 2. Sentence-Transformer embedder
    embedder = SentenceTransformer(embedding_model_path, device=device)

    # 3. MLP probe
    input_dim = embedder.get_sentence_embedding_dimension() * 2
    mlp_probe = RepeatDetector(input_dim, hidden_dim=mlp_hidden_dim).to(device)
    mlp_probe.load_state_dict(torch.load(mlp_probe_path, map_location=device))
    mlp_probe.eval()

    return tokenizer, model, embedder, mlp_probe, device


def generate_with_pruning(
    question: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    embedder: SentenceTransformer,
    mlp_probe: RepeatDetector,
    device: torch.device,
    *,
    max_new_tokens: int = 256,
    generation_kwargs: Optional[dict] = None,
    pruning_kwargs: Optional[dict] = None,
    return_unpruned: bool = False,
) -> Tuple[str, Optional[str]]:
    """Generate an answer with the repetition-pruning mechanism.

    Parameters
    ----------
    question : str
        The GSM8K-style math word problem to solve.
    tokenizer, model, embedder, mlp_probe, device
        Components produced by :func:`load_pruning_components`.
    max_new_tokens : int, default 256
        Maximum tokens to generate.
    generation_kwargs : dict, optional
        Extra kwargs forwarded to ``model.generate`` (e.g. *do_sample*, *temperature*).
    pruning_kwargs : dict, optional
        Extra kwargs forwarded to :class:`RepetitionPruningLogitsProcessor` (e.g. *mlp_threshold*).
    return_unpruned : bool, default False
        If *True*, the function will also return an un-pruned baseline generation for
        easy comparison.

    Returns
    -------
    pruned_text : str
        The decoded output after pruning.
    unpruned_text : str | None
        The baseline output without pruning if *return_unpruned* is *True*, otherwise *None*.
    """
    generation_kwargs = generation_kwargs or {}
    pruning_kwargs = pruning_kwargs or {}

    # 1. Build the prompt
    prompt_template = (
        "You are an expert at solving math problems. Please think step by step. "
        "Enclose your thinking process in <think>...</think> tags.\n\n"
        "Question: {question}\n"
        "Answer: <think>"
    )
    prompt = prompt_template.format(question=question)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 2. Prepare logits-processor
    pruning_processor = RepetitionPruningLogitsProcessor(
        question=question,
        tokenizer=tokenizer,
        embedder=embedder,
        mlp_probe=mlp_probe,
        device=device,
        **pruning_kwargs,
    )

    # 3. Generation with pruning
    remove_strategy = pruning_kwargs.get("remove_strategy", "truncate_and_continue")
    
    if remove_strategy == "truncate_and_continue":
        # Enhanced generation with truncation and continuation
        pruned_text = _generate_with_truncation(
            input_ids, model, tokenizer, pruning_processor, max_new_tokens, generation_kwargs
        )
    else:
        # Original termination strategy
        pruned_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList([pruning_processor]),
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        pruned_text = tokenizer.decode(pruned_ids[0], skip_special_tokens=True)

    unpruned_text = None
    if return_unpruned:
        baseline_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        unpruned_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)

    return pruned_text, unpruned_text


def _generate_with_truncation(
    input_ids: torch.LongTensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pruning_processor: RepetitionPruningLogitsProcessor,
    max_new_tokens: int,
    generation_kwargs: dict,
) -> str:
    """Enhanced generation that handles truncation and continuation when repetition is detected."""
    
    current_ids = input_ids.clone()
    total_generated = 0
    truncation_count = 0
    max_truncations = 3  # Prevent infinite truncation loops
    
    while total_generated < max_new_tokens and truncation_count < max_truncations:
        # Reset processor state for this generation round
        pruning_processor.truncation_needed = False
        pruning_processor.truncation_point = 0
        pruning_processor.repetition_detected = False  # Important: reset this flag
        
        # Generate with remaining token budget
        remaining_tokens = max_new_tokens - total_generated
        
        print(f"\n[Generation] Round {truncation_count + 1}, remaining tokens: {remaining_tokens}")
        
        generated_ids = model.generate(
            current_ids,
            max_new_tokens=min(remaining_tokens, 64),  # Generate in chunks to check for truncation
            logits_processor=LogitsProcessorList([pruning_processor]),
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        
        # Check if truncation is needed
        if pruning_processor.truncation_needed:
            truncation_count += 1
            print(f"\n[Truncation] Round {truncation_count}: Repetition detected, truncating and continuing...")
            
            # Decode current progress
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"[Truncation] Full text before truncation: '{full_text[-200:]}'")
            
            # Find the <think> content and remove repetitive part
            truncated_text = _remove_repetitive_content(full_text, pruning_processor.question, tokenizer)
            print(f"[Truncation] Text after truncation: '{truncated_text[-200:]}'")
            
            # Re-encode truncated text and continue generation
            current_ids = tokenizer(truncated_text, return_tensors="pt").input_ids.to(generated_ids.device)
            
            # Update token count
            new_length = current_ids.shape[1]
            original_length = input_ids.shape[1]
            total_generated = new_length - original_length
            
            print(f"[Truncation] Token count - Original: {original_length}, New: {new_length}, Generated: {total_generated}")
            
        else:
            # No truncation needed, we're done
            final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n[Generation] Completed successfully after {truncation_count} truncations")
            return final_text
    
    # If we've hit the truncation limit or token limit, return what we have
    final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    if truncation_count >= max_truncations:
        print(f"\n[Generation] Stopped after {max_truncations} truncations to avoid infinite loop")
    
    return final_text


def _remove_repetitive_content(text: str, question: str, tokenizer: AutoTokenizer) -> str:
    """Remove repetitive content from the thinking process and prepare for continuation."""
    
    # Extract the <think> content
    think_match = re.search(r"(.*<think>\s*)(.*)", text, re.DOTALL)
    if not think_match:
        return text
    
    prefix = think_match.group(1)  # Everything up to and including <think>
    think_content = think_match.group(2).strip()
    
    print(f"[Cleanup] Original think content: '{think_content[:150]}...'")
    
    # More aggressive cleanup of repetitive content
    cleaned_content = think_content
    
    # 1. Remove any direct question repetition at the start
    question_clean = re.sub(r'[^\w\s]', '', question.lower())
    question_words = set(question_clean.split())
    
    # Split content into sentences and filter
    sentences = re.split(r'[.!?]', cleaned_content)
    filtered_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_clean = re.sub(r'[^\w\s]', '', sentence.lower())
        sentence_words = set(sentence_clean.split())
        
        # Calculate overlap ratio with question
        if len(sentence_words) > 0:
            overlap = len(question_words & sentence_words) / len(sentence_words)
            # Keep sentences with low overlap and sufficient length
            if overlap < 0.6 and len(sentence_words) > 3:
                filtered_sentences.append(sentence)
    
    # 2. If we removed too much, keep some context but start fresh
    if len(filtered_sentences) == 0:
        new_think_content = "Now I need to solve this step by step."
    elif len(filtered_sentences) < 2:
        # Keep some context and add transition
        new_think_content = filtered_sentences[0] + ". Now let me continue solving this."
    else:
        # Keep the non-repetitive content and add transition
        new_think_content = '. '.join(filtered_sentences[-2:])  # Keep last 2 sentences
        if not new_think_content.endswith('.'):
            new_think_content += '.'
        new_think_content += " Let me work through this systematically."
    
    print(f"[Cleanup] Cleaned think content: '{new_think_content}'")
    
    return prefix + new_think_content


# === Harness integration ===

from lm_eval.models.huggingface import HFLM


class MLPPruningHFLM(HFLM):
    """HuggingFace LM subclass that applies the MLP pruning logic during generation."""

    def __init__(
        self,
        main_model_path: str,
        embedding_model_path: str,
        mlp_probe_path: str,
        mlp_hidden_dim: int = 32,
        prefix_tokens: int = 32,
        remove_strategy: str = "truncate_and_continue",
        **hf_kwargs,
    ) -> None:
        super().__init__(pretrained=main_model_path, **hf_kwargs)
        # Load pruning components using the loaded device from HFLM
        self.embedder = SentenceTransformer(embedding_model_path, device=self.device)
        input_dim = self.embedder.get_sentence_embedding_dimension() * 2
        self.mlp_probe = RepeatDetector(input_dim, hidden_dim=mlp_hidden_dim).to(self.device)
        self.mlp_probe.load_state_dict(torch.load(mlp_probe_path, map_location=self.device))
        self.mlp_probe.eval()
        self.prefix_tokens = prefix_tokens
        self.remove_strategy = remove_strategy

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Override generation with pruning."""
        results = []
        for inst in requests:
            prompt, gen_kwargs = inst.args
            max_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            until = gen_kwargs.get("until", [])

            # Extract question from the prompt if possible
            m = re.search(r"Question:\s*(.*)\nAnswer:", prompt, re.DOTALL)
            question = m.group(1).strip() if m else prompt

            out, _ = generate_with_pruning(
                question,
                self.tokenizer,
                self.model,
                self.embedder,
                self.mlp_probe,
                self.device,
                max_new_tokens=max_toks,
                generation_kwargs={k: v for k, v in gen_kwargs.items() if k not in {"max_gen_toks", "until"}},
                pruning_kwargs={
                    "prefix_tokens": self.prefix_tokens,
                    "remove_strategy": self.remove_strategy,
                },
                return_unpruned=False,
            )

            for term in until:
                if term:
                    out = out.split(term)[0]
            results.append(out)

        return results


def get_harness_lm(
    main_model_path: str,
    embedding_model_path: str,
    mlp_probe_path: str,
    mlp_hidden_dim: int = 32,
    prefix_tokens: int = 32,
    remove_strategy: str = "truncate_and_continue",
    **hf_kwargs,
) -> MLPPruningHFLM:
    """Convenience helper to construct :class:`MLPPruningHFLM`."""

    return MLPPruningHFLM(
        main_model_path=main_model_path,
        embedding_model_path=embedding_model_path,
        mlp_probe_path=mlp_probe_path,
        mlp_hidden_dim=mlp_hidden_dim,
        prefix_tokens=prefix_tokens,
        remove_strategy=remove_strategy,
        **hf_kwargs,
    )


def evaluate_with_harness(lm: HFLM, tasks="gsm8k", **kwargs):
    """Run lm-evaluation-harness with the provided LM."""
    import lm_eval
    from lm_eval.utils import setup_logging

    setup_logging("INFO")
    return lm_eval.simple_evaluate(model=lm, tasks=[tasks], **kwargs)

# --- Original CLI entry point retained for quick manual testing ---

def main():
    parser = argparse.ArgumentParser(description="Prune LLM inference in real-time using a trained MLP probe.")
    parser.add_argument("--main_model_path", type=str, default="/path/to/your/main_model/", help="Path to the main causal LM.")
    parser.add_argument("--embedding_model_path", type=str, default="/path/to/your/embedding_model/", help="Path to the SentenceTransformer embedding model.")
    parser.add_argument("--mlp_probe_path", type=str, default="models/repeat_mlp.pt", help="Path to the trained MLP probe (.pt file).")
    parser.add_argument("--mlp_hidden_dim", type=int, default=32, help="Hidden dimension of the MLP probe. Must match the trained model.")
    parser.add_argument("--question", type=str, default="There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", help="Question to ask the model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load all models
    print("Loading models...")
    tokenizer, model, embedder, mlp_probe, device = load_pruning_components(
        args.main_model_path,
        args.embedding_model_path,
        args.mlp_probe_path,
        args.mlp_hidden_dim,
        device,
    )
    print("All models loaded successfully.")

    # 2. Generate text with and without the pruning processor
    
    print("\n--- Generating with MLP Pruning ---")
    pruned_text, unpruned_text = generate_with_pruning(
        args.question,
        tokenizer,
        model,
        embedder,
        mlp_probe,
        device,
        max_new_tokens=256,
        return_unpruned=True,
    )
    print("\n--- Pruned Output ---\n")
    print(pruned_text)

    if unpruned_text:
        print("\n--- Unpruned Output ---\n")
        print(unpruned_text)

    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main() 


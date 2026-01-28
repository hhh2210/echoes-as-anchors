#!/usr/bin/env python3
"""
Batch or single-sample analysis of log-probability differences before and after
trimming prompt-echo/repetitive content in thinking traces.

CLI modes:
1) Batch (used by compare_trimmed_accuracy.py)
   python src/evaluation/logp_trim_experiment.py \
     --input_file path/to/converted.jsonl \
     --output_file path/to/results.json \
     --model /path/to/model

   Input JSONL format (one per line), produced by convert_lm_eval_for_logp.py:
     {"idx": int, "problem": str, "pred": [str, ...], "is_correct": bool}

   Output JSON includes:
     {"summary": {...}, "details": [{...}, ...]}

"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import argparse
import json
import math
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data_processing.mlp_pipeline.inference import _remove_repetitive_content


def sanitize_input_ids(input_ids: torch.Tensor, pad_token_id: Optional[int], vocab_size: int) -> torch.Tensor:
    """Ensure token indices are valid longs; replace invalid with PAD/EOS/0."""
    ids = input_ids.to(dtype=torch.long, copy=True)
    invalid_mask = (ids < 0) | (ids >= vocab_size)
    if invalid_mask.any():
        safe_pad = pad_token_id
        if safe_pad is None:
            safe_pad = 0
        ids = ids.masked_fill(invalid_mask, safe_pad)
    return ids


def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Gather log p(label) for each position in a batch-friendly way.

    logits: [B, T, V], input_ids (labels): [B, T]
    returns: [B, T]
    """
    log_probs = logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return gathered


def compute_per_token_logps(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str) -> torch.Tensor:
    """Return log p for each token in ``text`` (teacher forcing)."""
    ids = tokenizer(text, return_tensors="pt").input_ids
    ids = sanitize_input_ids(ids, tokenizer.pad_token_id or tokenizer.eos_token_id, tokenizer.vocab_size)
    ids = ids.to(model.device)

    with torch.no_grad():
        out = model(ids, labels=ids)
    logits = out.logits[:, :-1, :]
    labels = ids[:, 1:]
    return get_per_token_logps(logits, labels).squeeze(0)


def build_prompt(question: str) -> str:
    return (
        "You are an expert at solving math problems. Please think step by step.\n"
        f"Question: {question}\n"
        "Answer: <think>"
    )


def compute_avg_answer_logp(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, full_text: str) -> float:
    """Compute average per-token log-likelihood over answer tokens only.

    We teacher-force on the combined string. We then drop contributions from
    prompt tokens by slicing labels starting at (len(prompt_tokens) - 1).
    """
    # Token counts for slicing
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    token_logps = compute_per_token_logps(model, tokenizer, full_text)
    # Labels correspond to tokens 1..N-1; answer labels start from position prompt_len-1
    start = max(prompt_len - 1, 0)
    answer_logps = token_logps[start:]
    if answer_logps.numel() == 0:
        # Fallback: avoid division by zero
        return float(token_logps.mean().item())
    return float(answer_logps.mean().item())


def analyze_sample(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str, original_prediction: str) -> Tuple[str, float, float, float]:
    """Return (trimmed_prediction_text_only, gL_full, gL_trim, delta_full_minus_trim)."""
    prompt = build_prompt(question)
    full_text = prompt + original_prediction

    # Trim using heuristic remover (consistent with current pipeline)
    trimmed_full_text = _remove_repetitive_content(full_text, question, tokenizer)

    # Compute average log-likelihood over answer tokens only
    gL_full = compute_avg_answer_logp(model, tokenizer, prompt, full_text)
    gL_trim = compute_avg_answer_logp(model, tokenizer, prompt, trimmed_full_text)

    delta = gL_full - gL_trim  # Δ = original - trimmed (README definition)

    # For JSON output, store only the answer portion (without the prompt)
    trimmed_answer_only = trimmed_full_text[len(prompt):]
    return trimmed_answer_only, gL_full, gL_trim, delta


def run_batch(input_file: str, output_file: str, model_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    details: List[Dict[str, Any]] = []
    deltas: List[float] = []
    total = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            if limit is not None and total >= limit:
                break
            sample = json.loads(line)
            question: str = sample.get("problem", "")
            predictions = sample.get("pred", [])
            if not predictions:
                continue
            # Use the last candidate for consistency with harness logging
            original_pred: str = predictions[-1] if isinstance(predictions[-1], str) else str(predictions[-1])

            try:
                trimmed_pred, gL_full, gL_trim, delta = analyze_sample(model, tokenizer, question, original_pred)
            except Exception as e:
                # Skip problematic samples but keep analysis running
                details.append({
                    "idx": sample.get("idx", line_idx),
                    "question": question,
                    "original_prediction": original_pred,
                    "trimmed_prediction": None,
                    "logp_delta": None,
                    "error": f"{type(e).__name__}: {e}",
                    "is_correct": sample.get("is_correct"),
                })
                continue

            details.append({
                "idx": sample.get("idx", line_idx),
                "question": question,
                "original_prediction": original_pred,
                "trimmed_prediction": trimmed_pred,
                "logp_delta": delta,
                "is_correct": sample.get("is_correct"),
            })
            deltas.append(delta)
            total += 1

    # Summary stats
    if deltas:
        mean_delta = float(sum(deltas) / len(deltas))
        # Unbiased std (population std would divide by N)
        variance = float(sum((d - mean_delta) ** 2 for d in deltas) / len(deltas))
        std_delta = math.sqrt(variance)
        negative_count = sum(1 for d in deltas if d < 0)
        negative_ratio = negative_count / len(deltas)
    else:
        mean_delta = 0.0
        std_delta = 0.0
        negative_count = 0
        negative_ratio = 0.0

    result: Dict[str, Any] = {
        "summary": {
            "total_samples": total,
            "mean_logp_delta": mean_delta,
            "std_logp_delta": std_delta,
            "negative_delta_count": negative_count,
            "negative_delta_ratio": negative_ratio,
        },
        "details": details,
    }

    # Persist
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    return result


def process_question_demo(model_name: str, question: str, max_new_tokens: int = 64) -> Tuple[str, str, float]:
    """Single-sample demo that generates an answer, trims it, and reports Δ."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    prompt = build_prompt(question)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        gen_out = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    answer_only = tokenizer.decode(gen_out[0], skip_special_tokens=True)[len(prompt):]

    trimmed_pred, gL_full, gL_trim, delta = analyze_sample(model, tokenizer, question, answer_only)
    return prompt + answer_only, prompt + trimmed_pred, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute log-probability difference after trimming repeated sentences (batch mode only).")
    parser.add_argument("--input_file", type=str, required=True, help="Converted JSONL input (from convert_lm_eval_for_logp.py)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write JSON results")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HF model path/name")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples")
    args = parser.parse_args()

    run_batch(args.input_file, args.output_file, args.model, limit=args.limit)


if __name__ == "__main__":
    main()

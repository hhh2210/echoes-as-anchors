#!/usr/bin/env python3
"""
Attention computation and analysis utilities
"""

from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .model_utils import Span, token_ids, find_question_token_span_in_prompt
except ImportError:
    from model_utils import Span, token_ids, find_question_token_span_in_prompt


def collect_attentions(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Collect attention weights from all layers."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = out.attentions  # type: ignore[attr-defined]
    if attentions is None:
        raise RuntimeError("Model did not return attentions; ensure output_attentions=True is supported.")
    return [a.squeeze(0).to(torch.float32) for a in attentions]


def mean_heads(attn: torch.Tensor) -> torch.Tensor:
    """Average attention across heads."""
    return attn.mean(dim=0)


def mean_layers(attns: List[torch.Tensor]) -> torch.Tensor:
    """Average attention across layers and heads."""
    return torch.stack([mean_heads(a) for a in attns], dim=0).mean(dim=0)


def gather_mass(attn_mat: torch.Tensor, src_idx: List[int], dst_idx: List[int]) -> float:
    """Calculate average attention mass from source to destination indices."""
    if not src_idx or not dst_idx:
        return float("nan")
    src = torch.tensor(src_idx, dtype=torch.long, device=attn_mat.device)
    dst = torch.tensor(dst_idx, dtype=torch.long, device=attn_mat.device)
    mass_per_src = attn_mat.index_select(0, src).index_select(1, dst).sum(dim=1)
    return float(mass_per_src.mean().item())


def gather_mass_per_head(attn_heads: torch.Tensor, src_idx: List[int], dst_idx: List[int]) -> List[float]:
    """
    Calculate attention mass per head.
    
    Args:
        attn_heads: Tensor[H, T, T]
        src_idx: Source token indices
        dst_idx: Destination token indices
    
    Returns:
        List of length H with mean mass from src to dst for each head
    """
    H = int(attn_heads.shape[0])
    if not src_idx or not dst_idx:
        return [float("nan")] * H
    
    src = torch.tensor(src_idx, dtype=torch.long, device=attn_heads.device)
    dst = torch.tensor(dst_idx, dtype=torch.long, device=attn_heads.device)
    vals: List[float] = []
    
    for h in range(H):
        mat = attn_heads[h]
        mass_per_src = mat.index_select(0, src).index_select(1, dst).sum(dim=1)
        vals.append(float(mass_per_src.mean().item()))
    
    return vals


def compute_attention_for_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer_text: str,
    answer_prefix_tokens: int,
    removed_prefix_tokens: Optional[int],
    return_per_layer: bool = False,
    use_removed_as_prefix: bool = False,
) -> Dict[str, Any]:
    """
    Compute attention metrics for a question-answer pair.
    
    Args:
        model: The model to analyze
        tokenizer: The tokenizer
        prompt: The prompt including the question
        answer_text: The answer text
        answer_prefix_tokens: Number of tokens to consider as answer prefix
        removed_prefix_tokens: Number of removed prefix tokens (if MLP detected)
        return_per_layer: Whether to return per-layer metrics
        use_removed_as_prefix: Whether to use removed tokens count as prefix window
    
    Returns:
        Dictionary containing attention metrics
    """
    full_text = prompt + answer_text
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    p_len = len(token_ids(prompt, tokenizer, add_special_tokens=True))
    T = int(input_ids.shape[1])
    ans_len = max(0, T - p_len)

    # Extract question from prompt for span detection
    # Find the actual question in the prompt (between "Question: " and "\nAnswer:")
    import re
    match = re.search(r"Question:\s*(.*?)\s*\nAnswer:", prompt, re.DOTALL)
    question = match.group(1) if match else ""
    q_span = find_question_token_span_in_prompt(tokenizer, prompt, question)

    attns = collect_attentions(model, input_ids, attn_mask)
    last = mean_heads(attns[-1])
    mean = mean_layers(attns)

    ans_indices = list(range(p_len, T))
    q_indices = list(range(q_span.start, q_span.end)) if q_span else []
    
    # Decide prefix window length
    if use_removed_as_prefix and removed_prefix_tokens is not None and removed_prefix_tokens > 0:
        k = min(int(removed_prefix_tokens), ans_len)
    else:
        k = min(answer_prefix_tokens, ans_len)
    ans_pref = list(range(p_len, p_len + k)) if k > 0 else []

    if removed_prefix_tokens is not None and removed_prefix_tokens > 0:
        r_k = min(removed_prefix_tokens, ans_len)
        removed_pref = list(range(p_len, p_len + r_k))
        ans_tail = list(range(p_len + r_k, T))
    else:
        removed_pref = []
        ans_tail = []

    out: Dict[str, Any] = {
        "seq_len": T,
        "prompt_len": p_len,
        "answer_len": ans_len,
        "used_ans_prefix_len": k,
        "question_span": [q_span.start, q_span.end] if q_span else None,
        "last_layer": {
            "ans_to_question": gather_mass(last, ans_indices, q_indices) if q_indices else float("nan"),
            "ans_to_ans_prefix": gather_mass(last, ans_indices, ans_pref) if ans_pref else float("nan"),
            "ans_tail_to_removed_prefix": gather_mass(last, ans_tail, removed_pref) if removed_pref and ans_tail else float("nan"),
        },
        "all_layers_mean": {
            "ans_to_question": gather_mass(mean, ans_indices, q_indices) if q_indices else float("nan"),
            "ans_to_ans_prefix": gather_mass(mean, ans_indices, ans_pref) if ans_pref else float("nan"),
            "ans_tail_to_removed_prefix": gather_mass(mean, ans_tail, removed_pref) if removed_pref and ans_tail else float("nan"),
        },
    }

    if return_per_layer:
        per_q: List[float] = []
        per_pref: List[float] = []
        for a in attns:
            m = mean_heads(a)
            vq = gather_mass(m, ans_indices, q_indices) if q_indices else float("nan")
            vp = gather_mass(m, ans_indices, ans_pref) if ans_pref else float("nan")
            per_q.append(float(vq))
            per_pref.append(float(vp))
        out["per_layer"] = {
            "ans_to_question": per_q,
            "ans_to_ans_prefix": per_pref,
        }

    return out


def compute_per_head_for_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer_text: str,
    answer_prefix_tokens: int,
    removed_prefix_tokens: Optional[int],
) -> Dict[str, Any]:
    """
    Compute per-head attention masses per layer.
    
    Returns:
        Dictionary with per_head metrics organized by layer and head
    """
    full_text = prompt + answer_text
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    p_len = len(token_ids(prompt, tokenizer, add_special_tokens=True))
    T = int(input_ids.shape[1])
    ans_len = max(0, T - p_len)

    # Extract question from prompt
    import re
    match = re.search(r"Question:\s*(.*?)\s*\nAnswer:", prompt, re.DOTALL)
    question = match.group(1) if match else ""
    q_span = find_question_token_span_in_prompt(tokenizer, prompt, question)

    attns = collect_attentions(model, input_ids, attn_mask)

    ans_indices = list(range(p_len, T))
    q_indices = list(range(q_span.start, q_span.end)) if q_span else []

    if removed_prefix_tokens is not None and removed_prefix_tokens > 0:
        k = min(int(removed_prefix_tokens), ans_len)
    else:
        k = min(answer_prefix_tokens, ans_len)
    ans_pref = list(range(p_len, p_len + k)) if k > 0 else []

    per_layer_heads_q: List[List[float]] = []
    per_layer_heads_pref: List[List[float]] = []
    
    for a in attns:
        # a: [H, T, T]
        per_layer_heads_q.append(gather_mass_per_head(a, ans_indices, q_indices) if q_indices else [])
        per_layer_heads_pref.append(gather_mass_per_head(a, ans_indices, ans_pref) if ans_pref else [])

    return {
        "per_head": {
            "ans_to_question": per_layer_heads_q,
            "ans_to_ans_prefix": per_layer_heads_pref,
        },
        "used_ans_prefix_len": k,
        "seq_len": T,
    }
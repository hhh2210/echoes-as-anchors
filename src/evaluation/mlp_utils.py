#!/usr/bin/env python3
"""
MLP-based echo removal utilities
"""

from typing import Optional, Tuple
from transformers import AutoTokenizer

try:
    from .utils import extract_think, longest_common_suffix_len
    from .model_utils import token_ids
except ImportError:
    from utils import extract_think, longest_common_suffix_len
    from model_utils import token_ids


def remove_echo_with_mlp(
    question: str,
    answer_text: str,
    embed_model,
    initial_threshold: float,
    drop_threshold: float,
) -> Tuple[str, Optional[int]]:
    """
    Remove <think> prefix echo using MLP detection.
    
    Args:
        question: The question text
        answer_text: The answer text containing potential echo
        embed_model: The embedding model for similarity computation
        initial_threshold: Initial similarity threshold
        drop_threshold: Drop threshold for boundary detection
    
    Returns:
        Tuple of (cleaned answer, estimated removed token count or None)
        If token count estimation is needed, returns "__ESTIMATE_REMOVED__" marker
    """
    if embed_model is None:
        return answer_text, None
    
    try:
        try:
            from src.data_processing.mlp_pipeline.utils import find_repetition_boundary  # type: ignore
        except Exception:
            from train_mlp.utils import find_repetition_boundary  # type: ignore

        prefix, think = extract_think(answer_text)
        if think is None or not think.strip():
            return answer_text, None
        
        is_rep, prefix_len_chars = find_repetition_boundary(
            question, think, embed_model, initial_threshold, drop_threshold
        )
        
        if is_rep != 1 or prefix_len_chars <= 0 or prefix_len_chars >= len(think):
            return answer_text, None
        
        # Remove character-level prefix
        kept = think[prefix_len_chars:]
        kept = kept.lstrip(" \n\t.,;:!?-")
        cleaned = prefix + kept
        
        # Mark for token estimation via suffix alignment
        return cleaned, "__ESTIMATE_REMOVED__"
    except Exception:
        return answer_text, None


def estimate_removed_tokens_via_suffix(
    tokenizer: AutoTokenizer,
    prompt: str,
    raw_answer: str,
    cleaned_answer: str,
) -> Optional[int]:
    """
    Estimate the number of removed tokens using suffix alignment.
    
    Args:
        tokenizer: The tokenizer
        prompt: The prompt text
        raw_answer: Original answer before cleaning
        cleaned_answer: Answer after echo removal
    
    Returns:
        Estimated number of removed tokens, or None if cannot estimate
    """
    raw_ids = token_ids(prompt + raw_answer, tokenizer, add_special_tokens=True)
    clean_ids = token_ids(prompt + cleaned_answer, tokenizer, add_special_tokens=True)
    prompt_ids = token_ids(prompt, tokenizer, add_special_tokens=True)
    
    p = len(prompt_ids)
    raw_ans_ids_only = raw_ids[p:]
    clean_ans_ids_only = clean_ids[p:]
    
    suf = longest_common_suffix_len(raw_ans_ids_only, clean_ans_ids_only)
    if suf <= 0:
        return None
    
    return max(0, len(raw_ans_ids_only) - suf)
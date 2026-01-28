#!/usr/bin/env python3
"""
File processing utilities for attention analysis
"""

import json
from typing import Any, Dict, List, Optional
from tqdm import tqdm

try:
    from .utils import truncate_at_next_question, write_jsonl
    from .mlp_utils import remove_echo_with_mlp, estimate_removed_tokens_via_suffix
    from .attention_metrics import compute_attention_for_pair, compute_per_head_for_pair
except ImportError:
    from utils import truncate_at_next_question, write_jsonl
    from mlp_utils import remove_echo_with_mlp, estimate_removed_tokens_via_suffix
    from attention_metrics import compute_attention_for_pair, compute_per_head_for_pair


def process_file(
    in_path: str,
    tokenizer,
    model,
    answer_prefix_tokens: int,
    embedder,
    initial_threshold: float,
    drop_threshold: float,
    file_label: str,
    use_removed_as_prefix: bool,
    want_per_layer: bool = False,
    build_prompt_fn=None,
) -> List[Dict[str, Any]]:
    """
    Process a JSONL file to compute attention metrics.
    
    Args:
        in_path: Path to input JSONL file
        tokenizer: The tokenizer
        model: The model
        answer_prefix_tokens: Number of answer prefix tokens
        embedder: Embedding model for MLP detection
        initial_threshold: Initial threshold for MLP
        drop_threshold: Drop threshold for MLP
        file_label: Label for progress bar
        use_removed_as_prefix: Whether to use removed tokens as prefix
        want_per_layer: Whether to compute per-layer metrics
        build_prompt_fn: Function to build prompt from question
    
    Returns:
        List of processed rows with attention metrics
    """
    if build_prompt_fn is None:
        from src.evaluation.logp_trim_experiment import build_prompt
        build_prompt_fn = build_prompt
    
    rows: List[Dict[str, Any]] = []
    
    # Count total lines for progress bar
    try:
        with open(in_path, "r", encoding="utf-8") as fcnt:
            total_lines = sum(1 for _ in fcnt)
    except Exception:
        total_lines = None

    with open(in_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, total=total_lines, desc=f"Processing {file_label}", unit="lines")
        for line in iterator:
            if not line.strip():
                continue
            s = json.loads(line)
            try:
                q = s.get("problem", "")
                preds = s.get("pred", [])
                if not isinstance(preds, list) or not preds:
                    continue
                a_raw = truncate_at_next_question(preds[-1] if isinstance(preds[-1], str) else str(preds[-1]))

                # MLP echo removal (optional)
                removed_tokens_est: Optional[int] = None
                if embedder is not None:
                    a_clean, marker = remove_echo_with_mlp(q, a_raw, embedder, initial_threshold, drop_threshold)
                    if marker == "__ESTIMATE_REMOVED__":
                        prompt = build_prompt_fn(q)
                        est = estimate_removed_tokens_via_suffix(tokenizer, prompt, a_raw, a_clean)
                        removed_tokens_est = est
                    a_for_attn = a_raw if (removed_tokens_est is None) else a_raw
                else:
                    a_for_attn = a_raw

                prompt = build_prompt_fn(q)
                metrics = compute_attention_for_pair(
                    model,
                    tokenizer,
                    prompt,
                    a_for_attn,
                    answer_prefix_tokens,
                    removed_tokens_est,
                    return_per_layer=want_per_layer,
                    use_removed_as_prefix=use_removed_as_prefix,
                )
                rows.append({
                    "idx": s.get("idx"),
                    "is_correct": s.get("is_correct"),
                    **metrics,
                })
            except Exception as e:
                rows.append({
                    "idx": s.get("idx"),
                    "error": f"{type(e).__name__}: {e}",
                })
    return rows


def export_per_layer(
    in_path: str,
    label: str,
    tokenizer,
    model,
    answer_prefix_tokens: int,
    embedder,
    initial_threshold: float,
    drop_threshold: float,
    use_probe_prefix_len_for_ans_prefix: bool,
    output_dir: str,
    build_prompt_fn=None,
) -> None:
    """
    Export per-layer attention metrics to JSONL file.
    
    Args:
        in_path: Input JSONL path
        label: Label for output file
        tokenizer: The tokenizer
        model: The model
        answer_prefix_tokens: Number of answer prefix tokens
        embedder: Embedding model
        initial_threshold: MLP initial threshold
        drop_threshold: MLP drop threshold
        use_probe_prefix_len_for_ans_prefix: Whether to use probe prefix length
        output_dir: Output directory
        build_prompt_fn: Function to build prompt
    """
    if build_prompt_fn is None:
        from src.evaluation.logp_trim_experiment import build_prompt
        build_prompt_fn = build_prompt
    
    from .utils import write_jsonl
    import os
    
    out_rows: List[Dict[str, Any]] = []
    
    # Count total lines
    try:
        with open(in_path, "r", encoding="utf-8") as fcnt:
            total_lines = sum(1 for _ in fcnt)
    except Exception:
        total_lines = None
    
    with open(in_path, "r", encoding="utf-8") as f:
        it = tqdm(f, total=total_lines, desc=f"Per-layer {label}", unit="lines")
        for line in it:
            if not line.strip():
                continue
            s = json.loads(line)
            try:
                q = s.get("problem", "")
                preds = s.get("pred", [])
                if not isinstance(preds, list) or not preds:
                    continue
                a_raw = truncate_at_next_question(preds[-1] if isinstance(preds[-1], str) else str(preds[-1]))
                
                # Optional: use probe-estimated prefix length
                removed_tokens_est_pl: Optional[int] = None
                if embedder is not None and use_probe_prefix_len_for_ans_prefix:
                    a_clean, marker = remove_echo_with_mlp(q, a_raw, embedder, initial_threshold, drop_threshold)
                    if marker == "__ESTIMATE_REMOVED__":
                        prompt = build_prompt_fn(q)
                        est = estimate_removed_tokens_via_suffix(tokenizer, prompt, a_raw, a_clean)
                        removed_tokens_est_pl = est
                
                prompt = build_prompt_fn(q)
                metrics = compute_attention_for_pair(
                    model,
                    tokenizer,
                    prompt,
                    a_raw,
                    answer_prefix_tokens,
                    removed_tokens_est_pl,
                    return_per_layer=True,
                    use_removed_as_prefix=use_probe_prefix_len_for_ans_prefix,
                )
                out_rows.append({
                    "idx": s.get("idx"),
                    "is_correct": s.get("is_correct"),
                    "per_layer": metrics.get("per_layer", {}),
                })
            except Exception as e:
                out_rows.append({
                    "idx": s.get("idx"),
                    "error": f"{type(e).__name__}: {e}",
                })
    
    write_jsonl(os.path.join(output_dir, f"per_layer_{label}.jsonl"), out_rows)


def export_per_head(
    in_path: str,
    label: str,
    tokenizer,
    model,
    answer_prefix_tokens: int,
    embedder_ph,
    initial_threshold: float,
    drop_threshold: float,
    use_probe_prefix_len_for_ans_prefix: bool,
    output_dir: str,
    build_prompt_fn=None,
) -> None:
    """
    Export per-head attention metrics to JSONL file.
    
    Args:
        in_path: Input JSONL path
        label: Label for output file
        tokenizer: The tokenizer
        model: The model
        answer_prefix_tokens: Number of answer prefix tokens
        embedder_ph: Embedding model for per-head analysis
        initial_threshold: MLP initial threshold
        drop_threshold: MLP drop threshold
        use_probe_prefix_len_for_ans_prefix: Whether to use probe prefix length
        output_dir: Output directory
        build_prompt_fn: Function to build prompt
    """
    if build_prompt_fn is None:
        from src.evaluation.logp_trim_experiment import build_prompt
        build_prompt_fn = build_prompt
    
    from .utils import write_jsonl
    import os
    
    out_rows: List[Dict[str, Any]] = []
    
    try:
        with open(in_path, "r", encoding="utf-8") as fcnt:
            total_lines = sum(1 for _ in fcnt)
    except Exception:
        total_lines = None
    
    with open(in_path, "r", encoding="utf-8") as f:
        it = tqdm(f, total=total_lines, desc=f"Per-head {label}", unit="lines")
        for line in it:
            if not line.strip():
                continue
            s = json.loads(line)
            try:
                q = s.get("problem", "")
                preds = s.get("pred", [])
                if not isinstance(preds, list) or not preds:
                    continue
                a_raw = truncate_at_next_question(preds[-1] if isinstance(preds[-1], str) else str(preds[-1]))
                
                removed_tokens_est_pl: Optional[int] = None
                if embedder_ph is not None and use_probe_prefix_len_for_ans_prefix:
                    a_clean, marker = remove_echo_with_mlp(q, a_raw, embedder_ph, initial_threshold, drop_threshold)
                    if marker == "__ESTIMATE_REMOVED__":
                        try:
                            prompt = build_prompt_fn(q)
                            est = estimate_removed_tokens_via_suffix(tokenizer, prompt, a_raw, a_clean)
                            removed_tokens_est_pl = est
                        except Exception:
                            removed_tokens_est_pl = None
                
                prompt = build_prompt_fn(q)
                m = compute_per_head_for_pair(
                    model, tokenizer, prompt, a_raw, answer_prefix_tokens, removed_tokens_est_pl
                )
                out_rows.append({
                    "idx": s.get("idx"),
                    "is_correct": s.get("is_correct"),
                    **m,
                })
            except Exception as e:
                out_rows.append({
                    "idx": s.get("idx"),
                    "error": f"{type(e).__name__}: {e}",
                })
    
    write_jsonl(os.path.join(output_dir, f"per_head_{label}.jsonl"), out_rows)
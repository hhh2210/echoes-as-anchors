#!/usr/bin/env python3
"""
Information Flow Route (attention-only)
======================================

This script builds an **attention-only information flow route** for a single
sample, closer in spirit to the Information Flow Routes figure from
Ferrando & Voita (2024) / the LLM Transparency Tool, but using only the
native attention tensors of a HF model (e.g., DeepSeek-R1-Distill-Llama-8B).

Key idea
--------
- Nodes: (layer, token) positions in a token×layer grid (after each block).
- Edges:
    - Residual edge: (layer-1, token) -> (layer, token)
    - Attention edge: for each layer ℓ and target token j, edges from
      source tokens i at layer ℓ-1 with high attention weight
      M_ℓ[j, i] (mean over heads).
- We start from the final answer token at the top layer and **walk
  backwards** layer by layer along high-attention edges, collecting a small
  route subgraph. The result is a polyline-style figure that shows how
  information flows from earlier tokens (question / echo prefix) through
  intermediate layers into the answer token.

Compared to `info_flow_case.py` (which shows per-layer top-K columns), this
script explicitly constructs cross-layer edges and is much closer visually to
the original "flow routes" figure.

Usage (example)
---------------

From repo root:

  python -m train_repeat.src.visualization.info_flow_route \\
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --logp_results_json train_repeat/analysis_results_20250809_141519/correct_converted.jsonl \\
    --idx 0 \\
    --answer_prefix_tokens 32 \\
    --attn_threshold 0.002 \\
    --topk_src 3 \\
    --max_seq_len 768 \\
    --output_path 675582843dfd537dbcfb6ef0/info_flow_route_deepseek_idx0.pdf \\
    --title "DeepSeek-R1: information flow route (correct, idx=0)"

This figure is intended for the ICLR rebuttal to directly illustrate how
prompt/question vs. echo/prefix tokens influence the final answer token
across layers in a reasoning model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class QAExample:
    question: str
    answer: str


def _load_qa_from_jsonl(path: str, idx: int) -> QAExample:
    """Load (question, answer) from converted JSONL (problem/pred/is_correct)."""
    q: Optional[str] = None
    a: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if int(s.get("idx", -1)) != int(idx):
                continue
            q = s.get("problem", "")
            preds = s.get("pred", [])
            if isinstance(preds, list) and preds:
                last = preds[-1]
                a = last if isinstance(last, str) else str(last)
            break
    if q is None or a is None:
        raise ValueError(f"Failed to find idx={idx} in {path}")
    return QAExample(question=q, answer=a)


def _prepare_model(model_path: str):
    """Load model & tokenizer via the shared helper to stay consistent."""
    from ..evaluation.attention_from_converted import _load_model_and_tokenizer  # type: ignore

    tokenizer, model, device = _load_model_and_tokenizer(model_path)
    return tokenizer, model, device


def _collect_attn_and_segments(
    model,
    tokenizer,
    question: str,
    answer: str,
    max_seq_len: Optional[int],
    query_answer_index: int,
) -> Tuple[List[torch.Tensor], List[str], int, Tuple[int, int], int]:
    """
    Run model once, return:
    - attn_mats: list of [T, T] (mean over heads) for each layer
    - tokens_disp: pretty token labels
    - ans_start: index of first answer token
    - q_span: (start, end) of question tokens inside prompt
    - q_idx: global token index of the chosen answer token
    """
    from ..evaluation.attention_from_converted import (  # type: ignore
        build_prompt,
        _collect_attentions,
        _token_ids,
        _mean_heads,
        _find_question_token_span_in_prompt,
    )
    from .attention_case_viz import (  # type: ignore
        _maybe_truncate_input_ids,
        _tokens_from_ids,
        _pretty_token_labels,
    )

    prompt = build_prompt(question)
    full_text = prompt + answer

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    p_len = len(_token_ids(prompt, tokenizer, add_special_tokens=True))

    # Truncate long sequences: keep full prompt + answer tail
    input_ids, p_len = _maybe_truncate_input_ids(input_ids, p_len, max_seq_len)
    if attn_mask is not None and input_ids.shape[1] != attn_mask.shape[1]:
        attn_mask = torch.ones_like(input_ids)

    T = int(input_ids.shape[1])
    ans_start = p_len
    ans_end = T
    if ans_start >= ans_end:
        raise ValueError(f"Answer segment is empty (prompt_len={p_len}, seq_len={T})")

    if query_answer_index < 0:
        q_idx = max(ans_start, ans_end + query_answer_index)
    else:
        q_idx = ans_start + query_answer_index
    if not (0 <= q_idx < T):
        raise ValueError(f"query index {q_idx} out of range for seq_len={T}")

    # Collect attentions
    attns = _collect_attentions(model, input_ids, attn_mask)
    attn_mats: List[torch.Tensor] = []
    for a in attns:
        # a: [num_heads, T, T]
        M = _mean_heads(a)  # [T, T]
        attn_mats.append(M.detach().cpu())

    # Question span (token indices inside prompt)
    q_span_obj = _find_question_token_span_in_prompt(tokenizer, prompt, question)
    if q_span_obj is None:
        q_span = (0, ans_start)
    else:
        q_span = (int(q_span_obj.start), int(q_span_obj.end))

    # Token labels
    ids = input_ids[0].detach().cpu().tolist()
    tokens_raw = _tokens_from_ids(tokenizer, ids)
    tokens_disp = _pretty_token_labels(tokenizer, tokens_raw)

    return attn_mats, tokens_disp, ans_start, q_span, q_idx


def _build_routes(
    attn_mats: List[torch.Tensor],
    q_span: Tuple[int, int],
    prefix_span: Tuple[int, int],
    ans_idx: int,
    attn_threshold: float,
    topk_src: int,
) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    """
    Build a small route subgraph starting from (L-1, ans_idx) and going
    backwards along high-attention edges.

    Returns:
    - active_nodes: set of (layer, token)
    - edges: list of (layer_from, token_from, layer_to, token_to)
    """
    import numpy as np

    L = len(attn_mats)
    if L == 0:
        raise ValueError("No attention matrices provided")

    T = int(attn_mats[0].shape[0])
    q_start, q_end = q_span
    p_start, p_end = prefix_span

    # Allowed tokens: question ∪ prefix ∪ {answer_idx}
    allowed = np.zeros(T, dtype=bool)
    allowed[q_start:q_end] = True
    allowed[p_start:p_end] = True
    if 0 <= ans_idx < T:
        allowed[ans_idx] = True

    active_nodes: Set[Tuple[int, int]] = set()
    edges: List[Tuple[int, int, int, int]] = []

    # frontier_by_layer[ℓ] = set of token indices at layer ℓ that we are
    # currently expanding (starting from the answer at top layer).
    frontier_by_layer: Dict[int, Set[int]] = {L - 1: {ans_idx}}
    active_nodes.add((L - 1, ans_idx))

    for layer in range(L - 1, 0, -1):
        current = frontier_by_layer.get(layer, set())
        if not current:
            continue

        M = attn_mats[layer].numpy()  # [T, T], row = query (target), col = key (source)
        for tgt in current:
            if not (0 <= tgt < T):
                continue
            row = M[tgt]  # attention from tgt to all source tokens at previous layer

            idx = np.where(allowed)[0]
            if idx.size == 0:
                continue
            vals = row[idx]
            if attn_threshold > 0.0:
                keep = vals >= attn_threshold
                idx = idx[keep]
                vals = vals[keep]
                if idx.size == 0:
                    # still add a residual edge so the path is continuous
                    if 0 <= layer - 1 < L:
                        edges.append((layer - 1, tgt, layer, tgt))
                        active_nodes.add((layer - 1, tgt))
                        frontier_by_layer.setdefault(layer - 1, set()).add(tgt)
                    continue

            k = min(topk_src, idx.size)
            top_local = np.argsort(-vals)[:k]
            src_tokens = idx[top_local]

            for src in src_tokens:
                edges.append((layer - 1, int(src), layer, int(tgt)))
                active_nodes.add((layer - 1, int(src)))
                active_nodes.add((layer, int(tgt)))
                frontier_by_layer.setdefault(layer - 1, set()).add(int(src))

            # Also add a residual edge to keep a vertical backbone.
            edges.append((layer - 1, int(tgt), layer, int(tgt)))
            active_nodes.add((layer - 1, int(tgt)))
            frontier_by_layer.setdefault(layer - 1, set()).add(int(tgt))

    return active_nodes, edges


def _plot_route(
    tokens: List[str],
    active_nodes: Set[Tuple[int, int]],
    edges: List[Tuple[int, int, int, int]],
    q_span: Tuple[int, int],
    prefix_span: Tuple[int, int],
    ans_idx: int,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Render the route subgraph on a token×layer grid."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Set default font size to match paper
        plt.rcParams.update({'font.size': 10})
    except Exception as e:  # pragma: no cover
        print(f"[WARN] matplotlib import failed, skip plot: {e}")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not edges:
        print("[WARN] No edges in route; nothing to plot.")

    # Layers present in nodes/edges
    all_layers = {l for (l, _) in active_nodes}
    for (lf, _, lt, _) in edges:
        all_layers.add(lf)
        all_layers.add(lt)
    if not all_layers:
        all_layers = {0}
    L_min, L_max = min(all_layers), max(all_layers)
    layers = list(range(L_min, L_max + 1))
    L = len(layers)

    T = len(tokens)

    fig_w = max(6.0, min(14.0, 0.18 * T + 3.0))
    fig_h = 4.2
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    # Map real layer index -> y coordinate
    layer_to_y = {l: i for i, l in enumerate(reversed(layers))}  # top = highest layer

    # Background grid
    for x in range(T):
        for l in layers:
            y = layer_to_y[l]
            ax.scatter(
                x,
                y,
                s=6,
                color="#d0d0d0",
                alpha=0.35,
                zorder=1,
            )

    # Highlight mid-layers explicitly as [7, 18] (paper definition), if they exist.
    mid_start, mid_end = 7, 18
    if any(l == mid_start for l in layers) and any(l == mid_end for l in layers):
        y_hi = layer_to_y[mid_start] - 0.5  # smaller y -> visually higher
        y_lo = layer_to_y[mid_end] + 0.5    # larger y -> visually lower
        ax.axhspan(y_lo, y_hi, color="#d0ffd0", alpha=0.12, zorder=0)

    q_start, q_end = q_span
    p_start, p_end = prefix_span

    # Draw edges
    for (lf, tf, lt, tt) in edges:
        if lf not in layer_to_y or lt not in layer_to_y:
            continue
        x1, y1 = tf, layer_to_y[lf]
        x2, y2 = tt, layer_to_y[lt]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#6aa84f",  # greenish
            linewidth=1.3,
            alpha=0.9,
            zorder=2,
        )

    # Draw active nodes with colors: question vs prefix vs other
    for (l, t) in active_nodes:
        if l not in layer_to_y:
            continue
        x, y = t, layer_to_y[l]
        if q_start <= t < q_end:
            face = "#1f77b4"  # blue
        elif p_start <= t < p_end:
            face = "#2ca02c"  # green
        else:
            face = "#9467bd"  # purple-ish for other context
        ax.scatter(
            x,
            y,
            s=30,
            color=face,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    # Mark answer token at top with a triangle
    top_layer = max(all_layers)
    if top_layer in layer_to_y:
        ax.scatter(
            [ans_idx],
            [layer_to_y[top_layer]],
            s=70,
            color="#ff7f0e",
            marker="v",
            zorder=4,
        )

    # Axes / labels
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.invert_yaxis()

    ax.set_yticks([layer_to_y[l] for l in layers])
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xticks(range(T))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=5)
    ax.tick_params(axis="y", labelsize=7)

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Layer")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved info-flow route figure to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention-only information flow route for a single answer token."
    )
    parser.add_argument("--model", type=str, required=True, help="Path or name of HF model.")
    parser.add_argument("--logp_results_json", type=str, required=True, help="Converted JSONL with problem/pred.")
    parser.add_argument("--idx", type=int, required=True, help="Sample index in the JSONL.")
    parser.add_argument(
        "--answer_prefix_tokens",
        type=int,
        default=32,
        help="Number of answer prefix tokens treated as echo region.",
    )
    parser.add_argument(
        "--query_answer_index",
        type=int,
        default=-1,
        help="Answer token index used as the query (relative to answer start).",
    )
    parser.add_argument(
        "--attn_threshold",
        type=float,
        default=0.002,
        help="Minimum attention weight for adding a source token edge.",
    )
    parser.add_argument(
        "--topk_src",
        type=int,
        default=3,
        help="Max number of source tokens per (layer, target) to keep in the route.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=768,
        help="Optional max sequence length; keep full prompt and answer tail if longer.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save the resulting PDF/PNG.",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    qa = _load_qa_from_jsonl(args.logp_results_json, args.idx)
    tokenizer, model, _ = _prepare_model(args.model)

    attn_mats, tokens_disp, ans_start, q_span, q_idx = _collect_attn_and_segments(
        model=model,
        tokenizer=tokenizer,
        question=qa.question,
        answer=qa.answer,
        max_seq_len=args.max_seq_len,
        query_answer_index=args.query_answer_index,
    )

    T = len(tokens_disp)
    ans_end = T
    ans_len = max(0, ans_end - ans_start)
    k = min(max(args.answer_prefix_tokens, 0), ans_len)
    prefix_span = (ans_start, ans_start + k)

    active_nodes, edges = _build_routes(
        attn_mats=attn_mats,
        q_span=q_span,
        prefix_span=prefix_span,
        ans_idx=q_idx,
        attn_threshold=args.attn_threshold,
        topk_src=args.topk_src,
    )

    title = args.title or "Information flow route: question vs. answer-prefix"

    _plot_route(
        tokens=tokens_disp,
        active_nodes=active_nodes,
        edges=edges,
        q_span=q_span,
        prefix_span=prefix_span,
        ans_idx=q_idx,
        output_path=args.output_path,
        title=title,
    )


if __name__ == "__main__":
    main()

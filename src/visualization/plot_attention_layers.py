#!/usr/bin/env python3
"""
Plot per-layer attention trajectories and group differences.

Inputs:
- per_layer_correct.jsonl / per_layer_wrong.jsonl produced by attention_from_converted.py --per_layer_trajectories

Outputs:
- PDF/PNG curves with mean ± s.e.m. across samples per layer, and difference curve
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    xs = [x for x in values if isinstance(x, (int, float))]
    if not xs:
        return float("nan"), float("nan")
    import math
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sem = (var ** 0.5) / (n ** 0.5)
    return mean, sem


def _aggregate_per_layer(rows: List[dict], key: str) -> Tuple[List[float], List[float]]:
    # key in {"ans_to_question", "ans_to_ans_prefix"}
    # rows[i]["per_layer"][key] is a list over layers
    per_layer_lists: List[List[float]] = []
    L = None
    for r in rows:
        try:
            seq = r.get("per_layer", {}).get(key, None)
            if isinstance(seq, list) and seq:
                if L is None:
                    L = len(seq)
                if len(seq) == L:
                    per_layer_lists.append(seq)
        except Exception:
            continue
    if L is None or not per_layer_lists:
        return [], []
    means: List[float] = []
    sems: List[float] = []
    for l in range(L):
        vals = [s[l] for s in per_layer_lists]
        m, s = _mean_sem(vals)
        means.append(m)
        sems.append(s)
    return means, sems


def _plot(curves: Dict[str, Tuple[List[float], List[float]]], out_path: str, title: str, ylabel: str, figsize: Tuple[float, float], show_title: bool) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skip plot.")
        return
    plt.figure(figsize=figsize)
    for label, (means, sems) in curves.items():
        if not means:
            continue
        xs = list(range(len(means)))
        line_style = "-" if label.lower() == "correct" else "--" if label.lower() == "wrong" else "-"
        plt.plot(xs, means, label=label, linestyle=line_style)
        if sems:
            import numpy as np
            means_np = np.array(means)
            sems_np = np.array(sems)
            plt.fill_between(xs, means_np - sems_np, means_np + sems_np, alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    if show_title:
        plt.title(title)
    
    # Add light gray vertical dashed lines at layer 7 and layer 18
    plt.axvline(x=7, color='lightgray', linestyle='--', alpha=0.7, linewidth=1)
    plt.axvline(x=18, color='lightgray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add visible numbers on the x-axis at the vertical lines
    ax = plt.gca()
    # Fix x-axis range with a small left margin before 0
    ax.set_xlim(-0.5, 31)
    # Custom ticks: remove 20 and 30; keep 0, 7, 10, 18, 31
    xticks = [0, 7, 10, 18, 31]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(t)) for t in xticks])
    
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-layer attention trajectories")
    parser.add_argument("--metrics_dir", type=str, default="/path/to/your/metrics_dir")
    parser.add_argument("--output", type=str, default="attention_layers.pdf")
    parser.add_argument("--compare", type=str, default="correctness", choices=["correctness"])  # extend later
    parser.add_argument("--focus_layers", type=int, nargs=2, default=None)
    # Publication style controls
    parser.add_argument("--pub", action="store_true", default=True, help="Use publication style (9pt fonts, compact size)")
    parser.add_argument("--font_size", type=float, default=9.0)
    parser.add_argument("--tick_font_size", type=float, default=8.0)
    parser.add_argument("--legend_font_size", type=float, default=8.0)
    parser.add_argument("--figsize", type=float, nargs=2, default=[3.4, 2.1])
    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--percent", action="store_true", help="Scale y to percentage (x100) to match table caption")
    args = parser.parse_args()

    # Apply publication rc if requested
    if args.pub:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "font.size": args.font_size,
                "axes.labelsize": args.font_size,
                "axes.titlesize": args.font_size,
                "xtick.labelsize": args.tick_font_size,
                "ytick.labelsize": args.tick_font_size,
                "legend.fontsize": args.legend_font_size,
                "lines.linewidth": 1.4,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
            })
        except Exception:
            pass

    correct_path = os.path.join(args.metrics_dir, "per_layer_correct.jsonl")
    wrong_path = os.path.join(args.metrics_dir, "per_layer_wrong.jsonl")
    correct = _read_jsonl(correct_path)
    wrong = _read_jsonl(wrong_path)

    # Curves: ans->question
    cq_m, cq_s = _aggregate_per_layer(correct, "ans_to_question")
    wq_m, wq_s = _aggregate_per_layer(wrong, "ans_to_question")
    if args.percent and cq_m and wq_m:
        cq_m = [x * 100 for x in cq_m]
        wq_m = [x * 100 for x in wq_m]
        cq_s = [s * 100 for s in cq_s]
        wq_s = [s * 100 for s in wq_s]
    curves_q = {
        "Correct": (cq_m, cq_s),
        "Wrong": (wq_m, wq_s),
    }
    _plot(curves_q, "fig_layers_q.pdf", "Answer→Question (per-layer)", "Attention weight (%)" if args.percent else "Attention weight", tuple(args.figsize), not args.no_title)

    # Curves: ans->answer-prefix
    cp_m, cp_s = _aggregate_per_layer(correct, "ans_to_ans_prefix")
    wp_m, wp_s = _aggregate_per_layer(wrong, "ans_to_ans_prefix")
    if args.percent and cp_m and wp_m:
        cp_m = [x * 100 for x in cp_m]
        wp_m = [x * 100 for x in wp_m]
        cp_s = [s * 100 for s in cp_s]
        wp_s = [s * 100 for s in wp_s]
    curves_p = {
        "Correct": (cp_m, cp_s),
        "Wrong": (wp_m, wp_s),
    }
    _plot(curves_p, "fig_layers_pref.pdf", "Answer→Answer-prefix (per-layer)", "Attention weight (%)" if args.percent else "Attention weight", tuple(args.figsize), not args.no_title)

    if args.focus_layers and len(args.focus_layers) == 2:
        l0, l1 = args.focus_layers
        def _avg(seg: List[float]) -> float:
            if not seg:
                return float("nan")
            return sum(seg) / len(seg)
        import math
        if cq_m and wq_m:
            avg_cq = _avg(cq_m[l0:l1+1])
            avg_wq = _avg(wq_m[l0:l1+1])
            print(f"TABLE_VALUE:Layers 7-18: answer -> question:Correct:{avg_cq:.4f}")
            print(f"TABLE_VALUE:Layers 7-18: answer -> question:Wrong:{avg_wq:.4f}")
            avg_diff_q = avg_cq - avg_wq
            val_q = avg_diff_q if args.percent else avg_diff_q * 100
            print(f"Focus layers [{l0}-{l1}] Answer→Question: Δ(C−W)={val_q:.2f}%")
        if cp_m and wp_m:
            avg_cp = _avg(cp_m[l0:l1+1])
            avg_wp = _avg(wp_m[l0:l1+1])
            print(f"TABLE_VALUE:Layers 7-18: answer -> answer-prefix:Correct:{avg_cp:.4f}")
            print(f"TABLE_VALUE:Layers 7-18: answer -> answer-prefix:Wrong:{avg_wp:.4f}")
            avg_diff_p = avg_cp - avg_wp
            val_p = avg_diff_p if args.percent else avg_diff_p * 100
            print(f"Focus layers [{l0}-{l1}] Answer→Answer-prefix: Δ(C−W)={val_p:.2f}%")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Compare attention patterns between Baseline and ED-SFT models (Qwen3-8B family)
===============================================================================

目的：严格对齐论文段落“中间层(7-18)的 answer→answer-prefix 注意力增强”。

本脚本对两个模型（Baseline 与 ED-SFT）分别在相同评测集上计算注意力指标，
对每个模型内部进行“正确 vs 错误”分组的层级统计，并特别输出中层桶（默认 mid:7-18）
的 answer→answer-prefix 平均注意力差值（正确均值 - 错误均值）。随后比较 ED-SFT 与 Baseline
在该差值上的提升幅度（百分比点），以复现实证分析。

依赖：
- src.evaluation.processing.process_file（按行前向并产出 per_layer 指标）
- src.evaluation.model_utils.load_model_and_tokenizer（加载模型/分词器）
- src.evaluation.statistical_analysis.compute_layer_stats（AUC与Cohen's d 等）
- src.evaluation.utils.parse_bucket_def / write_jsonl / read_jsonl_rows

输入：
- --base_model, --edsft_model：两个模型路径或 repo_id
- --base_correct_converted, --base_wrong_converted：Baseline 模型对应的正确/错误样本（由 split_* 得到）
- --edsft_correct_converted, --edsft_wrong_converted：ED-SFT 模型对应的正确/错误样本

注意：正确/错误样本应来自各自模型的 lm-eval 结果转换与分组，以保证“正确 vs 错误”在每个模型内部自洽。

输出：
- {output_dir}/baseline/ 和 {output_dir}/edsft/ 下的原始行级指标与统计文件
- {output_dir}/comparison_summary.json：包含中层差值与提升幅度，以及 AUC / d 的对比
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.model_utils import load_model_and_tokenizer
from src.evaluation.processing import process_file
from src.evaluation.statistical_analysis import compute_layer_stats
from src.evaluation.utils import write_jsonl, read_jsonl_rows, parse_bucket_def


def _nanmean(values: List[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _collect_mid_means(rows: List[Dict[str, Any]], layer_range: Tuple[int, int]) -> List[float]:
    """对每一行样本，计算中层区间内的 per_layer.ans_to_ans_prefix 的均值。
    返回样本级均值列表，用于再求 across-sample 的均值。
    """
    a, b = layer_range
    out: List[float] = []
    for r in rows:
        pl = r.get("per_layer")
        if not isinstance(pl, dict):
            continue
        arr = pl.get("ans_to_ans_prefix")
        if not isinstance(arr, list) or not arr:
            continue
        # clamp
        L = len(arr)
        i = max(0, min(a, L - 1))
        j = max(0, min(b, L - 1))
        if i > j:
            i, j = j, i
        seg = [float(x) for x in arr[i : j + 1] if isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)]
        if not seg:
            continue
        out.append(sum(seg) / len(seg))
    return out


def _run_one_model(
    label: str,
    model_path: str,
    correct_converted: str,
    wrong_converted: str,
    output_dir: str,
    answer_prefix_tokens: int,
    use_mlp_echo_removal: bool,
    embedding_model_path: Optional[str],
    initial_threshold: float,
    drop_threshold: float,
    use_probe_prefix_len_for_ans_prefix: bool,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model, _ = load_model_and_tokenizer(model_path)
    embedder = None
    if use_mlp_echo_removal:
        # 延迟导入，避免硬依赖
        try:
            from src.evaluation.model_utils import load_embedder
            import torch
            embedder = load_embedder(embedding_model_path, model.device)  # type: ignore[arg-type]
        except Exception:
            embedder = None

    # per-line rows with per_layer present
    correct_rows = process_file(
        correct_converted,
        tokenizer,
        model,
        answer_prefix_tokens,
        embedder,
        initial_threshold,
        drop_threshold,
        file_label=f"{label}-correct",
        use_removed_as_prefix=use_probe_prefix_len_for_ans_prefix,
        want_per_layer=True,
    )
    wrong_rows = process_file(
        wrong_converted,
        tokenizer,
        model,
        answer_prefix_tokens,
        embedder,
        initial_threshold,
        drop_threshold,
        file_label=f"{label}-wrong",
        use_removed_as_prefix=use_probe_prefix_len_for_ans_prefix,
        want_per_layer=True,
    )

    write_jsonl(os.path.join(output_dir, "per_model_correct.jsonl"), correct_rows)
    write_jsonl(os.path.join(output_dir, "per_model_wrong.jsonl"), wrong_rows)

    # layer-wise statistics within the model (correct vs wrong)
    layer_stats = compute_layer_stats(correct_rows, wrong_rows, bucket_def=None)

    return {
        "rows": {"correct_path": os.path.join(output_dir, "per_model_correct.jsonl"),
                  "wrong_path": os.path.join(output_dir, "per_model_wrong.jsonl")},
        "layer_stats": layer_stats,
        "correct_rows": correct_rows,  # 后续用于 mid 均值差
        "wrong_rows": wrong_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attention between Baseline and ED-SFT (Qwen3-8B)")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--edsft_model", type=str, required=True)

    parser.add_argument("--base_correct_converted", type=str, required=True)
    parser.add_argument("--base_wrong_converted", type=str, required=True)
    parser.add_argument("--edsft_correct_converted", type=str, required=True)
    parser.add_argument("--edsft_wrong_converted", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answer_prefix_tokens", type=int, default=32)
    parser.add_argument("--bucket_def", type=str, default="early:0-6,mid:7-18,late:19-31",
                        help="层桶定义，例如 'early:0-6,mid:7-18,late:19-31'")
    parser.add_argument("--use_mlp_echo_removal", action="store_true")
    parser.add_argument("--embedding_model_path", type=str, default=None)
    parser.add_argument("--initial_threshold", type=float, default=0.6)
    parser.add_argument("--drop_threshold", type=float, default=0.15)
    parser.add_argument("--use_probe_prefix_len_for_ans_prefix", action="store_true",
                        help="当可用时，使用探针估计的 echo 前缀 token 数作为窗口长度")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_out = os.path.join(args.output_dir, "baseline")
    edsft_out = os.path.join(args.output_dir, "edsft")
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(edsft_out, exist_ok=True)

    # Run both models
    base = _run_one_model(
        label="baseline",
        model_path=args.base_model,
        correct_converted=args.base_correct_converted,
        wrong_converted=args.base_wrong_converted,
        output_dir=base_out,
        answer_prefix_tokens=args.answer_prefix_tokens,
        use_mlp_echo_removal=args.use_mlp_echo_removal,
        embedding_model_path=args.embedding_model_path,
        initial_threshold=args.initial_threshold,
        drop_threshold=args.drop_threshold,
        use_probe_prefix_len_for_ans_prefix=args.use_probe_prefix_len_for_ans_prefix,
    )

    edsft = _run_one_model(
        label="edsft",
        model_path=args.edsft_model,
        correct_converted=args.edsft_correct_converted,
        wrong_converted=args.edsft_wrong_converted,
        output_dir=edsft_out,
        answer_prefix_tokens=args.answer_prefix_tokens,
        use_mlp_echo_removal=args.use_mlp_echo_removal,
        embedding_model_path=args.embedding_model_path,
        initial_threshold=args.initial_threshold,
        drop_threshold=args.drop_threshold,
        use_probe_prefix_len_for_ans_prefix=args.use_probe_prefix_len_for_ans_prefix,
    )

    # Determine mid bucket indices based on the number of layers in either result
    # Fallback to default mid:7-18 if parse fails.
    def _infer_num_layers(rows: List[Dict[str, Any]]) -> Optional[int]:
        for r in rows:
            pl = r.get("per_layer")
            if isinstance(pl, dict):
                arr = pl.get("ans_to_ans_prefix")
                if isinstance(arr, list) and len(arr) > 0:
                    return len(arr)
        return None

    L_base = _infer_num_layers(base["correct_rows"]) or _infer_num_layers(base["wrong_rows"]) or 0
    L_edsft = _infer_num_layers(edsft["correct_rows"]) or _infer_num_layers(edsft["wrong_rows"]) or 0
    L = max(L_base, L_edsft)
    buckets = parse_bucket_def(args.bucket_def, L)
    mid_range = buckets.get("mid", (7, 18))

    # Compute mid means within each model
    base_corr_mid_means = _collect_mid_means(base["correct_rows"], mid_range)
    base_wrong_mid_means = _collect_mid_means(base["wrong_rows"], mid_range)
    edsft_corr_mid_means = _collect_mid_means(edsft["correct_rows"], mid_range)
    edsft_wrong_mid_means = _collect_mid_means(edsft["wrong_rows"], mid_range)

    base_corr_mid = _nanmean(base_corr_mid_means)
    base_wrong_mid = _nanmean(base_wrong_mid_means)
    edsft_corr_mid = _nanmean(edsft_corr_mid_means)
    edsft_wrong_mid = _nanmean(edsft_wrong_mid_means)

    base_diff = base_corr_mid - base_wrong_mid
    edsft_diff = edsft_corr_mid - edsft_wrong_mid
    improvement = edsft_diff - base_diff  # 绝对提升（注意力质量点差），可 ×100 视为百分比点

    # Extract AUC/d for mid bucket from layer_stats if available
    def _get_mid_stats(layer_stats: Dict[str, Any]) -> Dict[str, Optional[float]]:
        try:
            b = layer_stats.get("buckets", {})
            m = b.get("mid", {})
            return {
                "auc": float(m.get("ans_to_ans_prefix_auc")) if m.get("ans_to_ans_prefix_auc") is not None else None,
                "d": float(m.get("ans_to_ans_prefix_d")) if m.get("ans_to_ans_prefix_d") is not None else None,
            }
        except Exception:
            return {"auc": None, "d": None}

    base_mid_stats = _get_mid_stats(base["layer_stats"]) if isinstance(base.get("layer_stats"), dict) else {"auc": None, "d": None}
    edsft_mid_stats = _get_mid_stats(edsft["layer_stats"]) if isinstance(edsft.get("layer_stats"), dict) else {"auc": None, "d": None}

    summary = {
        "config": {
            "base_model": args.base_model,
            "edsft_model": args.edsft_model,
            "answer_prefix_tokens": args.answer_prefix_tokens,
            "bucket_def": args.bucket_def,
            "mid_range": list(mid_range),
            "use_mlp_echo_removal": bool(args.use_mlp_echo_removal),
            "use_probe_prefix_len_for_ans_prefix": bool(args.use_probe_prefix_len_for_ans_prefix),
        },
        "base": {
            "layer_stats_path": os.path.join(base_out, "layer_stats.json"),
            "mid_means": {
                "correct": base_corr_mid,
                "wrong": base_wrong_mid,
                "diff": base_diff,
            },
            "mid_auc": base_mid_stats.get("auc"),
            "mid_d": base_mid_stats.get("d"),
        },
        "edsft": {
            "layer_stats_path": os.path.join(edsft_out, "layer_stats.json"),
            "mid_means": {
                "correct": edsft_corr_mid,
                "wrong": edsft_wrong_mid,
                "diff": edsft_diff,
            },
            "mid_auc": edsft_mid_stats.get("auc"),
            "mid_d": edsft_mid_stats.get("d"),
        },
        "delta": {
            "mid_diff_improvement": improvement,  # ×100 即百分比点
            "mid_auc_improvement": (edsft_mid_stats.get("auc") - base_mid_stats.get("auc")) if (base_mid_stats.get("auc") is not None and edsft_mid_stats.get("auc") is not None) else None,
            "mid_d_improvement": (edsft_mid_stats.get("d") - base_mid_stats.get("d")) if (base_mid_stats.get("d") is not None and edsft_mid_stats.get("d") is not None) else None,
        },
    }

    # Persist per-model layer_stats for reproducibility
    try:
        with open(os.path.join(base_out, "layer_stats.json"), "w", encoding="utf-8") as f:
            json.dump(base["layer_stats"], f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        with open(os.path.join(edsft_out, "layer_stats.json"), "w", encoding="utf-8") as f:
            json.dump(edsft["layer_stats"], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("对比完成：")
    print(f"  mid 区间 {mid_range} 的 ans→ans-prefix 差值(正确-错误)：")
    print(f"    Baseline: {base_diff:.6f}  (≈ {base_diff * 100:.2f} pp)")
    print(f"    ED-SFT  : {edsft_diff:.6f}  (≈ {edsft_diff * 100:.2f} pp)")
    print(f"    提升幅度: {improvement:.6f}  (≈ {improvement * 100:.2f} pp)")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Attention from converted JSONL
==============================

用途：直接读取 lm-eval 转换后的 JSONL（每行包含 problem/pred/is_correct），
可选启用 MLP 语义阈值法删除 <think> 开头的 prompt echo，随后计算注意力指标：
 - answer→question
 - answer→answer 前缀（首 K 个）
 - answer 尾段→被删前缀（仅在能稳定估计被删前缀长度时启用）

输入 JSONL（由 convert_lm_eval_for_logp.py 生成）：
  {"idx": int, "problem": str, "pred": [str, ...], "is_correct": bool}

输出：
  attention_metrics_correct.jsonl / attention_metrics_wrong.jsonl / attention_summary.json

与 .cursorrules/README 对齐：
 - 采用与 logp 流水线一致的 build_prompt(question)
 - 可选使用 train_mlp/utils.py 的 echo 检测阈值 (initial_threshold, drop_threshold)
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.model_utils import load_model_and_tokenizer, load_embedder
from src.evaluation.utils import write_jsonl, read_jsonl_rows, aggregate_metrics, parse_bucket_def
from src.evaluation.statistical_analysis import compute_layer_stats
from src.evaluation.processing import process_file, export_per_layer, export_per_head

# Import build_prompt function
try:
    from src.evaluation.logp_trim_experiment import build_prompt
except Exception:
    def build_prompt(question: str) -> str:  # type: ignore
        return (
            "You are an expert at solving math problems. Please think step by step.\n"
            f"Question: {question}\n"
            "Answer: <think>"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute attention metrics directly from converted JSONL (problem/pred)")
    parser.add_argument("--correct_converted", type=str, required=True, help="Path to *_correct_converted.jsonl")
    parser.add_argument("--wrong_converted", type=str, required=True, help="Path to *_wrong_converted.jsonl")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answer_prefix_tokens", type=int, default=32)
    parser.add_argument("--per_layer_trajectories", action="store_true", help="导出每层指标轨迹到 JSONL")
    parser.add_argument("--compute_layer_stats", action="store_true", help="计算每层 AUC / Cohen's d 以及早/中/晚层桶统计")
    parser.add_argument("--bucket_def", type=str, default="early:0-6,mid:7-18,late:19-31", help="层桶定义，例如 'early:0-6,mid:7-18,late:19-31'")
    parser.add_argument("--reuse_per_layer_from", type=str, default=None, help="复用已有的 per_layer_{correct,wrong}.jsonl 直接计算统计（跳过模型前向）")
    parser.add_argument("--prefix_lengths", type=str, default=None, help="可选：用逗号分隔的多K列表，例如 '8,16,32,64'；逐K复算并输出 layer_stats_K.json")
    parser.add_argument("--per_head", action="store_true", help="可选：输出每层每个head的指标（正确/错误分组），用于一致性分析")
    parser.add_argument(
        "--use_probe_prefix_len_for_ans_prefix",
        action="store_true",
        help="当可用时，使用 MLP 探针估计的 echo 前缀 token 数作为 answer→answer-prefix 的窗口长度；否则回退到 --answer_prefix_tokens",
    )
    parser.add_argument("--use_mlp_echo_removal", action="store_true")
    parser.add_argument("--embedding_model_path", type=str, default=None)
    parser.add_argument("--initial_threshold", type=float, default=0.6)
    parser.add_argument("--drop_threshold", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.reuse_per_layer_from:
        # Reuse existing per_layer_* files, skip model loading
        correct_pl = os.path.join(args.reuse_per_layer_from, "per_layer_correct.jsonl")
        wrong_pl = os.path.join(args.reuse_per_layer_from, "per_layer_wrong.jsonl")
        if not os.path.exists(correct_pl) or not os.path.exists(wrong_pl):
            raise FileNotFoundError("--reuse_per_layer_from 目录下未找到 per_layer_correct.jsonl / per_layer_wrong.jsonl")
        correct_rows = read_jsonl_rows(correct_pl)
        wrong_rows = read_jsonl_rows(wrong_pl)
    else:
        tokenizer, model, device = load_model_and_tokenizer(args.model)
        embedder = load_embedder(args.embedding_model_path, device) if args.use_mlp_echo_removal else None

        correct_rows = process_file(
            args.correct_converted,
            tokenizer,
            model,
            args.answer_prefix_tokens,
            embedder,
            args.initial_threshold,
            args.drop_threshold,
            file_label="correct",
            use_removed_as_prefix=args.use_probe_prefix_len_for_ans_prefix,
            want_per_layer=args.per_layer_trajectories or args.compute_layer_stats,
            build_prompt_fn=build_prompt,
        )
        wrong_rows = process_file(
            args.wrong_converted,
            tokenizer,
            model,
            args.answer_prefix_tokens,
            embedder,
            args.initial_threshold,
            args.drop_threshold,
            file_label="wrong",
            use_removed_as_prefix=args.use_probe_prefix_len_for_ans_prefix,
            want_per_layer=args.per_layer_trajectories or args.compute_layer_stats,
            build_prompt_fn=build_prompt,
        )

    # Write raw attention metrics (only when not reusing)
    if not args.reuse_per_layer_from:
        write_jsonl(os.path.join(args.output_dir, "attention_metrics_correct.jsonl"), correct_rows)
        write_jsonl(os.path.join(args.output_dir, "attention_metrics_wrong.jsonl"), wrong_rows)

    if args.per_layer_trajectories and not args.reuse_per_layer_from:
        export_per_layer(
            args.correct_converted, "correct", tokenizer, model,
            args.answer_prefix_tokens, embedder, args.initial_threshold, args.drop_threshold,
            args.use_probe_prefix_len_for_ans_prefix, args.output_dir, build_prompt
        )
        export_per_layer(
            args.wrong_converted, "wrong", tokenizer, model,
            args.answer_prefix_tokens, embedder, args.initial_threshold, args.drop_threshold,
            args.use_probe_prefix_len_for_ans_prefix, args.output_dir, build_prompt
        )

    if args.compute_layer_stats:
        layer_stats = compute_layer_stats(correct_rows, wrong_rows, args.bucket_def)
        with open(os.path.join(args.output_dir, "layer_stats.json"), "w", encoding="utf-8") as f:
            json.dump(layer_stats, f, ensure_ascii=False, indent=2)
        
        # Print mid-layer stats
        mid = parse_bucket_def(args.bucket_def, layer_stats.get("num_layers", 0)).get("mid")
        if mid is not None:
            a, b = mid
            print(f"中层层段 mid={a}-{b} AUC(d) for ans→ans-prefix: "
                  f"{layer_stats['buckets']['mid']['ans_to_ans_prefix_auc']:.4f} "
                  f"({layer_stats['buckets']['mid']['ans_to_ans_prefix_d']:.4f})")

    # Multi-K prefix length comparison
    if args.prefix_lengths and args.reuse_per_layer_from:
        ks = [int(x) for x in args.prefix_lengths.split(',') if x.strip().isdigit()]
        if not os.path.exists(args.correct_converted) or not os.path.exists(args.wrong_converted):
            print("prefix_lengths 需要可用的 --correct_converted/--wrong_converted 以便逐K重算；当前路径不可用，跳过K对比。")
        else:
            try:
                if 'tokenizer' not in locals() or 'model' not in locals():
                    tokenizer, model, _ = load_model_and_tokenizer(args.model)
            except Exception as e:
                print(f"无法加载模型以执行K对比：{e}")
                ks = []
            for K in ks:
                print(f"按K={K} 重算per-layer并输出 layer_stats_K={K}.json …")
                c_rows = process_file(
                    args.correct_converted, tokenizer, model, K, None, args.initial_threshold, args.drop_threshold,
                    file_label=f"correct-K{K}", use_removed_as_prefix=False, want_per_layer=True,
                    build_prompt_fn=build_prompt,
                )
                w_rows = process_file(
                    args.wrong_converted, tokenizer, model, K, None, args.initial_threshold, args.drop_threshold,
                    file_label=f"wrong-K{K}", use_removed_as_prefix=False, want_per_layer=True,
                    build_prompt_fn=build_prompt,
                )
                stats_k = compute_layer_stats(c_rows, w_rows, args.bucket_def)
                with open(os.path.join(args.output_dir, f"layer_stats_K{K}.json"), "w", encoding="utf-8") as f:
                    json.dump(stats_k, f, ensure_ascii=False, indent=2)

    # Per-head analysis
    if args.per_head and not args.reuse_per_layer_from:
        try:
            tokenizer, model, device = load_model_and_tokenizer(args.model)
        except Exception as e:
            print(f"无法加载模型以执行per-head分析：{e}")
            model = None
        if model is not None:
            embedder_ph = load_embedder(args.embedding_model_path, device) if args.use_mlp_echo_removal else None
            export_per_head(
                args.correct_converted, "correct", tokenizer, model,
                args.answer_prefix_tokens, embedder_ph, args.initial_threshold, args.drop_threshold,
                args.use_probe_prefix_len_for_ans_prefix, args.output_dir, build_prompt
            )
            export_per_head(
                args.wrong_converted, "wrong", tokenizer, model,
                args.answer_prefix_tokens, embedder_ph, args.initial_threshold, args.drop_threshold,
                args.use_probe_prefix_len_for_ans_prefix, args.output_dir, build_prompt
            )

    # Compute final summary
    correct_sum = aggregate_metrics([r for r in correct_rows if "error" not in r])
    wrong_sum = aggregate_metrics([r for r in wrong_rows if "error" not in r])
    summary = {
        "correct": correct_sum,
        "wrong": wrong_sum,
        "difference": {k: correct_sum.get(k, float("nan")) - wrong_sum.get(k, float("nan")) for k in correct_sum.keys()},
    }
    with open(os.path.join(args.output_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("注意力核验报告已保存:")
    print("  - attention_metrics_correct.jsonl")
    print("  - attention_metrics_wrong.jsonl")
    print("  - attention_summary.json")
    if args.per_layer_trajectories:
        print("  - per_layer_correct.jsonl")
        print("  - per_layer_wrong.jsonl")


if __name__ == "__main__":
    main()
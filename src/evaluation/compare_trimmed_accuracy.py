#!/usr/bin/env python3
"""
Compare log probability differences between correct and incorrect answers
before and after removing repetitive sentences.

Usage example:
python src/evaluation/compare_trimmed_accuracy.py \
    --correct_file /path/to/results_correct.jsonl \
    --wrong_file /path/to/results_wrong.jsonl \
    --model /path/to/your/model \
    --output_dir ./comparison_results
"""

import argparse
import json
import os
import subprocess
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse existing per-token logp calculation and prompt construction to ensure consistency with Step 2
try:
    from src.evaluation.logp_trim_experiment import compute_per_token_logps, build_prompt
except Exception:
    compute_per_token_logps = None
    build_prompt = None

# Add parent directory to path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optionally import attention verification (located in train_repeat)
try:
    from src.evaluation.attention_analysis import run_attention_checks as _run_attention_checks
except Exception:
    _run_attention_checks = None


def run_logp_experiment(input_file: str, output_file: str, model: str) -> dict:
    """Call logp_trim_experiment.py to process a single file"""
    # First convert file format
    converted_file = input_file.replace('.jsonl', '_converted.jsonl')
    convert_cmd = [
        "python", "src/evaluation/convert_lm_eval_for_logp.py",
        "--input", input_file,
        "--output", converted_file
    ]
    
    print(f"转换文件格式: {' '.join(convert_cmd)}")
    env = os.environ.copy()
    # Get repository root from current file location
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = repo_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    convert_result = subprocess.run(
        convert_cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    
    if convert_result.returncode != 0:
        print(f"格式转换错误: {convert_result.stderr}")
        raise RuntimeError(f"格式转换失败，返回码: {convert_result.returncode}")
    
    # 然后调用 logp_trim_experiment.py
    cmd = [
        "python", "src/evaluation/logp_trim_experiment.py",
        "--input_file", converted_file,
        "--output_file", output_file,
        "--model", model
    ]
    
    # 统计样本数量
    with open(converted_file, 'r', encoding='utf-8') as f:
        sample_count = sum(1 for _ in f)
    
    print(f"📊 样本数量: {sample_count}")
    print(f"🚀 运行命令: {' '.join(cmd)}")
    # 让 tqdm 进度条能够实时显示，只捕获 stderr 用于错误处理
    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        cwd=repo_root,
        env=env,
    )
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        raise RuntimeError(f"logp_trim_experiment.py 执行失败，返回码: {result.returncode}")
    
    # 检查输出文件是否存在
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"logp_trim_experiment.py 执行成功但未生成输出文件: {output_file}")
    
    # 读取结果
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def compare_groups(correct_data: dict, wrong_data: dict) -> dict:
    """比较两组数据的统计指标"""
    correct_summary = correct_data['summary']
    wrong_summary = wrong_data['summary']
    
    comparison = {
        'correct_group': {
            'sample_count': correct_summary['total_samples'],
            'mean_logp_delta': correct_summary['mean_logp_delta'],
            'std_logp_delta': correct_summary['std_logp_delta'],
            'negative_delta_ratio': correct_summary['negative_delta_ratio']
        },
        'wrong_group': {
            'sample_count': wrong_summary['total_samples'],
            'mean_logp_delta': wrong_summary['mean_logp_delta'],
            'std_logp_delta': wrong_summary['std_logp_delta'],
            'negative_delta_ratio': wrong_summary['negative_delta_ratio']
        },
        'difference': {
            'mean_delta_diff': correct_summary['mean_logp_delta'] - wrong_summary['mean_logp_delta'],
            'negative_ratio_diff': correct_summary['negative_delta_ratio'] - wrong_summary['negative_delta_ratio']
        }
    }
    
    return comparison


def generate_report(comparison: dict, output_dir: str):
    """生成比较报告"""
    # 创建表格数据
    table_data = []
    
    for group_name, group_data in [('正确答案组', comparison['correct_group']), 
                                   ('错误答案组', comparison['wrong_group'])]:
        table_data.append({
            '组别': group_name,
            '样本数': group_data['sample_count'],
            '平均Δ对数概率': f"{group_data['mean_logp_delta']:.4f}",
            '标准差': f"{group_data['std_logp_delta']:.4f}",
            '负值比例': f"{group_data['negative_delta_ratio']:.2%}"
        })
    
    df = pd.DataFrame(table_data)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 生成文本报告
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 正确vs错误答案的对数概率差异比较 ===\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n=== 关键差异指标 ===\n")
        f.write(f"平均Δ对数概率差异: {comparison['difference']['mean_delta_diff']:.4f}\n")
        f.write(f"负值比例差异: {comparison['difference']['negative_ratio_diff']:.2%}\n")
        f.write("\n=== 解释 ===\n")
        f.write("- Δ对数概率 = 原始答案的对数概率 - 去重后答案的对数概率\n")
        f.write("- 负值表示去重后概率更高（重复降低了概率）\n")
        f.write("- 正值表示去重后概率更低（重复提高了概率）\n")
        
        if comparison['difference']['mean_delta_diff'] < 0:
            f.write("- 结果显示：错误答案组的重复问题更严重\n")
        else:
            f.write("- 结果显示：正确答案组的重复问题更严重\n")
    
    print(f"比较报告已保存:")
    print(f"  表格: {csv_path}")
    print(f"  报告: {report_path}")
    
    return df


# ============== 额外核验：长度分层与公共后缀对齐 ==============

def _load_model_and_tokenizer(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    return tokenizer, model, device


def _answer_token_logps(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    answer_text: str,
) -> Tuple[torch.Tensor, List[int]]:
    """返回答案段的逐 token logp 序列与对应 token ids（与 teacher-forcing 对齐）。"""
    # 构造与 logp_trim_experiment 完全一致的 prompt
    if build_prompt is None:
        prompt = (
            "You are an expert at solving math problems. Please think step by step.\n"
            f"Question: {question}\n"
            "Answer: <think>"
        )
    else:
        prompt = build_prompt(question)

    full_text = prompt + answer_text
    # 先拿到 prompt token 数用于切分答案段
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = int(prompt_ids.shape[1])

    # 逐 token logp（包含整个文本）
    token_logps = compute_per_token_logps(model, tokenizer, full_text) if compute_per_token_logps else _fallback_compute_per_token_logps(model, tokenizer, full_text)

    # teacher-forcing 的 labels 对齐到 ids[:, 1:]
    # 答案段的起始索引为 (prompt_len - 1)
    start = max(prompt_len - 1, 0)
    answer_logps = token_logps[start:]

    # 为了做后缀对齐，取与答案段严格一致的 token ids（切分自 full ids）
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    answer_token_ids = full_ids[0, prompt_len:].tolist()

    return answer_logps, answer_token_ids


def _fallback_compute_per_token_logps(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str) -> torch.Tensor:
    """在无法从 logp_trim_experiment 导入时，使用本地实现计算逐 token logp。"""
    ids = tokenizer(text, return_tensors="pt").input_ids
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    logits = out.logits[:, :-1, :]
    labels = ids[:, 1:]
    log_probs = logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered.squeeze(0)


def _longest_common_suffix_len(a: List[int], b: List[int]) -> int:
    i = len(a) - 1
    j = len(b) - 1
    count = 0
    while i >= 0 and j >= 0 and a[i] == b[j]:
        count += 1
        i -= 1
        j -= 1
    return count


def _lcs_length(a: List[int], b: List[int]) -> int:
    """Token 级 LCS 长度，用于估计最少删除数：len(a) - LCS(a,b)."""
    # 为节省内存，使用两行 DP
    if not a or not b:
        return 0
    # 确保 b 是较短的一维（优化）
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[len(b)]


def _bin_removed_tokens(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    return "21+"


def _extend_samples_with_metrics(
    samples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    group_label: str,
) -> List[Dict[str, Any]]:
    extended: List[Dict[str, Any]] = []
    for s in samples:
        if s.get("trimmed_prediction") in (None, ""):
            continue
        try:
            question = s.get("question", "")
            raw_ans = s.get("original_prediction", "")
            trim_ans = s.get("trimmed_prediction", "")

            raw_logps, raw_token_ids = _answer_token_logps(model, tokenizer, question, raw_ans)
            trim_logps, trim_token_ids = _answer_token_logps(model, tokenizer, question, trim_ans)

            n_raw = int(raw_logps.numel())
            n_trim = int(trim_logps.numel())

            # 公共后缀长度（严格相等的 token 后缀）与 LCS 删除估计
            suf_len = _longest_common_suffix_len(raw_token_ids, trim_token_ids)
            lcs_len = _lcs_length(raw_token_ids, trim_token_ids)
            removed_tokens = max(0, n_raw - lcs_len)
            added_tokens = max(0, n_trim - lcs_len)

            # Δ̄_per_token（均值差，与现有 logp_delta 一致），以及 Δ_sum
            raw_mean = float(raw_logps.mean().item()) if n_raw > 0 else 0.0
            trim_mean = float(trim_logps.mean().item()) if n_trim > 0 else 0.0
            delta_bar = raw_mean - trim_mean
            delta_sum = float(raw_logps.sum().item() - trim_logps.sum().item())

            # 每删除 1 个回声 token 的 Δ 代价：两种口径
            delta_sum_per_removed = (delta_sum / removed_tokens) if removed_tokens > 0 else None
            delta_bar_per_removed = (delta_bar / removed_tokens) if removed_tokens > 0 else None

            # 仅在公共后缀上的 Δ（总和差 与 均值差）
            if suf_len > 0:
                raw_suf = raw_logps[-suf_len:]
                trim_suf = trim_logps[-suf_len:]
                delta_suffix_sum = float(raw_suf.sum().item() - trim_suf.sum().item())
                delta_suffix_mean = float(raw_suf.mean().item() - trim_suf.mean().item())
            else:
                delta_suffix_sum = None
                delta_suffix_mean = None

            extended.append({
                "group": group_label,
                "idx": s.get("idx"),
                "is_correct": s.get("is_correct"),
                "raw_token_count": n_raw,
                "trim_token_count": n_trim,
                "removed_tokens": removed_tokens,
                "added_tokens": added_tokens,
                "suffix_len": suf_len,
                "delta_bar_per_token": delta_bar,
                "delta_sum": delta_sum,
                "delta_sum_per_removed": delta_sum_per_removed,
                "delta_bar_per_removed": delta_bar_per_removed,
                "delta_suffix_sum": delta_suffix_sum,
                "delta_suffix_mean": delta_suffix_mean,
            })
        except Exception as e:
            # 出错样本保留最小信息
            extended.append({
                "group": group_label,
                "idx": s.get("idx"),
                "error": f"{type(e).__name__}: {e}",
            })
    return extended


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _aggregate_and_save(extended_rows: List[Dict[str, Any]], output_dir: str):
    df = pd.DataFrame(extended_rows)
    df_path = os.path.join(output_dir, "per_sample_extended.csv")
    df.to_csv(df_path, index=False)

    # 删除前缀长度分布
    dist = (
        df.dropna(subset=["removed_tokens"])  # 过滤异常
          .assign(bin=lambda x: x["removed_tokens"].astype(int).map(_bin_removed_tokens))
          .groupby(["group", "bin"], as_index=False)["idx"].count()
          .rename(columns={"idx": "count"})
    )
    dist_path = os.path.join(output_dir, "removed_prefix_length_distribution.csv")
    dist.to_csv(dist_path, index=False)

    # 长度分层报告（按 bin）
    strat_cols = [
        "delta_bar_per_token",
        "delta_sum_per_removed",
        "delta_bar_per_removed",
        "delta_suffix_mean",
    ]
    strat = (
        df.dropna(subset=["removed_tokens"])  # 异常
          .assign(bin=lambda x: x["removed_tokens"].astype(int).map(_bin_removed_tokens))
          .groupby(["group", "bin"], as_index=False)[strat_cols].mean()
    )
    strat_path = os.path.join(output_dir, "length_stratified_summary.csv")
    strat.to_csv(strat_path, index=False)

    # 组间总览（未分层）
    overall = df.groupby("group", as_index=False)[strat_cols + ["removed_tokens", "raw_token_count", "trim_token_count", "suffix_len"]].mean()
    overall_path = os.path.join(output_dir, "overall_extra_checks_summary.csv")
    overall.to_csv(overall_path, index=False)

    print("额外核验报告已保存:")
    print(f"  每样本扩展指标: {df_path}")
    print(f"  删除前缀长度分布: {dist_path}")
    print(f"  长度分层汇总: {strat_path}")
    print(f"  组间总体汇总: {overall_path}")


def run_extra_checks(correct_data: dict, wrong_data: dict, model_path: str, output_dir: str):
    print("\n" + "="*60)
    print("🔎 额外核验：长度混杂与公共后缀对齐")
    print("="*60)

    tokenizer, model, device = _load_model_and_tokenizer(model_path)

    correct_rows = _extend_samples_with_metrics(correct_data.get("details", []), tokenizer, model, "correct")
    wrong_rows = _extend_samples_with_metrics(wrong_data.get("details", []), tokenizer, model, "wrong")

    all_rows = correct_rows + wrong_rows

    # 保存 JSONL 与 CSV
    _write_jsonl(os.path.join(output_dir, "extended_metrics_correct.jsonl"), correct_rows)
    _write_jsonl(os.path.join(output_dir, "extended_metrics_wrong.jsonl"), wrong_rows)

    _aggregate_and_save(all_rows, output_dir)


def main():
    parser = argparse.ArgumentParser(description="比较正确和错误答案的对数概率差异")
    parser.add_argument("--correct_file", required=True, help="正确答案的JSONL文件")
    parser.add_argument("--wrong_file", required=True, help="错误答案的JSONL文件")
    parser.add_argument("--model", required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", default="./comparison_results", help="输出目录")
    # 注意力核验（可选）
    parser.add_argument("--run_attention", action="store_true", help="在额外核验后，运行注意力质量核验")
    parser.add_argument("--attention_prefix_tokens", type=int, default=32, help="answer 前缀作为注意力目标的 token 数 K")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理正确答案组
    print("\n" + "="*60)
    print("🔍 Step 2.1: 处理正确答案组...")
    print("="*60)
    correct_output = os.path.join(args.output_dir, "correct_logp_results.json")
    correct_data = run_logp_experiment(args.correct_file, correct_output, args.model)
    
    if correct_data is None:
        print("❌ 处理正确答案组失败")
        return
    print("✅ 正确答案组处理完成")
    
    # 处理错误答案组
    print("\n" + "="*60)
    print("🔍 Step 2.2: 处理错误答案组...")
    print("="*60)
    wrong_output = os.path.join(args.output_dir, "wrong_logp_results.json")
    wrong_data = run_logp_experiment(args.wrong_file, wrong_output, args.model)
    
    if wrong_data is None:
        print("❌ 处理错误答案组失败")
        return
    print("✅ 错误答案组处理完成")
    
    # 比较两组结果
    print("生成比较报告...")
    comparison = compare_groups(correct_data, wrong_data)
    
    # 保存完整比较结果
    comparison_file = os.path.join(args.output_dir, "full_comparison.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    df = generate_report(comparison, args.output_dir)
    
    print("\n=== 快速预览 ===")
    print(df.to_string(index=False))

    # 额外核验：长度混杂与公共后缀对齐
    try:
        run_extra_checks(correct_data, wrong_data, args.model, args.output_dir)
    except Exception as e:
        print(f"[额外核验] 跳过（发生错误）：{type(e).__name__}: {e}")

    # 注意力核验：answer→question / answer→answer-prefix / tail→removed-prefix
    if args.run_attention:
        try:
            if _run_attention_checks is None:
                # 延迟导入一次，若上面失败则再试
                from src.evaluation.attention_analysis import run_attention_checks as _run_attention_checks  # type: ignore
            if _run_attention_checks is None:
                raise ImportError("attention_analysis.run_attention_checks 未找到")

            print("\n" + "="*60)
            print("🔎 注意力核验：answer→question / answer→prefix / tail→removed")
            print("="*60)
            _run_attention_checks(
                correct_data,
                wrong_data,
                args.model,
                args.output_dir,
                answer_prefix_tokens=args.attention_prefix_tokens,
            )
        except Exception as e:
            print(f"[注意力核验] 跳过（发生错误）：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
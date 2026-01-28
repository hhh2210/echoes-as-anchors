#!/usr/bin/env python3
"""
Compute echo-free conditional accuracy (trimmed accuracy)

Theoretical basis (rejection sampling identity):
  E_{y~τ_θ}[f(y)] = E_{y~π_θ}[f(y)·1_{y∈Y_trim}] / Z_x,
where f(y)=1[ExactMatch(y)=1] operates on original output y, 1_{y∈Y_trim} indicates "whether original output y is echo-free".

This implementation prioritizes MLP probe to approximate 1_{y∈Y_trim}, falling back to length heuristics when probe resources unavailable (not recommended for paper main results).

Path resolution rules:
  - Relative paths are first resolved against current working directory; if not found, fallback to relative to `train_repeat` root.
  - Recommend providing absolute paths for critical files (e.g., `--mlp_probe_path`) to avoid ambiguity.

Usage (MLP, recommended):
python src/evaluation/compute_trimmed_accuracy.py \
    --correct_file path/to/correct_logp_results.json \
    --wrong_file path/to/wrong_logp_results.json \
    --method mlp \
    --embedding_model_path path/to/Qwen3-Embedding-0.6B/ \
    --mlp_probe_path path/to/repeat_mlp.pt \
    --mlp_threshold 0.9 \
    --prefix_tokens 32
"""

import argparse
import json
import re
import os
import sys
from typing import Dict, Any, Tuple

import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

# 确保可以按包名导入 src.*（将 train_repeat 根目录加入 sys.path）
try:
    _HERE = os.path.abspath(os.path.dirname(__file__))
    _PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # /.../train_repeat
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    from src.data_processing.mlp_pipeline.inference import RepeatDetector
except Exception:
    RepeatDetector = None  # type: ignore


def _resolve_path_maybe(p: str, *search_roots: str) -> str:
    """解析路径：
    - 若为绝对路径且存在，直接返回；
    - 若为相对路径，优先以 CWD 解析；若不存在，再依次以给定根目录解析；
    - 返回第一个存在的绝对路径；若都不存在，返回原字符串（由上层决定是否报错）。
    """
    if not p:
        return p
    p_expanded = os.path.expanduser(p)
    if os.path.isabs(p_expanded) and os.path.exists(p_expanded):
        return p_expanded
    # try CWD
    if not os.path.isabs(p_expanded):
        cand = os.path.abspath(os.path.join(os.getcwd(), p_expanded))
        if os.path.exists(cand):
            return cand
    # try given roots
    for root in search_roots:
        cand = os.path.abspath(os.path.join(root, p_expanded))
        if os.path.exists(cand):
            return cand
    return p


def _ensure_existing_file(p: str, desc: str, *search_roots: str) -> str:
    """解析并校验文件存在性，否则抛出带候选路径的 FileNotFoundError。"""
    resolved = _resolve_path_maybe(p, *search_roots)
    if os.path.exists(resolved):
        return resolved
    tried: list[str] = []  # type: ignore[var-annotated]
    p_expanded = os.path.expanduser(p)
    tried.append(p_expanded)
    tried.append(os.path.abspath(os.path.join(os.getcwd(), p_expanded)))
    for root in search_roots:
        tried.append(os.path.abspath(os.path.join(root, p_expanded)))
    msg = (
        f"{desc} 未找到: {p}\n"
        f"已尝试的路径:\n  - " + "\n  - ".join(tried)
    )
    raise FileNotFoundError(msg)


def load_logp_results(input_file: str) -> Dict[str, Any]:
    """加载log-probability实验结果"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _extract_think_prefix(answer_text: str, prefix_tokens: int) -> str:
    """提取 <think> 内容的前 prefix_tokens 个词，若无 <think> 则退化为答案前缀。"""
    if not isinstance(answer_text, str):
        return ""
    m = re.search(r"<think>\s*(.*?)(?:</think>|$)", answer_text, re.DOTALL)
    content = m.group(1).strip() if m else answer_text.strip()
    words = content.split()
    if not words:
        return ""
    return " ".join(words[: max(1, int(prefix_tokens))])


def _load_probe(embedding_model_path: str, mlp_probe_path: str, mlp_hidden_dim: int) -> Tuple["SentenceTransformer", "RepeatDetector", torch.device]:  # type: ignore[name-defined]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if SentenceTransformer is None or RepeatDetector is None:
    #     raise ImportError("缺少 sentence-transformers 或 RepeatDetector 依赖。")
    embedder = SentenceTransformer(embedding_model_path, device=device)
    input_dim = embedder.get_sentence_embedding_dimension() * 2
    mlp = RepeatDetector(input_dim, hidden_dim=mlp_hidden_dim).to(device)
    # 解析并校验 MLP 探针权重路径
    resolved_probe = _ensure_existing_file(
        mlp_probe_path,
        "MLP 探针权重文件",
        _PKG_ROOT,
    )
    print(f"[Info] 使用 MLP 探针权重: {resolved_probe}")
    mlp.load_state_dict(torch.load(resolved_probe, map_location=device))
    mlp.eval()
    return embedder, mlp, device


def _probe_is_repeat(question: str, answer_text: str, embedder: "SentenceTransformer", mlp: "RepeatDetector", device: torch.device, prefix_tokens: int) -> float:  # type: ignore[name-defined]
    prefix_text = _extract_think_prefix(answer_text, prefix_tokens)
    if not question or not prefix_text:
        return 0.0
    q_emb = embedder.encode(question, convert_to_tensor=True, device=device)
    p_emb = embedder.encode(prefix_text, convert_to_tensor=True, device=device)
    if len(q_emb.shape) == 1:
        q_emb = q_emb.unsqueeze(0)
    if len(p_emb.shape) == 1:
        p_emb = p_emb.unsqueeze(0)
    features = torch.cat([q_emb, p_emb], dim=1).to(device)
    with torch.no_grad():
        logits = mlp(features)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def compute_combined_accuracy_length_heuristic(correct_data: Dict[str, Any], wrong_data: Dict[str, Any]) -> Dict[str, float]:
    """长度回退：以“未发生剪裁”近似 echo-free，仅用于回退。
    echo-free 指示：len(trimmed) >= len(original)
    """
    correct_details = correct_data.get('details', [])
    wrong_details = wrong_data.get('details', [])
    
    all_samples = correct_details + wrong_details
    total_samples = len(all_samples)
    
    if total_samples == 0:
        return {}
    
    original_accuracy = len(correct_details) / total_samples

    echo_free_count = 0
    echo_free_correct = 0

    for sample in all_samples:
        if sample.get('error'):
            continue
        orig = sample.get('original_prediction', '')
        trim = sample.get('trimmed_prediction', '')
        is_echo_free = len(trim) >= len(orig)
        if is_echo_free:
            echo_free_count += 1
            if bool(sample.get('is_correct')):
                echo_free_correct += 1

    Z_x = echo_free_count / total_samples if total_samples > 0 else 0.0
    trimmed_accuracy = (echo_free_correct / echo_free_count) if echo_free_count > 0 else original_accuracy

    return {
        'method': 'length',
        'total_samples': total_samples,
        'correct_samples': len(correct_details),
        'wrong_samples': len(wrong_details),
        'original_accuracy': original_accuracy,
        'echo_free_count': echo_free_count,
        'echo_free_correct': echo_free_correct,
        'acceptance_rate_Zx': Z_x,
        'trimmed_accuracy': trimmed_accuracy,
        'accuracy_change': trimmed_accuracy - original_accuracy,
    }


def compute_combined_accuracy_mlp(
    correct_data: Dict[str, Any],
    wrong_data: Dict[str, Any],
    embedding_model_path: str,
    mlp_probe_path: str,
    mlp_hidden_dim: int = 32,
    mlp_threshold: float = 0.9,
    prefix_tokens: int = 32,
) -> Dict[str, float]:
    """用 MLP 探针估计 echo-free 条件准确率：P(EM=1 | echo-free)。"""
    correct_details = correct_data.get('details', [])
    wrong_details = wrong_data.get('details', [])
    all_samples = correct_details + wrong_details
    total_samples = len(all_samples)
    if total_samples == 0:
        return {}

    embedder, mlp, device = _load_probe(embedding_model_path, mlp_probe_path, mlp_hidden_dim)

    original_accuracy = len(correct_details) / total_samples

    echo_free_count = 0
    echo_free_correct = 0

    for sample in all_samples:
        if sample.get('error'):
            continue
        question = sample.get('question', '')
        answer = sample.get('original_prediction', '')
        prob_repeat = _probe_is_repeat(question, answer, embedder, mlp, device, prefix_tokens)
        is_echo_free = (prob_repeat <= float(mlp_threshold))
        if is_echo_free:
            echo_free_count += 1
            if bool(sample.get('is_correct')):
                echo_free_correct += 1

    Z_x = echo_free_count / total_samples if total_samples > 0 else 0.0
    trimmed_accuracy = (echo_free_correct / echo_free_count) if echo_free_count > 0 else original_accuracy

    return {
        'method': 'mlp',
        'total_samples': total_samples,
        'correct_samples': len(correct_details),
        'wrong_samples': len(wrong_details),
        'original_accuracy': original_accuracy,
        'echo_free_count': echo_free_count,
        'echo_free_correct': echo_free_correct,
        'acceptance_rate_Zx': Z_x,
        'trimmed_accuracy': trimmed_accuracy,
        'accuracy_change': trimmed_accuracy - original_accuracy,
        'mlp_threshold': float(mlp_threshold),
        'prefix_tokens': int(prefix_tokens),
    }


def print_results(results: Dict[str, float]):
    """打印结果报告"""
    print(f"\n{'='*60}")
    method = results.get('method', 'mlp')
    title = "📊 echo-free 条件准确率（基于拒绝采样恒等式）"
    if method == 'length':
        title += " [长度回退]"
    print(title)
    print(f"{'='*60}")
    
    print(f"\n基础统计:")
    print(f"  总样本数: {results['total_samples']}")
    print(f"  - 正确组: {results['correct_samples']}")
    print(f"  - 错误组: {results['wrong_samples']}")
    print(f"  原始准确率: {results['original_accuracy']:.2%}")
    
    print(f"\n条件集统计（echo-free）:")
    print(f"  echo-free 样本数: {results['echo_free_count']}")
    print(f"  echo-free ∧ 正确: {results['echo_free_correct']}")
    print(f"  接受率 Z_x: {results['acceptance_rate_Zx']:.2%}")
    
    print(f"\n准确率计算（Rejection Sampling）:")
    print(f"  公式: Acc_trim = E[f(y)·1_{'{'}echo-free{'}'}] / Z_x")
    print(f"  分母 Z_x: {results['acceptance_rate_Zx']:.4f}")
    print(f"  剪裁后准确率: {results['trimmed_accuracy']:.2%}")
    print(f"  准确率变化: {results['accuracy_change']:+.2%}")


def save_results(results: Dict[str, Any], output_file: str):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="计算 echo-free 条件准确率（trimmed accuracy）")
    parser.add_argument("--correct_file", required=True, 
                       help="正确组的logp_results.json文件")
    parser.add_argument("--wrong_file", required=True,
                       help="错误组的logp_results.json文件")
    parser.add_argument("--output_file", default="trimmed_accuracy_results.json",
                       help="输出文件路径")
    parser.add_argument("--method", choices=["mlp", "length"], default="mlp",
                       help="估计方法：mlp（推荐）或 length（回退）")
    parser.add_argument("--embedding_model_path", type=str, default=None,
                       help="SentenceTransformer 嵌入模型路径（method=mlp 需要）")
    parser.add_argument("--mlp_probe_path", type=str, default=None,
                       help="训练好的 MLP 探针路径（method=mlp 需要）")
    parser.add_argument("--mlp_hidden_dim", type=int, default=32,
                       help="MLP 隐藏层维度（需与训练一致）")
    parser.add_argument("--mlp_threshold", type=float, default=0.9,
                       help="探针阈值，越高越保守（默认0.9）")
    parser.add_argument("--prefix_tokens", type=int, default=32,
                       help="用于探针的 <think> 前缀词数 K")
    
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    correct_data = load_logp_results(args.correct_file)
    wrong_data = load_logp_results(args.wrong_file)
    
    # 计算 echo-free 条件准确率
    print("\n计算 echo-free 条件准确率...")
    use_mlp = (args.method == "mlp")
    combined_results: Dict[str, Any]
    if use_mlp and args.embedding_model_path and args.mlp_probe_path:
        combined_results = compute_combined_accuracy_mlp(
            correct_data,
            wrong_data,
            embedding_model_path=args.embedding_model_path,
            mlp_probe_path=args.mlp_probe_path,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_threshold=args.mlp_threshold,
            prefix_tokens=args.prefix_tokens,
        )
    else:
        if use_mlp:
            print("[警告] 未提供 embedding_model_path 或 mlp_probe_path，回退到长度启发式。")
        combined_results = compute_combined_accuracy_length_heuristic(correct_data, wrong_data)
    
    # 打印结果
    print_results(combined_results)
    
    # 分析Delta分布
    print(f"\n📈 各组Delta统计:")
    print(f"\n正确组:")
    print(f"  平均Delta: {correct_data['summary']['mean_logp_delta']:.4f}")
    print(f"  负Delta比例: {correct_data['summary']['negative_delta_ratio']:.2%}")
    print(f"\n错误组:")
    print(f"  平均Delta: {wrong_data['summary']['mean_logp_delta']:.4f}")
    print(f"  负Delta比例: {wrong_data['summary']['negative_delta_ratio']:.2%}")
    
    # 保存完整结果
    full_results = {
        'combined': combined_results,
        'correct_summary': correct_data['summary'],
        'wrong_summary': wrong_data['summary']
    }
    
    save_results(full_results, args.output_file)


if __name__ == "__main__":
    main()
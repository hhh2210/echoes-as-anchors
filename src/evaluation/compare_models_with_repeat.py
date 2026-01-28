#!/usr/bin/env python3
"""
Evaluate math accuracy of models before and after SFT and calculate their Repeat frequency

Uses lm-evaluation-harness for mathematical evaluation and trained MLP probe to detect repetition frequency in generated content.

Example:
    CUDA_VISIBLE_DEVICES=4,5 python src/evaluation/compare_models_with_repeat.py \
        --model_paths /path/to/deepseek-r1-distill-sft-full/ \
                      /path/to/DeepSeek-R1-Distill-Llama-8B/ \
        --embedding_model_path /path/to/Qwen3-Embedding-0.6B/ \
        --mlp_probe_path train_mlp/models/repeat_mlp.pt \
        --tasks gsm8k \
        --output_dir results/evaluation_results
"""

import os
import json
import argparse
import pandas as pd
import torch
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from torch import nn
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.harness_eval import evaluate_model, parse_harness_results


class RepeatDetector(nn.Module):
    """MLP probe for detecting repetition patterns"""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def analyze_repeat_frequency_and_save_tokens(
    samples_file: str,
    embedder: SentenceTransformer,
    mlp_probe: RepeatDetector,
    device: torch.device,
    output_dir: str,
    model_name: str,
    prefix_tokens: int = 32,
    mlp_threshold: float = 0.5
) -> Dict[str, float]:
    """
    分析 lm-evaluation-harness 生成的样本中的重复频率，并保存 think tokens
    
    Args:
        samples_file: harness 生成的样本文件路径 (JSONL 格式)
        embedder: 句子嵌入模型
        mlp_probe: 训练好的 MLP 探针
        device: 计算设备
        output_dir: 输出目录
        model_name: 模型名称
        prefix_tokens: 用于检测的前缀 token 数
        mlp_threshold: MLP 判断阈值
        
    Returns:
        包含重复统计信息的字典
    """
    total_samples = 0
    samples_with_think = 0
    repeat_count = 0
    repeat_scores = []
    
    # 保存所有样本的 Repeat tokens 和分析结果
    repeat_tokens_data = []
    detailed_results = []
    
    # 读取 JSONL 文件
    with open(samples_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            total_samples += 1
            
            # 提取问题
            # 新版 harness 使用不同的字段结构
            doc = sample.get('doc', {})
            question = doc.get('question', '')
            
            # 如果没有在 doc 中找到问题，尝试从其他字段提取
            if not question:
                # 从 arguments 字段提取
                if 'arguments' in sample:
                    args = sample['arguments']
                    if isinstance(args, list) and len(args) > 0:
                        # 从 prompt 中提取问题
                        prompt = args[0] if isinstance(args[0], str) else str(args[0])
                        question_match = re.search(r'Question:\s*(.*?)(?:\n|$)', prompt, re.DOTALL)
                        if question_match:
                            question = question_match.group(1).strip()
                # 从 query 字段提取
                elif 'query' in sample:
                    question = sample['query']
            
            # 获取生成的内容和思考内容
            generated = ""
            think_content = ""
            
            # 1. 优先查找 thoughts 字段 (新版 harness 格式)
            if 'thoughts' in sample:
                think_content = sample['thoughts']
                generated = sample.get('predicted', '') or sample.get('response', '')
            
            # 2. 从 resps 字段提取 (传统格式)
            elif 'resps' in sample:
                resps = sample['resps']
                if isinstance(resps, list) and len(resps) > 0:
                    if isinstance(resps[0], list) and len(resps[0]) > 0:
                        generated = resps[0][0] if isinstance(resps[0][0], str) else str(resps[0][0])
                    else:
                        generated = resps[0] if isinstance(resps[0], str) else str(resps[0])
                
                # 从生成内容中提取 thinking
                think_match = re.search(r'<think>\s*(.*?)(?:</think>|$)', generated, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
            
            # 3. 其他字段
            elif 'target' in sample or 'output' in sample:
                generated = sample.get('target', '') or sample.get('output', '')
                think_match = re.search(r'<think>\s*(.*?)(?:</think>|$)', generated, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
            
            # 如果没有找到 think 内容，但有生成内容，使用其他策略
            if not think_content and generated:
                # 使用 #### 作为分隔符
                if '####' in generated:
                    think_content = generated.split('####')[0].strip()
                # 或使用完整生成文本
                elif generated.strip():
                    think_content = generated.strip()
            
            # 保存原始 think tokens
            sample_data = {
                'sample_id': line_idx,
                'question': question,
                'think_content': think_content,
                'full_generated': generated,
                'is_repeat': False,
                'repeat_score': 0.0
            }
            
            if not think_content or not question:
                repeat_tokens_data.append(sample_data)
                continue
                
            samples_with_think += 1
            
            # 使用 MLP 检测是否重复
            q_embedding = embedder.encode(question, convert_to_tensor=True, device=device)
            
            # 使用前 prefix_tokens 个词作为前缀
            think_words = think_content.split()
            prefix_text = " ".join(think_words[:prefix_tokens])
            p_embedding = embedder.encode(prefix_text, convert_to_tensor=True, device=device)
            
            # 确保嵌入是 2D 的
            if len(q_embedding.shape) == 1:
                q_embedding = q_embedding.unsqueeze(0)
            if len(p_embedding.shape) == 1:
                p_embedding = p_embedding.unsqueeze(0)
                
            features = torch.cat([q_embedding, p_embedding], dim=1).to(device)
            
            # 获取 MLP 预测
            with torch.no_grad():
                logits = mlp_probe(features)
                prob_is_repeat = torch.sigmoid(logits).item()
                repeat_scores.append(prob_is_repeat)
                
                sample_data['repeat_score'] = prob_is_repeat
                
                if prob_is_repeat > mlp_threshold:
                    repeat_count += 1
                    sample_data['is_repeat'] = True
            
            repeat_tokens_data.append(sample_data)
    
    # 保存 think tokens 到文件
    tokens_output_file = os.path.join(output_dir, f"{model_name}_repeat_tokens.jsonl")
    with open(tokens_output_file, 'w', encoding='utf-8') as f:
        for item in repeat_tokens_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Repeat tokens 已保存到: {tokens_output_file}")
    
    # 计算统计信息
    stats = {
        'total_samples': total_samples,
        'samples_with_think': samples_with_think,
        'repeat_count': repeat_count,
        'repeat_frequency': repeat_count / samples_with_think if samples_with_think > 0 else 0,
        'avg_repeat_score': sum(repeat_scores) / len(repeat_scores) if repeat_scores else 0,
        'max_repeat_score': max(repeat_scores) if repeat_scores else 0,
        'min_repeat_score': min(repeat_scores) if repeat_scores else 0,
        'num_scores': len(repeat_scores),
        'repeat_tokens_file': tokens_output_file
    }
    
    return stats


def evaluate_model_with_repeat_analysis(
    model_path: str,
    output_dir: str,
    embedder: SentenceTransformer,
    mlp_probe: RepeatDetector,
    device: torch.device,
    tasks: str = "gsm8k",
    batch_size: str = "auto",
    use_multi_gpu: bool = False,
    tensor_parallel_size: int = 1,
    temperature: Optional[float] = None,
    mlp_threshold: float = 0.9
) -> Dict:
    """评估单个模型的数学准确率并分析重复频率"""
    
    model_name = Path(model_path).name
    model_output_dir = os.path.join(output_dir, model_name)
    
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 检查是否需要使用子目录（针对 llama_factory 的输出）
    actual_model_path = model_path
    if os.path.exists(os.path.join(model_path, "checkpoint-201")):
        checkpoint_path = os.path.join(model_path, "checkpoint-201")
        if os.path.exists(os.path.join(checkpoint_path, "tokenizer.json")):
            actual_model_path = checkpoint_path
            print(f"使用 checkpoint 目录: {actual_model_path}")
    
    # 1. 运行 harness 评估
    success, output = evaluate_model(
        model_path=actual_model_path,
        output_dir=model_output_dir,
        tasks=tasks,
        batch_size=batch_size,
        use_multi_gpu=use_multi_gpu,
        tensor_parallel_size=tensor_parallel_size,
        temperature=temperature
    )
    
    result = {
        'model_path': model_path,
        'model_name': model_name,
        'success': success,
        'timestamp': datetime.now().isoformat()
    }
    
    if not success:
        result['error'] = output
        return result
    
    # 在解析结果前增加延时，确保文件写入完成
    print("等待 15 秒，以确保文件写入完成...")
    time.sleep(15)
    
    # 2. 解析准确率结果
    metrics = parse_harness_results(model_output_dir, task_name=tasks)
    if metrics:
        result.update(metrics)
        print(f"✓ {tasks.upper()} 准确率: {metrics['accuracy']:.4f}")
    else:
        print("✗ 准确率解析失败")
        return result
    
    # 3. 分析重复频率
    # 查找样本文件 - harness 使用 _samples.jsonl 格式
    samples_file = None
    
    # 首先检查编码后的子目录（harness 的新版本行为）
    for root, dirs, files in os.walk(model_output_dir):
        for file in files:
            # 查找包含 samples 和 jsonl 的文件
            if 'samples' in file and file.endswith('.jsonl'):
                samples_file = os.path.join(root, file)
                print(f"找到样本文件: {samples_file}")
                break
        if samples_file:
            break
    
    # 如果还没找到，尝试标准命名
    if not samples_file:
        possible_names = [
            f"{tasks}_samples.jsonl",
            f"{tasks}_eval_samples.jsonl",
            "samples.jsonl"
        ]
        
        for name in possible_names:
            candidate = os.path.join(model_output_dir, name)
            if os.path.exists(candidate):
                samples_file = candidate
                break
    
    if samples_file and os.path.exists(samples_file):
        print(f"分析重复频率并保存 think tokens...")
        repeat_stats = analyze_repeat_frequency_and_save_tokens(
            samples_file,
            embedder,
            mlp_probe,
            device,
            output_dir,  # 传入输出目录
            model_name,  # 传入模型名称
            prefix_tokens=32,
            mlp_threshold=mlp_threshold
        )
        result['repeat_stats'] = repeat_stats
        print(f"✓ 重复频率: {repeat_stats['repeat_frequency']:.4f}")
        print(f"  - 重复样本数: {repeat_stats['repeat_count']}/{repeat_stats['samples_with_think']}")
        print(f"  - 总样本数: {repeat_stats['total_samples']}")
        print(f"  - 平均重复分数: {repeat_stats['avg_repeat_score']:.4f}")
    else:
        print("✗ 未找到样本文件，无法分析重复频率")
        result['repeat_stats'] = None
    
    return result


def create_comparison_report(results: List[Dict], output_dir: str):
    """创建模型比较报告 (仅 TXT)"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""模型评估与重复频率分析报告
{'='*80}
评估时间: {timestamp}
评估框架: lm-evaluation-harness
评估任务: GSM-8K (数学推理)
重复检测: MLP 探针

模型比较结果:
{'-'*80}
"""
    
    # 创建比较表格
    table_header = f"{'模型':<35} {'Strict Match':<15} {'Flexible Extract':<18} {'重复频率':<12} {'平均分数':<12}"
    report_content += table_header + "\n"
    report_content += "-" * 95 + "\n"
    
    for result in results:
        model_name = result['model_name'][:33]  # 截断长名称
        
        if result['success'] and 'accuracy' in result:
            strict_match = f"{result.get('strict_match', 0.0):.4f}"
            flexible_extract = f"{result.get('flexible_extract', 0.0):.4f}"
            
            if result.get('repeat_stats'):
                repeat_freq = f"{result['repeat_stats']['repeat_frequency']:.4f}"
                avg_score = f"{result['repeat_stats']['avg_repeat_score']:.4f}"
            else:
                repeat_freq = "N/A"
                avg_score = "N/A"
        else:
            strict_match = "失败"
            flexible_extract = "失败"
            repeat_freq = "N/A"
            avg_score = "N/A"
        
        report_content += f"{model_name:<35} {strict_match:<15} {flexible_extract:<18} {repeat_freq:<12} {avg_score:<12}\n"
    
    # 添加详细分析
    report_content += f"\n{'='*80}\n详细分析:\n\n"
    
    successful_results = [r for r in results if r['success'] and 'flexible_extract' in r]
    if successful_results:
        best_flex = max(successful_results, key=lambda x: x['flexible_extract'])
        worst_flex = min(successful_results, key=lambda x: x['flexible_extract'])
        
        report_content += f"最高 Flexible Extract: {best_flex['model_name']} - {best_flex['flexible_extract']:.4f}\n"
        report_content += f"最低 Flexible Extract: {worst_flex['model_name']} - {worst_flex['flexible_extract']:.4f}\n"
        
        # 分析重复频率
        results_with_repeat = [r for r in successful_results if r.get('repeat_stats')]
        if results_with_repeat:
            highest_repeat = max(results_with_repeat, key=lambda x: x['repeat_stats']['repeat_frequency'])
            lowest_repeat = min(results_with_repeat, key=lambda x: x['repeat_stats']['repeat_frequency'])
            
            report_content += f"\n最高重复频率: {highest_repeat['model_name']} - {highest_repeat['repeat_stats']['repeat_frequency']:.4f}\n"
            report_content += f"最低重复频率: {lowest_repeat['model_name']} - {lowest_repeat['repeat_stats']['repeat_frequency']:.4f}\n"
    
    # 保存报告
    report_path = os.path.join(output_dir, "model_comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n比较报告已保存: {report_path}")

    # 不再保存 JSON 和 CSV
    # print("已禁用 JSON 和 CSV 输出。")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model math accuracy and calculate repetition frequency")
    parser.add_argument("--model_paths", nargs='+', required=True,
                       help="List of model paths to evaluate")
    parser.add_argument("--embedding_model_path", type=str, default="Qwen3-Embedding-0.6B",
                       help="Sentence embedding model path")
    parser.add_argument("--mlp_probe_path", type=str, default="train_mlp/models/repeat_mlp.pt",
                       help="Path to trained MLP probe")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--tasks", type=str, default="gsm8k",
                       help="评估任务（默认: gsm8k）")
    parser.add_argument("--batch_size", type=str, default="auto",
                       help="评估批次大小")
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="使用多 GPU 评估")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="张量并行大小")
    parser.add_argument("--temperature", type=float, default=None,
                       help="生成温度")
    parser.add_argument("--mlp_hidden_dim", type=int, default=32,
                       help="MLP 隐藏层维度")
    parser.add_argument("--mlp-threshold", dest="mlp_threshold", type=float, default=0.9,
                       help="Threshold in [0,1] to decide repetition; higher is more conservative (default: 0.9)")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检测可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用 GPU")
    
    # 如果有多个 GPU，自动启用多 GPU 支持
    if num_gpus > 1 and not args.use_multi_gpu:
        print(f"自动启用多 GPU 支持，使用 {num_gpus} 个 GPU")
        args.use_multi_gpu = True
        if args.tensor_parallel_size == 1:
            args.tensor_parallel_size = num_gpus
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载嵌入模型和 MLP 探针
    print("加载嵌入模型和 MLP 探针...")
    embedder = SentenceTransformer(args.embedding_model_path, device=device)
    
    input_dim = embedder.get_sentence_embedding_dimension() * 2
    mlp_probe = RepeatDetector(input_dim, hidden_dim=args.mlp_hidden_dim).to(device)
    mlp_probe.load_state_dict(torch.load(args.mlp_probe_path, map_location=device))
    mlp_probe.eval()
    print("✓ 模型加载完成")
    
    # 评估所有模型
    results = []
    for model_path in args.model_paths:
        result = evaluate_model_with_repeat_analysis(
            model_path=model_path,
            output_dir=args.output_dir,
            embedder=embedder,
            mlp_probe=mlp_probe,
            device=device,
            tasks=args.tasks,
            batch_size=args.batch_size,
            use_multi_gpu=args.use_multi_gpu,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=args.temperature,
            mlp_threshold=args.mlp_threshold
        )
        results.append(result)
    
    # 创建比较报告
    create_comparison_report(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("所有评估完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
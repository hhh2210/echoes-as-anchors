#!/usr/bin/env python3
"""
将GSM8K lm-evaluation-harness结果按正确/错误答案分组保存。

用法示例:
python src/evaluation/split_gsm8k_by_accuracy.py \
    --input_file ./analysis_results_20250730_182827/evaluations/__data1__public__models__DeepSeek-R1-Distill-Llama-8B__/samples_gsm8k_2025-07-30T18-33-53.637932.jsonl \
    --output_dir ./analysis_results_20250730_182827/
"""

import argparse
import json
import os
from pathlib import Path


def split_gsm8k_samples(input_file: str, output_dir: str):
    """
    读取GSM8K样本文件，按正确/错误分组保存
    
    Args:
        input_file: GSM8K样本文件路径
        output_dir: 输出目录
    """
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    print(f"处理文件: {input_file}")
    
    correct_samples = []
    wrong_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                sample = json.loads(line.strip())
                
                # 根据exact_match字段分组
                if sample.get('exact_match', 0) >= 1.0:
                    correct_samples.append(sample)
                else:
                    wrong_samples.append(sample)
                    
            except json.JSONDecodeError as e:
                print(f"解析JSON错误 (行 {line_idx + 1}): {e}")
                continue
    
    print(f"正确答案样本数: {len(correct_samples)}")
    print(f"错误答案样本数: {len(wrong_samples)}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分组结果
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    correct_file = os.path.join(output_dir, f"{input_basename}_correct.jsonl")
    wrong_file = os.path.join(output_dir, f"{input_basename}_wrong.jsonl")
    
    with open(correct_file, 'w', encoding='utf-8') as f:
        for sample in correct_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(wrong_file, 'w', encoding='utf-8') as f:
        for sample in wrong_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ 正确答案保存至: {correct_file}")
    print(f"✓ 错误答案保存至: {wrong_file}")
    
    return correct_file, wrong_file


def main():
    parser = argparse.ArgumentParser(description="将GSM8K lm-eval结果按正确/错误答案分组")
    parser.add_argument("--input_file", required=True, help="GSM8K样本文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    
    try:
        correct_file, wrong_file = split_gsm8k_samples(args.input_file, args.output_dir)
        print(f"\n🎉 分组完成！")
        print(f"正确答案文件: {correct_file}")
        print(f"错误答案文件: {wrong_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
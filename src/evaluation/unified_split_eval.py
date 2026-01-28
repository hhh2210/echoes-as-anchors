#!/usr/bin/env python3
"""
统一的评估结果分割工具，支持多种分割模式：
1. simple: 简单的 exact_match 分割（正确/错误）
2. flexible: 考虑 flexible-extract 的分割（正确/错误/无效提取）
3. auto: 自动检测并选择合适的模式

用法示例:
python unified_split_eval.py --input_file results.jsonl --output_dir ./split_results --mode simple
python unified_split_eval.py --input_file results.jsonl --output_dir ./split_results --mode flexible
python unified_split_eval.py --input_file results.jsonl --output_dir ./split_results  # 默认auto模式
"""

import argparse
import json
import os
import gzip
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum


class SplitMode(Enum):
    """分割模式枚举"""
    SIMPLE = "simple"      # 简单的exact_match分割
    FLEXIBLE = "flexible"  # 考虑flexible-extract的分割
    AUTO = "auto"         # 自动检测模式


def load_lines(path: str):
    """从.jsonl或.jsonl.gz文件加载行"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def detect_mode(samples: List[Dict]) -> SplitMode:
    """自动检测应该使用哪种分割模式"""
    # 检查前几个样本是否包含filtered_resps字段
    check_count = min(10, len(samples))
    has_filtered_resps = 0
    
    for sample in samples[:check_count]:
        if 'filtered_resps' in sample:
            has_filtered_resps += 1
    
    # 如果大部分样本有filtered_resps字段，使用flexible模式
    if has_filtered_resps >= check_count * 0.5:
        print(f"📊 检测到filtered_resps字段，使用flexible模式")
        return SplitMode.FLEXIBLE
    else:
        print(f"📊 未检测到filtered_resps字段，使用simple模式")
        return SplitMode.SIMPLE


def split_simple(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """简单的exact_match分割"""
    correct_samples = []
    wrong_samples = []
    
    for sample in samples:
        em = sample.get("exact_match", 0)
        if em >= 1.0 or em is True:
            correct_samples.append(sample)
        else:
            wrong_samples.append(sample)
    
    return correct_samples, wrong_samples


def split_flexible(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """考虑flexible-extract的分割"""
    correct_samples = []
    wrong_samples = []
    invalid_samples = []
    
    for sample in samples:
        exact_match = sample.get('exact_match', 0.0)
        filtered_resps = sample.get('filtered_resps', [])
        
        # 判断分类
        if filtered_resps == ['[invalid]']:
            invalid_samples.append(sample)
        elif exact_match >= 1.0 or exact_match is True:
            correct_samples.append(sample)
        else:
            wrong_samples.append(sample)
    
    return correct_samples, wrong_samples, invalid_samples


def save_jsonl(data: List[Dict], filepath: str) -> None:
    """保存数据到JSONL文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="统一的评估结果分割工具，支持多种分割模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明:
  simple   - 简单的exact_match分割（正确/错误）
  flexible - 考虑flexible-extract的分割（正确/错误/无效提取）
  auto     - 自动检测并选择合适的模式（默认）

示例:
  %(prog)s --input_file results.jsonl --output_dir ./split --mode simple
  %(prog)s --input_file results.jsonl --output_dir ./split --mode flexible
  %(prog)s --input_file results.jsonl --output_dir ./split  # 自动检测模式
        """
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入的JSONL文件路径（支持.jsonl和.jsonl.gz）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "flexible", "auto"],
        default="auto",
        help="分割模式 (默认: auto)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📁 处理文件: {args.input_file}")
    
    # 加载所有样本
    samples = list(load_lines(args.input_file))
    total_count = len(samples)
    print(f"📊 总样本数: {total_count}")
    
    # 确定分割模式
    mode = SplitMode(args.mode)
    if mode == SplitMode.AUTO:
        mode = detect_mode(samples)
    else:
        print(f"📊 使用指定模式: {mode.value}")
    
    # 生成输出文件基础名
    base_name = Path(args.input_file).stem
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]
    
    # 根据模式执行分割
    if mode == SplitMode.SIMPLE:
        correct_samples, wrong_samples = split_simple(samples)
        
        # 保存结果
        correct_file = os.path.join(args.output_dir, f"{base_name}_correct.jsonl")
        wrong_file = os.path.join(args.output_dir, f"{base_name}_wrong.jsonl")
        
        if correct_samples:
            save_jsonl(correct_samples, correct_file)
            print(f"✅ 正确答案: {len(correct_samples)} ({len(correct_samples)/total_count*100:.1f}%) → {correct_file}")
        
        if wrong_samples:
            save_jsonl(wrong_samples, wrong_file)
            print(f"❌ 错误答案: {len(wrong_samples)} ({len(wrong_samples)/total_count*100:.1f}%) → {wrong_file}")
    
    elif mode == SplitMode.FLEXIBLE:
        correct_samples, wrong_samples, invalid_samples = split_flexible(samples)
        
        # 保存结果
        correct_file = os.path.join(args.output_dir, f"{base_name}_correct.jsonl")
        wrong_file = os.path.join(args.output_dir, f"{base_name}_wrong.jsonl")
        invalid_file = os.path.join(args.output_dir, f"{base_name}_invalid.jsonl")
        
        if correct_samples:
            save_jsonl(correct_samples, correct_file)
            print(f"✅ 正确答案: {len(correct_samples)} ({len(correct_samples)/total_count*100:.1f}%) → {correct_file}")
        
        if wrong_samples:
            save_jsonl(wrong_samples, wrong_file)
            print(f"❌ 错误答案: {len(wrong_samples)} ({len(wrong_samples)/total_count*100:.1f}%) → {wrong_file}")
        
        if invalid_samples:
            save_jsonl(invalid_samples, invalid_file)
            print(f"⚠️  提取失败: {len(invalid_samples)} ({len(invalid_samples)/total_count*100:.1f}%) → {invalid_file}")
    
    print(f"\n🎉 分割完成！")
    
    # 详细模式下显示更多统计信息
    if args.verbose:
        print(f"\n📈 详细统计:")
        print(f"  输入文件: {args.input_file}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  使用模式: {mode.value}")
        print(f"  总样本数: {total_count}")


if __name__ == "__main__":
    main()
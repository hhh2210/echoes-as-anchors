#!/usr/bin/env python3
"""
Direct evaluation script for single model
"""

import os
import sys
import argparse

# Add project path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.harness_eval import evaluate_model, parse_harness_results, install_lm_eval

def main():
    parser = argparse.ArgumentParser(description="Evaluate single model")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tasks", type=str, default="gsm8k", help="Evaluation tasks")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--use_multi_gpu", action="store_true", help="Use multi-GPU")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature")

    args = parser.parse_args()

    # Install lm-eval
    if not install_lm_eval():
        print("Cannot install lm-evaluation-harness")
        return
    
    # 评估模型
    print(f"开始评估模型: {args.model_path}")
    success, output = evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device,
        use_multi_gpu=args.use_multi_gpu,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature
    )
    
    if success:
        print("✓ 评估成功")
        # 解析结果
        metrics = parse_harness_results(args.output_dir, task_name=args.tasks)
        if metrics:
            print(f"准确率: {metrics['accuracy']:.4f}")
            print(f"样本数: {metrics['num_samples']}")
            print(f"标准误: {metrics['stderr']:.4f}")
    else:
        print(f"✗ 评估失败: {output}")

if __name__ == "__main__":
    main()
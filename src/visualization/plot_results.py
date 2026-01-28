#!/usr/bin/env python3
"""
绘制不同 w_repeat 实验的 GSM8K 准确率对比图
支持两种评估指标：flexible-extract 和 strict-match
"""

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_exp_results(exp_dir):
    """
    扫描 exp_dir/evaluations/step_* 下的 timestamped JSON，
    提取 exact_match 的两种指标，返回 DataFrame(step, flexible_acc, strict_acc).
    """
    records = []
    eval_dir = os.path.join(exp_dir, "evaluations")
    if not os.path.exists(eval_dir):
        return pd.DataFrame()
    
    for step_dir in sorted(glob.glob(os.path.join(eval_dir, "step_*"))):
        step_num = int(os.path.basename(step_dir).split("_")[1])
        
        # 查找编码子目录中的结果文件
        json_pattern = os.path.join(step_dir, "__*", "results_*.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            # 如果没有编码子目录，尝试直接查找
            direct_file = os.path.join(step_dir, "results.json")
            if os.path.exists(direct_file):
                json_files = [direct_file]
        
        if json_files:
            try:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                gsm8k_results = data.get("results", {}).get("gsm8k", {})
                
                flexible_acc = gsm8k_results.get("exact_match,flexible-extract", None)
                strict_acc = gsm8k_results.get("exact_match,strict-match", None)
                
                records.append({
                    "step": step_num,
                    "flexible_acc": flexible_acc,
                    "strict_acc": strict_acc
                })
                
            except Exception as e:
                print(f"Error reading {json_files[0]}: {e}")
                continue
    
    return pd.DataFrame(records)

def extract_w_repeat(exp_name):
    """从实验名称中提取 w_repeat 值"""
    try:
        if "w_repeat_" in exp_name:
            # 提取 w_repeat 值字符串，该字符串可能包含下划线表示小数，后面跟日期和时间两段
            suffix = exp_name.split("w_repeat_")[1]
            # 使用 rsplit 去掉末尾的日期和时间，只保留前面的值部分
            value_str = suffix.rsplit("_", 2)[0]
            # 将下划线转换为小数点 (例如 0_1 -> 0.1)
            return float(value_str.replace("_", "."))
    except:
        pass
    return None

def collect_all_experiments(experiments_root):
    """
    收集指定目录下所有实验的数据
    
    Args:
        experiments_root: 实验根目录
        
    Returns:
        一个字典，键为 w_repeat, 值为包含实验名称和数据的字典
    """
    all_experiments = {}
    
    for exp_dir in sorted(glob.glob(os.path.join(experiments_root, "exp_w_repeat_*"))):
        if not os.path.isdir(exp_dir):
            continue
            
        exp_name = os.path.basename(exp_dir)
        w_repeat = extract_w_repeat(exp_name)
        
        if w_repeat is None:
            print(f"跳过实验 {exp_name}：无法解析 w_repeat 值")
            continue
        
        df = load_exp_results(exp_dir)
        if df.empty:
            print(f"跳过实验 {exp_name}：未找到评估结果")
            continue
        
        # 过滤掉无效数据
        df = df.dropna().sort_values("step")
        if df.empty:
            continue
            
        all_experiments[w_repeat] = {
            'name': exp_name,
            'data': df
        }
        
        print(f"加载实验 {exp_name}: w_repeat={w_repeat}, {len(df)} 个数据点")
    
    return all_experiments

def plot_accuracy_comparison(all_experiments, save_dir="./", max_step=None):
    """
    绘制准确率对比图
    
    Args:
        all_experiments: 包含所有实验数据的字典
        save_dir: 图片保存目录
        max_step: 图表中 x 轴 (Training Step) 的最大值
    """
    if not all_experiments:
        print("未找到任何有效的实验数据用于绘图")
        return
    
    # 创建颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_experiments)))
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制 flexible-extract 结果
    ax1.set_title("GSM8K Accuracy: Flexible Extract", fontsize=14, fontweight='bold')
    for i, (w_repeat, exp_info) in enumerate(sorted(all_experiments.items())):
        df = exp_info['data']
        valid_data = df.dropna(subset=['flexible_acc'])
        if not valid_data.empty:
            ax1.plot(valid_data["step"], valid_data["flexible_acc"], 
                    marker="o", linewidth=2, markersize=6, 
                    color=colors[i], label=f"w_repeat={w_repeat}")
    
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Exact Match Accuracy (flexible-extract)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 0.9)
    if max_step is not None:
        ax1.set_xlim(left=0, right=max_step)
    
    # 绘制 strict-match 结果
    ax2.set_title("GSM8K Accuracy: Strict Match", fontsize=14, fontweight='bold')
    for i, (w_repeat, exp_info) in enumerate(sorted(all_experiments.items())):
        df = exp_info['data']
        valid_data = df.dropna(subset=['strict_acc'])
        if not valid_data.empty:
            ax2.plot(valid_data["step"], valid_data["strict_acc"], 
                    marker="s", linewidth=2, markersize=6, 
                    color=colors[i], label=f"w_repeat={w_repeat}")
    
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Exact Match Accuracy (strict-match)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 0.9)
    if max_step is not None:
        ax2.set_xlim(left=0, right=max_step)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(save_dir, "gsm8k_accuracy_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_file}")
    
    # 显示图片
    plt.show()

def plot_single_metric_comparison(all_experiments, metric="flexible", save_dir="./", max_step=None):
    """
    绘制单一指标的对比图
    
    Args:
        all_experiments: 包含所有实验数据的字典
        metric: 'flexible' 或 'strict'
        save_dir: 图片保存目录
        max_step: 图表中 x 轴 (Training Step) 的最大值
    """
    if not all_experiments:
        print("未找到任何有效的实验数据用于绘图")
        return
    
    # 选择指标
    if metric == "flexible":
        acc_column = "flexible_acc"
        title = "GSM8K Accuracy: Flexible Extract"
        ylabel = "Exact Match Accuracy (flexible-extract)"
        marker = "o"
    else:
        acc_column = "strict_acc"
        title = "GSM8K Accuracy: Strict Match"
        ylabel = "Exact Match Accuracy (strict-match)"
        marker = "s"
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_experiments)))
    
    for i, (w_repeat, exp_info) in enumerate(sorted(all_experiments.items())):
        df = exp_info['data']
        valid_data = df.dropna(subset=[acc_column])
        if not valid_data.empty:
            plt.plot(valid_data["step"], valid_data[acc_column], 
                    marker=marker, linewidth=2, markersize=6, 
                    color=colors[i], label=f"w_repeat={w_repeat}")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.7, 0.9)
    if max_step is not None:
        plt.xlim(left=0, right=max_step)
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(save_dir, f"gsm8k_accuracy_{metric}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_file}")
    
    plt.show()

def print_summary_table(all_experiments):
    """打印实验结果汇总表"""
    if not all_experiments:
        return

    print("\n" + "="*80)
    print("实验结果汇总表")
    print("="*80)
    print(f"{'w_repeat':<10} {'最佳Step':<10} {'Flexible-Extract':<18} {'Strict-Match':<15} {'实验名称':<30}")
    print("-"*80)
    
    for w_repeat in sorted(all_experiments.keys()):
        exp_info = all_experiments[w_repeat]
        df = exp_info['data']
        
        # 找到最佳结果
        if not df['flexible_acc'].isna().all():
            best_flexible_row = df.loc[df['flexible_acc'].idxmax()]
            best_flexible = best_flexible_row['flexible_acc']
            best_flexible_step = best_flexible_row['step']
        else:
            best_flexible = "N/A"
            best_flexible_step = "N/A"
        
        if not df['strict_acc'].isna().all():
            best_strict_row = df.loc[df['strict_acc'].idxmax()]
            best_strict = best_strict_row['strict_acc']
            best_strict_step = best_strict_row['step']
        else:
            best_strict = "N/A"
            best_strict_step = "N/A"
        
        # 使用较高的准确率对应的step
        if isinstance(best_flexible, float) and isinstance(best_strict, float):
            if best_flexible >= best_strict:
                best_step = best_flexible_step
            else:
                best_step = best_strict_step
        elif isinstance(best_flexible, float):
            best_step = best_flexible_step
        elif isinstance(best_strict, float):
            best_step = best_strict_step
        else:
            best_step = "N/A"
        
        flexible_str = f"{best_flexible:.4f}" if isinstance(best_flexible, float) else str(best_flexible)
        strict_str = f"{best_strict:.4f}" if isinstance(best_strict, float) else str(best_strict)
        
        print(f"{w_repeat:<10} {best_step:<10} {flexible_str:<18} {strict_str:<15} {exp_info['name']:<30}")

def export_all_results_to_md(all_experiments, output_md):
    """导出每个实验在所有 step 上的 flexible 和 strict 准确率到 Markdown 文件"""
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('# 实验数据汇总（所有 Step）\n\n')

        # 按实验名称排序以获得一致的输出
        sorted_exps = sorted(all_experiments.values(), key=lambda x: x['name'])

        for exp_info in sorted_exps:
            exp_name = exp_info['name']
            df = exp_info['data']
            if df.empty:
                continue
            
            f.write(f'## {exp_name}\n\n')
            f.write('| Step | Flexible-Extract | Strict-Match |\n')
            f.write('| --- | --- | --- |\n')
            for _, row in df.iterrows():
                f.write(f"| {int(row['step'])} | {row['flexible_acc']:.4f} | {row['strict_acc']:.4f} |\n")
            f.write('\n')

def main(experiments_root, save_dir, max_step=None):
    """主函数：协调数据收集、分析和可视化"""
    print("开始分析实验结果...")

    # 1. 统一收集数据
    all_experiments = collect_all_experiments(experiments_root)
    
    if not all_experiments:
        print(f"在目录 '{experiments_root}' 下未找到任何有效的实验数据，程序退出。")
        return

    # 2. 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 打印汇总表
    print_summary_table(all_experiments)
    
    # 4. 导出 Markdown
    md_file = os.path.join(save_dir, 'summary_all_data.md')
    export_all_results_to_md(all_experiments, md_file)
    print(f"\n所有实验数据已导出到: {md_file}")
    
    # 5. 绘制对比图
    print("\n绘制对比图...")
    plot_accuracy_comparison(all_experiments, save_dir, max_step)
    
    print("\n绘制 flexible-extract 图表...")
    plot_single_metric_comparison(all_experiments, "flexible", save_dir, max_step)
    
    print("\n绘制 strict-match 图表...")
    plot_single_metric_comparison(all_experiments, "strict", save_dir, max_step)
    
    print("\n分析完成！")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="绘制不同 w_repeat 实验的 GSM8K 准确率对比图。\n"
                    "该脚本会扫描指定目录下的 'exp_w_repeat_*' 子目录，\n"
                    "提取评估结果，并生成汇总表、Markdown报告和对比图。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--experiments_root",
        type=str,
        default="/path/to/your/output_dir",
        help="Root directory where all experiments are saved",
    )
    parser.add_argument("--save-dir", 
                        default=".", 
                        help="图片和报告的保存目录 (默认: 当前运行目录)")
    parser.add_argument("--max-step",
                        type=int,
                        default=None,
                        help="图表中 x 轴 (Training Step) 的最大值")
    args = parser.parse_args()

    main(args.experiments_root, args.save_dir, args.max_step)

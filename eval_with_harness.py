#!/usr/bin/env python3
"""
使用 lm-evaluation-harness 评估 repeat 实验中的模型
GPU 只能是 1,2,4
export CUDA_VISIBLE_DEVICES="1,2,3,4"
python eval_with_harness.py --exp_dir ./experiments/exp_w_repeat_0.5_20250602_134020 --use_multi_gpu 


"""

# It is recommended to set HuggingFace cache directories via environment variables
# before running the script. For example:
# export HF_HOME="/path/to/your/.cache/huggingface"
#
# import os
#
# # Example: set HuggingFace caches to a user-local directory
# os.environ['HF_HOME'] = '/path/to/your/.cache/huggingface'
# os.environ['HF_HUB_CACHE'] = '/path/to/your/.cache/huggingface/hub'
# os.environ['HF_DATASETS_CACHE'] = '/path/to/your/.cache/huggingface/datasets'


import json
import glob
import argparse
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil

def install_lm_eval():
    """安装 lm-evaluation-harness"""
    try:
        import lm_eval
        print("✓ lm-evaluation-harness 已安装")
        return True
    except ImportError:
        print("安装 lm-evaluation-harness...")
        try:
            subprocess.run([
                "pip", "install", "lm-eval[vllm]"
            ], check=True)
            print("✓ lm-evaluation-harness 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 安装失败: {e}")
            return False

def evaluate_model(model_path, output_dir, tasks="gsm8k", batch_size="auto", device="cuda:0", use_multi_gpu=False, tensor_parallel_size=1, temperature=None):
    """
    使用 lm-evaluation-harness 评估单个模型
    
    Args:
        model_path: 模型路径
        output_dir: 结果输出目录  
        tasks: 评估任务，默认 gsm8k
        batch_size: 批次大小 或 'auto'（vLLM 推荐使用 'auto'）
        device: 设备 (主要用于单GPU hf, vLLM 通常不直接使用此参数)
        use_multi_gpu: 是否强制启用多GPU支持
        tensor_parallel_size: vLLM 的张量并行数
        temperature: vLLM 的生成温度
    """
    print(f"开始评估模型: {model_path}")
    print(f"任务: {tasks}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建模型参数
    model_args_list = [f"pretrained={model_path}", "trust_remote_code=True"]
    
    # 新增：如果指定了 temperature，则添加到模型参数中
    if temperature is not None:
        model_args_list.append(f"temperature={temperature}")
        print(f"设置 vLLM temperature={temperature}")

    import torch
    num_gpus_available = torch.cuda.device_count()

    if use_multi_gpu and num_gpus_available > 1:
        # 对于 vLLM，我们使用 tensor_parallel_size
        # tensor_parallel_size 通常由调用者根据 CUDA_VISIBLE_DEVICES 决定
        # 确保 tensor_parallel_size 不超过实际可见的GPU数量
        actual_tensor_parallel_size = min(tensor_parallel_size, num_gpus_available)
        if actual_tensor_parallel_size > 1:
            model_args_list.append(f"tensor_parallel_size={actual_tensor_parallel_size}")
            print(f"检测到 {num_gpus_available} 个可用GPU，将为 vLLM 设置 tensor_parallel_size={actual_tensor_parallel_size}")
        else:
            print(f"检测到 {num_gpus_available} 个可用GPU，但 tensor_parallel_size=1，vLLM 将在单个GPU上运行。")
            # 如果只有一个GPU实际可用或者tensor_parallel_size设为1，vLLM默认在cuda:0上运行 (相对于可见设备)
            # 不需要显式传递 device 给 vLLM 命令，它会自行处理
    elif use_multi_gpu and num_gpus_available == 1:
        print(f"请求多GPU，但只检测到1个可用GPU。vLLM 将在单个GPU上运行。")
    else: # 单GPU模式
        print(f"单GPU模式或未请求多GPU。vLLM 将在默认GPU上运行。")
        # vLLM 默认使用 cuda:0 (相对于可见设备)，通常不需要显式传递 --device

    model_args = ",".join(model_args_list)
    
    # 构建评估命令
    cmd = [
        "lm_eval",
        "--model", "vllm", # 切换到 vLLM
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", batch_size, # vLLM 推荐 'auto'
        "--output_path", output_dir,
        "--log_samples"
    ]
    
    # 对于vLLM, --device 参数通常不需要，它会基于 CUDA_VISIBLE_DEVICES 和 tensor_parallel_size 工作
    # 如果 use_multi_gpu 为 True 且 tensor_parallel_size > 1，则不应传递 --device
    # 如果是单 GPU (或者 tensor_parallel_size=1), vLLM 默认使用第一个可见 GPU。
    # 我们移除显式的 --device 传递，让 vLLM 自行管理。
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行评估
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        print(f"lm_eval stdout:\n{result.stdout}")
        print(f"lm_eval stderr:\n{result.stderr}")

        if result.returncode == 0:
            print("✓ 评估完成 (lm_eval returncode 0)")
            # 即使返回0，也检查文件是否存在，因为有时 lm_eval 可能成功运行但未生成预期的文件
            result_file_path = os.path.join(output_dir, "results.json") # 通常 lm_eval 将结果存在 output_dir/results.json
            # 有些任务可能会生成更详细的子目录，例如 output_dir/gsm8k/results.json 或类似结构
            # 为了更通用，我们先检查根输出目录，如果找不到，可以考虑更复杂的查找
            if os.path.exists(result_file_path):
                print(f"✓ 结果文件 {result_file_path} 已找到。")
                return True, result.stdout
            else:
                # 尝试查找 output_dir 下子目录中的任何 results*.json 文件
                for item in os.listdir(output_dir):
                    subdir_path = os.path.join(output_dir, item)
                    if os.path.isdir(subdir_path):
                        for fname in os.listdir(subdir_path):
                            if fname.startswith("results") and fname.endswith(".json"):
                                potential_result_file = os.path.join(subdir_path, fname)
                                print(f"✓ 结果文件在子目录中找到: {potential_result_file}")
                                shutil.copy(potential_result_file, result_file_path)
                                print(f"✓ 已将结果文件复制到主目录: {result_file_path}")
                                return True, result.stdout

                print(f"✗ 评估命令成功，但结果文件 {result_file_path} 未在指定路径找到。")
                return False, "Evaluation command succeeded, but results.json not found at the primary path."

        else:
            print(f"✗ 评估失败 (lm_eval returncode {result.returncode})")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("✗ 评估超时")
        return False, "Evaluation timeout"
    except Exception as e:
        print(f"✗ 评估出错: {e}")
        return False, str(e)

def parse_harness_results(result_file_base_dir, task_name="gsm8k"):
    """解析 lm-evaluation-harness 的结果文件"""
    import glob
    
    # 路径1: 直接在基目录下找 results.json
    result_file_path1 = os.path.join(result_file_base_dir, "results.json")
    # 路径2: 在 基目录/任务名/results.json 找 (适配 lm-eval >= v0.4.1 的行为)
    result_file_path2 = os.path.join(result_file_base_dir, task_name, "results.json")
    # 路径3: 查找带时间戳的 results_*.json 文件
    result_file_path3_pattern = os.path.join(result_file_base_dir, "results_*.json")
    # 路径4: 查找编码路径子目录中的带时间戳文件 (新版本lm-eval的行为)
    encoded_subdir_pattern = os.path.join(result_file_base_dir, "__*", "results_*.json")

    actual_result_file = None
    
    # 按优先级顺序查找
    if os.path.exists(result_file_path1):
        actual_result_file = result_file_path1
        print(f"Found results file at: {actual_result_file}")
    elif os.path.exists(result_file_path2):
        actual_result_file = result_file_path2
        print(f"Found results file in task subdirectory: {actual_result_file}")
    else:
        # 查找带时间戳的文件
        timestamped_files = glob.glob(result_file_path3_pattern)
        if timestamped_files:
            actual_result_file = timestamped_files[0]  # 取第一个匹配的文件
            print(f"Found timestamped results file: {actual_result_file}")
        else:
            # 查找编码路径子目录中的文件
            encoded_files = glob.glob(encoded_subdir_pattern)
            if encoded_files:
                actual_result_file = encoded_files[0]  # 取第一个匹配的文件
                print(f"Found results file in encoded subdirectory: {actual_result_file}")
            else:
                print(f"✗ 结果文件未在以下期望路径找到: ")
                print(f"  - {result_file_path1}")
                print(f"  - {result_file_path2}")
                print(f"  - {result_file_path3_pattern}")
                print(f"  - {encoded_subdir_pattern}")
                return None

    try:
        with open(actual_result_file, 'r') as f:
            results_data = json.load(f)
        
        # 提取关键指标 (假设任务名是gsm8k，如果不是，需要更通用的解析)
        # lm-eval 的结果JSON结构可能包含一个 'results' 字典，键是任务名
        if 'results' in results_data and task_name in results_data['results']:
            task_results = results_data['results'][task_name]
            return {
                'accuracy': task_results.get('acc_norm', task_results.get('acc', 0.0)), # 尝试 acc_norm, 回退到 acc
                'exact_match': task_results.get('exact_match', 0.0),
                'num_samples': results_data.get('n_shot_samples', {}).get(task_name, results_data.get('config', {}).get('num_fewshot', 0)), # 尝试从不同地方获取样本数
                'stderr': task_results.get('acc_norm_stderr', task_results.get('acc_stderr', 0.0))
            }
        # 有些旧版本或者简单任务可能直接在顶层返回指标
        elif 'acc_norm' in results_data: # 直接就是gsm8k的结果内容
             return {
                'accuracy': results_data.get('acc_norm', 0.0),
                'exact_match': results_data.get('exact_match', 0.0),
                'num_samples': results_data.get('num_samples', 0),
                'stderr': results_data.get('acc_norm_stderr', 0.0)
            }
        else:
            print(f"未在结果JSON中找到 {task_name} 的结果部分。")
            print(f"可用的任务: {list(results_data.get('results', {}).keys())}")
            return None
            
    except Exception as e:
        print(f"解析结果文件 {actual_result_file} 出错: {e}")
        return None

def evaluate_experiment_models(exp_dir, eval_output_dir=None, tasks="gsm8k",
                                batch_size="auto", device="cuda:0", use_multi_gpu=False, tensor_parallel_size=1, temperature=None):
    """
    评估实验目录中的所有模型
    
    Args:
        exp_dir: 实验目录路径
        eval_output_dir: 评估结果输出目录，默认为 exp_dir/evaluations
        batch_size: 批次大小 或 'auto'（vLLM 推荐使用 'auto'）
        device: 设备
        use_multi_gpu: 是否强制启用多GPU支持
        tensor_parallel_size: vLLM 的张量并行数
        temperature: vLLM 的生成温度
    """
    if eval_output_dir is None:
        eval_output_dir = os.path.join(exp_dir, "evaluations")
    
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # 查找模型目录 (step_xxx 格式)
    model_dirs = glob.glob(os.path.join(exp_dir, "step_*"))
    model_dirs = [d for d in model_dirs if os.path.isdir(d) and not d.endswith('_info.txt')]
    model_dirs = sorted(model_dirs, key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    if not model_dirs:
        print(f"在 {exp_dir} 中未找到模型目录")
        return []
    
    print(f"找到 {len(model_dirs)} 个模型需要评估")
    
    evaluation_results = []
    
    for model_dir in model_dirs:
        step_num = os.path.basename(model_dir).split('_')[1]
        print(f"\n{'='*50}")
        print(f"评估 Step {step_num} 模型")
        print(f"{'='*50}")
        
        # 为每个step创建独立的评估输出目录
        step_eval_dir = os.path.join(eval_output_dir, f"step_{step_num}")
        
        # 运行评估
        success, output = evaluate_model(
            model_path=model_dir,
            output_dir=step_eval_dir,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            use_multi_gpu=use_multi_gpu,
            tensor_parallel_size=tensor_parallel_size,
            temperature=temperature
        )
        
        result_entry = {
            'step': int(step_num),
            'model_path': model_dir,
            'eval_output_dir': step_eval_dir,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            # 解析结果
            result_file = os.path.join(step_eval_dir, "results.json")
            if os.path.exists(result_file) or os.path.exists(os.path.join(step_eval_dir, tasks, "results.json")):
                metrics = parse_harness_results(step_eval_dir, task_name=tasks)
                if metrics:
                    result_entry.update(metrics)
                    print(f"✓ GSM-8K 准确率: {metrics['accuracy']:.4f}")
                else:
                    print("✗ 结果解析失败")
            else:
                print("✗ 结果文件未找到")
        else:
            print(f"✗ 评估失败: {output}")
            result_entry['error'] = output
        
        evaluation_results.append(result_entry)
    
    return evaluation_results

def create_evaluation_report(evaluation_results, exp_dir):
    """创建评估报告"""
    if not evaluation_results:
        print("没有评估结果可生成报告")
        return
    
    # 创建报告内容
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""GSM-8K 评估报告
================
实验目录: {exp_dir}
评估时间: {timestamp}
使用框架: lm-evaluation-harness
评估任务: GSM-8K (数学推理)

"""
    
    # 添加结果汇总表
    report_content += "评估结果汇总:\n"
    report_content += "-" * 80 + "\n"
    report_content += f"{'Step':<8} {'准确率':<12} {'样本数':<8} {'标准误':<12} {'状态':<8}\n"
    report_content += "-" * 80 + "\n"
    
    successful_results = []
    for result in evaluation_results:
        if result['success'] and 'accuracy' in result:
            successful_results.append(result)
            report_content += f"{result['step']:<8} {result['accuracy']:<12.4f} {result.get('num_samples', 'N/A'):<8} {result.get('stderr', 'N/A'):<12.4f} {'成功':<8}\n"
        else:
            report_content += f"{result['step']:<8} {'N/A':<12} {'N/A':<8} {'N/A':<12} {'失败':<8}\n"
    
    # 添加性能分析
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results]
        steps = [r['step'] for r in successful_results]
        
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        worst_result = min(successful_results, key=lambda x: x['accuracy'])
        
        report_content += f"\n性能分析:\n"
        report_content += f"最佳性能: Step {best_result['step']} - 准确率 {best_result['accuracy']:.4f}\n"
        report_content += f"最差性能: Step {worst_result['step']} - 准确率 {worst_result['accuracy']:.4f}\n"
        report_content += f"平均准确率: {sum(accuracies)/len(accuracies):.4f}\n"
        
        # 趋势分析
        if len(successful_results) > 1:
            if best_result['step'] > worst_result['step']:
                report_content += f"趋势: 训练过程中模型性能有所提升\n"
            else:
                report_content += f"趋势: 训练过程中模型性能有所下降\n"
    
    report_content += f"\n详细信息:\n"
    report_content += f"- 评估框架: EleutherAI lm-evaluation-harness\n"
    report_content += f"- 数据集: GSM-8K (Grade School Math 8K)\n"
    report_content += f"- 评估指标: 准确率 (accuracy)\n"
    report_content += f"- 评估模式: 少样本学习\n"
    
    # 保存报告
    report_path = os.path.join(exp_dir, "gsm8k_evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n评估报告已保存: {report_path}")
    
    # 保存 CSV 格式结果
    if successful_results:
        df = pd.DataFrame(successful_results)
        csv_path = os.path.join(exp_dir, "gsm8k_evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV 结果已保存: {csv_path}")

def compare_experiments(exp_dirs, output_dir="./comparison_results"):
    """比较多个实验的 GSM-8K 评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_experiment_results = {}
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        
        # 查找评估结果
        results_file = os.path.join(exp_dir, "gsm8k_evaluation_results.csv")
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            all_experiment_results[exp_name] = df
            print(f"✓ 加载实验结果: {exp_name}")
        else:
            print(f"✗ 未找到评估结果: {exp_dir}")
    
    if not all_experiment_results:
        print("没有找到可比较的实验结果")
        return
    
    # 创建比较报告
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    comparison_content = f"""GSM-8K 实验比较报告
====================
比较时间: {timestamp}
实验数量: {len(all_experiment_results)}

"""
    
    # 提取每个实验的最佳结果
    best_results = []
    for exp_name, results_df in all_experiment_results.items():
        if not results_df.empty:
            best_row = results_df.loc[results_df['accuracy'].idxmax()]
            w_repeat = exp_name.split('_')[2] if 'w_repeat' in exp_name else 'Unknown'
            
            best_results.append({
                'experiment': exp_name,
                'w_repeat': w_repeat,
                'best_step': best_row['step'],
                'best_accuracy': best_row['accuracy'],
                'num_samples': best_row.get('num_samples', 'N/A')
            })
    
    # 按准确率排序
    best_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    comparison_content += "各实验最佳表现:\n"
    comparison_content += "-" * 80 + "\n"
    comparison_content += f"{'实验名称':<30} {'w_repeat':<10} {'最佳Step':<10} {'准确率':<12}\n"
    comparison_content += "-" * 80 + "\n"
    
    for result in best_results:
        comparison_content += f"{result['experiment']:<30} {result['w_repeat']:<10} {result['best_step']:<10} {result['best_accuracy']:<12.4f}\n"
    
    # w_repeat 参数影响分析
    if len(best_results) > 1:
        comparison_content += f"\nw_repeat 参数影响分析:\n"
        comparison_content += f"最佳配置: {best_results[0]['experiment']} (w_repeat={best_results[0]['w_repeat']}, 准确率={best_results[0]['best_accuracy']:.4f})\n"
        comparison_content += f"最差配置: {best_results[-1]['experiment']} (w_repeat={best_results[-1]['w_repeat']}, 准确率={best_results[-1]['best_accuracy']:.4f})\n"
        
        # 尝试分析趋势
        w_repeat_results = [(float(r['w_repeat']), r['best_accuracy']) for r in best_results if r['w_repeat'] != 'Unknown']
        if len(w_repeat_results) > 1:
            w_repeat_results.sort(key=lambda x: x[0])
            comparison_content += f"\nw_repeat 趋势分析:\n"
            for w_repeat, acc in w_repeat_results:
                comparison_content += f"  w_repeat={w_repeat}: 准确率={acc:.4f}\n"
    
    # 保存比较报告
    comparison_path = os.path.join(output_dir, "gsm8k_experiments_comparison.txt")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(comparison_content)
    
    print(f"比较报告已保存: {comparison_path}")
    
    # 保存详细数据
    comparison_df = pd.DataFrame(best_results)
    csv_path = os.path.join(output_dir, "gsm8k_experiments_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"比较数据已保存: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="使用 lm-evaluation-harness 评估 repeat 实验")
    parser.add_argument("--exp_dir", type=str, help="单个实验目录路径")
    parser.add_argument("--exp_dirs", nargs='+', help="多个实验目录路径（用于比较）")
    parser.add_argument("--experiments_root", type=str, default="./experiments", 
                       help="实验根目录，自动发现所有实验")
    parser.add_argument("--comparison_output_dir", type=str, default="./comparison_results",
                       help="比较报告输出目录 (默认为 ./comparison_results)")
    parser.add_argument("--tasks", type=str, default="gsm8k",
                       help="评估任务（默认: gsm8k）")
    parser.add_argument("--batch_size", type=str, default="auto",
                       help="评估批次大小（默认: auto，使用 vLLM 时推荐 'auto'）")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="评估设备（默认: cuda:0，vLLM通常不直接使用此参数，而是依赖CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="强制启用多GPU支持（即使只有一个GPU可见）")
    parser.add_argument("--gpu_ids", type=str, default=None, # 改为None，优先使用环境变量
                       help="指定使用的GPU ID，逗号分隔（例如 '0,1,2,3'）。如果设置，将覆盖CUDA_VISIBLE_DEVICES")
    parser.add_argument("--compare_only", action="store_true",
                       help="仅进行比较，不运行新的评估")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM 的张量并行大小 (用于多GPU评估)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="设置 vLLM 的生成温度 (例如 0.7)。默认由harness控制。")
    
    args = parser.parse_args()
    
    # 设置GPU环境变量
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"通过 --gpu_ids 设置 CUDA_VISIBLE_DEVICES={args.gpu_ids}")
    
    # 获取实际可用的GPU数量 (在设置CUDA_VISIBLE_DEVICES之后)
    import torch
    num_visible_gpus = torch.cuda.device_count()
    print(f"当前可见的GPU数量: {num_visible_gpus}")

    actual_tensor_parallel_size = 1
    if args.use_multi_gpu:
        if args.tensor_parallel_size > 1:
            if num_visible_gpus >= args.tensor_parallel_size:
                actual_tensor_parallel_size = args.tensor_parallel_size
                print(f"多GPU模式：将使用 tensor_parallel_size={actual_tensor_parallel_size}")
            else:
                actual_tensor_parallel_size = num_visible_gpus
                print(f"警告: 请求的 tensor_parallel_size ({args.tensor_parallel_size}) 大于可见GPU数量 ({num_visible_gpus}). "
                      f"将使用 tensor_parallel_size={actual_tensor_parallel_size}")
        elif num_visible_gpus > 1: # 如果用户指定了 use_multi_gpu 但没指定 tensor_parallel_size > 1，则默认使用所有可见GPU
            actual_tensor_parallel_size = num_visible_gpus
            print(f"多GPU模式：未指定有效的 tensor_parallel_size，将使用所有可见GPU，tensor_parallel_size={actual_tensor_parallel_size}")
        else: # num_visible_gpus <= 1
            print(f"多GPU模式：但只有一个或没有可见GPU。将在单GPU模式下运行。")
    else:
        print("单GPU模式。")

    # 确保安装了 lm-evaluation-harness
    if not install_lm_eval():
        print("无法安装 lm-evaluation-harness，退出")
        return
    
    if args.compare_only:
        # 仅比较模式
        # 使用 --comparison_output_dir 作为比较结果的输出目录
        comparison_out = args.comparison_output_dir
        if args.exp_dirs:
            compare_experiments(args.exp_dirs, comparison_out)
        elif args.experiments_root and os.path.exists(args.experiments_root):
            exp_dirs_to_compare = glob.glob(os.path.join(args.experiments_root, "exp_w_repeat_*"))
            if exp_dirs_to_compare:
                compare_experiments(exp_dirs_to_compare, comparison_out)
            else:
                print(f"在 {args.experiments_root} 中未找到实验目录")
        else:
            print("比较模式需要指定实验目录 (--exp_dirs) 或有效的实验根目录 (--experiments_root)")
        return
    
    # 评估模式
    if args.exp_dir:
        # 评估单个实验
        print(f"评估单个实验: {args.exp_dir}")
        # 评估结果将存储在 exp_dir/evaluations/
        eval_output_directory_for_exp = os.path.join(args.exp_dir, "evaluations")
        results = evaluate_experiment_models(
            args.exp_dir,
            # eval_output_dir is now handled inside evaluate_experiment_models based on exp_dir
            tasks=args.tasks,
            batch_size=args.batch_size,
            device=args.device,
            use_multi_gpu=args.use_multi_gpu,
            tensor_parallel_size=actual_tensor_parallel_size,
            temperature=args.temperature
        )
        create_evaluation_report(results, args.exp_dir) # 报告也基于 exp_dir
        
    elif args.exp_dirs:
        # 评估多个实验
        all_evaluated_exp_dirs = []
        for exp_dir_item in args.exp_dirs:
            print(f"评估实验: {exp_dir_item}")
            eval_output_directory_for_exp = os.path.join(exp_dir_item, "evaluations")
            results = evaluate_experiment_models(
                exp_dir_item,
                tasks=args.tasks,
                batch_size=args.batch_size,
                device=args.device,
                use_multi_gpu=args.use_multi_gpu,
                tensor_parallel_size=actual_tensor_parallel_size,
                temperature=args.temperature
            )
            create_evaluation_report(results, exp_dir_item)
            all_evaluated_exp_dirs.append(exp_dir_item)
        
        # 生成比较报告，使用 --comparison_output_dir
        if all_evaluated_exp_dirs:
            compare_experiments(all_evaluated_exp_dirs, args.comparison_output_dir)
        
    else:
        # 自动发现并评估所有实验
        if os.path.exists(args.experiments_root):
            discovered_exp_dirs = glob.glob(os.path.join(args.experiments_root, "exp_w_repeat_*"))
            discovered_exp_dirs = [d for d in discovered_exp_dirs if os.path.isdir(d)]
            all_evaluated_exp_dirs = []

            if discovered_exp_dirs:
                print(f"发现 {len(discovered_exp_dirs)} 个实验，开始评估...")
                
                for exp_dir_item in discovered_exp_dirs:
                    print(f"\n评估实验: {os.path.basename(exp_dir_item)}")
                    eval_output_directory_for_exp = os.path.join(exp_dir_item, "evaluations")
                    results = evaluate_experiment_models(
                        exp_dir_item,
                        tasks=args.tasks,
                        batch_size=args.batch_size,
                        device=args.device,
                        use_multi_gpu=args.use_multi_gpu,
                        tensor_parallel_size=actual_tensor_parallel_size,
                        temperature=args.temperature
                    )
                    create_evaluation_report(results, exp_dir_item)
                    all_evaluated_exp_dirs.append(exp_dir_item)
                
                # 生成比较报告，使用 --comparison_output_dir
                if all_evaluated_exp_dirs:
                    compare_experiments(all_evaluated_exp_dirs, args.comparison_output_dir)
                
            else:
                print(f"在 {args.experiments_root} 中未找到实验目录")
        else:
            print(f"实验根目录不存在: {args.experiments_root}")

if __name__ == "__main__":
    main()


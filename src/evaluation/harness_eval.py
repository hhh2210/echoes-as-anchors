#!/usr/bin/env python3
"""
使用 lm-evaluation-harness 评估 repeat 实验中的模型
"""

import os
import json
import glob
import argparse
import subprocess
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import shutil

# Configure HuggingFace cache directories via environment variables if provided
HF_HOME = os.getenv('HF_HOME')
HF_HUB_CACHE = os.getenv('HF_HUB_CACHE')
HF_DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE')
if HF_HOME:
    os.environ['HF_HOME'] = HF_HOME
if HF_HUB_CACHE:
    os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
if HF_DATASETS_CACHE:
    os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE


def install_lm_eval():
    """Install lm-evaluation-harness"""
    try:
        import lm_eval
        print("✓ lm-evaluation-harness is already installed")
        return True
    except ImportError:
        print("Installing lm-evaluation-harness...")
        try:
            subprocess.run([
                "pip", "install", "lm-eval[vllm]"
            ], check=True)
            print("✓ lm-evaluation-harness installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Installation failed: {e}")
            return False


def evaluate_model(
    model_path,
    output_dir,
    tasks="gsm8k",
    batch_size="auto",
    device="cuda:0",
    use_multi_gpu=False,
    tensor_parallel_size=1,
    temperature=None,
    enable_thinking=None,
    think_end_token=None,
):
    """
    使用 lm-evaluation-harness 评估单个模型
    """
    print(f"开始评估模型: {model_path}")
    print(f"任务: {tasks}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建模型参数
    model_args_list = [f"pretrained={model_path}", "trust_remote_code=True"]
    
    # 如果指定了 temperature，则添加到模型参数中
    if temperature is not None:
        model_args_list.append(f"temperature={temperature}")
        print(f"设置 vLLM temperature={temperature}")

    import torch
    num_gpus_available = torch.cuda.device_count()

    if use_multi_gpu and num_gpus_available > 1:
        actual_tensor_parallel_size = min(tensor_parallel_size, num_gpus_available)
        if actual_tensor_parallel_size > 1:
            model_args_list.append(f"tensor_parallel_size={actual_tensor_parallel_size}")
            print(f"检测到 {num_gpus_available} 个可用GPU，将为 vLLM 设置 tensor_parallel_size={actual_tensor_parallel_size}")

    # Optional: pass enable_thinking flag through to model backend (vLLM/HF)
    if enable_thinking is True:
        model_args_list.append("enable_thinking=True")
    # Optional: pass think_end_token to trim CoT for scoring while preserving raw generations
    if think_end_token:
        model_args_list.append(f"think_end_token={think_end_token}")

    model_args = ",".join(model_args_list)
    
    # 构建评估命令
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--output_path", output_dir,
        "--log_samples"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        print(f"lm_eval stdout:\n{result.stdout}")
        print(f"lm_eval stderr:\n{result.stderr}")

        if result.returncode == 0:
            print("✓ 评估完成")
            result_file_path = os.path.join(output_dir, "results.json")
            if os.path.exists(result_file_path):
                print(f"✓ 结果文件 {result_file_path} 已找到")
                return True, result.stdout
            else:
                # 尝试查找子目录中的结果文件
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

                print(f"✗ 评估命令成功，但结果文件 {result_file_path} 未在指定路径找到")
                return False, "Evaluation command succeeded, but results.json not found"
        else:
            print(f"✗ 评估失败 (返回码 {result.returncode})")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("✗ 评估超时")
        return False, "Evaluation timeout"
    except Exception as e:
        print(f"✗ 评估出错: {e}")
        return False, str(e)


def parse_harness_results(result_file_base_dir, task_name="gsm8k"):
    """解析 lm-evaluation-harness 的结果文件"""
    # 路径1: 直接在基目录下找 results.json
    result_file_path1 = os.path.join(result_file_base_dir, "results.json")
    # 路径2: 在 基目录/任务名/results.json 找
    result_file_path2 = os.path.join(result_file_base_dir, task_name, "results.json")
    # 路径3: 查找带时间戳的 results_*.json 文件
    result_file_path3_pattern = os.path.join(result_file_base_dir, "results_*.json")
    # 路径4: 查找编码路径子目录中的带时间戳文件
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
            actual_result_file = timestamped_files[0]
            print(f"Found timestamped results file: {actual_result_file}")
        else:
            # 查找编码路径子目录中的文件
            encoded_files = glob.glob(encoded_subdir_pattern)
            if encoded_files:
                actual_result_file = encoded_files[0]
                print(f"Found results file in encoded subdirectory: {actual_result_file}")
            else:
                print(f"✗ 结果文件未在以下期望路径找到:")
                print(f"  - {result_file_path1}")
                print(f"  - {result_file_path2}")
                print(f"  - {result_file_path3_pattern}")
                print(f"  - {encoded_subdir_pattern}")
                return None

    try:
        with open(actual_result_file, 'r') as f:
            results_data = json.load(f)
        
        # 提取关键指标
        if 'results' in results_data and task_name in results_data['results']:
            task_results = results_data['results'][task_name]
            
            # 新版本 lm-eval 使用不同的字段名
            accuracy = (
                task_results.get('exact_match,flexible-extract') or
                task_results.get('exact_match,strict-match') or 
                task_results.get('acc_norm') or
                task_results.get('acc') or
                0.0
            )
            
            # 获取标准误差
            stderr = (
                task_results.get('exact_match_stderr,flexible-extract') or
                task_results.get('exact_match_stderr,strict-match') or
                task_results.get('acc_norm_stderr') or
                task_results.get('acc_stderr') or
                0.0
            )
            
            # 样本数应来自 n-samples.effective（或 original），而非 num_fewshot（那是fewshot数量）
            num_samples = 0
            try:
                ns = results_data.get('n-samples', {}).get(task_name, {})
                num_eff = ns.get('effective')
                num_ori = ns.get('original')
                if isinstance(num_eff, int) and num_eff > 0:
                    num_samples = num_eff
                elif isinstance(num_ori, int) and num_ori > 0:
                    num_samples = num_ori
            except Exception:
                pass
            
            return {
                'accuracy': accuracy,
                'exact_match': accuracy,
                'num_samples': num_samples,
                'stderr': stderr
            }
        elif 'acc_norm' in results_data:
             return {
                'accuracy': results_data.get('acc_norm', 0.0),
                'exact_match': results_data.get('exact_match', 0.0),
                'num_samples': results_data.get('num_samples', 0),
                'stderr': results_data.get('acc_norm_stderr', 0.0)
            }
        else:
            print(f"未在结果JSON中找到 {task_name} 的结果部分")
            print(f"可用的任务: {list(results_data.get('results', {}).keys())}")
            return None
            
    except Exception as e:
        print(f"解析结果文件 {actual_result_file} 出错: {e}")
        return None


def evaluate_experiment_models(exp_dir, eval_output_dir=None, tasks="gsm8k",
                                batch_size="auto", device="cuda:0", 
                                use_multi_gpu=False, tensor_parallel_size=1, 
                                temperature=None):
    """评估实验目录中的所有模型"""
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
        
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        worst_result = min(successful_results, key=lambda x: x['accuracy'])
        
        report_content += f"\n性能分析:\n"
        report_content += f"最佳性能: Step {best_result['step']} - 准确率 {best_result['accuracy']:.4f}\n"
        report_content += f"最差性能: Step {worst_result['step']} - 准确率 {worst_result['accuracy']:.4f}\n"
        report_content += f"平均准确率: {sum(accuracies)/len(accuracies):.4f}\n"
    
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


def main():
    parser = argparse.ArgumentParser(description="使用 lm-evaluation-harness 评估 repeat 实验")
    parser.add_argument("--exp_dir", type=str, help="单个实验目录路径")
    parser.add_argument("--exp_dirs", nargs='+', help="多个实验目录路径（用于比较）")
    parser.add_argument("--experiments_root", type=str, default="/path/to/experiments_root",
                       help="实验根目录，自动发现所有实验")
    parser.add_argument("--comparison_output_dir", type=str, default="/path/to/comparison_output",
                       help="比较报告输出目录")
    parser.add_argument("--tasks", type=str, default="gsm8k",
                       help="评估任务（默认: gsm8k）")
    parser.add_argument("--batch_size", type=str, default="auto",
                       help="评估批次大小（默认: auto）")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="评估设备（默认: cuda:0）")
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="强制启用多GPU支持")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="指定使用的GPU ID，逗号分隔（例如 '0,1,2,3'）")
    parser.add_argument("--compare_only", action="store_true",
                       help="仅进行比较，不运行新的评估")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM 的张量并行大小 (用于多GPU评估)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="设置 vLLM 的生成温度")
    
    args = parser.parse_args()
    
    # 设置GPU环境变量
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"通过 --gpu_ids 设置 CUDA_VISIBLE_DEVICES={args.gpu_ids}")
    
    # 获取实际可用的GPU数量
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
                print(f"警告: 请求的 tensor_parallel_size ({args.tensor_parallel_size}) 大于可见GPU数量 ({num_visible_gpus})")
        elif num_visible_gpus > 1:
            actual_tensor_parallel_size = num_visible_gpus
            print(f"多GPU模式：未指定有效的 tensor_parallel_size，将使用所有可见GPU")

    # 确保安装了 lm-evaluation-harness
    if not install_lm_eval():
        print("无法安装 lm-evaluation-harness，退出")
        return
    
    if args.exp_dir:
        # 评估单个实验
        print(f"评估单个实验: {args.exp_dir}")
        results = evaluate_experiment_models(
            args.exp_dir,
            tasks=args.tasks,
            batch_size=args.batch_size,
            device=args.device,
            use_multi_gpu=args.use_multi_gpu,
            tensor_parallel_size=actual_tensor_parallel_size,
            temperature=args.temperature
        )
        create_evaluation_report(results, args.exp_dir)


if __name__ == "__main__":
    main()

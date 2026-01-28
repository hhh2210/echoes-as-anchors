#!/usr/bin/env python3
"""
数据集下载工具

用于下载和预处理 GSM8K 数据集，供训练和评估使用。
"""

import os

# It is recommended to set HuggingFace cache directories via environment variables.
# For example:
#   export HF_HOME="/path/to/your/.cache/huggingface"
#
# You can also configure them programmatically (example, commented out by default):
#   os.environ['HF_HOME'] = '/path/to/your/.cache/huggingface'
#   os.environ['HF_HUB_CACHE'] = '/path/to/your/.cache/huggingface/hub'
#   os.environ['HF_DATASETS_CACHE'] = '/path/to/your/.cache/huggingface/datasets'

from datasets import load_dataset
from tqdm import tqdm

def main():
    """下载并处理 GSM8K 数据集"""
    # 可以在此处根据需要设置 HuggingFace 缓存目录，例如：
    # os.environ['HF_HOME'] = '/path/to/your/.cache/huggingface'
    # os.environ['HF_HUB_CACHE'] = '/path/to/your/.cache/huggingface/hub'
    # os.environ['HF_DATASETS_CACHE'] = '/path/to/your/.cache/huggingface/datasets'
    
    print("[数据下载] 正在从缓存加载 GSM8K 数据集（训练集）...")
    
    try:
        dataset = load_dataset("openai/gsm8k", name="main", split="train")
        print("[数据下载] 数据集加载成功")
        
        # 处理数据为 Q-A 对格式
        qas = [
            {"Q": question, "A": answer.split("####")[-1].strip()}
            for question, answer in zip(dataset["question"], dataset["answer"])
        ]
        
        print(f"[数据下载] 已处理 {len(qas)} 个问答对")
        print(f"[数据下载] 示例问题: {qas[0]['Q'][:100]}...")
        print(f"[数据下载] 示例答案: {qas[0]['A']}")
        
        return qas
        
    except Exception as e:
        print(f"[数据下载] 加载失败: {e}")
        return None


if __name__ == "__main__":
    main()

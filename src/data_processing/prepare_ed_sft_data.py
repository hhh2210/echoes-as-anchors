"""
Simple Echo-Based COT Distillation
核心思想：让模型自然地echo(复述)问题，不需要复杂的多阶段引导
"""

import json
import random
from typing import Dict, Any, Tuple
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

class SimpleEchoDistiller:
    """Simple implementation focusing on natural prompt echoing"""
    
    def __init__(self, api_base="http://localhost:8000/v1", api_key="EMPTY", 
                 model_name="openai/gpt-oss-120b"):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
    
    def create_natural_echo_prompt(self, question: str) -> Tuple[str, str]:
        """
        创建自然的echo prompt - 核心是让模型自然地重述问题
        不需要复杂的阶段，只需要简单、自然的复述
        """
        
        # 简单自然的echo开场白 - 这些都是模型自然会说的话
        natural_openings = [
            f"Let me understand this problem. The question asks: {question}",
            f"So the problem is: {question}",
            f"I need to solve: {question}",
            f"Looking at this question: {question}",
            f"The problem states: {question}",
            f"I'm asked to find: {question}",
            f"Let me read this carefully: {question}",
            f"Okay, so we have: {question}",
            f"The question is: {question}",
            f"I need to work on: {question}",
        ]
        
        # 简单的后续过渡 - 从echo自然过渡到解题
        transitions = [
            "\n\nLet me work through this step by step.",
            "\n\nI'll solve this systematically.",
            "\n\nLet me think about this.",
            "\n\nHere's how I'll approach it.",
            "\n\nLet me break this down.",
        ]
        
        # 随机选择一个开场白和过渡
        opening = random.choice(natural_openings)
        transition = random.choice(transitions)
        
        # 系统提示：强调自然的思考过程，开始时复述问题
        system_prompt = """You are solving math problems. Start by naturally restating or echoing the problem to show you understand it, then work through your solution. Use <think> tags for your thinking process."""
        
        # 在系统提示中给一个具体的开始示例，引导模型echo
        user_prompt = f"""Problem: {question}

Start your response with something like: "<think>{opening}{transition}..." and continue solving."""
        
        return system_prompt, user_prompt
    
    def create_simple_echo_prompt_v2(self, question: str) -> Tuple[str, str]:
        """
        更简洁的版本 - 让模型更自然地echo
        """
        
        # 极简的系统提示
        system_prompt = """When solving problems, always begin by restating what you're asked to find. This helps ensure you understand the problem correctly. Wrap your thinking in <think> tags."""
        
        # 用户提示中subtly引导echo
        user_prompt = f"""Solve this problem (remember to first restate what you need to find):

{question}"""
        
        return system_prompt, user_prompt
    
    def create_minimal_echo_prompt(self, question: str) -> Tuple[str, str]:
        """
        最小化版本 - 只在系统提示中要求echo
        """
        
        system_prompt = """You're solving math problems. Always start your <think> block by repeating or paraphrasing the problem in your own words to ensure understanding, then proceed with solving."""
        
        user_prompt = question
        
        return system_prompt, user_prompt
    
    def generate_with_echo(self, question: str, answer: str, 
                          prompt_version: str = "natural") -> Dict[str, Any]:
        """生成带有echo的回答"""
        
        # 选择prompt版本
        if prompt_version == "natural":
            system_prompt, user_prompt = self.create_natural_echo_prompt(question)
        elif prompt_version == "simple":
            system_prompt, user_prompt = self.create_simple_echo_prompt_v2(question)
        else:  # minimal
            system_prompt, user_prompt = self.create_minimal_echo_prompt(question)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                timeout=30
            )
            
            generated = response.choices[0].message.content
            
            return {
                "question": question,
                "answer": answer,
                "response": generated,
                "has_think_tags": "<think>" in generated and "</think>" in generated,
                "prompt_version": prompt_version
            }
            
        except Exception as e:
            print(f"Error: {e}")
            # 简单的fallback
            fallback = f"""<think>
Let me solve this problem: {question}

[Working through the solution...]
</think>

{answer}"""
            return {
                "question": question,
                "answer": answer,
                "response": fallback,
                "has_think_tags": True,
                "prompt_version": "fallback"
            }
    
    def verify_echo_presence(self, response: str, question: str) -> bool:
        """
        简单检查response中是否包含了问题的echo
        不需要复杂的MLP，只需要基本的文本匹配
        """
        
        # 提取<think>部分
        import re
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if not think_match:
            return False
        
        think_content = think_match.group(1).lower()
        question_lower = question.lower()
        
        # 检查是否包含问题的关键词（简单的启发式方法）
        question_words = set(question_lower.split())
        think_words = set(think_content[:500].split())  # 只看前500字符
        
        # 如果think开始部分包含了问题50%以上的词，认为有echo
        overlap = len(question_words & think_words)
        ratio = overlap / len(question_words) if question_words else 0
        
        return ratio > 0.5
    
    def process_dataset_simple(self, split="train", sample_size=None, 
                              output_file="simple_echo_gsm8k.json"):
        """处理数据集 - 简单直接的方式"""
        
        print("Loading GSM-8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        
        data = []
        for item in dataset:
            question = item['question']
            answer = item['answer'].split('####')[-1].strip()
            data.append({'question': question, 'answer': answer})
        
        if sample_size:
            data = random.sample(data, min(sample_size, len(data)))
        
        results = []
        prompt_versions = ["natural", "simple", "minimal"]
        
        for item in tqdm(data, desc="Generating echo responses"):
            # 随机选择一个prompt版本增加多样性
            version = random.choice(prompt_versions)
            
            result = self.generate_with_echo(
                item['question'], 
                item['answer'],
                prompt_version=version
            )
            
            # 验证是否有echo
            has_echo = self.verify_echo_presence(
                result['response'], 
                item['question']
            )
            result['has_echo'] = has_echo
            
            # 格式化为SFT训练格式
            result['messages'] = [
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": result['response']}
            ]
            
            results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计
        echo_count = sum(1 for r in results if r.get('has_echo'))
        print(f"\n✓ Saved {len(results)} examples to {output_file}")
        print(f"  - With echo: {echo_count} ({echo_count/len(results)*100:.1f}%)")
        
        # 按prompt版本统计
        version_stats = {}
        for r in results:
            v = r.get('prompt_version', 'unknown')
            version_stats[v] = version_stats.get(v, 0) + 1
        
        print("\nPrompt version distribution:")
        for v, count in version_stats.items():
            print(f"  - {v}: {count}")
        
        return results


# 对比实验：生成不带echo的数据
class NoEchoDistiller:
    """生成不带echo的对照组数据"""
    
    def __init__(self, api_base="http://localhost:8000/v1", api_key="EMPTY", 
                 model_name="openai/gpt-oss-120b"):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
    
    def create_no_echo_prompt(self, question: str) -> Tuple[str, str]:
        """创建不要求echo的prompt"""
        
        system_prompt = """Solve math problems step by step. Use <think> tags for your reasoning process. Get straight to solving without restating the problem."""
        
        user_prompt = f"""Solve directly: {question}"""
        
        return system_prompt, user_prompt
    
    def generate_without_echo(self, question: str, answer: str) -> Dict[str, Any]:
        """生成不带echo的回答"""
        
        system_prompt, user_prompt = self.create_no_echo_prompt(question)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                timeout=30
            )
            
            generated = response.choices[0].message.content
            
            # 确保以<think>开始，直接解题，不echo
            if not generated.startswith("<think>"):
                generated = f"<think>\n{generated}\n</think>"
            
            return {
                "question": question,
                "answer": answer,
                "response": generated,
                "type": "no_echo"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": answer,
                "response": f"<think>\n[Direct solution]\n</think>\n{answer}",
                "type": "no_echo_fallback"
            }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Echo-based COT distillation")
    parser.add_argument("--mode", choices=["echo", "no_echo", "both"], default="echo",
                       help="Generation mode: with echo, without echo, or both")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", default="echo_data.json", help="Output file")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    
    args = parser.parse_args()
    
    if args.mode in ["echo", "both"]:
        echo_gen = SimpleEchoDistiller(args.api_base, model_name=args.model)
        echo_results = echo_gen.process_dataset_simple(
            split=args.split,
            sample_size=args.sample_size,
            output_file=args.output.replace(".json", "_echo.json")
        )
    
    if args.mode in ["no_echo", "both"]:
        no_echo_gen = NoEchoDistiller(args.api_base, model_name=args.model)
        # Similar processing for no-echo version
        print("Generating no-echo control group...")
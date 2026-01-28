#!/usr/bin/env python3
"""
Use OpenAI-compatible interface and vLLM direct invocation for GSM8K inference + MLP repeat detection + probability statistics (independent of harness).

Usage:
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m src.evaluation.direct_gsm8k_openai_mlp \
  --base_url http://localhost:8000/v1  \
  --models qwen3-8B DeepSeek-8B gpt-oss \
  --qwen3_path /path/to/Qwen3-8B/ \
  --deepseek_path /path/to/DeepSeek-R1-Distill-Llama-8B/ \
  --vllm_tensor_parallel_size 4 --vllm_trust_remote_code \
  --embedding_model_path /path/to/Qwen3-Embedding-0.6B/ \
  --mlp_probe_path train_mlp/models/repeat_mlp.pt \
Output:
  - Per-model JSONL: {model}_gsm8k_samples.jsonl (contains question/response/think_content/is_repeat/repeat_score)
  - Summary JSON: repeat_summary.json (repeat probability for each model)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- OpenAI Compatible Client ---
try:
    from openai import OpenAI  # Official openai>=1.x client
except Exception as e:
    raise ImportError("Please install openai first: pip install openai>=1.0.0") from e

# --- vLLM Direct Invocation (Optional) ---
try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


# --- MLP 探针定义（避免引入 harness 依赖，复制最小可用结构） ---
class RepeatDetector(nn.Module):
    """两层 MLP（二分类）用于检测重复模式。需要与训练时定义一致。"""

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


# --- 文本解析与特征 ---
_THINK_RE = re.compile(r"<think>\s*(.*?)(?:</think>|$)", re.DOTALL)


def extract_think_content(text: str) -> str:
    """从生成文本中尽力提取 <think>...</think> 内的思考内容。
    若不存在，则回退为 '####' 之前的前缀或完整文本（保守）。"""
    if not text:
        return ""
    m = _THINK_RE.search(text)
    if m:
        content = m.group(1).strip()
        if content:
            return content
    # 回退：若包含 '####'（GSM8K 常见答案分隔），取其前缀
    if "####" in text:
        return text.split("####", 1)[0].strip()
    return text.strip()


def build_messages(question: str) -> List[Dict[str, str]]:
    """构造 ChatCompletions 消息，鼓励在 <think> 中展示推理。"""
    system = (
        "You are a helpful mathematician. Think step by step and put your reasoning in "
        "<think>...</think> tags. After reasoning, provide the final numeric answer after '####'."
    )
    user = (
        "Solve the following math word problem. Show your reasoning inside <think> tags, then give the final answer after '####'.\n\n"
        f"Question: {question}\nAnswer: <think>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_prompt(question: str) -> str:
    """构造用于 vLLM.generate 的纯文本 prompt。"""
    return (
        "You are an expert at solving math problems. Please think step by step. "
        "Enclose your thinking process in <think>...</think> tags.\n\n"
        f"Question: {question}\n"
        "Answer: <think>"
    )


@dataclass
class ProbeResources:
    embedder: SentenceTransformer
    mlp: RepeatDetector
    device: torch.device
    prefix_tokens: int
    threshold: float


def score_repetition(question: str, think_content: str, res: ProbeResources) -> Tuple[float, bool]:
    """计算重复概率（sigmoid 后）与是否重复判定。"""
    if not question or not think_content:
        return 0.0, False

    # 取思考前缀词
    words = think_content.split()
    prefix_text = " ".join(words[: res.prefix_tokens]) if words else ""

    # 计算嵌入
    q_emb = res.embedder.encode(question, convert_to_tensor=True, device=res.device)
    p_emb = res.embedder.encode(prefix_text, convert_to_tensor=True, device=res.device)

    if len(q_emb.shape) == 1:
        q_emb = q_emb.unsqueeze(0)
    if len(p_emb.shape) == 1:
        p_emb = p_emb.unsqueeze(0)

    feats = torch.cat([q_emb, p_emb], dim=1).to(res.device)
    with torch.no_grad():
        prob = torch.sigmoid(res.mlp(feats)).item()
    return prob, bool(prob > res.threshold)


def ensure_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def _process_one_question_openai(
    q: str,
    client: OpenAI,
    model_name: str,
    res: ProbeResources,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
) -> Tuple[Dict, float, bool, bool]:
    """Process a single question: API call with retry, scoring, and result packaging."""
    text = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=build_messages(q),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60,  # Add timeout to prevent hanging
            )
            text = resp.choices[0].message.content or ""
            break  # Success, exit retry loop
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                # Retry for 500 errors
                if "500" in error_msg or "Internal Server Error" in error_msg:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
            # Final attempt failed or non-retryable error
            if "500" in error_msg:
                text = f"[ERROR] Error code: 500"
            else:
                text = f"[ERROR] Error code: {getattr(e, 'status_code', 'unknown')} - {e}"

    think = extract_think_content(text)
    has_think = bool(think)
    prob, is_rep = score_repetition(q, think, res) if think else (0.0, False)

    row = {
        "question": q,
        "model": model_name,
        "response": text,
        "think_content": think,
        "repeat_score": prob,
        "is_repeat": is_rep,
    }
    return row, prob, is_rep, has_think


def infer_one_model_openai(
    client: OpenAI,
    model_name: str,
    questions: Iterable[str],
    res: ProbeResources,
    *,
    max_tokens: int,
    temperature: float,
    output_dir: str,
    concurrency: int = 8,
) -> Dict[str, float]:
    """通过 OpenAI 兼容接口推理并打分。"""
    out_path = os.path.join(output_dir, f"{model_name}_gsm8k_samples.jsonl")
    total = 0
    have_think = 0
    rep_cnt = 0
    scores: List[float] = []
    question_list = list(questions)

    with open(out_path, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _process_one_question_openai, q, client, model_name, res, max_tokens, temperature
                )
                for q in question_list
            ]

            for future in tqdm(
                as_completed(futures), total=len(question_list), desc=f"{model_name}", unit="q"
            ):
                try:
                    row, prob, is_rep, has_think = future.result()
                    total += 1
                    if has_think:
                        have_think += 1
                        scores.append(prob)
                        if is_rep:
                            rep_cnt += 1

                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error processing a question for model {model_name}: {e}")

    stats = {
        "model": model_name,
        "total_samples": total,
        "samples_with_think": have_think,
        "repeat_count": rep_cnt,
        "repeat_frequency": (rep_cnt / have_think) if have_think > 0 else 0.0,
        "avg_repeat_score": (sum(scores) / len(scores)) if scores else 0.0,
        "output_file": out_path,
    }
    return stats


def infer_one_model_vllm(
    model_path: str,
    model_alias: str,
    questions: Iterable[str],
    res: ProbeResources,
    *,
    max_tokens: int,
    temperature: float,
    output_dir: str,
    tensor_parallel_size: int = 1,
    dtype: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Dict[str, float]:
    """通过 vLLM Python API 直调本地模型并打分。"""
    if LLM is None or SamplingParams is None:
        raise ImportError("未安装 vllm，请先安装：pip install vllm")

    out_path = os.path.join(output_dir, f"{model_alias}_gsm8k_samples.jsonl")
    total = 0
    have_think = 0
    rep_cnt = 0
    scores: List[float] = []

    init_kwargs: Dict[str, object] = {
        "model": model_path,
        "tensor_parallel_size": int(tensor_parallel_size),
        "trust_remote_code": bool(trust_remote_code),
    }
    if dtype:
        init_kwargs["dtype"] = dtype

    engine = LLM(**init_kwargs)  # type: ignore

    sp_kwargs: Dict[str, object] = {"max_tokens": int(max_tokens)}
    if temperature and temperature > 0:
        sp_kwargs["temperature"] = float(temperature)
    sp = SamplingParams(**sp_kwargs)  # type: ignore

    with open(out_path, "w", encoding="utf-8") as fout:
        for q in tqdm(questions, desc=f"{model_alias}", unit="q"):
            total += 1
            prompt = build_prompt(q)
            try:
                outs = engine.generate([prompt], sampling_params=sp, use_tqdm=False)
                text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            except Exception as e:
                text = f"[ERROR] {e}"

            think = extract_think_content(text)
            prob, is_rep = score_repetition(q, think, res) if think else (0.0, False)
            if think:
                have_think += 1
                scores.append(prob)
                if is_rep:
                    rep_cnt += 1

            row = {
                "question": q,
                "model": model_alias,
                "response": text,
                "think_content": think,
                "repeat_score": prob,
                "is_repeat": is_rep,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "model": model_alias,
        "total_samples": total,
        "samples_with_think": have_think,
        "repeat_count": rep_cnt,
        "repeat_frequency": (rep_cnt / have_think) if have_think > 0 else 0.0,
        "avg_repeat_score": (sum(scores) / len(scores)) if scores else 0.0,
        "output_file": out_path,
    }
    return stats


def load_gsm8k_questions(split: str, limit: Optional[int]) -> List[str]:
    ds = load_dataset("openai/gsm8k", name="main", split=split)
    questions = list(ds["question"])  # type: ignore[index]
    if limit is not None:
        questions = questions[: int(limit)]
    return questions


def main() -> None:
    ap = argparse.ArgumentParser(description="直接通过 OpenAI 兼容接口评估 GSM8K 的重复概率（MLP 探针）")
    ap.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI 兼容 API base_url")
    ap.add_argument("--api_key", type=str, default="EMPTY", help="API Key（本地服务通常填 EMPTY）")
    ap.add_argument("--models", nargs="+", default=["qwen3-8B", "DeepSeek-8B", "gpt-oss"], help="模型名称列表")
    # vLLM 本地模型路径（用于 qwen3-8B / DeepSeek-8B 直调）
    ap.add_argument("--qwen3_path", type=str, default="/path/to/your/Qwen3-8B/", help="Qwen3-8B local model path (vLLM)")
    ap.add_argument("--deepseek_path", type=str, default="/path/to/your/DeepSeek-R1-Distill-Llama-8B/", help="DeepSeek-8B local model path (vLLM)")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM 张量并行大小")
    ap.add_argument("--vllm_dtype", type=str, default=None, help="vLLM dtype，例如 float16/bfloat16/auto")
    ap.add_argument("--vllm_trust_remote_code", action="store_true", help="vLLM 启动时允许 trust_remote_code")
    # gpt-oss 专属 OpenAI 端口/密钥/模型名（可覆盖全局 base_url、api_key 与模型名）
    ap.add_argument("--gpt_base_url", type=str, default=None, help="gpt-oss 使用的 base_url（默认继承 --base_url）")
    ap.add_argument("--gpt_api_key", type=str, default=None, help="gpt-oss 使用的 api_key（默认继承 --api_key）")
    ap.add_argument("--gpt_model_name", type=str, default="gpt-oss", help="gpt-oss 服务端模型名（默认使用完整路径）")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"], help="GSM8K 数据集切分")
    ap.add_argument("--limit", type=int, default=None, help="可选限制样本数，便于快速试跑")
    ap.add_argument("--output_dir", type=str, default="results/direct_openai_mlp", help="输出目录")
    ap.add_argument("--embedding_model_path", type=str, default="/path/to/your/embedding_model/", help="Sentence embedding model path")
    ap.add_argument("--mlp_probe_path", type=str, default="train_mlp/models/repeat_mlp.pt", help="已训练的 MLP 探针路径")
    ap.add_argument("--mlp_hidden_dim", type=int, default=32, help="MLP 隐藏层维度（与训练一致）")
    ap.add_argument("--answer_prefix_tokens", type=int, default=32, help="用于打分的思考前缀词数")
    ap.add_argument("--mlp_threshold", type=float, default=0.9, help="重复判定阈值（sigmoid 概率 > 阈值则视为重复）")
    ap.add_argument("--max_tokens", type=int, default=2048, help="生成最大 token 数")
    ap.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    ap.add_argument("--concurrency", type=int, default=8, help="OpenAI API 并发请求数")
    args = ap.parse_args()

    # 设备与资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)

    # 加载嵌入与 MLP
    embedder = SentenceTransformer(args.embedding_model_path, device=device)
    input_dim = embedder.get_sentence_embedding_dimension() * 2
    mlp = RepeatDetector(input_dim, hidden_dim=args.mlp_hidden_dim).to(device)
    mlp.load_state_dict(torch.load(args.mlp_probe_path, map_location=device))
    mlp.eval()

    probe_res = ProbeResources(
        embedder=embedder,
        mlp=mlp,
        device=device,
        prefix_tokens=args.answer_prefix_tokens,
        threshold=args.mlp_threshold,
    )

    # OpenAI 客户端（仅给 gpt-oss 使用）
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 数据
    questions = load_gsm8k_questions(args.split, args.limit)
    print(f"Loaded GSM8K[{args.split}] questions: {len(questions)}")

    # 逐模型推理与统计
    all_stats: List[Dict[str, float]] = []
    results_dict: Dict[str, Dict[str, float]] = {}
    gpt_thread: Optional[threading.Thread] = None

    # 默认并行：若包含 gpt-oss，则后台线程跑 gpt-oss（在 CPU 上进行嵌入与 MLP）
    if "gpt-oss" in args.models:
        def _run_gpt_oss_worker() -> None:
            try:
                gpt_base_url = args.gpt_base_url or args.base_url
                gpt_api_key = args.gpt_api_key or args.api_key
                gpt_model = args.gpt_model_name or "gpt-oss"
                gpt_client = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)
                cpu_device = torch.device("cpu")
                embedder_cpu = SentenceTransformer(args.embedding_model_path, device=cpu_device)
                input_dim_cpu = embedder_cpu.get_sentence_embedding_dimension() * 2
                mlp_cpu = RepeatDetector(input_dim_cpu, hidden_dim=args.mlp_hidden_dim).to(cpu_device)
                mlp_cpu.load_state_dict(torch.load(args.mlp_probe_path, map_location=cpu_device))
                mlp_cpu.eval()
                probe_res_cpu = ProbeResources(
                    embedder=embedder_cpu,
                    mlp=mlp_cpu,
                    device=cpu_device,
                    prefix_tokens=args.answer_prefix_tokens,
                    threshold=args.mlp_threshold,
                )
                stats_cpu = infer_one_model_openai(
                    gpt_client,
                    gpt_model,
                    questions,
                    probe_res_cpu,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    output_dir=output_dir,
                    concurrency=args.concurrency,
                )
                results_dict["gpt-oss"] = stats_cpu
            except Exception as e:
                results_dict["gpt-oss"] = {
                    "model": "gpt-oss",
                    "total_samples": 0,
                    "samples_with_think": 0,
                    "repeat_count": 0,
                    "repeat_frequency": 0.0,
                    "avg_repeat_score": 0.0,
                    "output_file": f"[ERROR] {e}",
                }

        gpt_thread = threading.Thread(target=_run_gpt_oss_worker, daemon=True)
        gpt_thread.start()
    for model_name in args.models:
        # 路由：qwen3-8B / DeepSeek-8B -> vLLM；gpt-oss -> OpenAI
        if model_name == "qwen3-8B":
            stats = infer_one_model_vllm(
                model_path=args.qwen3_path,
                model_alias=model_name,
                questions=questions,
                res=probe_res,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=output_dir,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                dtype=args.vllm_dtype,
                trust_remote_code=bool(args.vllm_trust_remote_code),
            )
        elif model_name == "DeepSeek-8B":
            stats = infer_one_model_vllm(
                model_path=args.deepseek_path,
                model_alias=model_name,
                questions=questions,
                res=probe_res,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=output_dir,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                dtype=args.vllm_dtype,
                trust_remote_code=bool(args.vllm_trust_remote_code),
            )
        elif model_name == "gpt-oss":
            # 若已经并行启动，则主循环中跳过，待汇总
            continue
        else:
            # 默认走 OpenAI 兼容接口（例如 gpt-oss）
            stats = infer_one_model_openai(
                client,
                model_name,
                questions,
                probe_res,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=output_dir,
                concurrency=args.concurrency,
            )
        print(
            f"Model={model_name} | with_think={stats['samples_with_think']}/{stats['total_samples']} | "
            f"repeat={stats['repeat_count']} | freq={stats['repeat_frequency']:.4f}"
        )
        all_stats.append(stats)

    # 等待 gpt-oss 完成并加入汇总
    if gpt_thread is not None:
        gpt_thread.join()
        s = results_dict.get("gpt-oss")
        if s is not None:
            print(
                f"Model=gpt-oss | with_think={s['samples_with_think']}/{s['total_samples']} | "
                f"repeat={s['repeat_count']} | freq={s['repeat_frequency']:.4f}"
            )
            all_stats.append(s)

    # 写入汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "num_questions": len(questions),
        "models": args.models,
        "stats": all_stats,
    }
    with open(os.path.join(output_dir, "repeat_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Summary written to: {output_dir}/repeat_summary.json")


if __name__ == "__main__":
    main()

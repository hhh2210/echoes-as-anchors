#!/usr/bin/env python3
"""
Echo-insertion ablation:
Given an echo-free full trace, truncate it by length to form a prefix,
then continue generation under two conditions:
  1) echo: insert a fixed echo phrase before continuing
  2) echo_free: continue directly
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.echo_free_insert_decoding \
  --input_jsonl /path/to/qwen3_8b_wrong_samples.jsonl \
  --output_jsonl /path/to/experiments/echo_prefix_ablation/gsm8k_qwen3_8b_wrong_cut0.5.jsonl \
  --model /path/to/Qwen3-8B/ \
  --cut_ratio 0.5 \
  --max_new_tokens 2048 --temperature 0.7 --top_p 0.9
Outputs a JSONL where each row contains both continuations, with `resps` holding the actual model output (continuation only).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple, Callable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Reuse the project-standard prompt builder
try:
    from src.evaluation.logp_trim_experiment import build_prompt
except Exception:
    # Fallback to a default prompt if import path differs
    def build_prompt(question: str) -> str:  # type: ignore
        return (
            "You are an expert at solving math problems. Please think step by step.\n"
            f"Question: {question}\n"
            "Answer: <think>"
        )


def _build_gsm8k_cot_fewshot_prompt_builder(
    yaml_path: str,
    num_fewshot: Optional[int] = None,
) -> Callable[[str], str]:
    """
    Construct a prompt builder that prepends GSM8K-CoT few-shot examples
    (from lm-eval's gsm8k-cot.yaml) before the current question.

    The resulting prompt roughly matches:
        Q: <fs_q1>
        A: <fs_a1>

        ...

        Q: <fs_qN>
        A: <fs_aN>

        Q: <current question>
        A: <think>
    """
    if yaml is None:
        raise ImportError("PyYAML is not available; cannot load fewshot config.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fewshot_cfg = (cfg or {}).get("fewshot_config") or {}
    samples: List[Dict[str, Any]] = fewshot_cfg.get("samples") or []
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No fewshot samples found in {yaml_path}")

    if num_fewshot is None:
        num_fewshot = cfg.get("num_fewshot", len(samples))
    num_fewshot = max(0, min(int(num_fewshot), len(samples)))

    blocks: List[str] = []
    for s in samples[:num_fewshot]:
        q = s.get("question")
        a = s.get("target") or s.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        blocks.append(f"Q: {q}\nA: {a}")

    fewshot_prefix = "\n\n".join(blocks).strip()

    def _builder(question: str) -> str:
        current_block = f"Q: {question}\nA: <think>"
        if fewshot_prefix:
            return fewshot_prefix + "\n\n" + current_block
        return current_block

    return _builder


def _pick_question(obj: Dict[str, Any]) -> str:
    return (
        obj.get("question")
        or obj.get("problem")
        or (obj.get("doc") or {}).get("question")
        or ""
    )


def _pick_full_trace(obj: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit fields if present
    for k in ("think", "trace", "answer", "full_trace", "echo_free_trace", "resp"):
        val = obj.get(k)
        if isinstance(val, str) and val.strip():
            return val
    # Harness-like format: use the last element of 'pred' or 'resps'
    preds = obj.get("pred")
    if isinstance(preds, list) and preds:
        last = preds[-1]
        if isinstance(last, str) and last.strip():
            return last
    resps = obj.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, str) and first.strip():
            return first
        if isinstance(first, list) and first:
            inner = first[0]
            if isinstance(inner, str) and inner.strip():
                return inner
    return None


def _truncate_prefix_by_ratio(tokenizer: AutoTokenizer, text: str, ratio: float) -> Tuple[list[int], str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [], ""
    cut = max(1, min(len(ids) - 1, int(round(len(ids) * ratio))))
    prefix_ids = ids[:cut]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    return prefix_ids, prefix_text


@torch.no_grad()
def _continue_from_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    prefix_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """
    Returns (continuation_only, full_response_text = prefix_text + continuation_only)
    """
    device = model.device
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0) if prefix_ids else None

    if prefix_tensor is not None:
        full_input_ids = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        full_input_ids = prompt_ids

    gen = model.generate(
        input_ids=full_input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Slice off the input portion to get only the continuation tokens
    cont_ids = gen[0][full_input_ids.shape[1]:]
    continuation = tokenizer.decode(cont_ids, skip_special_tokens=True)
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True) if prefix_ids else ""
    return continuation, (prefix_text + continuation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo insertion ablation over echo-free traces.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to JSONL with fields question + echo-free trace")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Where to write combined JSONL with both conditions")
    parser.add_argument("--model", type=str, default="/path/to/Qwen3-8B/", help="HF model path/name")
    parser.add_argument("--cut_ratio", type=float, default=0.5, help="Fraction of trace tokens to keep as prefix (0,1)")
    parser.add_argument("--echo_phrase", type=str, default="now i need to look back at the question again:", help="Echo phrase to insert")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use_gsm8k_fewshot",
        action="store_true",
        help="If set, build the prompt using GSM8K-CoT few-shot examples from lm-eval's gsm8k-cot.yaml.",
    )
    parser.add_argument(
        "--gsm8k_fewshot_yaml",
        type=str,
        default="path/to/gsm8k-cot.yaml",
        help="Path to gsm8k-cot.yaml used to construct few-shot prompts.",
    )
    parser.add_argument(
        "--gsm8k_num_fewshot",
        type=int,
        default=None,
        help="Override number of GSM8K few-shot examples (default: use num_fewshot from yaml).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    out_f = open(args.output_jsonl, "w", encoding="utf-8")

    total = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:

        # Decide prompt builder: original single-question prompt vs few-shot GSM8K-CoT prompt
        if args.use_gsm8k_fewshot:
            prompt_builder: Callable[[str], str] = _build_gsm8k_cot_fewshot_prompt_builder(
                args.gsm8k_fewshot_yaml,
                num_fewshot=args.gsm8k_num_fewshot,
            )
        else:
            prompt_builder = build_prompt

        for line in tqdm(fin, desc="Echo ablation", unit="lines"):
            if args.limit is not None and total >= args.limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            q = _pick_question(obj)
            full_trace = _pick_full_trace(obj)
            if not q or not full_trace:
                continue

            prompt = prompt_builder(q)
            # Build prefix from echo-free full trace
            prefix_ids, prefix_text = _truncate_prefix_by_ratio(tokenizer, full_trace, args.cut_ratio)
            if not prefix_ids:
                continue

            # Condition A: echo-free (continue directly)
            ef_cont, ef_full = _continue_from_prefix(
                model, tokenizer, prompt, prefix_ids,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
            )

            # Condition B: echo (insert phrase then continue)
            echo_insert_ids = tokenizer.encode(" " + args.echo_phrase, add_special_tokens=False)
            echo_prefix_ids = prefix_ids + echo_insert_ids
            echo_cont, echo_full = _continue_from_prefix(
                model, tokenizer, prompt, echo_prefix_ids,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
            )

            row = {
                "idx": obj.get("idx"),
                "question": q,
                "prefix_ratio": args.cut_ratio,
                "prefix_len_tokens": len(prefix_ids),
                "prefix_text": prefix_text,
                "echo_phrase": args.echo_phrase,
                # echo_free condition
                "echo_free": {
                    "resps": ef_cont,  # 注意：resps 字段才是 LLM 的实际输出（续写部分）
                    "full_response": ef_full,
                },
                # echo condition
                "echo": {
                    "resps": echo_cont,
                    "full_response": echo_full,
                },
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    out_f.close()


if __name__ == "__main__":
    main()

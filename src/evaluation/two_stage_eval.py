#!/usr/bin/env python3
"""
Evaluate models with Two-Stage HFLM in lm-evaluation-harness.
Usage examples:
  # Baseline 2048
  python -m src.evaluation.two_stage_eval \
      --model_path /path/to/your/model \
      --tasks gsm8k --mode baseline2048 --output_dir results/two_stage/baseline2048
Notes:
  - This script uses the Python API (simple_evaluate) and writes results.json;
    if sample logs are returned, it will also write a JSONL for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

try:
    import lm_eval
except Exception as e:
    raise ImportError("Please install lm-evaluation-harness: pip install lm-eval[vllm]") from e

from .two_stage_hflm import TwoStageHFLM
try:
    from .two_stage_vllm import TwoStageVLLM  # optional
except Exception:
    TwoStageVLLM = None  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_results(output_dir: str, results: Dict[str, Any], task_name: str) -> None:
    _ensure_dir(output_dir)
    # Write full results
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Write samples if available
    samples = results.get("samples")
    if isinstance(samples, list) and samples:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        samp_path = os.path.join(output_dir, f"samples_{task_name}_{timestamp}.jsonl")
        with open(samp_path, "w", encoding="utf-8") as fout:
            for row in samples:
                try:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        print(f"✓ 样本日志已写入: {samp_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate with Two-Stage HFLM using lm-eval Python API")
    ap.add_argument("--model_path", type=str, required=True, help="HF model path/name for the base LM")
    ap.add_argument("--tasks", type=str, default="gsm8k", help="Tasks to evaluate (single task supported)")
    ap.add_argument("--mode", type=str, default="baseline2048",
                    choices=["baseline2048", "baseline4096", "two_stage_echo", "two_stage_continue"],
                    help="Evaluation mode")
    ap.add_argument("--first_stage_tokens", type=int, default=2048, help="Stage-1 tokens for two-stage modes")
    ap.add_argument("--second_stage_tokens", type=int, default=2048, help="Stage-2 tokens for two-stage modes")
    ap.add_argument("--injection_template", type=str, default="Look back again: {question}\nSo now I know that ",
                    help="Injection template used in two_stage_echo mode (supports {question})")
    ap.add_argument("--continue_template", type=str, default="",
                    help="Neutral continuation text used in two_stage_continue mode")
    ap.add_argument("--limit", type=int, default=None, help="Optional dataset subset for quick runs")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to write results")
    ap.add_argument("--dtype", type=str, default=None, help="Optional dtype for backend (e.g., float16, bfloat16, auto)")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "hf"], help="Backend selection: auto prefers vLLM if available")
    ap.add_argument("--tensor_parallel_size", type=int, default=None, help="vLLM tensor parallel world size")
    ap.add_argument("--hf_device_map", type=str, default=None, help="HF device_map for multi-GPU sharding (e.g., 'auto')")
    args = ap.parse_args()

    # Construct LM with selected backend
    lm: Any
    backend = "auto"
    if args.backend == "vllm":
        if TwoStageVLLM is None:
            raise ImportError("vLLM is not installed. Please: pip install vllm")
        lm = TwoStageVLLM(
            pretrained=args.model_path,
            mode=args.mode,
            first_stage_tokens=args.first_stage_tokens,
            second_stage_tokens=args.second_stage_tokens,
            injection_template=args.injection_template,
            continue_template=args.continue_template,
            dtype=args.dtype,
            tensor_parallel_size=(args.tensor_parallel_size or 1),
        )
        backend = "vllm"
    elif args.backend == "hf":
        lm_kwargs: Dict[str, Any] = {
            "pretrained": args.model_path,
            "trust_remote_code": True,
        }
        if args.dtype:
            lm_kwargs["dtype"] = args.dtype
        if args.hf_device_map:
            lm_kwargs["device_map"] = args.hf_device_map
        lm = TwoStageHFLM(
            mode=args.mode,
            first_stage_tokens=args.first_stage_tokens,
            second_stage_tokens=args.second_stage_tokens,
            injection_template=args.injection_template,
            continue_template=args.continue_template,
            **lm_kwargs,
        )
        backend = "hf"
    else:  # auto
        if TwoStageVLLM is not None:
            lm = TwoStageVLLM(
                pretrained=args.model_path,
                mode=args.mode,
                first_stage_tokens=args.first_stage_tokens,
                second_stage_tokens=args.second_stage_tokens,
                injection_template=args.injection_template,
                continue_template=args.continue_template,
                dtype=args.dtype,
                tensor_parallel_size=(args.tensor_parallel_size or 1),
            )
            backend = "vllm"
        else:
            lm_kwargs = {
                "pretrained": args.model_path,
                "trust_remote_code": True,
            }
            if args.dtype:
                lm_kwargs["dtype"] = args.dtype
            if args.hf_device_map:
                lm_kwargs["device_map"] = args.hf_device_map
            lm = TwoStageHFLM(
                mode=args.mode,
                first_stage_tokens=args.first_stage_tokens,
                second_stage_tokens=args.second_stage_tokens,
                injection_template=args.injection_template,
                continue_template=args.continue_template,
                **lm_kwargs,
            )
            backend = "hf"

    out_dir = Path(args.output_dir).expanduser().resolve().as_posix()
    print(f"Starting evaluation: task={args.tasks}, mode={args.mode}, backend={backend}, output: {out_dir}")
    # 兼容旧版 harness：不传 output_path / log_samples
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.tasks],
        limit=args.limit,
    )

    # Persist outputs
    _write_results(out_dir, results, args.tasks)
    print(f"✓ Evaluation completed, results written to: {out_dir}/results.json")


if __name__ == "__main__":
    main()



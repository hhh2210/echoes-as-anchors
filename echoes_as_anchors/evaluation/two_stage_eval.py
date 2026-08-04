#!/usr/bin/env python3
"""Canonical packaged entry point for new two-stage lm-eval runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .registry import register_models
from .two_stage_common import (
    DEFAULT_INJECTION_TEMPLATE,
    DEFAULT_PROTOCOL_ID,
    VALID_MODES,
    run_manifest,
)


def _load_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    allowed = {
        "task",
        "mode",
        "protocol_id",
        "first_stage_tokens",
        "second_stage_tokens",
        "injection_template",
        "continue_template",
        "seed",
        "backend",
        "limit",
        "dtype",
        "tensor_parallel_size",
        "hf_device_map",
        "revision",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unsupported config keys: {', '.join(sorted(unknown))}")
    return value


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    defaults = _load_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Run a recorded Echoes two-stage evaluation with lm-eval 0.4.12"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--task", "--tasks", dest="task", default="gsm8k")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="baseline2048")
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument(
        "--first-stage-tokens",
        "--first_stage_tokens",
        dest="first_stage_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--second-stage-tokens",
        "--second_stage_tokens",
        dest="second_stage_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--injection-template",
        "--injection_template",
        dest="injection_template",
        default=DEFAULT_INJECTION_TEMPLATE,
    )
    parser.add_argument(
        "--continue-template",
        "--continue_template",
        dest="continue_template",
        default="",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto")
    parser.add_argument("--dtype")
    parser.add_argument(
        "--tensor-parallel-size",
        "--tensor_parallel_size",
        dest="tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument("--hf-device-map", "--hf_device_map", dest="hf_device_map")
    parser.add_argument("--revision")
    parser.set_defaults(**defaults)
    return parser


def _select_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "vllm" if importlib.util.find_spec("vllm") is not None else "hf"


def _write_results(
    output_dir: Path,
    results: dict[str, Any],
    manifest: dict[str, Any],
    task: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    samples = results.get("samples")
    sample_groups = samples if isinstance(samples, dict) else {task: samples}
    for sample_task, rows in sample_groups.items():
        if not isinstance(rows, list) or not rows:
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        safe_task = "".join(char if char.isalnum() or char in "._-" else "-" for char in str(sample_task))
        sample_path = output_dir / f"samples_{safe_task}_{stamp}.jsonl"
        with sample_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    if args.backend not in {"auto", "hf", "vllm"}:
        parser.error("backend must be auto, hf, or vllm")
    if args.first_stage_tokens <= 0 or args.second_stage_tokens <= 0:
        parser.error("stage token budgets must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    register_models()
    import lm_eval

    backend = _select_backend(args.backend)
    alias = f"echoes-{backend}"
    model_args: dict[str, Any] = {
        "pretrained": args.model_path,
        "mode": args.mode,
        "first_stage_tokens": args.first_stage_tokens,
        "second_stage_tokens": args.second_stage_tokens,
        "injection_template": args.injection_template,
        "continue_template": args.continue_template,
        "trust_remote_code": True,
    }
    if args.dtype:
        model_args["dtype"] = args.dtype
    if args.revision:
        model_args["revision"] = args.revision
    if backend == "vllm":
        model_args["tensor_parallel_size"] = args.tensor_parallel_size
    elif args.hf_device_map:
        model_args["device_map"] = args.hf_device_map

    manifest = run_manifest(
        model=args.model_path,
        task=args.task,
        backend=backend,
        mode=args.mode,
        protocol_id=args.protocol_id,
        first_stage_tokens=args.first_stage_tokens,
        second_stage_tokens=args.second_stage_tokens,
        injection_template=args.injection_template,
        continue_template=args.continue_template,
        seed=args.seed,
        limit=args.limit,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_device_map=args.hf_device_map,
        model_revision=args.revision,
        lm_eval_version=version("lm-eval"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    results = lm_eval.simple_evaluate(
        model=alias,
        model_args=model_args,
        tasks=[args.task],
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        metadata=manifest,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")
    _write_results(args.output_dir, results, manifest, args.task)
    print(f"evaluation complete: {args.output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

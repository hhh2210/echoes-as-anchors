"""Dependency-free helpers shared by the HF and vLLM two-stage adapters."""

from __future__ import annotations

import re
from typing import Any


VALID_MODES = frozenset(
    {"baseline2048", "baseline4096", "two_stage_echo", "two_stage_continue"}
)
DEFAULT_PROTOCOL_ID = "repo-standalone-v1-not-figure4"
DEFAULT_INJECTION_TEMPLATE = "Look back again: {question}\nSo now I know that "


def validate_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def validate_injection_template(template: str) -> str:
    if template.count("{question}") != 1:
        raise ValueError("injection template must contain {question} exactly once")
    return template


def extract_question(prompt: str) -> str:
    pairs = list(
        re.finditer(r"\bQuestion:\s*(.*?)\r?\n\s*Answer:\s*", prompt, re.DOTALL)
    )
    if pairs:
        return pairs[-1].group(1).strip()
    questions = list(re.finditer(r"\bQuestion:\s*(.*)$", prompt, re.DOTALL))
    if questions:
        segment = questions[-1].group(1)
        next_question = segment.find("\n\nQuestion:")
        if next_question != -1:
            segment = segment[:next_question]
        return segment.strip()
    return prompt


def apply_until(text: str, until: list[str] | None) -> str:
    trimmed = text
    for term in until or []:
        if term and term in trimmed:
            trimmed = trimmed.split(term, 1)[0]
    return trimmed


def sanitize_until(until: list[str] | None) -> list[str]:
    return [
        term
        for term in until or []
        if term is not None and str(term).strip().lower() != "question:"
    ]


def run_manifest(
    *,
    model: str,
    task: str,
    backend: str,
    mode: str,
    protocol_id: str,
    first_stage_tokens: int,
    second_stage_tokens: int,
    injection_template: str,
    continue_template: str,
    seed: int,
    limit: int | None,
    dtype: str | None,
    tensor_parallel_size: int,
    hf_device_map: str | None,
    model_revision: str | None,
    lm_eval_version: str,
) -> dict[str, Any]:
    validate_mode(mode)
    if mode == "two_stage_echo":
        validate_injection_template(injection_template)
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "paper_figure_claim": None,
        "model": model,
        "task": task,
        "backend": backend,
        "mode": mode,
        "first_stage_tokens": first_stage_tokens,
        "second_stage_tokens": second_stage_tokens,
        "injection_template": injection_template,
        "continue_template": continue_template,
        "seed": seed,
        "limit": limit,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "hf_device_map": hf_device_map,
        "model_revision": model_revision,
        "trust_remote_code": True,
        "decoding": {"do_sample": False},
        "lm_eval_version": lm_eval_version,
    }

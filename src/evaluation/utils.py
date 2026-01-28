#!/usr/bin/env python3
"""
Utility functions for attention analysis
"""

import json
import math
from typing import Any, Dict, List, Optional


def truncate_at_next_question(text: str) -> str:
    """Truncate text at the next question marker."""
    if not isinstance(text, str):
        return text
    markers = ["\n\nQuestion:", "\nQuestion:"]
    cut = len(text)
    for m in markers:
        pos = text.find(m)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut]


def longest_common_suffix_len(a: List[int], b: List[int]) -> int:
    """Find the length of the longest common suffix between two token lists."""
    i, j, cnt = len(a) - 1, len(b) - 1, 0
    while i >= 0 and j >= 0 and a[i] == b[j]:
        cnt += 1
        i -= 1
        j -= 1
    return cnt


def extract_think(answer_text: str) -> tuple[str, Optional[str]]:
    """Extract the part before and after <think> tag."""
    import re
    m = re.search(r"(.*?<think>\s*)(.*)", answer_text, re.DOTALL)
    if not m:
        return "", None
    return m.group(1), m.group(2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    """Read rows from a JSONL file."""
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")
    return rows


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate attention metrics from rows."""
    def _mean(xs: List[float]) -> float:
        xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)]
        return sum(xs) / len(xs) if xs else float("nan")

    def _get(key: str) -> List[float]:
        parts = key.split(".")
        vals = []
        for r in rows:
            x = r
            try:
                for p in parts:
                    x = x[p]
                vals.append(float(x))
            except Exception:
                pass
        return vals

    keys = [
        "last_layer.ans_to_question",
        "last_layer.ans_to_ans_prefix",
        "last_layer.ans_tail_to_removed_prefix",
        "all_layers_mean.ans_to_question",
        "all_layers_mean.ans_to_ans_prefix",
        "all_layers_mean.ans_tail_to_removed_prefix",
    ]
    return {k: _mean(_get(k)) for k in keys}


def parse_bucket_def(bucket_def: str, num_layers: int) -> Dict[str, tuple[int, int]]:
    """Parse bucket definition string into layer ranges."""
    buckets: Dict[str, tuple[int, int]] = {}
    try:
        parts = [p.strip() for p in bucket_def.split(',') if p.strip()]
        for p in parts:
            name, rng = p.split(':')
            a, b = rng.split('-')
            i = max(0, int(a))
            j = min(num_layers - 1, int(b))
            if i > j:
                i, j = j, i
            buckets[name] = (i, j)
    except Exception:
        # fallback: evenly split 3 buckets
        t = num_layers
        a = (0, max(0, t // 3 - 1))
        b = (a[1] + 1, max(a[1] + 1, 2 * t // 3 - 1))
        c = (b[1] + 1, t - 1)
        buckets = {"early": a, "mid": b, "late": c}
    return buckets
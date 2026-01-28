#!/usr/bin/env python3
"""
Statistical analysis utilities for attention metrics
"""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from .utils import parse_bucket_def
except ImportError:
    from utils import parse_bucket_def


def cohen_d(x: List[float], y: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Args:
        x: First group values
        y: Second group values
    
    Returns:
        Cohen's d effect size
    """
    x = [v for v in x if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    y = [v for v in y if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) * (v - mx) for v in x) / (len(x) - 1)
    vy = sum((v - my) * (v - my) for v in y) / (len(y) - 1)
    
    # pooled standard deviation
    s = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    s = math.sqrt(s) if s > 0 else float("nan")
    
    if not s or math.isnan(s):
        return float("nan")
    
    return (mx - my) / s


def auc(pos: List[float], neg: List[float]) -> float:
    """
    Calculate AUC (Area Under Curve) using Mann-Whitney U statistic.
    
    Args:
        pos: Positive class values
        neg: Negative class values
    
    Returns:
        AUC score
    """
    pos = [v for v in pos if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    neg = [v for v in neg if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return float("nan")
    
    # Create combined list with labels
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    
    # Assign average ranks for ties
    ranks: List[float] = [0.0] * (m + n)
    i = 0
    while i < m + n:
        j = i + 1
        while j < m + n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    
    # Sum ranks for positives
    R_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    U = R_pos - m * (m + 1) / 2.0
    return U / (m * n)


def compute_layer_stats(
    correct_rows: List[Dict[str, Any]], 
    wrong_rows: List[Dict[str, Any]], 
    bucket_def: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute layer-wise statistics comparing correct and wrong answers.
    
    Args:
        correct_rows: Rows containing correct answer metrics
        wrong_rows: Rows containing wrong answer metrics
        bucket_def: Bucket definition string (e.g., "early:0-6,mid:7-18,late:19-31")
    
    Returns:
        Dictionary containing layer statistics and bucket analysis
    """
    try:
        from .utils import parse_bucket_def
    except ImportError:
        from src.evaluation.utils import parse_bucket_def
    
    # Find first row with per_layer data
    def _first_per_layer(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for r in rows:
            pl = r.get("per_layer")
            if isinstance(pl, dict):
                return pl
        return None

    pl_c = _first_per_layer(correct_rows)
    pl_w = _first_per_layer(wrong_rows)
    
    if pl_c is None and pl_w is None:
        return {"error": "no per_layer data present"}
    
    # Determine number of layers
    arr = None
    if pl_c and isinstance(pl_c.get("ans_to_ans_prefix"), list):
        arr = pl_c["ans_to_ans_prefix"]
    elif pl_w and isinstance(pl_w.get("ans_to_ans_prefix"), list):
        arr = pl_w["ans_to_ans_prefix"]
    
    if arr is None:
        return {"error": "per_layer.ans_to_ans_prefix missing"}
    
    L = len(arr)  # number of layers

    def _collect(rows: List[Dict[str, Any]], key: str) -> List[List[float]]:
        out: List[List[float]] = []
        for r in rows:
            pl = r.get("per_layer")
            if not isinstance(pl, dict):
                continue
            vals = pl.get(key)
            if isinstance(vals, list) and len(vals) == L:
                out.append([float(v) for v in vals])
        return out

    C_pref = _collect(correct_rows, "ans_to_ans_prefix")
    W_pref = _collect(wrong_rows, "ans_to_ans_prefix")
    C_q = _collect(correct_rows, "ans_to_question")
    W_q = _collect(wrong_rows, "ans_to_question")

    # Layer-wise statistics
    layer_indices = list(range(L))
    d_pref: List[float] = []
    auc_pref: List[float] = []
    d_q: List[float] = []
    auc_q: List[float] = []

    for i in range(L):
        c_i_pref = [row[i] for row in C_pref]
        w_i_pref = [row[i] for row in W_pref]
        c_i_q = [row[i] for row in C_q]
        w_i_q = [row[i] for row in W_q]
        
        d_pref.append(cohen_d(c_i_pref, w_i_pref))
        auc_pref.append(auc(c_i_pref, w_i_pref))
        d_q.append(cohen_d(c_i_q, w_i_q))
        auc_q.append(auc(c_i_q, w_i_q))

    # Bucket analysis
    buckets = parse_bucket_def(bucket_def or "early:0-6,mid:7-18,late:19-31", L)
    bucket_stats: Dict[str, Dict[str, float]] = {}

    def _avg_in_bucket(samples: List[List[float]], a: int, b: int) -> List[float]:
        if a < 0 or b >= L or a > b:
            return []
        out: List[float] = []
        span = b - a + 1
        for row in samples:
            s = 0.0
            for j in range(a, b + 1):
                s += row[j]
            out.append(s / span)
        return out

    for name, (a, b) in buckets.items():
        c_pref_bucket = _avg_in_bucket(C_pref, a, b)
        w_pref_bucket = _avg_in_bucket(W_pref, a, b)
        c_q_bucket = _avg_in_bucket(C_q, a, b)
        w_q_bucket = _avg_in_bucket(W_q, a, b)
        
        bucket_stats[name] = {
            "ans_to_ans_prefix_auc": auc(c_pref_bucket, w_pref_bucket),
            "ans_to_ans_prefix_d": cohen_d(c_pref_bucket, w_pref_bucket),
            "ans_to_question_auc": auc(c_q_bucket, w_q_bucket),
            "ans_to_question_d": cohen_d(c_q_bucket, w_q_bucket),
            "layers": [a, b],
        }

    # Identify peak layers
    def _topk(vals: List[float], k: int = 5) -> List[int]:
        idx = list(range(len(vals)))
        idx.sort(key=lambda i: (float('-inf') if math.isnan(vals[i]) else vals[i]), reverse=True)
        return idx[:k]

    top_auc_pref = _topk(auc_pref)
    top_d_pref = _topk(d_pref)

    return {
        "num_layers": L,
        "layer_indices": layer_indices,
        "ans_to_ans_prefix": {
            "auc": auc_pref, 
            "cohen_d": d_pref, 
            "top_auc_layers": top_auc_pref, 
            "top_d_layers": top_d_pref
        },
        "ans_to_question": {
            "auc": auc_q, 
            "cohen_d": d_q
        },
        "buckets": bucket_stats,
        "counts": {
            "correct": len(C_pref), 
            "wrong": len(W_pref)
        },
    }
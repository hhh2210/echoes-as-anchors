#!/usr/bin/env python3
"""
Attention from converted JSONL
==============================

用途：直接读取 lm-eval 转换后的 JSONL（每行包含 problem/pred/is_correct），
可选启用 MLP 语义阈值法删除 <think> 开头的 prompt echo，随后计算注意力指标：
 - answer→question
 - answer→answer 前缀（首 K 个）
 - answer 尾段→被删前缀（仅在能稳定估计被删前缀长度时启用）

输入 JSONL（由 convert_lm_eval_for_logp.py 生成）：
  {"idx": int, "problem": str, "pred": [str, ...], "is_correct": bool}

输出：
  attention_metrics_correct.jsonl / attention_metrics_wrong.jsonl / attention_summary.json

与 .cursorrules/README 对齐：
 - 采用与 logp 流水线一致的 build_prompt(question)
 - 可选使用 train_mlp/utils.py 的 echo 检测阈值 (initial_threshold, drop_threshold)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math as _math

try:
    # 复用与 logp 流水线一致的 prompt
    from src.evaluation.logp_trim_experiment import build_prompt
except Exception:
    def build_prompt(question: str) -> str:  # type: ignore
        return (
            "You are an expert at solving math problems. Please think step by step.\n"
            f"Question: {question}\n"
            "Answer: <think>"
        )


@dataclass
class Span:
    start: int
    end: int  # end exclusive


def _load_model_and_tokenizer(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 若提供本地目录，则优先只用本地文件，避免被当作 Hub repo_id 校验
    local_only = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=local_only,
    ).to(device).eval()
    return tokenizer, model, device


def _token_ids(text: str, tokenizer: AutoTokenizer, add_special_tokens: bool = True) -> List[int]:
    enc = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors=None)
    if isinstance(enc, dict):
        return enc["input_ids"]  # type: ignore[index]
    return enc.input_ids  # type: ignore[attr-defined]


def _find_question_token_span_in_prompt(
    tokenizer: AutoTokenizer,
    prompt_with_question: str,
    raw_question: str,
) -> Optional[Span]:
    try:
        start_char = prompt_with_question.find(raw_question)
        if start_char < 0:
            return None
        end_char = start_char + len(raw_question)
        enc = tokenizer(prompt_with_question, add_special_tokens=True, return_offsets_mapping=True)  # type: ignore[arg-type]
        offsets = enc["offset_mapping"]  # type: ignore[index]
        token_start = None
        token_end = None
        for idx, (s, e) in enumerate(offsets):
            if e == 0 and s == 0:
                continue
            if e > start_char and s < end_char:
                if token_start is None:
                    token_start = idx
                token_end = idx + 1
        if token_start is None or token_end is None:
            return None
        return Span(start=token_start, end=token_end)
    except Exception:
        return None


def _collect_attentions(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = out.attentions  # type: ignore[attr-defined]
    if attentions is None:
        raise RuntimeError("Model did not return attentions; ensure output_attentions=True is supported.")
    return [a.squeeze(0).to(torch.float32) for a in attentions]


def _mean_heads(attn: torch.Tensor) -> torch.Tensor:
    return attn.mean(dim=0)


def _mean_layers(attns: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([_mean_heads(a) for a in attns], dim=0).mean(dim=0)


def _gather_mass(attn_mat: torch.Tensor, src_idx: List[int], dst_idx: List[int]) -> float:
    if not src_idx or not dst_idx:
        return float("nan")
    src = torch.tensor(src_idx, dtype=torch.long, device=attn_mat.device)
    dst = torch.tensor(dst_idx, dtype=torch.long, device=attn_mat.device)
    mass_per_src = attn_mat.index_select(0, src).index_select(1, dst).sum(dim=1)
    return float(mass_per_src.mean().item())


def _truncate_at_next_question(text: str) -> str:
    if not isinstance(text, str):
        return text
    markers = ["\n\nQuestion:", "\nQuestion:"]
    cut = len(text)
    for m in markers:
        pos = text.find(m)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut]


def _longest_common_suffix_len(a: List[int], b: List[int]) -> int:
    i, j, cnt = len(a) - 1, len(b) - 1, 0
    while i >= 0 and j >= 0 and a[i] == b[j]:
        cnt += 1
        i -= 1
        j -= 1
    return cnt


def _extract_think(answer_text: str) -> Tuple[str, Optional[str]]:
    import re
    m = re.search(r"(.*?<think>\s*)(.*)", answer_text, re.DOTALL)
    if not m:
        return "", None
    return m.group(1), m.group(2)


def _load_embedder(embedding_model_path: Optional[str], device: torch.device):
    if embedding_model_path is None:
        return None
    try:
        from src.data_processing.mlp_pipeline.utils import load_embedding_model, init_nltk  # type: ignore
        init_nltk()
        return load_embedding_model(embedding_model_path, device=str(device))
    except Exception:
        try:
            from train_mlp.utils import load_embedding_model, init_nltk  # type: ignore
            init_nltk()
            return load_embedding_model(embedding_model_path, device=str(device))
        except Exception:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(embedding_model_path, device=str(device))


def _remove_echo_with_mlp(
    question: str,
    answer_text: str,
    embed_model,
    initial_threshold: float,
    drop_threshold: float,
) -> Tuple[str, Optional[int]]:
    """
    删除 <think> 开头的重复前缀。返回 (清洗后的答案, 估计被删的前缀 token 数)。
    若无法估计 token 数，第二项为 None。
    """
    if embed_model is None:
        return answer_text, None
    try:
        try:
            from src.data_processing.mlp_pipeline.utils import find_repetition_boundary  # type: ignore
        except Exception:
            from train_mlp.utils import find_repetition_boundary  # type: ignore

        prefix, think = _extract_think(answer_text)
        if think is None or not think.strip():
            return answer_text, None
        is_rep, prefix_len_chars = find_repetition_boundary(
            question, think, embed_model, initial_threshold, drop_threshold
        )
        if is_rep != 1 or prefix_len_chars <= 0 or prefix_len_chars >= len(think):
            return answer_text, None
        # 删除字符级前缀
        kept = think[prefix_len_chars:]
        kept = kept.lstrip(" \n\t.,;:!?-")
        cleaned = prefix + kept
        # 估计被删 token 数：在答案段上做“公共后缀”对齐
        return cleaned, "__ESTIMATE_REMOVED__"
    except Exception:
        return answer_text, None


def _estimate_removed_tokens_via_suffix(
    tokenizer: AutoTokenizer,
    question: str,
    raw_answer: str,
    cleaned_answer: str,
) -> Optional[int]:
    prompt = build_prompt(question)
    raw_ids = _token_ids(prompt + raw_answer, tokenizer, add_special_tokens=True)
    clean_ids = _token_ids(prompt + cleaned_answer, tokenizer, add_special_tokens=True)
    prompt_ids = _token_ids(prompt, tokenizer, add_special_tokens=True)
    p = len(prompt_ids)
    raw_ans_ids_only = raw_ids[p:]
    clean_ans_ids_only = clean_ids[p:]
    suf = _longest_common_suffix_len(raw_ans_ids_only, clean_ans_ids_only)
    if suf <= 0:
        return None
    return max(0, len(raw_ans_ids_only) - suf)


def _compute_attention_for_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    answer_text: str,
    answer_prefix_tokens: int,
    removed_prefix_tokens: Optional[int],
    return_per_layer: bool = False,
    use_removed_as_prefix: bool = False,
) -> Dict[str, Any]:
    prompt = build_prompt(question)
    full_text = prompt + answer_text
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    p_len = len(_token_ids(prompt, tokenizer, add_special_tokens=True))
    T = int(input_ids.shape[1])
    ans_len = max(0, T - p_len)

    q_span = _find_question_token_span_in_prompt(tokenizer, prompt, question)

    attns = _collect_attentions(model, input_ids, attn_mask)
    last = _mean_heads(attns[-1])
    mean = _mean_layers(attns)

    ans_indices = list(range(p_len, T))
    q_indices = list(range(q_span.start, q_span.end)) if q_span else []
    # Decide prefix window length: prefer MLP-estimated removed length when requested and available
    if use_removed_as_prefix and removed_prefix_tokens is not None and removed_prefix_tokens > 0:
        k = min(int(removed_prefix_tokens), ans_len)
    else:
        k = min(answer_prefix_tokens, ans_len)
    ans_pref = list(range(p_len, p_len + k)) if k > 0 else []

    if removed_prefix_tokens is not None and removed_prefix_tokens > 0:
        r_k = min(removed_prefix_tokens, ans_len)
        removed_pref = list(range(p_len, p_len + r_k))
        ans_tail = list(range(p_len + r_k, T))
    else:
        removed_pref = []
        ans_tail = []

    out: Dict[str, Any] = {
        "seq_len": T,
        "prompt_len": p_len,
        "answer_len": ans_len,
        "used_ans_prefix_len": k,
        "question_span": [q_span.start, q_span.end] if q_span else None,
        "last_layer": {
            "ans_to_question": _gather_mass(last, ans_indices, q_indices) if q_indices else float("nan"),
            "ans_to_ans_prefix": _gather_mass(last, ans_indices, ans_pref) if ans_pref else float("nan"),
            "ans_tail_to_removed_prefix": _gather_mass(last, ans_tail, removed_pref) if removed_pref and ans_tail else float("nan"),
        },
        "all_layers_mean": {
            "ans_to_question": _gather_mass(mean, ans_indices, q_indices) if q_indices else float("nan"),
            "ans_to_ans_prefix": _gather_mass(mean, ans_indices, ans_pref) if ans_pref else float("nan"),
            "ans_tail_to_removed_prefix": _gather_mass(mean, ans_tail, removed_pref) if removed_pref and ans_tail else float("nan"),
        },
    }

    if return_per_layer:
        per_q: List[float] = []
        per_pref: List[float] = []
        for a in attns:
            m = _mean_heads(a)
            vq = _gather_mass(m, ans_indices, q_indices) if q_indices else float("nan")
            vp = _gather_mass(m, ans_indices, ans_pref) if ans_pref else float("nan")
            per_q.append(float(vq))
            per_pref.append(float(vp))
        out["per_layer"] = {
            "ans_to_question": per_q,
            "ans_to_ans_prefix": per_pref,
        }

    return out


def _gather_mass_per_head(attn_heads: torch.Tensor, src_idx: List[int], dst_idx: List[int]) -> List[float]:
    """
    attn_heads: Tensor[H, T, T]
    returns: list of length H with mean mass from src to dst for each head
    """
    H = int(attn_heads.shape[0])
    if not src_idx or not dst_idx:
        return [float("nan")] * H
    src = torch.tensor(src_idx, dtype=torch.long, device=attn_heads.device)
    dst = torch.tensor(dst_idx, dtype=torch.long, device=attn_heads.device)
    vals: List[float] = []
    for h in range(H):
        mat = attn_heads[h]
        mass_per_src = mat.index_select(0, src).index_select(1, dst).sum(dim=1)
        vals.append(float(mass_per_src.mean().item()))
    return vals


def _compute_per_head_for_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    answer_text: str,
    answer_prefix_tokens: int,
    removed_prefix_tokens: Optional[int],
) -> Dict[str, Any]:
    """Compute per-head attention masses per layer for two metrics.
    Returns dict with keys: per_head -> { ans_to_question: List[List[float]], ans_to_ans_prefix: List[List[float]] }
    Shapes: [num_layers][num_heads]
    """
    prompt = build_prompt(question)
    full_text = prompt + answer_text
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    p_len = len(_token_ids(prompt, tokenizer, add_special_tokens=True))
    T = int(input_ids.shape[1])
    ans_len = max(0, T - p_len)

    q_span = _find_question_token_span_in_prompt(tokenizer, prompt, question)

    attns = _collect_attentions(model, input_ids, attn_mask)

    ans_indices = list(range(p_len, T))
    q_indices = list(range(q_span.start, q_span.end)) if q_span else []

    if removed_prefix_tokens is not None and removed_prefix_tokens > 0:
        k = min(int(removed_prefix_tokens), ans_len)
    else:
        k = min(answer_prefix_tokens, ans_len)
    ans_pref = list(range(p_len, p_len + k)) if k > 0 else []

    per_layer_heads_q: List[List[float]] = []
    per_layer_heads_pref: List[List[float]] = []
    for a in attns:
        # a: [H, T, T]
        per_layer_heads_q.append(_gather_mass_per_head(a, ans_indices, q_indices) if q_indices else [])
        per_layer_heads_pref.append(_gather_mass_per_head(a, ans_indices, ans_pref) if ans_pref else [])

    return {
        "per_head": {
            "ans_to_question": per_layer_heads_q,
            "ans_to_ans_prefix": per_layer_heads_pref,
        },
        "used_ans_prefix_len": k,
        "seq_len": T,
    }


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
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


def _process_file(
    in_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    answer_prefix_tokens: int,
    embedder,
    initial_threshold: float,
    drop_threshold: float,
    file_label: str,
    use_removed_as_prefix: bool,
    want_per_layer: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # 统计总行数用于 tqdm 进度条
    try:
        with open(in_path, "r", encoding="utf-8") as fcnt:
            total_lines = sum(1 for _ in fcnt)
    except Exception:
        total_lines = None

    with open(in_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, total=total_lines, desc=f"Processing {file_label}", unit="lines")
        for line in iterator:
            if not line.strip():
                continue
            s = json.loads(line)
            try:
                q = s.get("problem", "")
                preds = s.get("pred", [])
                if not isinstance(preds, list) or not preds:
                    continue
                a_raw = _truncate_at_next_question(preds[-1] if isinstance(preds[-1], str) else str(preds[-1]))

                # MLP echo 去除（可选）
                removed_tokens_est: Optional[int] = None
                if embedder is not None:
                    a_clean, marker = _remove_echo_with_mlp(q, a_raw, embedder, initial_threshold, drop_threshold)
                    if marker == "__ESTIMATE_REMOVED__":
                        est = _estimate_removed_tokens_via_suffix(tokenizer, q, a_raw, a_clean)
                        removed_tokens_est = est
                    a_for_attn = a_raw if (removed_tokens_est is None) else a_raw
                else:
                    a_for_attn = a_raw

                metrics = _compute_attention_for_pair(
                    model,
                    tokenizer,
                    q,
                    a_for_attn,
                    answer_prefix_tokens,
                    removed_tokens_est,
                    return_per_layer=want_per_layer,
                    use_removed_as_prefix=use_removed_as_prefix,
                )
                rows.append({
                    "idx": s.get("idx"),
                    "is_correct": s.get("is_correct"),
                    **metrics,
                })
            except Exception as e:
                rows.append({
                    "idx": s.get("idx"),
                    "error": f"{type(e).__name__}: {e}",
                })
    return rows


def _cohen_d(x: List[float], y: List[float]) -> float:
    x = [v for v in x if isinstance(v, (int, float)) and not _math.isnan(v) and not _math.isinf(v)]
    y = [v for v in y if isinstance(v, (int, float)) and not _math.isnan(v) and not _math.isinf(v)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) * (v - mx) for v in x) / (len(x) - 1)
    vy = sum((v - my) * (v - my) for v in y) / (len(y) - 1)
    # pooled std
    s = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    s = _math.sqrt(s) if s > 0 else float("nan")
    if not s or _math.isnan(s):
        return float("nan")
    return (mx - my) / s


def _auc(pos: List[float], neg: List[float]) -> float:
    pos = [v for v in pos if isinstance(v, (int, float)) and not _math.isnan(v) and not _math.isinf(v)]
    neg = [v for v in neg if isinstance(v, (int, float)) and not _math.isnan(v) and not _math.isinf(v)]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return float("nan")
    # Mann-Whitney U via ranks
    # Create combined list with labels
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    # assign average ranks for ties
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
    # sum ranks for positives
    R_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    U = R_pos - m * (m + 1) / 2.0
    return U / (m * n)


def _parse_bucket_def(bucket_def: str, num_layers: int) -> Dict[str, Tuple[int, int]]:
    buckets: Dict[str, Tuple[int, int]] = {}
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


def _compute_layer_stats(correct_rows: List[Dict[str, Any]], wrong_rows: List[Dict[str, Any]], bucket_def: Optional[str] = None) -> Dict[str, Any]:
    # find first row with per_layer
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
    # determine num_layers by ans_to_ans_prefix length
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

    # layer-wise stats
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
        d_pref.append(_cohen_d(c_i_pref, w_i_pref))
        auc_pref.append(_auc(c_i_pref, w_i_pref))
        d_q.append(_cohen_d(c_i_q, w_i_q))
        auc_q.append(_auc(c_i_q, w_i_q))

    # buckets
    buckets = _parse_bucket_def(bucket_def or "early:0-6,mid:7-18,late:19-31", L)
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
            "ans_to_ans_prefix_auc": _auc(c_pref_bucket, w_pref_bucket),
            "ans_to_ans_prefix_d": _cohen_d(c_pref_bucket, w_pref_bucket),
            "ans_to_question_auc": _auc(c_q_bucket, w_q_bucket),
            "ans_to_question_d": _cohen_d(c_q_bucket, w_q_bucket),
            "layers": [a, b],
        }

    # identify peak layers for interpretability
    def _topk(vals: List[float], k: int = 5) -> List[int]:
        idx = list(range(len(vals)))
        idx.sort(key=lambda i: (float('-inf') if _math.isnan(vals[i]) else vals[i]), reverse=True)
        return idx[:k]

    top_auc_pref = _topk(auc_pref)
    top_d_pref = _topk(d_pref)

    return {
        "num_layers": L,
        "layer_indices": layer_indices,
        "ans_to_ans_prefix": {"auc": auc_pref, "cohen_d": d_pref, "top_auc_layers": top_auc_pref, "top_d_layers": top_d_pref},
        "ans_to_question": {"auc": auc_q, "cohen_d": d_q},
        "buckets": bucket_stats,
        "counts": {"correct": len(C_pref), "wrong": len(W_pref)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute attention metrics directly from converted JSONL (problem/pred)")
    parser.add_argument("--correct_converted", type=str, required=True, help="Path to *_correct_converted.jsonl")
    parser.add_argument("--wrong_converted", type=str, required=True, help="Path to *_wrong_converted.jsonl")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answer_prefix_tokens", type=int, default=32)
    parser.add_argument("--per_layer_trajectories", action="store_true", help="导出每层指标轨迹到 JSONL")
    parser.add_argument("--compute_layer_stats", action="store_true", help="计算每层 AUC / Cohen's d 以及早/中/晚层桶统计")
    parser.add_argument("--bucket_def", type=str, default="early:0-6,mid:7-18,late:19-31", help="层桶定义，例如 'early:0-6,mid:7-18,late:19-31'")
    parser.add_argument("--reuse_per_layer_from", type=str, default=None, help="复用已有的 per_layer_{correct,wrong}.jsonl 直接计算统计（跳过模型前向）")
    parser.add_argument("--prefix_lengths", type=str, default=None, help="可选：用逗号分隔的多K列表，例如 '8,16,32,64'；逐K复算并输出 layer_stats_K.json")
    parser.add_argument("--per_head", action="store_true", help="可选：输出每层每个head的指标（正确/错误分组），用于一致性分析")
    parser.add_argument(
        "--use_probe_prefix_len_for_ans_prefix",
        action="store_true",
        help="当可用时，使用 MLP 探针估计的 echo 前缀 token 数作为 answer→answer-prefix 的窗口长度；否则回退到 --answer_prefix_tokens",
    )
    parser.add_argument("--use_mlp_echo_removal", action="store_true")
    parser.add_argument("--embedding_model_path", type=str, default=None)
    parser.add_argument("--initial_threshold", type=float, default=0.6)
    parser.add_argument("--drop_threshold", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def _read_jsonl_rows(p: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        except Exception as e:
            raise RuntimeError(f"Failed to read {p}: {e}")
        return rows

    if args.reuse_per_layer_from:
        # 复用已有 per_layer_* 文件，直接做层级统计，跳过模型加载与前向
        correct_pl = os.path.join(args.reuse_per_layer_from, "per_layer_correct.jsonl")
        wrong_pl = os.path.join(args.reuse_per_layer_from, "per_layer_wrong.jsonl")
        if not os.path.exists(correct_pl) or not os.path.exists(wrong_pl):
            raise FileNotFoundError("--reuse_per_layer_from 目录下未找到 per_layer_correct.jsonl / per_layer_wrong.jsonl")
        correct_rows = _read_jsonl_rows(correct_pl)
        wrong_rows = _read_jsonl_rows(wrong_pl)
    else:
        tokenizer, model, _ = _load_model_and_tokenizer(args.model)
        embedder = _load_embedder(args.embedding_model_path, model.device) if args.use_mlp_echo_removal else None

        correct_rows = _process_file(
            args.correct_converted,
            tokenizer,
            model,
            args.answer_prefix_tokens,
            embedder,
            args.initial_threshold,
            args.drop_threshold,
            file_label="correct",
            use_removed_as_prefix=args.use_probe_prefix_len_for_ans_prefix,
            want_per_layer=args.per_layer_trajectories or args.compute_layer_stats,
        )
        wrong_rows = _process_file(
            args.wrong_converted,
            tokenizer,
            model,
            args.answer_prefix_tokens,
            embedder,
            args.initial_threshold,
            args.drop_threshold,
            file_label="wrong",
            use_removed_as_prefix=args.use_probe_prefix_len_for_ans_prefix,
            want_per_layer=args.per_layer_trajectories or args.compute_layer_stats,
        )

    # 仅当不是复用模式时，才写 attention_metrics 原始输出
    if not args.reuse_per_layer_from:
        _write_jsonl(os.path.join(args.output_dir, "attention_metrics_correct.jsonl"), correct_rows)
        _write_jsonl(os.path.join(args.output_dir, "attention_metrics_wrong.jsonl"), wrong_rows)

    if args.per_layer_trajectories and not args.reuse_per_layer_from:
        # 从 raw 再跑一次以输出 per-layer；避免重复前向的话可在 _process_file 内加开关，这里简化：读取已写 rows 的最小字段不够
        def _export_per_layer(in_path: str, label: str) -> None:
            out_rows: List[Dict[str, Any]] = []
            # 统计总行
            try:
                with open(in_path, "r", encoding="utf-8") as fcnt:
                    total_lines = sum(1 for _ in fcnt)
            except Exception:
                total_lines = None
            with open(in_path, "r", encoding="utf-8") as f:
                it = tqdm(f, total=total_lines, desc=f"Per-layer {label}", unit="lines")
                for line in it:
                    if not line.strip():
                        continue
                    s = json.loads(line)
                    try:
                        q = s.get("problem", "")
                        preds = s.get("pred", [])
                        if not isinstance(preds, list) or not preds:
                            continue
                        a_raw = _truncate_at_next_question(preds[-1] if isinstance(preds[-1], str) else str(preds[-1]))
                        # 可选：用探针估计的前缀长度参与窗口选择
                        removed_tokens_est_pl: Optional[int] = None
                        if embedder is not None and args.use_probe_prefix_len_for_ans_prefix:
                            a_clean, marker = _remove_echo_with_mlp(q, a_raw, embedder, args.initial_threshold, args.drop_threshold)
                            if marker == "__ESTIMATE_REMOVED__":
                                est = _estimate_removed_tokens_via_suffix(tokenizer, q, a_raw, a_clean)
                                removed_tokens_est_pl = est
                        metrics = _compute_attention_for_pair(
                            model,
                            tokenizer,
                            q,
                            a_raw,
                            args.answer_prefix_tokens,
                            removed_tokens_est_pl,
                            return_per_layer=True,
                            use_removed_as_prefix=args.use_probe_prefix_len_for_ans_prefix,
                        )
                        out_rows.append({
                            "idx": s.get("idx"),
                            "is_correct": s.get("is_correct"),
                            "per_layer": metrics.get("per_layer", {}),
                        })
                    except Exception as e:
                        out_rows.append({
                            "idx": s.get("idx"),
                            "error": f"{type(e).__name__}: {e}",
                        })
            _write_jsonl(os.path.join(args.output_dir, f"per_layer_{label}.jsonl"), out_rows)

        _export_per_layer(args.correct_converted, "correct")
        _export_per_layer(args.wrong_converted, "wrong")

    if args.compute_layer_stats:
        layer_stats = _compute_layer_stats(correct_rows, wrong_rows, args.bucket_def)
        with open(os.path.join(args.output_dir, "layer_stats.json"), "w", encoding="utf-8") as f:
            json.dump(layer_stats, f, ensure_ascii=False, indent=2)
        # 便捷打印：强调中层（默认为 7-18）
        mid = _parse_bucket_def(args.bucket_def, layer_stats.get("num_layers", 0)).get("mid")
        if mid is not None:
            a, b = mid
            print(f"中层层段 mid={a}-{b} AUC(d) for ans→ans-prefix: "
                  f"{layer_stats['buckets']['mid']['ans_to_ans_prefix_auc']:.4f} "
                  f"({layer_stats['buckets']['mid']['ans_to_ans_prefix_d']:.4f})")

    # 前缀长度多取值对比：在复用模式下仅做统计重算；否则需要重前向（此处仅支持复用per_layer）
    if args.prefix_lengths and args.reuse_per_layer_from:
        ks = [int(x) for x in args.prefix_lengths.split(',') if x.strip().isdigit()]
        # 复用已加载的 correct_rows / wrong_rows 的 per_layer 向量做K改变并不准确（因窗口随K变更需要重新取mass）
        # 为保证正确性，这里读取原始pred并逐K重前向计算per_layer（代价较大）；若不可用，则给出提示。
        base_dir = args.reuse_per_layer_from
        # 尝试读取原始converted文件路径（从命令入参）并逐K重跑前向
        if not os.path.exists(args.correct_converted) or not os.path.exists(args.wrong_converted):
            print("prefix_lengths 需要可用的 --correct_converted/--wrong_converted 以便逐K重算；当前路径不可用，跳过K对比。")
        else:
            try:
                tokenizer, model, _ = _load_model_and_tokenizer(args.model)
            except Exception as e:
                print(f"无法加载模型以执行K对比：{e}")
                ks = []
            for K in ks:
                print(f"按K={K} 重算per-layer并输出 layer_stats_K={K}.json …")
                c_rows = _process_file(
                    args.correct_converted, tokenizer, model, K, None, args.initial_threshold, args.drop_threshold,
                    file_label=f"correct-K{K}", use_removed_as_prefix=False, want_per_layer=True,
                )
                w_rows = _process_file(
                    args.wrong_converted, tokenizer, model, K, None, args.initial_threshold, args.drop_threshold,
                    file_label=f"wrong-K{K}", use_removed_as_prefix=False, want_per_layer=True,
                )
                stats_k = _compute_layer_stats(c_rows, w_rows, args.bucket_def)
                with open(os.path.join(args.output_dir, f"layer_stats_K{K}.json"), "w", encoding="utf-8") as f:
                    json.dump(stats_k, f, ensure_ascii=False, indent=2)

    # 每头分析：输出每层每个head的质量（正确与错误分组分别写文件）
    if args.per_head and not args.reuse_per_layer_from:
        try:
            tokenizer, model, _ = _load_model_and_tokenizer(args.model)
        except Exception as e:
            print(f"无法加载模型以执行per-head分析：{e}")
            model = None
        if model is not None:
            embedder_ph = _load_embedder(args.embedding_model_path, model.device) if args.use_mlp_echo_removal else None
            def _export_per_head(in_path: str, label: str) -> None:
                out_rows: List[Dict[str, Any]] = []
                try:
                    with open(in_path, "r", encoding="utf-8") as fcnt:
                        total_lines = sum(1 for _ in fcnt)
                except Exception:
                    total_lines = None
                with open(in_path, "r", encoding="utf-8") as f:
                    it = tqdm(f, total=total_lines, desc=f"Per-head {label}", unit="lines")
                    for line in it:
                        if not line.strip():
                            continue
                        s = json.loads(line)
                        try:
                            q = s.get("problem", "")
                            preds = s.get("pred", [])
                            if not isinstance(preds, list) or not preds:
                                continue
                            a_raw = _truncate_at_next_question(preds[-1] if isinstance(preds[-1], str) else str(preds[-1]))
                            removed_tokens_est_pl: Optional[int] = None
                            if embedder_ph is not None and args.use_probe_prefix_len_for_ans_prefix:
                                a_clean, marker = _remove_echo_with_mlp(q, a_raw, embedder_ph, args.initial_threshold, args.drop_threshold)
                                if marker == "__ESTIMATE_REMOVED__":
                                    try:
                                        est = _estimate_removed_tokens_via_suffix(tokenizer, q, a_raw, a_clean)
                                        removed_tokens_est_pl = est
                                    except Exception:
                                        removed_tokens_est_pl = None
                            m = _compute_per_head_for_pair(
                                model, tokenizer, q, a_raw, args.answer_prefix_tokens, removed_tokens_est_pl
                            )
                            out_rows.append({
                                "idx": s.get("idx"),
                                "is_correct": s.get("is_correct"),
                                **m,
                            })
                        except Exception as e:
                            out_rows.append({
                                "idx": s.get("idx"),
                                "error": f"{type(e).__name__}: {e}",
                            })
                _write_jsonl(os.path.join(args.output_dir, f"per_head_{label}.jsonl"), out_rows)

            _export_per_head(args.correct_converted, "correct")
            _export_per_head(args.wrong_converted, "wrong")

    correct_sum = _aggregate([r for r in correct_rows if "error" not in r])
    wrong_sum = _aggregate([r for r in wrong_rows if "error" not in r])
    summary = {
        "correct": correct_sum,
        "wrong": wrong_sum,
        "difference": {k: correct_sum.get(k, float("nan")) - wrong_sum.get(k, float("nan")) for k in correct_sum.keys()},
    }
    with open(os.path.join(args.output_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("注意力核验报告已保存:")
    print("  - attention_metrics_correct.jsonl")
    print("  - attention_metrics_wrong.jsonl")
    print("  - attention_summary.json")
    if args.per_layer_trajectories:
        print("  - per_layer_correct.jsonl")
        print("  - per_layer_wrong.jsonl")


if __name__ == "__main__":
    main()



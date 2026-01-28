#!/usr/bin/env python3
"""
Attention Case Visualization
============================

用途：对单个样本进行注意力可视化（case study）。
- 支持从已转换的 JSONL（包含 problem/pred）中按 idx 选样本，或直接传入 question/answer 文本
- 指定层与 head，对选定 query token 的注意力分配进行着色可视化（HTML）
- 可选生成整张 head 的 (seq_len × seq_len) 热力图（PNG）

依赖：
- 复用 `src.evaluation.attention_from_converted` 中的加载与收集注意力逻辑

示例：
    # Example usage in a shell script:
    # python /path/to/this/script.py \
    #   --model /path/to/your/model \
    #   --logp_results_json /path/to/your/logp_results.json \
    #   ...
"""

from __future__ import annotations

import argparse
import html
import json
import os
from typing import List, Optional, Tuple

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# 复用已有评估脚本中的工具
try:
    from src.evaluation.attention_from_converted import (
        _load_model_and_tokenizer,
        _collect_attentions,
        _token_ids,
        build_prompt,
        _truncate_at_next_question,
        _mean_heads,
        _mean_layers,
        _gather_mass,
        _find_question_token_span_in_prompt,
    )
except Exception:
    # 降级：在极端情况下直接在本脚本内实现最小功能
    def _load_model_and_tokenizer(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:  # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device).eval()
        return tokenizer, model, device

    def _collect_attentions(
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:  # type: ignore
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

    def _token_ids(text: str, tokenizer: AutoTokenizer, add_special_tokens: bool = True) -> List[int]:  # type: ignore
        enc = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors=None)
        if isinstance(enc, dict):
            return enc["input_ids"]  # type: ignore[index]
        return enc.input_ids  # type: ignore[attr-defined]

    def build_prompt(question: str) -> str:  # type: ignore
        return (
            "You are an expert at solving math problems. Please think step by step.\n"
            f"Question: {question}\n"
            "Answer: <think>"
        )

    def _truncate_at_next_question(text: str) -> str:  # type: ignore
        if not isinstance(text, str):
            return text
        markers = ["\n\nQuestion:", "\nQuestion:"]
        cut = len(text)
        for m in markers:
            pos = text.find(m)
            if pos != -1:
                cut = min(cut, pos)
        return text[:cut]

    def _mean_heads(attn: torch.Tensor) -> torch.Tensor:  # type: ignore
        return attn.mean(dim=0)

    def _mean_layers(attns: List[torch.Tensor]) -> torch.Tensor:  # type: ignore
        return torch.stack([_mean_heads(a) for a in attns], dim=0).mean(dim=0)

    def _gather_mass(attn_mat: torch.Tensor, src_idx: List[int], dst_idx: List[int]) -> float:  # type: ignore
        if not src_idx or not dst_idx:
            return float("nan")
        src = torch.tensor(src_idx, dtype=torch.long, device=attn_mat.device)
        dst = torch.tensor(dst_idx, dtype=torch.long, device=attn_mat.device)
        mass_per_src = attn_mat.index_select(0, src).index_select(1, dst).sum(dim=1)
        return float(mass_per_src.mean().item())

    def _find_question_token_span_in_prompt(tokenizer: AutoTokenizer, prompt_with_question: str, raw_question: str):  # type: ignore
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
            from dataclasses import dataclass
            @dataclass
            class Span:  # type: ignore
                start: int
                end: int
            return Span(start=token_start, end=token_end)
        except Exception:
            return None


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _load_case_from_converted(jsonl_path: str, idx: int) -> Tuple[str, str]:
    """从转换后的JSONL中按 idx 读取 (question, answer) 对。
    JSONL 每行为 {"idx": int, "problem": str, "pred": [str,...]}
    取最后一个 pred 作为 answer 文本。
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if int(s.get("idx", -1)) != int(idx):
                continue
            q = s.get("problem", "")
            preds = s.get("pred", [])
            if not isinstance(preds, list) or not preds:
                raise ValueError(f"idx={idx} 的样本缺少 pred 列表")
            a = preds[-1] if isinstance(preds[-1], str) else str(preds[-1])
            return q, a
    raise ValueError(f"未在 {jsonl_path} 中找到 idx={idx} 的样本")


def _tokens_from_ids(tokenizer: AutoTokenizer, ids: List[int]) -> List[str]:
    toks = tokenizer.convert_ids_to_tokens(ids)
    return toks


def _pretty_token_labels(tokenizer: AutoTokenizer, tokens: List[str]) -> List[str]:
    """将 tokenizer 子词标记转为更人类可读的标签。"""
    labels: List[str] = []
    for t in tokens:
        label = ""
        try:
            # 尝试用 tokenizer 的 detokenize 单个token的方式得到更自然的片段
            label = tokenizer.convert_tokens_to_string([t])  # type: ignore[attr-defined]
        except Exception:
            label = t
        if not label:
            label = t
        # 常见子词前缀清理
        label = label.replace("▁", " ")  # SentencePiece 下划线
        label = label.replace("Ġ", " ")  # GPT BPE 空格标识
        label = label.replace("\n", "⏎")
        label = label.replace("\t", "⇥")
        # 避免空字符串标签
        label = label if label.strip() else (t.replace("▁", " ") or t)
        labels.append(label)
    return labels


def _normalize_weights_for_color(weights: torch.Tensor) -> torch.Tensor:
    """按最大值归一化以提升可视对比度，不改变相对排序。"""
    w = weights.detach().float().clamp(min=0)
    maxv = float(w.max().item()) if w.numel() > 0 else 0.0
    if maxv <= 0:
        return torch.zeros_like(w)
    return (w / maxv).clamp(0, 1)


def _build_html(tokens: List[str], attn_weights: List[float], query_index: int, prompt_len: int, title: str) -> str:
    """构建HTML，将每个token按注意力强度着色，query token加边框标记。"""
    esc_tokens = [html.escape(t) for t in tokens]

    # 将比值映射为 rgba alpha；加一个轻微的非线性增强
    import math

    def alpha_of(x: float) -> float:
        # 轻微gamma矫正
        return max(0.0, min(0.95, x ** 0.8))

    spans = []
    for i, (tok, w) in enumerate(zip(esc_tokens, attn_weights)):
        a = alpha_of(float(w))
        border = "2px solid #000" if i == query_index else "1px solid #ddd"
        bg = f"rgba(255, 87, 34, {a:.3f})"  # 深橙色映射
        role = "prompt" if i < prompt_len else "answer"
        spans.append(
            f"<span class='tok {role}' title='idx={i}, w={float(w):.4f}' "
            f"style='background:{bg}; border:{border}; padding:2px 3px; margin:1px; border-radius:3px; display:inline-block;'>"
            f"{tok}</span>"
        )

    legend = (
        "<div style='margin-top:8px; font-size:13px; color:#444'>"
        "深色=该 head 对所选 query token 的注意力更强；黑框=所选 query token。"
        "</div>"
    )

    head = (
        "<style>body{font-family:Arial,Helvetica,sans-serif;} .tok.prompt{opacity:.9} .tok.answer{opacity:1}</style>"
        f"<h3 style='margin:6px 0'>{html.escape(title)}</h3>"
    )

    prompt_marker = (
        "<div style='margin:6px 0; color:#666'>灰色区域约为提示(prompt)部分，白色为答案(answer)部分。</div>"
    )

    return head + prompt_marker + "<div style='line-height:2.0'>" + "".join(spans) + "</div>" + legend


def _save_heatmap_png(attn_head: torch.Tensor, tokens: List[str], out_png: str, max_ticks: int = 128) -> None:
    """保存 head 的完整 (seq_len × seq_len) 热力图。
    为避免过大，token刻度最多显示 max_ticks 个（等距抽样）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except Exception:
        print("[WARN] 未安装 matplotlib/seaborn，跳过PNG热力图生成。")
        return

    data = attn_head.detach().cpu().float().numpy()
    seq_len = data.shape[0]

    plt.figure(figsize=(min(18, 0.12 * seq_len + 4), min(14, 0.12 * seq_len + 3)))
    sns.heatmap(data, cmap="YlOrRd", cbar=True)
    plt.title("Attention Heatmap (head) — rows=query, cols=key")

    # 抽样设置坐标轴刻度
    if seq_len > 0:
        import numpy as np
        tick_idx = np.linspace(0, seq_len - 1, num=min(max_ticks, seq_len), dtype=int)
        tick_labels = [tokens[i] if i < len(tokens) else str(i) for i in tick_idx]
        plt.xticks(tick_idx + 0.5, tick_labels, rotation=90, fontsize=7)
        plt.yticks(tick_idx + 0.5, tick_labels, rotation=0, fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] 保存热力图：{out_png}")


def _maybe_truncate_input_ids(
    input_ids: torch.Tensor,
    prompt_len: int,
    max_seq_len: Optional[int],
) -> Tuple[torch.Tensor, int]:
    """当序列过长时，保留全部 prompt token，并尽量保留答案后段，限制总长不超过 max_seq_len。
    返回 (裁剪后的 input_ids, 新的 prompt_len)。
    """
    if max_seq_len is None:
        return input_ids, prompt_len
    T = int(input_ids.shape[1])
    if T <= max_seq_len:
        return input_ids, prompt_len
    drop = T - max_seq_len
    ans_len = T - prompt_len
    if drop <= ans_len and prompt_len < max_seq_len:
        # 丢弃答案开头的 drop 个 token，保留全部 prompt
        kept_ans = input_ids[:, prompt_len + drop :]
        new_ids = torch.cat([input_ids[:, :prompt_len], kept_ans], dim=1)
        return new_ids, prompt_len
    else:
        # 连同一部分 prompt 也不得不裁掉：保留末尾 max_seq_len 个 token
        new_ids = input_ids[:, -max_seq_len:]
        # 计算新的 prompt_len（可能小于原来的）
        drop_excess = drop - max(ans_len, 0)
        new_prompt_len = max(0, prompt_len - max(drop_excess, 0))
        return new_ids, new_prompt_len


def visualize_case(
    model_path: str,
    question: str,
    answer: str,
    layer: int,
    head: int,
    query_full_index: Optional[int],
    query_answer_index: Optional[int],
    output_dir: str,
    basename_hint: str = "case",
    save_png: bool = False,
    max_seq_len: Optional[int] = 1024,
    heatmap_region: str = "answer",
) -> Tuple[str, Optional[str]]:
    """执行一次可视化，返回 (html_path, png_path)。"""
    _ensure_dir(output_dir)

    tokenizer, model, _ = _load_model_and_tokenizer(model_path)
    prompt = build_prompt(question)
    full_text = prompt + answer

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    p_len = len(_token_ids(prompt, tokenizer, add_special_tokens=True))
    # 长序列截断（优先保留 prompt + 答案末尾）
    input_ids, p_len = _maybe_truncate_input_ids(input_ids, p_len, max_seq_len)
    if attn_mask is not None and input_ids.shape[1] != attn_mask.shape[1]:
        attn_mask = torch.ones_like(input_ids)
    T = int(input_ids.shape[1])

    attns = _collect_attentions(model, input_ids, attn_mask)
    num_layers = len(attns)
    if layer < 0:
        layer = num_layers - 1
    if not (0 <= layer < num_layers):
        raise ValueError(f"layer 越界：{layer}，共有 {num_layers} 层")

    attn_layer = attns[layer]  # (num_heads, T, T)
    num_heads = int(attn_layer.shape[0])
    if not (0 <= head < num_heads):
        raise ValueError(f"head 越界：{head}，该层共有 {num_heads} 个head")

    if query_full_index is not None:
        q_idx = int(query_full_index)
    elif query_answer_index is not None:
        qi = int(query_answer_index)
        if qi < 0:
            q_idx = max(p_len, T + qi)  # 允许负数从末尾回数
        else:
            q_idx = p_len + qi
    else:
        q_idx = T - 1  # 默认最后一个token

    if not (0 <= q_idx < T):
        raise ValueError(f"query 索引越界：{q_idx}，序列长度 {T}")

    attn_head = attn_layer[head]  # (T, T)
    weights = attn_head[q_idx]    # (T,)
    weights_norm = _normalize_weights_for_color(weights)

    ids = input_ids[0].detach().cpu().tolist()
    tokens_raw = _tokens_from_ids(tokenizer, ids)
    tokens_disp = _pretty_token_labels(tokenizer, tokens_raw)

    # 计算与表格指标一致的聚合量（仅供标题提示）：
    # last-layer: ans->question, ans->answer-prefix; all-layers mean 同理
    ans_indices = list(range(p_len, T))
    q_span = _find_question_token_span_in_prompt(tokenizer, prompt, question)
    q_indices = list(range(q_span.start, q_span.end)) if q_span else []
    k = min(32, max(0, T - p_len))
    ans_pref = list(range(p_len, p_len + k)) if k > 0 else []

    last = _mean_heads(attns[-1])
    mean_all = _mean_layers(attns)
    m_last_q = _gather_mass(last, ans_indices, q_indices) if q_indices else float("nan")
    m_last_pref = _gather_mass(last, ans_indices, ans_pref) if ans_pref else float("nan")
    m_mean_q = _gather_mass(mean_all, ans_indices, q_indices) if q_indices else float("nan")
    m_mean_pref = _gather_mass(mean_all, ans_indices, ans_pref) if ans_pref else float("nan")

    title = (
        f"L{layer} H{head} | q={q_idx} | last(ans→Q)={m_last_q:.3f}, last(ans→A₀:{k})={m_last_pref:.3f}, "
        f"mean(ans→Q)={m_mean_q:.3f}, mean(ans→A₀:{k})={m_mean_pref:.3f}"
    )
    html_str = _build_html(tokens_disp, weights_norm.detach().cpu().tolist(), q_idx, p_len, title)

    html_path = os.path.join(output_dir, f"{basename_hint}_L{layer}_H{head}_Q{q_idx}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"[OK] 保存HTML：{html_path}")

    png_path: Optional[str] = None
    if save_png:
        # 根据区域选择要绘制的子矩阵
        if heatmap_region == "answer":
            sub = attn_head[p_len:, p_len:] if p_len < T else attn_head
            sub_tokens = tokens_disp[p_len:]
        else:
            sub = attn_head
            sub_tokens = tokens_disp
        png_path = os.path.join(output_dir, f"{basename_hint}_L{layer}_H{head}.png")
        _save_heatmap_png(sub, sub_tokens, png_path)

    return html_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention case visualization (HTML/PNG)")
    parser.add_argument("--model", required=True, help="Model path")

    # 选择输入来源（二选一）
    parser.add_argument("--converted_jsonl", type=str, default=None, help="路径：*_converted.jsonl")
    parser.add_argument("--logp_results_json", required=True, help="Path to ..._logp_results.json")
    parser.add_argument("--idx", type=int, default=None, help="样本 idx（配合 --converted_jsonl/--logp_results_json 使用）")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--answer", type=str, default=None)

    parser.add_argument("--layer", type=int, default=-1, help="层索引（-1 表示最后一层）")
    parser.add_argument("--head", type=int, default=0, help="head 索引")

    # 选择 query token
    parser.add_argument("--query_full_index", type=int, default=None, help="序列内绝对索引（含prompt）")
    parser.add_argument("--query_answer_index", type=int, default=None, help="答案段内相对索引（可为负，-1为末尾）")

    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--basename_hint", type=str, default="case")
    parser.add_argument("--save_png", action="store_true", help="同时保存 head 的整矩阵热力图")
    parser.add_argument("--prefer_original", action="store_true", help="优先使用 original_prediction（默认优先 trimmed_prediction）")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度（超过将裁剪，仅保证保留全部prompt和答案的末尾）")
    parser.add_argument("--heatmap_region", type=str, default="answer", choices=["full", "answer"], help="热图绘制区域：全序列或仅答案区域")

    args = parser.parse_args()

    # 校验输入来源（优先顺序：converted_jsonl > logp_results_json > question/answer）
    if args.converted_jsonl is not None:
        if args.idx is None:
            raise SystemExit("提供 --converted_jsonl 时必须给出 --idx")
        question, answer = _load_case_from_converted(args.converted_jsonl, args.idx)
        basename = args.basename_hint or f"idx{args.idx}"
    elif args.logp_results_json is not None:
        if args.idx is None:
            raise SystemExit("提供 --logp_results_json 时必须给出 --idx")
        with open(args.logp_results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        details = data.get("details", [])
        # 允许 doc_id != 连续位置；按 idx 匹配
        match = None
        for d in details:
            if int(d.get("idx", -1)) == int(args.idx):
                match = d
                break
        if match is None and 0 <= args.idx < len(details):
            # 回退：按位置索引
            match = details[args.idx]
        if match is None:
            raise SystemExit(f"在 {args.logp_results_json} 中找不到 idx={args.idx} 对应条目")
        question = match.get("question", "")
        # 默认优先使用 trimmed_prediction，避免掺入后续多轮 "Question:" 的串接
        if args.prefer_original:
            answer = match.get("original_prediction") or match.get("trimmed_prediction") or ""
        else:
            answer = match.get("trimmed_prediction") or match.get("original_prediction") or ""
        # 进一步在文本层面切断后续无关的题干（与 metrics 侧一致）
        answer = _truncate_at_next_question(answer)
        basename = args.basename_hint or f"idx{args.idx}"
    else:
        if args.question is None or args.answer is None:
            raise SystemExit("必须提供 --converted_jsonl/--logp_results_json + --idx，或直接提供 --question 与 --answer")
        question, answer = args.question, args.answer
        basename = args.basename_hint or "case"

    visualize_case(
        model_path=args.model,
        question=question,
        answer=answer,
        layer=args.layer,
        head=args.head,
        query_full_index=args.query_full_index,
        query_answer_index=args.query_answer_index,
        output_dir=args.output_dir,
        basename_hint=basename,
        save_png=bool(args.save_png),
        max_seq_len=args.max_seq_len,
        heatmap_region=args.heatmap_region,
    )


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Two-Stage HFLM for lm-evaluation-harness
========================================

实现一个自定义的 HFLM 子类，用于在评测时进行“两阶段生成”对照实验：

- baseline4096: 一次性生成 4096 tokens
- two_stage_echo: 先生成 2048，再插入 "Look back again: {question}\nSo now I know that "（可配置），再生成 2048
- two_stage_continue: 先生成 2048，再生成 2048

注意：本类覆写了 generate_until，直接在 HF 后端完成分阶段生成；
      返回文本在末尾会根据 until 列表做截断，保持与 harness 行为一致。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch

try:
    # 优先从安装的 lm-evaluation-harness 导入
    from lm_eval.models.huggingface import HFLM  # type: ignore
except Exception as e:  # pragma: no cover - 仅在缺失依赖时提示
    raise ImportError(
        "lm-evaluation-harness 未安装或版本不兼容，请先安装: pip install lm-eval[vllm]"
    ) from e


class TwoStageHFLM(HFLM):
    """自定义 HFLM：支持两阶段生成与中途注入提示。

    参数
    ----
    mode: str
        取值 {"baseline2048", "baseline4096", "two_stage_echo", "two_stage_continue"}
    first_stage_tokens: int
        第一阶段生成 token 数（默认 2048）。
    second_stage_tokens: int
        第二阶段生成 token 数（默认 2048）。
    injection_template: str
        echo 模式下的注入模版，包含 {question} 占位符。
    continue_template: str
        非 echo 模式下的中性续写提示。
    """

    def __init__(
        self,
        *,
        mode: str = "baseline2048",
        first_stage_tokens: int = 2048,
        second_stage_tokens: int = 2048,
        injection_template: str = "Look back again: {question}\nSo now I know that ",
        continue_template: str = "",
        **hf_kwargs: Any,
    ) -> None:
        super().__init__(**hf_kwargs)
        assert mode in {"baseline2048", "baseline4096", "two_stage_echo", "two_stage_continue"}, (
            f"Unsupported mode: {mode}"
        )
        self.mode = mode
        self.first_stage_tokens = int(first_stage_tokens)
        self.second_stage_tokens = int(second_stage_tokens)
        self.injection_template = str(injection_template)
        self.continue_template = str(continue_template)

    # ---- Core helpers ----
    def _extract_question_from_prompt(self, prompt: str) -> str:
        """尽力从 harness prompt 中提取 Question 文本（针对 GSM8K 常见模板）。

        若未匹配到，则回退为原始 prompt（保持健壮性）。
        """
        try:
            # 1) 取最后一个 Question: ... \n Answer: 的 pair
            pairs = list(re.finditer(r"\bQuestion:\s*(.*?)\r?\n\s*Answer:\s*", prompt, re.DOTALL))
            if pairs:
                return pairs[-1].group(1).strip()
            # 2) 若没有显式 Answer:，取最后一个 Question: 到文本末尾（切掉后续再次出现的 Question:）
            qs = list(re.finditer(r"\bQuestion:\s*(.*)$", prompt, re.DOTALL))
            if qs:
                seg = qs[-1].group(1)
                nxt = seg.find("\n\nQuestion:")
                if nxt != -1:
                    seg = seg[:nxt]
                return seg.strip()
        except Exception:
            pass
        # 回退为原始 prompt，避免注入失败
        return prompt

    def _decode(self, ids: torch.LongTensor) -> str:
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    def _apply_until(self, text: str, until: List[str]) -> str:
        if not until:
            return text
        trimmed = text
        for term in until:
            if not term:
                continue
            if term in trimmed:
                trimmed = trimmed.split(term)[0]
        return trimmed

    def _sanitize_until(self, until: List[str]) -> List[str]:
        """移除会误杀 CoT 的 stop 词（例如 Question:）。

        说明：很多 CoT 会自然出现 "Question:"，若作为 stop，将导致在尚未给出
        最终答案前被截断，进而使 strict-match 提取失败。
        """
        if not until:
            return until
        sanitized: List[str] = []
        for u in until:
            if u is None:
                continue
            s = str(u).strip().lower()
            if s == "question:":
                # 跳过该 stop 词
                continue
            sanitized.append(u)
        return sanitized

    def _generate_once(self, prompt: str, max_new_tokens: int, gen_kwargs: Dict[str, Any]) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        else:
            attn_mask = attn_mask.to(self.device)

        # 允许透传常见采样参数；过滤掉 harness 的控制字段
        safe_kwargs: Dict[str, Any] = {}
        pass_keys = [
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "num_beams",
        ]
        for k in pass_keys:
            if k in gen_kwargs:
                safe_kwargs[k] = gen_kwargs[k]

        # 若未显式启用采样，则移除采样相关参数以避免告警
        if not bool(safe_kwargs.get("do_sample", False)):
            safe_kwargs.pop("temperature", None)
            safe_kwargs.pop("top_p", None)
            safe_kwargs.pop("top_k", None)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=self.tokenizer.eos_token_id,
                **safe_kwargs,
            )
        # 仅返回补全部分（去掉前缀 prompt），与 harness 期望一致
        gen_only = out_ids[0, input_ids.shape[1]:]
        if gen_only.numel() == 0:
            return ""
        return self.tokenizer.decode(gen_only, skip_special_tokens=True)

    # ---- Override HFLM API ----
    def generate_until(self, requests, disable_tqdm: bool = False):  # type: ignore[override]
        from tqdm import tqdm  # lazy import

        results: List[str] = []
        pbar = None if disable_tqdm else tqdm(total=len(requests), desc="Evaluating", unit="samples")

        for inst in requests:
            prompt, gen_kwargs = inst.args
            max_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            until = self._sanitize_until(gen_kwargs.get("until", []) or [])

            # 解析 question（两阶段 echo 模式需要）
            question = self._extract_question_from_prompt(prompt)

            if self.mode == "baseline2048":
                text = self._generate_once(prompt, max_new_tokens=min(2048, max_toks), gen_kwargs=gen_kwargs)
                text = self._apply_until(text, until)
                results.append(text)
            elif self.mode == "baseline4096":
                text = self._generate_once(prompt, max_new_tokens=min(4096, max_toks), gen_kwargs=gen_kwargs)
                text = self._apply_until(text, until)
                results.append(text)
            elif self.mode in {"two_stage_echo", "two_stage_continue"}:
                # Stage 1：生成补全部分
                stage1_gen = self._generate_once(prompt, max_new_tokens=min(self.first_stage_tokens, max_toks), gen_kwargs=gen_kwargs)
                # 用于第二阶段的完整上下文
                stage1_full = prompt + stage1_gen

                # 注入提示
                if self.mode == "two_stage_echo":
                    injection = self.injection_template.format(question=question)
                else:
                    injection = self.continue_template

                connector = "\n" if (not stage1_full.endswith("\n") and not injection.startswith("\n")) else ""
                prompt2 = stage1_full + connector + injection

                # Stage 2（继续使用 gen_kwargs 的采样参数），返回仅补全部分
                stage2_gen = self._generate_once(
                    prompt2,
                    max_new_tokens=min(self.second_stage_tokens, max_toks),
                    gen_kwargs=gen_kwargs,
                )
                final_text = self._apply_until(stage2_gen, until)
                results.append(final_text)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return results


__all__ = ["TwoStageHFLM"]



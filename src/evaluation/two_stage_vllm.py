#!/usr/bin/env python3
"""
Two-Stage VLLM for lm-evaluation-harness
=======================================

使用 vLLM 引擎实现两阶段生成的自定义 LM。该实现直接对接
lm-evaluation-harness 的 Python API（simple_evaluate），覆写
generate_until 以支持：

- baseline2048 / baseline4096
- two_stage_echo（2048 + 注入 + 2048）
- two_stage_continue（2048 + 续写 + 2048）

注意：当前实现为简洁起见未实现 loglikelihood 系列方法，仅
用于基于生成任务（如 gsm8k）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("未找到 vLLM，请先安装：pip install vllm") from e

try:
    from lm_eval.api.model import LM  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("lm-evaluation-harness 未安装或版本不兼容，请安装 lm-eval") from e


class TwoStageVLLM(LM):
    def __init__(
        self,
        *,
        pretrained: str,
        mode: str = "baseline2048",
        tensor_parallel_size: int = 1,
        dtype: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = True,
        first_stage_tokens: int = 2048,
        second_stage_tokens: int = 2048,
        injection_template: str = "Look back again: {question}\nSo now I know that ",
        continue_template: str = "Continue.",
        **unused: Any,
    ) -> None:
        super().__init__()
        assert mode in {"baseline2048", "baseline4096", "two_stage_echo", "two_stage_continue"}
        self.mode = mode
        self.first_stage_tokens = int(first_stage_tokens)
        self.second_stage_tokens = int(second_stage_tokens)
        self.injection_template = str(injection_template)
        self.continue_template = str(continue_template)

        init_kwargs: Dict[str, Any] = {
            "model": pretrained,
            "tensor_parallel_size": int(tensor_parallel_size),
            "trust_remote_code": trust_remote_code,
        }
        if dtype:
            init_kwargs["dtype"] = dtype
        if revision:
            init_kwargs["revision"] = revision

        self.engine = LLM(**init_kwargs)
        # 默认最大生成 tokens（可被每条请求覆盖）
        self.max_gen_toks = 256

    # ----- helpers -----
    def _extract_question_from_prompt(self, prompt: str) -> str:
        try:
            pairs = list(re.finditer(r"\bQuestion:\s*(.*?)\r?\n\s*Answer:\s*", prompt, re.DOTALL))
            if pairs:
                return pairs[-1].group(1).strip()
            qs = list(re.finditer(r"\bQuestion:\s*(.*)$", prompt, re.DOTALL))
            if qs:
                seg = qs[-1].group(1)
                nxt = seg.find("\n\nQuestion:")
                if nxt != -1:
                    seg = seg[:nxt]
                return seg.strip()
        except Exception:
            pass
        return prompt

    def _apply_until(self, text: str, until: List[str]) -> str:
        if not until:
            return text
        out = text
        for term in until:
            if term and term in out:
                out = out.split(term)[0]
        return out

    def _sanitize_until(self, until: List[str] | None) -> List[str] | None:
        if not until:
            return until
        sanitized: List[str] = []
        for u in until:
            if u is None:
                continue
            s = str(u).strip().lower()
            if s == "question:":
                continue
            sanitized.append(u)
        return sanitized

    def _sampling_params_from(self, gen_kwargs: Dict[str, Any], max_tokens: int, stops: List[str] | None = None) -> SamplingParams:
        do_sample = bool(gen_kwargs.get("do_sample", False))
        params: Dict[str, Any] = {"max_tokens": int(max_tokens)}
        if stops:
            params["stop"] = [s for s in stops if s]
        if do_sample:
            if "temperature" in gen_kwargs:
                params["temperature"] = gen_kwargs["temperature"]
            if "top_p" in gen_kwargs:
                params["top_p"] = gen_kwargs["top_p"]
            if "top_k" in gen_kwargs:
                params["top_k"] = gen_kwargs["top_k"]
        return SamplingParams(**params)

    def _generate_once(self, prompt: str, gen_kwargs: Dict[str, Any], max_tokens: int, stops: List[str] | None) -> str:
        sp = self._sampling_params_from(gen_kwargs, max_tokens=max_tokens, stops=stops)
        outs = self.engine.generate([prompt], sampling_params=sp, use_tqdm=False)
        # vLLM 返回仅生成内容，不包含 prompt
        return outs[0].outputs[0].text if outs and outs[0].outputs else ""

    # ----- LM interface -----
    def generate_until(self, requests, disable_tqdm: bool = False):  # type: ignore[override]
        from tqdm import tqdm  # lazy import

        results: List[str] = []
        pbar = None if disable_tqdm else tqdm(total=len(requests), desc="Evaluating", unit="samples")

        for inst in requests:
            prompt, gen_kwargs = inst.args
            max_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            until = self._sanitize_until(gen_kwargs.get("until", []) or [])

            if self.mode == "baseline2048":
                gen_text = self._generate_once(prompt, gen_kwargs, max_tokens=min(2048, max_toks), stops=until)
                results.append(self._apply_until(gen_text, until))
            elif self.mode == "baseline4096":
                gen_text = self._generate_once(prompt, gen_kwargs, max_tokens=min(4096, max_toks), stops=until)
                results.append(self._apply_until(gen_text, until))
            else:
                # Stage 1：生成补全部分，并拼回完整上下文
                stage1_gen = self._generate_once(prompt, gen_kwargs, max_tokens=min(self.first_stage_tokens, max_toks), stops=None)
                stage1_full = prompt + stage1_gen

                # 注入提示
                if self.mode == "two_stage_echo":
                    question = self._extract_question_from_prompt(prompt)
                    injection = self.injection_template.format(question=question)
                else:
                    injection = self.continue_template

                connector = "\n" if (not stage1_full.endswith("\n") and not injection.startswith("\n")) else ""
                prompt2 = stage1_full + connector + injection

                stage2_gen = self._generate_once(prompt2, gen_kwargs, max_tokens=min(self.second_stage_tokens, max_toks), stops=until)
                # 返回第二阶段新增内容（与 HFLM 行为保持一致：只返回补全部分）
                results.append(self._apply_until(stage2_gen, until))

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return results

    # ---- Required abstract methods (not used in GSM8K generation path) ----
    def greedy_until(self, requests, disable_tqdm: bool = False):  # type: ignore[override]
        # Most harness generations call greedy_until; delegate to generate_until
        return self.generate_until(requests, disable_tqdm=disable_tqdm)

    def loglikelihood(self, requests):  # type: ignore[override]
        # Not needed for this experiment; implement to satisfy abstract base
        raise NotImplementedError("TwoStageVLLM.loglikelihood is not implemented for this experiment.")

    def loglikelihood_rolling(self, requests):  # type: ignore[override]
        # Not needed for this experiment; implement to satisfy abstract base
        raise NotImplementedError("TwoStageVLLM.loglikelihood_rolling is not implemented for this experiment.")


__all__ = ["TwoStageVLLM"]



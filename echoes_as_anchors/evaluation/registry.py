"""Register Echoes adapters lazily with lm-evaluation-harness."""

from __future__ import annotations


def register_models() -> tuple[str, str]:
    try:
        from lm_eval.api.registry import model_registry
    except ImportError as exc:
        raise ImportError(
            'Install an evaluation extra first: pip install "echoes-as-anchors[hf]"'
        ) from exc

    package = __package__ or "echoes_as_anchors.evaluation"
    targets = {
        "echoes-hf": f"{package}.two_stage_hflm:TwoStageHFLM",
        "echoes-vllm": f"{package}.two_stage_vllm:TwoStageVLLM",
    }
    for alias, target in targets.items():
        if alias not in model_registry:
            model_registry.register(alias, target=target)
    return tuple(targets)

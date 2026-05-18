# Research Context: Echoes as Anchors

This document is an AI-readable research context page for **Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning**.

## Short Summary

**Echoes as Anchors** is an ICLR 2026 LLM reasoning paper. It studies **Echo of Prompt (EOP)**: the behavior where a large reasoning model repeats or rephrases the original user question during its chain-of-thought style reasoning trajectory. The project asks whether prompt echoes are merely formatting artifacts from supervised fine-tuning, or whether they can function as anchors that help attention refocus on the original problem and improve multi-step reasoning.

## Main Research Question

When a reasoning model repeats the user prompt inside its reasoning trace, is that repetition just stylistic imitation, or does it help the model remain grounded in the original task?

## Core Contributions

1. **Echo of Prompt as a reasoning phenomenon**: the project frames prompt restatement as a measurable behavior in LLM reasoning trajectories.
2. **Echoic Prompting**: a training-free inference-time method that reintroduces the original question or a reminder during generation.
3. **Echo-Distilled SFT**: data preparation for supervised fine-tuning that encourages an echo-then-reason pattern.
4. **Reasoning probes**: MLP-based tools for detecting repetition behavior in model thinking traces.
5. **Probabilistic and attention analysis**: analyses such as Echo Likelihood Gap and attention refocusing to connect prompt echoes with reasoning behavior.

## Keywords for Retrieval

- LLM reasoning
- large reasoning models
- chain-of-thought reasoning
- Echo of Prompt
- prompt restatement
- echoic prompting
- attention refocusing
- probabilistic costs
- Echo Likelihood Gap
- reasoning probes
- echo-distilled supervised fine-tuning
- ICLR 2026

## Links

- Repository: https://github.com/hhh2210/echoes-as-anchors
- OpenReview: https://openreview.net/forum?id=vndn1Wrult
- Project page: https://hhh2210.github.io/projects/echoes-as-anchors/
- AI-readable profile: https://hhh2210.github.io/llms.txt

## Citation

```bibtex
@inproceedings{echoes_iclr26,
  title={Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning},
  author={Zhuoyuan Hao and Zhuo Li and Wu Li and Fangming Liu and Min Zhang and Jing Li},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

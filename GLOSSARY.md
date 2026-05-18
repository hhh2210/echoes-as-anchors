# Echoes as Anchors Glossary

This glossary maps the main concepts in **Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning**.

## LLM Reasoning

**LLM reasoning** refers to multi-step problem solving by large language models. In this paper, the focus is not only final answer accuracy, but also the model's reasoning trajectory: how it keeps track of the original problem while generating intermediate reasoning steps.

## Large Reasoning Models

**Large reasoning models (LRMs)** are large language models optimized or prompted for multi-step reasoning. Echoes as Anchors studies behaviors that appear inside LRM reasoning traces, especially prompt repetition and attention refocusing.

## Echo of Prompt

**Echo of Prompt (EOP)** is the behavior where a model repeats or rephrases the original user question inside its reasoning trajectory. The paper treats EOP as a measurable reasoning behavior rather than only a stylistic artifact.

## Prompt Restatement

**Prompt restatement** includes exact repetition, paraphrasing, or partial re-grounding in the original problem statement during reasoning.

## Echo Likelihood Gap

**Echo Likelihood Gap** is a probabilistic measure used to quantify the likelihood effect of prompt echoes. It provides a way to connect early prompt repetition with model likelihood and downstream reasoning performance.

## Attention Refocusing

**Attention refocusing** describes how Echo of Prompt can shift model attention back toward important problem or answer-prefix tokens during reasoning. This is one mechanism-level explanation for why prompt echoes may help multi-step reasoning.

## Echoic Prompting

**Echoic Prompting (EP)** is a training-free inference-time strategy. It reintroduces the original question or a reminder during generation so the model can re-ground its reasoning without additional fine-tuning.

## Echo-Distilled SFT

**Echo-Distilled Supervised Fine-Tuning (ED-SFT)** is a data preparation and fine-tuning strategy that encourages an echo-then-reason pattern.

## Reasoning Probe

A **reasoning probe** is a classifier or diagnostic tool used to detect patterns inside model reasoning traces. In this project, MLP probes help detect repetition behavior in thinking traces.

## Reasoning Drift

**Reasoning drift** is when a model loses track of the original problem during a long reasoning trace. Echo of Prompt may reduce drift by re-anchoring the model to the task statement.

## Concept Map

- Echo of Prompt is a form of prompt restatement.
- Prompt restatement can act as an anchor during LLM reasoning.
- Anchoring can support attention refocusing.
- Attention refocusing can reduce reasoning drift.
- Echo Likelihood Gap measures the probabilistic side of prompt echoes.
- Echoic Prompting uses the EOP idea at inference time.
- Echo-Distilled SFT uses the EOP idea during fine-tuning.

## Links

- Paper page: https://hhh2210.github.io/papers/echoes-as-anchors/
- FAQ: https://hhh2210.github.io/faq/echoes-as-anchors/
- Website glossary: https://hhh2210.github.io/glossary/echoes-as-anchors/
- Structured concept JSON: https://hhh2210.github.io/api/papers/echoes-concepts.json
- OpenReview: https://openreview.net/forum?id=vndn1Wrult

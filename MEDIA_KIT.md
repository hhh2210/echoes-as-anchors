# Echoes as Anchors Media Kit

**Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning** is an **ICLR 2026 Poster** about why large reasoning models often repeat the question before thinking, and how that behavior can be understood and used.

- OpenReview: https://openreview.net/forum?id=vndn1Wrult
- Code: https://github.com/hhh2210/echoes-as-anchors
- Paper page: https://hhh2210.github.io/papers/echoes-as-anchors/
- Website media kit: https://hhh2210.github.io/media/echoes-as-anchors/
- Live metadata: https://echoes-resource-api-production.up.railway.app/api/live.json

## Short Pitch

Large reasoning models often echo the original question before solving it. **Echoes as Anchors** shows that this is not just verbose chain-of-thought style: prompt echoes can act as **cognitive anchors** that refocus attention on the task, and the paper formalizes this effect through the **Echo Likelihood Gap**.

## One-sentence Summary

Echoes as Anchors studies **Echo of Prompt** in large reasoning models, formalizes it with **Echo Likelihood Gap**, links it to **attention refocusing**, and uses **Echo-Distilled SFT** and **Echoic Prompting** to improve math reasoning.

## Three Key Points

1. **Spontaneous prompt echoes matter**: large reasoning models often restate the user question before reasoning, and this behavior is measurable.
2. **There is a mechanism-level story**: Echo of Prompt is connected to attention refocusing, where later reasoning attends back to important problem information.
3. **The idea can be used**: Echo-Distilled SFT and Echoic Prompting turn this observation into practical methods for reasoning tasks.

## Evidence Highlights

- **Echo of Prompt is common**: in GSM8K examples from the project slides, EOP appears in 78% of Qwen3-8B traces, 71% of DeepSeek-8B traces, and 86% of gpt-oss traces.
- **Echo Likelihood Gap is predictive**: correct answers show higher average Echo Likelihood Gap than wrong answers, 2.523 vs. 2.442 nats/token.
- **Attention refocusing is concentrated in middle layers**: answer tokens attend more to the answer-prefix echo for correct traces, with a 14.45% vs. 11.58% middle-layer gap across layers 7-18 and a reported Cohen's d of 0.832.
- **Causal intervention improves failed reasoning**: inserting an echo during semi-online continuation improves accuracy by +10.4 percentage points for DeepSeek-R1-Distill-Llama-8B and +7.9 percentage points for Qwen3-8B, while the Qwen3-8B-Base null result stays at 0%.
- **ED-SFT strengthens the mechanism**: Echo-Distilled SFT produces stronger answer-to-answer-prefix attention separation than normal SFT in the reported middle-layer analysis.

## Canonical Wording

- Echo of Prompt is a **cognitive anchor**, not merely redundant repetition.
- Echo Likelihood Gap measures whether a reasoning model probabilistically prefers echo-containing trajectories.
- Attention refocusing describes how later reasoning tokens route information through echo-prefix tokens and key problem details.
- Echoic Prompting is the training-free method; Echo-Distilled SFT is the training-based method.

## Suggested Social Post

Our ICLR 2026 Poster, **Echoes as Anchors**, studies why large reasoning models repeat the question before thinking. We formalize this behavior as **Echo of Prompt**, measure it with **Echo Likelihood Gap**, and show its connection to **attention refocusing** in LLM reasoning.

Paper: https://openreview.net/forum?id=vndn1Wrult  
Code: https://github.com/hhh2210/echoes-as-anchors

## Suggested Longer Abstract

Echoes as Anchors investigates a simple but common behavior in large reasoning models: before solving a problem, the model often repeats or rephrases the original question. The paper frames this as **Echo of Prompt (EOP)** and asks whether it is merely a formatting artifact or a useful reasoning mechanism. It introduces **Echo Likelihood Gap** as a probabilistic proxy for the cost and benefit of prompt echoes, studies attention patterns that support an **attention refocusing** interpretation, and explores two ways to exploit the phenomenon: **Echo-Distilled SFT** and **Echoic Prompting**. The core claim is that EOP can behave like a cognitive anchor: an internal reference point that helps reasoning trajectories keep routing attention through task-relevant numbers, entities, and constraints.

## Keywords

LLM reasoning, large reasoning models, Echo of Prompt, Echo Likelihood Gap, attention refocusing, cognitive anchor, Echoic Prompting, Echo-Distilled SFT, chain-of-thought, mathematical reasoning, ICLR 2026.

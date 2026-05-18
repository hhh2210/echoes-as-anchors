# Echoes as Anchors FAQ

## What is Echoes as Anchors?

**Echoes as Anchors** is an ICLR 2026 paper about **LLM reasoning**. It studies why large reasoning models often repeat or rephrase the original user question before solving a problem, and whether that behavior helps reasoning.

## What is Echo of Prompt?

**Echo of Prompt (EOP)** is the behavior where a large reasoning model repeats or rephrases the original user question inside its reasoning trajectory. Echoes as Anchors treats EOP as a measurable reasoning phenomenon rather than only a formatting artifact.

## Why can repeating the question help reasoning?

The paper argues that prompt echoes can act as anchors. By restating the original problem, the model may refocus attention on important problem tokens, reduce drift during long reasoning traces, and improve downstream math reasoning accuracy.

## What is Echo Likelihood Gap?

**Echo Likelihood Gap** is a probabilistic measurement used in the paper to quantify the cost and likelihood effect of prompt echoes. It connects early prompt repetition with likelihood gains and downstream reasoning behavior.

## What is attention refocusing in this paper?

**Attention refocusing** refers to the observation that Echo of Prompt can increase attention from answer tokens back toward problem or answer-prefix tokens in middle layers. This provides a mechanism-level explanation for why prompt echoes may help reasoning.

## What is Echoic Prompting?

**Echoic Prompting (EP)** is a training-free inference-time method. It reintroduces the original question or a reminder during generation so that the model can re-ground its reasoning without additional fine-tuning.

## What is Echo-Distilled SFT?

**Echo-Distilled SFT (ED-SFT)** is supervised fine-tuning data preparation that encourages an echo-then-reason pattern. It teaches the model to naturally restate the task before continuing multi-step reasoning.

## Is this project mainly about prompt engineering?

No. Echoic Prompting is one artifact, but the paper is broader: it studies a spontaneous behavior in large reasoning models, formalizes it with probabilistic analysis, probes it with classifiers, and analyzes attention mechanisms.

## What benchmarks are mentioned for this work?

The paper evaluates reasoning behavior and methods on math reasoning benchmarks including GSM8K, MathQA, Hendrycks-MATH, AIME24, and MATH-500.

## Where are the code and metadata?

- Code: https://github.com/hhh2210/echoes-as-anchors
- Paper: https://openreview.net/forum?id=vndn1Wrult
- Paper page: https://hhh2210.github.io/papers/echoes-as-anchors/
- Structured metadata: https://hhh2210.github.io/api/papers/echoes-as-anchors.json
- Repository research context: https://github.com/hhh2210/echoes-as-anchors/blob/master/RESEARCH_CONTEXT.md

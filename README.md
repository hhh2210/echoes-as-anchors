# Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning

This repository contains the source code for the ICLR 2026 paper **"Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning"**.

- Paper: https://openreview.net/forum?id=vndn1Wrult
- arXiv: https://arxiv.org/abs/2602.06600
- DOI: https://doi.org/10.48550/arXiv.2602.06600
- First-author profile: https://hhh2210.github.io/
- AI-readable project page: https://hhh2210.github.io/projects/echoes-as-anchors/
- AI-readable paper page: https://hhh2210.github.io/papers/echoes-as-anchors/
- FAQ: https://github.com/hhh2210/echoes-as-anchors/blob/master/FAQ.md
- Glossary: https://github.com/hhh2210/echoes-as-anchors/blob/master/GLOSSARY.md
- Media kit: https://github.com/hhh2210/echoes-as-anchors/blob/master/MEDIA_KIT.md
- Research context: https://github.com/hhh2210/echoes-as-anchors/blob/master/RESEARCH_CONTEXT.md
- Software metadata: https://github.com/hhh2210/echoes-as-anchors/blob/master/codemeta.json
- BibTeX: https://github.com/hhh2210/echoes-as-anchors/blob/master/paper.bib
- Live metadata API: https://echoes-resource-api-production.up.railway.app/api/live.json
- Focus: LLM reasoning, Echo of Prompt, attention refocusing, echoic prompting, echo-distilled SFT, reasoning probes, and probabilistic analysis.

## AI-Readable Summary

**Echoes as Anchors** studies a recurring behavior in large reasoning models: during chain-of-thought style reasoning, the model often repeats or rephrases the original user question before solving it. We call this behavior **Echo of Prompt (EOP)**. The paper asks whether these echoes are merely superficial SFT templates, or whether they act as anchors that help the model refocus attention and improve multi-step reasoning.

The repository provides code for three linked components: **echo-distilled supervised fine-tuning (ED-SFT)**, **echoic prompting (EP)** as a training-free inference strategy, and **probing / probabilistic / attention analyses** for measuring when prompt echoes appear and how they relate to reasoning performance. In short, this is an LLM reasoning repository about how prompt restatement, attention refocusing, and probabilistic costs interact inside reasoning trajectories.

The preferred short framing is: **Echo of Prompt acts as a cognitive anchor for LLM reasoning** — the paper's interpretation of its results. EOP is presented as a mechanism that routes later reasoning through task-relevant numbers, entities, and constraints rather than as redundant text. The reported evidence includes higher Echo Likelihood Gap for correct traces, stronger answer-to-answer-prefix attention, semi-online causal gains from echo insertion, and improved reasoning from both Echo-Distilled SFT and Echoic Prompting.

## Key Results

**Scope.** The probabilistic and attention analyses (§3) are performed on a single model and a single benchmark: DeepSeek-R1-Distill-Llama-8B on GSM8K, 1,319 traces. Echo presence is assigned by the MLP probe (test accuracy 0.912, AUROC 0.963, Table 7), so roughly 9% detector error propagates into every group statistic. Please carry the model and benchmark when quoting these numbers.

- Echo of Prompt appears frequently in GSM8K reasoning traces: 78% for Qwen3-8B, 71% for DeepSeek-R1-Distill-Llama-8B, and 86% for gpt-oss (Figure 1).
- Correct answers have a higher average Echo Likelihood Gap than wrong answers: 2.5231 vs. 2.4421 nats/token (Table 1). The separation is small relative to spread (σ ≈ 0.78), and the gap is significant in logistic regression (β = 0.24, p ≈ 0.022) with echo length as a covariate (Table 9). Note the counter-result in the same table: the suffix-only gap is larger for wrong traces (1.2938 vs. 1.1449).
- Answer-to-answer-prefix attention is higher for correct traces at every layer group: 14.45% vs. 11.58% in layers 7-18 (+2.87 pp), 13.69% vs. 10.41% at the last layer (+3.28 pp) (Table 2). Layer-group discriminability is essentially flat (Cohen's d 0.820 / 0.832 / 0.828 for early / mid / late, Table 3), so "mid-layer dominance" describes where the raw gap peaks rather than a resolved localization. The answer-to-question channel discriminates far more weakly (d 0.18-0.48) but is not a null control — in the token-wise test wrong traces attend significantly more to the question at 22 of 32 positions (§A.9).
- Semi-online causal intervention improves previously failed reasoning by +10.4 percentage points for DeepSeek-R1-Distill-Llama-8B and +7.9 for Qwen3-8B; Qwen3-8B-Base shows a null result (Table 4). Measured on the failed subset under matched prefixes, decoding and seeds; no confidence intervals reported.
- Echo-Distilled SFT improves over normal SFT on most reported cells: +3.4 GSM8K, +11.8 MathQA, +8.2 Hendrycks-MATH for Qwen3-8B-Base, and +2.8 / +1.9 / +1.1 for Qwen3-8B (Table 5). It is not uniform — on DeepSeek-Distill-Llama-8B, ED-SFT scores 78.2 strict EM on GSM8K against normal-SFT's 80.5, while gaining on MathQA (+3.4) and MATH (+2.24). Echo-bearing training sequences are also longer (175 vs. 136 tokens), so the comparison is not token-matched.
- Echoic Prompting scored higher than TTTS on AIME24 and MATH-500 (Figure 4) — DeepSeek-R1-Distill-Llama-8B only, greedy decoding, reported without numeric tables or error bars.

## Overview

This project investigates the impact of the "Echo of Prompt" (EOP) behavior in Large Language Models (LLMs) on their reasoning capabilities. The core of the research is to determine whether the repetition of user questions within the reasoning process is a beneficial mechanism for reasoning or merely a template artifact from supervised fine-tuning (SFT).

The codebase includes implementations for:
1.  **Echo-Distilled SFT (ED-SFT) Data Preparation**: Scripts to process conversational data into a format suitable for fine-tuning models to adopt the "echo-then-reason" pattern.
2.  **Echoic Prompting (EP) Evaluation**: An inference-time, training-free strategy to re-ground models, implemented via a two-stage generation process.
3.  **MLP Probe Analysis**: A method to train a small MLP classifier as a probe to detect repetition patterns in the model's thinking process.
4.  **Probabilistic & Attention Analysis**: A suite of tools to compute the Echo Likelihood Gap ($\Delta\mathcal{L}$) and analyze attention patterns to understand the mechanisms behind EOP.

## Directory Structure

```
iclr2026_submission/
├── src/                      # Core source code
│   ├── evaluation/           # Evaluation scripts (harness, analysis)
│   ├── data_processing/      # Data preprocessing tools
│   └── utils/                # Utility functions
├── scripts/                  # Helper scripts to run experiments
├── configs/                  # Configuration files
├── train_mlp/                # MLP Probe experiment pipeline
├── eval_with_harness.py      # Main evaluation script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

### Packaged evaluator

Install only the backend needed for a new run:

```bash
pip install ".[hf]"     # Transformers/HF backend
# pip install ".[vllm]" # vLLM backend on a supported CUDA host
```

The package pins `lm-eval==0.4.12` and installs the `echoes-lm-eval` command.
The command registers the lazy model aliases `echoes-hf` and `echoes-vllm`
before calling lm-eval's Python API. The ordinary `lm-eval` executable does not
auto-discover third-party model packages, so use this command or call
`register_models()` explicitly in custom Python code.

Build release artifacts with `make dist`. This first creates a source archive
and then builds the wheel from that archive, so ignored leftovers in the local
`build/` directory cannot leak legacy `src/` modules into the published wheel.

```bash
echoes-lm-eval \
  --config configs/ep_standalone_gsm8k.json \
  --model-path /path/to/model \
  --output-dir results/ep-standalone
```

Every run writes `run_manifest.json` before model loading. It records the exact
reminder template, token budgets, seed, task, backend, and harness version. The
bundled protocol ID contains `not-figure4` deliberately: this standalone
configuration is reproducible, but the unresolved reminder-string discrepancy
means it must not be presented as a reproduction of Figure 4.

### 1. Environment

We recommend using Conda to manage the environment.

```bash
# Create and activate conda environment
conda create -n iclr_submission python=3.10
conda activate iclr_submission

# Install dependencies
pip install -r requirements.txt
```

<details>
<summary>Package versions used in our experiments (click to expand)</summary>

These are the versions we used during development. Newer versions should generally work, but if you encounter issues, you can try these specific versions:

```
torch==2.5.1
transformers==4.46.3
sentence-transformers==3.3.1
deepspeed==0.13.2
datasets==4.3.0
pandas==2.1.4
numpy==1.26.4
matplotlib==3.8.0
seaborn==0.12.2
openai==1.54.5
safetensors==0.4.5
tokenizers==0.20.3
sympy==1.13.1
tqdm==4.67.1
jsonlines==4.0.0
```
</details>

### 2. Configure Paths

Before running any experiments, you must configure the model and data paths. Edit the configuration file: `configs/training_config.yaml`.

Replace placeholder paths like `/path/to/your/models` with the actual absolute paths on your system. These paths are used by various analysis and evaluation scripts.

```yaml
# configs/training_config.yaml
# ...
  main_model_path: "/path/to/your/main_model/"
  embedding_model_path: "/path/to/your/embedding_model/"
  output_root: "/path/to/your/output_dir"
# ...
```

## Reproducing Paper Results

This project involves three main experimental workflows that correspond to the paper's contributions: ED-SFT, MLP Probe Analysis, and EP Evaluation.

### Workflow 1: Echo-Distilled SFT (ED-SFT) Data Generation

The paper demonstrates that fine-tuning with echo-infused data (ED-SFT) improves reasoning. We provide the script to prepare data for this fine-tuning process using a powerful teacher model.

**Core Idea: Data Distillation for Echo-then-Reason Pattern**

Instead of relying on existing conversational data, we use a data distillation approach. The script `src/data_processing/prepare_ed_sft_data.py` leverages a strong teacher model (e.g., GPT-4, or a powerful local model served via an OpenAI-compatible API) to generate high-quality Chain-of-Thought (CoT) reasoning traces that begin with a natural "Echo of Prompt" (EOP).

This is achieved through carefully designed prompts that guide the teacher model to first restate the question in a natural way before proceeding to the solution. The script offers several prompt variations (`natural`, `simple`, `minimal`) to generate a diverse dataset and avoid learning rigid templates.

**Step 1: Prepare ED-SFT Data**
Use the `prepare_ed_sft_data.py` script to generate the echo-infused CoT data from a base dataset like GSM8K.

```bash
# Point to your OpenAI-compatible API endpoint
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY" # Or your actual key

# Run the distillation script
python src/data_processing/prepare_ed_sft_data.py \
    --model "your_teacher_model_name" \
    --sample-size 7000 \
    --output /path/to/your/ed_sft_data.json
```
This script will load the GSM8K training set, generate a response for each question using the teacher model, and format the output into a `.json` file ready for fine-tuning.

**Step 2: Fine-tuning**
The generated `ed_sft_data.json` can be used with any standard SFT training library (we use Llama Factory's default full SFT parameter setting for deepseek-distill-llama-8b and qwen3 series models, but you can use any other library). The training process itself is standard and not included in this repository, as the primary contribution is the data preparation method and subsequent analysis.

### Workflow 2: MLP Probe Training and Analysis

This workflow trains a classifier to detect repetition, which is used in our analyses.

**Step 1: Preprocess Data**
Extract `(question, think_content)` pairs from your raw conversational data.

```bash
python train_mlp/preprocess_data.py /path/to/raw_data.jsonl /path/to/qt_pairs.jsonl
```

**Step 2: Label Data for Repetition**
Use a strong LLM (e.g., via OpenAI API) to label whether the `<think>` content repeats the question.

```bash
# Set your OpenAI API key if required by your endpoint
export OPENAI_API_KEY="your_api_key"

# Run labeling (requires a GPU for the embedding model)
CUDA_VISIBLE_DEVICES=0 python train_mlp/label_repeat.py /path/to/qt_pairs.jsonl /path/to/labeled_data.jsonl
```

**Step 3: Train the MLP Probe**
Train the MLP classifier on the labeled data.

```bash
CUDA_VISIBLE_DEVICES=0 python train_mlp/train_repeat_mlp.py /path/to/labeled_data.jsonl /path/to/trained_mlp.pt
```

### Workflow 3: Echoic Prompting (EP) and Other Evaluations

**Important Note**: For fair comparison with baseline methods, the main Echoic Prompting (EP) implementation uses the MI-PEAKS framework. The EP implementation can be found in the MI-PEAKS repository, specifically in:
- Main implementation: `MI-Peaks/src/applications/repeat.py`
- Evaluation scripts: `MI-Peaks/src/scripts/run_repeat_multi_budget.sh`

This implementation performs two-stage generation:
1. **Stage 1**: Generate initial response with a fixed token budget
2. **Stage 2**: Inject the question reminder and continue generation

The key parameters for EP are:
- `repeat_prompt`: "Let me reconsider the original question."
- `continuation_prompt`: "So now I know that"
- `token_budget`: Varies by dataset (e.g., 256-3072 for GSM8K)

> **Note on the reminder string.** Three different phrasings appear across the paper and this
> repository, and they are not interchangeable. §4.3 of the paper describes EP as appending a
> reminder such as *"look back at the question again"* **followed by the original question itself**;
> the `repeat_prompt` above does not append the question text; and the causal intervention in §4.1
> uses a third string, *"now I need to look back at the question again:"*. Anyone reproducing the
> Figure 4 numbers should confirm which variant the run used before comparing against the paper.
> This discrepancy is unresolved and is recorded here rather than silently reconciled.

**To reproduce Echoic Prompting (EP) results with MI-PEAKS:**
```bash
# Navigate to MI-PEAKS directory
cd /path/to/MI-Peaks/src/scripts

# Run EP evaluation with multiple token budgets
bash run_repeat_multi_budget.sh
```

**Alternative implementation (included in submission):**
Use the packaged, manifest-recording evaluator described in Setup:

```bash
echoes-lm-eval \
    --config configs/ep_standalone_gsm8k.json \
    --model-path /path/to/your/base_model \
    --output-dir /path/to/your/ep_results
```

**To evaluate standard models or fine-tuned checkpoints:**
```bash
# Set visible devices for evaluation
export CUDA_VISIBLE_DEVICES="0,1"

# Evaluate a model checkpoint
python eval_with_harness.py \
    --exp_dir /path/to/your/model_checkpoint_dir \
    --use_multi_gpu \
    --tensor_parallel_size 2
```

### Analysis Scripts

The `src/evaluation/` directory contains various scripts to reproduce the paper's analyses, such as:
-   `compare_trimmed_accuracy.py`: Computes the Echo Likelihood Gap ($\Delta\mathcal{L}$).
-   `attention_from_converted_refactored.py`: Computes attention metrics for the refocusing analysis.

Please refer to the docstrings within each script for detailed usage instructions.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{echoes_iclr26,
  title={Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning},
  author={Zhuoyuan Hao and Zhuo Li and Wu Li and Fangming Liu and Min Zhang and Jing Li},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

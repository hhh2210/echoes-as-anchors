# Echoic Prompting (EP) Implementation Details

## Overview

The Echoic Prompting (EP) method is a training-free inference strategy that re-grounds the model on the original question mid-generation to improve reasoning performance. This document describes the implementation used in our ICLR 2026 submission.

## Main Implementation (MI-PEAKS Framework)

For fair comparison with baseline methods and to ensure consistent evaluation across different token budgets, we implement EP within the MI-PEAKS framework, which provides standardized evaluation infrastructure for multi-step reasoning.

### Location

The primary EP implementation is located in the MI-PEAKS repository:
- **Main script**: `MI-Peaks/src/applications/repeat.py`
- **Evaluation script**: `MI-Peaks/src/scripts/run_repeat_multi_budget.sh`

### How It Works

The EP method performs two-stage generation:

1. **Stage 1 (Initial Generation)**:
   - Generate response with a fixed token budget (`max_tokens_per_call`)
   - This allows the model to start reasoning naturally

2. **Stage 2 (Echo Injection + Continuation)**:
   - Inject a reminder of the original question
   - Continue generation with remaining token budget
   - Format: `[Initial_Response] [Repeat_Prompt] [Original_Question] [Continuation_Prompt] [Continued_Response]`

### Key Parameters

```python
# Prompt components
repeat_prompt = "Let me reconsider the original question."
continuation_prompt = "So now I know that"

# Token budgets (vary by dataset)
# GSM8K: [256, 512, 1024, 1536, 2048, 3072]
# MATH: [512, 772, 1024, 2048, 3072, 4096, 5120]
# AIME: [1024, 2048, 3072, 4096, 6144, 8192, 12288]
```

### Implementation Details

The core EP logic in `repeat.py`:

```python
def batch_repeat_generation(prompts, llm, tokenizer, args, stop_words):
    # Stage 1: Base generation
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens_per_call,
        temperature=0.0,
        ...
    )
    vllm_outputs = llm.generate(prompts, sampling_params)

    # Stage 2: Generation with repeat prompt
    repeat_prompts = []
    for q, a in zip(prompts, outputs):
        # Construct repeat prompt with question reminder
        concat_prompt = f'{q} {a} {args.repeat_prompt} {q} {args.continuation_prompt}'
        repeat_prompts.append(concat_prompt)

    # Continue generation with remaining budget
    remaining_budget = args.token_budget - args.max_tokens_per_call
    ...
```

## Running EP Experiments

### Prerequisites

1. Install MI-PEAKS dependencies:
```bash
pip install vllm transformers torch
```

2. Set up model paths in the script

### Evaluation

To run EP evaluation across multiple token budgets:

```bash
cd MI-Peaks/src/scripts
bash run_repeat_multi_budget.sh
```

This will:
- Evaluate EP across predefined token budgets
- Generate detailed results for each budget
- Create a summary report with accuracy metrics

### Customization

To modify EP parameters:

1. Edit `run_repeat_multi_budget.sh`:
```bash
# Modify prompts
repeat_prompt="Your custom repeat prompt"
continuation_prompt="Your custom continuation"

# Modify token budgets
token_budgets=(256 512 1024 2048)
```

2. Run with custom model:
```bash
model="/path/to/your/model"
dataset="gsm8k"  # or "math", "aime"
```

## Alternative Implementation

We also provide a standalone implementation in `src/evaluation/two_stage_eval.py` that doesn't require the MI-PEAKS framework. This is useful for quick tests or integration with other evaluation pipelines.

## Results Interpretation

The EP method shows improvements especially at intermediate token budgets where:
- There's enough initial generation to establish reasoning context
- The reminder helps refocus attention on key problem details
- There's sufficient remaining budget for correction/completion

Key metrics to examine:
- Accuracy across different token budgets
- Comparison with baseline (no repeat) at same total budget
- Analysis of where in generation the echo helps most

## Citation

When using the EP implementation, please cite both:
1. Our ICLR 2026 paper (Echoes as Anchors)
2. The MI-PEAKS framework (if using that implementation)
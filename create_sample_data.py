#!/usr/bin/env python3
"""
Sample data generator for ICLR 2026 submission.
Creates small sample datasets for testing the pipeline.
"""

import json
import random
from pathlib import Path

def create_gsm8k_samples(output_file="data/samples/gsm8k_sample.jsonl", n_samples=10):
    """Create sample GSM8K-style math problems."""

    samples = []

    # Sample math problems
    problems = [
        {
            "question": "John has 5 apples. He gives 2 to Mary. How many apples does John have left?",
            "answer": "3"
        },
        {
            "question": "A store sells pencils for $0.50 each. If Sarah buys 6 pencils, how much does she pay?",
            "answer": "3.00"
        },
        {
            "question": "Tom reads 20 pages per hour. How many pages can he read in 3 hours?",
            "answer": "60"
        },
        {
            "question": "A pizza is cut into 8 slices. If 3 people eat 2 slices each, how many slices are left?",
            "answer": "2"
        },
        {
            "question": "Lisa has 15 marbles. She loses 7. How many marbles does she have now?",
            "answer": "8"
        },
        {
            "question": "A car travels 60 miles per hour. How far does it travel in 2.5 hours?",
            "answer": "150"
        },
        {
            "question": "A box contains 24 chocolates. If eaten equally over 6 days, how many per day?",
            "answer": "4"
        },
        {
            "question": "Mark scored 85, 90, and 95 on three tests. What is his average score?",
            "answer": "90"
        },
        {
            "question": "A recipe needs 3 cups of flour for 12 cookies. How much flour for 20 cookies?",
            "answer": "5"
        },
        {
            "question": "If 5 workers can complete a job in 10 days, how long would it take 10 workers?",
            "answer": "5"
        }
    ]

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write samples
    with open(output_file, 'w') as f:
        for i, problem in enumerate(problems[:n_samples]):
            sample = {
                "id": f"sample_{i:04d}",
                "question": problem["question"],
                "answer": problem["answer"],
                "source": "gsm8k_sample"
            }
            f.write(json.dumps(sample) + '\n')

    print(f"Created {n_samples} GSM8K samples in {output_file}")

def create_conversation_samples(output_file="data/samples/conversation_sample.jsonl", n_samples=5):
    """Create sample conversation data with thinking traces."""

    conversations = []

    # Sample conversations with echo patterns
    templates = [
        {
            "question": "What is 15 + 27?",
            "think": "<think>Let me calculate 15 + 27. I need to add these two numbers together. 15 + 27 = 42.</think>",
            "answer": "42"
        },
        {
            "question": "If x + 5 = 12, what is x?",
            "think": "<think>So I need to find x where x + 5 = 12. To solve for x, I'll subtract 5 from both sides: x = 12 - 5 = 7.</think>",
            "answer": "x = 7"
        },
        {
            "question": "Calculate the area of a rectangle with length 8 and width 5.",
            "think": "<think>I need to find the area of a rectangle with length 8 and width 5. The formula for area is length × width. So area = 8 × 5 = 40.</think>",
            "answer": "40 square units"
        },
        {
            "question": "What is 20% of 150?",
            "think": "<think>To find 20% of 150, I need to multiply 150 by 0.20. So 150 × 0.20 = 30.</think>",
            "answer": "30"
        },
        {
            "question": "Solve: 3x - 7 = 11",
            "think": "<think>Looking at the equation 3x - 7 = 11. First, I'll add 7 to both sides: 3x = 18. Then divide by 3: x = 6.</think>",
            "answer": "x = 6"
        }
    ]

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write samples
    with open(output_file, 'w') as f:
        for i, conv in enumerate(templates[:n_samples]):
            sample = {
                "id": f"conv_{i:04d}",
                "messages": [
                    {"role": "user", "content": conv["question"]},
                    {"role": "assistant", "content": conv["think"] + " " + conv["answer"]}
                ]
            }
            f.write(json.dumps(sample) + '\n')

    print(f"Created {n_samples} conversation samples in {output_file}")

def create_mlp_training_samples(output_file="data/samples/mlp_training_sample.jsonl", n_samples=10):
    """Create sample data for MLP training (question, think_content, label)."""

    samples = []

    # Mix of echo and non-echo examples
    for i in range(n_samples):
        if i % 2 == 0:  # Echo example
            question = f"What is {random.randint(1,20)} times {random.randint(1,20)}?"
            think = f"Let me calculate what is asked in the problem. The question asks {question.lower()} I need to multiply these numbers."
            label = 1
        else:  # Non-echo example
            question = f"Calculate {random.randint(1,100)} plus {random.randint(1,100)}."
            think = f"To solve this, I'll use addition. The result is clearly a sum."
            label = 0

        sample = {
            "id": f"mlp_{i:04d}",
            "question": question,
            "think_content": think,
            "label": label
        }
        samples.append(sample)

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write samples
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Created {n_samples} MLP training samples in {output_file}")

if __name__ == "__main__":
    # Create all sample datasets
    create_gsm8k_samples()
    create_conversation_samples()
    create_mlp_training_samples()

    print("\nAll sample datasets created successfully!")
    print("Sample data is available in the data/samples/ directory")
#!/usr/bin/env python
import json
import argparse
import os
import gzip

def load_lines(path):
    """
    Helper to load lines from a .jsonl or .jsonl.gz file.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    """
    Splits a lm-eval-harness samples.jsonl file into correct and wrong
    subsets based on the 'exact_match' metric.
    """
    parser = argparse.ArgumentParser(
        description="Split lm-eval samples by 'exact_match' metric."
    )
    parser.add_argument(
        "--samples", 
        type=str, 
        required=True, 
        help="Path to the samples.jsonl file from lm-evaluation-harness."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the split files."
    )
    args = parser.parse_args()

    correct_samples = []
    wrong_samples = []

    print(f"Reading samples from: {args.samples}")
    for sample in load_lines(args.samples):
        # The 'exact_match' can be 1.0, 1, or True.
        em = sample.get("exact_match")
        if em == 1 or em == 1.0 or em is True:
            correct_samples.append(sample)
        else:
            wrong_samples.append(sample)

    base_name = os.path.splitext(os.path.basename(args.samples))[0]
    correct_file_path = os.path.join(args.output_dir, f"{base_name}_correct.jsonl")
    wrong_file_path = os.path.join(args.output_dir, f"{base_name}_wrong.jsonl")

    # Write correct samples
    with open(correct_file_path, "w", encoding="utf-8") as f:
        for sample in correct_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Write wrong samples
    with open(wrong_file_path, "w", encoding="utf-8") as f:
        for sample in wrong_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Splitting complete.")
    print(f"  - {len(correct_samples)} correct samples -> {correct_file_path}")
    print(f"  - {len(wrong_samples)} wrong samples   -> {wrong_file_path}")

if __name__ == "__main__":
    main()

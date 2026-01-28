#!/usr/bin/env python
"""
Convert lm-eval output format to logp_trim_experiment.py expected format.
"""
import json
import argparse

def convert_lm_eval_to_logp_format(input_file, output_file):
    """
    Convert from lm-eval format:
    {
        "doc": {"question": "...", "answer": "..."},
        "filtered_resps": ["..."],
        "exact_match": 1.0
    }
    
    To logp_trim_experiment format:
    {
        "problem": "...",
        "pred": ["..."],
        "is_correct": true,
        "idx": 0
    }
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            sample = json.loads(line)
            
            # Extract question
            question = sample.get('doc', {}).get('question', '')
            
            # Extract predictions: prefer raw resps (may include <think>), fallback to filtered_resps
            predictions = sample.get('resps')
            if predictions is None:
                predictions = sample.get('filtered_resps', [])

            # Flatten nested singletons like [["..."], ["..."]] -> ["...", "..."]
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
                # keep last for consistency with later selection
                predictions = [p[0] if isinstance(p, list) and p else p for p in predictions]

            if not isinstance(predictions, list):
                predictions = [predictions]
            
            # Extract correctness
            exact_match = sample.get('exact_match', 0)
            is_correct = bool(exact_match == 1 or exact_match == 1.0 or exact_match is True)
            
            # Create converted sample
            converted = {
                'idx': sample.get('doc_id', idx),
                'problem': question,
                'pred': predictions,
                'is_correct': is_correct
            }
            
            fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
    
    print(f"Converted {idx + 1} samples from {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input file from lm-eval')
    parser.add_argument('--output', required=True, help='Output file for logp_trim_experiment')
    args = parser.parse_args()
    
    convert_lm_eval_to_logp_format(args.input, args.output)

if __name__ == "__main__":
    main()
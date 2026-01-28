#!/usr/bin/env python3
"""Evaluation using lm-evaluation-harness with MLP pruning inference model.
export CUDA_VISIBLE_DEVICES=4
python -m src.evaluation.pruning_eval \
      --main_model_path path/to/Qwen2.5-7B-Instruct/ \
      --embedding_model_path path/to/Qwen3-Embedding-0.6B/ \
      --mlp_probe_path data/models/mlp_models/repeat_mlp.pt \
      --tasks gsm8k
This is a simple wrapper for :mod:`train_mlp.prune_inference_with_mlp`,
making it easy to evaluate on GSM8K or other tasks without writing Python scripts for each run.
"""

import argparse
from src.data_processing.mlp_pipeline.inference import get_harness_lm, evaluate_with_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pruned model with lm-evaluation-harness")
    parser.add_argument("--main_model_path", type=str, required=True, help="Path to the main causal LM")
    parser.add_argument("--embedding_model_path", type=str, required=True, help="Path to the embedding model")
    parser.add_argument("--mlp_probe_path", type=str, required=True, help="Path to the trained MLP probe")
    parser.add_argument("--mlp_hidden_dim", type=int, default=32, help="Hidden dimension used by the probe")
    parser.add_argument("--prefix_tokens", type=int, default=32, help="Number of prefix tokens for pruning")
    parser.add_argument("--remove_strategy", type=str, default="truncate_and_continue", 
                        choices=["terminate", "truncate_and_continue"],
                        help="Strategy for handling repetition: 'terminate' stops generation, 'truncate_and_continue' removes repetitive content and continues")
    parser.add_argument("--tasks", type=str, default="gsm8k", help="Evaluation task(s)")
    parser.add_argument("--limit", type=int, default=None, help="Subset of the dataset for quick evaluation")
    args = parser.parse_args()

    lm = get_harness_lm(
        main_model_path=args.main_model_path,
        embedding_model_path=args.embedding_model_path,
        mlp_probe_path=args.mlp_probe_path,
        mlp_hidden_dim=args.mlp_hidden_dim,
        prefix_tokens=args.prefix_tokens,
        remove_strategy=args.remove_strategy,
    )

    # print(f"Using remove strategy: {args.remove_strategy}")
    results = evaluate_with_harness(lm, tasks=args.tasks, limit=args.limit)
    
    # Extract and display final results
    task_results = results["results"][args.tasks]
    print(f"\n{'='*50}")
    print(f"Final Results for {args.tasks}:")
    print(f"{'='*50}")
    for metric, value in task_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


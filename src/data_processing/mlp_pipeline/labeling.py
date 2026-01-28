import os
import json
import argparse
import asyncio
import logging
import sys
from typing import Iterator, Coroutine, Any, List
from collections import Counter

import openai
from tqdm.asyncio import tqdm
from sentence_transformers import SentenceTransformer

# Import utilities from the same module
from .utils import (
    iter_dataset,
    load_embedding_model,
    get_truncated_think_content,
    init_nltk,
)

# --- LLM Configuration ---
# Read credentials and base URL from environment variables to avoid hardcoding secrets.
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
# --- End LLM Configuration ---

# --- Semantic Truncation Configuration ---
# Default path is a placeholder; in practice, set EMBEDDING_MODEL_PATH via
# environment variable or pass an explicit path into the loading utilities.
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/path/to/Qwen3-Embedding-0.6B/")
# --- End Semantic Truncation Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SYSTEM_PROMPT = (
    "You are a data annotator tasked with determining whether the model's <think> content repeats the user's question at the beginning. "
    "Given a question q and <think> text t, please answer with ONLY the following JSON format:\n"
    "{\n"
    '  "repeat": 1 or 0 (1 indicates repetition occurs),\n'
    '  "repeat_prefix_tokens": "number of tokens in the beginning repetition, 0 if no repetition"\n'
    "}"
)

def parse_llm_response(text: str) -> tuple[int, int]:
    """Parses the JSON response from the LLM."""
    try:
        data = json.loads(text)
        repeat = int(data.get("repeat", 0))
        prefix_len = int(data.get("repeat_prefix_tokens", 0))
        return repeat, prefix_len
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logging.warning(f"Failed to parse LLM response: {text}. Error: {e}")
        # Fallback to old parsing method for robustness
        repeat = 0
        prefix_len = 0
        for line in text.splitlines():
            # In case the response is not a valid JSON but contains the fields
            if '"repeat":' in line or 'repeat:' in line:
                try:
                    # Extract number, removing potential trailing commas or whitespace
                    val = line.split(":", 1)[1].strip()
                    if val.endswith(','):
                        val = val[:-1]
                    repeat = int(val)
                except (ValueError, IndexError):
                    pass
            if '"repeat_prefix_tokens":' in line or 'repeat_prefix_tokens:' in line:
                try:
                    val = line.split(":", 1)[1].strip()
                    if val.endswith(','):
                        val = val[:-1]
                    prefix_len = int(val)
                except (ValueError, IndexError):
                    pass
        return repeat, prefix_len


async def query_llm(q: str, t: str, model: str, client: openai.AsyncOpenAI, temperature: float) -> tuple[int, int]:
    """Queries the LLM API and returns the parsed labels."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题: {q}\n<think>: {t}"},
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    return parse_llm_response(text)


async def query_llm_with_retry(
    q: str, t: str, model: str, client: openai.AsyncOpenAI, temperature: float, max_retries: int = 3
) -> tuple[int, int]:
    """Queries the LLM with retries on failure."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await query_llm(q, t, model, client, temperature)
        except (
            openai.APIError,
            getattr(openai, "APITimeoutError", Exception),  # present in >=1.14
            openai.APIConnectionError,
            openai.RateLimitError,
        ) as e:
            logging.warning(f"API call failed on attempt {attempt + 1}/{max_retries} for Q: {q[:30]}... Error: {e}")
            last_exception = e
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logging.error(f"An unexpected error occurred for Q: {q[:30]}... Error: {e}")
            last_exception = e
            break  # Don't retry on unexpected errors

    logging.error(f"All {max_retries} retries failed for Q: {q[:30]}... Last exception: {last_exception}")
    return -1, -1  # Indicate failure


async def label_item(
    item: dict,
    llm_model: str,
    llm_client: openai.AsyncOpenAI,
    embed_model: SentenceTransformer,  # Can be None
    use_semantic_truncation: bool,
    n_samples: int,
    max_retries: int,
) -> dict:
    """
    Labels a single item, handling consensus ('pass@k'), retries,
    and optional semantic truncation.
    """
    q = item.get("q") or item.get("Q")
    t = item.get("t") or item.get("T") or item.get("answer")

    if not q or not t:
        item["repeat"] = -1
        item["prefix_len"] = -1
        logging.warning(f"Skipping item with missing 'q' or 't': {item}")
        return item

    t_to_label = t
    orig_chars = len(t)  # NEW: record original length
    if use_semantic_truncation and embed_model:
        original_len = orig_chars
        # These thresholds can be exposed as CLI args if needed
        t_to_label = get_truncated_think_content(
            question=q,
            think_content=t,
            embed_model=embed_model,
            initial_threshold=0.6,
            drop_threshold=0.15,
        )
        logging.info(
            f"Semantic truncation used. Original chars: {original_len}, "
            f"Truncated chars: {len(t_to_label)}"
        )
    trunc_chars = len(t_to_label)  # NEW: record truncated length
    # Store lengths for later stats (prefixed with underscore to avoid interfering with downstream tasks)
    item["_orig_chars"] = orig_chars
    item["_trunc_chars"] = trunc_chars
    # Set temperature higher for multiple samples to introduce variance for consensus check
    temperature = 0.4 if n_samples > 1 else 0

    tasks = [
        query_llm_with_retry(q, t_to_label, llm_model, llm_client, temperature, max_retries)
        for _ in range(n_samples)
    ]
    results = await asyncio.gather(*tasks)

    valid_results = [r for r in results if r != (-1, -1)]
    if not valid_results:
        logging.error(f"All API calls failed for question: {q[:50]}...")
        item["repeat"] = -1
        item["prefix_len"] = -1
        return item

    if n_samples == 1:
        repeat, prefix_len = valid_results[0]
    else:  # Consensus logic
        repeat_votes = Counter(r[0] for r in valid_results)
        most_common = repeat_votes.most_common(1)
        
        if not most_common:
             # This case should ideally not be reached if valid_results is not empty
            repeat, prefix_len = -1, -1
        else:
            best_repeat_vote, count = most_common[0]
            # Use majority vote if it exists, otherwise default to the most common
            if count <= len(valid_results) / 2:
                logging.warning(
                    f"No majority consensus for question: {q[:50]}... "
                    f"Votes: {repeat_votes}. Taking most common: {best_repeat_vote}."
                )
            repeat = best_repeat_vote

            if repeat == 1:
                # Average prefix_len for the majority vote
                prefix_lens = [r[1] for r in valid_results if r[0] == repeat and r[1] >= 0]
                prefix_len = int(sum(prefix_lens) / len(prefix_lens)) if prefix_lens else 0
            else:
                prefix_len = 0
    item["repeat"] = repeat
    item["prefix_len"] = prefix_len
    return item


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Label repeat tokens with LLM (async, with optional semantic truncation).")
    parser.add_argument("input", help="Input JSONL with fields q and t")
    parser.add_argument("output", help="Output JSONL with repeat labels")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of samples for consensus check (pass@k)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for API calls")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent API calls")
    parser.add_argument(
        "--use-semantic-truncation",
        action="store_true",
        help="Enable semantic truncation before sending to LLM to reduce cost and improve speed."
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Device identifier for the embedding model, e.g. 'cuda:0', 'cuda:1' or 'cpu'.\n"
             "If not provided, the script defaults to the first visible CUDA device or CPU."
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=60.0,
        help="Timeout (seconds) for the OpenAI HTTP request. Increase if your model needs long thinking time." 
    )
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("Please set environment variable OPENAI_API_KEY before running this script.")

    # Initialize models
    embed_model = None
    if args.use_semantic_truncation:
        init_nltk()
        embed_model = load_embedding_model(EMBEDDING_MODEL_PATH, device=args.embedding_device)

    llm_client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        timeout=args.api_timeout,
    )
    dataset = list(iter_dataset(args.input))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_with_semaphore(item: dict) -> Coroutine[Any, Any, dict]:
        async with semaphore:
            return await label_item(
                item,
                MODEL,
                llm_client,
                embed_model,
                args.use_semantic_truncation,
                args.n_samples,
                args.max_retries,
            )

    tasks = [process_with_semaphore(item) for item in dataset]

    results = await tqdm.gather(*tasks, desc="Labeling data")

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # --- Summary statistics for semantic truncation ---
    if args.use_semantic_truncation:
        total_items = len(results)
        truncated_items = 0
        total_orig_chars = 0
        total_trunc_chars = 0

        for itm in results:
            o = itm.get("_orig_chars", 0)
            tlen = itm.get("_trunc_chars", o)
            total_orig_chars += o
            total_trunc_chars += tlen
            if tlen < o:
                truncated_items += 1

        if total_orig_chars > 0:
            avg_reduction_ratio = 1 - (total_trunc_chars / total_orig_chars)
        else:
            avg_reduction_ratio = 0.0

        logging.info(
            "Semantic truncation summary: "
            f"{truncated_items}/{total_items} items truncated "
            f"({truncated_items / total_items:.2%}). "
            f"Average character reduction: {avg_reduction_ratio:.2%} across dataset."
        )

    # Optionally remove the helper keys to keep output clean
    # (Uncomment if you do not wish to keep these fields in the output file)
    # if args.use_semantic_truncation:
    #     cleaned_path = args.output + ".cleaned"
    #     with open(args.output, "r", encoding="utf-8") as fin, open(cleaned_path, "w", encoding="utf-8") as fout:
    #         for line in fin:
    #             obj = json.loads(line)
    #             obj.pop("_orig_chars", None)
    #             obj.pop("_trunc_chars", None)
    #             fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    #     logging.info(f"Cleaned output (without helper keys) written to {cleaned_path}")
    # --- End summary ---

    logging.info(f"Finished processing. Results saved to {args.output}")


def main() -> None:
    # To support running on Windows if needed, though the primary environment is Linux
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(amain())


if __name__ == "__main__":
    main()

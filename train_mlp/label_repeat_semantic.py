import argparse
import json
import logging
from typing import Iterator

import torch
import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
# It is recommended to load this path from a configuration file or pass as an argument.
EMBEDDING_MODEL_PATH = "/path/to/your/embedding_model/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- End Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download sentence tokenizer data if not present, useful for first-time run
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt')
    logging.info("'punkt' model downloaded successfully.")

def iter_dataset(path: str) -> Iterator[dict]:
    """Iterates over a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def find_repetition_boundary(
    question: str,
    think_content: str,
    model: SentenceTransformer,
    initial_threshold: float,
    drop_threshold: float
) -> tuple[int, int]:
    """
    Finds the boundary of question repetition in think_content using semantic similarity.

    The logic is as follows:
    1. A repetition is considered to exist only if the first sentence of the `think_content`
       is semantically similar to the question (above `initial_threshold`).
    2. If it exists, we then find where it ends by looking for a significant drop
       in similarity (`drop_threshold`) as we add more sentences.

    Returns:
        A tuple of (is_repetition, prefix_length_in_chars).
        `is_repetition` is 1 if found, 0 otherwise.
        `prefix_length_in_chars` is the character count of the repeated content.
    """
    q_embedding = model.encode(question, convert_to_tensor=True, device=DEVICE)
    sentences = nltk.sent_tokenize(think_content)

    if not sentences:
        return 0, 0

    # Step 1: Check if the first sentence constitutes a repetition
    first_sent_embedding = model.encode(sentences[0], convert_to_tensor=True, device=DEVICE)
    initial_sim = util.pytorch_cos_sim(q_embedding, first_sent_embedding).item()

    if initial_sim < initial_threshold:
        return 0, 0  # Not a repetition from the start

    # Step 2: A repetition is confirmed. Now find where it ends.
    is_repetition = 1
    repetition_end_char_index = len(sentences[0])
    cumulative_sentences = [sentences[0]]
    peak_similarity = initial_sim

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        cumulative_sentences.append(sentence)
        current_text = " ".join(cumulative_sentences)

        current_embedding = model.encode(current_text, convert_to_tensor=True, device=DEVICE)
        current_sim = util.pytorch_cos_sim(q_embedding, current_embedding).item()

        # Update peak similarity if the text is getting more aligned with the question
        if current_sim > peak_similarity:
            peak_similarity = current_sim

        # Check for a significant drop from the peak similarity
        if (peak_similarity - current_sim) > drop_threshold:
            # The drop occurred when adding the current sentence.
            # The boundary is the end of the *previous* cumulative text.
            repetition_end_char_index = len(" ".join(cumulative_sentences[:-1]))
            break
        else:
            # No significant drop, the repetition continues.
            repetition_end_char_index = len(current_text)

    return is_repetition, repetition_end_char_index


def main():
    parser = argparse.ArgumentParser(
        description="Label repeat content semantically using a sentence transformer model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", help="Input JSONL file.")
    parser.add_argument("output", help="Output JSONL file with semantic labels.")
    parser.add_argument(
        "--initial-threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for the first sentence to be considered a repetition."
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.15,
        help="Similarity drop from the peak to detect the end of a repetition."
    )
    args = parser.parse_args()

    # 1. Load the sentence transformer model
    logging.info(f"Loading embedding model from: {EMBEDDING_MODEL_PATH} on device: {DEVICE}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    except Exception as e:
        logging.error(f"Failed to load model from {EMBEDDING_MODEL_PATH}. Please ensure the path is correct and dependencies are installed.")
        logging.error(f"Error: {e}")
        return
    logging.info("Model loaded successfully.")

    # 2. Process the dataset
    with open(args.output, "w", encoding="utf-8") as fout:
        # Use tqdm for progress bar
        for item in tqdm(iter_dataset(args.input), desc="Processing data"):
            q, t = None, None
            # Handle the nested format from 'am_0.9M_sample_1k.jsonl'
            try:
                if 'messages' in item and len(item['messages']) > 1:
                    q = item['messages'][0]['content']
                    assistant_content = item['messages'][1]['content']
                    if '<think>' in assistant_content:
                        t = assistant_content.split('<think>', 1)[1].split('</think>', 1)[0].strip()
            except (KeyError, IndexError, AttributeError):
                pass  # Will be handled by the fallback below

            # Fallback to simple key-value format if the above fails or is not applicable
            if not q or not t:
                q = item.get("q") or item.get("Q")
                t = item.get("t") or item.get("T") or item.get("think_content")

            if not q or not t:
                logging.warning(f"Skipping item due to missing 'q' or 't' fields: {item.get('id', 'N/A')}")
                item["repeat_semantic"] = -1
                item["prefix_len_semantic"] = -1
            else:
                is_rep, prefix_len = find_repetition_boundary(
                    q, t, model, args.initial_threshold, args.drop_threshold
                )
                item["repeat_semantic"] = is_rep
                item["prefix_len_semantic"] = prefix_len

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Processing complete. Output written to {args.output}")

if __name__ == "__main__":
    main() 
import json
import logging
from typing import Iterator, Optional

import torch
import nltk
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
# It is recommended to load this path from a configuration file or pass as an argument.
EMBEDDING_MODEL_PATH = "/path/to/your/embedding_model/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- End Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def init_nltk():
    """Download NLTK sentence tokenizer data if not present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("NLTK 'punkt' model not found. Downloading...")
        nltk.download('punkt', quiet=True)
        logging.info("'punkt' model downloaded successfully.")

    # Handle newer NLTK versions where 'punkt_tab' may be required
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logging.info("NLTK 'punkt_tab' resource not found. Attempting to download...")
        # 'punkt_tab' is sometimes packaged inside the punkt download; downloading again is harmless.
        nltk.download('punkt_tab', quiet=True)
        logging.info("'punkt_tab' resource downloaded (or already present).")


def iter_dataset(path: str) -> Iterator[dict]:
    """Iterates over a JSONL file, yielding each line as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_embedding_model(
    model_path: str = EMBEDDING_MODEL_PATH,
    device: "Optional[str]" = None,
) -> SentenceTransformer:
    """Loads and returns a SentenceTransformer model.

    Args:
        model_path: Local path or HF model id for the embedding model.
        device: Explicit device string (e.g., "cuda:0", "cpu"). If ``None``, it
            falls back to the global *DEVICE* constant which in turn respects
            ``CUDA_VISIBLE_DEVICES``. This makes it easy to control the target
            GPU from the outside without modifying the code.
    """

    # Resolve target device. If the caller did not specify it, we fall back to
    # the module-level *DEVICE* constant which is derived from torch's default
    # device semantics (the first visible GPU when CUDA is available).
    target_device = device or DEVICE

    logging.info(f"Loading embedding model from: {model_path} on device: {target_device}")

    try:
        model = SentenceTransformer(model_path, device=target_device)
        logging.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        logging.error(
            f"Failed to load embedding model from '{model_path}' on device '{target_device}'."
        )
        raise e


def find_repetition_boundary(
    question: str,
    think_content: str,
    model: SentenceTransformer,
    initial_threshold: float,
    drop_threshold: float
) -> tuple[int, int]:
    """
    Finds the boundary of question repetition in think_content using semantic similarity.
    Returns a tuple of (is_repetition, prefix_length_in_chars).
    """
    if not think_content:
        return 0, 0
        
    q_embedding = model.encode(question, convert_to_tensor=True)
    sentences = nltk.sent_tokenize(think_content)

    if not sentences:
        return 0, 0

    first_sent_embedding = model.encode(sentences[0], convert_to_tensor=True)
    initial_sim = util.pytorch_cos_sim(q_embedding, first_sent_embedding).item()

    logging.debug(
        f"[find_repetition_boundary] initial_sim: {initial_sim:.4f} (threshold: {initial_threshold})"
    )

    if initial_sim < initial_threshold:
        return 0, 0

    is_repetition = 1
    repetition_end_char_index = len(sentences[0])
    cumulative_sentences = [sentences[0]]
    peak_similarity = initial_sim

    # For verbose analysis, track similarity progression (debug only)
    similarity_trace = [(0, initial_sim)]

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        current_text_list = cumulative_sentences + [sentence]
        current_text = " ".join(current_text_list)
        current_embedding = model.encode(current_text, convert_to_tensor=True)
        current_sim = util.pytorch_cos_sim(q_embedding, current_embedding).item()

        similarity_trace.append((i, current_sim))

        if current_sim > peak_similarity:
            peak_similarity = current_sim
            repetition_end_char_index = len(current_text)
            cumulative_sentences.append(sentence)
        elif (peak_similarity - current_sim) > drop_threshold:
            break
        else:
            repetition_end_char_index = len(current_text)
            cumulative_sentences.append(sentence)
            
    # Emit similarity trace every 50 sentences or on end (debug)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "[find_repetition_boundary] similarity progression (first 10 points): "
            + ", ".join(f"{idx}:{sim:.3f}" for idx, sim in similarity_trace[:10])
        )
        logging.debug(
            f"[find_repetition_boundary] detected repetition_end_char_index: {repetition_end_char_index} "
            f"(peak_sim: {peak_similarity:.4f})"
        )

    return is_repetition, repetition_end_char_index


def get_truncated_think_content(
    question: str,
    think_content: str,
    embed_model: SentenceTransformer,
    initial_threshold: float,
    drop_threshold: float,
    buffer_chars: int = 150,
    fallback_sentences: int = 4
) -> str:
    """
    Truncates think_content based on semantic repetition detection.
    If repetition is found, it returns the repeated part plus a buffer.
    If not, it returns a fixed number of initial sentences.
    """
    is_rep, prefix_len_chars = find_repetition_boundary(
        question, think_content, embed_model, initial_threshold, drop_threshold
    )

    if is_rep == 1:
        truncated_content = think_content[:prefix_len_chars + buffer_chars]
        logging.debug(f"Repetition detected. Original length: {len(think_content)}, Truncated to: {len(truncated_content)}")
        return truncated_content
    else:
        sentences = nltk.sent_tokenize(think_content)
        fallback_content = " ".join(sentences[:fallback_sentences])
        logging.debug(f"No repetition detected. Original length: {len(think_content)}, Truncated to fallback: {len(fallback_content)}")
        return fallback_content 
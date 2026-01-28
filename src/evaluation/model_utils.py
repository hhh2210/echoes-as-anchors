#!/usr/bin/env python3
"""
Model loading and tokenization utilities
"""

import os
from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass


@dataclass
class Span:
    """Token span representation."""
    start: int
    end: int  # end exclusive


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load model and tokenizer with appropriate settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If provided local directory, use only local files
    local_only = os.path.isdir(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=local_only,
    ).to(device).eval()
    
    return tokenizer, model, device


def token_ids(text: str, tokenizer: AutoTokenizer, add_special_tokens: bool = True) -> List[int]:
    """Get token IDs for text."""
    enc = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors=None)
    if isinstance(enc, dict):
        return enc["input_ids"]  # type: ignore[index]
    return enc.input_ids  # type: ignore[attr-defined]


def find_question_token_span_in_prompt(
    tokenizer: AutoTokenizer,
    prompt_with_question: str,
    raw_question: str,
) -> Optional[Span]:
    """Find the token span of the question within the prompt."""
    try:
        start_char = prompt_with_question.find(raw_question)
        if start_char < 0:
            return None
        end_char = start_char + len(raw_question)
        
        enc = tokenizer(prompt_with_question, add_special_tokens=True, return_offsets_mapping=True)  # type: ignore[arg-type]
        offsets = enc["offset_mapping"]  # type: ignore[index]
        
        token_start = None
        token_end = None
        for idx, (s, e) in enumerate(offsets):
            if e == 0 and s == 0:
                continue
            if e > start_char and s < end_char:
                if token_start is None:
                    token_start = idx
                token_end = idx + 1
                
        if token_start is None or token_end is None:
            return None
        return Span(start=token_start, end=token_end)
    except Exception:
        return None


def load_embedder(embedding_model_path: Optional[str], device: torch.device):
    """Load embedding model for MLP-based repeat detection."""
    if embedding_model_path is None:
        return None
    
    try:
        from src.data_processing.mlp_pipeline.utils import load_embedding_model, init_nltk  # type: ignore
        init_nltk()
        return load_embedding_model(embedding_model_path, device=str(device))
    except Exception:
        try:
            from train_mlp.utils import load_embedding_model, init_nltk  # type: ignore
            init_nltk()
            return load_embedding_model(embedding_model_path, device=str(device))
        except Exception:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(embedding_model_path, device=str(device))
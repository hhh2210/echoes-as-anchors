#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

import tiktoken

THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\n?")

def build_tokenizer(model_name: Optional[str], encoding_name: Optional[str]):
    if encoding_name:
        return tiktoken.get_encoding(encoding_name)
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
    return tiktoken.get_encoding("cl100k_base")

def extract_assistant_message(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages or []):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            return m["content"]
    return None

def get_text(item: Dict[str, Any], field: str) -> Optional[str]:
    if field == "assistant":
        return extract_assistant_message(item.get("messages", []))
    if field == "assistant_or_thinking":
        text = extract_assistant_message(item.get("messages", []))
        if text:
            return text
        return item.get("thinking_response")
    return item.get(field)

def main():
    ap = argparse.ArgumentParser(description="Compute average response token length with tiktoken.")
    ap.add_argument("--path", action="append", default=None, help="Path to JSON file (can be passed multiple times)")
    ap.add_argument("--paths", nargs="+", default=None, help="One or more JSON file paths")
    ap.add_argument("--model", default=None, help="Model name for tokenizer, e.g. gpt-4")
    ap.add_argument("--encoding", default="cl100k_base", help="Encoding name, e.g. cl100k_base")
    ap.add_argument("--field", default="assistant_or_thinking",
                    choices=["assistant", "thinking_response", "response", "assistant_or_thinking"],
                    help="Which field to read as response text")
    ap.add_argument("--strip-think", action="store_true",
                    help="Remove <think>...</think> blocks before tokenizing")
    args = ap.parse_args()

    selected_paths = []
    if args.path:
        selected_paths.extend(args.path)
    if args.paths:
        selected_paths.extend(args.paths)
    if not selected_paths:
        # Provide a default example path - user should replace with actual data
        print("Warning: No path specified. Please provide --path or --paths argument.")
        print("Example: python avg_response_tokens.py --path results/gsm8k_responses.json")
        sys.exit(1)

    enc = build_tokenizer(args.model, args.encoding)

    grand_total_tokens = 0
    grand_total_items = 0

    for p in selected_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON '{p}': {e}", file=sys.stderr)
            continue

        file_total_tokens = 0
        file_total_items = 0

        items = data if isinstance(data, list) else []
        for item in items:
            text = get_text(item, args.field)
            if not text or not isinstance(text, str):
                continue
            if args.strip_think:
                text = THINK_TAG_RE.sub("", text)
            file_total_tokens += len(enc.encode(text))
            file_total_items += 1

        if file_total_items == 0:
            print(f"No responses found for the chosen field in: {p}")
        else:
            avg = file_total_tokens / file_total_items
            print(f"File: {p}")
            print(f"  Counted items: {file_total_items}")
            print(f"  Total tokens: {file_total_tokens}")
            print(f"  Average response length (tokens): {avg:.2f}")

        grand_total_tokens += file_total_tokens
        grand_total_items += file_total_items

    if grand_total_items > 0 and len(selected_paths) > 1:
        overall_avg = grand_total_tokens / grand_total_items
        print("Overall:")
        print(f"  Counted items: {grand_total_items}")
        print(f"  Total tokens: {grand_total_tokens}")
        print(f"  Average response length (tokens): {overall_avg:.2f}")

if __name__ == "__main__":
    main()

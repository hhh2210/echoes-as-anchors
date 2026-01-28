import json
import argparse
import re
from tqdm import tqdm
import os

def extract_think_content(text: str) -> str:
    """Extracts content from within the <think>...</think> tags."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def process_raw_data(input_path: str, output_path: str):
    """
    Processes raw conversation data and extracts (question, think_content) pairs.
    It expects the input to be a JSONL file where each line is a JSON object
    with a 'messages' field containing the conversation.
    """
    print(f"Starting processing of {input_path}...")
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(tqdm(fin, desc="Processing conversations")):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Extract messages from the data object
                if 'messages' not in data:
                    print(f"Warning: Skipping line {i+1} - no 'messages' field found.")
                    continue
                    
                conversation = data['messages']
                
                question = None
                think_content = None
                
                # Flexible extraction based on roles
                for turn in conversation:
                    if not isinstance(turn, dict):
                        continue
                        
                    role = turn.get("role")
                    if role == "user":
                        question = turn.get("content")
                    elif role == "assistant":
                        assistant_content = turn.get("content")
                        if assistant_content:
                            think_content = extract_think_content(assistant_content)
                            
                        # Also check if think_content is directly available in info
                        info = turn.get("info", {})
                        if not think_content and isinstance(info, dict):
                            direct_think = info.get("think_content")
                            if direct_think:
                                think_content = direct_think

                if question and think_content:
                    output_item = {"q": question, "t": think_content}
                    fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    # Log a warning if a conversation doesn't match the expected format
                    print(f"Warning: Skipping line {i+1} due to missing question or think content.")

            except json.JSONDecodeError:
                print(f"Warning: Skipping line {i+1} due to JSON decoding error.")
            except Exception as e:
                print(f"An error occurred on line {i+1}: {e}")
    
    print(f"Successfully processed {processed_count} conversations.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess conversation data for repeat labeling.")
    parser.add_argument("input", help="Input JSONL file with raw conversation data.")
    parser.add_argument("output", help="Output JSONL file with (q, t) pairs.")
    args = parser.parse_args()
    
    process_raw_data(args.input, args.output)
    print(f"\nProcessing complete. Output written to {args.output}")

if __name__ == "__main__":
    main() 
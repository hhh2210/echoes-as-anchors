#!/usr/bin/env python3
"""
Script to collect model responses using Hugging Face Transformers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from datetime import datetime

# Define model paths
# It is recommended to load these from a config file or pass as arguments
MODELS = {
    "model1": "/path/to/your/echo_sft_model/",
    "model2": "/path/to/your/Qwen3-8B-Base_model/"
}

def get_model_response(question, model_path, max_new_tokens=512, temperature=0.7, top_p=0.95, device="cuda"):
    """
    Get response from Hugging Face model
    
    Args:
        question: The input question to ask the model
        model_path: Path to the model
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p value for nucleus sampling
        device: Device to run the model on
    
    Returns:
        dict: Response containing the answer and metadata
    """
    try:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Tokenize input
        inputs = tokenizer(question, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        result = {
            "question": question,
            "response": response_text.strip(),
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": outputs.shape[1] - input_length,
                "total_tokens": outputs.shape[1]
            }
        }
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"Error getting response: {e}")
        return {
            "question": question,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    parser = argparse.ArgumentParser(description="Collect responses from Hugging Face models")
    parser.add_argument("--question", type=str, default="Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles. How many different groups of two marbles can Tom choose?", help="Question to ask the model")
    parser.add_argument("--model", type=str, choices=["model1", "model2"], default="model1", 
                       help="Which model to use (model1: natural_sft, model2: Qwen3-8B-Base)")
    parser.add_argument("--model-path", type=str, help="Custom model path (overrides --model)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--output", type=str, help="Output file to save the response (optional)")
    parser.add_argument("--compare", action="store_true", help="Compare responses from both models")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare responses from both models
        results = []
        for model_name, model_path in MODELS.items():
            print(f"\n{'='*50}")
            print(f"Testing {model_name}: {model_path}")
            print(f"{'='*50}")
            result = get_model_response(
                args.question, 
                model_path,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.device
            )
            result["model_name"] = model_name
            results.append(result)
            
            if "error" not in result:
                print(f"\n{model_name} Response:\n{result['response']}\n")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nComparison saved to {args.output}")
    else:
        # Single model response
        model_path = args.model_path if args.model_path else MODELS[args.model]
        
        print(f"Question: {args.question}\n")
        
        result = get_model_response(
            args.question,
            model_path,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.device
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Model Response:\n{result['response']}\n")
            print(f"Token Usage: {result['usage']}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResponse saved to {args.output}")
    
    return result if not args.compare else results

if __name__ == "__main__":
    main()
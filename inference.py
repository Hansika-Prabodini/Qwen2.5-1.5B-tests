#!/usr/bin/env python3
"""
Interactive Inference Utility for Fine-tuned Derivatives Query Model

This script provides both single-query and interactive modes for running
inference with a fine-tuned LoRA model to convert natural language queries
into structured JSON filters.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import custom utilities
from utils.data_utils import format_prompt


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments as Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Interactive inference for derivatives query model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint directory"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode (default if --query not provided)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (lower = more deterministic)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="./Llama-3.2-1B",
        help="Path to base model (local path or HuggingFace model ID)"
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, base_model: str = "./Llama-3.2-1B") -> Tuple[Any, Any]:
    """
    Load fine-tuned PEFT model and tokenizer from checkpoint.

    Args:
        model_path: Path to model checkpoint directory
        base_model: Path to base model (local path or HuggingFace model ID)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        FileNotFoundError: If model path doesn't exist
    """
    print("Loading model and tokenizer...")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load tokenizer
    print("  Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("  ✓ Tokenizer loaded from fine-tuned model")
    except Exception as e:
        print(f"  Warning: Could not load tokenizer from model path: {e}")
        print(f"  Attempting to load tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        print("  ✓ Tokenizer loaded from base model")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Load model
    print("  Loading PEFT model...")
    try:
        # Try loading as PEFT model
        # Load base model
        print(f"  Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load PEFT weights
        print(f"  Loading PEFT adapter from: {model_path}")
        model = PeftModel.from_pretrained(
            base_model_obj,
            model_path,
            torch_dtype=torch.float16
        )
        
        # Merge weights for faster inference
        print("  Merging adapter weights...")
        model = model.merge_and_unload()
        
        print("  ✓ PEFT model loaded and merged successfully")
        
    except Exception as e:
        print(f"  Error loading PEFT model: {e}")
        print("  Attempting to load as regular model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("  ✓ Model loaded as regular checkpoint")
    
    # Set model to evaluation mode
    model.eval()
    
    print("✓ Model and tokenizer ready for inference\n")
    
    return model, tokenizer


def extract_json_from_generation(text: str) -> Optional[str]:
    """
    Extract JSON filter from generated text.
    
    Handles special tokens and extracts the JSON portion.
    
    Args:
        text: Generated text from model
        
    Returns:
        Extracted JSON string or None if not found
    """
    # Remove special tokens
    text = text.replace("<|eot_id|>", "").strip()
    text = text.replace("<|begin_of_text|>", "").strip()
    text = text.replace("<|start_header_id|>", "").strip()
    text = text.replace("<|end_header_id|>", "").strip()
    
    # Try to find JSON object in the text
    # Look for content between { and }
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return None


def parse_and_validate_json(text: str) -> Tuple[Optional[dict], bool, Optional[str]]:
    """
    Parse generated text to extract and validate JSON filter.
    
    Args:
        text: Generated text from model
        
    Returns:
        Tuple of (parsed_dict, success_flag, error_message)
    """
    json_str = extract_json_from_generation(text)
    
    if json_str is None:
        return None, False, "No JSON object found in generated text"
    
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return None, False, "Parsed JSON is not a dictionary"
        
        # Basic validation
        if "table" not in parsed:
            return parsed, False, "Missing 'table' key in JSON"
        if "filters" not in parsed:
            return parsed, False, "Missing 'filters' key in JSON"
        if not isinstance(parsed["filters"], list):
            return parsed, False, "'filters' must be a list"
        
        return parsed, True, None
    except json.JSONDecodeError as e:
        return None, False, f"JSON parsing error: {e.msg}"


def pretty_print_json(json_obj: dict, use_color: bool = True) -> None:
    """
    Pretty print JSON with optional color formatting.
    
    Args:
        json_obj: Dictionary to print
        use_color: Whether to use ANSI color codes
    """
    json_str = json.dumps(json_obj, indent=2, sort_keys=False)
    
    if use_color and sys.stdout.isatty():
        # Add simple ANSI color codes for better readability
        # Keys in cyan, strings in green, numbers in yellow
        colored_str = re.sub(r'"([^"]+)":', r'\033[36m"\1"\033[0m:', json_str)  # Keys in cyan
        colored_str = re.sub(r': "([^"]*)"', r': \033[32m"\1"\033[0m', colored_str)  # String values in green
        colored_str = re.sub(r': (\d+)', r': \033[33m\1\033[0m', colored_str)  # Numbers in yellow
        print(colored_str)
    else:
        print(json_str)


def clear_screen() -> None:
    """Clear the terminal screen (cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')


def generate_filter(
    model: Any,
    tokenizer: Any,
    query: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float
) -> Tuple[str, float]:
    """
    Generate JSON filter for a query.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        query: Natural language query
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Tuple of (generated_text, generation_time)
    """
    # Format prompt (system + user only, no assistant response)
    prompt = format_prompt(query, include_response=False)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate with timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Decode generated text (only the new tokens)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return generated_text, generation_time


def process_query(
    model: Any,
    tokenizer: Any,
    query: str,
    args: argparse.Namespace,
    show_timing: bool = True
) -> None:
    """
    Process a single query and display results.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        query: Natural language query
        args: Command-line arguments
        show_timing: Whether to display generation time
    """
    # Generate
    generated_text, gen_time = generate_filter(
        model,
        tokenizer,
        query,
        args.max_new_tokens,
        args.temperature,
        args.top_p
    )
    
    # Parse and validate
    parsed_json, success, error_msg = parse_and_validate_json(generated_text)
    
    if success and parsed_json is not None:
        # Successfully parsed valid JSON
        print("\n" + "="*60)
        print("Generated JSON Filter:")
        print("="*60)
        pretty_print_json(parsed_json, use_color=True)
        print("="*60)
        
        if show_timing:
            print(f"\nGeneration time: {gen_time:.3f} seconds")
    
    elif parsed_json is not None:
        # Parsed but with validation warnings
        print("\n" + "="*60)
        print("Generated JSON Filter (with warnings):")
        print("="*60)
        print(f"⚠ Warning: {error_msg}")
        print()
        pretty_print_json(parsed_json, use_color=True)
        print("="*60)
        
        if show_timing:
            print(f"\nGeneration time: {gen_time:.3f} seconds")
    
    else:
        # Failed to parse JSON
        print("\n" + "="*60)
        print("Error: Failed to parse JSON")
        print("="*60)
        print(f"Error message: {error_msg}")
        print("\nRaw generated text:")
        print("-"*60)
        print(generated_text[:500])  # Truncate for readability
        if len(generated_text) > 500:
            print(f"\n... (truncated, total length: {len(generated_text)} chars)")
        print("="*60)
        
        if show_timing:
            print(f"\nGeneration time: {gen_time:.3f} seconds")


def single_query_mode(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    """
    Run single query mode: process one query and exit.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("Single Query Mode")
    print("="*60)
    print(f"Query: {args.query}\n")
    
    process_query(model, tokenizer, args.query, args, show_timing=True)
    
    print()


def interactive_mode(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    """
    Run interactive mode: loop with user input for multiple queries.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("Enter natural language queries to generate JSON filters.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'clear' to clear the screen")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Enter query: ").strip()
            
            # Check for commands
            if query.lower() in ['quit', 'exit']:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            if query.lower() == 'clear':
                clear_screen()
                print("="*60)
                print("Interactive Mode")
                print("="*60)
                continue
            
            # Skip empty queries
            if not query:
                print("⚠ Please enter a query or type 'quit' to exit.")
                continue
            
            # Process the query
            process_query(model, tokenizer, query, args, show_timing=True)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("Please try again or type 'quit' to exit.")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)

        # Determine mode
        if args.query and not args.interactive:
            # Single query mode
            single_query_mode(model, tokenizer, args)
        else:
            # Interactive mode (default if no query or explicit --interactive)
            interactive_mode(model, tokenizer, args)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

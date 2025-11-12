#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Derivatives Query Model

This script evaluates a fine-tuned LoRA model on the test dataset,
calculating various metrics including exact match accuracy, per-field accuracy,
and parse success rate.
"""

import argparse
import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Import custom utilities
from utils.data_utils import load_jsonl_dataset, format_prompt


def setup_logging() -> logging.Logger:
    """
    Set up logging to console.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments as Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Llama-3.2-1B model on test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint directory"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Path to output JSON file for detailed results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (currently only supports 1)"
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
        "--base-model",
        type=str,
        default="./Llama-3.2-1B",
        help="Path to base model (local path or HuggingFace model ID)"
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, logger: logging.Logger, base_model: str = "./Llama-3.2-1B") -> Tuple[Any, Any]:
    """
    Load fine-tuned PEFT model and tokenizer from checkpoint.

    Args:
        model_path: Path to model checkpoint directory
        logger: Logger instance
        base_model: Path to base model (local path or HuggingFace model ID)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading model and tokenizer...")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded from fine-tuned model")
    except Exception as e:
        logger.warning(f"Could not load tokenizer from model path: {e}")
        logger.info(f"Attempting to load tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded from base model")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Load model
    logger.info("Loading PEFT model...")
    try:
        # Try loading as PEFT model directly
        # Load base model
        logger.info(f"Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load PEFT weights
        logger.info(f"Loading PEFT adapter from: {model_path}")
        model = PeftModel.from_pretrained(
            base_model_obj,
            model_path,
            torch_dtype=torch.float16
        )
        
        # Merge weights for faster inference
        logger.info("Merging adapter weights...")
        model = model.merge_and_unload()
        
        logger.info("✓ PEFT model loaded and merged successfully")
        
    except Exception as e:
        logger.error(f"Error loading PEFT model: {e}")
        logger.info("Attempting to load as regular model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✓ Model loaded as regular checkpoint")
    
    # Set model to evaluation mode
    model.eval()
    
    # Log model info
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Model dtype: {model.dtype}")
    
    return model, tokenizer


def extract_json_from_generation(text: str) -> Optional[str]:
    """
    Extract JSON filter from generated text.
    
    Handles special tokens like <|eot_id|> and extracts the JSON portion.
    
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


def normalize_json(json_obj: dict) -> str:
    """
    Normalize JSON object for fair comparison.
    
    Sorts keys and handles whitespace consistently.
    
    Args:
        json_obj: Dictionary to normalize
        
    Returns:
        Normalized JSON string
    """
    # Sort keys recursively
    def sort_dict(obj):
        if isinstance(obj, dict):
            return {k: sort_dict(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [sort_dict(item) for item in obj]
        else:
            return obj
    
    sorted_obj = sort_dict(json_obj)
    # Use compact JSON with sorted keys
    return json.dumps(sorted_obj, sort_keys=True, separators=(',', ':'))


def parse_generated_json(text: str) -> Tuple[Optional[dict], bool]:
    """
    Parse generated text to extract JSON filter.
    
    Args:
        text: Generated text from model
        
    Returns:
        Tuple of (parsed_dict, success_flag)
    """
    json_str = extract_json_from_generation(text)
    
    if json_str is None:
        return None, False
    
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return None, False
        return parsed, True
    except json.JSONDecodeError:
        return None, False


def calculate_per_field_accuracy(predicted: dict, ground_truth: dict) -> Dict[str, bool]:
    """
    Calculate per-field accuracy metrics.
    
    Args:
        predicted: Predicted filter dictionary
        ground_truth: Ground truth filter dictionary
        
    Returns:
        Dictionary with per-field match indicators
    """
    metrics = {
        "table_match": False,
        "filter_count_match": False,
        "all_filters_match": False,
        "matched_filters": 0,
        "total_filters": 0
    }
    
    # Check table match
    if predicted.get("table") == ground_truth.get("table"):
        metrics["table_match"] = True
    
    # Check filter count
    pred_filters = predicted.get("filters", [])
    gt_filters = ground_truth.get("filters", [])
    metrics["total_filters"] = len(gt_filters)
    
    if len(pred_filters) == len(gt_filters):
        metrics["filter_count_match"] = True
    
    # Check individual filter matches
    # Normalize filters for comparison (sort by column name)
    def normalize_filter(f):
        return (f.get("column"), f.get("op"), json.dumps(f.get("value"), sort_keys=True))
    
    pred_filter_set = set(normalize_filter(f) for f in pred_filters if isinstance(f, dict))
    gt_filter_set = set(normalize_filter(f) for f in gt_filters if isinstance(f, dict))
    
    # Count matching filters
    matched = pred_filter_set.intersection(gt_filter_set)
    metrics["matched_filters"] = len(matched)
    
    # Check if all filters match
    if pred_filter_set == gt_filter_set:
        metrics["all_filters_match"] = True
    
    return metrics


def run_inference(
    model: Any,
    tokenizer: Any,
    test_data: List[Dict],
    args: argparse.Namespace,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Run inference on test dataset and collect predictions.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_data: List of test examples
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        List of result dictionaries with predictions and metrics
    """
    logger.info(f"Running inference on {len(test_data)} examples...")
    
    results = []
    
    # Inference loop with progress bar
    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        query = example["query"]
        ground_truth = example["filter"]
        
        # Format prompt (system + user only, no assistant response)
        prompt = format_prompt(query, include_response=False)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode generated text (only the new tokens)
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            # Parse JSON from generated text
            predicted_json, parse_success = parse_generated_json(generated_text)
            
            # Calculate metrics
            exact_match = False
            per_field_metrics = {}
            
            if parse_success and predicted_json is not None:
                # Normalize and compare
                try:
                    pred_normalized = normalize_json(predicted_json)
                    gt_normalized = normalize_json(ground_truth)
                    exact_match = (pred_normalized == gt_normalized)
                    
                    # Calculate per-field accuracy
                    per_field_metrics = calculate_per_field_accuracy(predicted_json, ground_truth)
                except Exception as e:
                    logger.warning(f"Error normalizing JSON for example {i}: {e}")
                    per_field_metrics = calculate_per_field_accuracy(predicted_json, ground_truth)
            else:
                # If parsing failed, calculate basic per-field metrics with empty dict
                per_field_metrics = calculate_per_field_accuracy({}, ground_truth)
            
            # Store result
            result = {
                "example_id": i,
                "query": query,
                "ground_truth": ground_truth,
                "predicted": predicted_json if parse_success else None,
                "generated_text": generated_text[:500],  # Truncate for readability
                "parse_success": parse_success,
                "exact_match": exact_match,
                "per_field_metrics": per_field_metrics
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error generating for example {i}: {e}")
            # Store failed result
            per_field_metrics = calculate_per_field_accuracy({}, ground_truth)
            result = {
                "example_id": i,
                "query": query,
                "ground_truth": ground_truth,
                "predicted": None,
                "generated_text": "",
                "parse_success": False,
                "exact_match": False,
                "per_field_metrics": per_field_metrics,
                "error": str(e)
            }
            results.append(result)
    
    return results


def calculate_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with summary metrics
    """
    total = len(results)
    
    if total == 0:
        return {
            "total_examples": 0,
            "exact_match_accuracy": 0.0,
            "parse_success_rate": 0.0,
            "table_accuracy": 0.0,
            "filter_count_accuracy": 0.0,
            "all_filters_accuracy": 0.0,
            "average_filter_match_rate": 0.0
        }
    
    # Count successes
    exact_matches = sum(1 for r in results if r.get("exact_match", False))
    parse_successes = sum(1 for r in results if r.get("parse_success", False))
    table_matches = sum(1 for r in results if r.get("per_field_metrics", {}).get("table_match", False))
    filter_count_matches = sum(1 for r in results if r.get("per_field_metrics", {}).get("filter_count_match", False))
    all_filters_matches = sum(1 for r in results if r.get("per_field_metrics", {}).get("all_filters_match", False))
    
    # Calculate average filter match rate
    total_filters = sum(r.get("per_field_metrics", {}).get("total_filters", 0) for r in results)
    matched_filters = sum(r.get("per_field_metrics", {}).get("matched_filters", 0) for r in results)
    avg_filter_match_rate = (matched_filters / total_filters * 100) if total_filters > 0 else 0.0
    
    return {
        "total_examples": total,
        "exact_match_accuracy": exact_matches / total * 100,
        "parse_success_rate": parse_successes / total * 100,
        "table_accuracy": table_matches / total * 100,
        "filter_count_accuracy": filter_count_matches / total * 100,
        "all_filters_accuracy": all_filters_matches / total * 100,
        "average_filter_match_rate": avg_filter_match_rate,
        "counts": {
            "exact_matches": exact_matches,
            "parse_successes": parse_successes,
            "table_matches": table_matches,
            "filter_count_matches": filter_count_matches,
            "all_filters_matches": all_filters_matches,
            "total_filters": total_filters,
            "matched_filters": matched_filters
        }
    }


def display_summary_statistics(metrics: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Display summary statistics to console.
    
    Args:
        metrics: Summary metrics dictionary
        logger: Logger instance
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Examples: {metrics['total_examples']}")
    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"  Exact Match Accuracy:       {metrics['exact_match_accuracy']:.2f}%  ({metrics['counts']['exact_matches']}/{metrics['total_examples']})")
    logger.info(f"  Parse Success Rate:         {metrics['parse_success_rate']:.2f}%  ({metrics['counts']['parse_successes']}/{metrics['total_examples']})")
    logger.info(f"\nPer-Field Accuracy:")
    logger.info(f"  Table Name Match:           {metrics['table_accuracy']:.2f}%  ({metrics['counts']['table_matches']}/{metrics['total_examples']})")
    logger.info(f"  Filter Count Match:         {metrics['filter_count_accuracy']:.2f}%  ({metrics['counts']['filter_count_matches']}/{metrics['total_examples']})")
    logger.info(f"  All Filters Match:          {metrics['all_filters_accuracy']:.2f}%  ({metrics['counts']['all_filters_matches']}/{metrics['total_examples']})")
    logger.info(f"  Individual Filter Match:    {metrics['average_filter_match_rate']:.2f}%  ({metrics['counts']['matched_filters']}/{metrics['counts']['total_filters']})")
    logger.info("="*80)


def display_sample_predictions(results: List[Dict[str, Any]], num_samples: int, logger: logging.Logger) -> None:
    """
    Display sample predictions with ground truth comparison.
    
    Args:
        results: List of result dictionaries
        num_samples: Number of samples to display
        logger: Logger instance
    """
    logger.info("\n" + "="*80)
    logger.info(f"SAMPLE PREDICTIONS (First {num_samples})")
    logger.info("="*80)
    
    for i, result in enumerate(results[:num_samples]):
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(f"Query: {result['query']}")
        logger.info(f"\nGround Truth:")
        logger.info(f"  {json.dumps(result['ground_truth'], indent=2)}")
        logger.info(f"\nPredicted:")
        if result['predicted'] is not None:
            logger.info(f"  {json.dumps(result['predicted'], indent=2)}")
        else:
            logger.info(f"  [PARSE FAILED]")
            logger.info(f"  Raw output: {result['generated_text'][:200]}...")
        
        logger.info(f"\nMetrics:")
        logger.info(f"  Exact Match: {'✓' if result['exact_match'] else '✗'}")
        logger.info(f"  Parse Success: {'✓' if result['parse_success'] else '✗'}")
        if result.get('per_field_metrics'):
            pfm = result['per_field_metrics']
            logger.info(f"  Table Match: {'✓' if pfm.get('table_match') else '✗'}")
            logger.info(f"  Filter Count Match: {'✓' if pfm.get('filter_count_match') else '✗'}")
            logger.info(f"  Filters Matched: {pfm.get('matched_filters', 0)}/{pfm.get('total_filters', 0)}")
    
    logger.info("\n" + "="*80)


def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_path: str, logger: logging.Logger) -> None:
    """
    Save detailed results to JSON file.
    
    Args:
        results: List of result dictionaries
        metrics: Summary metrics
        output_path: Path to output file
        logger: Logger instance
    """
    logger.info(f"\nSaving results to {output_path}...")
    
    output_data = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "summary_metrics": metrics,
        "detailed_results": results
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Results saved successfully")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    """Main evaluation function."""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    try:
        logger.info("="*80)
        logger.info("Starting Model Evaluation")
        logger.info("="*80)
        logger.info(f"Model: {args.model}")
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Max new tokens: {args.max_new_tokens}")
        logger.info(f"Temperature: {args.temperature}")
        
        # Check CUDA availability
        logger.info("\n" + "="*80)
        logger.info("Device Information")
        logger.info("="*80)
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available")
            logger.info(f"  Device count: {torch.cuda.device_count()}")
            logger.info(f"  Current device: {torch.cuda.current_device()}")
            logger.info(f"  Device name: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠ CUDA not available, using CPU")
        
        # Load model and tokenizer
        logger.info("\n" + "="*80)
        logger.info("Loading Model")
        logger.info("="*80)
        model, tokenizer = load_model_and_tokenizer(args.model, logger, args.base_model)
        
        # Load test dataset
        logger.info("\n" + "="*80)
        logger.info("Loading Test Dataset")
        logger.info("="*80)
        test_data = load_jsonl_dataset(args.test_data)
        logger.info(f"✓ Test examples loaded: {len(test_data)}")
        
        # Run inference
        logger.info("\n" + "="*80)
        logger.info("Running Inference")
        logger.info("="*80)
        results = run_inference(model, tokenizer, test_data, args, logger)
        
        # Calculate summary metrics
        logger.info("\nCalculating metrics...")
        metrics = calculate_summary_metrics(results)
        
        # Display results
        display_summary_statistics(metrics, logger)
        display_sample_predictions(results, num_samples=10, logger=logger)
        
        # Save results
        save_results(results, metrics, args.output, logger)
        
        logger.info("\n" + "="*80)
        logger.info("Evaluation Complete!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

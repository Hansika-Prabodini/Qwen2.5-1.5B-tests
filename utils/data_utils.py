#!/usr/bin/env python3
"""
Data Utilities for Derivatives Query Dataset

This module provides functions for loading, formatting, and tokenizing data
for fine-tuning models to convert natural language queries into structured
JSON filters.
"""

import json
from typing import Dict, List, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
import datasets


def validate_filter_json(filter_json: dict) -> None:
    """
    Validate that filter_json conforms to the expected schema.
    
    Expected schema:
    {
        "table": str,
        "filters": [
            {
                "column": str,
                "op": str,
                "value": Any
            },
            ...
        ]
    }
    
    Args:
        filter_json: Dictionary to validate
        
    Raises:
        ValueError: If the structure doesn't conform to the expected schema
    """
    if not isinstance(filter_json, dict):
        raise ValueError(f"filter_json must be a dictionary, got {type(filter_json).__name__}")
    
    # Check required top-level keys
    if "table" not in filter_json:
        raise ValueError("filter_json must contain 'table' key")
    if "filters" not in filter_json:
        raise ValueError("filter_json must contain 'filters' key")
    
    # Validate table is a string
    if not isinstance(filter_json["table"], str):
        raise ValueError(f"'table' must be a string, got {type(filter_json['table']).__name__}")
    
    # Validate filters is a list
    if not isinstance(filter_json["filters"], list):
        raise ValueError(f"'filters' must be a list, got {type(filter_json['filters']).__name__}")
    
    # Validate each filter in the list
    for i, filter_item in enumerate(filter_json["filters"]):
        if not isinstance(filter_item, dict):
            raise ValueError(f"Filter at index {i} must be a dictionary, got {type(filter_item).__name__}")
        
        # Check required keys in each filter
        required_keys = {"column", "op", "value"}
        missing_keys = required_keys - set(filter_item.keys())
        if missing_keys:
            raise ValueError(f"Filter at index {i} missing required keys: {missing_keys}")
        
        # Validate types
        if not isinstance(filter_item["column"], str):
            raise ValueError(f"Filter at index {i}: 'column' must be a string")
        if not isinstance(filter_item["op"], str):
            raise ValueError(f"Filter at index {i}: 'op' must be a string")
        # value can be any type, so no validation needed


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """
    Load a JSONL dataset file.
    
    Reads a JSONL (JSON Lines) file where each line is a separate JSON object
    containing 'query' and 'filter' keys.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, each containing 'query' and 'filter' keys
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If a line contains malformed JSON or missing required keys
        IOError: If there's an error reading the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {file_path}")
    
    dataset = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Malformed JSON at line {line_num}: {e.msg}\n"
                        f"Line content: {line[:100]}..."
                    )
                
                # Validate required keys
                if not isinstance(data, dict):
                    raise ValueError(
                        f"Line {line_num}: Expected a JSON object (dict), got {type(data).__name__}"
                    )
                
                if "query" not in data:
                    raise ValueError(f"Line {line_num}: Missing required key 'query'")
                if "filter" not in data:
                    raise ValueError(f"Line {line_num}: Missing required key 'filter'")
                
                # Validate filter structure
                try:
                    validate_filter_json(data["filter"])
                except ValueError as e:
                    raise ValueError(f"Line {line_num}: Invalid filter structure - {str(e)}")
                
                dataset.append(data)
    
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")
    
    if not dataset:
        raise ValueError(f"No valid data found in file: {file_path}")
    
    return dataset


def format_prompt(query: str, filter_json: dict = None, include_response: bool = True) -> str:
    """
    Format a prompt using the Llama-3 template with special tokens.
    
    Applies the Llama-3 instruction format with proper special tokens for
    system message, user query, and optional assistant response.
    
    Args:
        query: The natural language query from the user
        filter_json: The structured filter dictionary (optional if include_response=False)
        include_response: If True, include the assistant's response with filter_json.
                         If False, omit the assistant message (for inference mode).
        
    Returns:
        Formatted prompt string with Llama-3 special tokens
        
    Raises:
        ValueError: If filter_json is invalid when include_response=True
    """
    # System message
    system_message = (
        "You are an assistant that converts natural language queries into "
        "structured JSON filters for data grids."
    )
    
    # Build the prompt
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}<|eot_id|>"
    )
    
    # Add assistant response if requested
    if include_response:
        if filter_json is None:
            raise ValueError("filter_json must be provided when include_response=True")
        
        # Validate the filter structure
        validate_filter_json(filter_json)
        
        # Convert filter_json to formatted JSON string
        filter_json_str = json.dumps(filter_json, indent=2)
        
        prompt += (
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{filter_json_str}<|eot_id|>"
        )
    else:
        # For inference, add the assistant header but no response
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt


def create_hf_dataset(jsonl_path: str, tokenizer) -> datasets.Dataset:
    """
    Create a HuggingFace Dataset from a JSONL file.
    
    Loads the JSONL file, formats each example with the Llama-3 template,
    and tokenizes the data for training.
    
    Args:
        jsonl_path: Path to the JSONL dataset file
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        HuggingFace Dataset object ready for training
        
    Raises:
        ValueError: If the JSONL file is invalid or empty
        FileNotFoundError: If the file doesn't exist
    """
    # Load the JSONL dataset
    data = load_jsonl_dataset(jsonl_path)
    
    # Format each example
    formatted_examples = []
    for example in data:
        formatted_text = format_prompt(
            query=example["query"],
            filter_json=example["filter"],
            include_response=True
        )
        formatted_examples.append({"text": formatted_text})
    
    # Create HuggingFace Dataset
    dataset = datasets.Dataset.from_dict({
        "text": [ex["text"] for ex in formatted_examples]
    })
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # We'll pad dynamically during training
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


class TokenizeDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenizing training data with label masking.
    
    This dataset wrapper tokenizes text data and applies label masking to ensure
    that the loss is only computed on the assistant's response tokens, not the
    prompt tokens.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Initialize the TokenizeDataset.
        
        Args:
            data: List of dictionaries with 'query' and 'filter' keys
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized example with label masking.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary containing 'input_ids', 'attention_mask', and 'labels' tensors
        """
        example = self.data[idx]
        query = example["query"]
        filter_json = example["filter"]
        
        # Format the full prompt with response
        full_text = format_prompt(query, filter_json, include_response=True)
        
        # Format just the prompt (without response) to find the boundary
        prompt_only = format_prompt(query, filter_json=None, include_response=False)
        
        # Tokenize both
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_encoding = self.tokenizer(
            prompt_only,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False  # Don't add extra special tokens
        )
        
        # Get the length of the prompt (number of tokens)
        prompt_length = len(prompt_encoding["input_ids"])
        
        # Create labels by copying input_ids
        input_ids = full_encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask the prompt tokens (set to -100)
        # Only compute loss on assistant response tokens
        labels[:prompt_length] = -100
        
        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": full_encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

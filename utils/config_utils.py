#!/usr/bin/env python3
"""
Configuration Utilities for Model Training

This module provides functions for loading, merging, and validating
configuration files for model training with LoRA fine-tuning.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    
    Reads and parses a YAML configuration file, returning it as a nested
    dictionary structure. Provides helpful error messages for common issues
    like missing files or invalid YAML syntax.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the parsed configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the YAML syntax is invalid or file is empty
        
    Examples:
        >>> config = load_config('configs/default_config.yaml')
        >>> print(config['model']['name'])
        'meta-llama/Llama-3.2-1B'
    """
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure the file exists at the specified path."
        )
    
    # Check if file is empty
    if config_path.stat().st_size == 0:
        raise ValueError(
            f"Configuration file is empty: {config_path}\n"
            f"Please provide a valid YAML configuration."
        )
    
    # Load and parse YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML syntax in {config_path}:\n{str(e)}\n"
            f"Please check your YAML formatting."
        )
    except Exception as e:
        raise ValueError(
            f"Error reading configuration file {config_path}: {str(e)}"
        )
    
    # Check if config is None (empty YAML file with just comments/whitespace)
    if config is None:
        raise ValueError(
            f"Configuration file contains no valid data: {config_path}\n"
            f"Please provide a valid YAML configuration."
        )
    
    # Check if config is a dictionary
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a YAML mapping/dictionary, got {type(config).__name__}\n"
            f"Please ensure your YAML file has a proper structure."
        )
    
    return config


def merge_args_with_config(config: dict, cli_args: argparse.Namespace) -> dict:
    """
    Merge CLI arguments with configuration dictionary.
    
    Takes a loaded configuration dictionary and CLI arguments, overriding
    config values where CLI arguments are provided. Supports both flat
    argument names (e.g., 'epochs', 'lr') and nested key overrides.
    
    The function creates a deep copy of the config to avoid modifying the
    original, then applies CLI argument overrides based on a predefined
    mapping between argument names and config paths.
    
    Supported CLI argument mappings:
        - epochs → training.num_epochs
        - lr/learning_rate → training.learning_rate
        - batch_size → training.batch_size
        - lora_r → lora.r
        - lora_alpha → lora.lora_alpha
        And more... (see _CLI_ARG_MAPPING)
    
    Args:
        config: Loaded configuration dictionary
        cli_args: Parsed CLI arguments from argparse.Namespace
        
    Returns:
        New dictionary with merged configuration (original config unchanged)
        
    Examples:
        >>> config = load_config('config.yaml')
        >>> args = argparse.Namespace(epochs=5, lr=1e-4)
        >>> merged = merge_args_with_config(config, args)
        >>> print(merged['training']['num_epochs'])
        5
    """
    import copy
    
    # Create a deep copy to avoid modifying the original config
    merged_config = copy.deepcopy(config)
    
    # Mapping from CLI argument names to config paths
    # Format: 'cli_arg_name': ('section', 'key')
    _CLI_ARG_MAPPING = {
        # Training arguments
        'epochs': ('training', 'num_epochs'),
        'num_epochs': ('training', 'num_epochs'),
        'batch_size': ('training', 'batch_size'),
        'lr': ('training', 'learning_rate'),
        'learning_rate': ('training', 'learning_rate'),
        'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
        'warmup_steps': ('training', 'warmup_steps'),
        'logging_steps': ('training', 'logging_steps'),
        'save_steps': ('training', 'save_steps'),
        'eval_steps': ('training', 'eval_steps'),
        'max_seq_length': ('training', 'max_seq_length'),
        'output_dir': ('training', 'output_dir'),
        'fp16': ('training', 'fp16'),
        
        # Model arguments
        'model_name': ('model', 'name'),
        'cache_dir': ('model', 'cache_dir'),
        'use_flash_attention': ('model', 'use_flash_attention'),
        
        # LoRA arguments
        'lora_r': ('lora', 'r'),
        'r': ('lora', 'r'),
        'lora_alpha': ('lora', 'lora_alpha'),
        'lora_dropout': ('lora', 'lora_dropout'),
        'target_modules': ('lora', 'target_modules'),
        'bias': ('lora', 'bias'),
        
        # Data arguments
        'train_file': ('data', 'train_file'),
        'val_file': ('data', 'val_file'),
        'test_file': ('data', 'test_file'),
    }
    
    # Convert Namespace to dict for easier handling
    if isinstance(cli_args, argparse.Namespace):
        cli_args_dict = vars(cli_args)
    else:
        cli_args_dict = cli_args
    
    # Apply overrides from CLI arguments
    for arg_name, arg_value in cli_args_dict.items():
        # Skip None values (arguments not provided)
        if arg_value is None:
            continue
        
        # Check if this argument has a mapping
        if arg_name in _CLI_ARG_MAPPING:
            section, key = _CLI_ARG_MAPPING[arg_name]
            
            # Ensure the section exists in merged_config
            if section not in merged_config:
                merged_config[section] = {}
            
            # Override the value
            merged_config[section][key] = arg_value
        
        # Support nested key overrides (e.g., "training.learning_rate")
        elif '.' in arg_name:
            parts = arg_name.split('.')
            if len(parts) == 2:
                section, key = parts
                # Ensure the section exists in merged_config
                if section not in merged_config:
                    merged_config[section] = {}
                merged_config[section][key] = arg_value
    
    return merged_config


def validate_config(config: dict) -> bool:
    """
    Validate that configuration contains all required fields with correct types.
    
    Performs comprehensive validation of a configuration dictionary, checking:
    - Presence of all required fields
    - Correct data types for each field
    - Valid value ranges where applicable
    - File paths that should exist or will be created
    
    Required fields:
        model:
            - name (str): Model identifier
            - cache_dir (str): Cache directory path
        training:
            - output_dir (str): Output directory path
            - num_epochs (int): Number of training epochs
            - batch_size (int): Training batch size
            - learning_rate (float): Learning rate
        lora:
            - r (int): LoRA rank
            - lora_alpha (int/float): LoRA alpha parameter
            - target_modules (list): List of modules to apply LoRA
        data:
            - train_file (str): Training data file path
            - val_file (str): Validation data file path
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If any required field is missing, has wrong type, or invalid value
        
    Examples:
        >>> config = load_config('config.yaml')
        >>> validate_config(config)
        True
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a dictionary, got {type(config).__name__}"
        )
    
    # Define required fields and their expected types
    # Format: (section, key, type, optional_validator_function)
    required_fields = [
        # Model section
        ('model', 'name', str, None),
        
        # Training section
        ('training', 'output_dir', str, None),
        ('training', 'num_epochs', int, lambda x: x > 0),
        ('training', 'batch_size', int, lambda x: x > 0),
        ('training', 'learning_rate', (int, float), lambda x: x > 0),
        
        # LoRA section
        ('lora', 'r', int, lambda x: x > 0),
        ('lora', 'lora_alpha', (int, float), lambda x: x > 0),
        ('lora', 'target_modules', list, lambda x: len(x) > 0),
        
        # Data section
        ('data', 'train_file', str, None),
        ('data', 'val_file', str, None),
    ]
    
    # Check each required field
    for section, key, expected_type, validator in required_fields:
        # Check if section exists
        if section not in config:
            raise ValueError(
                f"Missing required section '{section}' in configuration.\n"
                f"Please add the '{section}' section to your config file."
            )
        
        # Check if section is a dictionary
        if not isinstance(config[section], dict):
            raise ValueError(
                f"Section '{section}' must be a dictionary, got {type(config[section]).__name__}"
            )
        
        # Check if key exists in section
        if key not in config[section]:
            raise ValueError(
                f"Missing required field '{section}.{key}' in configuration.\n"
                f"Please add '{key}' to the '{section}' section."
            )
        
        value = config[section][key]
        
        # Check data type
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = ' or '.join(t.__name__ for t in expected_type)
                raise ValueError(
                    f"Field '{section}.{key}' must be of type {type_names}, "
                    f"got {type(value).__name__}"
                )
            else:
                raise ValueError(
                    f"Field '{section}.{key}' must be of type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        
        # Run custom validator if provided
        if validator is not None:
            try:
                if not validator(value):
                    raise ValueError(
                        f"Field '{section}.{key}' has invalid value: {value}"
                    )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Validation failed for '{section}.{key}' with value {value}: {str(e)}"
                )
    
    # Validate target_modules contains only strings
    target_modules = config['lora']['target_modules']
    for i, module in enumerate(target_modules):
        if not isinstance(module, str):
            raise ValueError(
                f"lora.target_modules[{i}] must be a string, got {type(module).__name__}"
            )
    
    # Additional validation for file paths
    # Note: We don't check if files exist yet as they might be generated
    # But we validate they are reasonable paths
    for file_key in ['train_file', 'val_file', 'test_file']:
        if file_key in config.get('data', {}):
            file_path = config['data'][file_key]
            if not isinstance(file_path, str):
                raise ValueError(
                    f"data.{file_key} must be a string path, got {type(file_path).__name__}"
                )
            if not file_path.strip():
                raise ValueError(
                    f"data.{file_key} cannot be an empty string"
                )
    
    return True

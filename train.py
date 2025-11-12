#!/usr/bin/env python3
"""
Training Script for LoRA Fine-tuning of Llama-3.2-1B

This script implements the complete training pipeline for fine-tuning
Llama-3.2-1B using LoRA (Low-Rank Adaptation) on the derivatives query dataset.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel

# Import custom utilities
from utils.config_utils import load_config, merge_args_with_config, validate_config
from utils.data_utils import create_hf_dataset


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Set up logging to both console and file.
    
    Args:
        output_dir: Directory where log file will be saved
        
    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
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
    
    # File handler
    log_file = Path(output_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments as Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Train Llama-3.2-1B with LoRA for derivatives query parsing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model and outputs"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data JSONL file"
    )
    
    # Optional override parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size per device (overrides config)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (overrides config)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    return parser.parse_args()


def validate_paths(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Validate that all required paths exist before training.
    
    Args:
        args: Parsed command-line arguments
        logger: Logger instance
        
    Raises:
        FileNotFoundError: If required files don't exist
    """
    logger.info("Validating paths...")
    
    # Check config file
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    logger.info(f"✓ Config file found: {args.config}")
    
    # Check training data
    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    logger.info(f"✓ Training data found: {args.train_data}")
    
    # Check validation data
    if not Path(args.val_data).exists():
        raise FileNotFoundError(f"Validation data not found: {args.val_data}")
    logger.info(f"✓ Validation data found: {args.val_data}")
    
    # Check resume checkpoint if provided
    if args.resume_from_checkpoint:
        if not Path(args.resume_from_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from_checkpoint}")
        logger.info(f"✓ Resume checkpoint found: {args.resume_from_checkpoint}")
    
    logger.info("All paths validated successfully")


def load_model_and_tokenizer(config: Dict[str, Any], logger: logging.Logger):
    """
    Load the base model and tokenizer with proper configuration.
    
    Args:
        config: Merged configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading base model and tokenizer...")
    
    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)
    use_flash_attention = config['model'].get('use_flash_attention', False)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Flash attention: {use_flash_attention}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Ensure special tokens are properly configured
    logger.info("Verifying Llama-3 special tokens...")
    special_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ]
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            logger.warning(f"Special token not found in vocabulary: {token}")
    
    # Load model
    logger.info("Loading base model...")
    model_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if config['training'].get('fp16', True) else torch.float32,
        "device_map": "auto",
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        logger.info(f"✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Retrying without flash attention...")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    
    # Log model info
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Model dtype: {model.dtype}")
    
    return model, tokenizer


def apply_lora(model, config: Dict[str, Any], logger: logging.Logger):
    """
    Apply LoRA configuration to the model.
    
    Args:
        model: Base model to apply LoRA to
        config: Merged configuration dictionary
        logger: Logger instance
        
    Returns:
        PEFT model with LoRA applied
    """
    logger.info("Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type="CAUSAL_LM"
    )
    
    logger.info(f"LoRA config:")
    logger.info(f"  - r: {lora_config.r}")
    logger.info(f"  - lora_alpha: {lora_config.lora_alpha}")
    logger.info(f"  - target_modules: {lora_config.target_modules}")
    logger.info(f"  - lora_dropout: {lora_config.lora_dropout}")
    logger.info(f"  - bias: {lora_config.bias}")
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    logger.info(f"✓ LoRA applied successfully")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model


def load_datasets(config: Dict[str, Any], tokenizer, logger: logging.Logger):
    """
    Load and tokenize training and validation datasets.
    
    Args:
        config: Merged configuration dictionary
        tokenizer: Tokenizer instance
        logger: Logger instance
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Loading datasets...")
    
    train_file = config['data']['train_file']
    val_file = config['data']['val_file']
    
    logger.info(f"Training data: {train_file}")
    logger.info(f"Validation data: {val_file}")
    
    # Load training dataset
    logger.info("Loading training dataset...")
    train_dataset = create_hf_dataset(train_file, tokenizer)
    logger.info(f"✓ Training examples: {len(train_dataset)}")
    
    # Load validation dataset
    logger.info("Loading validation dataset...")
    val_dataset = create_hf_dataset(val_file, tokenizer)
    logger.info(f"✓ Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_training_arguments(config: Dict[str, Any], resume_checkpoint: str = None) -> TrainingArguments:
    """
    Create TrainingArguments from configuration.
    
    Args:
        config: Merged configuration dictionary
        resume_checkpoint: Optional checkpoint path to resume from
        
    Returns:
        TrainingArguments instance
    """
    training_config = config['training']
    
    args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        fp16=training_config.get('fp16', True),
        bf16=False,
        optim="adamw_torch",
        logging_dir=f"{training_config['output_dir']}/logs",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    return args


def save_training_metadata(output_dir: str, config: Dict[str, Any], trainer_state: Any, logger: logging.Logger) -> None:
    """
    Save training metadata including config and final metrics.
    
    Args:
        output_dir: Directory to save metadata
        config: Configuration used for training
        trainer_state: Trainer state with metrics
        logger: Logger instance
    """
    logger.info("Saving training metadata...")
    
    metadata = {
        "config": config,
        "training_completed": datetime.now().isoformat(),
        "final_train_loss": None,
        "final_eval_loss": None,
        "total_steps": trainer_state.global_step if trainer_state else None,
    }
    
    # Extract final metrics if available
    if trainer_state and hasattr(trainer_state, 'log_history') and trainer_state.log_history:
        # Get last training loss
        train_losses = [entry.get('loss') for entry in trainer_state.log_history if 'loss' in entry]
        if train_losses:
            metadata['final_train_loss'] = train_losses[-1]
        
        # Get last eval loss
        eval_losses = [entry.get('eval_loss') for entry in trainer_state.log_history if 'eval_loss' in entry]
        if eval_losses:
            metadata['final_eval_loss'] = eval_losses[-1]
    
    # Save to JSON
    metadata_path = Path(output_dir) / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Training metadata saved to: {metadata_path}")
    if metadata['final_train_loss']:
        logger.info(f"  Final training loss: {metadata['final_train_loss']:.4f}")
    if metadata['final_eval_loss']:
        logger.info(f"  Final validation loss: {metadata['final_eval_loss']:.4f}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging first (before anything else)
    logger = setup_logging(args.output_dir)
    
    try:
        logger.info("="*80)
        logger.info("Starting Llama-3.2-1B LoRA Training")
        logger.info("="*80)
        
        # Validate paths
        validate_paths(args, logger)
        
        # Load and merge configuration
        logger.info("\n" + "="*80)
        logger.info("Loading Configuration")
        logger.info("="*80)
        config = load_config(args.config)
        logger.info(f"✓ Configuration loaded from: {args.config}")
        
        # Override with CLI arguments
        config = merge_args_with_config(config, args)
        logger.info("✓ CLI arguments merged with configuration")
        
        # Update data paths from CLI args
        config['data']['train_file'] = args.train_data
        config['data']['val_file'] = args.val_data
        config['training']['output_dir'] = args.output_dir
        
        # Validate configuration
        validate_config(config)
        logger.info("✓ Configuration validated successfully")
        
        # Log key configuration values
        logger.info("\nKey configuration:")
        logger.info(f"  Model: {config['model']['name']}")
        logger.info(f"  Output dir: {config['training']['output_dir']}")
        logger.info(f"  Epochs: {config['training']['num_epochs']}")
        logger.info(f"  Batch size: {config['training']['batch_size']}")
        logger.info(f"  Learning rate: {config['training']['learning_rate']}")
        logger.info(f"  LoRA rank (r): {config['lora']['r']}")
        
        # Check CUDA availability
        logger.info("\n" + "="*80)
        logger.info("Device Information")
        logger.info("="*80)
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available")
            logger.info(f"  Device count: {torch.cuda.device_count()}")
            logger.info(f"  Current device: {torch.cuda.current_device()}")
            logger.info(f"  Device name: {torch.cuda.get_device_name()}")
            logger.info(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"  Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        else:
            logger.warning("⚠ CUDA not available, training will use CPU (very slow)")
        
        # Load model and tokenizer
        logger.info("\n" + "="*80)
        logger.info("Loading Model and Tokenizer")
        logger.info("="*80)
        model, tokenizer = load_model_and_tokenizer(config, logger)
        
        # Apply LoRA
        logger.info("\n" + "="*80)
        logger.info("Applying LoRA")
        logger.info("="*80)
        model = apply_lora(model, config, logger)
        
        # Load datasets
        logger.info("\n" + "="*80)
        logger.info("Loading Datasets")
        logger.info("="*80)
        train_dataset, val_dataset = load_datasets(config, tokenizer, logger)
        
        # Create data collator
        logger.info("\n" + "="*80)
        logger.info("Setting up Data Collator")
        logger.info("="*80)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        logger.info("✓ Data collator created for causal language modeling")
        
        # Create training arguments
        logger.info("\n" + "="*80)
        logger.info("Creating Training Arguments")
        logger.info("="*80)
        training_args = create_training_arguments(config, args.resume_from_checkpoint)
        logger.info("✓ Training arguments configured")
        logger.info(f"  Output directory: {training_args.output_dir}")
        logger.info(f"  Num epochs: {training_args.num_train_epochs}")
        logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        logger.info(f"  FP16: {training_args.fp16}")
        logger.info(f"  Logging steps: {training_args.logging_steps}")
        logger.info(f"  Save steps: {training_args.save_steps}")
        logger.info(f"  Eval steps: {training_args.eval_steps}")
        
        # Initialize Trainer
        logger.info("\n" + "="*80)
        logger.info("Initializing Trainer")
        logger.info("="*80)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        logger.info("✓ Trainer initialized successfully")
        
        # Start training
        logger.info("\n" + "="*80)
        logger.info("Starting Training")
        logger.info("="*80)
        logger.info(f"Training will begin now...")
        
        try:
            if args.resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
                train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            else:
                train_result = trainer.train()
            
            logger.info("\n" + "="*80)
            logger.info("Training Completed Successfully!")
            logger.info("="*80)
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            logger.info(f"Training steps: {train_result.global_step}")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("\n" + "="*80)
            logger.error("CUDA Out of Memory Error")
            logger.error("="*80)
            logger.error(f"Error: {e}")
            logger.error("\nSuggestions to fix:")
            logger.error("  1. Reduce batch size (--batch-size)")
            logger.error("  2. Increase gradient accumulation steps (--gradient-accumulation-steps)")
            logger.error("  3. Reduce max sequence length (--max-seq-length)")
            logger.error("  4. Use a smaller LoRA rank (r) in config")
            logger.error("  5. Clear CUDA cache and restart")
            raise
        
        # Save final model
        logger.info("\n" + "="*80)
        logger.info("Saving Model and Tokenizer")
        logger.info("="*80)
        final_model_path = Path(args.output_dir) / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {final_model_path}")
        model.save_pretrained(final_model_path)
        logger.info("✓ Model saved successfully")
        
        logger.info(f"Saving tokenizer to: {final_model_path}")
        tokenizer.save_pretrained(final_model_path)
        logger.info("✓ Tokenizer saved successfully")
        
        # Save training metadata
        logger.info("\n" + "="*80)
        logger.info("Saving Training Metadata")
        logger.info("="*80)
        save_training_metadata(
            args.output_dir,
            config,
            trainer.state,
            logger
        )
        
        logger.info("\n" + "="*80)
        logger.info("All Done! 🎉")
        logger.info("="*80)
        logger.info(f"Model and outputs saved to: {args.output_dir}")
        logger.info(f"Final model location: {final_model_path}")
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("Training Failed")
        logger.error("="*80)
        logger.error(f"Error: {e}", exc_info=True)
        raise
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

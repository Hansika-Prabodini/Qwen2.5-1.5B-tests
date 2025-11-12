# Derivatives Terminology Fine-Tuning Project

A fine-tuning project for training language models to understand derivatives trading terminology, jargon, and domain-specific language used in financial markets.

## Overview

This project implements a fine-tuning pipeline for language models using LoRA (Low-Rank Adaptation) to teach models derivatives market terminology from the `Derivatives_Lingo.csv` dataset. The dataset contains 500+ terms including institution shorthands, FX pair nicknames, and trading jargon mapped to their real names and definitions.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- CUDA-compatible GPU (recommended for training)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd use-cases-and-demos
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

```
.
├── data/               # Raw CSV and generated JSONL datasets
├── models/             # Model checkpoints and training outputs
├── logs/               # Training logs and metrics
├── scripts/            # Data generation utilities
├── configs/            # YAML configuration files
├── utils/              # Utility modules
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Directory Purposes

- **`data/`** - Stores the raw `Derivatives_Lingo.csv` file and generated training datasets in JSONL format
- **`models/`** - Saves fine-tuned model checkpoints, adapter weights, and final model outputs
- **`logs/`** - Contains training logs, loss curves, and evaluation metrics
- **`scripts/`** - Houses Python scripts for synthetic data generation and preprocessing
- **`configs/`** - Stores YAML configuration files for training parameters and model settings
- **`utils/`** - Contains reusable utility modules for data processing and model operations

## Quick Start

*(Coming soon - to be populated in later tasks)*

This section will include:
- Data generation commands
- Model training instructions
- Inference examples
- Evaluation procedures

## Dataset

The project uses `Derivatives_Lingo.csv`, which contains:
- **500+ entries** of derivatives trading terminology
- **Categories**: Banks/Institutions, FX Pairs/FX Slang, and more
- **Fields**: alias, real_name, definition, category, db_table, db_column, db_type

## Dependencies

Key libraries used in this project:
- `transformers` - HuggingFace model loading and training
- `peft` - LoRA implementation for efficient fine-tuning
- `torch` - PyTorch deep learning framework
- `datasets` - HuggingFace dataset handling
- `accelerate` - Training optimization and distributed training
- `pandas` - CSV data processing
- `pyyaml` - Configuration file parsing

See `requirements.txt` for complete dependency list with version pins.


## Generate Data: 

python scripts/generate_data.py \
    --output data/generated \
    --size 1000 \
    --csv-path Derivatives_Lingo.csv

## Training: 

python train.py \
    --config configs/default_config.yaml \
    --output-dir models/output \
    --train-data data/generated/train.jsonl \
    --val-data data/generated/val.jsonl

## Testing: 

python test.py \
    --model models/output/final_model \
    --test-data data/generated/test.jsonl \
    --output test_results.json

## Inference:

python inference.py \
      --model models/output/final_model \
      --interactive

## License

*(To be determined)*

## Contributing

*(To be determined)*

# ViAG - Vietnamese Answer Generation

ViAG (Vietnamese Answer Generation) is a project that fine-tunes encoder-decoder models on Vietnamese question-answering tasks. This project provides tools for training, evaluating, and deploying models that can generate answers to questions in Vietnamese.

## Features

- Fine-tune pre-trained encoder-decoder models (like ViT5) for answer generation
- Support for local CSV datasets
- Comprehensive evaluation metrics (ROUGE, BLEU, METEOR, BERTScore)
- Command-line interface for training and evaluation
- Weights & Biases integration for experiment tracking
- Modular and extensible codebase

## Project Structure

```markdown
ViAG/
├── configs/              # Configuration files
├── datasets/             # Data files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── src/                  # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model configuration and training
│   ├── evaluation/       # Evaluation metrics and utilities
│   └── utils/            # Helper functions and constants
├── scripts/              # Training and evaluation scripts
├── models/               # Directory for saved models
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/              # Training outputs and logs
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ntphuc149/ViAG.git
cd ViAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the Vietnamese SpaCy model:

```
pip install https://gitlab.com/trungtv/vi_spacy/-/raw/master/packages/vi_core_news_lg-3.6.0/dist/vi_core_news_lg-3.6.0.tar.gz
```

4. Create a `.env` file with your API keys (optional):

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

## Data Format

The expected data format is a CSV file with the following columns:

- `context`: The context passage
- `question`: The question to be answered
- `answer`: The target generative answer

## Usage

### Training

Train a model using the command-line interface:

```bash
python scripts/train.py \
    --train_data datasets/train.csv \
    --val_data datasets/val.csv \
    --test_data datasets/test.csv \
    --model_name VietAI/vit5-base \
    --output_dir outputs/experiment1 \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --use_wandb
```

For more options, run:

```bash
python scripts/train.py --help
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/run_evaluate.py \
    --test_data datasets/test.csv \
    --model_path outputs/experiment1 \
    --output_dir outputs/evaluation1 \
    --batch_size 1
```

For more options, run:

```bash
python scripts/run_evaluate.py --help
```

```markdown
## LLM Instruction Fine-tuning (New Feature)

ViAG now supports instruction fine-tuning for Large Language Models (LLMs) using QLoRA technique. This allows you to fine-tune models like Qwen, Llama, and Mistral on Vietnamese QA tasks with limited GPU memory.

### Features

- **QLoRA Integration**: 4-bit quantization with LoRA for memory-efficient training
- **Multiple Instruction Formats**: Support for ChatML, Alpaca, Vicuna, Llama, and custom templates
- **Flexible Workflow**: Separate training, inference, and evaluation phases for long-running jobs
- **Automatic Data Splitting**: Split single dataset into train/val/test with customizable ratios

### Quick Start

#### 1. Full Pipeline (Train + Infer + Eval)

```bash
python scripts/train_llm.py \
    --do_train --do_infer --do_eval \
    --data_path Truong-Phuc/ViBidLQA \
    --model_name Qwen/Qwen2-0.5B \
    --instruction_template chatml \
    --output_dir outputs/qwen2-vibidlqa
```

#### 2. Separate Phases (for Kaggle/Colab sessions)

**Phase 1: Training (~11 hours)**
```bash
python scripts/train_llm.py \
    --do_train \
    --data_path data/train.csv \
    --model_name Qwen/Qwen2-0.5B \
    --num_epochs 10 \
    --output_dir outputs/qwen2-checkpoint
```

**Phase 2: Inference (~1 hour)**
```bash
python scripts/train_llm.py \
    --do_infer \
    --test_data data/test.csv \
    --checkpoint_path outputs/qwen2-checkpoint \
    --predictions_file outputs/predictions.csv
```

**Phase 3: Evaluation (~10 minutes)**
```bash
python scripts/train_llm.py \
    --do_eval \
    --predictions_file outputs/predictions.csv \
    --metrics_file outputs/metrics.json
```

### Instruction Templates

The framework supports multiple instruction formats:

#### ChatML (default)
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{context}
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

#### Alpaca
```
Below is an instruction that describes a task...

### Instruction:
{question}

### Input:
{context}

### Response:
{answer}
```

#### Custom Template
```bash
python scripts/train_llm.py \
    --instruction_template custom \
    --custom_template "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
```

### Configuration

You can use a JSON configuration file:

```bash
python scripts/train_llm.py --config configs/llm_config.json
```

Example configuration:
```json
{
  "model": {
    "name": "Qwen/Qwen2-0.5B",
    "instruction_template": "chatml"
  },
  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05
  },
  "training": {
    "num_epochs": 5,
    "batch_size": 1,
    "gradient_accumulation_steps": 16
  }
}
```

### Supported Models

- Qwen/Qwen2 series (0.5B, 1.5B, 7B)
- meta-llama/Llama-2 series
- mistralai/Mistral series
- Any other causal LM compatible with Transformers

### Advanced Options

```bash
python scripts/train_llm.py --help
```

Key parameters:
- `--train_ratio`, `--val_ratio`, `--test_ratio`: Data split ratios (default: 8:1:1)
- `--lora_r`: LoRA rank (default: 16)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--max_new_tokens`: Max tokens to generate (default: 512)
- `--use_wandb`: Enable W&B logging
```

## Configuration

You can customize the training process using a JSON configuration file:

```json
{
  "model": {
    "name": "vinai/bartpho-syllable-base",
    "max_input_length": 1024,
    "max_target_length": 256
  },
  "training": {
    "num_epochs": 5,
    "learning_rate": 3e-5,
    "batch_size": 2,
    "gradient_accumulation_steps": 16
  },
  "data": {
    "train_path": "datasets/train.csv",
    "val_path": "datasets/val.csv",
    "test_path": "datasets/test.csv"
  }
}
```

Then use it with:

```bash
python scripts/train.py --config configs/my_config.json
```

## Metrics

The project uses the following metrics to evaluate answer quality:

- `ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `ROUGE-L-SUM`: Measures n-gram overlap between generated and reference answers
- `BLEU-1`, `BLEU-2`, `BLEU-3`, `BLEU-4`: Measures precision of n-grams in generated answers
- `METEOR`: Measures unigram alignment between generated and reference answers
- `BERTScore`: Measures semantic similarity using BERT embeddings

## Models

The project currently supports the following models:

- `VietAI/vit5-base`
- `VietAI/vit5-large`
- `vinai/bartpho-syllable`
- `vinai/bartpho-syllable-base`
- Other encoder-decoder models compatible with the Hugging Face Transformers library

## License

This project is licensed under the MIT License - see the LICENSE file for details.

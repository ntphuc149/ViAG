# ViAG - Vietnamese Answer Generation

ViAG (Vietnamese Answer Generation) is a project for fine-tuning encoder-decoder models on Vietnamese question-answering tasks. This project provides tools for training, evaluating, and deploying models that can generate answers to questions in Vietnamese.

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
├── configs/ # Configuration files
├── datasets/ # Data files
│ ├── train.csv
│ ├── val.csv
│ └── test.csv
├── src/ # Source code
│ ├── data/ # Data loading and preprocessing
│ ├── models/ # Model configuration and training
│ ├── evaluation/ # Evaluation metrics and utilities
│ └── utils/ # Helper functions and constants
├── scripts/ # Training and evaluation scripts
├── models/ # Directory for saved models
├── notebooks/ # Jupyter notebooks for exploration
├── outputs/ # Training outputs and logs
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ViAG.git
cd ViAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Vietnamese SpaCy model:

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
python scripts/evaluate.py \
    --test_data datasets/test.csv \
    --model_path outputs/experiment1 \
    --output_dir outputs/evaluation1
```

For more options, run:

```bash
python scripts/evaluate.py --help
```

## Configuration

You can customize the training process using a JSON configuration file:

```json
{
  "model": {
    "name": "VietAI/vit5-base",
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

- `ROUGE-1`, `ROUGE-2`, `ROUGE-L`: Measures n-gram overlap between generated and reference answers
- `BLEU-1`, `BLEU-2`, `BLEU-3`, `BLEU-4`: Measures precision of n-grams in generated answers
- `METEOR`: Measures unigram alignment between generated and reference answers
- `BERTScore`: Measures semantic similarity using BERT embeddings

## Models

The project currently supports the following models:

- `VietAI/vit5-base`
- `VietAI/vit5-large`
- `vinai/phobert-base`
- `vinai/phobert-large`
- Other encoder-decoder models compatible with the Hugging Face Transformers library

## License

This project is licensed under the MIT License - see the LICENSE file for details.

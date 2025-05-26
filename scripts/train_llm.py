#!/usr/bin/env python
"""Training script for LLM instruction fine-tuning."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import argparse
import json
from datetime import datetime
import torch
import pandas as pd
from datasets import Dataset, load_dataset

from src.utils.helper import (
    setup_logging,
    load_config,
    save_config,
    check_required_files,
    load_environment_variables,
    check_gpu,
    create_directory_if_not_exists
)
from src.data.instruction_preprocessor import InstructionPreprocessor
from src.models.llm_trainer import LLMTrainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import calculate_metrics, calculate_bleu

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train, infer, or evaluate LLMs for instruction following.")
    
    # Mode arguments
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_infer", action="store_true",
                        help="Whether to run inference")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to single dataset file or HuggingFace dataset")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data CSV file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data CSV file")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data CSV file")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio when splitting single file")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio when splitting single file")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test set ratio when splitting single file")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B",
                        help="Name or path of the base model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint for inference/evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/llm",
                        help="Output directory")
    
    # Instruction format arguments
    parser.add_argument("--instruction_template", type=str, default="chatml",
                        choices=["chatml", "alpaca", "vicuna", "llama", "custom"],
                        help="Instruction template format")
    parser.add_argument("--custom_template", type=str, default=None,
                        help="Custom template string (use {context}, {question}, {answer} placeholders)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="System prompt to use")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every X steps")
    parser.add_argument("--use_packing", action="store_true",
                        help="Use packing for training efficiency")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", 
                                "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")
    
    # Quantization arguments
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                        choices=["fp4", "nf4"],
                        help="Quantization type")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True,
                        help="Use double quantization")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.75,
                        help="Top-p sampling")
    
    # Output arguments
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="Path to save/load predictions")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="Path to save metrics")
    
    # Other arguments
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="viag-llm",
                        help="W&B project name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()


def load_data(args, logger):
    """Load and prepare datasets based on arguments."""
    datasets = {}
    
    if args.data_path:
        # Load from single file or HuggingFace dataset
        logger.info(f"Loading data from {args.data_path}")
        
        try:
            # Try loading as HuggingFace dataset
            dataset = load_dataset(args.data_path, use_auth_token=True)
            if isinstance(dataset, dict):
                # Already split dataset
                datasets = dataset
            else:
                # Single dataset, need to split
                preprocessor = InstructionPreprocessor()
                datasets = preprocessor.split_dataset(
                    dataset,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.seed
                )
        except:
            # Load as CSV file
            df = pd.read_csv(args.data_path)
            dataset = Dataset.from_pandas(df, preserve_index=False)
            
            # Split dataset
            preprocessor = InstructionPreprocessor()
            datasets = preprocessor.split_dataset(
                dataset,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
    else:
        # Load from separate files
        if args.train_data and os.path.exists(args.train_data):
            df_train = pd.read_csv(args.train_data)
            datasets['train'] = Dataset.from_pandas(df_train, preserve_index=False)
            
        if args.val_data and os.path.exists(args.val_data):
            df_val = pd.read_csv(args.val_data)
            datasets['validation'] = Dataset.from_pandas(df_val, preserve_index=False)
            
        if args.test_data and os.path.exists(args.test_data):
            df_test = pd.read_csv(args.test_data)
            datasets['test'] = Dataset.from_pandas(df_test, preserve_index=False)
    
    return datasets


def do_train(args, datasets, logger):
    """Execute training phase."""
    logger.info("Starting training phase")
    
    # Prepare LoRA config
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    # Prepare quantization config
    quantization_config = {
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant
    }
    
    # Prepare training arguments
    training_args = {
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": 10,
        "report_to": "wandb" if args.use_wandb else "none"
    }
    
    # Initialize trainer
    trainer = LLMTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        quantization_config=quantization_config,
        lora_config=lora_config,
        training_args=training_args
    )
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Prepare instruction preprocessor
    instruction_template = args.custom_template if args.instruction_template == "custom" else args.instruction_template
    preprocessor = InstructionPreprocessor(
        instruction_template=instruction_template,
        system_prompt=args.system_prompt,
        max_length=args.max_seq_length
    )
    
    # Apply chat format to tokenizer
    trainer.tokenizer = preprocessor.create_chat_format(trainer.tokenizer)
    
    # Prepare datasets
    train_dataset = preprocessor.prepare_dataset(
        datasets['train'],
        include_output=True
    )
    
    eval_dataset = None
    if 'validation' in datasets:
        eval_dataset = preprocessor.prepare_dataset(
            datasets['validation'],
            include_output=True
        )
    
    # Setup trainer
    trainer.setup_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=args.use_packing
    )
    
    # Train
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
    
    train_result = trainer.train()
    logger.info(f"Training completed: {train_result}")
    
    if args.use_wandb:
        wandb.finish()


def do_infer(args, datasets, logger):
    """Execute inference phase."""
    logger.info("Starting inference phase")
    
    if not args.checkpoint_path:
        raise ValueError("--checkpoint_path required for inference")
    
    if 'test' not in datasets:
        raise ValueError("Test dataset not found")
    
    # Initialize trainer for inference
    trainer = LLMTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Load model from checkpoint
    trainer.load_model_and_tokenizer(checkpoint_path=args.checkpoint_path)
    
    # Get instruction template
    instruction_template = args.custom_template if args.instruction_template == "custom" else args.instruction_template
    
    # Set default predictions file if not specified
    if not args.predictions_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.predictions_file = os.path.join(args.output_dir, f"predictions_{timestamp}.csv")
    
    # Generate predictions
    predictions = trainer.generate_predictions(
        test_dataset=datasets['test'],
        instruction_template=instruction_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_file=args.predictions_file
    )
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Predictions saved to {args.predictions_file}")


def do_eval(args, logger):
    """Execute evaluation phase."""
    logger.info("Starting evaluation phase")
    
    if not args.predictions_file:
        raise ValueError("--predictions_file required for evaluation")
    
    # Load predictions
    if args.predictions_file.endswith('.csv'):
        df = pd.read_csv(args.predictions_file)
    elif args.predictions_file.endswith('.json'):
        df = pd.read_json(args.predictions_file)
    else:
        raise ValueError("Predictions file must be CSV or JSON")
    
    predictions = df['predictions'].tolist()
    references = df['references'].tolist()
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    
    # BLEU scores
    bleu_scores = calculate_bleu(predictions, references)
    
    # Other metrics (ROUGE, METEOR, BERTScore)
    other_metrics = calculate_metrics(predictions, references)
    
    # Combine all metrics
    all_metrics = {**bleu_scores, **other_metrics}
    
    # Set default metrics file if not specified
    if not args.metrics_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.metrics_file = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
    
    # Save metrics
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    with open(args.metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    
    # Print metrics
    print("\n===== EVALUATION METRICS =====")
    for key, value in all_metrics.items():
        print(f"{key}: {value:.4f}")
        logger.info(f"{key}: {value:.4f}")
    print("==============================\n")
    
    logger.info(f"Metrics saved to {args.metrics_file}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    if not any([args.do_train, args.do_infer, args.do_eval]):
        raise ValueError("At least one of --do_train, --do_infer, --do_eval must be specified")
    
    # Set up logging
    log_file = args.log_file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.output_dir, f"llm_training_{timestamp}.log")
    
    create_directory_if_not_exists(args.output_dir)
    setup_logging(log_file=log_file, level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LLM training/inference/evaluation script")
    logger.info(f"Arguments: {args}")
    
    # Load environment variables
    load_environment_variables()
    
    # Check GPU availability
    check_gpu()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration from file if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        # Update args with config values (command line args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Load datasets if needed for training or inference
    datasets = {}
    if args.do_train or args.do_infer:
        datasets = load_data(args, logger)
    
    # Execute requested phases
    if args.do_train:
        do_train(args, datasets, logger)
    
    if args.do_infer:
        do_infer(args, datasets, logger)
    
    if args.do_eval:
        do_eval(args, logger)
    
    logger.info("Script completed successfully")


if __name__ == "__main__":
    main()
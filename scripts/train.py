#!/usr/bin/env python
"""Training script for the ViAG project."""

import os
import logging
import argparse
import json
from datetime import datetime
import torch
import nltk
import wandb
from dotenv import load_dotenv

from src.utils.helper import (
    setup_logging, 
    load_config, 
    save_config, 
    check_required_files,
    load_environment_variables,
    check_gpu,
    create_directory_if_not_exists
)
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_config import ModelConfig
from src.models.trainer import ViAGTrainer
from src.evaluation.metrics import compute_metrics_for_seq2seq

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model for Vietnamese answer generation.")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="datasets/train.csv",
                        help="Path to training data CSV file")
    parser.add_argument("--val_data", type=str, default="datasets/val.csv",
                        help="Path to validation data CSV file")
    parser.add_argument("--test_data", type=str, default="datasets/test.csv",
                        help="Path to test data CSV file")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="VietAI/vit5-base",
                        help="Name or path of the model to use")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for training results")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every X steps")
    
    # Config and other arguments
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="viag",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_file = args.log_file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.output_dir, f"training_{timestamp}.log")
    
    setup_logging(log_file=log_file, level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training script")
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
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    
    # Update configuration with command line arguments
    config_model = config.get('model', {})
    config_model['name'] = args.model_name
    config_model['max_input_length'] = args.max_input_length
    config_model['max_target_length'] = args.max_target_length
    
    config_training = config.get('training', {})
    config_training['num_epochs'] = args.num_epochs
    config_training['batch_size'] = args.batch_size
    config_training['eval_batch_size'] = args.eval_batch_size
    config_training['learning_rate'] = args.learning_rate
    config_training['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    config_training['warmup_ratio'] = args.warmup_ratio
    config_training['weight_decay'] = args.weight_decay
    config_training['save_steps'] = args.save_steps
    config_training['eval_steps'] = args.eval_steps
    
    config_data = config.get('data', {})
    config_data['train_path'] = args.train_data
    config_data['val_path'] = args.val_data
    config_data['test_path'] = args.test_data
    
    config.update({
        'model': config_model,
        'training': config_training,
        'data': config_data,
        'output_dir': args.output_dir
    })
    
    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(args.output_dir, f"config_{timestamp}.json")
    create_directory_if_not_exists(args.output_dir)
    save_config(config, config_path)
    
    # Initialize Weights & Biases logging
    if args.use_wandb:
        logger.info("Initializing Weights & Biases logging")
        wandb_run_name = args.wandb_run_name or f"run_{timestamp}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, config=config)
    
    # Check if required files exist
    data_files = [args.train_data, args.val_data, args.test_data]
    if not check_required_files(data_files):
        logger.error("Required data files not found. Exiting.")
        return
    
    # Load data
    logger.info("Loading data")
    data_loader = DataLoader(args.train_data, args.val_data, args.test_data)
    datasets = data_loader.load_data()
    
    # Verify data format
    if not data_loader.verify_data_format():
        logger.error("Data format verification failed. Exiting.")
        return
    
    logger.info(f"Dataset sizes: {data_loader.get_dataset_sizes()}")
    
    # Initialize model and tokenizer
    logger.info(f"Initializing model: {args.model_name}")
    model_config = ModelConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        training_args={
            "num_train_epochs": args.num_epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "logging_steps": 100,
            "report_to": "wandb" if args.use_wandb else "none"
        }
    )
    
    tokenizer = model_config.load_tokenizer()
    model = model_config.load_model()
    data_collator = model_config.create_data_collator()
    training_args = model_config.get_training_arguments()
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length
    )
    
    train_dataset = preprocessor.preprocess_dataset(datasets['train'])
    val_dataset = preprocessor.preprocess_dataset(datasets['validation']) if 'validation' in datasets else None
    
    # Set up trainer with metrics
    compute_metrics = compute_metrics_for_seq2seq(tokenizer)
    
    logger.info("Setting up trainer")
    trainer = ViAGTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.setup_trainer(
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model
    logger.info("Starting training")
    train_output = trainer.train()
    logger.info(f"Training completed. Output: {train_output}")
    
    # Evaluate model
    if val_dataset:
        logger.info("Evaluating model")
        eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
        logger.info(f"Evaluation metrics: {eval_metrics}")
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
    logger.info("Training script completed successfully")

if __name__ == "__main__":
    main()
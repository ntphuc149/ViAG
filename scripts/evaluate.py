#!/usr/bin/env python
"""Evaluation script for the ViAG project."""

import os
import logging
import argparse
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.helper import (
    setup_logging, 
    load_config, 
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
from src.evaluation.evaluator import Evaluator

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model for Vietnamese answer generation.")
    
    # Data arguments
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data CSV file")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for evaluation results")
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="Path to save predictions. If not provided, will use 'predictions.csv' in output_dir")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="Path to save metrics. If not provided, will use 'metrics.json' in output_dir")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for prediction")
    
    # Other arguments
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
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
        log_file = os.path.join(args.output_dir, f"evaluation_{timestamp}.log")
    
    setup_logging(log_file=log_file, level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting evaluation script")
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
    
    # Check if required files exist
    if not check_required_files([args.test_data, args.model_path]):
        logger.error("Required files not found. Exiting.")
        return
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    data_loader = DataLoader(test_path=args.test_data)
    datasets = data_loader.load_data()
    
    if 'test' not in datasets:
        logger.error("Test dataset not found. Exiting.")
        return
    
    test_dataset = datasets['test']
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Preprocess test data
    logger.info("Preprocessing test data")
    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length
    )
    
    test_dataset_processed = preprocessor.preprocess_dataset(test_dataset)
    
    # Set up training arguments for prediction
    training_args = ModelConfig(
        model_name=args.model_path,
        output_dir=args.output_dir,
        training_args={
            "per_device_eval_batch_size": args.batch_size,
            "predict_with_generate": True,
            "generation_max_length": args.max_length,
            "generation_num_beams": args.num_beams
        }
    ).get_training_arguments()
    
    # Set up trainer
    trainer = ViAGTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=compute_metrics_for_seq2seq(tokenizer)
    )
    
    trainer.setup_trainer(
        train_dataset=None,
        eval_dataset=test_dataset_processed
    )
    
    # Set up evaluator
    evaluator = Evaluator(
        trainer=trainer.trainer,
        tokenizer=tokenizer
    )
    
    # Generate predictions and compute metrics
    logger.info("Generating predictions")
    
    # Set up output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.predictions_file:
        predictions_file = os.path.join(args.output_dir, f"predictions_{timestamp}.csv")
    else:
        predictions_file = args.predictions_file
    
    if not args.metrics_file:
        metrics_file = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
    else:
        metrics_file = args.metrics_file
    
    # Create output directory
    create_directory_if_not_exists(args.output_dir)
    
    # Generate predictions
    predictions, references = evaluator.predict(
        dataset=test_dataset_processed,
        output_file=predictions_file,
        max_length=args.max_length,
        num_beams=args.num_beams
    )
    
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Compute metrics
    logger.info("Computing metrics")
    metrics = evaluator.compute_metrics(predictions, references)
    
    # Save metrics
    evaluator.save_metrics(metrics, metrics_file)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Print metrics
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("Evaluation script completed successfully")

if __name__ == "__main__":
    main()
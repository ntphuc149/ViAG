"""Trainer for the ViAG project."""

import os
import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import Seq2SeqTrainer
from datasets import Dataset

logger = logging.getLogger(__name__)

class ViAGTrainer:
    """Trainer for the ViAG project.
    
    This class handles model training and prediction.
    """
    
    def __init__(self, model, tokenizer, training_args, 
                 data_collator=None, compute_metrics=None):
        """Initialize the ViAGTrainer.
        
        Args:
            model: Model to train.
            tokenizer: Tokenizer for the model.
            training_args: Training arguments.
            data_collator: Data collator for the model.
            compute_metrics: Function to compute metrics during evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        
        self.trainer = None
    
    def setup_trainer(self, train_dataset: Dataset, 
                     eval_dataset: Optional[Dataset] = None):
        """Set up the trainer.
        
        Args:
            train_dataset (Dataset): Training dataset.
            eval_dataset (Dataset, optional): Evaluation dataset. Defaults to None.
        """
        logger.info("Setting up Seq2SeqTrainer")
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
    
    def train(self):
        """Train the model.
        
        Returns:
            TrainOutput: Training output.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        
        logger.info("Starting training")
        return self.trainer.train()
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate the model.
        
        Args:
            eval_dataset (Dataset, optional): Evaluation dataset. Defaults to None.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        
        logger.info("Starting evaluation")
        return self.trainer.evaluate(eval_dataset=eval_dataset)
    
    def predict(self, test_dataset: Dataset, 
               max_length: int = 256, 
               num_beams: int = 4):
        """Generate predictions for the test dataset.
        
        Args:
            test_dataset (Dataset): Test dataset.
            max_length (int, optional): Maximum output length. Defaults to 256.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            
        Returns:
            List[str]: Predictions.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        
        logger.info("Generating predictions")
        
        # Use trainer to generate predictions
        predictions = self.trainer.predict(test_dataset=test_dataset).predictions
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        return decoded_preds
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the model.
        
        Args:
            output_dir (str, optional): Output directory. Defaults to None.
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        
        save_dir = output_dir or self.training_args.output_dir
        logger.info(f"Saving model to {save_dir}")
        
        self.trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
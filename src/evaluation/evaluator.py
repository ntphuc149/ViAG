"""Evaluator for the ViAG project."""

import os
import logging
import pandas as pd
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from datasets import Dataset

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluator for the ViAG project.
    
    This class handles model evaluation and prediction.
    """
    
    def __init__(self, trainer, tokenizer, metrics_fn=None):
        """Initialize the Evaluator.
        
        Args:
            trainer: The trainer to use for evaluation.
            tokenizer: The tokenizer to use for decoding predictions.
            metrics_fn: Function to compute metrics.
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.metrics_fn = metrics_fn
    
    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset.
        
        Args:
            dataset (Dataset): The dataset to evaluate on.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        logger.info("Starting evaluation")
        metrics = self.trainer.evaluate(eval_dataset=dataset)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    # def predict(self, dataset: Dataset, 
    #            output_file: Optional[str] = None,
    #            max_length: int = 256,
    #            num_beams: int = 4) -> List[str]:
    #     """Generate predictions for a dataset.
        
    #     Args:
    #         dataset (Dataset): The dataset to generate predictions for.
    #         output_file (str, optional): Path to save predictions. Defaults to None.
    #         max_length (int, optional): Maximum length of predictions. Defaults to 256.
    #         num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            
    #     Returns:
    #         List[str]: Generated predictions.
    #     """
    #     logger.info("Generating predictions")
        
    #     # Generate predictions
    #     preds = self.trainer.predict(
    #         test_dataset=dataset,
    #         max_length=max_length,
    #         num_beams=num_beams
    #     )
        
    #     # Decode predictions
    #     decoded_preds = self.tokenizer.batch_decode(
    #         preds.predictions, 
    #         skip_special_tokens=True, 
    #         clean_up_tokenization_spaces=True
    #     )
        
    #     # Get references
    #     labels = preds.label_ids
    #     labels = labels.reshape(labels.shape[0], -1)
        
    #     # Replace -100 with pad token id
    #     labels = labels.copy()
    #     labels[labels == -100] = self.tokenizer.pad_token_id
        
    #     # Decode references
    #     decoded_refs = self.tokenizer.batch_decode(
    #         labels, 
    #         skip_special_tokens=True, 
    #         clean_up_tokenization_spaces=True
    #     )
        
    #     # Format text
    #     decoded_preds = [self.format_text(pred) for pred in decoded_preds]
    #     decoded_refs = [self.format_text(ref) for ref in decoded_refs]
        
    #     # Save predictions if output file is provided
    #     if output_file:
    #         self.save_predictions(decoded_preds, decoded_refs, output_file)
        
    #     return decoded_preds, decoded_refs

    def predict(self, dataset: Dataset, 
           output_file: Optional[str] = None,
           max_length: int = 256,
           num_beams: int = 4) -> List[str]:
        """Generate predictions for a dataset."""
        logger.info("Generating predictions")
        
        # Sử dụng model trực tiếp thay vì qua trainer
        model = self.trainer.model
        model.eval()
        device = model.device
        
        all_preds = []
        all_refs = []
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Move to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Generate
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_length,
                    num_beams=num_beams
                )
                
                # Decode predictions
                pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                all_preds.append(pred)
                
                # Decode references (handle -100)
                labels = batch["labels"][0].cpu().numpy()
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                ref = self.tokenizer.decode(labels, skip_special_tokens=True)
                all_refs.append(ref)
        
        # Format text
        all_preds = [self.format_text(pred) for pred in all_preds]
        all_refs = [self.format_text(ref) for ref in all_refs]
        
        # Save predictions
        if output_file:
            self.save_predictions(all_preds, all_refs, output_file)
        
        return all_preds, all_refs
    
    @staticmethod
    def format_text(text: str) -> str:
        """Format text for evaluation.
        
        Args:
            text (str): The text to format.
            
        Returns:
            str: Formatted text.
        """
        import re
        # Remove special tokens and clean text
        text = re.sub(r'<unk>|<pad>|\</s>|<pad>\*|[^\w\s,.;:!?()\-]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def save_predictions(self, predictions: List[str], 
                        references: List[str], 
                        output_file: str):
        """Save predictions to a file.
        
        Args:
            predictions (List[str]): Predicted texts.
            references (List[str]): Reference texts.
            output_file (str): Path to save predictions.
        """
        logger.info(f"Saving predictions to {output_file}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({'predictions': predictions, 'references': references})
        
        # Save to file
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        elif output_file.endswith('.json'):
            df.to_json(output_file, orient='records', force_ascii=False, indent=4)
        else:
            # Default to CSV
            df.to_csv(f"{output_file}.csv", index=False, encoding='utf-8-sig')
    
    # def compute_metrics(self, predictions: List[str], 
    #                    references: List[str]) -> Dict[str, float]:
    #     """Compute metrics for predictions.
        
    #     Args:
    #         predictions (List[str]): Predicted texts.
    #         references (List[str]): Reference texts.
            
    #     Returns:
    #         Dict[str, float]: Metrics.
    #     """
    #     if self.metrics_fn is None:
    #         logger.warning("No metrics function provided. Cannot compute metrics.")
    #         return {}
        
    #     metrics = self.metrics_fn(predictions, references)
    #     logger.info(f"Metrics: {metrics}")
        
    #     return metrics

    def compute_metrics(self, predictions: List[str], 
                    references: List[str]) -> Dict[str, float]:
        """Compute metrics for predictions."""
        if self.metrics_fn is None:
            # Thay vì sử dụng metrics_fn, sử dụng các hàm trong metrics.py
            from src.evaluation.metrics import calculate_metrics, calculate_bleu
            
            # Tính toán các loại metrics
            bleu_scores = calculate_bleu(predictions, references)
            other_metrics = calculate_metrics(predictions, references)
            
            # Kết hợp kết quả
            metrics = {**bleu_scores, **other_metrics}
            logger.info(f"Metrics: {metrics}")
            return metrics
        else:
            metrics = self.metrics_fn(predictions, references)
            logger.info(f"Metrics: {metrics}")
            return metrics
    
    def save_metrics(self, metrics: Dict[str, float], 
                    output_file: str):
        """Save metrics to a file.
        
        Args:
            metrics (Dict[str, float]): Metrics to save.
            output_file (str): Path to save metrics.
        """
        logger.info(f"Saving metrics to {output_file}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

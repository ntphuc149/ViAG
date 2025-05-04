"""Model configuration for the ViAG project."""

import os
import logging
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)

logger = logging.getLogger(__name__)

class ModelConfig:
    """Model configuration for the ViAG project.
    
    This class handles model, tokenizer, and training arguments configuration.
    """
    
    def __init__(self, model_name: str, output_dir: str, 
                 training_args: Optional[Dict[str, Any]] = None):
        """Initialize the ModelConfig.
        
        Args:
            model_name (str): Name or path of the model.
            output_dir (str): Output directory for training results.
            training_args (Dict[str, Any], optional): Training arguments. Defaults to None.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.training_args = training_args or {}
        
        self.tokenizer = None
        self.model = None
        self.data_collator = None
    
    def load_tokenizer(self):
        """Load tokenizer.
        
        Returns:
            AutoTokenizer: Loaded tokenizer.
        """
        logger.info(f"Loading tokenizer: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_name
            )
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_model(self):
        """Load model.
        
        Returns:
            AutoModelForSeq2SeqLM: Loaded model.
        """
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name
            )
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_data_collator(self):
        """Create data collator.
        
        Returns:
            DataCollatorForSeq2Seq: Created data collator.
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        if self.model is None:
            self.load_model()
        
        logger.info("Creating data collator")
        
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            return_tensors='pt'
        )
        return self.data_collator
    
    def get_training_arguments(self):
        """Get training arguments.
        
        Returns:
            Seq2SeqTrainingArguments: Training arguments.
        """
        # Default training arguments
        default_args = {
            "output_dir": self.output_dir,
            "do_train": True,
            "do_eval": True,
            "num_train_epochs": 5,
            "learning_rate": 3e-5,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "predict_with_generate": True,
            "group_by_length": True,
            "save_total_limit": 1,
            "gradient_accumulation_steps": 16,
            "evaluation_strategy": "steps"
        }
        
        # Override defaults with user-provided arguments
        args = {**default_args, **self.training_args}
        
        logger.info(f"Training arguments: {args}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        return Seq2SeqTrainingArguments(**args)
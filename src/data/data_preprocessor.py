"""Data preprocessing utilities for the ViAG project."""

import logging
from typing import Dict, List, Optional, Union, Callable
import nltk
import spacy
from transformers import PreTrainedTokenizer
from datasets import Dataset

logger = logging.getLogger(__name__)

try:
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

try:
    # Load Vietnamese Spacy model
    nlp = spacy.load('vi_core_news_lg')
except Exception as e:
    logger.warning(f"Failed to load Spacy model: {e}. Using en_core_web_sm as fallback.")
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        logger.error("Failed to load any Spacy model.")
        nlp = None

class DataPreprocessor:
    """Data preprocessor for the ViAG project.
    
    This class handles data preprocessing for training and evaluation.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 max_input_length: int = 1024, 
                 max_target_length: int = 256):
        """Initialize the DataPreprocessor.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to use.
            max_input_length (int, optional): Maximum input length. Defaults to 1024.
            max_target_length (int, optional): Maximum target length. Defaults to 256.
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess function for the dataset.
        
        Args:
            examples (Dict[str, List]): Examples to preprocess.
            
        Returns:
            Dict[str, List]: Preprocessed examples.
        """
        # Determine padding side
        pad_on_right = self.tokenizer.padding_side == 'right'
        
        # Tokenize inputs
        inputs = self.tokenizer(
            examples['question' if pad_on_right else 'context'],
            examples['context' if pad_on_right else 'question'],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=self.max_input_length,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            examples['answer'], 
            max_length=self.max_target_length, 
            truncation=True, 
            padding=True
        )
        
        # Set labels
        inputs['labels'] = labels['input_ids']
        
        return inputs
    
    def preprocess_dataset(self, dataset: Dataset, 
                          num_proc: int = 8) -> Dataset:
        """Preprocess a dataset.
        
        Args:
            dataset (Dataset): Dataset to preprocess.
            num_proc (int, optional): Number of processes to use. Defaults to 8.
            
        Returns:
            Dataset: Preprocessed dataset.
        """
        logger.info(f"Preprocessing dataset with {num_proc} processes")
        
        # Verify dataset has required columns
        required_columns = ['context', 'question', 'answer']
        for col in required_columns:
            if col not in dataset.column_names:
                raise ValueError(f"Dataset missing required column: {col}")
        
        # Apply preprocessing function
        processed_dataset = dataset.map(
            self.preprocess_function,
            remove_columns=['context', 'question', 'answer'],
            num_proc=num_proc
        )
        
        # Set format for PyTorch
        processed_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        return processed_dataset
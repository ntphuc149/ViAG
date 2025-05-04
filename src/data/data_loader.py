"""Data loading utilities for the ViAG project."""

import os
import logging
import pandas as pd
from typing import Optional, Dict, List, Union
from datasets import Dataset

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for the ViAG project.
    
    This class handles loading and preparing data for training and evaluation.
    """
    
    def __init__(self, train_path: Optional[str] = None, 
                 val_path: Optional[str] = None, 
                 test_path: Optional[str] = None):
        """Initialize the DataLoader.
        
        Args:
            train_path (str, optional): Path to the training data.
            val_path (str, optional): Path to the validation data.
            test_path (str, optional): Path to the test data.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def load_data(self) -> Dict[str, Dataset]:
        """Load data from CSV files.
        
        Returns:
            Dict[str, Dataset]: Dictionary with train, val, and test datasets.
        """
        datasets_dict = {}
        
        if self.train_path and os.path.exists(self.train_path):
            logger.info(f"Loading training data from {self.train_path}")
            df_train = pd.read_csv(self.train_path)
            self.train_data = Dataset.from_pandas(df_train, preserve_index=False)
            datasets_dict['train'] = self.train_data
        
        if self.val_path and os.path.exists(self.val_path):
            logger.info(f"Loading validation data from {self.val_path}")
            df_val = pd.read_csv(self.val_path)
            self.val_data = Dataset.from_pandas(df_val, preserve_index=False)
            datasets_dict['validation'] = self.val_data
        
        if self.test_path and os.path.exists(self.test_path):
            logger.info(f"Loading test data from {self.test_path}")
            df_test = pd.read_csv(self.test_path)
            self.test_data = Dataset.from_pandas(df_test, preserve_index=False)
            datasets_dict['test'] = self.test_data
            
        return datasets_dict
    
    def split_train_validation(self, 
                              val_size: float = 0.1, 
                              random_state: int = 42) -> Dict[str, Dataset]:
        """Split training data into train and validation sets.
        
        Args:
            val_size (float, optional): Size of the validation set. Defaults to 0.1.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            
        Returns:
            Dict[str, Dataset]: Dictionary with train and validation datasets.
        """
        if self.train_data is None:
            if not self.train_path:
                raise ValueError("Training data path not specified.")
            self.load_data()
            
        if self.train_data is None:
            raise ValueError("Failed to load training data.")
            
        logger.info(f"Splitting train data into train and validation sets (val_size={val_size})")
        df_train = self.train_data.to_pandas()
        
        # Split data
        val_size_int = int(len(df_train) * val_size)
        df_val = df_train.sample(n=val_size_int, random_state=random_state)
        df_train = df_train.drop(index=df_val.index)
        
        logger.info(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}")
        
        # Convert back to Dataset
        self.train_data = Dataset.from_pandas(df_train, preserve_index=False)
        self.val_data = Dataset.from_pandas(df_val, preserve_index=False)
        
        return {
            'train': self.train_data,
            'validation': self.val_data
        }
    
    def verify_data_format(self, required_columns: List[str] = None) -> bool:
        """Verify that the data has the required columns.
        
        Args:
            required_columns (List[str], optional): List of required columns. 
                Defaults to ['context', 'question', 'answer'].
                
        Returns:
            bool: True if data has required columns, False otherwise.
        """
        if required_columns is None:
            required_columns = ['context', 'question', 'answer']
            
        datasets_to_check = []
        if self.train_data is not None:
            datasets_to_check.append(('train', self.train_data))
        if self.val_data is not None:
            datasets_to_check.append(('validation', self.val_data))
        if self.test_data is not None:
            datasets_to_check.append(('test', self.test_data))
            
        for name, dataset in datasets_to_check:
            columns = dataset.column_names
            missing_columns = [col for col in required_columns if col not in columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in {name} data: {missing_columns}")
                return False
                
        return True
    
    def get_dataset_sizes(self) -> Dict[str, int]:
        """Get the sizes of the datasets.
        
        Returns:
            Dict[str, int]: Dictionary with dataset sizes.
        """
        sizes = {}
        
        if self.train_data is not None:
            sizes['train'] = len(self.train_data)
        if self.val_data is not None:
            sizes['validation'] = len(self.val_data)
        if self.test_data is not None:
            sizes['test'] = len(self.test_data)
            
        return sizes
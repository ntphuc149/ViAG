"""Helper functions for the ViAG project."""

import os
import json
import logging
import torch
from pathlib import Path
from dotenv import load_dotenv

def setup_logging(log_file=None, level=logging.INFO):
    """Set up logging configuration.
    
    Args:
        log_file (str, optional): Path to the log file. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_config(config_path):
    """Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def save_config(config, config_path):
    """Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary.
        config_path (str): Path to save the configuration file.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def check_required_files(files_list):
    """Check if all required files exist.
    
    Args:
        files_list (list): List of file paths to check.
        
    Returns:
        bool: True if all files exist, False otherwise.
    """
    for file_path in files_list:
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            return False
    return True

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check if required environment variables are set
    required_vars = []
    
    if os.getenv('WANDB_API_KEY'):
        logging.info("Found WANDB API key.")
    else:
        logging.warning("WANDB API key not found. Wandb logging will be disabled.")
    
    if os.getenv('HF_TOKEN'):
        logging.info("Found Hugging Face token.")
    else:
        logging.warning("Hugging Face token not found. Some features may be limited.")
    
    return True

def check_gpu():
    """Check if GPU is available.
    
    Returns:
        bool: True if GPU is available, False otherwise.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"GPU is available. Found {device_count} device(s).")
        for i in range(device_count):
            logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        logging.warning("No GPU available. Training will be slow.")
        return False

def create_directory_if_not_exists(directory_path):
    """Create directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory.
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)
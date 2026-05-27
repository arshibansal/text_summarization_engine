"""
General utility functions for the FLAN-T5 Summarization Project.
"""

import os
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
from config import load_config

config = load_config("config.yaml")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (slower training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed set to {seed}")


def get_device() -> str:
    """
    Get the best available device.
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("⚠️ Using CPU (training will be slower)")
    
    return device


def create_directories():
    """
    Create all necessary project directories.
    """
    directories = [
        config["training"]["output_dir"],
        "./logs",
        "./outputs/models",
        "./outputs/predictions",
        "./outputs/results",
        "./data/processed",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created successfully.")


def setup_logging(log_file: Optional[str] = None):
    """
    Setup logging configuration.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"./logs/training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed.")
    return logger


def count_parameters(model):
    """
    Count trainable parameters in a model.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    
    return trainable


def format_time(seconds: float) -> str:
    """
    Convert seconds to human readable time format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_model_size_mb(model) -> float:
    """
    Get model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return round(size_mb, 2)
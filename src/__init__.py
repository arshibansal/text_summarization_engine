"""
FLAN-T5 Abstractive Summarization Project
Source package initialization.
"""

__version__ = "0.1.0"
__author__ = "Arshi"

# Package-level imports for convenient access
from .data.data_loader import load_dataset, preprocess_function
from .model.model import load_model, load_tokenizer
from .utils.helpers import set_seed, get_device
from .utils.logging import setup_logging

__all__ = [
    # Data
    "load_dataset",
    "preprocess_function",
    
    # Model
    "load_model",
    "load_tokenizer",
    
    # Utils
    "set_seed",
    "get_device",
    "setup_logging",
    
    # Version
    "__version__",
]
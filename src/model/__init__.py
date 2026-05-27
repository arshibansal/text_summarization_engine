"""
Model module for FLAN-T5 Abstractive Summarization.
Contains model loading, trainer, and related utilities.
"""

from .model import (
    load_model,
    load_tokenizer,
    load_model_for_training,
    SummarizationModel,
)

from .trainer import (
    FLANT5Trainer,
    compute_metrics,
    custom_train_loop,
)

__all__ = [
    # Model loading
    "load_model",
    "load_tokenizer",
    "load_model_for_training",
    "SummarizationModel",
    
    # Training
    "FLANT5Trainer",
    "compute_metrics",
    "custom_train_loop",
]
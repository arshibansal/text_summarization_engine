"""
Model loading utilities for FLAN-T5 summarization.
"""

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5TokenizerFast,
)
from typing import Tuple, Optional
from config import load_config

config = load_config("config.yaml")


def load_tokenizer(model_name: Optional[str] = None) -> T5TokenizerFast:
    """
    Load FLAN-T5 tokenizer.
    
    Args:
        model_name: Model name or path
    
    Returns:
        FLAN-T5 tokenizer
    """
    model_name = model_name or config["model"]["name"]
    
    print(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=config["model"].get("cache_dir"),
        use_fast=True
    )
    
    # FLAN-T5 doesn't have a pad token by default, so we set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer loaded successfully")
    return tokenizer


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None
) -> Tuple[AutoModelForSeq2SeqLM, T5TokenizerFast]:
    """
    Load FLAN-T5 model and tokenizer.
    
    Args:
        model_name: Model name or local path
        device: Device to load model on ('cuda', 'cpu', etc.)
        torch_dtype: Data type for model weights
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or config["model"]["name"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine torch dtype
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name} on {device} ({torch_dtype})")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        cache_dir=config["model"].get("cache_dir"),
        trust_remote_code=True
    )

    tokenizer = load_tokenizer(model_name)

    # Move to device if not using device_map
    if device != "cuda" or not hasattr(model, "hf_device_map"):
        model = model.to(device)

    print("✅ Model loaded successfully!")
    print(f"   Model parameters: {model.num_parameters():,}")

    return model, tokenizer


def load_model_for_training(model_name: Optional[str] = None):
    """
    Load model optimized for training (full precision + gradient checkpointing).
    """
    model_name = model_name or config["model"]["name"]

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Better for training stability
        cache_dir=config["model"].get("cache_dir"),
    )

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    tokenizer = load_tokenizer(model_name)

    return model, tokenizer


# Optional: Simple Model Wrapper
class SummarizationModel:
    """
    Wrapper class for easier inference and management.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model, self.tokenizer = load_model(model_name)
        self.device = self.model.device

    def summarize(self, text: str, **kwargs) -> str:
        """Generate summary for a single text."""
        from .inference import generate_summary  # Will create later
        return generate_summary(self.model, self.tokenizer, text, **kwargs)
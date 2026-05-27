"""
Configuration package for FLAN-T5 Summarization Project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).parent.parent.resolve()

# Config paths
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
PROCESSED_DATA_DIR = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
MODEL_OUTPUT_DIR = BASE_DIR / os.getenv("MODEL_OUTPUT_DIR", "outputs/models")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_config(config_name: str = "config.yaml"):
    """
    Load YAML configuration file.
    
    Args:
        config_name: Name of the config file (default: config.yaml)
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_inference_config():
    """Load inference specific configuration."""
    return load_config("inference_config.yaml")


# Expose commonly used variables
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUTPUT_DIR",
    "MODEL_OUTPUT_DIR",
    "LOG_DIR",
    "load_config",
    "load_inference_config",
]
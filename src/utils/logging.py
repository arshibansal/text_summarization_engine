"""
Logging utilities for the FLAN-T5 Summarization Project.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import load_config

config = load_config("config.yaml")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
        log_file: Path to log file. If None, auto-generates timestamped file.
        console: Whether to also log to console
    
    Returns:
        Logger instance
    """
    # Create logs directory
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"flan_t5_summarization_{timestamp}.log"
    else:
        log_file = Path(log_file)

    # Configure root logger
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        )
        logging.getLogger().addHandler(console_handler)

    logger = logging.getLogger("flan_t5_summarization")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "flan_t5_summarization") -> logging.Logger:
    """
    Get a logger with the project name prefix.
    """
    return logging.getLogger(f"flan_t5_summarization.{name}")


def log_training_info(logger: logging.Logger, trainer=None):
    """
    Log important training information.
    """
    logger.info("=" * 60)
    logger.info("🚀 FLAN-T5 SUMMARIZATION TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Dataset: {config['dataset']['name']} ({config['dataset']['version']})")
    logger.info(f"Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"Batch Size: {config['training']['per_device_train_batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Using FP16: {config['training']['fp16']}")
    logger.info("=" * 60)


# Example usage
if __name__ == "__main__":
    logger = setup_logging(log_level="INFO")
    logger.info("This is a test log message")
    logger.debug("This is a debug message (only visible if log_level=DEBUG)")
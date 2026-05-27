"""
Data Loader for CNN/DailyMail Dataset using Hugging Face datasets library.
"""

import os
from typing import Dict, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from config import load_config

config = load_config("config.yaml")


def load_dataset(
    dataset_name: str = None,
    version: str = None,
    subset: Optional[int] = None
) -> DatasetDict:
    """
    Load CNN/DailyMail dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset
        version: Dataset version
        subset: Use small subset for quick testing/debugging
    
    Returns:
        DatasetDict containing train, validation, test splits
    """
    dataset_name = dataset_name or config["dataset"]["name"]
    version = version or config["dataset"]["version"]

    print(f"Loading dataset: {dataset_name} ({version})...")

    dataset = load_dataset(
        dataset_name,
        version,
        trust_remote_code=True
    )

    # Use small subset for testing (optional)
    if subset:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(min(subset, len(dataset[split]))))

    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset.get('validation', []))}")
    print(f"Test samples: {len(dataset.get('test', []))}")

    return dataset


def preprocess_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_input_length: int = None,
    max_target_length: int = None
) -> Dict:
    """
    Preprocess function for tokenizing articles and summaries.
    """
    max_input_length = max_input_length or config["model"]["max_input_length"]
    max_target_length = max_target_length or config["model"]["max_target_length"]

    # FLAN-T5 prefix for summarization
    inputs = ["summarize: " + doc for doc in examples["article"]]

    # Tokenize inputs (articles)
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets (highlights/summaries)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    subset: Optional[int] = None
) -> DatasetDict:
    """
    Tokenize the entire dataset.
    """
    if subset:
        dataset = load_dataset(
            config["dataset"]["name"],
            config["dataset"]["version"],
            trust_remote_code=True
        )
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(min(subset, len(dataset[split]))))

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count() or 4,
        desc="Tokenizing dataset"
    )

    return tokenized_dataset


def get_dataloader(
    tokenized_dataset: DatasetDict,
    batch_size: int = None,
    shuffle: bool = True
):
    """Optional: Return PyTorch DataLoader (if not using Trainer)."""
    from torch.utils.data import DataLoader

    batch_size = batch_size or config["training"]["per_device_train_batch_size"]

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloader
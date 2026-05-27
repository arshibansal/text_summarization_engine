"""
Preprocessing utilities for CNN/DailyMail dataset for FLAN-T5 summarization.
"""

import os
from typing import Dict, Optional
from datasets import DatasetDict
from transformers import PreTrainedTokenizer
from config import load_config

config = load_config("config.yaml")


def preprocess_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_input_length: Optional[int] = None,
    max_target_length: Optional[int] = None
) -> Dict:
    """
    Tokenize articles and summaries for FLAN-T5 model.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: FLAN-T5 tokenizer
        max_input_length: Maximum length for input articles
        max_target_length: Maximum length for target summaries
    
    Returns:
        Tokenized inputs and labels
    """
    max_input_length = max_input_length or config["model"]["max_input_length"]
    max_target_length = max_target_length or config["model"]["max_target_length"]

    # Add task prefix for FLAN-T5
    inputs = ["summarize: " + article for article in examples["article"]]

    # Tokenize the input (articles)
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors=None  # Keep as lists for datasets map
    )

    # Tokenize the target summaries
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_tokenized_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    subset: Optional[int] = None,
    num_proc: Optional[int] = None
) -> DatasetDict:
    """
    Apply preprocessing to the entire dataset.
    
    Args:
        dataset: Raw CNN/DailyMail dataset
        tokenizer: FLAN-T5 tokenizer
        subset: Number of samples to use (for testing)
        num_proc: Number of processes for parallel processing
    
    Returns:
        Tokenized dataset ready for training
    """
    if subset:
        print(f"Using subset of {subset} samples per split for faster processing.")
        for split in list(dataset.keys()):
            if split in dataset:
                dataset[split] = dataset[split].select(range(min(subset, len(dataset[split]))))

    num_proc = num_proc or (os.cpu_count() or 4)

    print("Tokenizing dataset... This may take some time.")

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="Tokenizing CNN/DailyMail dataset"
    )

    print(f"✅ Tokenization completed!")
    print(f"   Train samples: {len(tokenized_dataset['train'])}")
    print(f"   Validation samples: {len(tokenized_dataset.get('validation', []))}")

    return tokenized_dataset


def filter_long_articles(
    dataset: DatasetDict,
    max_length: int = 1024
) -> DatasetDict:
    """
    Optional: Filter out very long articles to reduce noise.
    """
    def is_reasonable_length(example):
        return len(example["article"].split()) <= max_length // 2  # rough word count

    filtered_dataset = dataset.filter(is_reasonable_length)
    print(f"Filtered dataset: {len(filtered_dataset['train'])} training samples remaining.")
    
    return filtered_dataset
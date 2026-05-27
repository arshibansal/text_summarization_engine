"""
Inference module for FLAN-T5 Abstractive Summarization.
Handles model inference, summarization pipeline, and utilities.
"""

from .pipeline import (
    generate_summary,
    batch_summarize,
    SummarizationPipeline,
)

from .utils import (
    clean_text,
    truncate_article,
    post_process_summary,
    print_summary_example,
    load_examples,
    get_generation_config,
)

__all__ = [
    # Main inference functions
    "generate_summary",
    "batch_summarize",
    "SummarizationPipeline",
    
    # Utility functions
    "clean_text",
    "truncate_article",
    "post_process_summary",
    "print_summary_example",
    "load_examples",
    "get_generation_config",
]
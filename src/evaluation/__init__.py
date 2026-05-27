"""
Evaluation module for FLAN-T5 Abstractive Summarization.
Contains metrics computation and evaluation utilities.
"""

from .metrics import (
    compute_rouge,
    compute_bertscore,
    evaluate_summaries,
    evaluate_dataset,
)

__all__ = [
    # Core evaluation functions
    "compute_rouge",
    "compute_bertscore",
    "evaluate_summaries",
    "evaluate_dataset",
]
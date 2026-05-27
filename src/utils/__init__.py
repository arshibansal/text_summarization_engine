"""
Utilities module for FLAN-T5 Abstractive Summarization Project.
Contains helper functions, logging, and visualization tools.
"""

from .helpers import (
    set_seed,
    get_device,
    create_directories,
    setup_logging as setup_logging_helpers,
    count_parameters,
    format_time,
    get_model_size_mb,
)

from .logging import (
    setup_logging,
    get_logger,
    log_training_info,
)

from .visualizations import (
    plot_training_curves,
    plot_rouge_scores,
    compare_summaries,
    save_training_history,
)

__all__ = [
    # General Helpers
    "set_seed",
    "get_device",
    "create_directories",
    "count_parameters",
    "format_time",
    "get_model_size_mb",
    
    # Logging
    "setup_logging",
    "get_logger",
    "log_training_info",
    
    # Visualizations
    "plot_training_curves",
    "plot_rouge_scores",
    "compare_summaries",
    "save_training_history",
    
    # Legacy compatibility
    "setup_logging_helpers",   # if someone imported from helpers
]
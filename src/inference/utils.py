"""
Utility functions for the Inference module.
Helper functions for summarization, text processing, and evaluation.
"""

import re
from typing import List, Dict, Optional
from config import load_inference_config

inf_config = load_inference_config()


def clean_text(text: str) -> str:
    """
    Basic text cleaning for input articles.
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might confuse the model
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    return text.strip()


def truncate_article(text: str, max_words: int = 400) -> str:
    """
    Truncate very long articles to prevent exceeding max input length.
    """
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text


def post_process_summary(summary: str) -> str:
    """
    Advanced post-processing for generated summaries.
    """
    if not summary:
        return ""
    
    # Remove common artifacts
    summary = summary.strip()
    
    # Remove "Summary:" or "summarize:" prefixes if they appear
    summary = re.sub(r'^(summarize|summary|article summary)[:\-]\s*', '', summary, flags=re.IGNORECASE)
    
    # Fix spacing
    summary = re.sub(r'\s+', ' ', summary)
    
    # Capitalize first letter if needed
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    # Remove trailing punctuation issues
    summary = re.sub(r'\s+([.,!?])$', r'\1', summary)
    
    return summary.strip()


def print_summary_example(article: str, summary: str, reference: Optional[str] = None):
    """
    Pretty print article, generated summary, and reference (if available).
    """
    print("=" * 80)
    print("📝 ARTICLE (truncated):")
    print("-" * 40)
    print(truncate_article(article, 150))
    print("\n" + "=" * 80)
    
    print("✅ GENERATED SUMMARY:")
    print("-" * 40)
    print(summary)
    print("=" * 80)
    
    if reference:
        print("📌 REFERENCE SUMMARY:")
        print("-" * 40)
        print(reference)
        print("=" * 80)


def load_examples() -> List[Dict]:
    """
    Load sample examples from inference config for quick testing.
    """
    examples = inf_config.get("examples", [])
    return [{"text": ex} for ex in examples] if examples else []


def get_generation_config(overrides: Optional[Dict] = None) -> Dict:
    """
    Get generation parameters with optional overrides.
    """
    base_config = inf_config["generation"].copy()
    if overrides:
        base_config.update(overrides)
    return base_config
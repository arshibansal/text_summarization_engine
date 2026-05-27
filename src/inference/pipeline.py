"""
Inference Pipeline for FLAN-T5 Summarization.
Handles text summarization with proper generation settings.
"""

import torch
from typing import List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from config import load_inference_config, load_config

inf_config = load_inference_config()
train_config = load_config("config.yaml")


def generate_summary(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    num_beams: Optional[int] = None,
    **kwargs
) -> str:
    """
    Generate summary for a single article.
    
    Args:
        model: FLAN-T5 model
        tokenizer: FLAN-T5 tokenizer
        text: Input article text
        max_length: Maximum summary length
        min_length: Minimum summary length
        num_beams: Number of beams for beam search
    
    Returns:
        Generated summary
    """
    max_length = max_length or inf_config["generation"]["max_length"]
    min_length = min_length or inf_config["generation"]["min_length"]
    num_beams = num_beams or inf_config["generation"]["num_beams"]

    # Add FLAN-T5 prompt prefix
    input_text = f"summarize: {text}"

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=inf_config["input"]["max_input_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=inf_config["generation"]["length_penalty"],
            early_stopping=inf_config["generation"]["early_stopping"],
            no_repeat_ngram_size=inf_config["generation"]["no_repeat_ngram_size"],
            temperature=inf_config["generation"].get("temperature", 0.7),
            top_p=inf_config["generation"].get("top_p", 0.9),
            top_k=inf_config["generation"].get("top_k", 50),
            do_sample=inf_config["generation"]["do_sample"],
            num_return_sequences=inf_config["generation"]["num_return_sequences"],
            **kwargs
        )

    # Decode the output
    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=inf_config["output"]["skip_special_tokens"],
        clean_up_tokenization_spaces=inf_config["output"]["clean_up_tokenization_spaces"]
    )

    # Post-processing
    summary = post_process_summary(summary)
    return summary


def batch_summarize(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    Generate summaries for multiple texts in batches.
    """
    summaries = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_summaries = []
        
        for text in batch:
            summary = generate_summary(model, tokenizer, text, **kwargs)
            batch_summaries.append(summary)
        
        summaries.extend(batch_summaries)
        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} articles")
    
    return summaries


def post_process_summary(summary: str) -> str:
    """
    Clean up the generated summary.
    """
    # Remove any leftover prompt artifacts
    if summary.lower().startswith("summarize:"):
        summary = summary[len("summarize:"):].strip()
    
    # Basic cleaning
    summary = summary.strip()
    
    if inf_config["post_processing"].get("strip_whitespace", True):
        summary = " ".join(summary.split())
    
    # Remove bullet points if requested
    if inf_config["post_processing"].get("remove_bullets", True):
        summary = summary.replace("• ", "").replace("- ", "")
    
    return summary


class SummarizationPipeline:
    """
    High-level pipeline class for easy usage.
    """
    def __init__(self, model_name: Optional[str] = None):
        from src.model.model import load_model
        self.model, self.tokenizer = load_model(model_name)
        self.device = self.model.device

    def summarize(self, text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Main method to generate summary/summaries."""
        if isinstance(text, list):
            return batch_summarize(self.model, self.tokenizer, text, **kwargs)
        else:
            return generate_summary(self.model, self.tokenizer, text, **kwargs)

    def save(self, save_path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
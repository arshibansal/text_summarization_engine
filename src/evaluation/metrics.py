"""
Evaluation metrics for Abstractive Summarization using FLAN-T5.
Supports ROUGE, BERTScore, and other metrics.
"""

import evaluate
import numpy as np
from typing import Dict, List, Optional
from datasets import Dataset
from config import load_config

config = load_config("config.yaml")


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        rouge_types: List of ROUGE variants to compute
    
    Returns:
        Dictionary of ROUGE scores
    """
    rouge_types = rouge_types or config["evaluation"].get("rouge_types", 
                                                         ["rouge1", "rouge2", "rougeL", "rougeLsum"])
    
    rouge = evaluate.load("rouge")
    
    result = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=rouge_types,
        use_stemmer=True,
        use_aggregator=True
    )
    
    # Convert to percentage and round
    return {k: round(v * 100, 4) for k, v in result.items()}


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en"
) -> Dict[str, float]:
    """
    Compute BERTScore (semantic similarity).
    """
    try:
        bertscore = evaluate.load("bertscore")
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang=lang,
            model_type="microsoft/deberta-xlarge-mnli"
        )
        
        return {
            "bertscore_precision": round(np.mean(result["precision"]) * 100, 4),
            "bertscore_recall": round(np.mean(result["recall"]) * 100, 4),
            "bertscore_f1": round(np.mean(result["f1"]) * 100, 4),
        }
    except Exception as e:
        print(f"⚠️ BERTScore computation failed: {e}")
        return {"bertscore_f1": 0.0}


def evaluate_summaries(
    predictions: List[str],
    references: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of summaries.
    
    Args:
        predictions: Generated summaries
        references: Ground truth summaries
        metrics: List of metrics to compute
    
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = metrics or config["evaluation"].get("metrics", ["rouge", "bertscore"])
    
    results = {}
    
    # Compute ROUGE
    if "rouge" in metrics or "rouge" in [m.lower() for m in metrics]:
        rouge_scores = compute_rouge(predictions, references)
        results.update(rouge_scores)
    
    # Compute BERTScore
    if "bertscore" in metrics or "bertscore" in [m.lower() for m in metrics]:
        bert_scores = compute_bertscore(predictions, references)
        results.update(bert_scores)
    
    # Add average ROUGE (common practice)
    if "rouge1" in results and "rouge2" in results and "rougeL" in results:
        results["rouge_avg"] = round(
            (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3, 4
        )
    
    return results


def evaluate_dataset(
    model,
    tokenizer,
    test_dataset: Dataset,
    max_samples: Optional[int] = 500,
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluate model on a dataset split.
    """
    from src.inference.pipeline import generate_summary  # Will be created later
    
    print(f"Evaluating on {min(max_samples, len(test_dataset))} samples...")
    
    # Take subset if specified
    if max_samples and max_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(max_samples))
    
    predictions = []
    references = []
    
    for example in test_dataset:
        article = example["article"]
        reference = example["highlights"]
        
        # Generate summary
        summary = generate_summary(
            model, 
            tokenizer, 
            article,
            max_length=config["generation"]["max_length"],
            min_length=config["generation"]["min_length"]
        )
        
        predictions.append(summary)
        references.append(reference)
    
    # Compute metrics
    metrics = evaluate_summaries(predictions, references)
    
    print("\n📊 Evaluation Results:")
    for metric, score in metrics.items():
        print(f"   {metric.upper():<20}: {score:.4f}")
    
    return metrics
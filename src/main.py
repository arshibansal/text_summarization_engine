"""
Main entry point for FLAN-T5 Abstractive Summarization Project.
Supports training, evaluation, and inference modes.
"""

import argparse
import sys
from pathlib import Path

from config import load_config
from src.utils import (
    set_seed,
    get_device,
    create_directories,
    setup_logging,
    get_logger,
)
from src.data.data_loader import load_dataset
from src.data.preprocessing import prepare_tokenized_dataset
from src.model.model import load_model, load_tokenizer
from src.model.trainer import FLANT5Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLAN-T5 Abstractive Text Summarization Engine"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train",
        choices=["train", "evaluate", "infer", "test"],
        help="Mode to run: train, evaluate, infer, or test"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="Model name or path"
    )
    
    parser.add_argument(
        "--subset", 
        type=int, 
        default=None,
        help="Use subset of data for quick testing"
    )
    
    parser.add_argument(
        "--text", 
        type=str, 
        default=None,
        help="Input text for inference mode"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    create_directories()
    logger = setup_logging()
    
    logger.info("🚀 Starting FLAN-T5 Summarization Engine")
    logger.info(f"Mode: {args.mode}")
    
    config = load_config("config.yaml")
    
    if args.mode == "train":
        logger.info("📚 Loading dataset...")
        raw_dataset = load_dataset(subset=args.subset)
        
        logger.info("🔤 Loading tokenizer and model...")
        tokenizer = load_tokenizer(args.model_name)
        model, _ = load_model(args.model_name)  # Only for reference
        
        logger.info("🧹 Tokenizing dataset...")
        tokenized_dataset = prepare_tokenized_dataset(raw_dataset, tokenizer, subset=args.subset)
        
        logger.info("🏋️ Starting training...")
        trainer = FLANT5Trainer(args.model_name)
        trainer.train(tokenized_dataset)
        
        logger.info("✅ Training completed successfully!")
        
    elif args.mode == "infer":
        if not args.text:
            logger.error("❌ Please provide --text for inference mode")
            sys.exit(1)
            
        from src.inference.pipeline import SummarizationPipeline
        
        logger.info("🔄 Loading inference pipeline...")
        pipeline = SummarizationPipeline(args.model_name)
        
        logger.info("✍️ Generating summary...")
        summary = pipeline.summarize(args.text)
        
        print("\n" + "="*80)
        print("📝 INPUT TEXT:")
        print("-" * 40)
        print(args.text[:500] + "..." if len(args.text) > 500 else args.text)
        print("\n✅ GENERATED SUMMARY:")
        print("-" * 40)
        print(summary)
        print("="*80)
        
    elif args.mode == "evaluate":
        logger.info("📊 Starting evaluation...")
        from src.evaluation.metrics import evaluate_dataset
        from src.model.model import load_model, load_tokenizer
        
        model, tokenizer = load_model(args.model_name)
        raw_dataset = load_dataset()
        
        results = evaluate_dataset(
            model, 
            tokenizer, 
            raw_dataset["test"], 
            max_samples=200
        )
        logger.info(f"Evaluation completed with results: {results}")
        
    elif args.mode == "test":
        # Quick test mode
        logger.info("🧪 Running quick test...")
        dataset = load_dataset(subset=100)
        print(f"✅ Test successful! Loaded {len(dataset['train'])} training samples.")
        
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
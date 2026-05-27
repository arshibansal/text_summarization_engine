"""
Custom Trainer for FLAN-T5 Summarization using Accelerate + Transformers.
"""

import torch
from typing import Dict, Optional
from datasets import DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load
import numpy as np
from config import load_config
from .model import load_model_for_training, load_tokenizer

config = load_config("config.yaml")


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE scores for evaluation."""
    rouge = load("rouge")
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects newlines
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Return average ROUGE scores
    return {
        "rouge1": result["rouge1"] * 100,
        "rouge2": result["rouge2"] * 100,
        "rougeL": result["rougeL"] * 100,
        "rougeLsum": result["rougeLsum"] * 100,
    }


class FLANT5Trainer:
    """
    Trainer class for FLAN-T5 summarization model.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config["model"]["name"]
        self.model, self.tokenizer = load_model_for_training(self.model_name)
        self.config = config

    def train(self, tokenized_dataset: DatasetDict):
        """
        Train the model using Seq2SeqTrainer.
        """
        print("Initializing trainer...")

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            fp16=self.config["training"]["fp16"],
            bf16=self.config["training"]["bf16"],
            logging_steps=self.config["training"]["logging_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            save_steps=self.config["training"]["save_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            metric_for_best_model=self.config["training"]["metric_for_best_model"],
            greater_is_better=True,
            predict_with_generate=True,
            seed=self.config["training"]["seed"],
            report_to="wandb" if self.config["logging"]["use_wandb"] else "none",
            run_name=self.config["logging"].get("wandb_run_name"),
        )

        # Data collator (handles padding)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            return_tensors="pt"
        )

        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, self.tokenizer),
        )

        print("🚀 Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model(self.config["training"]["output_dir"])
        self.tokenizer.save_pretrained(self.config["training"]["output_dir"])
        
        print(f"✅ Training completed! Model saved to: {self.config['training']['output_dir']}")
        return trainer


# Optional: Custom training loop using Accelerate (more control)
def custom_train_loop(model, tokenizer, tokenized_dataset):
    """
    Future-ready custom training loop using Accelerate.
    Use this if you want more flexibility than HF Trainer.
    """
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    
    accelerator = Accelerator()
    
    # Prepare dataloaders
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=True,
    )
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, torch.optim.AdamW(model.parameters(), lr=3e-5), train_dataloader
    )
    
    # Training loop here...
    print("Custom Accelerate training loop initialized.")
    # Implementation can be expanded later
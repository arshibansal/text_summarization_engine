"""
Visualization utilities for FLAN-T5 Summarization Project.
Includes training curves, evaluation metrics, and summary comparison plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from config import load_config

config = load_config("config.yaml")

sns.set_style("whitegrid")


def plot_training_curves(
    trainer_logs: List[Dict],
    save_path: Optional[str] = None,
    title: str = "FLAN-T5 Training Progress"
):
    """
    Plot training and evaluation loss curves.
    """
    if not trainer_logs:
        print("⚠️ No training logs available to plot.")
        return

    steps = [log['step'] for log in trainer_logs if 'loss' in log]
    train_loss = [log['loss'] for log in trainer_logs if 'loss' in log]
    eval_loss = [log.get('eval_loss') for log in trainer_logs if 'eval_loss' in log]

    plt.figure(figsize=(12, 6))
    
    plt.plot(steps[:len(train_loss)], train_loss, label='Training Loss', marker='o', linewidth=2)
    
    if eval_loss and len(eval_loss) > 0:
        eval_steps = steps[::config["training"]["eval_steps"]][:len(eval_loss)]
        plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='s', linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curve saved to: {save_path}")
    else:
        plt.show()


def plot_rouge_scores(
    rouge_scores: Dict[str, float],
    save_path: Optional[str] = None
):
    """
    Plot ROUGE scores as a bar chart.
    """
    plt.figure(figsize=(10, 6))
    
    metrics = list(rouge_scores.keys())
    values = list(rouge_scores.values())
    
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.title('ROUGE Scores Comparison', fontsize=16)
    plt.ylabel('Score (%)')
    plt.ylim(0, max(values) * 1.15)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROUGE scores plot saved to: {save_path}")
    else:
        plt.show()


def compare_summaries(
    article: str,
    reference: str,
    generated: str,
    save_path: Optional[str] = None
):
    """
    Create a visual comparison of reference vs generated summary.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    # Article
    axs[0].text(0.02, 0.5, article[:800] + "..." if len(article) > 800 else article,
                va='center', ha='left', wrap=True, fontsize=10)
    axs[0].set_title('📝 Original Article (truncated)', fontsize=14)
    axs[0].axis('off')
    
    # Reference
    axs[1].text(0.02, 0.5, reference, va='center', ha='left', wrap=True, fontsize=11)
    axs[1].set_title('📌 Reference Summary', fontsize=14, color='green')
    axs[1].axis('off')
    
    # Generated
    axs[2].text(0.02, 0.5, generated, va='center', ha='left', wrap=True, fontsize=11)
    axs[2].set_title('🤖 Generated Summary (FLAN-T5)', fontsize=14, color='blue')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Summary comparison saved to: {save_path}")
    else:
        plt.show()


def save_training_history(history, save_dir: str = "./outputs/results"):
    """
    Save training metrics to CSV for later analysis.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(history)
    csv_path = f"{save_dir}/training_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Training history saved to: {csv_path}")
    
    return df
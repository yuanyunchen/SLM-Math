"""
Training visualization and analysis script.
Generates loss curves, training reports, and compares different training runs.

Usage:
    python -m utils.visualize_training --checkpoint_dir results/sft_checkpoints/run_name_1120_1234
    python -m utils.visualize_training --compare run1/ run2/ run3/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_loss_history(checkpoint_dir: Path) -> Optional[pd.DataFrame]:
    """Load loss history from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        DataFrame with loss history or None if not found
    """
    loss_file = checkpoint_dir / "loss_history.csv"
    
    if not loss_file.exists():
        logger.warning(f"Loss history file not found: {loss_file}")
        return None
    
    try:
        df = pd.read_csv(loss_file)
        logger.info(f"Loaded {len(df)} loss records from {checkpoint_dir.name}")
        return df
    except Exception as e:
        logger.error(f"Error loading loss history: {e}")
        return None


def plot_loss_curves(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Training Loss Curves"
):
    """Plot training and evaluation loss curves.
    
    Args:
        df: DataFrame with loss history
        output_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Train loss
    train_data = df[df['train_loss'].notna()]
    if not train_data.empty:
        ax1.plot(train_data['step'], train_data['train_loss'], 
                label='Train Loss', color='blue', linewidth=2)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Eval loss
    eval_data = df[df['eval_loss'].notna()]
    if not eval_data.empty:
        ax2.plot(eval_data['step'], eval_data['eval_loss'], 
                label='Eval Loss', color='orange', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Evaluation Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved loss curves to {output_path}")
    plt.close()


def plot_learning_rate(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Learning Rate Schedule"
):
    """Plot learning rate schedule.
    
    Args:
        df: DataFrame with loss history
        output_path: Path to save the plot
        title: Plot title
    """
    lr_data = df[df['learning_rate'].notna()]
    
    if lr_data.empty:
        logger.warning("No learning rate data found")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(lr_data['step'], lr_data['learning_rate'], 
            color='green', linewidth=2)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved learning rate plot to {output_path}")
    plt.close()


def generate_training_report(
    checkpoint_dir: Path,
    df: pd.DataFrame
) -> str:
    """Generate a text report of training metrics.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        df: DataFrame with loss history
    
    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Training Report")
    lines.append("=" * 80)
    lines.append(f"Checkpoint: {checkpoint_dir.name}")
    lines.append("")
    
    # Load config if available
    config_file = checkpoint_dir / "training_config.yaml"
    if config_file.exists():
        lines.append("Configuration:")
        with open(config_file, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            lines.append(f"  Model: {config.get('model', {}).get('name', 'N/A')}")
            lines.append(f"  LoRA: {config.get('lora', {}).get('enabled', 'N/A')}")
            if config.get('lora', {}).get('enabled'):
                lines.append(f"    - Rank: {config['lora'].get('r', 'N/A')}")
                lines.append(f"    - Alpha: {config['lora'].get('alpha', 'N/A')}")
            lines.append(f"  Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
            lines.append(f"  Batch Size: {config.get('training', {}).get('per_device_train_batch_size', 'N/A')}")
            lines.append(f"  Gradient Accumulation: {config.get('training', {}).get('gradient_accumulation_steps', 'N/A')}")
            lines.append("")
    
    # Training metrics
    lines.append("Training Metrics:")
    lines.append(f"  Total steps: {df['step'].max()}")
    lines.append(f"  Final epoch: {df['epoch'].max():.2f}")
    
    # Train loss statistics
    train_data = df[df['train_loss'].notna()]
    if not train_data.empty:
        lines.append(f"\nTrain Loss:")
        lines.append(f"  Initial: {train_data['train_loss'].iloc[0]:.4f}")
        lines.append(f"  Final: {train_data['train_loss'].iloc[-1]:.4f}")
        lines.append(f"  Best: {train_data['train_loss'].min():.4f} (step {train_data.loc[train_data['train_loss'].idxmin(), 'step']:.0f})")
        lines.append(f"  Mean: {train_data['train_loss'].mean():.4f}")
    
    # Eval loss statistics
    eval_data = df[df['eval_loss'].notna()]
    if not eval_data.empty:
        lines.append(f"\nEval Loss:")
        lines.append(f"  Initial: {eval_data['eval_loss'].iloc[0]:.4f}")
        lines.append(f"  Final: {eval_data['eval_loss'].iloc[-1]:.4f}")
        lines.append(f"  Best: {eval_data['eval_loss'].min():.4f} (step {eval_data.loc[eval_data['eval_loss'].idxmin(), 'step']:.0f})")
        lines.append(f"  Mean: {eval_data['eval_loss'].mean():.4f}")
    
    # Learning rate
    lr_data = df[df['learning_rate'].notna()]
    if not lr_data.empty:
        lines.append(f"\nLearning Rate:")
        lines.append(f"  Initial: {lr_data['learning_rate'].iloc[0]:.2e}")
        lines.append(f"  Final: {lr_data['learning_rate'].iloc[-1]:.2e}")
    
    # Load final metrics if available
    metrics_file = checkpoint_dir / "train_results.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        lines.append(f"\nFinal Training Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    return report


def compare_runs(checkpoint_dirs: List[Path], output_dir: Path):
    """Compare multiple training runs.
    
    Args:
        checkpoint_dirs: List of checkpoint directories
        output_dir: Directory to save comparison plots
    """
    logger.info(f"Comparing {len(checkpoint_dirs)} training runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all loss histories
    all_data = []
    for checkpoint_dir in checkpoint_dirs:
        df = load_loss_history(checkpoint_dir)
        if df is not None:
            df['run_name'] = checkpoint_dir.name
            all_data.append(df)
    
    if not all_data:
        logger.error("No valid loss histories found")
        return
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(range(len(all_data)))
    
    for i, df in enumerate(all_data):
        run_name = df['run_name'].iloc[0]
        color = colors[i]
        
        # Train loss
        train_data = df[df['train_loss'].notna()]
        if not train_data.empty:
            ax1.plot(train_data['step'], train_data['train_loss'], 
                    label=run_name, color=color, linewidth=2, alpha=0.8)
        
        # Eval loss
        eval_data = df[df['eval_loss'].notna()]
        if not eval_data.empty:
            ax2.plot(eval_data['step'], eval_data['eval_loss'], 
                    label=run_name, color=color, linewidth=2, marker='o', 
                    markersize=4, alpha=0.8)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Evaluation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()
    
    # Generate comparison table
    comparison_data = []
    for df in all_data:
        run_name = df['run_name'].iloc[0]
        train_data = df[df['train_loss'].notna()]
        eval_data = df[df['eval_loss'].notna()]
        
        comparison_data.append({
            'Run': run_name,
            'Final Train Loss': train_data['train_loss'].iloc[-1] if not train_data.empty else None,
            'Best Train Loss': train_data['train_loss'].min() if not train_data.empty else None,
            'Final Eval Loss': eval_data['eval_loss'].iloc[-1] if not eval_data.empty else None,
            'Best Eval Loss': eval_data['eval_loss'].min() if not eval_data.empty else None,
            'Total Steps': df['step'].max(),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Saved comparison table to {comparison_file}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("Training Runs Comparison")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize and analyze SFT training results")
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Compare multiple checkpoint directories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as checkpoint_dir)'
    )
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        checkpoint_dirs = [Path(d) for d in args.compare]
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/training_comparison")
        compare_runs(checkpoint_dirs, output_dir)
    
    elif args.checkpoint_dir:
        # Single checkpoint visualization
        checkpoint_dir = Path(args.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            return
        
        logger.info(f"Analyzing checkpoint: {checkpoint_dir}")
        
        # Load loss history
        df = load_loss_history(checkpoint_dir)
        
        if df is None:
            logger.error("Could not load loss history")
            return
        
        # Determine output directory
        output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plot_loss_curves(
            df,
            output_dir / "loss_curves.png",
            title=f"Training Loss Curves - {checkpoint_dir.name}"
        )
        
        plot_learning_rate(
            df,
            output_dir / "learning_rate.png",
            title=f"Learning Rate Schedule - {checkpoint_dir.name}"
        )
        
        # Generate report
        report = generate_training_report(checkpoint_dir, df)
        
        # Save report
        report_file = output_dir / "training_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved training report to {report_file}")
        
        # Print report
        print("\n" + report + "\n")
    
    else:
        parser.print_help()
        logger.error("Please specify either --checkpoint_dir or --compare")


if __name__ == "__main__":
    main()


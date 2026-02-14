"""Checkpoint utilities for saving and loading models."""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    name: str = None,
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoints
        name: Custom checkpoint name (default: checkpoint_{epoch}.pt)
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if name is None:
        name = f"checkpoint_{epoch:04d}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, name)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
        device: Device to load onto
    
    Returns:
        Dictionary with checkpoint information (epoch, metrics, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
    }


def save_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_value: float,
    checkpoint_dir: str,
    metric_name: str = 'loss',
) -> str:
    """
    Save best checkpoint based on metric.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metric_value: Metric value for this checkpoint
        checkpoint_dir: Directory to save checkpoints
        metric_name: Name of the metric being optimized
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = save_checkpoint(
        model, optimizer, epoch,
        {metric_name: metric_value},
        checkpoint_dir,
        name=f"best_{metric_name}.pt"
    )
    return checkpoint_path

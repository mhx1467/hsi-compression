"""Data transforms for HSI preprocessing."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable


class Normalize(nn.Module):
    """Normalize tensor to [0, 1] range using min-max normalization."""
    
    def __init__(self, per_channel: bool = False):
        super().__init__()
        self.per_channel = per_channel
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize tensor.
        
        Args:
            x: Input tensor of shape (C, H, W) or (B, C, H, W)
        
        Returns:
            Normalized tensor
        """
        if self.per_channel:
            # Normalize per channel
            if x.dim() == 3:  # (C, H, W)
                x_min = x.view(x.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
                x_max = x.view(x.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
            else:  # (B, C, H, W)
                x_min = x.view(x.shape[0], x.shape[1], -1).min(dim=2)[0].view(x.shape[0], x.shape[1], 1, 1)
                x_max = x.view(x.shape[0], x.shape[1], -1).max(dim=2)[0].view(x.shape[0], x.shape[1], 1, 1)
        else:
            # Normalize globally
            x_min = x.min()
            x_max = x.max()
        
        return (x - x_min) / (x_max - x_min + 1e-7)


class Standardize(nn.Module):
    """Standardize tensor using z-score normalization."""
    
    def __init__(self, per_channel: bool = False):
        super().__init__()
        self.per_channel = per_channel
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Standardize tensor.
        
        Args:
            x: Input tensor
        
        Returns:
            Standardized tensor
        """
        if self.per_channel:
            if x.dim() == 3:  # (C, H, W)
                mean = x.view(x.shape[0], -1).mean(dim=1).view(-1, 1, 1)
                std = x.view(x.shape[0], -1).std(dim=1).view(-1, 1, 1)
            else:  # (B, C, H, W)
                mean = x.view(x.shape[0], x.shape[1], -1).mean(dim=2).view(x.shape[0], x.shape[1], 1, 1)
                std = x.view(x.shape[0], x.shape[1], -1).std(dim=2).view(x.shape[0], x.shape[1], 1, 1)
        else:
            mean = x.mean()
            std = x.std()
        
        return (x - mean) / (std + 1e-7)


class ToTensor:
    """Convert numpy array to tensor if not already."""
    
    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.float()


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


def get_default_transforms(model_type: str, normalize_input: bool = True) -> Callable:
    """
    Get default transforms for a model type.
    
    Args:
        model_type: 'lossless' or 'lossy'
        normalize_input: Whether to normalize input
    
    Returns:
        Composed transform
    """
    transforms = [ToTensor()]
    
    if normalize_input:
        if model_type == 'lossless':
            # For lossless, keep integer values or minimal normalization
            # Usually lossless compression works better with original values
            pass
        elif model_type == 'lossy':
            # For lossy, normalize to [0, 1]
            transforms.append(Normalize(per_channel=False))
    
    return Compose(transforms)

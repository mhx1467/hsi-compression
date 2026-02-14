"""Distortion losses for compression models."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: Original tensor
            y: Reconstructed tensor
        
        Returns:
            MSE loss scalar
        """
        return torch.mean((x - y) ** 2)


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss for hyperspectral data."""
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute spectral angle mapper (SAM) loss.
        
        Args:
            x: Original HSI tensor of shape (B, C, H, W)
            y: Reconstructed HSI tensor of shape (B, C, H, W)
        
        Returns:
            SAM loss scalar
        """
        # Reshape to (B*H*W, C) for per-pixel SAM
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        y_flat = y.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize to unit vectors
        x_norm = torch.nn.functional.normalize(x_flat, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(y_flat, p=2, dim=1)
        
        # Compute cosine similarity and clip to [-1, 1] to handle numerical errors
        cos_angle = torch.sum(x_norm * y_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # Compute spectral angle in radians and convert to degrees if needed
        angles = torch.acos(cos_angle)
        
        # Return mean angle in radians
        return torch.mean(angles)


class MSESAMLoss(nn.Module):
    """Weighted combination of MSE and SAM losses."""
    
    def __init__(self, mse_weight: float = 0.5, sam_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.sam_weight = sam_weight
        self.mse_loss = MSELoss()
        self.sam_loss = SAMLoss()
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: Original tensor
            y: Reconstructed tensor
        
        Returns:
            Weighted MSE + SAM loss
        """
        mse = self.mse_loss(x, y)
        sam = self.sam_loss(x, y)
        return self.mse_weight * mse + self.sam_weight * sam


class NegativeLogLikelihoodLoss(nn.Module):
    """Negative Log-Likelihood loss for lossless compression."""
    
    def __init__(self, distribution_type: str = 'gaussian'):
        super().__init__()
        self.distribution_type = distribution_type
    
    def forward(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """
        Compute NLL loss given predicted distribution parameters.
        
        Args:
            x: Original values
            mean: Predicted mean of distribution
            scale: Predicted scale/std of distribution
        
        Returns:
            NLL loss scalar
        """
        if self.distribution_type == 'gaussian':
            # Gaussian: NLL = -log(1/sqrt(2*pi*sigma^2)) - (x-mu)^2/(2*sigma^2)
            #         = 0.5*log(2*pi*sigma^2) + (x-mu)^2/(2*sigma^2)
            loss = 0.5 * torch.log(2 * torch.tensor(3.14159265) * scale ** 2) + (x - mean) ** 2 / (2 * scale ** 2)
        elif self.distribution_type == 'logistic':
            # Logistic: NLL = log(scale) + (x-mu)/scale + 2*log(1 + exp(-(x-mu)/scale))
            z = (x - mean) / scale
            loss = torch.log(scale) + z + 2 * torch.nn.functional.softplus(-z)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution_type}")
        
        return torch.mean(loss)

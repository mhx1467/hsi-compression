"""Rate estimation losses for lossy compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class RateDistortionLoss(nn.Module):
    """Combined Rate-Distortion loss for lossy compression."""
    
    def __init__(self, 
                 distortion_loss: nn.Module,
                 rate_weight: float = 0.01):
        """
        Args:
            distortion_loss: Loss function for distortion term (e.g., MSELoss)
            rate_weight: Weight/lambda for rate term in R-D tradeoff
        """
        super().__init__()
        self.distortion_loss = distortion_loss
        self.rate_weight = rate_weight
    
    def forward(self, 
                original: Tensor,
                reconstruction: Tensor,
                estimated_bpp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute R-D loss: Loss = D + lambda * R
        
        Args:
            original: Original input tensor
            reconstruction: Reconstructed tensor
            estimated_bpp: Estimated bits per pixel (from entropy model)
        
        Returns:
            Tuple of (total_loss, distortion, rate)
        """
        # Compute distortion
        distortion = self.distortion_loss(original, reconstruction)
        
        # Normalize rate by pixels
        rate = torch.mean(estimated_bpp)
        
        # Combined loss
        total_loss = distortion + self.rate_weight * rate
        
        return total_loss, distortion, rate


class EntropybottleneckBpp(nn.Module):
    """Estimate bits per pixel using entropy bottleneck."""
    
    def forward(self, 
                latent: Tensor,
                mean: Tensor,
                scale: Tensor) -> Tensor:
        """
        Estimate BPP using Gaussian entropy.
        
        Args:
            latent: Quantized latent representation (B, M, H, W)
            mean: Mean of Gaussian distribution (B, M, H, W)
            scale: Scale/std of Gaussian distribution (B, M, H, W)
        
        Returns:
            BPP tensor of shape (B, M, H, W)
        """
        # Gaussian entropy: H = 0.5*log(2*pi*e*sigma^2)
        # This is proportional to -log2(p(x)) in bits
        # Using natural log and converting to bits per dimension
        
        # Clamp scale to avoid log(0)
        scale = torch.clamp(scale, min=1e-5)
        
        # Compute negative log-likelihood (in nats)
        nll = 0.5 * torch.log(2 * torch.tensor(3.14159265, device=latent.device) * scale ** 2) + \
              (latent - mean) ** 2 / (2 * scale ** 2)
        
        # Convert nats to bits
        bpp = nll / torch.log(torch.tensor(2.0, device=latent.device))
        
        return bpp

"""Hyperprior network for entropy estimation in lossy compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


class HyperpriorEncoder(nn.Module):
    """CNN encoder for hyperprior side information."""
    
    def __init__(
        self,
        latent_channels: int = 192,
        channels: List[int] = None,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        """
        Args:
            latent_channels: Number of input latent channels
            channels: List of channel dimensions for each layer
            kernel_size: Convolution kernel size
            stride: Stride for downsampling
        """
        super().__init__()
        
        if channels is None:
            channels = [192, 192, 128]
        
        padding = kernel_size // 2
        
        layers = []
        in_ch = latent_channels
        
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())
            in_ch = out_ch
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode latent to side information.
        
        Args:
            x: Latent tensor of shape (B, M, H, W)
        
        Returns:
            Side information tensor of shape (B, C_z, H', W')
        """
        return self.layers(x)


class HyperpriorDecoder(nn.Module):
    """CNN decoder for entropy model parameter estimation."""
    
    def __init__(
        self,
        latent_channels: int = 192,
        z_channels: int = 128,
        kernel_size: int = 3,
        upsample_factor: int = 2,
    ):
        """
        Args:
            latent_channels: Number of latent channels
            z_channels: Number of hyperprior channels
            kernel_size: Convolution kernel size
            upsample_factor: Upsampling factor
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        # Two upsampling steps (stride 2 each, so 4x total)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(z_channels, z_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(z_channels, z_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        
        # Output: mean and scale for each latent channel
        self.mean = nn.Conv2d(z_channels, latent_channels, kernel_size, padding=padding)
        self.log_scale = nn.Conv2d(z_channels, latent_channels, kernel_size, padding=padding)
    
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decode side information to entropy model parameters.
        
        Args:
            z: Side information tensor of shape (B, C_z, H', W')
        
        Returns:
            Tuple of (mean, scale) tensors of shape (B, M, H, W)
        """
        x = self.upsample1(z)
        x = self.upsample2(x)
        
        mean = self.mean(x)
        log_scale = self.log_scale(x)
        scale = torch.exp(log_scale)
        scale = torch.clamp(scale, min=1e-3)
        
        return mean, scale


class Hyperprior(nn.Module):
    """Complete hyperprior module for entropy coding."""
    
    def __init__(
        self,
        latent_channels: int = 192,
        z_channels: int = 128,
        hyperprior_channels: List[int] = None,
        kernel_size: int = 3,
    ):
        """
        Args:
            latent_channels: Number of latent channels
            z_channels: Number of hyperprior channels
            hyperprior_channels: Channel dimensions for encoder
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        if hyperprior_channels is None:
            hyperprior_channels = [192, 192, 128]
        
        self.encoder = HyperpriorEncoder(
            latent_channels,
            hyperprior_channels,
            kernel_size,
        )
        
        self.decoder = HyperpriorDecoder(
            latent_channels,
            hyperprior_channels[-1],
            kernel_size,
        )
    
    def forward(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Estimate entropy model parameters for latent.
        
        Args:
            latent: Latent tensor of shape (B, M, H, W)
        
        Returns:
            Tuple of (mean, scale) tensors for entropy coding
        """
        z = self.encoder(latent)
        mean, scale = self.decoder(z)
        return mean, scale

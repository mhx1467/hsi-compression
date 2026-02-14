"""TCN layers for lossless hyperspectral compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class CausalDilatedConv1d(nn.Module):
    """1D Causal Dilated Convolution for spectral processing."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            dilation: Dilation factor
            dropout: Dropout probability
        """
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Calculate padding to maintain causality
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, L) where L is sequence length
        
        Returns:
            Output tensor of shape (B, C_out, L)
        """
        # Apply convolution
        y = self.conv(x)
        
        # Remove future information to maintain causality
        # Padding was applied on both sides, so we need to remove the right part
        padding_to_remove = (self.kernel_size - 1) * self.dilation
        if padding_to_remove > 0:
            y = y[:, :, :-padding_to_remove]
        
        y = self.dropout(y)
        return y


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv1 = CausalDilatedConv1d(
            channels, channels, kernel_size, dilation, dropout
        )
        self.conv2 = CausalDilatedConv1d(
            channels, channels, kernel_size, dilation, dropout
        )
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, L)
        
        Returns:
            Output tensor of shape (B, C, L)
        """
        residual = x
        
        # First conv
        y = self.conv1(x)
        y = self.relu(y)
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        
        # Second conv
        y = self.conv2(y)
        y = self.relu(y)
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        
        # Residual connection
        return residual + y


class SpectralTCN(nn.Module):
    """Spectral component using stacked TCN blocks with dilations."""
    
    def __init__(
        self,
        channels: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            channels: List of channel dimensions for each TCN block
            kernel_size: Convolution kernel size
            dilations: List of dilation factors (if None, use [1, 2, 4, 8, ...])
            dropout: Dropout probability
        """
        super().__init__()
        
        if dilations is None:
            dilations = [2 ** i for i in range(len(channels))]
        
        self.layers = nn.ModuleList()
        
        # First layer: 1 -> channels[0]
        self.layers.append(nn.Conv1d(1, channels[0], 1))
        self.layers.append(TCNBlock(channels[0], kernel_size, dilations[0], dropout))
        
        # Subsequent layers: channels[i] -> channels[i+1]
        for i in range(1, len(channels)):
            if channels[i] != channels[i-1]:
                self.layers.append(nn.Conv1d(channels[i-1], channels[i], 1))
            self.layers.append(TCNBlock(channels[i], kernel_size, dilations[i], dropout))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, L) for a single band
        
        Returns:
            Output tensor of shape (B, C_out, L)
        """
        for layer in self.layers:
            x = layer(x)
        
        return x


class SpatialConv(nn.Module):
    """Spatial convolution for capturing 2D patterns."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        return self.conv(x)


class DistributionHead(nn.Module):
    """Output layer for distribution parameters (mean and scale)."""
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
    ):
        """
        Args:
            in_channels: Number of input channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        # Output 2 channels: mean and log(scale)
        self.conv_mean = nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size//2)
        self.conv_scale = nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size//2)
    
    def forward(self, x: Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Tuple of (mean, scale) tensors of shape (B, 1, H, W)
        """
        mean = self.conv_mean(x)
        log_scale = self.conv_scale(x)
        scale = torch.exp(log_scale)  # Ensure positivity
        scale = torch.clamp(scale, min=1e-3)  # Prevent too small scales
        
        return mean, scale

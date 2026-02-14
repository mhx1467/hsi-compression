"""Mamba-based encoder for lossy hyperspectral compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple

# Note: mamba-ssm should be installed separately
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class SpectralMambaBlock(nn.Module):
    """Mamba block for spectral dimension processing."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: State dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        if Mamba is None:
            raise ImportError("mamba-ssm not installed. Install with: pip install mamba-ssm")
        
        self.mamba = Mamba(d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for spectral processing.
        
        Args:
            x: Input tensor of shape (B, L, D) where L is sequence length (num_pixels)
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class SpatialMambaBlock(nn.Module):
    """Vision Mamba block for spatial dimension processing."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        window_size: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: State dimension
            window_size: Local window size (e.g., 16x16 patches)
            dropout: Dropout probability
        """
        super().__init__()
        
        if Mamba is None:
            raise ImportError("mamba-ssm not installed. Install with: pip install mamba-ssm")
        
        self.window_size = window_size
        self.mamba = Mamba(d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for spatial processing.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Patch the image into local windows
        # Simplified: treat entire image as one sequence
        # In practice, you might want local window processing
        
        x_seq = x.reshape(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        residual = x_seq
        x_seq = self.norm(x_seq)
        x_seq = self.mamba(x_seq)
        x_seq = self.dropout(x_seq)
        
        x_out = residual + x_seq
        return x_out.transpose(1, 2).reshape(B, C, H, W)


class MambaEncoder(nn.Module):
    """Mamba-based encoder for hyperspectral compression."""
    
    def __init__(
        self,
        in_channels: int = 224,
        latent_channels: int = 192,
        spectral_d_model: int = 256,
        spectral_d_state: int = 16,
        spectral_n_layers: int = 4,
        spatial_d_model: int = 256,
        spatial_d_state: int = 16,
        spatial_n_layers: int = 4,
        spatial_window_size: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input spectral bands
            latent_channels: Number of latent channels
            spectral_d_model: Spectral Mamba model dimension
            spectral_d_state: Spectral Mamba state dimension
            spectral_n_layers: Number of spectral Mamba layers
            spatial_d_model: Spatial Mamba model dimension
            spatial_d_state: Spatial Mamba state dimension
            spatial_n_layers: Number of spatial Mamba layers
            spatial_window_size: Size of spatial processing windows
            dropout: Dropout probability
        """
        super().__init__()
        
        # Initial projection to model dimension
        self.input_proj = nn.Conv2d(in_channels, spectral_d_model, 1)
        
        # Spectral processing path
        self.spectral_mamba_layers = nn.ModuleList([
            SpectralMambaBlock(spectral_d_model, spectral_d_state, dropout)
            for _ in range(spectral_n_layers)
        ])
        
        # Spatial processing path
        self.spatial_mamba_layers = nn.ModuleList([
            SpatialMambaBlock(spectral_d_model, spatial_d_state, spatial_window_size, dropout)
            for _ in range(spatial_n_layers)
        ])
        
        # Feature fusion and bottleneck
        self.fusion = nn.Sequential(
            nn.Conv2d(spectral_d_model + spectral_d_model, spectral_d_model, 1),
            nn.ReLU(),
            nn.Conv2d(spectral_d_model, latent_channels, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Latent tensor of shape (B, M, H, W)
        """
        B, C, H, W = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, D_spectral, H, W)
        
        # Spectral path
        spectral_feat = x
        for layer in self.spectral_mamba_layers:
            # Reshape for sequence processing
            spec_seq = spectral_feat.reshape(B, -1, H*W).transpose(1, 2)  # (B, H*W, D)
            spec_seq = layer(spec_seq)
            spectral_feat = spec_seq.transpose(1, 2).reshape(B, -1, H, W)
        
        # Spatial path
        spatial_feat = x
        for layer in self.spatial_mamba_layers:
            spatial_feat = layer(spatial_feat)
        
        # Fusion
        fused = torch.cat([spectral_feat, spatial_feat], dim=1)
        latent = self.fusion(fused)
        
        return latent

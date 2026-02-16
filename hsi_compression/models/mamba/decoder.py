"""Mamba-based decoder for lossy hyperspectral compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def _check_mamba_available(function_name: str):
    """Helper to provide informative error message if mamba-ssm is not installed."""
    if Mamba is None:
        raise ImportError(
            f"Cannot create {function_name}: mamba-ssm is not installed.\n"
            "\nTo use Mamba-based models, install mamba-ssm:\n"
            "  pip install 'hsi-compression[mamba]'\n"
            "\nIf that fails due to CUDA compilation issues, ensure CUDA development tools are installed:\n"
            "  apt-get install nvidia-cuda-toolkit nvidia-cuda-dev\n"
            "  pip install mamba-ssm\n"
            "\nAlternatively, use the TCN model which doesn't require mamba-ssm:\n"
            "  python train.py --config hsi_compression/configs/models/tcn_lossless.yaml"
        )


class SpectralMambaBlock(nn.Module):
    """Mamba block for spectral dimension processing in decoder."""
    
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
        
        _check_mamba_available("SpectralMambaBlock")
        
        self.mamba = Mamba(d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for spectral processing.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence format
        residual = x
        x_seq = x.reshape(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        x_seq = self.norm(x_seq)
        x_seq = self.mamba(x_seq)
        x_seq = self.dropout(x_seq)
        
        # Reshape back to 2D
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        return residual + x


class SpatialMambaBlock(nn.Module):
    """Vision Mamba block for spatial dimension processing in decoder."""
    
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
        
        _check_mamba_available("SpatialMambaBlock")
        
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
        
        residual_2d = x
        x_seq = x.reshape(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        residual_seq = x_seq
        x_seq = self.norm(x_seq)
        x_seq = self.mamba(x_seq)
        x_seq = self.dropout(x_seq)
        
        x_seq = residual_seq + x_seq
        return x_seq.transpose(1, 2).reshape(B, C, H, W) + residual_2d


class MambaDecoder(nn.Module):
    """Mamba-based decoder for hyperspectral compression."""
    
    def __init__(
        self,
        latent_channels: int = 192,
        out_channels: int = 224,
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
            latent_channels: Number of latent channels
            out_channels: Number of output spectral bands
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
        
        _check_mamba_available("MambaDecoder")
        
        # Initial projection from latent to model dimension
        self.latent_proj = nn.Conv2d(latent_channels, spectral_d_model, 1)
        
        # Spectral Mamba layers (mirror of encoder)
        self.spectral_mamba_layers = nn.ModuleList([
            SpectralMambaBlock(spectral_d_model, spectral_d_state, dropout)
            for _ in range(spectral_n_layers)
        ])
        
        # Spatial Mamba layers (mirror of encoder)
        self.spatial_mamba_layers = nn.ModuleList([
            SpatialMambaBlock(spectral_d_model, spatial_d_state, spatial_window_size, dropout)
            for _ in range(spatial_n_layers)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(spectral_d_model + spectral_d_model, spectral_d_model, 1),
            nn.ReLU(),
        )
        
        # Upsampling and final reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(spectral_d_model, out_channels * 4, 1),
            nn.PixelShuffle(2) if out_channels >= 56 else nn.Identity(),
            nn.Conv2d(out_channels if out_channels >= 56 else out_channels * 4, out_channels, 3, padding=1),
        )
    
    def forward(self, latent: Tensor) -> Tensor:
        """
        Decode latent representation back to image space.
        
        Args:
            latent: Latent tensor of shape (B, M, H, W)
        
        Returns:
            Reconstructed image tensor of shape (B, C, H, W)
        """
        B, M, H, W = latent.shape
        
        # Project latent
        x = self.latent_proj(latent)
        
        # Spectral path (mirror of encoder)
        spectral_feat = x
        for layer in self.spectral_mamba_layers:
            spectral_feat = layer(spectral_feat)
        
        # Spatial path (mirror of encoder)
        spatial_feat = x
        for layer in self.spatial_mamba_layers:
            spatial_feat = layer(spatial_feat)
        
        # Fusion
        fused = torch.cat([spectral_feat, spatial_feat], dim=1)
        x = self.fusion(fused)
        
        # Reconstruct to original resolution and channels
        reconstruction = self.reconstruct(x)
        
        # Ensure output is in valid range
        reconstruction = torch.clamp(reconstruction, 0, 1)
        
        return reconstruction

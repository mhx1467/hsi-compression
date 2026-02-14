"""Mamba-based decoder for lossy hyperspectral compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


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
        
        if Mamba is None:
            raise ImportError("mamba-ssm not installed. Install with: pip install mamba-ssm")
        
        # Initial projection from latent to model dimension
        self.latent_proj = nn.Conv2d(latent_channels, spectral_d_model, 1)
        
        # Spectral Mamba layers (mirror of encoder)
        self.spectral_mamba_layers = nn.ModuleList([
            self._make_mamba_block(spectral_d_model, spectral_d_state, dropout)
            for _ in range(spectral_n_layers)
        ])
        
        # Spatial Mamba layers (mirror of encoder)
        self.spatial_mamba_layers = nn.ModuleList([
            self._make_spatial_mamba_block(spectral_d_model, spatial_d_state, spatial_window_size, dropout)
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
    
    @staticmethod
    def _make_mamba_block(d_model: int, d_state: int, dropout: float):
        """Create a Mamba block for spectral processing."""
        mamba = Mamba(d_model, d_state=d_state)
        norm = nn.LayerNorm(d_model)
        
        class MambaBlock(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                residual = x
                x = norm(x)
                x = mamba(x)
                x = nn.Dropout(dropout)(x)
                return residual + x
        
        return MambaBlock()
    
    @staticmethod
    def _make_spatial_mamba_block(d_model: int, d_state: int, window_size: int, dropout: float):
        """Create a Mamba block for spatial processing."""
        mamba = Mamba(d_model, d_state=d_state)
        norm = nn.LayerNorm(d_model)
        
        class SpatialMambaBlock(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                B, C, H, W = x.shape
                
                residual_2d = x
                x_seq = x.reshape(B, C, -1).transpose(1, 2)  # (B, H*W, C)
                
                residual_seq = x_seq
                x_seq = norm(x_seq)
                x_seq = mamba(x_seq)
                x_seq = nn.Dropout(dropout)(x_seq)
                
                x_seq = residual_seq + x_seq
                return x_seq.transpose(1, 2).reshape(B, C, H, W) + residual_2d
        
        return SpatialMambaBlock()
    
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

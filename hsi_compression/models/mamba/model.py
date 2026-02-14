"""Mamba-based lossy hyperspectral image compression model."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, List, Optional
import numpy as np
import io
import pickle

from ..base import LossyCompressor
from ..registry import register_model
from .encoder import MambaEncoder
from .decoder import MambaDecoder
from .hyperprior import Hyperprior


@register_model('mamba_lossy')
class MambaLossyCompressor(LossyCompressor):
    """
    Mamba-based lossy compression model for hyperspectral images.
    
    Uses dual-path Mamba encoders for spectral/spatial processing,
    with hyperprior for entropy estimation and rate-distortion optimization.
    """
    
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
        hyperprior_channels: List[int] = None,
        normalize_input: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            in_channels: Number of spectral bands
            latent_channels: Dimension of latent representation
            spectral_d_model: Spectral Mamba model dimension
            spectral_d_state: Spectral Mamba state dimension
            spectral_n_layers: Number of spectral Mamba layers
            spatial_d_model: Spatial Mamba model dimension
            spatial_d_state: Spatial Mamba state dimension
            spatial_n_layers: Number of spatial Mamba layers
            spatial_window_size: Size of spatial windows
            hyperprior_channels: Channel dimensions for hyperprior
            normalize_input: Whether to normalize input
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.normalize_input = normalize_input
        
        if hyperprior_channels is None:
            hyperprior_channels = [192, 192, 128]
        
        # Encoder (spectral and spatial Mamba paths)
        self.encoder = MambaEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            spectral_d_model=spectral_d_model,
            spectral_d_state=spectral_d_state,
            spectral_n_layers=spectral_n_layers,
            spatial_d_model=spatial_d_model,
            spatial_d_state=spatial_d_state,
            spatial_n_layers=spatial_n_layers,
            spatial_window_size=spatial_window_size,
            dropout=dropout,
        )
        
        # Decoder (mirror of encoder)
        self.decoder = MambaDecoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            spectral_d_model=spectral_d_model,
            spectral_d_state=spectral_d_state,
            spectral_n_layers=spectral_n_layers,
            spatial_d_model=spatial_d_model,
            spatial_d_state=spatial_d_state,
            spatial_n_layers=spatial_n_layers,
            spatial_window_size=spatial_window_size,
            dropout=dropout,
        )
        
        # Hyperprior for entropy estimation
        self.hyperprior = Hyperprior(
            latent_channels=latent_channels,
            z_channels=hyperprior_channels[-1],
            hyperprior_channels=hyperprior_channels,
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass for training.
        
        Args:
            x: Input HSI tensor of shape (B, C, H, W)
        
        Returns:
            Dictionary with latent, reconstruction, and entropy parameters
        """
        # Normalize if needed
        if self.normalize_input:
            x_norm = self._normalize(x)
        else:
            x_norm = x
        
        # Encode
        latent = self.encode(x_norm)
        
        # Hyperprior for entropy estimation
        entropy_mean, entropy_scale = self.hyperprior(latent)
        
        # Decode
        reconstruction = self.decode(latent)
        
        return {
            'latent': latent,
            'reconstruction': reconstruction,
            'entropy_mean': entropy_mean,
            'entropy_scale': entropy_scale,
        }
    
    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Latent tensor of shape (B, M, H, W)
        """
        latent = self.encoder(x)
        return latent
    
    def decode(self, latent: Tensor) -> Tensor:
        """
        Decode latent back to image space.
        
        Args:
            latent: Latent tensor of shape (B, M, H, W)
        
        Returns:
            Reconstruction of shape (B, C, H, W)
        """
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def compress(self, x: Tensor) -> bytes:
        """
        Compress input using the model.
        
        Args:
            x: Input tensor
        
        Returns:
            Compressed bitstream
        """
        # Placeholder: actual entropy coding would use torchac/compressai
        with torch.no_grad():
            latent = self.encode(self._normalize(x) if self.normalize_input else x)
        
        import io
        import pickle
        
        buffer = io.BytesIO()
        pickle.dump(latent.cpu().numpy(), buffer)
        return buffer.getvalue()
    
    def decompress(self, bitstream: bytes, shape: Tuple) -> Tensor:
        """
        Decompress bitstream back to tensor.
        
        Args:
            bitstream: Compressed bitstream
            shape: Expected output shape
        
        Returns:
            Decompressed tensor
        """
        import io
        import pickle
        
        buffer = io.BytesIO(bitstream)
        latent_data = pickle.load(buffer)
        latent = torch.from_numpy(latent_data).float()
        
        with torch.no_grad():
            reconstruction = self.decode(latent)
        
        return reconstruction
    
    def _normalize(self, x: Tensor) -> Tensor:
        """Normalize input to [0, 1] range."""
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)
        
        # Global normalization
        x_min = x_flat.min()
        x_max = x_flat.max()
        
        return (x - x_min) / (x_max - x_min + 1e-7)

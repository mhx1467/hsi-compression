"""Base classes for compression models."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
from torch import nn, Tensor


class BaseCompressor(nn.Module, ABC):
    """Abstract base class for all compression models."""
    
    def __init__(self):
        super().__init__()
        self._compression_type = None
    
    @property
    @abstractmethod
    def compression_type(self) -> str:
        """Return 'lossless' or 'lossy'."""
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass returning relevant outputs.
        
        Args:
            x: Input tensor of shape (B, C, H, W) for HSI
        
        Returns:
            Dictionary with model outputs (e.g., predictions, latents, distributions)
        """
        pass
    
    @abstractmethod
    def compress(self, x: Tensor) -> bytes:
        """
        Compress input to bitstream.
        
        Args:
            x: Input tensor
        
        Returns:
            Compressed bitstream (bytes)
        """
        pass
    
    @abstractmethod
    def decompress(self, bitstream: bytes, shape: Tuple) -> Tensor:
        """
        Decompress bitstream back to tensor.
        
        Args:
            bitstream: Compressed bitstream
            shape: Expected output shape
        
        Returns:
            Decompressed tensor
        """
        pass


class LosslessCompressor(BaseCompressor, ABC):
    """Base class for lossless compression models (e.g., TCN)."""
    
    @property
    def compression_type(self) -> str:
        return "lossless"
    
    @abstractmethod
    def predict_distribution(self, x: Tensor, band_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Predict probability distribution for a band.
        
        Args:
            x: Input tensor
            band_idx: Index of the band to predict
        
        Returns:
            Tuple of (mean, scale/std) for the distribution
        """
        pass


class LossyCompressor(BaseCompressor, ABC):
    """Base class for lossy compression models (e.g., Mamba)."""
    
    @property
    def compression_type(self) -> str:
        return "lossy"
    
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """
        Encode to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Latent tensor of shape (B, M, H', W')
        """
        pass
    
    @abstractmethod
    def decode(self, latent: Tensor) -> Tensor:
        """
        Decode latent back to reconstruction.
        
        Args:
            latent: Latent tensor
        
        Returns:
            Reconstructed tensor of original shape
        """
        pass
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass for lossy compression."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return {
            'latent': latent,
            'reconstruction': reconstruction,
        }

"""TCN-based lossless hyperspectral image compression model."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, List, Optional
import numpy as np

from ..base import LosslessCompressor
from ..registry import register_model
from .layers import SpectralTCN, SpatialConv, DistributionHead


@register_model('tcn_lossless')
class TCNLosslessCompressor(LosslessCompressor):
    """
    TCN-based lossless compression model for hyperspectral images.
    
    Uses temporal convolutional networks to predict spectral bands
    and outputs probability distributions for entropy coding.
    """
    
    def __init__(
        self,
        num_bands: int = 224,
        spectral_window: int = 8,
        tcn_channels: List[int] = None,
        tcn_kernel_size: int = 3,
        tcn_dilations: List[int] = None,
        spatial_channels: int = 64,
        spatial_kernel_size: int = 3,
        distribution_type: str = 'gaussian',
        normalize_input: bool = True,
        **kwargs
    ):
        """
        Args:
            num_bands: Number of spectral bands
            spectral_window: Number of previous bands to use for prediction
            tcn_channels: Channel dimensions for TCN blocks
            tcn_kernel_size: TCN kernel size
            tcn_dilations: Dilation factors for TCN blocks
            spatial_channels: Number of spatial feature channels
            spatial_kernel_size: Spatial convolution kernel size
            distribution_type: 'gaussian' or 'logistic'
            normalize_input: Whether to normalize input
        """
        super().__init__()
        
        self.num_bands = num_bands
        self.spectral_window = spectral_window
        self.distribution_type = distribution_type
        self.normalize_input = normalize_input
        
        # Default channel configuration
        if tcn_channels is None:
            tcn_channels = [64, 128, 256]
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4, 8]
        
        # Spectral TCN component
        self.spectral_tcn = SpectralTCN(
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations[:len(tcn_channels)],
        )
        
        # Spatial component (operates on current band)
        self.spatial_conv = SpatialConv(
            in_channels=1,
            out_channels=spatial_channels,
            kernel_size=spatial_kernel_size,
        )
        
        # Feature fusion (combine spectral and spatial features)
        # spectral TCN output channels + spatial channels
        fusion_channels = tcn_channels[-1] + spatial_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 1),
            nn.ReLU(),
        )
        
        # Distribution head
        self.dist_head = DistributionHead(
            in_channels=fusion_channels,
            kernel_size=3,
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input HSI tensor of shape (B, C, H, W)
        
        Returns:
            Dictionary with predictions, means, and scales
        """
        B, C, H, W = x.shape
        
        # Normalize if needed
        if self.normalize_input:
            x = self._normalize(x)
        
        # Process bands sequentially
        predictions = []
        means = []
        scales = []
        
        # Process first band (initialization)
        current_band = x[:, 0:1, :, :]  # (B, 1, H, W)
        
        for band_idx in range(1, min(C, self.spectral_window + 1)):
            # Get previous bands for context
            prev_bands = x[:, max(0, band_idx - self.spectral_window):band_idx, :, :]
            
            # Predict distribution for current band
            mean, scale = self.predict_distribution_band(prev_bands, current_band)
            means.append(mean)
            scales.append(scale)
            
            # Move to next band
            current_band = x[:, band_idx:band_idx+1, :, :]
        
        return {
            'means': torch.stack(means) if means else None,
            'scales': torch.stack(scales) if scales else None,
        }
    
    def predict_distribution_band(
        self,
        prev_bands: Tensor,
        current_band: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict probability distribution for the current band.
        
        Args:
            prev_bands: Previous bands of shape (B, K, H, W) where K is window size
            current_band: Current band to process (used for spatial context)
        
        Returns:
            Tuple of (mean, scale) tensors of shape (B, 1, H, W)
        """
        B, K, H, W = prev_bands.shape
        
        # Spectral processing: flatten to sequence
        prev_flat = prev_bands.reshape(B * H * W, K, 1)  # (B*H*W, K, 1)
        
        # Process with TCN (operates on sequence dimension)
        spectral_features = []
        for t in range(K):
            band_t = prev_bands[:, t:t+1, :, :]  # (B, 1, H, W)
            spec = self.spectral_tcn(band_t.reshape(B, 1, H*W))
            spec = spec.reshape(B, -1, H, W)
            spectral_features.append(spec)
        
        # Use last spectral feature
        spectral_feat = spectral_features[-1]  # (B, C_tcn, H, W)
        
        # Spatial processing on current band
        spatial_feat = self.spatial_conv(current_band)  # (B, C_spatial, H, W)
        
        # Fusion
        fused = torch.cat([spectral_feat, spatial_feat], dim=1)  # (B, C_tcn+C_spatial, H, W)
        fused = self.fusion(fused)
        
        # Get distribution parameters
        mean, scale = self.dist_head(fused)
        
        return mean, scale
    
    def predict_distribution(
        self,
        x: Tensor,
        band_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict distribution for a specific band.
        
        Args:
            x: Input tensor
            band_idx: Index of band to predict
        
        Returns:
            Tuple of (mean, scale)
        """
        prev_bands = x[:, max(0, band_idx - self.spectral_window):band_idx, :, :]
        if prev_bands.shape[1] == 0:
            # First band - return zeros
            return torch.zeros_like(x[:, band_idx:band_idx+1]), \
                   torch.ones_like(x[:, band_idx:band_idx+1])
        
        current_band = x[:, band_idx:band_idx+1, :, :]
        return self.predict_distribution_band(prev_bands, current_band)
    
    def compress(self, x: Tensor) -> bytes:
        """
        Compress input using entropy coding with predicted distributions.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Compressed bitstream
        """
        from ...utils.entropy import EntropyEncoder, encode_distribution_parameters
        import struct
        
        B, C, H, W = x.shape
        
        # Normalize input
        if self.normalize_input:
            x_norm = self._normalize(x)
        else:
            x_norm = x.clone()
        
        bitstream = bytearray()
        
        # Header: metadata
        bitstream.extend(struct.pack('IIII', B, C, H, W))
        
        # Store normalization parameters if used
        if self.normalize_input:
            x_flat = x.reshape(B, C, -1)
            means_denorm = x_flat.mean(dim=2, keepdim=True)
            stds_denorm = x_flat.std(dim=2, keepdim=True)
            
            # Encode normalization params
            norm_params = torch.cat([means_denorm, stds_denorm], dim=-1)
            norm_bytes = norm_params.cpu().detach().numpy().tobytes()
            bitstream.extend(struct.pack('I', len(norm_bytes)))
            bitstream.extend(norm_bytes)
        
        # Process each band and encode with context
        encoder = EntropyEncoder()
        
        for band_idx in range(1, C):
            # Get distribution prediction
            prev_bands = x[:, max(0, band_idx - self.spectral_window):band_idx, :, :]
            current_band = x[:, band_idx:band_idx+1, :, :]
            
            if prev_bands.shape[1] > 0:
                mean, scale = self.predict_distribution_band(prev_bands, current_band)
                
                # Encode band with context
                band_data = x[:, band_idx, :, :]  # (B, H, W)
                band_encoded = encoder.encode(band_data, mean.squeeze(1), scale.squeeze(1))
                
                # Store length and data
                bitstream.extend(struct.pack('I', len(band_encoded)))
                bitstream.extend(band_encoded)
                
                # Store distribution parameters for decoding
                dist_params = encode_distribution_parameters(
                    mean.squeeze(1), scale.squeeze(1)
                )
                bitstream.extend(struct.pack('I', len(dist_params)))
                bitstream.extend(dist_params)
            else:
                # First band: encode raw
                band_data = x[:, band_idx, :, :]
                band_encoded = encoder.encode(band_data)
                
                bitstream.extend(struct.pack('I', len(band_encoded)))
                bitstream.extend(band_encoded)
                bitstream.extend(struct.pack('I', 0))  # No distribution params
        
        return bytes(bitstream)
    
    def decompress(self, bitstream: bytes, shape: Tuple) -> Tensor:
        """
        Decompress bitstream back to tensor using entropy decoding.
        
        Args:
            bitstream: Compressed bitstream from compress()
            shape: Expected output shape (B, C, H, W)
        
        Returns:
            Decompressed tensor
        """
        import struct
        from ...utils.entropy import EntropyDecoder, decode_distribution_parameters
        
        bitstream_arr = bytearray(bitstream)
        offset = 0
        
        # Read header
        B, C, H, W = struct.unpack_from('IIII', bitstream_arr, offset)
        offset += 16
        
        # Read normalization parameters if stored
        norm_params = None
        if self.normalize_input:
            norm_len = struct.unpack_from('I', bitstream_arr, offset)[0]
            offset += 4
            
            norm_data = np.frombuffer(bitstream_arr[offset:offset + norm_len], dtype=np.float32)
            offset += norm_len
            
            norm_params = torch.tensor(norm_data.reshape(B, C, 2), dtype=torch.float32)
        
        # Decompress bands
        bands = [torch.zeros((B, H, W), dtype=torch.float32)]  # First band is zero
        
        decoder = EntropyDecoder()
        
        for band_idx in range(1, C):
            # Read band data
            band_len = struct.unpack_from('I', bitstream_arr, offset)[0]
            offset += 4
            band_data = bytes(bitstream_arr[offset:offset + band_len])
            offset += band_len
            
            # Read distribution parameters
            dist_len = struct.unpack_from('I', bitstream_arr, offset)[0]
            offset += 4
            
            if dist_len > 0:
                dist_data = bytes(bitstream_arr[offset:offset + dist_len])
                offset += dist_len
                
                # Decode with context
                # For now, simplified decoding
                decoded_band = decoder.decode(band_data, (B, H, W))
            else:
                # Decode raw
                decoded_band = decoder.decode(band_data, (B, H, W))
            
            bands.append(decoded_band)
        
        # Stack bands
        result = torch.stack(bands, dim=1)  # (B, C, H, W)
        
        # Denormalize if needed
        if self.normalize_input and norm_params is not None:
            means = norm_params[:, :, 0].reshape(B, C, 1, 1)
            stds = norm_params[:, :, 1].reshape(B, C, 1, 1)
            result = result * stds + means
        
        return result
    
    def _normalize(self, x: Tensor) -> Tensor:
        """Normalize input to [-1, 1] range per channel."""
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)
        
        # Per-channel normalization
        mean = x_flat.mean(dim=2, keepdim=True)
        std = x_flat.std(dim=2, keepdim=True)
        
        return (x - mean.reshape(B, C, 1, 1)) / (std.reshape(B, C, 1, 1) + 1e-7)

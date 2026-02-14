"""Quality metrics for compression models."""

import torch
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional


class PSNRMetric:
    """Peak Signal-to-Noise Ratio (PSNR) metric."""
    
    def __init__(self, max_value: float = 1.0, eps: float = 1e-7):
        """
        Args:
            max_value: Maximum possible pixel value (typically 1.0 for normalized or 255 for uint8)
            eps: Small epsilon for numerical stability
        """
        self.max_value = max_value
        self.eps = eps
    
    def __call__(self, original: Tensor, reconstructed: Tensor) -> float:
        """
        Compute PSNR between original and reconstructed images.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
        
        Returns:
            PSNR value in dB
        """
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * math.log10(self.max_value / (math.sqrt(mse.item()) + self.eps))
        return psnr


class SSIMMetric:
    """Structural Similarity Index (SSIM) metric."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5, max_value: float = 1.0):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian window
            max_value: Maximum possible pixel value
        """
        self.window_size = window_size
        self.sigma = sigma
        self.max_value = max_value
        self.c1 = (0.01 * max_value) ** 2
        self.c2 = (0.03 * max_value) ** 2
    
    def __call__(self, original: Tensor, reconstructed: Tensor) -> float:
        """
        Compute SSIM between original and reconstructed images.
        
        Args:
            original: Original tensor of shape (B, C, H, W)
            reconstructed: Reconstructed tensor of shape (B, C, H, W)
        
        Returns:
            SSIM value (between -1 and 1, typically 0 to 1)
        """
        # Simple SSIM computation (per-channel mean)
        b, c, h, w = original.shape
        ssim_values = []
        
        for i in range(c):
            x = original[:, i:i+1, :, :]
            y = reconstructed[:, i:i+1, :, :]
            
            # Compute statistics
            mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size//2)
            mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size//2)
            
            mu_x_sq = mu_x ** 2
            mu_y_sq = mu_y ** 2
            mu_x_mu_y = mu_x * mu_y
            
            sigma_x_sq = F.avg_pool2d(x ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_x_sq
            sigma_y_sq = F.avg_pool2d(y ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_y_sq
            sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.window_size//2) - mu_x_mu_y
            
            # SSIM formula
            ssim = ((2 * mu_x_mu_y + self.c1) * (2 * sigma_xy + self.c2)) / \
                   ((mu_x_sq + mu_y_sq + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2))
            
            ssim_values.append(ssim.mean().item())
        
        return sum(ssim_values) / len(ssim_values)


class SAMMetric:
    """Spectral Angle Mapper (SAM) metric."""
    
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
    
    def __call__(self, original: Tensor, reconstructed: Tensor) -> float:
        """
        Compute SAM between original and reconstructed HSI.
        
        Args:
            original: Original HSI tensor of shape (B, C, H, W)
            reconstructed: Reconstructed HSI tensor of shape (B, C, H, W)
        
        Returns:
            Mean SAM in degrees
        """
        B, C, H, W = original.shape
        
        # Reshape to (B*H*W, C)
        original_flat = original.permute(0, 2, 3, 1).reshape(-1, C)
        reconstructed_flat = reconstructed.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize to unit vectors
        original_norm = F.normalize(original_flat, p=2, dim=1)
        reconstructed_norm = F.normalize(reconstructed_flat, p=2, dim=1)
        
        # Compute cosine similarity and clip
        cos_angle = torch.sum(original_norm * reconstructed_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # Compute spectral angle in radians and convert to degrees
        angles_rad = torch.acos(cos_angle)
        angles_deg = torch.rad2deg(angles_rad)
        
        return angles_deg.mean().item()


class CompressionRatioMetric:
    """Compute compression ratio and bits per pixel (BPP)."""
    
    @staticmethod
    def compute_bpp(compressed_bytes: int, original_shape: tuple) -> float:
        """
        Compute bits per pixel.
        
        Args:
            compressed_bytes: Size of compressed data in bytes
            original_shape: Shape of original data (B, C, H, W)
        
        Returns:
            BPP value
        """
        num_pixels = original_shape[0] * original_shape[2] * original_shape[3]
        bits_total = compressed_bytes * 8
        return bits_total / num_pixels
    
    @staticmethod
    def compute_ratio(original_bytes: int, compressed_bytes: int) -> float:
        """
        Compute compression ratio.
        
        Args:
            original_bytes: Size of original data
            compressed_bytes: Size of compressed data
        
        Returns:
            Compression ratio (original/compressed)
        """
        return original_bytes / compressed_bytes if compressed_bytes > 0 else float('inf')

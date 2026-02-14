"""Evaluator for model evaluation and compression metrics."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple, Any
import io
import pickle

from ..models.base import BaseCompressor
from ..metrics import PSNRMetric, SSIMMetric, SAMMetric, CompressionRatioMetric


class Evaluator:
    """Evaluator for compression models."""
    
    def __init__(
        self,
        model: BaseCompressor,
        test_loader: DataLoader,
        config: DictConfig,
        device: torch.device = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Compression model
            test_loader: Test data loader
            config: Configuration object
            device: Device to evaluate on
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Setup metrics
        self.psnr = PSNRMetric(max_value=1.0)
        self.ssim = SSIMMetric()
        self.sam = SAMMetric()
        self.compression_ratio = CompressionRatioMetric()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of metrics
        """
        all_metrics = {
            'psnr': [],
            'ssim': [],
            'sam': [],
            'bpp': [],
            'compression_ratio': [],
        }
        
        total_original_bytes = 0
        total_compressed_bytes = 0
        
        for batch_idx, x in enumerate(self.test_loader):
            x = x.to(self.device)
            B, C, H, W = x.shape
            
            # Forward pass to get reconstruction
            if self.model.compression_type == 'lossy':
                outputs = self.model(x)
                reconstruction = outputs['reconstruction']
            else:
                # For lossless, need actual compression/decompression
                reconstruction = x  # Placeholder
            
            # Clamp reconstruction to valid range
            reconstruction = torch.clamp(reconstruction, 0, 1)
            
            # Compute metrics
            psnr = self.psnr(x, reconstruction)
            ssim = self.ssim(x, reconstruction)
            sam = self.sam(x, reconstruction)
            
            all_metrics['psnr'].append(psnr)
            all_metrics['ssim'].append(ssim)
            all_metrics['sam'].append(sam)
            
            # Estimate compression (simplified)
            original_bytes = self._estimate_bytes(x)
            compressed_bytes = self._estimate_compressed_bytes(x, reconstruction)
            
            total_original_bytes += original_bytes
            total_compressed_bytes += compressed_bytes
            
            bpp = self.compression_ratio.compute_bpp(compressed_bytes, (B, C, H, W))
            all_metrics['bpp'].append(bpp)
            
            ratio = self.compression_ratio.compute_ratio(original_bytes, compressed_bytes)
            all_metrics['compression_ratio'].append(ratio)
        
        # Aggregate results
        results = {
            'psnr': sum(all_metrics['psnr']) / len(all_metrics['psnr']),
            'ssim': sum(all_metrics['ssim']) / len(all_metrics['ssim']),
            'sam': sum(all_metrics['sam']) / len(all_metrics['sam']),
            'bpp': sum(all_metrics['bpp']) / len(all_metrics['bpp']),
            'compression_ratio': sum(all_metrics['compression_ratio']) / len(all_metrics['compression_ratio']),
        }
        
        return results
    
    @staticmethod
    def _estimate_bytes(x: torch.Tensor) -> int:
        """Estimate original data size in bytes."""
        return x.numel() * 4  # Assume float32
    
    @staticmethod
    def _estimate_compressed_bytes(original: torch.Tensor, reconstruction: torch.Tensor) -> int:
        """Estimate compressed data size (simplified)."""
        # This is a placeholder - actual implementation would use entropy coding
        # For now, estimate based on MSE (lower MSE = smaller entropy)
        mse = torch.mean((original - reconstruction) ** 2).item()
        
        # Very simplified: assume compression is proportional to information loss
        # In reality, this would be from the entropy model
        compression_factor = max(0.1, 1.0 - mse)
        return int(Evaluator._estimate_bytes(original) * compression_factor)

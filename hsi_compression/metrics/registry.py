"""Metrics registry for dynamic metric loading."""

from typing import Dict, Callable, Any
from .quality import PSNRMetric, SSIMMetric, SAMMetric, CompressionRatioMetric


class MetricsRegistry:
    """Registry for metric functions."""
    
    def __init__(self):
        self._metrics: Dict[str, Callable] = {
            'psnr': PSNRMetric,
            'ssim': SSIMMetric,
            'sam': SAMMetric,
            'compression_ratio': CompressionRatioMetric,
        }
    
    def get(self, name: str, **kwargs) -> Any:
        """Get metric by name."""
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}. "
                           f"Available: {list(self._metrics.keys())}")
        return self._metrics[name](**kwargs)
    
    def list_metrics(self):
        """List available metrics."""
        return list(self._metrics.keys())


# Global registry instance
_global_metrics_registry = MetricsRegistry()


def get_metric(name: str, **kwargs) -> Any:
    """Get metric from global registry."""
    return _global_metrics_registry.get(name, **kwargs)


def list_metrics():
    """List all registered metrics."""
    return _global_metrics_registry.list_metrics()

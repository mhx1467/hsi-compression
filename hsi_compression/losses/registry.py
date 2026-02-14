"""Loss registry for dynamic loss loading."""

from typing import Dict, Callable, Any
from .distortion import MSELoss, SAMLoss, MSESAMLoss, NegativeLogLikelihoodLoss
from .rate import RateDistortionLoss


class LossRegistry:
    """Registry for loss functions."""
    
    def __init__(self):
        self._distortion_losses: Dict[str, Callable] = {
            'mse': MSELoss,
            'sam': SAMLoss,
            'mse_sam': MSESAMLoss,
            'nll': NegativeLogLikelihoodLoss,
        }
        self._rate_losses: Dict[str, Callable] = {
            'rate_distortion': RateDistortionLoss,
        }
    
    def get_distortion_loss(self, name: str, **kwargs) -> Any:
        """Get distortion loss by name."""
        if name not in self._distortion_losses:
            raise ValueError(f"Unknown distortion loss: {name}. "
                           f"Available: {list(self._distortion_losses.keys())}")
        return self._distortion_losses[name](**kwargs)
    
    def get_rate_loss(self, name: str, **kwargs) -> Any:
        """Get rate loss by name."""
        if name not in self._rate_losses:
            raise ValueError(f"Unknown rate loss: {name}. "
                           f"Available: {list(self._rate_losses.keys())}")
        return self._rate_losses[name](**kwargs)
    
    def list_distortion_losses(self):
        """List available distortion losses."""
        return list(self._distortion_losses.keys())
    
    def list_rate_losses(self):
        """List available rate losses."""
        return list(self._rate_losses.keys())


# Global registry instance
_global_loss_registry = LossRegistry()


def get_distortion_loss(name: str, **kwargs) -> Any:
    """Get distortion loss from global registry."""
    return _global_loss_registry.get_distortion_loss(name, **kwargs)


def get_rate_loss(name: str, **kwargs) -> Any:
    """Get rate loss from global registry."""
    return _global_loss_registry.get_rate_loss(name, **kwargs)


def list_distortion_losses():
    """List all registered distortion losses."""
    return _global_loss_registry.list_distortion_losses()


def list_rate_losses():
    """List all registered rate losses."""
    return _global_loss_registry.list_rate_losses()

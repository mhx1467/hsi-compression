"""Model registry for dynamic model loading."""

from typing import Dict, Type, Callable, Any, Optional
from .base import BaseCompressor


class ModelRegistry:
    """Registry for compression models."""
    
    def __init__(self):
        self._registry: Dict[str, Type[BaseCompressor]] = {}
        self._builders: Dict[str, Callable] = {}
    
    def register(self, name: str, model_class: Type[BaseCompressor]):
        """Register a model class."""
        self._registry[name] = model_class
    
    def register_builder(self, name: str, builder: Callable):
        """Register a builder function for custom model construction."""
        self._builders[name] = builder
    
    def get(self, name: str, **kwargs) -> BaseCompressor:
        """
        Get model instance by name.
        
        Args:
            name: Model name
            **kwargs: Model configuration parameters
        
        Returns:
            Model instance
        """
        if name in self._builders:
            return self._builders[name](**kwargs)
        
        if name not in self._registry:
            raise ValueError(f"Unknown model: {name}. Available: {list(self._registry.keys())}")
        
        return self._registry[name](**kwargs)
    
    def list_models(self):
        """List all registered models."""
        return list(self._registry.keys()) + list(self._builders.keys())


# Global registry instance
_global_registry = ModelRegistry()


def register_model(name: str):
    """Decorator to register a model."""
    def decorator(cls):
        _global_registry.register(name, cls)
        return cls
    return decorator


def get_model(name: str, **kwargs) -> BaseCompressor:
    """Get model by name from global registry."""
    return _global_registry.get(name, **kwargs)


def list_models():
    """List all registered models."""
    return _global_registry.list_models()

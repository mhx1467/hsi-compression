"""Dataset registry for dynamic dataset loading."""

from typing import Dict, Type, Callable, Any, Optional
from torch.utils.data import Dataset
from .hyspecnet11k import HySpecNet11k


class DatasetRegistry:
    """Registry for datasets."""
    
    def __init__(self):
        self._registry: Dict[str, Type[Dataset]] = {
            'hyspecnet11k': HySpecNet11k,
        }
    
    def register(self, name: str, dataset_class: Type[Dataset]):
        """Register a dataset class."""
        self._registry[name] = dataset_class
    
    def get(self, name: str, **kwargs) -> Dataset:
        """
        Get dataset instance by name.
        
        Args:
            name: Dataset name
            **kwargs: Dataset parameters
        
        Returns:
            Dataset instance
        """
        if name not in self._registry:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self._registry.keys())}")
        
        return self._registry[name](**kwargs)
    
    def list_datasets(self):
        """List all registered datasets."""
        return list(self._registry.keys())


# Global registry instance
_global_dataset_registry = DatasetRegistry()


def get_dataset(name: str, **kwargs) -> Dataset:
    """Get dataset by name from global registry."""
    return _global_dataset_registry.get(name, **kwargs)


def list_datasets():
    """List all registered datasets."""
    return _global_dataset_registry.list_datasets()

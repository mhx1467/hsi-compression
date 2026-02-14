"""Models package initialization."""

from .base import BaseCompressor, LosslessCompressor, LossyCompressor
from .registry import register_model, get_model, list_models

# Import concrete models to register them
from .tcn import TCNLosslessCompressor
from .mamba import MambaLossyCompressor

__all__ = [
    'BaseCompressor',
    'LosslessCompressor',
    'LossyCompressor',
    'register_model',
    'get_model',
    'list_models',
    'TCNLosslessCompressor',
    'MambaLossyCompressor',
]

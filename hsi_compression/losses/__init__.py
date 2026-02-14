"""Losses package initialization."""

from .distortion import MSELoss, SAMLoss, MSESAMLoss, NegativeLogLikelihoodLoss
from .rate import RateDistortionLoss, EntropybottleneckBpp
from .registry import (
    get_distortion_loss,
    get_rate_loss,
    list_distortion_losses,
    list_rate_losses,
)

__all__ = [
    'MSELoss',
    'SAMLoss',
    'MSESAMLoss',
    'NegativeLogLikelihoodLoss',
    'RateDistortionLoss',
    'EntropybottleneckBpp',
    'get_distortion_loss',
    'get_rate_loss',
    'list_distortion_losses',
    'list_rate_losses',
]

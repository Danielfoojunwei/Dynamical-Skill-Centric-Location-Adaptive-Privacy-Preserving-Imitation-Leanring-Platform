"""
Spatial Intelligence Module - Vision-Language-Action Models

This module provides spatial reasoning capabilities through:
- Pi0 VLA model for action prediction
- Spatial encoders and decoders
- Multi-modal fusion

Components:
- pi0/: Pi0 Vision-Language-Action model implementation

Usage:
    from src.spatial_intelligence.pi0 import Pi0, Pi0Config

    config = Pi0Config(action_dim=7)
    model = Pi0(config)
    actions = model.sample_actions(images, instruction, proprio)
"""

from .pi0 import model as pi0_model
from .pi0.model import Pi0, Pi0Config
from .pi0.modules import (
    VisionEncoder,
    LanguageEncoder,
    ActionDecoder,
)

__all__ = [
    'Pi0',
    'Pi0Config',
    'VisionEncoder',
    'LanguageEncoder',
    'ActionDecoder',
    'pi0_model',
]

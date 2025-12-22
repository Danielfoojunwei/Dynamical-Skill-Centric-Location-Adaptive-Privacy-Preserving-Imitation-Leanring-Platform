"""
Pi0 VLA Model - Vision-Language-Action for Robot Control

Pi0 is a diffusion-based VLA model that provides:
- Multi-view image encoding
- Language instruction processing
- Flow-matching action generation
- Proprioception integration

Reference: Physical Intelligence Pi0

Usage:
    from src.spatial_intelligence.pi0 import Pi0, Pi0Config

    config = Pi0Config(
        action_dim=7,
        action_horizon=16,
    )
    model = Pi0(config)

    actions = model.sample_actions(
        images=images,  # [B, N, 3, H, W]
        instruction="pick up the cup",
        proprio=proprio,  # [B, DOF]
    )
"""

from .model import Pi0, Pi0Config
from .modules import (
    VisionEncoder,
    LanguageEncoder,
    ActionDecoder,
    FlowMatchingHead,
)

__all__ = [
    'Pi0',
    'Pi0Config',
    'VisionEncoder',
    'LanguageEncoder',
    'ActionDecoder',
    'FlowMatchingHead',
]

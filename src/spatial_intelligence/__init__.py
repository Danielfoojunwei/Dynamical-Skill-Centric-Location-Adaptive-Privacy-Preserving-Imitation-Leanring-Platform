"""
Spatial Intelligence Module - Vision-Language-Action Models

This module provides spatial reasoning capabilities through:
- Pi0.5 VLA model (RECOMMENDED) - Official Physical Intelligence implementation
- Legacy Pi0 model (EXPERIMENTAL) - Custom implementation requiring PyTorch

Components:
- pi0/: Pi0 Vision-Language-Action model implementations

Usage (Pi0.5 - Recommended):
    from src.spatial_intelligence.pi0 import (
        Pi05Model, Pi05Config, Pi05Observation
    )

    model = Pi05Model.for_jetson_thor()
    model.load()
    result = model.infer(Pi05Observation(images=img, instruction="pick up cup"))

Usage (Legacy - Requires PyTorch):
    from src.spatial_intelligence.pi0 import Pi0, Pi0Config

    config = Pi0Config(action_dim=7)
    model = Pi0(config)
    actions = model.sample_actions(images, instruction, proprio)
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Pi0.5 - Official Physical Intelligence (RECOMMENDED)
# =============================================================================
try:
    from .pi0 import (
        Pi05Model,
        Pi05Config,
        Pi05Variant,
        Pi05Observation,
        Pi05Result,
        load_pi05,
        list_variants,
        check_installation,
        HAS_OPENPI,
    )
except ImportError as e:
    logger.debug(f"Pi0.5 not available: {e}")
    HAS_OPENPI = False
    Pi05Model = None
    Pi05Config = None
    Pi05Variant = None
    Pi05Observation = None
    Pi05Result = None
    load_pi05 = None
    list_variants = None
    check_installation = None

# =============================================================================
# Legacy Pi0 - Custom Implementation (EXPERIMENTAL - requires PyTorch)
# =============================================================================
try:
    from .pi0 import (
        Pi0,
        Pi0Config,
        VLMBackbone,
        HAS_LEGACY_PI0,
    )
    from .pi0.modules import (
        VisionEncoder,
        LanguageEncoder,
        ActionDecoder,
    )
except ImportError as e:
    logger.debug(f"Legacy Pi0 not available (PyTorch required): {e}")
    HAS_LEGACY_PI0 = False
    Pi0 = None
    Pi0Config = None
    VLMBackbone = None
    VisionEncoder = None
    LanguageEncoder = None
    ActionDecoder = None

__all__ = [
    # Pi0.5 Official (RECOMMENDED)
    'Pi05Model',
    'Pi05Config',
    'Pi05Variant',
    'Pi05Observation',
    'Pi05Result',
    'load_pi05',
    'list_variants',
    'check_installation',
    'HAS_OPENPI',

    # Legacy Custom Pi0 (EXPERIMENTAL)
    'Pi0',
    'Pi0Config',
    'VLMBackbone',
    'VisionEncoder',
    'LanguageEncoder',
    'ActionDecoder',
    'HAS_LEGACY_PI0',
]

"""
Spatial Intelligence Module - Pi0.5 Vision-Language-Action Model

This module provides spatial reasoning capabilities through Pi0.5,
the official VLA model from Physical Intelligence.

Pi0.5 Features:
- Pre-trained on 10k+ hours of robot data
- Open-world generalization to unseen environments
- Semantic task understanding
- Multi-camera support
- Proprioceptive conditioning

Usage:
    from src.spatial_intelligence.pi0 import (
        Pi05Model, Pi05Config, Pi05Observation
    )

    model = Pi05Model.for_jetson_thor()
    model.load()
    result = model.infer(Pi05Observation(images=img, instruction="pick up cup"))
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Pi0.5 - Official Physical Intelligence Implementation
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
# VLA Interface
# =============================================================================
try:
    from .vla_interface import (
        VLAInterface,
        VLAConfig,
        VLAObservation,
        VLAResult,
        HardwareTarget,
    )
except ImportError as e:
    logger.debug(f"VLA interface not available: {e}")
    VLAInterface = None
    VLAConfig = None
    VLAObservation = None
    VLAResult = None
    HardwareTarget = None


__all__ = [
    # Pi0.5 Official
    'Pi05Model',
    'Pi05Config',
    'Pi05Variant',
    'Pi05Observation',
    'Pi05Result',
    'load_pi05',
    'list_variants',
    'check_installation',
    'HAS_OPENPI',

    # VLA Interface
    'VLAInterface',
    'VLAConfig',
    'VLAObservation',
    'VLAResult',
    'HardwareTarget',
]

"""
Pi0.5 VLA Model - Vision-Language-Action for Robot Control

This module provides the Pi0.5 VLA model from Physical Intelligence
for robot control using natural language instructions.

Pi0.5 Features:
- Pre-trained on 10k+ hours of robot data
- Open-world generalization to unseen environments
- Semantic task understanding
- Multi-camera support
- Proprioceptive conditioning

Installation:
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi && pip install -e .

Usage:
    from src.spatial_intelligence.pi0 import (
        Pi05Model, Pi05Config, Pi05Variant, Pi05Observation
    )

    # Create and load model
    model = Pi05Model.for_jetson_thor()
    model.load()

    # Run inference
    result = model.infer(Pi05Observation(
        images=camera_images,
        instruction="pick up the red cup"
    ))
    actions = result.actions

References:
- Pi0.5 Paper: https://arxiv.org/abs/2504.16054
- OpenPI Repo: https://github.com/Physical-Intelligence/openpi
- Blog: https://www.physicalintelligence.company/blog/pi05
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Pi0.5 - Official Physical Intelligence Implementation
# =============================================================================
try:
    from .pi05_model import (
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
    logger.warning(f"Pi0.5 model not available: {e}")
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
# Backwards compatibility aliases
# =============================================================================
Pi05Backend = Pi05Model  # Alias for backwards compatibility
create_pi05_for_thor = lambda: Pi05Model.for_jetson_thor() if Pi05Model else None


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

    # Backwards compatibility
    'Pi05Backend',
    'create_pi05_for_thor',
]

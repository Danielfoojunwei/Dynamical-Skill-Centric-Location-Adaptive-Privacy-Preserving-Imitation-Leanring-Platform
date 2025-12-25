"""
Pi0 VLA Models - Vision-Language-Action for Robot Control

This module provides VLA model implementations for robot control:

1. **Pi0.5 (RECOMMENDED)** - Official Physical Intelligence implementation
   - Pre-trained on 10k+ hours of robot data
   - Open-world generalization to unseen environments
   - Use AS-IS from openpi library

2. **Legacy Pi0** - Custom implementation (EXPERIMENTAL)
   - Uses PaliGemma 3B backbone only
   - For development/research only
   - NOT recommended for production

IMPORTANT: Use Pi0.5 for production deployments.

Installation (Pi0.5):
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi && pip install -e .

Usage (Pi0.5 - Recommended):
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

# =============================================================================
# Pi0.5 - Official Physical Intelligence (RECOMMENDED FOR PRODUCTION)
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
    import logging
    logging.getLogger(__name__).warning(f"Pi0.5 model not available: {e}")
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
# Legacy Pi0 - Custom Implementation (EXPERIMENTAL - NOT FOR PRODUCTION)
# =============================================================================
try:
    from .model import Pi0, Pi0Config, VLMBackbone
    from .modules import (
        ActionEncoder,
        GemmaMoE,
        MoeExpertConfig,
        SinusoidalPosEmb,
    )
    HAS_LEGACY_PI0 = True
except ImportError:
    HAS_LEGACY_PI0 = False
    Pi0 = None
    Pi0Config = None
    VLMBackbone = None
    ActionEncoder = None
    GemmaMoE = None
    MoeExpertConfig = None
    SinusoidalPosEmb = None

# =============================================================================
# Backwards compatibility - redirect old imports to Pi0.5
# =============================================================================
# These were in the old openpi_backend.py - redirect to new pi05_model.py
Pi05Backend = Pi05Model  # Alias for backwards compatibility
create_pi05_for_thor = lambda: Pi05Model.for_jetson_thor() if Pi05Model else None


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

    # Backwards compatibility
    'Pi05Backend',
    'create_pi05_for_thor',

    # Legacy Custom Pi0 (EXPERIMENTAL)
    'Pi0',
    'Pi0Config',
    'VLMBackbone',
    'ActionEncoder',
    'GemmaMoE',
    'MoeExpertConfig',
    'SinusoidalPosEmb',
    'HAS_LEGACY_PI0',
]

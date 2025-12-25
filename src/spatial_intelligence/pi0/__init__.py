"""
Pi0 VLA Model - Vision-Language-Action for Robot Control

Pi0 is a diffusion-based VLA model that provides:
- Multi-view image encoding
- Language instruction processing
- Flow-matching action generation
- Proprioception integration

Supported Backends:
==================
1. **Custom Pi0** (this module): Gemma 3 backbone with custom MoE
2. **OpenPI Pi0.5**: Official Physical Intelligence implementation

VLM Backbones:
=============
- PaliGemma 3B (legacy)
- Gemma 3 4B/12B/27B (multimodal, recommended for Thor)

Jetson Thor Optimizations:
=========================
With 128GB memory and 2070 TFLOPS:
- Run Gemma 3-27B at full precision
- 10Hz control loop with full perception
- Native FP8 support for faster inference

Reference:
- Physical Intelligence Pi0: https://www.physicalintelligence.company/blog/pi0
- Pi0.5 Paper: https://arxiv.org/abs/2504.16054
- Gemma 3: https://huggingface.co/blog/gemma3

Usage:
    # Custom Pi0 with Gemma 3 backbone
    from src.spatial_intelligence.pi0 import Pi0, Pi0Config, VLMBackbone

    # For Jetson Thor (uses Gemma 3-27B)
    model = Pi0.for_jetson_thor()

    # Custom configuration
    config = Pi0Config(
        vlm_backbone=VLMBackbone.GEMMA3_12B,
        action_dim=7,
        action_horizon=16,
    )
    model = Pi0.from_config(config)

    # Official Pi0.5 via OpenPI
    from src.spatial_intelligence.pi0 import Pi05Backend, Pi05Config, Pi05Variant

    backend = Pi05Backend(Pi05Config(variant=Pi05Variant.PI05_BASE))
    backend.load()
    actions = backend.infer(observation)
"""

from .model import Pi0, Pi0Config, VLMBackbone
from .modules import (
    ActionEncoder,
    GemmaMoE,
    MoeExpertConfig,
    SinusoidalPosEmb,
)

# Try to import OpenPI backend
try:
    from .openpi_backend import (
        Pi05Backend,
        Pi05Config,
        Pi05Variant,
        Pi05Observation,
        create_pi05_for_thor,
    )
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    Pi05Backend = None
    Pi05Config = None
    Pi05Variant = None
    Pi05Observation = None
    create_pi05_for_thor = None

__all__ = [
    # Core Pi0 with Gemma 3
    'Pi0',
    'Pi0Config',
    'VLMBackbone',
    # MoE components
    'ActionEncoder',
    'GemmaMoE',
    'MoeExpertConfig',
    'SinusoidalPosEmb',
    # OpenPI Pi0.5 (if available)
    'Pi05Backend',
    'Pi05Config',
    'Pi05Variant',
    'Pi05Observation',
    'create_pi05_for_thor',
    'HAS_OPENPI',
]

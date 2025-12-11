"""
Meta AI Model Integration for Dynamical Edge Platform

This module integrates Meta's latest AI models for enhanced perception and planning:

1. DINOv3 - Self-supervised vision encoder for feature extraction
2. SAM3 - Text-driven segmentation for manipulation tasks
3. V-JEPA 2 - World model for predictive planning and action generation

All models are integrated with:
- MoE (Mixture of Experts) skill routing
- N2HE/FHE privacy-preserving encryption
- 4-Tier timing system (1kHz safety to 0.1Hz cloud)

Architecture:
============

┌─────────────────────────────────────────────────────────────────────────────┐
│                    META AI ENHANCED PERCEPTION STACK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   DINOv3     │───▶│    SAM3      │───▶│   V-JEPA 2   │                   │
│  │   (Vision)   │    │ (Segmentation)│    │ (World Model)│                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                    │                           │
│         └───────────────────┴────────────────────┘                           │
│                             │                                                │
│                    ┌────────▼────────┐                                       │
│                    │  N2HE Privacy   │                                       │
│                    │   Wrapper       │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│                    ┌────────▼────────┐                                       │
│                    │   MoE Skill     │                                       │
│                    │    Router       │                                       │
│                    └─────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

License: Apache 2.0 (Meta AI models) + Proprietary (Integration)
"""

from .dinov3 import DINOv3Encoder, DINOv3Config
from .sam3 import SAM3Segmenter, SAM3Config, SegmentationResult
from .vjepa2 import VJEPA2WorldModel, VJEPA2Config, WorldModelPrediction
from .unified_perception import (
    UnifiedPerceptionPipeline,
    PerceptionConfig,
    PerceptionResult
)
from .privacy_wrapper import (
    MetaAIPrivacyWrapper,
    EncryptedFeatures,
    PrivacyConfig
)

__all__ = [
    # DINOv3
    'DINOv3Encoder',
    'DINOv3Config',
    # SAM3
    'SAM3Segmenter',
    'SAM3Config',
    'SegmentationResult',
    # V-JEPA 2
    'VJEPA2WorldModel',
    'VJEPA2Config',
    'WorldModelPrediction',
    # Unified Pipeline
    'UnifiedPerceptionPipeline',
    'PerceptionConfig',
    'PerceptionResult',
    # Privacy
    'MetaAIPrivacyWrapper',
    'EncryptedFeatures',
    'PrivacyConfig',
]

__version__ = "0.3.2"

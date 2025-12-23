"""
MoAI: Mixture of All Intelligence

This module provides the MOAI neural network compression and FHE encryption
for privacy-preserving edge-to-cloud inference.

Components:
- n2he: Neural Network Homomorphic Encryption (LWE-based, post-quantum secure)
- moai_fhe: MOAI-FHE integration and PyTorch components for encrypted transformer inference

Reference:
K.Y. Lam et al., "Efficient FHE-based Privacy-Enhanced Neural Network for
Trustworthy AI-as-a-Service", IEEE TDSC.
https://github.com/HintSight-Technology/N2HE-hexl

Usage:
    # FHE operations
    from src.moai import MoaiFHESystem, MoaiFHEContext
    system = MoaiFHESystem()

    # PyTorch components (optional, requires torch)
    from src.moai import MoaiConfig, MoaiPolicy
    config = MoaiConfig(d_model=256)
    policy = MoaiPolicy(config)
"""

# Core N2HE (always available)
from .n2he import (
    N2HEContext,
    N2HEParams,
    N2HE_128,
    N2HE_192,
    LWECiphertext,
    RLWECiphertext,
    N2HEKeyGenerator,
    KeyGenerator,
    Encryptor,
    Decryptor,
    Evaluator,
    Ciphertext,
)

# MOAI FHE and PyTorch components (unified module)
from .moai_fhe import (
    # FHE components
    MoaiFHEContext,
    MoaiFHEConfig,
    MoaiFHESystem,
    MoaiTransformerFHE,
    # PyTorch components (config always available)
    MoaiConfig,
    MoaiTransformerBlockPT,
    MoaiPolicy,
)

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    # N2HE core
    'N2HEContext',
    'N2HEParams',
    'N2HE_128',
    'N2HE_192',
    'LWECiphertext',
    'RLWECiphertext',
    'N2HEKeyGenerator',
    'KeyGenerator',
    'Encryptor',
    'Decryptor',
    'Evaluator',
    'Ciphertext',
    # MOAI FHE
    'MoaiFHEContext',
    'MoaiFHEConfig',
    'MoaiFHESystem',
    'MoaiTransformerFHE',
    # MOAI PyTorch (optional)
    'MoaiConfig',
    'MoaiPolicy',
    'MoaiTransformerBlockPT',
    'HAS_TORCH',
]

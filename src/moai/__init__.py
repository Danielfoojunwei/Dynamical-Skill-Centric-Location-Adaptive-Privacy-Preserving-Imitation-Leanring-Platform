"""
MoAI: Mixture of All Intelligence

This module provides the MOAI neural network compression and FHE encryption
for privacy-preserving edge-to-cloud inference.

Components:
- n2he: Neural Network Homomorphic Encryption (LWE-based, post-quantum secure)
- moai_fhe: MOAI-FHE integration for encrypted transformer inference
- moai_pt: PyTorch MOAI transformer implementation

Reference:
K.Y. Lam et al., "Efficient FHE-based Privacy-Enhanced Neural Network for
Trustworthy AI-as-a-Service", IEEE TDSC.
https://github.com/HintSight-Technology/N2HE-hexl
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

# MOAI FHE (always available)
from .moai_fhe import (
    MoaiFHEContext,
    MoaiFHEConfig,
    MoaiFHESystem,
    MoaiTransformerFHE,
)

# PyTorch components (optional - requires torch)
try:
    from .moai_pt import (
        MoaiConfig,
        MoaiPolicy,
        MoaiTransformerBlockPT,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    MoaiConfig = None
    MoaiPolicy = None
    MoaiTransformerBlockPT = None

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

"""
Shared Cryptographic Primitives for Federated Learning

This module provides a unified interface for Fully Homomorphic Encryption (FHE)
used in privacy-preserving federated learning across the Dynamical platform.

Supported Backends:
1. TenSEAL (Microsoft SEAL) - Production-grade CKKS encryption
2. N2HE (LWE-based) - Research/fallback implementation
3. Mock - Development/testing without encryption

Usage:
    from src.shared.crypto import create_fhe_backend, FHEConfig

    # Create backend (auto-selects best available)
    backend = create_fhe_backend()

    # Encrypt gradients
    encrypted = backend.encrypt(gradients)

    # Homomorphic sum (on server)
    summed = backend.homomorphic_sum([enc1, enc2, enc3])

    # Decrypt result
    result = backend.decrypt(summed)
"""

from .fhe_backend import (
    FHEBackend,
    FHEConfig,
    TenSEALBackend,
    N2HEBackend,
    MockFHEBackend,
    create_fhe_backend,
    get_available_backends,
)

__all__ = [
    'FHEBackend',
    'FHEConfig',
    'TenSEALBackend',
    'N2HEBackend',
    'MockFHEBackend',
    'create_fhe_backend',
    'get_available_backends',
]

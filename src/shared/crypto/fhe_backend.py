"""
FHE Backend - Unified Fully Homomorphic Encryption Interface

This module provides a unified interface for FHE operations used in
privacy-preserving federated learning. It abstracts the underlying
cryptographic library (TenSEAL, N2HE, or Mock).

Security Properties:
- 128-bit post-quantum security (LWE-based)
- Additive homomorphism for gradient aggregation
- Efficient linear operations for federated averaging

Architecture:
    Edge Device          Cloud Server
    ───────────          ────────────
    1. Compute gradients
    2. Encrypt(grad) ──────► 3. Receive encrypted
                             4. Homomorphic sum
                             5. Decrypt aggregate
                        ◄─── 6. Return updated model
"""

import time
import gzip
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Backend Availability Detection
# =============================================================================

# Try TenSEAL (production-grade FHE)
try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False
    ts = None

# Try N2HE (research implementation)
try:
    from src.moai.n2he import N2HEContext, N2HEParams, LWECiphertext
    HAS_N2HE = True
except ImportError:
    HAS_N2HE = False
    N2HEContext = None
    N2HEParams = None
    LWECiphertext = None

# Try PyTorch (optional)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_available_backends() -> List[str]:
    """Return list of available FHE backends."""
    backends = []
    if HAS_TENSEAL:
        backends.append('tenseal')
    if HAS_N2HE:
        backends.append('n2he')
    backends.append('mock')  # Always available
    return backends


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FHEConfig:
    """Configuration for FHE backend.

    Attributes:
        backend: Preferred backend ('tenseal', 'n2he', 'mock', or 'auto')
        poly_modulus_degree: CKKS polynomial modulus degree (TenSEAL)
        coeff_mod_bit_sizes: Coefficient modulus bit sizes (TenSEAL)
        global_scale: Global scale for CKKS encoding (TenSEAL)
        n: Lattice dimension (N2HE)
        q: Ciphertext modulus (N2HE)
        t: Plaintext modulus (N2HE)
        sigma: Noise standard deviation (N2HE)
        security_bits: Target security level
    """
    backend: str = 'auto'

    # TenSEAL CKKS parameters
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    global_scale: int = 2**40

    # N2HE LWE parameters
    n: int = 1024
    q: int = 2**32
    t: int = 2**16
    sigma: float = 3.2

    # Security
    security_bits: int = 128


# =============================================================================
# Abstract Backend Interface
# =============================================================================

class FHEBackend(ABC):
    """Abstract base class for FHE backends.

    Implementations must provide:
    - encrypt(): Encrypt a numpy array or tensor
    - decrypt(): Decrypt encrypted data
    - homomorphic_sum(): Sum encrypted vectors without decryption
    - serialize()/deserialize(): Convert to/from bytes
    """

    def __init__(self, config: FHEConfig = None):
        self.config = config or FHEConfig()
        self.stats = {
            'encryptions': 0,
            'decryptions': 0,
            'homomorphic_ops': 0,
            'bytes_encrypted': 0,
            'total_time_ms': 0.0,
        }

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @abstractmethod
    def encrypt(self, data: np.ndarray) -> bytes:
        """Encrypt numpy array to bytes."""
        pass

    @abstractmethod
    def decrypt(self, encrypted: bytes) -> np.ndarray:
        """Decrypt bytes to numpy array."""
        pass

    @abstractmethod
    def homomorphic_sum(self, encrypted_list: List[bytes]) -> bytes:
        """Homomorphically sum encrypted vectors."""
        pass

    def get_public_context(self) -> bytes:
        """Get serialized public context for clients."""
        return b''

    def get_stats(self) -> Dict[str, Any]:
        """Get encryption statistics."""
        return self.stats.copy()


# =============================================================================
# TenSEAL Backend (Production)
# =============================================================================

class TenSEALBackend(FHEBackend):
    """TenSEAL CKKS backend using Microsoft SEAL.

    This is the production-grade FHE backend with:
    - CKKS scheme for approximate arithmetic
    - 128-bit security level
    - Efficient SIMD operations
    """

    def __init__(self, config: FHEConfig = None):
        super().__init__(config)

        if not HAS_TENSEAL:
            raise ImportError("TenSEAL not available. Install with: pip install tenseal")

        # Create CKKS context
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.config.poly_modulus_degree,
            coeff_mod_bit_sizes=self.config.coeff_mod_bit_sizes,
        )
        self.ctx.generate_galois_keys()
        self.ctx.global_scale = self.config.global_scale

        # Store secret key for decryption
        self._secret_key = self.ctx.secret_key()

        logger.info(f"[TenSEAL] Initialized CKKS with {self.config.security_bits}-bit security")

    @property
    def name(self) -> str:
        return 'tenseal'

    def encrypt(self, data: np.ndarray) -> bytes:
        """Encrypt numpy array using CKKS."""
        start = time.time()

        # Flatten and convert to float64
        flat = data.flatten().astype(np.float64)
        original_shape = data.shape

        # Encrypt
        encrypted_vec = ts.ckks_vector(self.ctx, flat.tolist())

        # Serialize
        result = gzip.compress(pickle.dumps({
            'backend': 'tenseal',
            'data': encrypted_vec.serialize(),
            'shape': original_shape,
            'num_elements': len(flat),
        }))

        # Update stats
        self.stats['encryptions'] += 1
        self.stats['bytes_encrypted'] += len(result)
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result

    def decrypt(self, encrypted: bytes) -> np.ndarray:
        """Decrypt CKKS ciphertext."""
        start = time.time()

        payload = pickle.loads(gzip.decompress(encrypted))
        encrypted_vec = ts.ckks_vector_from(self.ctx, payload['data'])

        result = np.array(encrypted_vec.decrypt(self._secret_key), dtype=np.float32)

        if 'shape' in payload:
            result = result[:np.prod(payload['shape'])].reshape(payload['shape'])

        self.stats['decryptions'] += 1
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result

    def homomorphic_sum(self, encrypted_list: List[bytes]) -> bytes:
        """Homomorphically sum CKKS vectors."""
        if not encrypted_list:
            return b''

        start = time.time()

        # Deserialize first vector
        payload0 = pickle.loads(gzip.decompress(encrypted_list[0]))
        sum_vec = ts.ckks_vector_from(self.ctx, payload0['data'])

        # Add remaining vectors
        for enc_bytes in encrypted_list[1:]:
            payload = pickle.loads(gzip.decompress(enc_bytes))
            vec = ts.ckks_vector_from(self.ctx, payload['data'])
            sum_vec = sum_vec + vec  # Homomorphic addition

        # Serialize result
        result = gzip.compress(pickle.dumps({
            'backend': 'tenseal',
            'data': sum_vec.serialize(),
            'shape': payload0.get('shape'),
            'num_clients': len(encrypted_list),
        }))

        self.stats['homomorphic_ops'] += len(encrypted_list)
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result

    def get_public_context(self) -> bytes:
        """Get public context for client-side encryption."""
        # Create context without secret key
        public_ctx = self.ctx.copy()
        public_ctx.make_context_public()
        return public_ctx.serialize()


# =============================================================================
# N2HE Backend (Research)
# =============================================================================

class N2HEBackend(FHEBackend):
    """N2HE LWE-based backend for research/fallback.

    Features:
    - LWE-based encryption (post-quantum secure)
    - Per-element encryption (simpler but less efficient)
    - Good for small vectors and research purposes
    """

    def __init__(self, config: FHEConfig = None):
        super().__init__(config)

        if not HAS_N2HE:
            raise ImportError("N2HE not available")

        # Create N2HE context
        params = N2HEParams(
            n=self.config.n,
            q=self.config.q,
            t=self.config.t,
            sigma=self.config.sigma,
            security_bits=self.config.security_bits,
        )
        self.ctx = N2HEContext(params=params, use_mock=False)
        self.ctx.generate_keys(generate_boot_key=False)

        self._scale = 2**15  # Fixed-point scaling

        logger.info(f"[N2HE] Initialized with n={self.config.n}, {self.config.security_bits}-bit security")

    @property
    def name(self) -> str:
        return 'n2he'

    def encrypt(self, data: np.ndarray) -> bytes:
        """Encrypt numpy array using LWE."""
        start = time.time()

        flat = data.flatten().astype(np.float64)
        original_shape = data.shape

        # Scale to integers
        scaled = np.clip(flat * self._scale, -32767, 32767).astype(np.int64)

        # Encrypt each element
        ciphertexts = [self.ctx.encrypt(int(val)).serialize() for val in scaled]

        result = gzip.compress(pickle.dumps({
            'backend': 'n2he',
            'ciphertexts': ciphertexts,
            'shape': original_shape,
            'scale': self._scale,
            'num_elements': len(flat),
        }))

        self.stats['encryptions'] += 1
        self.stats['bytes_encrypted'] += len(result)
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result

    def decrypt(self, encrypted: bytes) -> np.ndarray:
        """Decrypt LWE ciphertexts."""
        start = time.time()

        payload = pickle.loads(gzip.decompress(encrypted))

        # Decrypt each element
        ciphertexts = [LWECiphertext.deserialize(s) for s in payload['ciphertexts']]
        values = [self.ctx.decrypt(ct) for ct in ciphertexts]

        result = np.array(values, dtype=np.float32) / payload['scale']

        if 'shape' in payload:
            result = result.reshape(payload['shape'])

        self.stats['decryptions'] += 1
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result

    def homomorphic_sum(self, encrypted_list: List[bytes]) -> bytes:
        """Homomorphically sum LWE ciphertexts."""
        if not encrypted_list:
            return b''

        start = time.time()

        payloads = [pickle.loads(gzip.decompress(e)) for e in encrypted_list]
        num_elements = len(payloads[0]['ciphertexts'])

        summed_cts = []
        for i in range(num_elements):
            ct_sum = LWECiphertext.deserialize(payloads[0]['ciphertexts'][i])
            for payload in payloads[1:]:
                ct = LWECiphertext.deserialize(payload['ciphertexts'][i])
                ct_sum = ct_sum + ct  # Homomorphic addition
            summed_cts.append(ct_sum.serialize())

        result = gzip.compress(pickle.dumps({
            'backend': 'n2he',
            'ciphertexts': summed_cts,
            'shape': payloads[0]['shape'],
            'scale': payloads[0]['scale'],
            'num_clients': len(encrypted_list),
        }))

        self.stats['homomorphic_ops'] += len(encrypted_list)
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result


# =============================================================================
# Mock Backend (Development)
# =============================================================================

class MockFHEBackend(FHEBackend):
    """Mock backend for development and testing.

    This backend provides no actual encryption - it just compresses data.
    Use only for development and testing, never in production.
    """

    def __init__(self, config: FHEConfig = None):
        super().__init__(config)
        logger.warning("[MockFHE] Using mock backend - NO ENCRYPTION!")

    @property
    def name(self) -> str:
        return 'mock'

    def encrypt(self, data: np.ndarray) -> bytes:
        """'Encrypt' by compressing (no actual encryption)."""
        start = time.time()

        flat = data.flatten().astype(np.float64)

        result = gzip.compress(pickle.dumps({
            'backend': 'mock',
            'data': flat.tobytes(),
            'shape': data.shape,
            'dtype': str(data.dtype),
        }))

        self.stats['encryptions'] += 1
        self.stats['bytes_encrypted'] += len(result)
        self.stats['total_time_ms'] += (time.time() - start) * 1000

        return result

    def decrypt(self, encrypted: bytes) -> np.ndarray:
        """'Decrypt' by decompressing."""
        payload = pickle.loads(gzip.decompress(encrypted))
        result = np.frombuffer(payload['data'], dtype=np.float64).astype(np.float32)

        if 'shape' in payload:
            result = result.reshape(payload['shape'])

        self.stats['decryptions'] += 1
        return result

    def homomorphic_sum(self, encrypted_list: List[bytes]) -> bytes:
        """Sum by decrypting, adding, and re-encrypting."""
        if not encrypted_list:
            return b''

        payloads = [pickle.loads(gzip.decompress(e)) for e in encrypted_list]

        summed = np.frombuffer(payloads[0]['data'], dtype=np.float64)
        for payload in payloads[1:]:
            summed = summed + np.frombuffer(payload['data'], dtype=np.float64)

        result = gzip.compress(pickle.dumps({
            'backend': 'mock',
            'data': summed.tobytes(),
            'shape': payloads[0].get('shape'),
            'num_clients': len(encrypted_list),
        }))

        self.stats['homomorphic_ops'] += len(encrypted_list)
        return result


# =============================================================================
# Factory Function
# =============================================================================

def create_fhe_backend(
    backend: str = 'auto',
    config: FHEConfig = None,
) -> FHEBackend:
    """Create an FHE backend instance.

    Args:
        backend: Backend type ('tenseal', 'n2he', 'mock', or 'auto')
        config: FHE configuration (optional)

    Returns:
        FHEBackend instance

    Priority for 'auto':
        1. TenSEAL (if available)
        2. N2HE (if available)
        3. Mock (always available, but warns)
    """
    if config is None:
        config = FHEConfig(backend=backend)
    else:
        config.backend = backend

    if backend == 'auto':
        if HAS_TENSEAL:
            return TenSEALBackend(config)
        elif HAS_N2HE:
            return N2HEBackend(config)
        else:
            logger.warning("No FHE backend available - using mock (NO ENCRYPTION)")
            return MockFHEBackend(config)

    elif backend == 'tenseal':
        if not HAS_TENSEAL:
            raise ImportError("TenSEAL not available. Install: pip install tenseal")
        return TenSEALBackend(config)

    elif backend == 'n2he':
        if not HAS_N2HE:
            raise ImportError("N2HE not available")
        return N2HEBackend(config)

    elif backend == 'mock':
        return MockFHEBackend(config)

    else:
        raise ValueError(f"Unknown backend: {backend}. Available: {get_available_backends()}")

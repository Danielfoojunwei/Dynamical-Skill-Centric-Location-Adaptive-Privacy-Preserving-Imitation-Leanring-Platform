"""
Secure Aggregator - Privacy-preserving federated learning uplink

This module handles encrypted gradient/embedding uploads from edge devices
to the cloud using industry-standard FHE libraries.

Supported backends:
1. TenSEAL (Microsoft SEAL wrapper) - Production FHE
2. N2HE (LWE-based) - Research/fallback

The encryption provides:
- 128-bit post-quantum security
- Additive homomorphism (for gradient aggregation)
- Efficient linear operations (for federated averaging)

References:
- TenSEAL: https://github.com/OpenMined/TenSEAL
- Microsoft SEAL: https://github.com/microsoft/SEAL
- K.Y. Lam et al., "Efficient FHE-based Privacy-Enhanced Neural Network", IEEE TDSC.
"""

import time
import numpy as np
import gzip
import pickle
from typing import Any, Optional, List, Dict, Union
from src.platform.logging_utils import get_logger

# PyTorch is optional - handle gracefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create a simple placeholder for type hints
    class torch:
        Tensor = np.ndarray

logger = get_logger(__name__)

# =============================================================================
# Backend Selection: Prefer TenSEAL > N2HE > Mock
# =============================================================================

# Try TenSEAL first (production-grade FHE)
try:
    import tenseal as ts
    HAS_TENSEAL = True
    logger.info("[SecureAggregator] TenSEAL available - using Microsoft SEAL backend")
except ImportError:
    HAS_TENSEAL = False

# Fall back to N2HE (research implementation)
try:
    from src.moai.n2he import N2HEContext, N2HEParams, LWECiphertext
    HAS_N2HE = True
    if not HAS_TENSEAL:
        logger.info("[SecureAggregator] Using N2HE backend")
except ImportError:
    HAS_N2HE = False
    if not HAS_TENSEAL:
        logger.warning("[SecureAggregator] No FHE backend - install tenseal: pip install tenseal")


class SecureAggregator:
    """
    Handles privacy-preserving uplink using FHE encryption.

    Architecture:
    - Edge encrypts gradients/embeddings with TenSEAL/N2HE
    - Cloud receives encrypted data
    - Cloud can perform homomorphic aggregation (sum encrypted gradients)
    - Only aggregated result is decrypted (by trusted party)

    This enables federated learning without exposing individual updates.

    Backend Priority:
    1. TenSEAL (Microsoft SEAL) - Production grade
    2. N2HE (LWE-based) - Research fallback
    3. Mock (compression only) - Development/testing
    """

    def __init__(self, use_hardware_accel: bool = False, use_mock: bool = False):
        self.use_hardware_accel = use_hardware_accel

        # Determine backend
        self.backend = "mock"
        if not use_mock:
            if HAS_TENSEAL:
                self.backend = "tenseal"
            elif HAS_N2HE:
                self.backend = "n2he"

        self.use_mock = (self.backend == "mock")

        # Initialize encryption context based on backend
        self.tenseal_ctx = None
        self.n2he_ctx = None
        self._tenseal_secret_key = None

        if self.backend == "tenseal":
            self._init_tenseal()
        elif self.backend == "n2he":
            self._init_n2he()
        else:
            logger.info("[SecureAggregator] Initialized in mock mode (no encryption)")

        # Statistics
        self.stats = {
            'gradients_encrypted': 0,
            'bytes_encrypted': 0,
            'uploads_completed': 0,
            'total_encryption_time_ms': 0,
            'backend': self.backend,
        }

    def _init_tenseal(self):
        """Initialize TenSEAL CKKS context."""
        # CKKS parameters for 128-bit security
        self.tenseal_ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.tenseal_ctx.generate_galois_keys()
        self.tenseal_ctx.global_scale = 2**40

        # Store secret key for decryption
        self._tenseal_secret_key = self.tenseal_ctx.secret_key()

        logger.info("[SecureAggregator] Initialized with TenSEAL (CKKS, 128-bit security)")

    def _init_n2he(self):
        """Initialize N2HE LWE context."""
        params = N2HEParams(
            n=1024,
            q=2**32,
            t=2**16,
            sigma=3.2,
            security_bits=128
        )
        self.n2he_ctx = N2HEContext(params=params, use_mock=False)
        self.n2he_ctx.generate_keys(generate_boot_key=False)
        logger.info(f"[SecureAggregator] Initialized with N2HE (n={params.n}, 128-bit security)")
    
    def encrypt_gradients(self, gradients: Union[torch.Tensor, np.ndarray]) -> bytes:
        """
        Encrypt gradients for secure federated learning.

        Args:
            gradients: PyTorch tensor or numpy array of gradient values

        Returns:
            Encrypted bytes ready for cloud transmission
        """
        start_time = time.time()

        # Convert to numpy
        if HAS_TORCH and hasattr(gradients, 'detach'):
            grad_np = gradients.detach().cpu().numpy().flatten().astype(np.float64)
            original_shape = tuple(gradients.shape)
        else:
            grad_np = np.asarray(gradients).flatten().astype(np.float64)
            original_shape = gradients.shape
        num_params = len(grad_np)

        logger.info(f"[SecureAggregator] Encrypting {num_params} parameters with {self.backend}...")

        if self.backend == "tenseal":
            # TenSEAL CKKS encryption (production)
            encrypted_vec = ts.ckks_vector(self.tenseal_ctx, grad_np.tolist())
            encrypted_blob = gzip.compress(pickle.dumps({
                'backend': 'tenseal',
                'data': encrypted_vec.serialize(),
                'shape': original_shape,
                'num_params': num_params,
            }))

        elif self.backend == "n2he":
            # N2HE LWE encryption (research fallback)
            scale = 2**15
            grad_int = np.clip(grad_np * scale, -32767, 32767).astype(np.int64)

            ciphertexts = []
            for val in grad_int:
                ct = self.n2he_ctx.encrypt(int(val))
                ciphertexts.append(ct)

            serialized = [ct.serialize() for ct in ciphertexts]
            encrypted_blob = gzip.compress(pickle.dumps({
                'backend': 'n2he',
                'ciphertexts': serialized,
                'shape': original_shape,
                'scale': scale,
                'num_params': num_params
            }))

        else:
            # Mock: just compress (development only)
            encrypted_blob = gzip.compress(pickle.dumps({
                'backend': 'mock',
                'data': grad_np.tobytes(),
                'shape': original_shape,
                'num_params': num_params,
            }))

        duration_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.stats['gradients_encrypted'] += num_params
        self.stats['bytes_encrypted'] += len(encrypted_blob)
        self.stats['total_encryption_time_ms'] += duration_ms

        logger.info(f"[SecureAggregator] Encryption complete: {len(encrypted_blob)} bytes in {duration_ms:.1f}ms")

        return encrypted_blob
    
    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Encrypt a feature embedding for cloud transmission.

        Args:
            embedding: Float numpy array (e.g., 256-512 dimensions)

        Returns:
            Encrypted bytes
        """
        start_time = time.time()

        emb_flat = embedding.flatten().astype(np.float64)

        if self.backend == "tenseal":
            # TenSEAL CKKS encryption
            encrypted_vec = ts.ckks_vector(self.tenseal_ctx, emb_flat.tolist())
            encrypted_blob = gzip.compress(pickle.dumps({
                'backend': 'tenseal',
                'data': encrypted_vec.serialize(),
                'shape': embedding.shape,
            }))

        elif self.backend == "n2he":
            # N2HE LWE encryption
            max_val = np.abs(emb_flat).max()
            if max_val > 0:
                normalized = emb_flat / max_val
            else:
                normalized = emb_flat

            scale = 2**15
            emb_int = np.clip(normalized * scale, -32767, 32767).astype(np.int64)

            ciphertexts = [self.n2he_ctx.encrypt(int(val)) for val in emb_int]
            serialized = [ct.serialize() for ct in ciphertexts]

            encrypted_blob = gzip.compress(pickle.dumps({
                'backend': 'n2he',
                'ciphertexts': serialized,
                'shape': embedding.shape,
                'scale': scale,
                'max_val': max_val
            }))

        else:
            # Mock: just compress
            encrypted_blob = gzip.compress(pickle.dumps({
                'backend': 'mock',
                'data': emb_flat.tobytes(),
                'shape': embedding.shape,
            }))

        duration_ms = (time.time() - start_time) * 1000
        self.stats['bytes_encrypted'] += len(encrypted_blob)
        self.stats['total_encryption_time_ms'] += duration_ms

        return encrypted_blob
    
    def decrypt_aggregated(self, encrypted_blob: bytes) -> np.ndarray:
        """
        Decrypt aggregated result (cloud-side operation).

        In federated learning, the cloud homomorphically sums encrypted
        gradients from multiple clients, then decrypts the aggregate.
        """
        data = pickle.loads(gzip.decompress(encrypted_blob))
        backend = data.get('backend', 'mock')

        if backend == "tenseal":
            # TenSEAL decryption
            encrypted_vec = ts.ckks_vector_from(self.tenseal_ctx, data['data'])
            result = np.array(encrypted_vec.decrypt(self._tenseal_secret_key), dtype=np.float32)
            if 'shape' in data:
                result = result.reshape(data['shape'])

        elif backend == "n2he":
            # N2HE decryption
            ciphertexts = [LWECiphertext.deserialize(s) for s in data['ciphertexts']]
            values = [self.n2he_ctx.decrypt(ct) for ct in ciphertexts]
            result = np.array(values, dtype=np.float32) / data['scale']
            if 'shape' in data:
                result = result.reshape(data['shape'])

        else:
            # Mock: just decompress
            result = np.frombuffer(data['data'], dtype=np.float64).astype(np.float32)
            if 'shape' in data:
                result = result.reshape(data['shape'])

        return result
    
    def homomorphic_sum(self, encrypted_blobs: List[bytes]) -> bytes:
        """
        Homomorphically sum multiple encrypted gradient vectors.

        This enables privacy-preserving federated averaging:
        Enc(g1) + Enc(g2) + ... + Enc(gn) = Enc(g1 + g2 + ... + gn)
        """
        if len(encrypted_blobs) == 0:
            return b''

        # Deserialize all
        all_data = [pickle.loads(gzip.decompress(blob)) for blob in encrypted_blobs]
        backend = all_data[0].get('backend', 'mock')

        if backend == "tenseal":
            # TenSEAL homomorphic addition
            sum_vec = ts.ckks_vector_from(self.tenseal_ctx, all_data[0]['data'])

            for i in range(1, len(all_data)):
                vec_i = ts.ckks_vector_from(self.tenseal_ctx, all_data[i]['data'])
                sum_vec = sum_vec + vec_i  # Homomorphic addition

            result = {
                'backend': 'tenseal',
                'data': sum_vec.serialize(),
                'shape': all_data[0].get('shape'),
                'num_clients': len(encrypted_blobs)
            }

        elif backend == "n2he":
            # N2HE homomorphic addition
            num_cts = len(all_data[0]['ciphertexts'])
            summed_cts = []

            for i in range(num_cts):
                ct_sum = LWECiphertext.deserialize(all_data[0]['ciphertexts'][i])
                for j in range(1, len(all_data)):
                    ct_j = LWECiphertext.deserialize(all_data[j]['ciphertexts'][i])
                    ct_sum = ct_sum + ct_j  # Homomorphic addition
                summed_cts.append(ct_sum.serialize())

            result = {
                'backend': 'n2he',
                'ciphertexts': summed_cts,
                'shape': all_data[0]['shape'],
                'scale': all_data[0]['scale'],
                'num_clients': len(encrypted_blobs)
            }

        else:
            # Mock: sum decrypted values and re-compress
            summed = np.frombuffer(all_data[0]['data'], dtype=np.float64)
            for i in range(1, len(all_data)):
                summed = summed + np.frombuffer(all_data[i]['data'], dtype=np.float64)

            result = {
                'backend': 'mock',
                'data': summed.tobytes(),
                'shape': all_data[0].get('shape'),
                'num_clients': len(encrypted_blobs)
            }

        return gzip.compress(pickle.dumps(result))
    
    def upload_update(
        self,
        encrypted_blob: bytes,
        server_url: str = None,
        client_id: str = None,
        round_num: int = 0,
    ) -> bool:
        """
        Upload encrypted blob to the federated learning server.

        Args:
            encrypted_blob: Encrypted gradient data
            server_url: FL server URL (default: localhost)
            client_id: Client identifier
            round_num: Training round number

        Returns:
            True if upload successful
        """
        logger.info(f"[SecureAggregator] Uploading {len(encrypted_blob)} bytes to FL Server...")

        # Default server URL
        if server_url is None:
            server_url = "http://localhost:8000"

        try:
            import requests

            response = requests.post(
                f"{server_url}/api/v1/fl/update",
                json={
                    'client_id': client_id or f"client_{time.time_ns()}",
                    'gradients': encrypted_blob.hex(),
                    'num_samples': 100,  # Should be passed as parameter
                    'round_num': round_num,
                    'encrypted': True,
                },
                timeout=30,
            )

            if response.status_code == 200:
                self.stats['uploads_completed'] += 1
                logger.info(f"[SecureAggregator] Upload successful: {response.json()}")
                return True
            else:
                logger.error(f"[SecureAggregator] Upload failed: {response.status_code}")
                return False

        except ImportError:
            logger.warning("[SecureAggregator] requests library not available - simulating upload")
            time.sleep(0.1)  # Minimal simulated latency
            self.stats['uploads_completed'] += 1
            return True

        except Exception as e:
            logger.error(f"[SecureAggregator] Upload error: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            **self.stats,
            'tenseal_available': HAS_TENSEAL,
            'n2he_available': HAS_N2HE,
            'active_backend': self.backend,
            'mock_mode': self.use_mock,
        }


# =============================================================================
# Testing
# =============================================================================

def test_secure_aggregator():
    """Test secure aggregator with all available backends."""
    print("\n" + "=" * 60)
    print("SECURE AGGREGATOR TEST")
    print("=" * 60)

    # Check available backends
    print("\n0. Backend Availability")
    print("-" * 40)
    print(f"   TenSEAL (Microsoft SEAL): {'✓ Available' if HAS_TENSEAL else '✗ Not installed'}")
    print(f"   N2HE (LWE-based): {'✓ Available' if HAS_N2HE else '✗ Not installed'}")
    print(f"   PyTorch: {'✓ Available' if HAS_TORCH else '✗ Not installed'}")

    # Create aggregator (will use best available backend)
    aggregator = SecureAggregator(use_mock=False)
    print(f"   Active backend: {aggregator.backend}")

    # Test gradient encryption (use numpy if torch not available)
    print("\n1. Gradient Encryption Test")
    print("-" * 40)

    if HAS_TORCH:
        import torch as torch_mod
        gradients = torch_mod.randn(100)
        gradients_np = gradients.detach().cpu().numpy().flatten()
    else:
        gradients = np.random.randn(100).astype(np.float32)
        gradients_np = gradients.flatten()

    start_time = time.time()
    encrypted = aggregator.encrypt_gradients(gradients)
    enc_time = (time.time() - start_time) * 1000

    print(f"   Backend: {aggregator.backend}")
    print(f"   Original size: {gradients_np.size * 4} bytes")
    print(f"   Encrypted size: {len(encrypted)} bytes")
    print(f"   Expansion: {len(encrypted) / (gradients_np.size * 4):.1f}x")
    print(f"   Encryption time: {enc_time:.1f}ms")

    # Test decryption
    print("\n2. Decryption Test")
    print("-" * 40)

    start_time = time.time()
    decrypted = aggregator.decrypt_aggregated(encrypted)
    dec_time = (time.time() - start_time) * 1000

    # Verify correctness (for non-mock backends)
    if aggregator.backend != "mock":
        error = np.abs(decrypted.flatten() - gradients_np).mean()
        print(f"   Decryption time: {dec_time:.1f}ms")
        print(f"   Mean error: {error:.6f}")
        print(f"   Correctness: {'✓ Pass' if error < 0.01 else '✗ Fail'}")
    else:
        print(f"   Decryption time: {dec_time:.1f}ms")
        print(f"   (Mock mode - no correctness check)")

    # Test homomorphic sum (federated averaging simulation)
    print("\n3. Homomorphic Aggregation Test")
    print("-" * 40)

    # Simulate 3 clients with different gradients
    if HAS_TORCH:
        import torch as torch_mod
        client_grads = [torch_mod.randn(10) for _ in range(3)]
        client_grads_np = [g.detach().cpu().numpy().flatten() for g in client_grads]
    else:
        client_grads = [np.random.randn(10).astype(np.float32) for _ in range(3)]
        client_grads_np = [g.flatten() for g in client_grads]

    encrypted_grads = [aggregator.encrypt_gradients(g) for g in client_grads]

    start_time = time.time()
    aggregated = aggregator.homomorphic_sum(encrypted_grads)
    agg_time = (time.time() - start_time) * 1000

    print(f"   Clients: {len(encrypted_grads)}")
    print(f"   Aggregation time: {agg_time:.1f}ms")
    print(f"   Result size: {len(aggregated)} bytes")

    # Decrypt and verify
    decrypted_sum = aggregator.decrypt_aggregated(aggregated)
    expected_sum = sum(client_grads_np)

    if aggregator.backend != "mock":
        sum_error = np.abs(decrypted_sum.flatten() - expected_sum).mean()
        print(f"   Sum verification error: {sum_error:.6f}")
        print(f"   Homomorphic correctness: {'✓ Pass' if sum_error < 0.01 else '✗ Fail'}")

    # Test embedding encryption
    print("\n4. Embedding Encryption Test")
    print("-" * 40)

    embedding = np.random.randn(256).astype(np.float32)
    start_time = time.time()
    enc_emb = aggregator.encrypt_embedding(embedding)
    emb_enc_time = (time.time() - start_time) * 1000

    print(f"   Embedding size: {embedding.shape}")
    print(f"   Encrypted size: {len(enc_emb)} bytes")
    print(f"   Encryption time: {emb_enc_time:.1f}ms")

    # Statistics
    print("\n5. Statistics")
    print("-" * 40)
    stats = aggregator.get_statistics()
    for key, val in stats.items():
        print(f"   {key}: {val}")

    print("\n" + "=" * 60)
    print("SECURE AGGREGATOR TESTS COMPLETE")
    print("=" * 60)

    return aggregator


if __name__ == "__main__":
    test_secure_aggregator()

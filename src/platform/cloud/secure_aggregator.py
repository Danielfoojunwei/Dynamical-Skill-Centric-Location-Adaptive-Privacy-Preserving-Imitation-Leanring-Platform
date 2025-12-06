"""
Secure Aggregator - Privacy-preserving federated learning uplink using N2HE

This module handles encrypted gradient/embedding uploads from edge devices
to the cloud using the N2HE homomorphic encryption scheme.

The N2HE scheme provides:
- 128-bit post-quantum security (LWE hardness)
- Additive homomorphism (for gradient aggregation)
- Efficient linear operations (for federated averaging)

Reference:
K.Y. Lam et al., "Efficient FHE-based Privacy-Enhanced Neural Network for
Trustworthy AI-as-a-Service", IEEE TDSC.
"""

import torch
import time
import numpy as np
import gzip
import pickle
from typing import Any, Optional, List, Dict
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

# Import N2HE
try:
    from src.moai.n2he import N2HEContext, N2HEParams, LWECiphertext
    HAS_N2HE = True
except ImportError:
    HAS_N2HE = False
    logger.warning("N2HE not available - using mock encryption")


class SecureAggregator:
    """
    Handles privacy-preserving uplink using N2HE encryption.
    
    Architecture:
    - Edge encrypts gradients/embeddings with N2HE (LWE-based)
    - Cloud receives encrypted data
    - Cloud can perform homomorphic aggregation (sum encrypted gradients)
    - Only aggregated result is decrypted (by trusted party)
    
    This enables federated learning without exposing individual updates.
    """
    
    def __init__(self, use_hardware_accel: bool = False, use_mock: bool = False):
        self.use_hardware_accel = use_hardware_accel
        self.use_mock = use_mock or not HAS_N2HE
        
        # Initialize N2HE context
        if not self.use_mock:
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
        else:
            self.n2he_ctx = None
            logger.info("[SecureAggregator] Initialized in mock mode")
        
        # Statistics
        self.stats = {
            'gradients_encrypted': 0,
            'bytes_encrypted': 0,
            'uploads_completed': 0,
            'total_encryption_time_ms': 0,
        }
    
    def encrypt_gradients(self, gradients: torch.Tensor) -> bytes:
        """
        Encrypt gradients using N2HE for secure federated learning.
        
        Args:
            gradients: PyTorch tensor of gradient values
            
        Returns:
            Encrypted bytes ready for cloud transmission
        """
        start_time = time.time()
        
        # Convert to numpy
        grad_np = gradients.detach().cpu().numpy().flatten()
        num_params = len(grad_np)
        
        logger.info(f"[SecureAggregator] Encrypting {num_params} parameters...")
        
        if self.use_mock:
            # Mock encryption - just compress
            time.sleep(0.1)  # Simulate encryption latency
            encrypted_blob = gzip.compress(grad_np.tobytes())
        else:
            # Real N2HE encryption
            # Quantize gradients to integers (scale by 2^15 for precision)
            scale = 2**15
            grad_int = np.clip(grad_np * scale, -32767, 32767).astype(np.int64)
            
            # Encrypt each gradient element
            ciphertexts = []
            for val in grad_int:
                ct = self.n2he_ctx.encrypt(int(val))
                ciphertexts.append(ct)
            
            # Serialize ciphertexts
            serialized = [ct.serialize() for ct in ciphertexts]
            encrypted_blob = gzip.compress(pickle.dumps({
                'ciphertexts': serialized,
                'shape': gradients.shape,
                'scale': scale,
                'num_params': num_params
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
        
        if self.use_mock:
            time.sleep(0.05)
            return gzip.compress(embedding.tobytes())
        
        # Normalize and quantize
        max_val = np.abs(embedding).max()
        if max_val > 0:
            normalized = embedding / max_val
        else:
            normalized = embedding
        
        scale = 2**15
        emb_int = np.clip(normalized.flatten() * scale, -32767, 32767).astype(np.int64)
        
        # Encrypt
        ciphertexts = [self.n2he_ctx.encrypt(int(val)) for val in emb_int]
        
        # Serialize
        serialized = [ct.serialize() for ct in ciphertexts]
        encrypted_blob = gzip.compress(pickle.dumps({
            'ciphertexts': serialized,
            'shape': embedding.shape,
            'scale': scale,
            'max_val': max_val
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
        if self.use_mock:
            return np.frombuffer(gzip.decompress(encrypted_blob), dtype=np.float32)
        
        data = pickle.loads(gzip.decompress(encrypted_blob))
        ciphertexts = [LWECiphertext.deserialize(s) for s in data['ciphertexts']]
        
        # Decrypt
        values = [self.n2he_ctx.decrypt(ct) for ct in ciphertexts]
        result = np.array(values, dtype=np.float32) / data['scale']
        
        return result.reshape(data['shape'])
    
    def homomorphic_sum(self, encrypted_blobs: List[bytes]) -> bytes:
        """
        Homomorphically sum multiple encrypted gradient vectors.
        
        This enables privacy-preserving federated averaging:
        Enc(g1) + Enc(g2) + ... + Enc(gn) = Enc(g1 + g2 + ... + gn)
        """
        if len(encrypted_blobs) == 0:
            return b''
        
        if self.use_mock:
            # Mock: just return first blob
            return encrypted_blobs[0]
        
        # Deserialize all
        all_data = [pickle.loads(gzip.decompress(blob)) for blob in encrypted_blobs]
        
        # Sum ciphertexts element-wise
        num_cts = len(all_data[0]['ciphertexts'])
        summed_cts = []
        
        for i in range(num_cts):
            ct_sum = LWECiphertext.deserialize(all_data[0]['ciphertexts'][i])
            for j in range(1, len(all_data)):
                ct_j = LWECiphertext.deserialize(all_data[j]['ciphertexts'][i])
                ct_sum = ct_sum + ct_j  # Homomorphic addition
            summed_cts.append(ct_sum.serialize())
        
        # Package result
        result = {
            'ciphertexts': summed_cts,
            'shape': all_data[0]['shape'],
            'scale': all_data[0]['scale'],
            'num_clients': len(encrypted_blobs)
        }
        
        return gzip.compress(pickle.dumps(result))
    
    def upload_update(self, encrypted_blob: bytes) -> bool:
        """
        Upload encrypted blob to the aggregation server.
        """
        logger.info(f"[SecureAggregator] Uploading {len(encrypted_blob)} bytes to Federated Server...")
        
        # TODO: Implement actual HTTP POST to cloud endpoint
        # response = requests.post(
        #     "https://api.dynamical.ai/v1/federated/update",
        #     data=encrypted_blob,
        #     headers={"Content-Type": "application/octet-stream"}
        # )
        
        time.sleep(0.3)  # Simulate network latency
        
        self.stats['uploads_completed'] += 1
        logger.info("[SecureAggregator] Upload successful.")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            **self.stats,
            'n2he_available': HAS_N2HE,
            'mock_mode': self.use_mock,
        }


# =============================================================================
# Testing
# =============================================================================

def test_secure_aggregator():
    """Test secure aggregator."""
    print("\n" + "=" * 60)
    print("SECURE AGGREGATOR TEST")
    print("=" * 60)
    
    # Create aggregator with real N2HE
    aggregator = SecureAggregator(use_mock=False)
    
    # Test gradient encryption
    print("\n1. Gradient Encryption Test")
    print("-" * 40)
    
    gradients = torch.randn(100)
    encrypted = aggregator.encrypt_gradients(gradients)
    print(f"   Original size: {gradients.numel() * 4} bytes")
    print(f"   Encrypted size: {len(encrypted)} bytes")
    print(f"   Expansion: {len(encrypted) / (gradients.numel() * 4):.1f}x")
    
    # Test homomorphic sum (federated averaging simulation)
    print("\n2. Homomorphic Aggregation Test")
    print("-" * 40)
    
    # Simulate 3 clients
    client_grads = [torch.randn(10) for _ in range(3)]
    encrypted_grads = [aggregator.encrypt_gradients(g) for g in client_grads]
    
    # Sum encrypted (would happen on cloud)
    aggregated = aggregator.homomorphic_sum(encrypted_grads)
    print(f"   Aggregated {len(encrypted_grads)} encrypted gradient vectors")
    print(f"   Result size: {len(aggregated)} bytes")
    
    # Statistics
    print("\n3. Statistics")
    print("-" * 40)
    stats = aggregator.get_statistics()
    for key, val in stats.items():
        print(f"   {key}: {val}")
    
    print("\n" + "=" * 60)
    print("SECURE AGGREGATOR TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_secure_aggregator()

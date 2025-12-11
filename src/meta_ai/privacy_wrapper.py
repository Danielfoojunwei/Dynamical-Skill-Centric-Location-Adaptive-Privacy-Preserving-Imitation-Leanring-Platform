"""
Privacy Wrapper for Meta AI Models

Integrates N2HE/FHE encryption with Meta AI models (DINOv3, SAM3, V-JEPA 2)
for privacy-preserving perception and planning.

Key Features:
- Encrypted feature transmission to cloud
- Homomorphic operations on encrypted embeddings
- 128-bit post-quantum security (LWE-based)
- Maintains MoE skill routing on encrypted features

Architecture:
============

Edge Device (Jetson AGX Orin):
1. Run DINOv3/SAM3/V-JEPA 2 locally
2. Encrypt extracted features with N2HE
3. Send encrypted features to cloud

Cloud:
1. Receive encrypted features
2. Perform homomorphic MoE routing
3. Return encrypted skill weights

This ensures:
- Visual data never leaves edge unencrypted
- Cloud cannot see raw features
- Privacy-preserving skill learning
"""

import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Import N2HE encryption
try:
    from src.moai.n2he import (
        N2HEContext,
        N2HEParams,
        N2HE_128,
        LWECiphertext,
        Encryptor,
        Decryptor,
    )
    HAS_N2HE = True
except ImportError:
    HAS_N2HE = False
    logger.warning("N2HE not available - privacy features disabled")

# Import MOAI FHE
try:
    from src.moai.moai_fhe import MoaiFHEContext, MoaiFHEConfig
    HAS_MOAI_FHE = True
except ImportError:
    HAS_MOAI_FHE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PrivacyConfig:
    """Configuration for privacy wrapper."""
    # Encryption parameters
    lwe_dimension: int = 1024
    ciphertext_modulus: int = 2**32
    plaintext_modulus: int = 2**16
    noise_std: float = 3.2
    security_bits: int = 128

    # Feature quantization
    quantization_bits: int = 16
    quantization_scale: float = 2**15

    # Compression
    compress_ciphertexts: bool = True
    compression_level: int = 6

    # Cloud communication
    max_batch_size: int = 32
    timeout_ms: float = 5000.0

    # Enable/disable features
    enable_encryption: bool = True
    enable_homomorphic_routing: bool = True


@dataclass
class EncryptedFeatures:
    """Encrypted feature representation."""
    # Encrypted feature vector
    ciphertexts: List[bytes]        # Serialized LWE ciphertexts

    # Metadata (not encrypted)
    feature_dim: int
    source_model: str               # "dinov3", "sam3", "vjepa2"
    encryption_params: Dict[str, Any]

    # Timing
    encryption_time_ms: float = 0.0

    # Hash for integrity
    hash: str = ""

    @property
    def size_bytes(self) -> int:
        return sum(len(ct) for ct in self.ciphertexts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            'ciphertexts': self.ciphertexts,
            'feature_dim': self.feature_dim,
            'source_model': self.source_model,
            'encryption_params': self.encryption_params,
            'hash': self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedFeatures':
        """Create from dictionary."""
        return cls(
            ciphertexts=data['ciphertexts'],
            feature_dim=data['feature_dim'],
            source_model=data['source_model'],
            encryption_params=data['encryption_params'],
            hash=data.get('hash', ''),
        )


@dataclass
class EncryptedRoutingRequest:
    """Request for encrypted MoE routing."""
    encrypted_features: EncryptedFeatures
    task_description: Optional[str] = None
    max_skills: int = 3
    device_id: str = ""


@dataclass
class EncryptedRoutingResponse:
    """Response from encrypted MoE routing."""
    # Encrypted skill weights
    encrypted_weights: List[bytes]

    # Skill IDs (not encrypted - needed for skill loading)
    skill_ids: List[str]

    # Routing metadata
    routing_time_ms: float = 0.0


# =============================================================================
# Privacy Wrapper
# =============================================================================

class MetaAIPrivacyWrapper:
    """
    Privacy wrapper for Meta AI models.

    Provides:
    - Feature encryption using N2HE
    - Encrypted transmission to cloud
    - Homomorphic skill routing
    - Decryption of skill weights

    Usage:
        wrapper = MetaAIPrivacyWrapper()

        # Encrypt DINOv3 features
        features = dinov3.encode(image)
        encrypted = wrapper.encrypt_features(features, source="dinov3")

        # Send to cloud for routing
        response = wrapper.request_encrypted_routing(encrypted, task="pick up cup")

        # Decrypt skill weights locally
        weights = wrapper.decrypt_weights(response.encrypted_weights)
    """

    def __init__(self, config: PrivacyConfig = None):
        self.config = config or PrivacyConfig()

        # Initialize N2HE context
        if self.config.enable_encryption and HAS_N2HE:
            params = N2HEParams(
                n=self.config.lwe_dimension,
                q=self.config.ciphertext_modulus,
                t=self.config.plaintext_modulus,
                sigma=self.config.noise_std,
                security_bits=self.config.security_bits,
            )
            self.n2he_ctx = N2HEContext(params=params, use_mock=False)
            self.n2he_ctx.generate_keys(generate_boot_key=False)
            logger.info(f"Initialized N2HE with {self.config.security_bits}-bit security")
        else:
            self.n2he_ctx = None
            if self.config.enable_encryption:
                logger.warning("N2HE not available - using mock encryption")

        # Statistics
        self.stats = {
            'features_encrypted': 0,
            'features_decrypted': 0,
            'bytes_encrypted': 0,
            'total_encryption_time_ms': 0.0,
            'total_decryption_time_ms': 0.0,
        }

    def encrypt_features(
        self,
        features: np.ndarray,
        source_model: str = "dinov3",
    ) -> EncryptedFeatures:
        """
        Encrypt feature vector using N2HE.

        Args:
            features: Feature vector [D] or [B, D]
            source_model: Source model name

        Returns:
            EncryptedFeatures with serialized ciphertexts
        """
        start_time = time.time()

        # Flatten if needed
        if features.ndim > 1:
            features = features.flatten()

        feature_dim = len(features)

        if not self.config.enable_encryption or self.n2he_ctx is None:
            # Mock encryption - just serialize
            ciphertexts = [features.tobytes()]
            hash_val = hashlib.sha256(features.tobytes()).hexdigest()[:16]

            encryption_time = (time.time() - start_time) * 1000

            return EncryptedFeatures(
                ciphertexts=ciphertexts,
                feature_dim=feature_dim,
                source_model=source_model,
                encryption_params={'mock': True},
                encryption_time_ms=encryption_time,
                hash=hash_val,
            )

        # Quantize features to integers
        scale = self.config.quantization_scale
        max_val = np.abs(features).max()
        if max_val > 0:
            normalized = features / max_val
        else:
            normalized = features

        quantized = np.clip(normalized * scale, -scale + 1, scale - 1).astype(np.int64)

        # Encrypt each element
        ciphertexts = []
        for val in quantized:
            ct = self.n2he_ctx.encrypt(int(val))
            serialized = ct.serialize()
            ciphertexts.append(serialized)

        # Compute hash for integrity
        hash_input = b''.join(ciphertexts[:10])  # Hash first 10 for efficiency
        hash_val = hashlib.sha256(hash_input).hexdigest()[:16]

        encryption_time = (time.time() - start_time) * 1000

        # Update stats
        self.stats['features_encrypted'] += 1
        self.stats['bytes_encrypted'] += sum(len(ct) for ct in ciphertexts)
        self.stats['total_encryption_time_ms'] += encryption_time

        return EncryptedFeatures(
            ciphertexts=ciphertexts,
            feature_dim=feature_dim,
            source_model=source_model,
            encryption_params={
                'n': self.config.lwe_dimension,
                'q': self.config.ciphertext_modulus,
                't': self.config.plaintext_modulus,
                'scale': scale,
                'max_val': float(max_val),
            },
            encryption_time_ms=encryption_time,
            hash=hash_val,
        )

    def decrypt_features(
        self,
        encrypted: EncryptedFeatures,
    ) -> np.ndarray:
        """
        Decrypt encrypted features.

        Args:
            encrypted: Encrypted features

        Returns:
            Decrypted feature vector [D]
        """
        start_time = time.time()

        if not self.config.enable_encryption or self.n2he_ctx is None:
            # Mock decryption
            if encrypted.encryption_params.get('mock', False):
                features = np.frombuffer(encrypted.ciphertexts[0], dtype=np.float32)
            else:
                features = np.zeros(encrypted.feature_dim, dtype=np.float32)

            decryption_time = (time.time() - start_time) * 1000
            self.stats['features_decrypted'] += 1
            self.stats['total_decryption_time_ms'] += decryption_time

            return features

        # Get parameters
        scale = encrypted.encryption_params.get('scale', self.config.quantization_scale)
        max_val = encrypted.encryption_params.get('max_val', 1.0)

        # Decrypt each element
        values = []
        for ct_bytes in encrypted.ciphertexts:
            ct = LWECiphertext.deserialize(ct_bytes)
            val = self.n2he_ctx.decrypt(ct)
            values.append(val)

        # De-quantize
        features = np.array(values, dtype=np.float32) / scale * max_val

        decryption_time = (time.time() - start_time) * 1000

        self.stats['features_decrypted'] += 1
        self.stats['total_decryption_time_ms'] += decryption_time

        return features

    def encrypt_weights(
        self,
        weights: np.ndarray,
    ) -> List[bytes]:
        """
        Encrypt skill routing weights.

        Args:
            weights: Weight vector [N]

        Returns:
            List of serialized ciphertexts
        """
        if not self.config.enable_encryption or self.n2he_ctx is None:
            return [weights.tobytes()]

        scale = self.config.quantization_scale
        quantized = np.clip(weights * scale, -scale + 1, scale - 1).astype(np.int64)

        ciphertexts = []
        for val in quantized:
            ct = self.n2he_ctx.encrypt(int(val))
            ciphertexts.append(ct.serialize())

        return ciphertexts

    def decrypt_weights(
        self,
        encrypted_weights: List[bytes],
    ) -> np.ndarray:
        """
        Decrypt skill routing weights.

        Args:
            encrypted_weights: Serialized ciphertexts

        Returns:
            Decrypted weight vector
        """
        if not self.config.enable_encryption or self.n2he_ctx is None:
            if len(encrypted_weights) == 1:
                return np.frombuffer(encrypted_weights[0], dtype=np.float32)
            return np.zeros(len(encrypted_weights), dtype=np.float32)

        scale = self.config.quantization_scale

        values = []
        for ct_bytes in encrypted_weights:
            ct = LWECiphertext.deserialize(ct_bytes)
            val = self.n2he_ctx.decrypt(ct)
            values.append(val)

        return np.array(values, dtype=np.float32) / scale

    def homomorphic_dot_product(
        self,
        encrypted_features: EncryptedFeatures,
        plaintext_weights: np.ndarray,
    ) -> bytes:
        """
        Compute encrypted dot product: <enc(x), w>.

        This enables homomorphic MoE routing without decrypting features.

        Args:
            encrypted_features: Encrypted feature vector
            plaintext_weights: Plaintext weight vector

        Returns:
            Encrypted dot product result
        """
        if not self.config.enable_encryption or self.n2he_ctx is None:
            # Mock: just return encrypted sum
            return encrypted_features.ciphertexts[0]

        # Quantize weights
        scale = int(np.sqrt(self.config.quantization_scale))
        weights_int = np.clip(plaintext_weights * scale, -scale + 1, scale - 1).astype(np.int64)

        # Compute weighted sum homomorphically
        result = None
        for i, (ct_bytes, w) in enumerate(zip(encrypted_features.ciphertexts, weights_int)):
            ct = LWECiphertext.deserialize(ct_bytes)
            weighted = w * ct  # Scalar multiplication

            if result is None:
                result = weighted
            else:
                result = self.n2he_ctx.add(result, weighted)

        return result.serialize() if result else b''

    def request_encrypted_routing(
        self,
        encrypted_features: EncryptedFeatures,
        task_description: Optional[str] = None,
        max_skills: int = 3,
    ) -> EncryptedRoutingResponse:
        """
        Request encrypted MoE routing from cloud.

        In production, this would send to cloud service.
        Here we simulate the response.

        Args:
            encrypted_features: Encrypted features
            task_description: Optional task description
            max_skills: Maximum skills to return

        Returns:
            EncryptedRoutingResponse
        """
        start_time = time.time()

        # Simulate cloud routing
        # In production, this would be an API call

        # Mock skill selection based on hash
        hash_int = int(encrypted_features.hash[:8], 16)
        np.random.seed(hash_int)

        skill_ids = [f"skill_{i:03d}" for i in np.random.choice(100, max_skills, replace=False)]

        # Generate mock encrypted weights
        weights = np.random.dirichlet(np.ones(max_skills))
        encrypted_weights = self.encrypt_weights(weights)

        routing_time = (time.time() - start_time) * 1000

        return EncryptedRoutingResponse(
            encrypted_weights=encrypted_weights,
            skill_ids=skill_ids,
            routing_time_ms=routing_time,
        )

    def verify_integrity(
        self,
        encrypted: EncryptedFeatures,
    ) -> bool:
        """
        Verify integrity of encrypted features.

        Args:
            encrypted: Encrypted features to verify

        Returns:
            True if integrity check passes
        """
        if not encrypted.ciphertexts:
            return False

        # Recompute hash
        hash_input = b''.join(encrypted.ciphertexts[:10])
        computed_hash = hashlib.sha256(hash_input).hexdigest()[:16]

        return computed_hash == encrypted.hash

    def get_statistics(self) -> Dict[str, Any]:
        """Get privacy wrapper statistics."""
        avg_enc_time = 0.0
        avg_dec_time = 0.0

        if self.stats['features_encrypted'] > 0:
            avg_enc_time = self.stats['total_encryption_time_ms'] / self.stats['features_encrypted']
        if self.stats['features_decrypted'] > 0:
            avg_dec_time = self.stats['total_decryption_time_ms'] / self.stats['features_decrypted']

        return {
            **self.stats,
            'avg_encryption_time_ms': avg_enc_time,
            'avg_decryption_time_ms': avg_dec_time,
            'encryption_enabled': self.config.enable_encryption,
            'n2he_available': HAS_N2HE,
            'security_bits': self.config.security_bits,
        }


# =============================================================================
# Encrypted Pipeline Integration
# =============================================================================

class EncryptedPerceptionPipeline:
    """
    Full encrypted perception pipeline combining Meta AI models with N2HE.

    Pipeline:
    1. DINOv3 encodes image -> features
    2. SAM3 segments objects (optional)
    3. V-JEPA 2 predicts future (optional)
    4. Encrypt combined features
    5. Send to cloud for MoE routing
    6. Decrypt skill weights
    7. Execute skills locally
    """

    def __init__(
        self,
        privacy_config: PrivacyConfig = None,
    ):
        self.privacy_wrapper = MetaAIPrivacyWrapper(privacy_config)

        # Model references (set externally)
        self.dinov3 = None
        self.sam3 = None
        self.vjepa2 = None

        # Statistics
        self.stats = {
            'pipeline_runs': 0,
            'total_time_ms': 0.0,
        }

    def set_models(self, dinov3=None, sam3=None, vjepa2=None):
        """Set model references."""
        self.dinov3 = dinov3
        self.sam3 = sam3
        self.vjepa2 = vjepa2

    def process_frame(
        self,
        frame: np.ndarray,
        task_description: Optional[str] = None,
        include_segmentation: bool = False,
        include_prediction: bool = False,
    ) -> Dict[str, Any]:
        """
        Process frame through encrypted pipeline.

        Args:
            frame: Input frame [H, W, 3]
            task_description: Optional task for routing
            include_segmentation: Run SAM3 segmentation
            include_prediction: Run V-JEPA 2 prediction

        Returns:
            Dictionary with results
        """
        start_time = time.time()
        results = {}

        # 1. DINOv3 encoding
        if self.dinov3 is not None:
            dinov3_features = self.dinov3.encode(frame)
            features = dinov3_features.global_features.squeeze()
            results['dinov3_features'] = features
        else:
            # Mock features
            features = np.random.randn(1024).astype(np.float32)
            results['dinov3_features'] = features

        # 2. SAM3 segmentation (optional)
        if include_segmentation and self.sam3 is not None:
            if task_description:
                seg_result = self.sam3.segment_text(frame, task_description)
            else:
                seg_result = self.sam3.segment_all_objects(frame)
            results['segmentation'] = seg_result

        # 3. V-JEPA 2 prediction (optional)
        if include_prediction and self.vjepa2 is not None:
            self.vjepa2.add_frame(frame)
            prediction = self.vjepa2.predict()
            results['prediction'] = prediction

            # Combine features
            if prediction.future_embeddings is not None:
                combined = np.concatenate([
                    features,
                    prediction.future_embeddings[0]  # First future step
                ])
                features = combined

        # 4. Encrypt features
        encrypted = self.privacy_wrapper.encrypt_features(features, source_model="combined")
        results['encrypted_features'] = encrypted

        # 5. Request encrypted routing
        routing_response = self.privacy_wrapper.request_encrypted_routing(
            encrypted,
            task_description=task_description,
        )
        results['routing_response'] = routing_response

        # 6. Decrypt skill weights
        skill_weights = self.privacy_wrapper.decrypt_weights(routing_response.encrypted_weights)
        results['skill_weights'] = skill_weights
        results['skill_ids'] = routing_response.skill_ids

        total_time = (time.time() - start_time) * 1000

        self.stats['pipeline_runs'] += 1
        self.stats['total_time_ms'] += total_time

        results['total_time_ms'] = total_time

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_time = 0.0
        if self.stats['pipeline_runs'] > 0:
            avg_time = self.stats['total_time_ms'] / self.stats['pipeline_runs']

        return {
            **self.stats,
            'avg_time_ms': avg_time,
            'privacy_stats': self.privacy_wrapper.get_statistics(),
        }


# =============================================================================
# Testing
# =============================================================================

def test_privacy_wrapper():
    """Test privacy wrapper."""
    print("\n" + "=" * 60)
    print("META AI PRIVACY WRAPPER TEST")
    print("=" * 60)

    # Create wrapper
    config = PrivacyConfig(
        security_bits=128,
        enable_encryption=True,
    )
    wrapper = MetaAIPrivacyWrapper(config)

    print("\n1. Feature Encryption")
    print("-" * 40)

    # Test features
    features = np.random.randn(1024).astype(np.float32)
    features = features / np.abs(features).max()  # Normalize

    encrypted = wrapper.encrypt_features(features, source_model="dinov3")
    print(f"   Original shape: {features.shape}")
    print(f"   Encrypted size: {encrypted.size_bytes / 1024:.2f} KB")
    print(f"   Encryption time: {encrypted.encryption_time_ms:.2f}ms")
    print(f"   Hash: {encrypted.hash}")

    print("\n2. Feature Decryption")
    print("-" * 40)

    decrypted = wrapper.decrypt_features(encrypted)
    print(f"   Decrypted shape: {decrypted.shape}")

    # Check reconstruction error
    mse = np.mean((features - decrypted) ** 2)
    print(f"   MSE: {mse:.6f}")

    print("\n3. Integrity Verification")
    print("-" * 40)

    is_valid = wrapper.verify_integrity(encrypted)
    print(f"   Integrity valid: {is_valid}")

    print("\n4. Encrypted Routing")
    print("-" * 40)

    response = wrapper.request_encrypted_routing(
        encrypted,
        task_description="pick up the red cup",
        max_skills=3,
    )
    print(f"   Skill IDs: {response.skill_ids}")
    print(f"   Routing time: {response.routing_time_ms:.2f}ms")

    # Decrypt weights
    weights = wrapper.decrypt_weights(response.encrypted_weights)
    print(f"   Decrypted weights: {weights}")
    print(f"   Weights sum: {weights.sum():.3f}")

    print("\n5. Homomorphic Dot Product")
    print("-" * 40)

    # Test homomorphic computation
    query_weights = np.random.randn(encrypted.feature_dim).astype(np.float32)
    query_weights = query_weights / np.abs(query_weights).max()

    result_encrypted = wrapper.homomorphic_dot_product(encrypted, query_weights)
    print(f"   Encrypted result size: {len(result_encrypted)} bytes")

    print("\n6. Encrypted Pipeline")
    print("-" * 40)

    pipeline = EncryptedPerceptionPipeline(config)

    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results = pipeline.process_frame(
        test_frame,
        task_description="grasp the screwdriver",
    )

    print(f"   Pipeline total time: {results['total_time_ms']:.2f}ms")
    print(f"   Skills selected: {results['skill_ids']}")
    print(f"   Skill weights: {results['skill_weights']}")

    print("\n7. Statistics")
    print("-" * 40)

    stats = wrapper.get_statistics()
    print(f"   Features encrypted: {stats['features_encrypted']}")
    print(f"   Avg encryption time: {stats['avg_encryption_time_ms']:.2f}ms")
    print(f"   Security bits: {stats['security_bits']}")

    print("\n" + "=" * 60)
    print("PRIVACY WRAPPER TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_privacy_wrapper()

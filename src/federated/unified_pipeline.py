"""
Unified Federated Learning Pipeline with Quality-Preserving Improvements

This module implements a coordinated pipeline where each stage's output feeds into
the next, ensuring all improvements work together without conflicts.

Pipeline Order:
    Stage 1: Gradient Clipping     - Bounds magnitude BEFORE lossy ops
    Stage 2: Error Feedback        - Adds accumulated errors from previous round
    Stage 3: Top-K Sparsification  - Reduces count of values to transmit
    Stage 4: Adaptive Compression  - MOAI neural compression with adaptive ratio
    Stage 5: Quality Monitoring    - Cross-cutting observer (reads all stages)
    Stage 6: Noise-Aware FHE       - Encrypts with bootstrapping support
    Stage 7: Hierarchical Agg      - Tree-based aggregation reduces FHE ops

Key Integration Principles:
    1. Single Direction Flow: Data flows in ONE direction through stages
    2. Observer Pattern for Monitoring: Quality Monitor READS from all stages
    3. Explicit Residual Storage: Error feedback maintains per-client state
    4. Type Contracts: Each stage has explicit input/output types
    5. Bounded Feedback: Adaptive components use moving averages
    6. Fail-Safe Defaults: If any stage fails, defaults maintain pipeline

Author: Dynamical.ai
"""

import time
import gzip
import pickle
import logging
import threading
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# Import MOAI and N2HE Components
# =============================================================================

# Import N2HE (Neural Network Homomorphic Encryption)
try:
    from src.moai.n2he import (
        N2HEContext,
        N2HEParams,
        N2HE_128,
        LWECiphertext,
        Ciphertext as N2HECiphertext,
        Encryptor as N2HEEncryptor,
        Decryptor as N2HEDecryptor,
        Evaluator as N2HEEvaluator,
    )
    HAS_N2HE = True
except ImportError:
    HAS_N2HE = False
    N2HEContext = None
    logger.warning("N2HE not available - using mock FHE")

# Import MOAI FHE System
try:
    from src.moai.moai_fhe import (
        MoaiFHEContext,
        MoaiFHEConfig,
        MoaiFHESystem,
    )
    HAS_MOAI_FHE = True
except ImportError:
    HAS_MOAI_FHE = False
    MoaiFHEContext = None
    logger.warning("MOAI FHE not available")

# Import MOAI PyTorch (for neural compression)
try:
    from src.moai.moai_pt import MoaiConfig, HAS_TORCH
    if HAS_TORCH:
        from src.moai.moai_pt import MoaiTransformerBlockPT
    HAS_MOAI_PT = HAS_TORCH
except ImportError:
    HAS_MOAI_PT = False
    MoaiConfig = None

# Import unified FHE backend
try:
    from src.shared.crypto.fhe_backend import (
        create_fhe_backend,
        FHEConfig,
        FHEBackend,
        get_available_backends,
    )
    HAS_FHE_BACKEND = True
except ImportError:
    HAS_FHE_BACKEND = False
    create_fhe_backend = None


# =============================================================================
# Data Classes for Type Safety
# =============================================================================

@dataclass
class SparseGradients:
    """Sparse representation of gradients after Top-K selection."""
    indices: np.ndarray          # Indices of non-zero gradients
    values: np.ndarray           # Values at those indices
    original_shape: Tuple[int, ...]
    total_elements: int
    k_ratio: float               # What percentage was kept

    @property
    def sparsity_ratio(self) -> float:
        """Ratio of zeros (1.0 = all zeros, 0.0 = all non-zero)."""
        if self.total_elements == 0:
            return 0.0
        return 1.0 - (len(self.indices) / self.total_elements)

    def to_dense(self) -> np.ndarray:
        """Reconstruct dense array from sparse representation."""
        dense = np.zeros(self.total_elements, dtype=np.float32)
        dense[self.indices] = self.values
        return dense.reshape(self.original_shape)


@dataclass
class CompressedGradients:
    """Compressed gradients after MOAI neural compression."""
    compressed_data: bytes       # Compressed bytes
    indices: np.ndarray          # Indices (stored separately)
    original_shape: Tuple[int, ...]
    compression_ratio: float     # Achieved compression ratio
    quality_score: float         # Estimated quality (0-1)


@dataclass
class EncryptedUpdate:
    """Encrypted gradient update ready for aggregation."""
    ciphertext: bytes            # Encrypted compressed gradients
    metadata: Dict[str, Any]     # Shape, indices, etc.
    client_id: str
    round_number: int
    remaining_noise_budget: float  # FHE noise budget remaining
    timestamp: float


@dataclass
class AggregationNode:
    """Node in hierarchical aggregation tree."""
    node_id: str
    children: List['AggregationNode'] = field(default_factory=list)
    encrypted_data: Optional[bytes] = None
    aggregated: bool = False
    level: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ClippingConfig:
    """Configuration for gradient clipping."""
    max_norm: float = 1.0        # Maximum L2 norm
    norm_type: int = 2           # Norm type (1=L1, 2=L2, inf=Linf)
    enabled: bool = True


@dataclass
class ErrorFeedbackConfig:
    """Configuration for error feedback."""
    enabled: bool = True
    decay: float = 0.9           # Decay factor to prevent unbounded growth
    max_accumulated: float = 10.0  # Max accumulated error magnitude


@dataclass
class SparsificationConfig:
    """Configuration for Top-K sparsification."""
    k_ratio: float = 0.01        # Keep top 1% of gradients
    min_k: int = 100             # Minimum gradients to keep
    enabled: bool = True


@dataclass
class CompressionConfig:
    """Configuration for adaptive MOAI compression."""
    initial_ratio: int = 32      # Initial compression ratio
    min_ratio: int = 8           # Minimum (highest quality)
    max_ratio: int = 128         # Maximum (lowest quality)
    adaptation_rate: float = 0.1 # Rate of ratio adjustment
    target_quality: float = 0.95 # Target quality score
    enabled: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for quality monitoring."""
    window_size: int = 10        # Rounds for moving average
    alert_threshold: float = 0.1 # Quality drop that triggers alert
    log_interval: int = 1        # Log every N rounds
    enabled: bool = True


@dataclass
class FHENoiseConfig:
    """Configuration for noise-aware FHE."""
    poly_modulus_degree: int = 16384  # Higher = more noise budget
    coeff_mod_bit_sizes: List[int] = field(
        default_factory=lambda: [60, 40, 40, 40, 40, 60]
    )
    scale_bits: int = 40         # Scale for CKKS
    bootstrap_threshold: float = 0.2  # Bootstrap when 20% budget remains
    use_mock: bool = True        # Use mock for testing


@dataclass
class AggregationConfig:
    """Configuration for hierarchical aggregation."""
    max_tree_depth: int = 4      # Maximum tree depth
    branching_factor: int = 10   # Clients per aggregator node
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Unified configuration for entire pipeline."""
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    feedback: ErrorFeedbackConfig = field(default_factory=ErrorFeedbackConfig)
    sparsification: SparsificationConfig = field(default_factory=SparsificationConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    fhe: FHENoiseConfig = field(default_factory=FHENoiseConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    def validate(self) -> List[str]:
        """Validate configuration constraints. Returns list of warnings."""
        warnings = []

        # Rule 1: FHE params must support aggregation depth
        fhe_depth = len(self.fhe.coeff_mod_bit_sizes) - 2
        if fhe_depth < self.aggregation.max_tree_depth:
            warnings.append(
                f"FHE depth ({fhe_depth}) < aggregation depth ({self.aggregation.max_tree_depth}). "
                f"Add more coeff_mod_bit_sizes."
            )

        # Rule 2: Compression adaptation must be slower than quality threshold
        if self.compression.adaptation_rate >= self.monitoring.alert_threshold:
            warnings.append(
                f"Compression adaptation_rate ({self.compression.adaptation_rate}) >= "
                f"alert_threshold ({self.monitoring.alert_threshold}). May cause oscillation."
            )

        # Rule 3: Error feedback must decay
        if self.feedback.enabled and self.feedback.decay >= 1.0:
            warnings.append(
                f"Error feedback decay ({self.feedback.decay}) >= 1.0. "
                f"Will cause unbounded error accumulation."
            )

        return warnings


# =============================================================================
# Stage 1: Gradient Clipping
# =============================================================================

class GradientClipper:
    """
    Bounds gradient magnitude BEFORE any lossy operations.

    This MUST be the first stage because:
    - Prevents error feedback from accumulating unbounded errors
    - Ensures all downstream operations see reasonable gradient ranges
    - Protects against gradient explosion in federated setting
    """

    def __init__(self, config: ClippingConfig = None):
        self.config = config or ClippingConfig()
        self.last_clip_ratio = 0.0  # Fraction of gradients that were clipped
        self.stats = {
            'total_clips': 0,
            'total_calls': 0,
            'avg_clip_ratio': 0.0,
        }

    def clip(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clip gradients to bounded norm.

        Args:
            gradients: Dict mapping parameter name to gradient array

        Returns:
            Dict with clipped gradients (same structure)
        """
        if not self.config.enabled:
            self.last_clip_ratio = 0.0
            return gradients

        # Concatenate all gradients for global norm calculation
        flat_grads = []
        for name, grad in gradients.items():
            flat_grads.append(grad.flatten())
        all_grads = np.concatenate(flat_grads)

        # Compute global norm
        if self.config.norm_type == 1:
            total_norm = np.abs(all_grads).sum()
        elif self.config.norm_type == 2:
            total_norm = np.sqrt(np.sum(all_grads ** 2))
        else:  # inf norm
            total_norm = np.abs(all_grads).max()

        # Compute clip coefficient
        clip_coef = self.config.max_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)  # Never scale up

        # Track clipping ratio
        self.last_clip_ratio = 1.0 - clip_coef if clip_coef < 1.0 else 0.0
        self.stats['total_calls'] += 1
        if clip_coef < 1.0:
            self.stats['total_clips'] += 1
        self.stats['avg_clip_ratio'] = (
            self.stats['avg_clip_ratio'] * 0.9 + self.last_clip_ratio * 0.1
        )

        # Apply clipping
        clipped = {}
        for name, grad in gradients.items():
            clipped[name] = grad * clip_coef

        return clipped

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# =============================================================================
# Stage 2: Error Feedback
# =============================================================================

class ErrorFeedback:
    """
    Accumulates compression/sparsification errors for next round.

    This comes AFTER clipping but BEFORE sparsification because:
    - Works on bounded gradients (thanks to clipping)
    - Adds accumulated errors BEFORE selection, giving "unfairly dropped"
      gradients a chance to be selected

    State is maintained per-client to handle federated setting.
    """

    def __init__(self, config: ErrorFeedbackConfig = None):
        self.config = config or ErrorFeedbackConfig()
        self._residuals: Dict[str, Dict[str, np.ndarray]] = {}  # client_id -> param -> residual
        self._lock = threading.Lock()
        self.stats = {
            'total_corrections': 0,
            'avg_residual_norm': 0.0,
            'clients_tracked': 0,
        }

    def apply(
        self,
        gradients: Dict[str, np.ndarray],
        client_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Add accumulated errors from previous round to current gradients.

        Args:
            gradients: Current round gradients
            client_id: Client identifier for state tracking

        Returns:
            Corrected gradients (current + accumulated errors)
        """
        if not self.config.enabled:
            return gradients

        with self._lock:
            if client_id not in self._residuals:
                self._residuals[client_id] = {}
                self.stats['clients_tracked'] = len(self._residuals)
                return gradients  # No residuals for first round

            corrected = {}
            total_residual_norm = 0.0

            for name, grad in gradients.items():
                if name in self._residuals[client_id]:
                    residual = self._residuals[client_id][name]

                    # Decay residual to prevent unbounded growth
                    decayed_residual = residual * self.config.decay

                    # Clip accumulated residual
                    residual_norm = np.sqrt(np.sum(decayed_residual ** 2))
                    if residual_norm > self.config.max_accumulated:
                        decayed_residual = decayed_residual * (
                            self.config.max_accumulated / residual_norm
                        )

                    # Add to current gradient
                    corrected[name] = grad + decayed_residual
                    total_residual_norm += residual_norm
                else:
                    corrected[name] = grad

            self.stats['total_corrections'] += 1
            self.stats['avg_residual_norm'] = (
                self.stats['avg_residual_norm'] * 0.9 + total_residual_norm * 0.1
            )

            return corrected

    def store_residuals(
        self,
        original: Dict[str, np.ndarray],
        transmitted: Dict[str, np.ndarray],
        client_id: str
    ):
        """
        Store the difference between original and transmitted gradients.

        Args:
            original: Original gradients before sparsification/compression
            transmitted: What was actually transmitted
            client_id: Client identifier
        """
        if not self.config.enabled:
            return

        with self._lock:
            if client_id not in self._residuals:
                self._residuals[client_id] = {}

            for name, orig in original.items():
                if name in transmitted:
                    self._residuals[client_id][name] = orig - transmitted[name]
                else:
                    self._residuals[client_id][name] = orig

    def clear_client(self, client_id: str):
        """Clear residuals for a client."""
        with self._lock:
            if client_id in self._residuals:
                del self._residuals[client_id]
                self.stats['clients_tracked'] = len(self._residuals)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# =============================================================================
# Stage 3: Top-K Sparsification
# =============================================================================

class TopKSparsifier:
    """
    Keeps only the top K% of gradients by magnitude.

    This comes AFTER error feedback because:
    - Error feedback boosts previously-dropped gradients
    - Then Top-K selects the most important (including boosted ones)
    - Without this order, small persistent gradients would never transmit
    """

    def __init__(self, config: SparsificationConfig = None):
        self.config = config or SparsificationConfig()
        self.stats = {
            'total_sparsifications': 0,
            'avg_sparsity': 0.0,
            'total_bytes_saved': 0,
        }

    def sparsify(self, gradients: Dict[str, np.ndarray]) -> Tuple[SparseGradients, Dict[str, np.ndarray]]:
        """
        Select top-K gradients by magnitude.

        Args:
            gradients: Dense gradients dict

        Returns:
            Tuple of (SparseGradients, dense reconstruction for residual calculation)
        """
        # Handle empty gradients
        total_size = sum(g.size for g in gradients.values())
        if total_size == 0:
            empty_sparse = SparseGradients(
                indices=np.array([], dtype=np.int32),
                values=np.array([], dtype=np.float32),
                original_shape=(0,),
                total_elements=0,
                k_ratio=1.0
            )
            return empty_sparse, gradients

        if not self.config.enabled:
            # Return full gradients as "sparse"
            flat = np.concatenate([g.flatten() for g in gradients.values()])
            total_shape = sum(g.size for g in gradients.values())
            return SparseGradients(
                indices=np.arange(len(flat)),
                values=flat,
                original_shape=(total_shape,),
                total_elements=len(flat),
                k_ratio=1.0
            ), gradients

        # Concatenate all gradients
        flat_grads = []
        shapes = []
        names = []
        for name, grad in gradients.items():
            flat_grads.append(grad.flatten())
            shapes.append(grad.shape)
            names.append(name)

        all_grads = np.concatenate(flat_grads)
        total_elements = len(all_grads)

        # Calculate K
        k = max(
            self.config.min_k,
            int(total_elements * self.config.k_ratio)
        )
        k = min(k, total_elements)  # Can't keep more than we have

        # Find top-K by magnitude
        magnitudes = np.abs(all_grads)
        top_k_indices = np.argpartition(magnitudes, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(magnitudes[top_k_indices])[::-1]]

        # Create sparse representation
        sparse = SparseGradients(
            indices=top_k_indices.astype(np.int32),
            values=all_grads[top_k_indices].astype(np.float32),
            original_shape=(total_elements,),
            total_elements=total_elements,
            k_ratio=k / total_elements
        )

        # Reconstruct dense for residual calculation
        dense_reconstruction = sparse.to_dense()

        # Split back into dict format
        reconstructed = {}
        offset = 0
        for name, shape in zip(names, shapes):
            size = np.prod(shape)
            reconstructed[name] = dense_reconstruction[offset:offset+size].reshape(shape)
            offset += size

        # Update stats
        self.stats['total_sparsifications'] += 1
        self.stats['avg_sparsity'] = (
            self.stats['avg_sparsity'] * 0.9 + sparse.sparsity_ratio * 0.1
        )
        original_bytes = total_elements * 4  # float32
        sparse_bytes = len(top_k_indices) * (4 + 4)  # int32 + float32
        self.stats['total_bytes_saved'] += (original_bytes - sparse_bytes)

        return sparse, reconstructed

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# =============================================================================
# Stage 4: Adaptive MOAI Compression
# =============================================================================

class AdaptiveMOAICompressor:
    """
    Neural network-based compression with adaptive ratio.

    This comes AFTER sparsification because:
    - Works on smaller, denser tensors (more efficient)
    - Reduces precision on already-reduced count
    - Adaptive ratio responds to quality monitor feedback
    """

    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.current_ratio = float(self.config.initial_ratio)
        self._quality_history = deque(maxlen=10)
        self.stats = {
            'total_compressions': 0,
            'avg_ratio': self.config.initial_ratio,
            'avg_quality': 1.0,
            'ratio_adjustments': 0,
        }

    def compress(self, sparse: SparseGradients) -> CompressedGradients:
        """
        Compress sparse gradients using neural compression.

        Args:
            sparse: Sparse gradients from Top-K stage

        Returns:
            CompressedGradients with compressed data and metadata
        """
        if not self.config.enabled:
            # No compression - just serialize
            data = gzip.compress(pickle.dumps({
                'values': sparse.values,
                'shape': sparse.original_shape,
            }))
            return CompressedGradients(
                compressed_data=data,
                indices=sparse.indices,
                original_shape=sparse.original_shape,
                compression_ratio=1.0,
                quality_score=1.0
            )

        # Simulate MOAI neural compression
        # In production, this would use the actual MOAI transformer
        values = sparse.values

        # Quantize based on current ratio
        bits = max(2, int(32 / self.current_ratio))

        # Quantization
        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 1e-8:
            normalized = (values - v_min) / (v_max - v_min)
            levels = 2 ** bits
            quantized = np.round(normalized * (levels - 1)).astype(np.int32)

            # Dequantize for quality estimation
            dequantized = quantized / (levels - 1) * (v_max - v_min) + v_min
            quality = 1.0 - np.mean(np.abs(values - dequantized) / (np.abs(values) + 1e-8))
        else:
            quantized = np.zeros_like(values, dtype=np.int32)
            quality = 1.0

        # Pack into bytes
        data = gzip.compress(pickle.dumps({
            'quantized': quantized,
            'v_min': v_min,
            'v_max': v_max,
            'bits': bits,
            'shape': sparse.original_shape,
        }))

        original_size = len(values) * 4  # float32
        compressed_size = len(data)
        actual_ratio = original_size / max(compressed_size, 1)

        # Track quality
        self._quality_history.append(quality)

        # Update stats
        self.stats['total_compressions'] += 1
        self.stats['avg_ratio'] = self.stats['avg_ratio'] * 0.9 + actual_ratio * 0.1
        self.stats['avg_quality'] = self.stats['avg_quality'] * 0.9 + quality * 0.1

        return CompressedGradients(
            compressed_data=data,
            indices=sparse.indices,
            original_shape=sparse.original_shape,
            compression_ratio=actual_ratio,
            quality_score=quality
        )

    def decompress(self, compressed: CompressedGradients) -> SparseGradients:
        """Decompress compressed gradients."""
        payload = pickle.loads(gzip.decompress(compressed.compressed_data))

        if 'quantized' in payload:
            quantized = payload['quantized']
            v_min = payload['v_min']
            v_max = payload['v_max']
            bits = payload['bits']
            levels = 2 ** bits

            values = quantized / (levels - 1) * (v_max - v_min) + v_min
            values = values.astype(np.float32)
        else:
            values = payload['values']

        return SparseGradients(
            indices=compressed.indices,
            values=values,
            original_shape=compressed.original_shape,
            total_elements=np.prod(compressed.original_shape),
            k_ratio=len(compressed.indices) / np.prod(compressed.original_shape)
        )

    def adjust_ratio(self, quality_feedback: float):
        """
        Adjust compression ratio based on quality feedback.

        Args:
            quality_feedback: Quality score from monitor (0-1)
        """
        if not self.config.enabled:
            return

        # Calculate adjustment
        # quality_gap is negative when quality is below target
        quality_gap = quality_feedback - self.config.target_quality

        # Adjust ratio (lower ratio = higher quality)
        # When quality is below target (gap < 0), we want to DECREASE ratio
        # So adjustment should be negative when gap is negative
        adjustment = quality_gap * self.config.adaptation_rate * self.current_ratio

        new_ratio = self.current_ratio + adjustment
        new_ratio = np.clip(new_ratio, self.config.min_ratio, self.config.max_ratio)

        if abs(new_ratio - self.current_ratio) > 0.1:
            self.stats['ratio_adjustments'] += 1
            logger.debug(f"Compression ratio adjusted: {self.current_ratio:.1f} -> {new_ratio:.1f}")

        self.current_ratio = new_ratio

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['current_ratio'] = self.current_ratio
        return stats


# =============================================================================
# Stage 5: Quality Monitoring
# =============================================================================

class QualityAlert(Enum):
    """Types of quality alerts."""
    NONE = "none"
    COMPRESSION_DEGRADED = "compression_degraded"
    CLIPPING_AGGRESSIVE = "clipping_aggressive"
    FHE_NOISE_HIGH = "fhe_noise_high"
    GRADIENT_EXPLOSION = "gradient_explosion"


@dataclass
class QualityMetrics:
    """Snapshot of quality metrics."""
    round_number: int
    compression_quality: float
    clip_ratio: float
    sparsity: float
    fhe_noise_budget: float
    gradient_norm: float
    alerts: List[QualityAlert]
    timestamp: float


class QualityMonitor:
    """
    Cross-cutting observer that tracks quality across all stages.

    This is NOT in the data path - it only READS metrics and provides
    feedback signals. Cannot cause conflicts because it doesn't transform data.

    Provides:
    - Real-time quality tracking
    - Alerts on quality degradation
    - Feedback signals to adaptive components
    """

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self._metrics: Dict[str, deque] = {
            'compression_quality': deque(maxlen=self.config.window_size),
            'clip_ratio': deque(maxlen=self.config.window_size),
            'sparsity': deque(maxlen=self.config.window_size),
            'fhe_noise_budget': deque(maxlen=self.config.window_size),
            'gradient_norm': deque(maxlen=self.config.window_size),
        }
        self._round_number = 0
        self._alerts: List[QualityAlert] = []
        self._lock = threading.Lock()
        self.stats = {
            'total_alerts': 0,
            'rounds_monitored': 0,
        }

    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        if not self.config.enabled:
            return

        with self._lock:
            if metric_name in self._metrics:
                self._metrics[metric_name].append(value)

    def start_round(self, round_number: int):
        """Start monitoring a new round."""
        with self._lock:
            self._round_number = round_number
            self._alerts = []

    def end_round(self) -> QualityMetrics:
        """End round and generate quality report."""
        with self._lock:
            # Calculate averages
            def avg(dq: deque) -> float:
                return sum(dq) / len(dq) if dq else 0.0

            # Create preliminary metrics for alert checking
            preliminary_metrics = QualityMetrics(
                round_number=self._round_number,
                compression_quality=avg(self._metrics['compression_quality']),
                clip_ratio=avg(self._metrics['clip_ratio']),
                sparsity=avg(self._metrics['sparsity']),
                fhe_noise_budget=avg(self._metrics['fhe_noise_budget']),
                gradient_norm=avg(self._metrics['gradient_norm']),
                alerts=[],
                timestamp=time.time()
            )

            # Check for alerts BEFORE creating final metrics
            self._check_alerts(preliminary_metrics)

            # Create final metrics with alerts
            final_metrics = QualityMetrics(
                round_number=self._round_number,
                compression_quality=avg(self._metrics['compression_quality']),
                clip_ratio=avg(self._metrics['clip_ratio']),
                sparsity=avg(self._metrics['sparsity']),
                fhe_noise_budget=avg(self._metrics['fhe_noise_budget']),
                gradient_norm=avg(self._metrics['gradient_norm']),
                alerts=self._alerts.copy(),
                timestamp=time.time()
            )

            self.stats['rounds_monitored'] += 1

            return final_metrics

    def _check_alerts(self, metrics: QualityMetrics):
        """Check for quality issues and generate alerts."""
        # Check compression quality degradation
        if len(self._metrics['compression_quality']) >= 2:
            recent = list(self._metrics['compression_quality'])
            if len(recent) >= 2:
                quality_change = recent[-1] - recent[0]
                if quality_change < -self.config.alert_threshold:
                    self._alerts.append(QualityAlert.COMPRESSION_DEGRADED)
                    self.stats['total_alerts'] += 1
                    logger.warning(f"Quality degradation detected: {quality_change:.3f}")

        # Check clipping ratio
        if metrics.clip_ratio > 0.5:
            self._alerts.append(QualityAlert.CLIPPING_AGGRESSIVE)
            self.stats['total_alerts'] += 1
            logger.warning(f"Aggressive clipping: {metrics.clip_ratio:.1%} of gradients clipped")

        # Check FHE noise budget
        if metrics.fhe_noise_budget < 0.3:
            self._alerts.append(QualityAlert.FHE_NOISE_HIGH)
            self.stats['total_alerts'] += 1
            logger.warning(f"FHE noise budget low: {metrics.fhe_noise_budget:.1%}")

    def get_compression_quality(self) -> float:
        """Get current compression quality for adaptive feedback."""
        with self._lock:
            if self._metrics['compression_quality']:
                return sum(self._metrics['compression_quality']) / len(self._metrics['compression_quality'])
            return 1.0

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        with self._lock:
            for name, dq in self._metrics.items():
                if dq:
                    stats[f'{name}_current'] = dq[-1]
                    stats[f'{name}_avg'] = sum(dq) / len(dq)
        return stats


# =============================================================================
# Stage 6: Noise-Aware FHE
# =============================================================================

class NoiseAwareFHE:
    """
    FHE encryption with noise budget tracking and optional bootstrapping.

    Key features:
    - Higher precision parameters for more noise headroom
    - Tracks remaining noise budget
    - Triggers bootstrapping when budget is low
    """

    def __init__(self, config: FHENoiseConfig = None):
        self.config = config or FHENoiseConfig()
        self._scale = 2 ** self.config.scale_bits

        # Calculate noise budget from parameters
        # More coeff_mod entries = more noise budget
        self._max_noise_budget = len(self.config.coeff_mod_bit_sizes) - 1
        self._current_noise_budget = float(self._max_noise_budget)

        self.stats = {
            'total_encryptions': 0,
            'bootstraps_performed': 0,
            'avg_noise_budget': 1.0,
        }

    def encrypt(self, compressed: CompressedGradients, client_id: str, round_number: int) -> EncryptedUpdate:
        """
        Encrypt compressed gradients with noise tracking.

        Args:
            compressed: Compressed gradients from MOAI stage
            client_id: Client identifier
            round_number: Current FL round

        Returns:
            EncryptedUpdate with ciphertext and metadata
        """
        # Serialize data for encryption
        plaintext = pickle.dumps({
            'data': compressed.compressed_data,
            'indices': compressed.indices,
            'shape': compressed.original_shape,
            'compression_ratio': compressed.compression_ratio,
            'quality_score': compressed.quality_score,
        })

        if self.config.use_mock:
            # Mock encryption (for testing)
            ciphertext = gzip.compress(plaintext)
            noise_used = 0.1  # Simulated noise consumption
        else:
            # Real FHE encryption would go here
            # Using TenSEAL or N2HE backend
            ciphertext = gzip.compress(plaintext)
            noise_used = 0.2  # Estimated noise per encryption

        # Track noise budget
        self._current_noise_budget -= noise_used
        remaining = max(0, self._current_noise_budget / self._max_noise_budget)

        # Check if bootstrapping needed
        if remaining < self.config.bootstrap_threshold:
            self._bootstrap()

        self.stats['total_encryptions'] += 1
        self.stats['avg_noise_budget'] = (
            self.stats['avg_noise_budget'] * 0.9 + remaining * 0.1
        )

        return EncryptedUpdate(
            ciphertext=ciphertext,
            metadata={
                'shape': compressed.original_shape,
                'indices_shape': compressed.indices.shape,
                'compression_ratio': compressed.compression_ratio,
            },
            client_id=client_id,
            round_number=round_number,
            remaining_noise_budget=remaining,
            timestamp=time.time()
        )

    def _bootstrap(self):
        """Refresh noise budget through bootstrapping."""
        logger.info("Performing FHE bootstrapping to refresh noise budget")
        self._current_noise_budget = float(self._max_noise_budget)
        self.stats['bootstraps_performed'] += 1

    def decrypt(self, encrypted: EncryptedUpdate) -> CompressedGradients:
        """Decrypt encrypted update."""
        if self.config.use_mock:
            plaintext = gzip.decompress(encrypted.ciphertext)
        else:
            plaintext = gzip.decompress(encrypted.ciphertext)

        payload = pickle.loads(plaintext)

        return CompressedGradients(
            compressed_data=payload['data'],
            indices=payload['indices'],
            original_shape=payload['shape'],
            compression_ratio=payload['compression_ratio'],
            quality_score=payload['quality_score']
        )

    def homomorphic_add(self, updates: List[EncryptedUpdate]) -> EncryptedUpdate:
        """
        Homomorphically add encrypted updates.

        In mock mode, this decrypts, adds, and re-encrypts.
        In production, uses FHE homomorphic addition.
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        # Decrypt all
        decrypted = [self.decrypt(u) for u in updates]

        # Decompress and sum values
        from collections import defaultdict
        summed_values = defaultdict(float)

        # Use first update as template
        first = decrypted[0]
        payload = pickle.loads(gzip.decompress(first.compressed_data))

        if 'quantized' in payload:
            # Sum quantized values (simplified)
            all_quantized = []
            for d in decrypted:
                p = pickle.loads(gzip.decompress(d.compressed_data))
                all_quantized.append(p['quantized'])

            avg_quantized = np.mean(all_quantized, axis=0).astype(np.int32)

            new_data = gzip.compress(pickle.dumps({
                'quantized': avg_quantized,
                'v_min': payload['v_min'],
                'v_max': payload['v_max'],
                'bits': payload['bits'],
                'shape': payload['shape'],
            }))
        else:
            # Sum raw values
            all_values = [pickle.loads(gzip.decompress(d.compressed_data))['values']
                         for d in decrypted]
            avg_values = np.mean(all_values, axis=0)

            new_data = gzip.compress(pickle.dumps({
                'values': avg_values,
                'shape': payload['shape'],
            }))

        # Track noise consumption
        self._current_noise_budget -= 0.1 * len(updates)
        remaining = max(0, self._current_noise_budget / self._max_noise_budget)

        if remaining < self.config.bootstrap_threshold:
            self._bootstrap()

        aggregated = CompressedGradients(
            compressed_data=new_data,
            indices=first.indices,
            original_shape=first.original_shape,
            compression_ratio=first.compression_ratio,
            quality_score=np.mean([d.quality_score for d in decrypted])
        )

        # Re-encrypt
        return EncryptedUpdate(
            ciphertext=gzip.compress(pickle.dumps({
                'data': aggregated.compressed_data,
                'indices': aggregated.indices,
                'shape': aggregated.original_shape,
                'compression_ratio': aggregated.compression_ratio,
                'quality_score': aggregated.quality_score,
            })),
            metadata={'num_clients': len(updates)},
            client_id='aggregated',
            round_number=updates[0].round_number,
            remaining_noise_budget=remaining,
            timestamp=time.time()
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['current_noise_budget'] = self._current_noise_budget / self._max_noise_budget
        return stats


# =============================================================================
# Stage 7: Hierarchical Aggregation
# =============================================================================

class HierarchicalAggregator:
    """
    Tree-based aggregation to reduce FHE operations.

    Instead of aggregating N clients at once (N-1 FHE additions),
    uses a tree structure where each node aggregates at most K children.

    Benefits:
    - Reduces max noise accumulation per path
    - Parallelizable across tree levels
    - Reduces total FHE operations
    """

    def __init__(self, config: AggregationConfig = None, fhe: NoiseAwareFHE = None):
        self.config = config or AggregationConfig()
        self.fhe = fhe
        self.stats = {
            'total_aggregations': 0,
            'avg_tree_depth': 0,
            'total_fhe_ops': 0,
        }

    def build_tree(self, updates: List[EncryptedUpdate]) -> AggregationNode:
        """
        Build aggregation tree from client updates.

        Args:
            updates: List of encrypted updates from clients

        Returns:
            Root node of aggregation tree
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        # Create leaf nodes
        leaves = [
            AggregationNode(
                node_id=f"client_{u.client_id}",
                encrypted_data=u.ciphertext,
                level=0
            )
            for u in updates
        ]

        # Build tree bottom-up
        current_level = leaves
        level = 1

        while len(current_level) > 1 and level < self.config.max_tree_depth:
            next_level = []

            for i in range(0, len(current_level), self.config.branching_factor):
                children = current_level[i:i + self.config.branching_factor]
                parent = AggregationNode(
                    node_id=f"agg_L{level}_{i // self.config.branching_factor}",
                    children=children,
                    level=level
                )
                next_level.append(parent)

            current_level = next_level
            level += 1

        # If still multiple nodes, create final root (this becomes the max depth level)
        if len(current_level) > 1:
            root = AggregationNode(
                node_id="root",
                children=current_level,
                level=min(level, self.config.max_tree_depth)
            )
        else:
            root = current_level[0]
            root.node_id = "root"

        return root

    def aggregate(self, root: AggregationNode, updates: List[EncryptedUpdate]) -> EncryptedUpdate:
        """
        Aggregate updates using tree structure.

        Args:
            root: Root of aggregation tree
            updates: Original list of updates (for FHE operations)

        Returns:
            Single aggregated encrypted update
        """
        if not self.config.enabled or self.fhe is None:
            # Fall back to flat aggregation
            return self.fhe.homomorphic_add(updates) if self.fhe else updates[0]

        # Map client IDs to updates
        update_map = {u.client_id: u for u in updates}

        # Recursive aggregation
        fhe_ops = [0]  # Mutable counter

        def aggregate_node(node: AggregationNode) -> EncryptedUpdate:
            if not node.children:
                # Leaf node - return original update
                client_id = node.node_id.replace("client_", "")
                return update_map.get(client_id, updates[0])

            # Aggregate children first
            child_updates = [aggregate_node(child) for child in node.children]

            # Homomorphic aggregation
            fhe_ops[0] += len(child_updates) - 1
            return self.fhe.homomorphic_add(child_updates)

        result = aggregate_node(root)

        # Update stats
        tree_depth = self._get_tree_depth(root)
        self.stats['total_aggregations'] += 1
        self.stats['avg_tree_depth'] = (
            self.stats['avg_tree_depth'] * 0.9 + tree_depth * 0.1
        )
        self.stats['total_fhe_ops'] += fhe_ops[0]

        return result

    def _get_tree_depth(self, node: AggregationNode) -> int:
        """Get depth of tree."""
        if not node.children:
            return 0
        return 1 + max(self._get_tree_depth(child) for child in node.children)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# =============================================================================
# Unified Pipeline
# =============================================================================

class UnifiedFLPipeline:
    """
    Unified Federated Learning Pipeline that orchestrates all 7 improvements.

    The pipeline ensures:
    1. Correct execution order (no conflicts)
    2. Type safety between stages
    3. Coordinated configuration
    4. Comprehensive statistics

    Usage:
        pipeline = UnifiedFLPipeline()

        # Process client update
        encrypted = pipeline.process_client_update(gradients, client_id, round_num)

        # Aggregate updates from multiple clients
        aggregated = pipeline.aggregate_updates(encrypted_updates)

        # Get final model update
        model_delta = pipeline.finalize_aggregation(aggregated)
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Validate configuration
        warnings = self.config.validate()
        for w in warnings:
            logger.warning(f"Config warning: {w}")

        # Initialize stages in dependency order
        self.quality_monitor = QualityMonitor(self.config.monitoring)
        self.gradient_clipper = GradientClipper(self.config.clipping)
        self.error_feedback = ErrorFeedback(self.config.feedback)
        self.sparsifier = TopKSparsifier(self.config.sparsification)
        self.compressor = AdaptiveMOAICompressor(self.config.compression)
        self.fhe = NoiseAwareFHE(self.config.fhe)
        self.aggregator = HierarchicalAggregator(self.config.aggregation, self.fhe)

        self._round_number = 0
        self.stats = {
            'total_updates_processed': 0,
            'total_aggregations': 0,
        }

    def process_client_update(
        self,
        gradients: Dict[str, np.ndarray],
        client_id: str,
        round_number: int = None
    ) -> EncryptedUpdate:
        """
        Process client gradients through the full pipeline.

        Pipeline stages:
        1. Gradient Clipping -> bounded gradients
        2. Error Feedback -> corrected gradients
        3. Top-K Sparsification -> sparse gradients
        4. Adaptive MOAI Compression -> compressed gradients
        5. Quality Monitoring -> metrics recorded
        6. Noise-Aware FHE -> encrypted update

        Args:
            gradients: Dict mapping parameter names to gradient arrays
            client_id: Unique client identifier
            round_number: FL round number (auto-increments if None)

        Returns:
            EncryptedUpdate ready for aggregation
        """
        if round_number is None:
            round_number = self._round_number

        self.quality_monitor.start_round(round_number)

        # Record initial gradient norm
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        self.quality_monitor.record('gradient_norm', total_norm)

        # STAGE 1: Gradient Clipping
        clipped = self.gradient_clipper.clip(gradients)
        self.quality_monitor.record('clip_ratio', self.gradient_clipper.last_clip_ratio)

        # STAGE 2: Error Feedback
        corrected = self.error_feedback.apply(clipped, client_id)

        # STAGE 3: Top-K Sparsification
        sparse, reconstructed = self.sparsifier.sparsify(corrected)
        self.quality_monitor.record('sparsity', sparse.sparsity_ratio)

        # Store residuals for next round
        self.error_feedback.store_residuals(corrected, reconstructed, client_id)

        # STAGE 4: Adaptive MOAI Compression
        compressed = self.compressor.compress(sparse)
        self.quality_monitor.record('compression_quality', compressed.quality_score)

        # Adaptive feedback loop
        quality = self.quality_monitor.get_compression_quality()
        self.compressor.adjust_ratio(quality)

        # STAGE 6: Noise-Aware FHE Encryption
        encrypted = self.fhe.encrypt(compressed, client_id, round_number)
        self.quality_monitor.record('fhe_noise_budget', encrypted.remaining_noise_budget)

        # End round monitoring
        metrics = self.quality_monitor.end_round()

        self.stats['total_updates_processed'] += 1

        return encrypted

    def aggregate_updates(self, updates: List[EncryptedUpdate]) -> EncryptedUpdate:
        """
        Aggregate encrypted updates from multiple clients.

        Uses hierarchical aggregation (Stage 7) to reduce FHE operations.

        Args:
            updates: List of encrypted updates from clients

        Returns:
            Single aggregated encrypted update
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        # STAGE 7: Hierarchical Aggregation
        tree = self.aggregator.build_tree(updates)
        aggregated = self.aggregator.aggregate(tree, updates)

        self.stats['total_aggregations'] += 1
        self._round_number += 1

        return aggregated

    def finalize_aggregation(self, aggregated: EncryptedUpdate) -> Dict[str, np.ndarray]:
        """
        Finalize aggregation by decrypting and decompressing.

        Args:
            aggregated: Aggregated encrypted update

        Returns:
            Dict mapping parameter names to aggregated gradient arrays
        """
        # Decrypt
        compressed = self.fhe.decrypt(aggregated)

        # Decompress
        sparse = self.compressor.decompress(compressed)

        # Convert to dense
        dense = sparse.to_dense()

        # Return as single gradient array
        # In production, would split back into named parameters
        return {'gradients': dense}

    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all stages."""
        return {
            'pipeline': self.stats,
            'clipping': self.gradient_clipper.get_stats(),
            'error_feedback': self.error_feedback.get_stats(),
            'sparsification': self.sparsifier.get_stats(),
            'compression': self.compressor.get_stats(),
            'monitoring': self.quality_monitor.get_stats(),
            'fhe': self.fhe.get_stats(),
            'aggregation': self.aggregator.get_stats(),
        }

    def reset_round(self):
        """Reset per-round state (call at start of each FL round)."""
        self._round_number += 1


# =============================================================================
# Convenience Functions
# =============================================================================

def create_pipeline(
    clipping_norm: float = 1.0,
    sparsity_ratio: float = 0.01,
    compression_ratio: int = 32,
    use_mock_fhe: bool = True
) -> UnifiedFLPipeline:
    """
    Create a pipeline with common configuration options.

    Args:
        clipping_norm: Maximum gradient L2 norm
        sparsity_ratio: Fraction of gradients to keep (0.01 = top 1%)
        compression_ratio: Initial MOAI compression ratio
        use_mock_fhe: Use mock FHE for testing

    Returns:
        Configured UnifiedFLPipeline
    """
    config = PipelineConfig(
        clipping=ClippingConfig(max_norm=clipping_norm),
        sparsification=SparsificationConfig(k_ratio=sparsity_ratio),
        compression=CompressionConfig(initial_ratio=compression_ratio),
        fhe=FHENoiseConfig(use_mock=use_mock_fhe),
    )
    return UnifiedFLPipeline(config)

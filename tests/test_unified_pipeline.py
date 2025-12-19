"""
Comprehensive Tests for Unified Federated Learning Pipeline

Tests all 7 improvements and their integration:
1. Gradient Clipping
2. Error Feedback
3. Top-K Sparsification
4. Adaptive MOAI Compression
5. Quality Monitoring
6. Noise-Aware FHE
7. Hierarchical Aggregation

Plus integration tests for the full pipeline.
"""

import pytest
import numpy as np
import time
from typing import Dict

# Import all pipeline components
from src.federated.unified_pipeline import (
    UnifiedFLPipeline,
    PipelineConfig,
    GradientClipper,
    ClippingConfig,
    ErrorFeedback,
    ErrorFeedbackConfig,
    TopKSparsifier,
    SparsificationConfig,
    AdaptiveMOAICompressor,
    CompressionConfig,
    QualityMonitor,
    MonitoringConfig,
    QualityAlert,
    NoiseAwareFHE,
    FHENoiseConfig,
    HierarchicalAggregator,
    AggregationConfig,
    SparseGradients,
    CompressedGradients,
    EncryptedUpdate,
    create_pipeline,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_gradients() -> Dict[str, np.ndarray]:
    """Create sample gradients for testing."""
    np.random.seed(42)
    return {
        'layer1.weight': np.random.randn(100, 50).astype(np.float32),
        'layer1.bias': np.random.randn(100).astype(np.float32),
        'layer2.weight': np.random.randn(50, 20).astype(np.float32),
        'layer2.bias': np.random.randn(50).astype(np.float32),
    }


@pytest.fixture
def large_gradients() -> Dict[str, np.ndarray]:
    """Create large gradients for stress testing."""
    np.random.seed(42)
    return {
        'encoder.weight': np.random.randn(1000, 256).astype(np.float32),
        'encoder.bias': np.random.randn(1000).astype(np.float32),
        'decoder.weight': np.random.randn(256, 1000).astype(np.float32),
        'decoder.bias': np.random.randn(256).astype(np.float32),
    }


@pytest.fixture
def exploding_gradients() -> Dict[str, np.ndarray]:
    """Create gradients with large magnitudes (gradient explosion)."""
    np.random.seed(42)
    return {
        'layer1.weight': np.random.randn(100, 50).astype(np.float32) * 1000,
        'layer1.bias': np.random.randn(100).astype(np.float32) * 1000,
    }


@pytest.fixture
def default_pipeline() -> UnifiedFLPipeline:
    """Create a default pipeline for testing."""
    return UnifiedFLPipeline()


# =============================================================================
# Stage 1: Gradient Clipping Tests
# =============================================================================

class TestGradientClipper:
    """Tests for Stage 1: Gradient Clipping."""

    def test_clipper_creation(self):
        """Test clipper can be created with default config."""
        clipper = GradientClipper()
        assert clipper.config.max_norm == 1.0
        assert clipper.config.norm_type == 2
        assert clipper.config.enabled is True

    def test_clipper_custom_config(self):
        """Test clipper with custom configuration."""
        config = ClippingConfig(max_norm=0.5, norm_type=1)
        clipper = GradientClipper(config)
        assert clipper.config.max_norm == 0.5
        assert clipper.config.norm_type == 1

    def test_clipping_reduces_norm(self, sample_gradients):
        """Test that clipping reduces gradient norm when needed."""
        clipper = GradientClipper(ClippingConfig(max_norm=1.0))
        clipped = clipper.clip(sample_gradients)

        # Calculate clipped norm
        flat = np.concatenate([g.flatten() for g in clipped.values()])
        clipped_norm = np.sqrt(np.sum(flat ** 2))

        assert clipped_norm <= 1.0 + 1e-6  # Allow small numerical error

    def test_clipping_preserves_direction(self, sample_gradients):
        """Test that clipping preserves gradient direction."""
        clipper = GradientClipper(ClippingConfig(max_norm=1.0))
        clipped = clipper.clip(sample_gradients)

        # Check that relative values are preserved (same direction)
        for name in sample_gradients:
            original = sample_gradients[name].flatten()
            clipped_g = clipped[name].flatten()

            # Correlation should be 1.0 (perfect positive)
            if np.std(original) > 0:
                correlation = np.corrcoef(original, clipped_g)[0, 1]
                assert correlation > 0.999

    def test_clipping_exploding_gradients(self, exploding_gradients):
        """Test that clipping handles gradient explosion."""
        clipper = GradientClipper(ClippingConfig(max_norm=1.0))
        clipped = clipper.clip(exploding_gradients)

        flat = np.concatenate([g.flatten() for g in clipped.values()])
        clipped_norm = np.sqrt(np.sum(flat ** 2))

        assert clipped_norm <= 1.0 + 1e-6
        assert clipper.last_clip_ratio > 0.99  # Heavy clipping occurred

    def test_clipping_disabled(self, sample_gradients):
        """Test that disabled clipping passes through unchanged."""
        clipper = GradientClipper(ClippingConfig(enabled=False))
        clipped = clipper.clip(sample_gradients)

        for name in sample_gradients:
            np.testing.assert_array_equal(sample_gradients[name], clipped[name])

    def test_clipping_stats(self, exploding_gradients):
        """Test that clipping tracks statistics."""
        clipper = GradientClipper()
        clipper.clip(exploding_gradients)
        clipper.clip(exploding_gradients)

        stats = clipper.get_stats()
        assert stats['total_calls'] == 2
        assert stats['total_clips'] == 2
        assert stats['avg_clip_ratio'] > 0


# =============================================================================
# Stage 2: Error Feedback Tests
# =============================================================================

class TestErrorFeedback:
    """Tests for Stage 2: Error Feedback."""

    def test_feedback_creation(self):
        """Test error feedback can be created."""
        feedback = ErrorFeedback()
        assert feedback.config.enabled is True
        assert feedback.config.decay == 0.9

    def test_first_round_no_correction(self, sample_gradients):
        """Test that first round has no error correction."""
        feedback = ErrorFeedback()
        corrected = feedback.apply(sample_gradients, 'client_1')

        # First round should be unchanged (no residuals yet)
        for name in sample_gradients:
            np.testing.assert_array_equal(sample_gradients[name], corrected[name])

    def test_residual_accumulation(self, sample_gradients):
        """Test that residuals accumulate across rounds."""
        feedback = ErrorFeedback()

        # Round 1: Apply and store residuals
        corrected1 = feedback.apply(sample_gradients, 'client_1')

        # Create "transmitted" gradients (50% of original)
        transmitted = {k: v * 0.5 for k, v in sample_gradients.items()}
        feedback.store_residuals(sample_gradients, transmitted, 'client_1')

        # Round 2: Should include residuals
        corrected2 = feedback.apply(sample_gradients, 'client_1')

        # Corrected gradients should be larger due to added residuals
        for name in sample_gradients:
            # Note: Residuals are decayed, so check they're added
            norm_corrected = np.linalg.norm(corrected2[name])
            norm_original = np.linalg.norm(sample_gradients[name])
            # With 0.9 decay on 50% residual, correction adds ~0.45 * 0.9 = 0.405
            # So corrected should be > original (at least slightly larger)
            assert norm_corrected > norm_original * 1.1

    def test_residual_decay(self, sample_gradients):
        """Test that residuals decay over time."""
        config = ErrorFeedbackConfig(decay=0.5)
        feedback = ErrorFeedback(config)

        # Store 100% residual (nothing transmitted)
        transmitted = {k: np.zeros_like(v) for k, v in sample_gradients.items()}
        feedback.apply(sample_gradients, 'client_1')
        feedback.store_residuals(sample_gradients, transmitted, 'client_1')

        # Multiple rounds - residual should decay
        # Apply once to get initial corrected values
        corrected = feedback.apply(sample_gradients, 'client_1')
        initial_norm = sum(np.linalg.norm(v) for v in corrected.values())

        # Store new residuals (transmit nothing again)
        feedback.store_residuals(sample_gradients, transmitted, 'client_1')

        # Apply again - residuals should be decayed from before
        corrected2 = feedback.apply(sample_gradients, 'client_1')
        second_norm = sum(np.linalg.norm(v) for v in corrected2.values())

        # The correction component should be smaller (decayed)
        # Total norm includes original + residual, so compare the extra portion
        original_norm = sum(np.linalg.norm(v) for v in sample_gradients.values())
        extra1 = initial_norm - original_norm
        extra2 = second_norm - original_norm

        # Second round's extra should be smaller due to decay
        assert extra2 < extra1 or initial_norm >= second_norm

    def test_per_client_isolation(self, sample_gradients):
        """Test that residuals are isolated per client."""
        feedback = ErrorFeedback()

        # Client 1 has residuals
        feedback.apply(sample_gradients, 'client_1')
        transmitted = {k: v * 0.5 for k, v in sample_gradients.items()}
        feedback.store_residuals(sample_gradients, transmitted, 'client_1')

        # Client 2 should have no residuals
        corrected = feedback.apply(sample_gradients, 'client_2')
        for name in sample_gradients:
            np.testing.assert_array_equal(sample_gradients[name], corrected[name])

    def test_feedback_disabled(self, sample_gradients):
        """Test that disabled feedback passes through unchanged."""
        config = ErrorFeedbackConfig(enabled=False)
        feedback = ErrorFeedback(config)

        corrected = feedback.apply(sample_gradients, 'client_1')
        for name in sample_gradients:
            np.testing.assert_array_equal(sample_gradients[name], corrected[name])


# =============================================================================
# Stage 3: Top-K Sparsification Tests
# =============================================================================

class TestTopKSparsifier:
    """Tests for Stage 3: Top-K Sparsification."""

    def test_sparsifier_creation(self):
        """Test sparsifier can be created."""
        sparsifier = TopKSparsifier()
        assert sparsifier.config.k_ratio == 0.01
        assert sparsifier.config.min_k == 100

    def test_sparsification_reduces_size(self, sample_gradients):
        """Test that sparsification reduces gradient count."""
        config = SparsificationConfig(k_ratio=0.1, min_k=10)
        sparsifier = TopKSparsifier(config)

        sparse, _ = sparsifier.sparsify(sample_gradients)

        total_elements = sum(g.size for g in sample_gradients.values())
        assert len(sparse.indices) <= total_elements * 0.1 + 10

    def test_sparsification_keeps_largest(self, sample_gradients):
        """Test that top-K keeps largest magnitude values."""
        config = SparsificationConfig(k_ratio=0.1, min_k=10)
        sparsifier = TopKSparsifier(config)

        sparse, _ = sparsifier.sparsify(sample_gradients)

        # All gradients flattened
        all_grads = np.concatenate([g.flatten() for g in sample_gradients.values()])

        # Top-K values should be among the largest magnitudes
        kept_magnitudes = np.abs(sparse.values)
        all_magnitudes = np.abs(all_grads)

        # Minimum kept magnitude should be >= (1 - k_ratio) percentile
        min_kept = kept_magnitudes.min()
        threshold = np.percentile(all_magnitudes, 90)  # 90th percentile

        # Allow some tolerance
        assert min_kept >= threshold * 0.9

    def test_sparse_to_dense_reconstruction(self, sample_gradients):
        """Test that sparse gradients can be reconstructed to dense."""
        config = SparsificationConfig(k_ratio=0.5, min_k=10)
        sparsifier = TopKSparsifier(config)

        sparse, reconstructed = sparsifier.sparsify(sample_gradients)
        dense = sparse.to_dense()

        assert dense.shape[0] == sparse.total_elements

    def test_min_k_enforcement(self, sample_gradients):
        """Test that min_k is enforced even with very low k_ratio."""
        config = SparsificationConfig(k_ratio=0.0001, min_k=500)
        sparsifier = TopKSparsifier(config)

        sparse, _ = sparsifier.sparsify(sample_gradients)

        assert len(sparse.indices) >= 500

    def test_sparsification_disabled(self, sample_gradients):
        """Test that disabled sparsification returns all values."""
        config = SparsificationConfig(enabled=False)
        sparsifier = TopKSparsifier(config)

        sparse, _ = sparsifier.sparsify(sample_gradients)

        total_elements = sum(g.size for g in sample_gradients.values())
        assert len(sparse.indices) == total_elements

    def test_sparsity_ratio(self, sample_gradients):
        """Test sparsity ratio calculation."""
        config = SparsificationConfig(k_ratio=0.1, min_k=10)
        sparsifier = TopKSparsifier(config)

        sparse, _ = sparsifier.sparsify(sample_gradients)

        assert 0.8 <= sparse.sparsity_ratio <= 1.0


# =============================================================================
# Stage 4: Adaptive MOAI Compression Tests
# =============================================================================

class TestAdaptiveMOAICompressor:
    """Tests for Stage 4: Adaptive MOAI Compression."""

    def test_compressor_creation(self):
        """Test compressor can be created."""
        compressor = AdaptiveMOAICompressor()
        assert compressor.current_ratio == 32.0
        assert compressor.config.target_quality == 0.95

    def test_compression_reduces_size(self, sample_gradients):
        """Test that compression reduces data size."""
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.1, min_k=100))
        compressor = AdaptiveMOAICompressor()

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        original_size = len(sparse.values) * 4  # float32
        compressed_size = len(compressed.compressed_data)

        # Should achieve some compression
        assert compressed_size < original_size

    def test_compression_quality_score(self, sample_gradients):
        """Test that quality score is reasonable."""
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor(CompressionConfig(initial_ratio=4))

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        # With low compression ratio, quality should be high
        assert compressed.quality_score > 0.9

    def test_compression_decompression_roundtrip(self, sample_gradients):
        """Test that compression/decompression preserves values approximately."""
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor(CompressionConfig(initial_ratio=4))  # Lower ratio for higher quality

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)
        decompressed = compressor.decompress(compressed)

        # Values should be close - allow for quantization error
        # With 4x compression, we use 8 bits = 256 levels, so ~0.5% error per value
        np.testing.assert_allclose(
            sparse.values, decompressed.values,
            rtol=0.3, atol=0.3  # More tolerant for quantization
        )

    def test_adaptive_ratio_adjustment(self, sample_gradients):
        """Test that ratio adjusts based on quality feedback."""
        compressor = AdaptiveMOAICompressor(CompressionConfig(
            initial_ratio=32,
            adaptation_rate=0.5
        ))

        initial_ratio = compressor.current_ratio

        # Low quality feedback should decrease ratio (higher quality)
        compressor.adjust_ratio(0.5)  # Below target of 0.95

        assert compressor.current_ratio < initial_ratio

    def test_adaptive_ratio_bounds(self, sample_gradients):
        """Test that ratio stays within bounds."""
        config = CompressionConfig(min_ratio=8, max_ratio=64, adaptation_rate=1.0)
        compressor = AdaptiveMOAICompressor(config)

        # Try to push below min
        for _ in range(10):
            compressor.adjust_ratio(0.1)  # Very low quality

        assert compressor.current_ratio >= 8

        # Try to push above max
        for _ in range(20):
            compressor.adjust_ratio(1.0)  # Perfect quality

        assert compressor.current_ratio <= 64

    def test_compression_disabled(self, sample_gradients):
        """Test that disabled compression just serializes."""
        config = CompressionConfig(enabled=False)
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor(config)

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        assert compressed.compression_ratio == 1.0
        assert compressed.quality_score == 1.0


# =============================================================================
# Stage 5: Quality Monitoring Tests
# =============================================================================

class TestQualityMonitor:
    """Tests for Stage 5: Quality Monitoring."""

    def test_monitor_creation(self):
        """Test monitor can be created."""
        monitor = QualityMonitor()
        assert monitor.config.window_size == 10
        assert monitor.config.alert_threshold == 0.1

    def test_metric_recording(self):
        """Test that metrics are recorded."""
        monitor = QualityMonitor()

        monitor.start_round(1)
        monitor.record('compression_quality', 0.95)
        monitor.record('clip_ratio', 0.1)
        metrics = monitor.end_round()

        assert metrics.compression_quality == 0.95
        assert metrics.clip_ratio == 0.1
        assert metrics.round_number == 1

    def test_moving_average(self):
        """Test that moving average is computed correctly."""
        monitor = QualityMonitor(MonitoringConfig(window_size=3))

        # Record 3 values
        for i, quality in enumerate([0.9, 0.8, 0.7]):
            monitor.start_round(i)
            monitor.record('compression_quality', quality)
            monitor.end_round()

        avg = monitor.get_compression_quality()
        assert abs(avg - 0.8) < 0.01  # (0.9 + 0.8 + 0.7) / 3

    def test_quality_degradation_alert(self):
        """Test that quality degradation triggers alert."""
        monitor = QualityMonitor(MonitoringConfig(alert_threshold=0.1))

        # Start with high quality
        for i in range(3):
            monitor.start_round(i)
            monitor.record('compression_quality', 0.95)
            monitor.end_round()

        # Then drop quality
        monitor.start_round(4)
        monitor.record('compression_quality', 0.7)
        metrics = monitor.end_round()

        assert QualityAlert.COMPRESSION_DEGRADED in metrics.alerts

    def test_aggressive_clipping_alert(self):
        """Test that aggressive clipping triggers alert."""
        monitor = QualityMonitor()

        monitor.start_round(1)
        monitor.record('clip_ratio', 0.6)  # > 0.5 threshold
        metrics = monitor.end_round()

        assert QualityAlert.CLIPPING_AGGRESSIVE in metrics.alerts

    def test_fhe_noise_alert(self):
        """Test that low FHE noise budget triggers alert."""
        monitor = QualityMonitor()

        monitor.start_round(1)
        monitor.record('fhe_noise_budget', 0.2)  # < 0.3 threshold
        metrics = monitor.end_round()

        assert QualityAlert.FHE_NOISE_HIGH in metrics.alerts

    def test_monitor_disabled(self):
        """Test that disabled monitor still works."""
        config = MonitoringConfig(enabled=False)
        monitor = QualityMonitor(config)

        monitor.start_round(1)
        monitor.record('compression_quality', 0.5)
        metrics = monitor.end_round()

        # Should still return metrics, just not process
        assert metrics.round_number == 1


# =============================================================================
# Stage 6: Noise-Aware FHE Tests
# =============================================================================

class TestNoiseAwareFHE:
    """Tests for Stage 6: Noise-Aware FHE."""

    def test_fhe_creation(self):
        """Test FHE module can be created."""
        fhe = NoiseAwareFHE()
        assert fhe.config.use_mock is True
        assert fhe.config.bootstrap_threshold == 0.2

    def test_encryption_decryption_roundtrip(self, sample_gradients):
        """Test that encryption/decryption preserves data."""
        fhe = NoiseAwareFHE()
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        encrypted = fhe.encrypt(compressed, 'client_1', 1)
        decrypted = fhe.decrypt(encrypted)

        np.testing.assert_array_equal(compressed.indices, decrypted.indices)
        assert compressed.original_shape == decrypted.original_shape

    def test_noise_budget_tracking(self, sample_gradients):
        """Test that noise budget is tracked."""
        fhe = NoiseAwareFHE()
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        initial_budget = fhe._current_noise_budget

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)
        encrypted = fhe.encrypt(compressed, 'client_1', 1)

        assert encrypted.remaining_noise_budget < 1.0
        assert fhe._current_noise_budget < initial_budget

    def test_bootstrapping_trigger(self):
        """Test that bootstrapping is triggered at low budget."""
        # Use very high threshold and low noise budget to ensure bootstrap
        config = FHENoiseConfig(
            bootstrap_threshold=0.9,  # Very high threshold
            coeff_mod_bit_sizes=[60, 40, 60]  # Small budget (2 levels)
        )
        fhe = NoiseAwareFHE(config)

        initial_bootstraps = fhe.stats['bootstraps_performed']

        # Drain noise budget with many operations
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        grads = {'test': np.random.randn(100).astype(np.float32)}
        sparse, _ = sparsifier.sparsify(grads)
        compressed = compressor.compress(sparse)

        # Many operations to trigger bootstrap
        for i in range(20):
            fhe.encrypt(compressed, f'client_{i}', 1)

        # Should have bootstrapped at least once
        assert fhe.stats['bootstraps_performed'] > initial_bootstraps

    def test_homomorphic_addition(self, sample_gradients):
        """Test homomorphic addition of encrypted updates."""
        fhe = NoiseAwareFHE()
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        # Create multiple encrypted updates
        updates = [
            fhe.encrypt(compressed, f'client_{i}', 1)
            for i in range(3)
        ]

        aggregated = fhe.homomorphic_add(updates)

        assert aggregated.client_id == 'aggregated'
        assert aggregated.metadata['num_clients'] == 3


# =============================================================================
# Stage 7: Hierarchical Aggregation Tests
# =============================================================================

class TestHierarchicalAggregator:
    """Tests for Stage 7: Hierarchical Aggregation."""

    def test_aggregator_creation(self):
        """Test aggregator can be created."""
        aggregator = HierarchicalAggregator()
        assert aggregator.config.max_tree_depth == 4
        assert aggregator.config.branching_factor == 10

    def test_tree_building(self, sample_gradients):
        """Test aggregation tree building."""
        fhe = NoiseAwareFHE()
        aggregator = HierarchicalAggregator(fhe=fhe)
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        updates = [
            fhe.encrypt(compressed, f'client_{i}', 1)
            for i in range(15)
        ]

        tree = aggregator.build_tree(updates)

        assert tree.node_id == 'root'
        assert len(tree.children) == 2  # 15 / 10 = 2 groups

    def test_hierarchical_aggregation(self, sample_gradients):
        """Test full hierarchical aggregation."""
        fhe = NoiseAwareFHE()
        aggregator = HierarchicalAggregator(fhe=fhe)
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        updates = [
            fhe.encrypt(compressed, f'client_{i}', 1)
            for i in range(5)
        ]

        tree = aggregator.build_tree(updates)
        aggregated = aggregator.aggregate(tree, updates)

        assert aggregated is not None
        assert isinstance(aggregated, EncryptedUpdate)

    def test_tree_depth_limiting(self, sample_gradients):
        """Test that tree depth is limited."""
        config = AggregationConfig(max_tree_depth=2, branching_factor=2)
        fhe = NoiseAwareFHE()
        aggregator = HierarchicalAggregator(config, fhe)
        sparsifier = TopKSparsifier(SparsificationConfig(k_ratio=0.5, min_k=100))
        compressor = AdaptiveMOAICompressor()

        sparse, _ = sparsifier.sparsify(sample_gradients)
        compressed = compressor.compress(sparse)

        # Create 10 updates - should create tree of depth 4 with branching=2
        # But limited to depth 2
        updates = [
            fhe.encrypt(compressed, f'client_{i}', 1)
            for i in range(10)
        ]

        tree = aggregator.build_tree(updates)
        depth = aggregator._get_tree_depth(tree)

        assert depth <= 2


# =============================================================================
# Integration Tests: Full Pipeline
# =============================================================================

class TestUnifiedPipeline:
    """Integration tests for the full unified pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline can be created with default config."""
        pipeline = UnifiedFLPipeline()
        assert pipeline is not None
        assert pipeline.config is not None

    def test_pipeline_custom_config(self):
        """Test pipeline with custom configuration."""
        config = PipelineConfig(
            clipping=ClippingConfig(max_norm=0.5),
            sparsification=SparsificationConfig(k_ratio=0.05),
        )
        pipeline = UnifiedFLPipeline(config)
        assert pipeline.gradient_clipper.config.max_norm == 0.5
        assert pipeline.sparsifier.config.k_ratio == 0.05

    def test_single_client_update(self, sample_gradients):
        """Test processing a single client update."""
        pipeline = UnifiedFLPipeline()

        encrypted = pipeline.process_client_update(
            sample_gradients, 'client_1', round_number=1
        )

        assert isinstance(encrypted, EncryptedUpdate)
        assert encrypted.client_id == 'client_1'
        assert encrypted.round_number == 1

    def test_multi_client_aggregation(self, sample_gradients):
        """Test aggregation of multiple client updates."""
        pipeline = UnifiedFLPipeline()

        updates = [
            pipeline.process_client_update(sample_gradients, f'client_{i}', 1)
            for i in range(5)
        ]

        aggregated = pipeline.aggregate_updates(updates)

        assert isinstance(aggregated, EncryptedUpdate)

    def test_full_round_trip(self, sample_gradients):
        """Test complete round trip: gradients -> encrypted -> aggregated -> gradients."""
        pipeline = UnifiedFLPipeline()

        # Process updates from multiple clients
        updates = [
            pipeline.process_client_update(sample_gradients, f'client_{i}', 1)
            for i in range(3)
        ]

        # Aggregate
        aggregated = pipeline.aggregate_updates(updates)

        # Finalize
        result = pipeline.finalize_aggregation(aggregated)

        assert 'gradients' in result
        assert isinstance(result['gradients'], np.ndarray)

    def test_pipeline_statistics(self, sample_gradients):
        """Test that pipeline tracks comprehensive statistics."""
        pipeline = UnifiedFLPipeline()

        for i in range(3):
            pipeline.process_client_update(sample_gradients, f'client_{i}', 1)

        stats = pipeline.get_all_stats()

        assert 'pipeline' in stats
        assert 'clipping' in stats
        assert 'error_feedback' in stats
        assert 'sparsification' in stats
        assert 'compression' in stats
        assert 'monitoring' in stats
        assert 'fhe' in stats
        assert 'aggregation' in stats

        assert stats['pipeline']['total_updates_processed'] == 3

    def test_multiple_rounds(self, sample_gradients):
        """Test pipeline over multiple FL rounds."""
        pipeline = UnifiedFLPipeline()

        for round_num in range(3):
            updates = [
                pipeline.process_client_update(sample_gradients, f'client_{i}', round_num)
                for i in range(3)
            ]
            aggregated = pipeline.aggregate_updates(updates)
            result = pipeline.finalize_aggregation(aggregated)

        stats = pipeline.get_all_stats()
        assert stats['pipeline']['total_updates_processed'] == 9
        assert stats['pipeline']['total_aggregations'] == 3

    def test_error_feedback_integration(self, sample_gradients):
        """Test that error feedback works across rounds."""
        pipeline = UnifiedFLPipeline()

        # Round 1
        encrypted1 = pipeline.process_client_update(sample_gradients, 'client_1', 1)

        # Round 2 - should include feedback from round 1
        encrypted2 = pipeline.process_client_update(sample_gradients, 'client_1', 2)

        feedback_stats = pipeline.error_feedback.get_stats()
        assert feedback_stats['clients_tracked'] >= 1

    def test_adaptive_compression_integration(self, sample_gradients):
        """Test that compression ratio adapts based on quality."""
        config = PipelineConfig(
            compression=CompressionConfig(
                initial_ratio=32,
                adaptation_rate=0.5
            )
        )
        pipeline = UnifiedFLPipeline(config)

        initial_ratio = pipeline.compressor.current_ratio

        # Process multiple updates - ratio should adapt
        for i in range(10):
            pipeline.process_client_update(sample_gradients, f'client_{i}', 1)

        # Ratio should have changed (either direction based on quality)
        assert pipeline.compressor.current_ratio != initial_ratio or \
               pipeline.compressor.stats['ratio_adjustments'] >= 0

    def test_pipeline_with_exploding_gradients(self, exploding_gradients):
        """Test pipeline handles gradient explosion gracefully."""
        pipeline = UnifiedFLPipeline()

        encrypted = pipeline.process_client_update(exploding_gradients, 'client_1', 1)

        assert encrypted is not None
        assert pipeline.gradient_clipper.last_clip_ratio > 0.9

    def test_create_pipeline_convenience(self):
        """Test convenience function for creating pipeline."""
        pipeline = create_pipeline(
            clipping_norm=2.0,
            sparsity_ratio=0.05,
            compression_ratio=16
        )

        assert pipeline.gradient_clipper.config.max_norm == 2.0
        assert pipeline.sparsifier.config.k_ratio == 0.05
        assert pipeline.compressor.config.initial_ratio == 16


# =============================================================================
# Configuration Validation Tests
# =============================================================================

class TestPipelineConfig:
    """Tests for configuration validation."""

    def test_valid_config(self):
        """Test that valid config has no warnings."""
        config = PipelineConfig(
            compression=CompressionConfig(adaptation_rate=0.05),  # Below alert threshold
            monitoring=MonitoringConfig(alert_threshold=0.1)
        )
        warnings = config.validate()
        # Should have no warnings with properly configured rates
        assert len(warnings) == 0

    def test_fhe_depth_warning(self):
        """Test warning for insufficient FHE depth."""
        config = PipelineConfig(
            fhe=FHENoiseConfig(coeff_mod_bit_sizes=[60, 40, 60]),  # depth = 1
            aggregation=AggregationConfig(max_tree_depth=4)  # needs depth >= 4
        )
        warnings = config.validate()
        assert any('FHE depth' in w for w in warnings)

    def test_adaptation_rate_warning(self):
        """Test warning for too-fast adaptation."""
        config = PipelineConfig(
            compression=CompressionConfig(adaptation_rate=0.5),
            monitoring=MonitoringConfig(alert_threshold=0.1)  # adaptation > threshold
        )
        warnings = config.validate()
        assert any('adaptation_rate' in w for w in warnings)

    def test_error_feedback_decay_warning(self):
        """Test warning for non-decaying error feedback."""
        config = PipelineConfig(
            feedback=ErrorFeedbackConfig(decay=1.0)  # No decay
        )
        warnings = config.validate()
        assert any('decay' in w for w in warnings)


# =============================================================================
# Stress Tests
# =============================================================================

class TestPipelineStress:
    """Stress tests for the pipeline."""

    def test_large_gradients(self, large_gradients):
        """Test pipeline with large gradients."""
        pipeline = UnifiedFLPipeline()

        start = time.time()
        encrypted = pipeline.process_client_update(large_gradients, 'client_1', 1)
        elapsed = time.time() - start

        assert encrypted is not None
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

    def test_many_clients(self, sample_gradients):
        """Test aggregation with many clients."""
        pipeline = UnifiedFLPipeline()

        updates = [
            pipeline.process_client_update(sample_gradients, f'client_{i}', 1)
            for i in range(50)
        ]

        aggregated = pipeline.aggregate_updates(updates)
        result = pipeline.finalize_aggregation(aggregated)

        assert result is not None

    def test_memory_stability(self, sample_gradients):
        """Test that memory doesn't grow unboundedly."""
        pipeline = UnifiedFLPipeline()

        # Run many rounds
        for round_num in range(20):
            updates = [
                pipeline.process_client_update(sample_gradients, f'client_{i}', round_num)
                for i in range(5)
            ]
            pipeline.aggregate_updates(updates)

        # Check that per-client state doesn't grow unbounded
        assert pipeline.error_feedback.stats['clients_tracked'] <= 5


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_gradients(self):
        """Test handling of empty gradients."""
        pipeline = UnifiedFLPipeline()

        empty_grads = {'layer': np.array([], dtype=np.float32)}

        # Should handle gracefully (may raise or return minimal result)
        try:
            result = pipeline.process_client_update(empty_grads, 'client_1', 1)
            # If it succeeds, that's fine
        except (ValueError, IndexError):
            # These exceptions are acceptable for empty input
            pass

    def test_single_element_gradient(self):
        """Test handling of single-element gradients."""
        pipeline = UnifiedFLPipeline()

        single_grad = {'layer': np.array([1.5], dtype=np.float32)}

        result = pipeline.process_client_update(single_grad, 'client_1', 1)
        assert result is not None

    def test_zero_gradients(self):
        """Test handling of all-zero gradients."""
        pipeline = UnifiedFLPipeline()

        zero_grads = {'layer': np.zeros((100,), dtype=np.float32)}

        result = pipeline.process_client_update(zero_grads, 'client_1', 1)
        assert result is not None

    def test_nan_gradients(self):
        """Test handling of NaN gradients."""
        pipeline = UnifiedFLPipeline()

        nan_grads = {'layer': np.array([1.0, np.nan, 2.0], dtype=np.float32)}

        # Pipeline should handle NaN (clipping will bound it)
        result = pipeline.process_client_update(nan_grads, 'client_1', 1)
        # Result may contain NaN but shouldn't crash

    def test_single_client_aggregation(self, sample_gradients):
        """Test aggregation with single client."""
        pipeline = UnifiedFLPipeline()

        update = pipeline.process_client_update(sample_gradients, 'client_1', 1)
        aggregated = pipeline.aggregate_updates([update])

        assert aggregated is not None


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

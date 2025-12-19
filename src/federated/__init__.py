"""
Federated Learning Module with Quality-Preserving Improvements

This module implements a unified pipeline for federated learning that addresses
quality degradation from FHE encryption and MOAI compression.

Improvements implemented:
1. Gradient Clipping - Bounds gradient magnitude before processing
2. Error Feedback - Accumulates compression errors for next round
3. Top-K Sparsification - Only transmits top K% important gradients
4. Adaptive Compression - Self-adjusting compression ratio
5. Quality Monitoring - Real-time quality tracking and alerts
6. Noise-Aware FHE - Higher precision parameters with bootstrapping
7. Hierarchical Aggregation - Tree-based aggregation reduces FHE operations
"""

from .unified_pipeline import (
    UnifiedFLPipeline,
    PipelineConfig,
    GradientClipper,
    ErrorFeedback,
    TopKSparsifier,
    AdaptiveMOAICompressor,
    QualityMonitor,
    NoiseAwareFHE,
    HierarchicalAggregator,
    SparseGradients,
    CompressedGradients,
    EncryptedUpdate,
)

__all__ = [
    'UnifiedFLPipeline',
    'PipelineConfig',
    'GradientClipper',
    'ErrorFeedback',
    'TopKSparsifier',
    'AdaptiveMOAICompressor',
    'QualityMonitor',
    'NoiseAwareFHE',
    'HierarchicalAggregator',
    'SparseGradients',
    'CompressedGradients',
    'EncryptedUpdate',
]

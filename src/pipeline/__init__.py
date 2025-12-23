"""
Pipeline Module - Data Processing Pipelines

This module provides integrated pipelines for:
- Perception processing (cameras → features)
- Pi0 VLA inference
- Multi-modal data fusion

Usage:
    from src.pipeline import IntegratedPipeline

    pipeline = IntegratedPipeline()
    pipeline.initialize()
    result = pipeline.process(frames, proprio)
"""

from .integrated_pipeline import (
    IntegratedPipeline,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    'IntegratedPipeline',
    'PipelineConfig',
    'PipelineResult',
]

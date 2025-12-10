"""
Real Pose Inference Module

This module provides production-ready pose estimation using:
- RTMPose from MMPose/HuggingFace
- ONNX Runtime for efficient CPU/GPU inference
- TensorRT for Jetson deployment

Replaces all mock implementations with real inference.
"""

from .rtmpose_real import (
    RTMPoseRealInference,
    RTMPoseConfig,
    Pose2DResult,
    download_rtmpose_model,
)
from .person_detector import PersonDetector, DetectionResult

__all__ = [
    'RTMPoseRealInference',
    'RTMPoseConfig',
    'Pose2DResult',
    'PersonDetector',
    'DetectionResult',
    'download_rtmpose_model',
]

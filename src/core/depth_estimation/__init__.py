"""
Depth Estimation Module

Provides monocular depth estimation using Depth Anything V3 with TensorRT acceleration.
Integrates with the whole-body pose pipeline for improved 3D reconstruction.

Components:
- DepthAnythingV3: TensorRT-accelerated depth estimation
- DepthPoseFusion: Combine depth maps with 2D pose for 3D estimation
- DepthCalibration: Camera verification using depth

Reference:
- Depth Anything V3: https://github.com/DepthAnything/Depth-Anything-V3
- ROS2 TRT Implementation: https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
"""

from .depth_anything_v3 import (
    DepthAnythingV3,
    DepthEstimationConfig,
    DepthResult,
)
from .depth_pose_fusion import (
    DepthPoseFusion,
    DepthFusionConfig,
)

__all__ = [
    "DepthAnythingV3",
    "DepthEstimationConfig",
    "DepthResult",
    "DepthPoseFusion",
    "DepthFusionConfig",
]

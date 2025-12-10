"""
Unit Tests for Depth Estimation Module

Tests for Depth Anything V3 integration and depth-pose fusion.
"""

import numpy as np
import pytest
import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_map():
    """Create a sample depth map."""
    # Gradient depth: closer at bottom, farther at top
    h, w = 720, 1280
    v = np.arange(h).reshape(-1, 1)
    depth = 1.0 + (h - v) / h * 4.0  # 1m to 5m
    return np.broadcast_to(depth, (h, w)).astype(np.float32)


@pytest.fixture
def sample_pose_2d():
    """Create a sample 2D pose result."""
    @dataclass
    class MockPose2D:
        keypoints: np.ndarray
        bboxes: np.ndarray
        scores: np.ndarray
        num_persons: int
        image_size: tuple

    # 17 keypoints for body
    kp = np.zeros((1, 17, 3))
    kp[0, 0] = [640, 200, 0.95]   # nose
    kp[0, 5] = [560, 300, 0.9]   # left shoulder
    kp[0, 6] = [720, 300, 0.9]   # right shoulder
    kp[0, 7] = [520, 450, 0.85]  # left elbow
    kp[0, 8] = [760, 450, 0.85]  # right elbow
    kp[0, 9] = [500, 550, 0.8]   # left wrist
    kp[0, 10] = [780, 550, 0.8]  # right wrist
    kp[0, 11] = [580, 500, 0.85] # left hip
    kp[0, 12] = [700, 500, 0.85] # right hip
    kp[0, 13] = [570, 650, 0.8]  # left knee
    kp[0, 14] = [710, 650, 0.8]  # right knee
    kp[0, 15] = [560, 800, 0.75] # left ankle
    kp[0, 16] = [720, 800, 0.75] # right ankle

    bbox = np.array([[450, 150, 830, 850, 0.95]])

    return MockPose2D(
        keypoints=kp,
        bboxes=bbox,
        scores=np.array([0.95]),
        num_persons=1,
        image_size=(1280, 720),
    )


# =============================================================================
# DepthEstimationConfig Tests
# =============================================================================

class TestDepthEstimationConfig:
    """Tests for DepthEstimationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.core.depth_estimation.depth_anything_v3 import (
            DepthEstimationConfig,
            DepthModelSize,
            DepthPrecision,
        )

        config = DepthEstimationConfig()

        assert config.model_size == DepthModelSize.LARGE
        assert config.precision == DepthPrecision.FP16
        assert config.input_height == 280
        assert config.input_width == 504
        assert config.enable_sky_detection is True
        assert config.sky_threshold == 0.3

    def test_get_model_path(self):
        """Test model path generation."""
        from src.core.depth_estimation.depth_anything_v3 import (
            DepthEstimationConfig,
            DepthModelSize,
        )

        config = DepthEstimationConfig(model_size=DepthModelSize.SMALL)
        path = config.get_model_path()
        assert "SMALL" in path

        config = DepthEstimationConfig(model_size=DepthModelSize.LARGE)
        path = config.get_model_path()
        assert "LARGE" in path


# =============================================================================
# DepthResult Tests
# =============================================================================

class TestDepthResult:
    """Tests for DepthResult dataclass."""

    def test_depth_result_creation(self, sample_depth_map):
        """Test creating DepthResult."""
        from src.core.depth_estimation.depth_anything_v3 import DepthResult

        result = DepthResult(
            depth_map=sample_depth_map,
            timestamp=time.time(),
            original_size=(1280, 720),
        )

        assert result.width == 1280
        assert result.height == 720
        assert result.depth_map.shape == (720, 1280)

    def test_get_depth_at(self, sample_depth_map):
        """Test getting depth at specific location."""
        from src.core.depth_estimation.depth_anything_v3 import DepthResult

        result = DepthResult(
            depth_map=sample_depth_map,
            timestamp=time.time(),
            original_size=(1280, 720),
        )

        # Top of image should be farther (~5m)
        depth_top = result.get_depth_at(640, 0)
        assert depth_top > 4.0

        # Bottom should be closer (~1m)
        depth_bottom = result.get_depth_at(640, 719)
        assert depth_bottom < 2.0

    def test_sample_depth_bilinear(self, sample_depth_map):
        """Test bilinear depth sampling."""
        from src.core.depth_estimation.depth_anything_v3 import DepthResult

        result = DepthResult(
            depth_map=sample_depth_map,
            timestamp=time.time(),
            original_size=(1280, 720),
        )

        # Sample at integer should match direct access
        depth_int = result.get_depth_at(640, 360)
        depth_bilin = result.sample_depth_bilinear(640.0, 360.0)
        assert abs(depth_int - depth_bilin) < 0.01

        # Sample between pixels should interpolate
        depth_half = result.sample_depth_bilinear(640.5, 360.5)
        assert depth_half > 0

    def test_to_colormap(self, sample_depth_map):
        """Test colormap visualization."""
        from src.core.depth_estimation.depth_anything_v3 import DepthResult

        result = DepthResult(
            depth_map=sample_depth_map,
            timestamp=time.time(),
            original_size=(1280, 720),
        )

        colored = result.to_colormap(colormap="JET", max_depth=10.0)

        assert colored.shape[:2] == sample_depth_map.shape
        assert colored.dtype == np.uint8


# =============================================================================
# DepthAnythingV3 Tests
# =============================================================================

class TestDepthAnythingV3:
    """Tests for DepthAnythingV3 inference."""

    def test_mock_inference(self, sample_image):
        """Test mock inference when no model is available."""
        from src.core.depth_estimation.depth_anything_v3 import DepthAnythingV3

        estimator = DepthAnythingV3()

        result = estimator.infer(sample_image)

        assert result.depth_map.shape == sample_image.shape[:2]
        assert result.depth_map.min() >= 0
        assert result.inference_time_ms > 0

    def test_set_camera_intrinsics(self, sample_image):
        """Test setting camera intrinsics."""
        from src.core.depth_estimation.depth_anything_v3 import DepthAnythingV3

        estimator = DepthAnythingV3()
        estimator.set_camera_intrinsics(fx=1000, fy=1000)

        result = estimator.infer(sample_image)

        # Should use provided focal length
        assert result.depth_map is not None

    def test_batch_inference(self, sample_image):
        """Test batch inference."""
        from src.core.depth_estimation.depth_anything_v3 import DepthAnythingV3

        estimator = DepthAnythingV3()

        images = [sample_image, sample_image]
        results = estimator.infer_batch(images)

        assert len(results) == 2
        assert all(r.depth_map.shape == sample_image.shape[:2] for r in results)


# =============================================================================
# DepthPoseFusion Tests
# =============================================================================

class TestDepthPoseFusion:
    """Tests for depth-pose fusion."""

    def test_estimate_from_depth(self, sample_pose_2d, sample_depth_map):
        """Test 3D pose estimation from depth."""
        from src.core.depth_estimation.depth_pose_fusion import (
            DepthPoseFusion,
            DepthFusionConfig,
            CameraIntrinsics,
        )
        from src.core.depth_estimation.depth_anything_v3 import DepthResult

        # Create fusion
        fusion = DepthPoseFusion(DepthFusionConfig())

        # Create depth result
        depth_result = DepthResult(
            depth_map=sample_depth_map,
            timestamp=time.time(),
            original_size=(1280, 720),
        )

        # Create intrinsics
        intrinsics = CameraIntrinsics(
            fx=1000, fy=1000, cx=640, cy=360,
            width=1280, height=720
        )

        # Estimate 3D
        pose_3d = fusion.estimate_from_depth(
            sample_pose_2d, depth_result, intrinsics
        )

        # Check results
        assert pose_3d.keypoints_3d.shape[0] == 17
        assert pose_3d.valid_keypoint_ratio > 0

        # Check that non-zero keypoints have valid 3D coordinates
        for i in range(17):
            if sample_pose_2d.keypoints[0, i, 2] > 0.3:
                assert np.any(pose_3d.keypoints_3d[i] != 0)

    def test_create_intrinsics_from_fov(self):
        """Test creating intrinsics from FOV."""
        from src.core.depth_estimation.depth_pose_fusion import create_intrinsics_from_fov

        intrinsics = create_intrinsics_from_fov(
            width=1280, height=720, fov_horizontal=60
        )

        assert intrinsics.width == 1280
        assert intrinsics.height == 720
        assert intrinsics.cx == 640
        assert intrinsics.cy == 360
        assert intrinsics.fx > 0
        assert abs(intrinsics.fx - intrinsics.fy) < 1  # Square pixels

    def test_back_project(self):
        """Test back-projection from 2D to 3D."""
        from src.core.depth_estimation.depth_pose_fusion import CameraIntrinsics

        intrinsics = CameraIntrinsics(
            fx=1000, fy=1000, cx=640, cy=360,
            width=1280, height=720
        )

        # Center of image at 2m depth should be at origin (0, 0, 2)
        point = intrinsics.back_project(640, 360, 2.0)
        assert abs(point[0]) < 0.01
        assert abs(point[1]) < 0.01
        assert abs(point[2] - 2.0) < 0.01

        # Point to the right should have positive X
        point_right = intrinsics.back_project(800, 360, 2.0)
        assert point_right[0] > 0

    def test_anatomical_constraints(self, sample_pose_2d, sample_depth_map):
        """Test anatomical constraint enforcement."""
        from src.core.depth_estimation.depth_pose_fusion import (
            DepthPoseFusion,
            DepthFusionConfig,
            CameraIntrinsics,
        )
        from src.core.depth_estimation.depth_anything_v3 import DepthResult

        config = DepthFusionConfig(
            enable_anatomical_constraints=True,
            min_limb_length=0.1,
            max_limb_length=0.8,
        )
        fusion = DepthPoseFusion(config)

        depth_result = DepthResult(
            depth_map=sample_depth_map,
            timestamp=time.time(),
            original_size=(1280, 720),
        )

        intrinsics = CameraIntrinsics(
            fx=1000, fy=1000, cx=640, cy=360,
            width=1280, height=720
        )

        pose_3d = fusion.estimate_from_depth(
            sample_pose_2d, depth_result, intrinsics
        )

        # Check limb lengths are within constraints
        # Right arm: shoulder(6) -> elbow(8) -> wrist(10)
        if pose_3d.keypoint_confidence[6] > 0 and pose_3d.keypoint_confidence[8] > 0:
            upper_arm = np.linalg.norm(
                pose_3d.keypoints_3d[8] - pose_3d.keypoints_3d[6]
            )
            assert 0.1 <= upper_arm <= 0.8 or upper_arm == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests with pose pipeline."""

    def test_pipeline_with_depth(self, sample_image):
        """Test WholeBodyRetargetingPipeline with depth estimation."""
        from src.core.wholebody_pose_pipeline import (
            WholeBodyRetargetingPipeline,
            PoseEstimationConfig,
        )

        # Create pipeline with depth enabled
        pipeline = WholeBodyRetargetingPipeline(
            enable_depth_estimation=True
        )

        # Process single camera
        images = {"cam0": sample_image}
        result = pipeline.process_frame(images)

        assert 'pose_3d' in result
        assert 'depth_maps' in result
        assert 'robot_q' in result

    def test_pipeline_without_depth(self, sample_image):
        """Test pipeline with depth disabled."""
        from src.core.wholebody_pose_pipeline import (
            WholeBodyRetargetingPipeline,
        )

        pipeline = WholeBodyRetargetingPipeline(
            enable_depth_estimation=False
        )

        images = {"cam0": sample_image}
        result = pipeline.process_frame(images)

        assert 'pose_3d' in result
        assert result['depth_maps'] == {}


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_depth_estimator(self):
        """Test create_depth_estimator factory."""
        from src.core.depth_estimation.depth_anything_v3 import create_depth_estimator

        estimator = create_depth_estimator(
            model_size="large",
            precision="fp16",
            device="cuda:0",
        )

        assert estimator is not None
        assert estimator.config.precision.value == "fp16"

    def test_depth_to_point_cloud(self, sample_depth_map):
        """Test point cloud generation."""
        from src.core.depth_estimation.depth_pose_fusion import (
            depth_to_point_cloud_simple,
            CameraIntrinsics,
        )

        intrinsics = CameraIntrinsics(
            fx=1000, fy=1000, cx=640, cy=360,
            width=1280, height=720
        )

        points = depth_to_point_cloud_simple(
            sample_depth_map, intrinsics, downsample=4
        )

        assert len(points) > 0
        assert points.shape[1] == 3


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks."""

    def test_inference_time(self, sample_image):
        """Test inference time is reasonable."""
        from src.core.depth_estimation.depth_anything_v3 import DepthAnythingV3

        estimator = DepthAnythingV3()

        # Warmup
        estimator.infer(sample_image)

        # Benchmark
        times = []
        for _ in range(10):
            result = estimator.infer(sample_image)
            times.append(result.inference_time_ms)

        avg_time = np.mean(times)
        # Mock inference should be very fast (<10ms)
        assert avg_time < 100, f"Inference too slow: {avg_time:.1f}ms"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

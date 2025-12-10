"""
Depth-Pose Fusion Module

Combines monocular depth estimation with 2D pose detection for improved
3D pose reconstruction. Enables accurate single-camera 3D pose estimation
and enhances multi-camera triangulation.

Key Features:
- Single-camera 3D pose from depth
- Multi-camera depth-assisted triangulation
- Temporal smoothing
- Confidence-weighted fusion

This module replaces the hardcoded "2m depth" assumption with real
per-pixel depth values from Depth Anything V3.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import logging

from .depth_anything_v3 import DepthResult, DepthAnythingV3, DepthEstimationConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DepthFusionConfig:
    """Configuration for depth-pose fusion."""

    # Depth sampling
    use_bilinear_sampling: bool = True  # Sub-pixel accuracy
    depth_sample_radius: int = 3  # Pixels to sample around keypoint

    # Depth validation
    min_valid_depth: float = 0.3  # Minimum depth (meters)
    max_valid_depth: float = 10.0  # Maximum depth (meters)
    depth_confidence_threshold: float = 0.3  # Min keypoint confidence

    # Multi-view fusion weights
    triangulation_weight: float = 0.7  # Weight for DLT triangulation
    depth_weight: float = 0.3  # Weight for depth-based estimate

    # Single-camera settings
    single_cam_smoothing: float = 0.3  # Temporal smoothing alpha

    # Outlier rejection
    enable_outlier_rejection: bool = True
    max_depth_variance: float = 0.5  # Max std dev of depth samples

    # Anatomical constraints
    enable_anatomical_constraints: bool = True
    min_limb_length: float = 0.1  # Minimum limb segment (meters)
    max_limb_length: float = 0.8  # Maximum limb segment (meters)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> 'CameraIntrinsics':
        """Create from 3x3 intrinsic matrix."""
        return cls(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height,
        )

    def back_project(self, x: float, y: float, z: float) -> np.ndarray:
        """Back-project 2D point + depth to 3D."""
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy
        return np.array([X, Y, z])


@dataclass
class FusedPose3D:
    """3D pose from depth-pose fusion."""
    timestamp: float

    # 3D keypoints in camera frame: [N_keypoints, 3]
    keypoints_3d: np.ndarray

    # Per-keypoint confidence: [N_keypoints]
    keypoint_confidence: np.ndarray

    # Per-keypoint depth source: 'depth', 'triangulation', 'interpolated'
    depth_source: List[str]

    # Quality metrics
    mean_depth_confidence: float = 0.0
    valid_keypoint_ratio: float = 0.0

    @classmethod
    def empty(cls, n_keypoints: int = 133) -> 'FusedPose3D':
        return cls(
            timestamp=time.time(),
            keypoints_3d=np.zeros((n_keypoints, 3)),
            keypoint_confidence=np.zeros(n_keypoints),
            depth_source=['none'] * n_keypoints,
        )


# =============================================================================
# Depth-Pose Fusion
# =============================================================================

class DepthPoseFusion:
    """
    Fuses depth estimation with 2D pose for 3D reconstruction.

    Supports two modes:
    1. Single-camera: Uses depth map to lift 2D pose to 3D
    2. Multi-camera: Uses depth as additional constraint for triangulation

    Usage:
        fusion = DepthPoseFusion()

        # Single camera
        pose_3d = fusion.estimate_from_depth(pose_2d, depth_result, intrinsics)

        # Multi-camera with depth refinement
        pose_3d = fusion.refine_with_depth(
            triangulated_pose, pose_2d_dict, depth_dict, intrinsics_dict
        )
    """

    def __init__(self, config: DepthFusionConfig = None):
        self.config = config or DepthFusionConfig()

        # Temporal state for smoothing
        self._prev_pose: Optional[FusedPose3D] = None
        self._prev_timestamp: float = 0.0

        # Limb definitions for anatomical constraints
        self._limb_pairs = [
            (5, 7),   # left shoulder -> elbow
            (7, 9),   # left elbow -> wrist
            (6, 8),   # right shoulder -> elbow
            (8, 10),  # right elbow -> wrist
            (11, 13), # left hip -> knee
            (13, 15), # left knee -> ankle
            (12, 14), # right hip -> knee
            (14, 16), # right knee -> ankle
            (5, 11),  # left shoulder -> hip
            (6, 12),  # right shoulder -> hip
        ]

    def estimate_from_depth(
        self,
        pose_2d: 'Pose2DResult',
        depth_result: DepthResult,
        intrinsics: CameraIntrinsics,
    ) -> FusedPose3D:
        """
        Estimate 3D pose from 2D keypoints and depth map (single camera).

        This is the key method that replaces the hardcoded "2m depth" assumption.

        Args:
            pose_2d: 2D pose detection result
            depth_result: Depth estimation result
            intrinsics: Camera intrinsic parameters

        Returns:
            FusedPose3D with 3D keypoints in camera frame
        """
        timestamp = time.time()

        if pose_2d.num_persons == 0:
            return FusedPose3D.empty()

        # Get first person's keypoints: [N, 3] where 3 = (x, y, confidence)
        kp_2d = pose_2d.keypoints[0]
        n_keypoints = len(kp_2d)

        # Initialize outputs
        keypoints_3d = np.zeros((n_keypoints, 3))
        keypoint_confidence = np.zeros(n_keypoints)
        depth_source = ['none'] * n_keypoints

        valid_count = 0

        for i, (x, y, conf) in enumerate(kp_2d):
            if conf < self.config.depth_confidence_threshold:
                continue

            # Sample depth at keypoint location
            depth, depth_conf = self._sample_depth_robust(
                depth_result, x, y
            )

            if depth < self.config.min_valid_depth or depth > self.config.max_valid_depth:
                continue

            # Back-project to 3D
            point_3d = intrinsics.back_project(x, y, depth)

            keypoints_3d[i] = point_3d
            keypoint_confidence[i] = conf * depth_conf
            depth_source[i] = 'depth'
            valid_count += 1

        # Interpolate missing keypoints from neighbors
        keypoints_3d, depth_source = self._interpolate_missing(
            keypoints_3d, keypoint_confidence, depth_source
        )

        # Apply anatomical constraints
        if self.config.enable_anatomical_constraints:
            keypoints_3d = self._apply_anatomical_constraints(
                keypoints_3d, keypoint_confidence
            )

        # Create result
        result = FusedPose3D(
            timestamp=timestamp,
            keypoints_3d=keypoints_3d,
            keypoint_confidence=keypoint_confidence,
            depth_source=depth_source,
            mean_depth_confidence=np.mean(keypoint_confidence[keypoint_confidence > 0]) if valid_count > 0 else 0.0,
            valid_keypoint_ratio=valid_count / n_keypoints,
        )

        # Apply temporal smoothing
        if self._prev_pose is not None:
            result = self._apply_temporal_smoothing(result)

        self._prev_pose = result
        self._prev_timestamp = timestamp

        return result

    def _sample_depth_robust(
        self,
        depth_result: DepthResult,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """
        Robustly sample depth at a location.

        Uses multiple samples around the keypoint and rejects outliers.

        Returns:
            (depth, confidence) tuple
        """
        if self.config.use_bilinear_sampling and self.config.depth_sample_radius == 0:
            # Simple bilinear sampling
            depth = depth_result.sample_depth_bilinear(x, y)
            return depth, 1.0

        # Multi-sample approach for robustness
        radius = self.config.depth_sample_radius
        samples = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                sx, sy = x + dx, y + dy

                if 0 <= sx < depth_result.width and 0 <= sy < depth_result.height:
                    if self.config.use_bilinear_sampling:
                        d = depth_result.sample_depth_bilinear(sx, sy)
                    else:
                        d = depth_result.get_depth_at(int(sx), int(sy))

                    if self.config.min_valid_depth < d < self.config.max_valid_depth:
                        samples.append(d)

        if not samples:
            return 0.0, 0.0

        samples = np.array(samples)

        # Outlier rejection
        if self.config.enable_outlier_rejection and len(samples) > 3:
            median = np.median(samples)
            mad = np.median(np.abs(samples - median))

            if mad > 0:
                # Keep samples within 2 MAD of median
                mask = np.abs(samples - median) < 2 * mad
                samples = samples[mask] if mask.sum() > 0 else samples

        # Return median and confidence based on consistency
        depth = np.median(samples)
        std = np.std(samples)

        # Confidence based on depth consistency
        if std < 0.1:
            confidence = 1.0
        elif std < self.config.max_depth_variance:
            confidence = 1.0 - std / self.config.max_depth_variance
        else:
            confidence = 0.3

        return depth, confidence

    def _interpolate_missing(
        self,
        keypoints_3d: np.ndarray,
        confidence: np.ndarray,
        depth_source: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Interpolate missing keypoints from nearby valid keypoints.

        Uses skeletal structure to interpolate (e.g., if shoulder and wrist
        are known, elbow can be estimated).
        """
        # Interpolation pairs: (target, source1, source2, weight1)
        interpolations = [
            (7, 5, 9, 0.5),    # left elbow from shoulder and wrist
            (8, 6, 10, 0.5),   # right elbow from shoulder and wrist
            (13, 11, 15, 0.5), # left knee from hip and ankle
            (14, 12, 16, 0.5), # right knee from hip and ankle
        ]

        for target, src1, src2, w1 in interpolations:
            if target >= len(keypoints_3d):
                continue

            if confidence[target] > 0:
                continue  # Already have this keypoint

            if confidence[src1] > 0.3 and confidence[src2] > 0.3:
                # Interpolate
                keypoints_3d[target] = (
                    w1 * keypoints_3d[src1] + (1 - w1) * keypoints_3d[src2]
                )
                confidence[target] = 0.5 * min(confidence[src1], confidence[src2])
                depth_source[target] = 'interpolated'

        return keypoints_3d, depth_source

    def _apply_anatomical_constraints(
        self,
        keypoints_3d: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Apply anatomical constraints to ensure physically plausible pose.

        Checks limb lengths and adjusts if outside human body range.
        """
        for i, j in self._limb_pairs:
            if i >= len(keypoints_3d) or j >= len(keypoints_3d):
                continue

            if confidence[i] < 0.3 or confidence[j] < 0.3:
                continue

            # Calculate limb length
            limb = keypoints_3d[j] - keypoints_3d[i]
            length = np.linalg.norm(limb)

            if length < 1e-6:
                continue

            # Check constraints
            if length < self.config.min_limb_length:
                # Too short - extend to minimum
                scale = self.config.min_limb_length / length
                keypoints_3d[j] = keypoints_3d[i] + limb * scale

            elif length > self.config.max_limb_length:
                # Too long - shorten to maximum
                scale = self.config.max_limb_length / length
                keypoints_3d[j] = keypoints_3d[i] + limb * scale

        return keypoints_3d

    def _apply_temporal_smoothing(
        self,
        current: FusedPose3D,
    ) -> FusedPose3D:
        """
        Apply temporal smoothing to reduce jitter.

        Uses exponential smoothing with confidence-weighted blending.
        """
        if self._prev_pose is None:
            return current

        dt = current.timestamp - self._prev_timestamp
        if dt > 0.5:  # Too much time passed, don't smooth
            return current

        alpha = self.config.single_cam_smoothing

        # Blend keypoints
        for i in range(len(current.keypoints_3d)):
            if current.keypoint_confidence[i] > 0 and self._prev_pose.keypoint_confidence[i] > 0:
                # Weight by confidence
                w_curr = current.keypoint_confidence[i]
                w_prev = self._prev_pose.keypoint_confidence[i] * (1 - alpha)

                total_w = w_curr + w_prev
                if total_w > 0:
                    current.keypoints_3d[i] = (
                        w_curr * current.keypoints_3d[i] +
                        w_prev * self._prev_pose.keypoints_3d[i]
                    ) / total_w

        return current

    def refine_with_depth(
        self,
        triangulated_pose: 'Pose3DResult',
        pose_2d_dict: Dict[str, 'Pose2DResult'],
        depth_dict: Dict[str, DepthResult],
        intrinsics_dict: Dict[str, CameraIntrinsics],
        extrinsics_dict: Dict[str, np.ndarray] = None,
    ) -> 'Pose3DResult':
        """
        Refine triangulated pose using depth information from all cameras.

        Combines DLT triangulation with depth-based estimates using
        confidence-weighted fusion.

        Args:
            triangulated_pose: Pose3DResult from multi-view triangulation
            pose_2d_dict: Dict of camera_id -> Pose2DResult
            depth_dict: Dict of camera_id -> DepthResult
            intrinsics_dict: Dict of camera_id -> CameraIntrinsics
            extrinsics_dict: Optional dict of camera_id -> 4x4 world->camera transform

        Returns:
            Refined Pose3DResult
        """
        n_keypoints = len(triangulated_pose.keypoints_3d)

        # Get depth-based estimates from each camera
        depth_estimates = []

        for cam_id in depth_dict.keys():
            if cam_id not in pose_2d_dict or cam_id not in intrinsics_dict:
                continue

            # Estimate 3D from this camera's depth
            fused = self.estimate_from_depth(
                pose_2d_dict[cam_id],
                depth_dict[cam_id],
                intrinsics_dict[cam_id],
            )

            # Transform to world frame if extrinsics provided
            if extrinsics_dict and cam_id in extrinsics_dict:
                T_world_cam = np.linalg.inv(extrinsics_dict[cam_id])
                fused.keypoints_3d = self._transform_points(
                    fused.keypoints_3d, T_world_cam
                )

            depth_estimates.append((fused, cam_id))

        if not depth_estimates:
            return triangulated_pose

        # Fuse triangulation with depth estimates
        refined_keypoints = triangulated_pose.keypoints_3d.copy()
        refined_confidence = triangulated_pose.keypoint_confidence.copy()

        w_tri = self.config.triangulation_weight
        w_depth = self.config.depth_weight

        for i in range(n_keypoints):
            if refined_confidence[i] < 0.1:
                # Triangulation failed - use depth only
                depth_points = []
                depth_confs = []

                for fused, _ in depth_estimates:
                    if fused.keypoint_confidence[i] > 0.3:
                        depth_points.append(fused.keypoints_3d[i])
                        depth_confs.append(fused.keypoint_confidence[i])

                if depth_points:
                    # Weighted average of depth estimates
                    weights = np.array(depth_confs)
                    weights /= weights.sum()
                    refined_keypoints[i] = np.average(depth_points, weights=weights, axis=0)
                    refined_confidence[i] = np.mean(depth_confs)

            else:
                # Have triangulation - blend with depth
                depth_points = []
                depth_confs = []

                for fused, _ in depth_estimates:
                    if fused.keypoint_confidence[i] > 0.3:
                        depth_points.append(fused.keypoints_3d[i])
                        depth_confs.append(fused.keypoint_confidence[i])

                if depth_points:
                    # Weighted average of depth estimates
                    weights = np.array(depth_confs)
                    weights /= weights.sum()
                    depth_avg = np.average(depth_points, weights=weights, axis=0)

                    # Blend triangulation and depth
                    refined_keypoints[i] = (
                        w_tri * refined_keypoints[i] +
                        w_depth * depth_avg
                    )

        # Update pose
        triangulated_pose.keypoints_3d = refined_keypoints
        triangulated_pose.keypoint_confidence = refined_confidence

        return triangulated_pose

    def _transform_points(
        self,
        points: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Transform points by 4x4 transformation matrix."""
        n = len(points)
        points_h = np.hstack([points, np.ones((n, 1))])
        transformed = (T @ points_h.T).T
        return transformed[:, :3]

    def reset(self):
        """Reset temporal state."""
        self._prev_pose = None
        self._prev_timestamp = 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

def create_intrinsics_from_fov(
    width: int,
    height: int,
    fov_horizontal: float = 60.0,
) -> CameraIntrinsics:
    """
    Create camera intrinsics from horizontal field of view.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_horizontal: Horizontal FOV in degrees

    Returns:
        CameraIntrinsics instance
    """
    fov_rad = np.radians(fov_horizontal)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2

    return CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height
    )


def depth_to_point_cloud_simple(
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    downsample: int = 1,
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud.

    Args:
        depth_map: [H, W] depth in meters
        intrinsics: Camera intrinsic parameters
        downsample: Downsampling factor

    Returns:
        [N, 3] array of 3D points
    """
    h, w = depth_map.shape

    # Create pixel grids
    u = np.arange(0, w, downsample)
    v = np.arange(0, h, downsample)
    u_grid, v_grid = np.meshgrid(u, v)

    # Sample depth
    z = depth_map[v_grid, u_grid]

    # Valid mask
    valid = z > 0.1

    # Back-project
    x = (u_grid - intrinsics.cx) * z / intrinsics.fx
    y = (v_grid - intrinsics.cy) * z / intrinsics.fy

    points = np.stack([x, y, z], axis=-1)
    return points[valid]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Depth-Pose Fusion")
    print("=" * 50)

    # Create test data
    from dataclasses import dataclass

    @dataclass
    class MockPose2D:
        keypoints: np.ndarray
        num_persons: int = 1

    # Mock 2D pose (17 keypoints)
    kp_2d = np.zeros((1, 17, 3))
    kp_2d[0, 0] = [640, 200, 0.95]   # nose
    kp_2d[0, 5] = [560, 300, 0.9]    # left shoulder
    kp_2d[0, 6] = [720, 300, 0.9]    # right shoulder
    kp_2d[0, 7] = [520, 450, 0.85]   # left elbow
    kp_2d[0, 8] = [760, 450, 0.85]   # right elbow
    kp_2d[0, 9] = [500, 600, 0.8]    # left wrist
    kp_2d[0, 10] = [780, 600, 0.8]   # right wrist

    pose_2d = MockPose2D(keypoints=kp_2d)

    # Mock depth result
    depth_map = np.ones((720, 1280), dtype=np.float32) * 2.5  # 2.5m uniform
    depth_result = DepthResult(
        depth_map=depth_map,
        timestamp=time.time(),
        original_size=(1280, 720),
    )

    # Camera intrinsics
    intrinsics = create_intrinsics_from_fov(1280, 720, fov_horizontal=60)

    # Create fusion
    fusion = DepthPoseFusion()

    # Estimate 3D
    pose_3d = fusion.estimate_from_depth(pose_2d, depth_result, intrinsics)

    print(f"Valid keypoint ratio: {pose_3d.valid_keypoint_ratio:.1%}")
    print(f"Mean depth confidence: {pose_3d.mean_depth_confidence:.2f}")
    print(f"\nSample 3D keypoints:")
    for i in [0, 5, 9]:
        if pose_3d.keypoint_confidence[i] > 0:
            x, y, z = pose_3d.keypoints_3d[i]
            print(f"  Keypoint {i}: ({x:.2f}, {y:.2f}, {z:.2f})m")

    print("\nTest passed!")

"""
Isaac Lab Camera Manager

Manages virtual cameras in Isaac Lab simulation for multi-view perception.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a virtual camera."""
    camera_id: str
    position: Tuple[float, float, float]  # XYZ in world frame
    target: Tuple[float, float, float]  # Look-at point
    resolution: Tuple[int, int] = (640, 480)
    fov: float = 60.0  # Vertical FOV in degrees
    near_clip: float = 0.1
    far_clip: float = 10.0
    enable_rgb: bool = True
    enable_depth: bool = True
    enable_segmentation: bool = False
    enable_normals: bool = False


class IsaacCameraManager:
    """
    Manages multiple virtual cameras in Isaac Lab.

    Features:
    - Multi-camera setup for multi-view perception
    - RGB, depth, segmentation, and normal rendering
    - Camera intrinsics/extrinsics computation
    - Streaming to Dynamical platform
    """

    # Preset camera configurations for common setups
    PRESETS = {
        "manipulation_4cam": [
            CameraConfig("front", (1.0, 0.0, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("left", (0.4, 0.8, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("right", (0.4, -0.8, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("wrist", (0.5, 0.0, 0.5), (0.4, 0.0, 0.2)),
        ],
        "manipulation_12cam": [
            CameraConfig("front", (1.0, 0.0, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("front_left", (0.8, 0.5, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("front_right", (0.8, -0.5, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("left", (0.4, 0.8, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("right", (0.4, -0.8, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("back_left", (0.0, 0.5, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("back_right", (0.0, -0.5, 0.8), (0.4, 0.0, 0.2)),
            CameraConfig("top", (0.4, 0.0, 1.5), (0.4, 0.0, 0.2)),
            CameraConfig("top_left", (0.4, 0.3, 1.3), (0.4, 0.0, 0.2)),
            CameraConfig("top_right", (0.4, -0.3, 1.3), (0.4, 0.0, 0.2)),
            CameraConfig("wrist", (0.5, 0.0, 0.5), (0.4, 0.0, 0.2)),
            CameraConfig("gripper", (0.5, 0.0, 0.4), (0.5, 0.0, 0.2)),
        ],
        "warehouse": [
            CameraConfig("overhead_1", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (1280, 720)),
            CameraConfig("overhead_2", (5.0, 0.0, 5.0), (5.0, 0.0, 0.0), (1280, 720)),
            CameraConfig("aisle_1", (2.5, -3.0, 2.0), (2.5, 0.0, 1.0), (1280, 720)),
            CameraConfig("aisle_2", (2.5, 3.0, 2.0), (2.5, 0.0, 1.0), (1280, 720)),
        ],
    }

    def __init__(
        self,
        preset: str = "manipulation_4cam",
        custom_configs: Optional[List[CameraConfig]] = None,
    ):
        """
        Initialize camera manager.

        Args:
            preset: Preset camera configuration name
            custom_configs: Custom camera configurations (overrides preset)
        """
        if custom_configs:
            self.cameras = {cfg.camera_id: cfg for cfg in custom_configs}
        elif preset in self.PRESETS:
            self.cameras = {cfg.camera_id: cfg for cfg in self.PRESETS[preset]}
        else:
            self.cameras = {cfg.camera_id: cfg for cfg in self.PRESETS["manipulation_4cam"]}

        # Isaac Lab camera handles (populated during setup)
        self._camera_handles: Dict[str, Any] = {}

        # Cached intrinsics/extrinsics
        self._intrinsics: Dict[str, np.ndarray] = {}
        self._extrinsics: Dict[str, np.ndarray] = {}

        # Frame buffers
        self._rgb_buffers: Dict[str, np.ndarray] = {}
        self._depth_buffers: Dict[str, np.ndarray] = {}
        self._segmentation_buffers: Dict[str, np.ndarray] = {}

        # Compute camera matrices
        self._compute_camera_matrices()

        logger.info(f"Camera manager initialized with {len(self.cameras)} cameras")

    def _compute_camera_matrices(self) -> None:
        """Compute intrinsic and extrinsic matrices for all cameras."""
        for cam_id, config in self.cameras.items():
            self._intrinsics[cam_id] = self._compute_intrinsics(config)
            self._extrinsics[cam_id] = self._compute_extrinsics(config)

    def _compute_intrinsics(self, config: CameraConfig) -> np.ndarray:
        """Compute camera intrinsic matrix."""
        w, h = config.resolution
        fov_rad = np.deg2rad(config.fov)

        # Focal length from FOV
        fy = h / (2 * np.tan(fov_rad / 2))
        fx = fy  # Assuming square pixels

        cx, cy = w / 2, h / 2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def _compute_extrinsics(self, config: CameraConfig) -> np.ndarray:
        """Compute camera extrinsic matrix (world to camera)."""
        position = np.array(config.position)
        target = np.array(config.target)

        # Compute view direction (z-axis points into camera)
        z = position - target
        z = z / np.linalg.norm(z)

        # Compute right vector (x-axis)
        world_up = np.array([0, 0, 1])
        x = np.cross(world_up, z)
        if np.linalg.norm(x) < 1e-6:
            # Looking straight up/down, use different up vector
            world_up = np.array([0, 1, 0])
            x = np.cross(world_up, z)
        x = x / np.linalg.norm(x)

        # Compute up vector (y-axis)
        y = np.cross(z, x)

        # Build rotation matrix (camera to world)
        R_cw = np.stack([x, y, z], axis=1)

        # World to camera rotation
        R = R_cw.T

        # Translation
        t = -R @ position

        # Build 4x4 extrinsic matrix
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t

        return extrinsics

    def setup_isaac_cameras(self, sim_context: Any) -> None:
        """
        Setup cameras in Isaac Lab simulation.

        Args:
            sim_context: Isaac Lab simulation context
        """
        try:
            from omni.isaac.lab.sensors import Camera, CameraCfg
            import omni.isaac.core.utils.prims as prim_utils

            for cam_id, config in self.cameras.items():
                # Create camera prim
                prim_path = f"/World/Cameras/{cam_id}"

                cam_cfg = CameraCfg(
                    prim_path=prim_path,
                    update_period=1/30.0,  # 30 Hz
                    height=config.resolution[1],
                    width=config.resolution[0],
                    data_types=["rgb", "depth"] + (["instance_segmentation"] if config.enable_segmentation else []),
                )

                camera = Camera(cam_cfg)
                camera.set_world_pose(
                    pos=config.position,
                    quat=self._look_at_quaternion(config.position, config.target),
                )

                self._camera_handles[cam_id] = camera

                logger.info(f"Created Isaac Lab camera: {cam_id}")

        except ImportError:
            logger.warning("Isaac Lab not available, using standalone camera rendering")

    def _look_at_quaternion(
        self,
        position: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> np.ndarray:
        """Compute quaternion for camera looking at target."""
        position = np.array(position)
        target = np.array(target)

        # Forward direction
        forward = target - position
        forward = forward / np.linalg.norm(forward)

        # Right direction
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        # Up direction
        up = np.cross(right, forward)

        # Rotation matrix
        R = np.stack([right, up, -forward], axis=1)

        # Convert to quaternion
        return self._rotation_matrix_to_quaternion(R)

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (wxyz)."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        return np.array([w, x, y, z])

    def get_frames(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get frames from all cameras.

        Returns:
            Dictionary mapping camera_id to frame data
        """
        frames = {}

        for cam_id, config in self.cameras.items():
            if cam_id in self._camera_handles:
                # Get from Isaac Lab
                camera = self._camera_handles[cam_id]
                data = camera.data

                frames[cam_id] = {
                    "rgb": data.output["rgb"],
                    "depth": data.output.get("depth"),
                    "segmentation": data.output.get("instance_segmentation"),
                }
            else:
                # Standalone mode - return cached/synthetic data
                frames[cam_id] = {
                    "rgb": self._rgb_buffers.get(cam_id, np.zeros((*config.resolution[::-1], 3), dtype=np.uint8)),
                    "depth": self._depth_buffers.get(cam_id, np.zeros(config.resolution[::-1], dtype=np.float32)),
                    "segmentation": self._segmentation_buffers.get(cam_id),
                }

            # Add camera matrices
            frames[cam_id]["intrinsics"] = self._intrinsics[cam_id]
            frames[cam_id]["extrinsics"] = self._extrinsics[cam_id]

        return frames

    def get_frame(self, camera_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Get frame from specific camera."""
        frames = self.get_frames()
        return frames.get(camera_id)

    def get_intrinsics(self, camera_id: str) -> Optional[np.ndarray]:
        """Get intrinsic matrix for camera."""
        return self._intrinsics.get(camera_id)

    def get_extrinsics(self, camera_id: str) -> Optional[np.ndarray]:
        """Get extrinsic matrix for camera."""
        return self._extrinsics.get(camera_id)

    def project_point(
        self,
        point_3d: np.ndarray,
        camera_id: str,
    ) -> Optional[np.ndarray]:
        """
        Project 3D point to 2D image coordinates.

        Args:
            point_3d: 3D point in world frame (3,)
            camera_id: Camera ID

        Returns:
            2D point in image coordinates (2,) or None if behind camera
        """
        if camera_id not in self._intrinsics:
            return None

        K = self._intrinsics[camera_id]
        E = self._extrinsics[camera_id]

        # Transform to camera frame
        point_h = np.append(point_3d, 1)
        point_cam = E @ point_h

        # Check if behind camera
        if point_cam[2] <= 0:
            return None

        # Project
        point_2d_h = K @ point_cam[:3]
        point_2d = point_2d_h[:2] / point_2d_h[2]

        return point_2d

    def unproject_point(
        self,
        point_2d: np.ndarray,
        depth: float,
        camera_id: str,
    ) -> Optional[np.ndarray]:
        """
        Unproject 2D point to 3D using depth.

        Args:
            point_2d: 2D point in image coordinates (2,)
            depth: Depth value at point
            camera_id: Camera ID

        Returns:
            3D point in world frame (3,)
        """
        if camera_id not in self._intrinsics:
            return None

        K = self._intrinsics[camera_id]
        E = self._extrinsics[camera_id]

        # Unproject to camera frame
        K_inv = np.linalg.inv(K)
        point_2d_h = np.array([point_2d[0], point_2d[1], 1])
        ray_cam = K_inv @ point_2d_h
        point_cam = ray_cam * depth

        # Transform to world frame
        E_inv = np.linalg.inv(E)
        point_cam_h = np.append(point_cam, 1)
        point_world_h = E_inv @ point_cam_h

        return point_world_h[:3]

    def triangulate_point(
        self,
        observations: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from multi-view observations.

        Args:
            observations: Dictionary mapping camera_id to 2D point (2,)

        Returns:
            3D point in world frame (3,) or None if insufficient observations
        """
        if len(observations) < 2:
            return None

        # Build linear system for triangulation
        A = []

        for cam_id, point_2d in observations.items():
            if cam_id not in self._intrinsics:
                continue

            K = self._intrinsics[cam_id]
            E = self._extrinsics[cam_id]

            P = K @ E[:3, :]  # 3x4 projection matrix

            x, y = point_2d
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        if len(A) < 4:
            return None

        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        X = X[:3] / X[3]  # Dehomogenize

        return X

    def get_camera_ids(self) -> List[str]:
        """Get list of camera IDs."""
        return list(self.cameras.keys())

    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        """Get configuration for specific camera."""
        return self.cameras.get(camera_id)

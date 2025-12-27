"""
Robot Registry - Multi-Robot Configuration Management

Enables CROSS-ROBOT TRANSFER by:
1. Centralizing robot kinematic configurations
2. Providing per-robot camera calibrations
3. Supporting dynamic robot registration
4. Enabling embodiment-agnostic skill execution

This addresses MolmoAct Gap #3: "Dynamical's skills are trained per-robot"
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class RobotType(str, Enum):
    """Supported robot types."""
    MANIPULATOR = "manipulator"      # Single arm (UR, Franka, etc.)
    DUAL_ARM = "dual_arm"            # Two arms
    HUMANOID = "humanoid"            # Full humanoid
    MOBILE_MANIPULATOR = "mobile"    # Arm on mobile base
    CUSTOM = "custom"


class ControlMode(str, Enum):
    """Robot control modes."""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    HYBRID = "hybrid"


@dataclass
class CameraConfig:
    """Camera configuration for a robot."""
    camera_id: str
    name: str

    # Intrinsics [3, 3]
    intrinsics: np.ndarray

    # Extrinsics: camera pose in robot base frame [4, 4]
    extrinsics: np.ndarray

    # Image dimensions
    width: int = 640
    height: int = 480

    # Distortion coefficients [5] or [8]
    distortion: Optional[np.ndarray] = None

    # Camera type
    camera_type: str = "rgb"  # "rgb", "depth", "rgbd"

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to image coordinates."""
        # Transform to camera frame
        point_cam = np.linalg.inv(self.extrinsics)[:3, :3] @ point_3d + \
                    np.linalg.inv(self.extrinsics)[:3, 3]

        # Project
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        u = fx * point_cam[0] / point_cam[2] + cx
        v = fy * point_cam[1] / point_cam[2] + cy

        return np.array([u, v])

    def unproject_pixel(
        self,
        pixel: np.ndarray,
        depth: float,
    ) -> np.ndarray:
        """Unproject pixel to 3D point in robot base frame."""
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        # Camera frame
        x = (pixel[0] - cx) * depth / fx
        y = (pixel[1] - cy) * depth / fy
        z = depth

        point_cam = np.array([x, y, z])

        # Transform to base frame
        point_base = self.extrinsics[:3, :3] @ point_cam + self.extrinsics[:3, 3]

        return point_base


@dataclass
class RobotConfig:
    """
    Complete robot configuration for embodiment-agnostic planning.

    Contains all information needed to:
    1. Convert trajectory traces to joint commands
    2. Execute skills on this specific robot
    3. Validate actions against robot limits
    """
    # Identification
    robot_id: str
    name: str
    robot_type: RobotType = RobotType.MANIPULATOR

    # Kinematics
    dof: int = 7
    urdf_path: Optional[str] = None

    # Joint configuration
    joint_names: List[str] = field(default_factory=list)
    joint_limits_lower: np.ndarray = field(
        default_factory=lambda: np.array([-2*np.pi]*7)
    )
    joint_limits_upper: np.ndarray = field(
        default_factory=lambda: np.array([2*np.pi]*7)
    )
    velocity_limits: np.ndarray = field(
        default_factory=lambda: np.array([2.0]*7)
    )
    acceleration_limits: np.ndarray = field(
        default_factory=lambda: np.array([4.0]*7)
    )

    # End-effector
    ee_link: str = "tool0"
    base_link: str = "base_link"

    # Control
    default_control_mode: ControlMode = ControlMode.JOINT_POSITION
    control_frequency_hz: float = 100.0

    # Cameras
    cameras: List[CameraConfig] = field(default_factory=list)

    # Workspace limits [min, max] for x, y, z
    workspace_limits: Optional[np.ndarray] = None

    # Home position (joint angles)
    home_position: Optional[np.ndarray] = None

    # Gripper configuration
    has_gripper: bool = True
    gripper_dof: int = 1
    gripper_range: Tuple[float, float] = (0.0, 1.0)

    # Metadata
    manufacturer: str = ""
    model: str = ""
    payload_kg: float = 5.0
    reach_m: float = 1.0

    def __post_init__(self):
        """Validate and initialize configuration."""
        # Ensure numpy arrays
        self.joint_limits_lower = np.asarray(self.joint_limits_lower, dtype=np.float32)
        self.joint_limits_upper = np.asarray(self.joint_limits_upper, dtype=np.float32)
        self.velocity_limits = np.asarray(self.velocity_limits, dtype=np.float32)
        self.acceleration_limits = np.asarray(self.acceleration_limits, dtype=np.float32)

        # Validate dimensions
        if len(self.joint_limits_lower) != self.dof:
            raise ValueError(f"joint_limits_lower length {len(self.joint_limits_lower)} != dof {self.dof}")

        # Initialize joint names if not provided
        if not self.joint_names:
            self.joint_names = [f"joint_{i}" for i in range(self.dof)]

        # Initialize home position if not provided
        if self.home_position is None:
            self.home_position = (self.joint_limits_lower + self.joint_limits_upper) / 2

    @property
    def primary_camera(self) -> Optional[CameraConfig]:
        """Get the primary camera (first RGB camera)."""
        for cam in self.cameras:
            if cam.camera_type in ["rgb", "rgbd"]:
                return cam
        return self.cameras[0] if self.cameras else None

    def is_within_limits(self, joint_positions: np.ndarray) -> bool:
        """Check if joint positions are within limits."""
        return (
            np.all(joint_positions >= self.joint_limits_lower) and
            np.all(joint_positions <= self.joint_limits_upper)
        )

    def clip_to_limits(self, joint_positions: np.ndarray) -> np.ndarray:
        """Clip joint positions to limits."""
        return np.clip(
            joint_positions,
            self.joint_limits_lower,
            self.joint_limits_upper
        )

    def is_within_workspace(self, position: np.ndarray) -> bool:
        """Check if Cartesian position is within workspace."""
        if self.workspace_limits is None:
            return True
        return (
            np.all(position >= self.workspace_limits[0]) and
            np.all(position <= self.workspace_limits[1])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "robot_type": self.robot_type.value,
            "dof": self.dof,
            "urdf_path": self.urdf_path,
            "joint_names": self.joint_names,
            "joint_limits_lower": self.joint_limits_lower.tolist(),
            "joint_limits_upper": self.joint_limits_upper.tolist(),
            "velocity_limits": self.velocity_limits.tolist(),
            "ee_link": self.ee_link,
            "base_link": self.base_link,
            "has_gripper": self.has_gripper,
            "manufacturer": self.manufacturer,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        """Deserialize from dictionary."""
        return cls(
            robot_id=data["robot_id"],
            name=data["name"],
            robot_type=RobotType(data.get("robot_type", "manipulator")),
            dof=data["dof"],
            urdf_path=data.get("urdf_path"),
            joint_names=data.get("joint_names", []),
            joint_limits_lower=np.array(data["joint_limits_lower"]),
            joint_limits_upper=np.array(data["joint_limits_upper"]),
            velocity_limits=np.array(data.get("velocity_limits", [2.0] * data["dof"])),
            ee_link=data.get("ee_link", "tool0"),
            base_link=data.get("base_link", "base_link"),
            has_gripper=data.get("has_gripper", True),
            manufacturer=data.get("manufacturer", ""),
            model=data.get("model", ""),
        )


class RobotRegistry:
    """
    Central registry for robot configurations.

    Enables cross-robot transfer by:
    1. Storing all robot configurations in one place
    2. Providing lookup by robot ID or capabilities
    3. Supporting runtime robot registration
    """

    _instance: Optional["RobotRegistry"] = None
    _robots: Dict[str, RobotConfig] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._robots = {}
            cls._instance._initialize_defaults()
        return cls._instance

    def _initialize_defaults(self):
        """Initialize with default robot configurations."""
        # UR10e
        self.register(RobotConfig(
            robot_id="ur10e",
            name="Universal Robots UR10e",
            robot_type=RobotType.MANIPULATOR,
            dof=6,
            urdf_path="urdfs/ur10e/ur10e.urdf",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
            joint_limits_lower=np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            joint_limits_upper=np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
            velocity_limits=np.array([2.09, 2.09, 3.14, 3.14, 3.14, 3.14]),
            ee_link="tool0",
            base_link="base_link",
            manufacturer="Universal Robots",
            model="UR10e",
            payload_kg=10.0,
            reach_m=1.3,
            cameras=[
                CameraConfig(
                    camera_id="wrist_cam",
                    name="Wrist Camera",
                    intrinsics=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32),
                    extrinsics=np.eye(4, dtype=np.float32),
                    width=640,
                    height=480,
                )
            ],
        ))

        # Franka Panda
        self.register(RobotConfig(
            robot_id="franka_panda",
            name="Franka Emika Panda",
            robot_type=RobotType.MANIPULATOR,
            dof=7,
            urdf_path="urdfs/franka/panda.urdf",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            joint_limits_lower=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            joint_limits_upper=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
            velocity_limits=np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
            ee_link="panda_link8",
            base_link="panda_link0",
            manufacturer="Franka Emika",
            model="Panda",
            payload_kg=3.0,
            reach_m=0.855,
            cameras=[
                CameraConfig(
                    camera_id="external_cam",
                    name="External Camera",
                    intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
                    extrinsics=np.eye(4, dtype=np.float32),
                )
            ],
        ))

        # Humanoid 23-DOF
        self.register(RobotConfig(
            robot_id="humanoid_23dof",
            name="Humanoid 23-DOF",
            robot_type=RobotType.HUMANOID,
            dof=23,
            urdf_path="urdfs/humanoid/humanoid.urdf",
            joint_names=[
                # Torso (3)
                "torso_yaw", "torso_pitch", "torso_roll",
                # Right arm (7)
                "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
                "r_elbow_pitch", "r_elbow_roll",
                "r_wrist_pitch", "r_wrist_roll",
                # Left arm (7)
                "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
                "l_elbow_pitch", "l_elbow_roll",
                "l_wrist_pitch", "l_wrist_roll",
                # Right hand (3 simplified)
                "r_hand_thumb", "r_hand_index", "r_hand_grip",
                # Left hand (3 simplified)
                "l_hand_thumb", "l_hand_index", "l_hand_grip",
            ],
            joint_limits_lower=np.array([-np.pi]*23),
            joint_limits_upper=np.array([np.pi]*23),
            velocity_limits=np.array([1.5]*23),
            ee_link="r_hand_link",
            base_link="torso_base",
            manufacturer="Custom",
            model="Humanoid-23DOF",
            payload_kg=5.0,
            reach_m=0.8,
            cameras=[
                CameraConfig(
                    camera_id="head_cam",
                    name="Head Camera",
                    intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
                    extrinsics=np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0.5],
                        [0, 0, 1, 1.5],
                        [0, 0, 0, 1]
                    ], dtype=np.float32),
                )
            ],
        ))

    def register(self, config: RobotConfig) -> None:
        """Register a robot configuration."""
        self._robots[config.robot_id] = config
        logger.info(f"Registered robot: {config.robot_id} ({config.name})")

    def unregister(self, robot_id: str) -> bool:
        """Unregister a robot."""
        if robot_id in self._robots:
            del self._robots[robot_id]
            logger.info(f"Unregistered robot: {robot_id}")
            return True
        return False

    def get(self, robot_id: str) -> Optional[RobotConfig]:
        """Get robot configuration by ID."""
        return self._robots.get(robot_id)

    def get_or_raise(self, robot_id: str) -> RobotConfig:
        """Get robot configuration or raise error."""
        config = self.get(robot_id)
        if config is None:
            available = ", ".join(self._robots.keys())
            raise ValueError(f"Unknown robot: {robot_id}. Available: {available}")
        return config

    def list_robots(self) -> List[str]:
        """List all registered robot IDs."""
        return list(self._robots.keys())

    def list_by_type(self, robot_type: RobotType) -> List[str]:
        """List robots of a specific type."""
        return [
            rid for rid, config in self._robots.items()
            if config.robot_type == robot_type
        ]

    def find_by_dof(self, dof: int) -> List[str]:
        """Find robots with specific DOF."""
        return [
            rid for rid, config in self._robots.items()
            if config.dof == dof
        ]

    def get_all(self) -> Dict[str, RobotConfig]:
        """Get all robot configurations."""
        return self._robots.copy()

    def clear(self) -> None:
        """Clear all registrations (use with caution)."""
        self._robots.clear()

    @classmethod
    def instance(cls) -> "RobotRegistry":
        """Get singleton instance."""
        return cls()


# ============================================================================
# Convenience Functions
# ============================================================================

def get_robot(robot_id: str) -> RobotConfig:
    """Get robot configuration by ID."""
    return RobotRegistry.instance().get_or_raise(robot_id)


def list_robots() -> List[str]:
    """List all registered robots."""
    return RobotRegistry.instance().list_robots()


def register_robot(config: RobotConfig) -> None:
    """Register a new robot."""
    RobotRegistry.instance().register(config)


def create_camera_config(
    camera_id: str,
    fx: float = 500,
    fy: float = 500,
    cx: float = 320,
    cy: float = 240,
    position: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
) -> CameraConfig:
    """Create a camera configuration with common defaults."""
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    extrinsics = np.eye(4, dtype=np.float32)
    if position is not None:
        extrinsics[:3, 3] = position
    if rotation is not None:
        extrinsics[:3, :3] = rotation

    return CameraConfig(
        camera_id=camera_id,
        name=camera_id,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

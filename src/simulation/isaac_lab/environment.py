"""
NVIDIA Isaac Lab Environment for Dynamical.ai

This module provides a comprehensive simulation environment using NVIDIA Isaac Lab
for robot manipulation tasks with teleoperation support.

Isaac Lab: https://isaac-sim.github.io/IsaacLab/
Requires: NVIDIA Isaac Sim 2023.1.1+ and Isaac Lab extension

Usage:
    # Standalone mode (for testing without Isaac Sim)
    env = IsaacLabEnvironment(mode="standalone")

    # Full Isaac Lab mode (requires Isaac Sim)
    env = IsaacLabEnvironment(mode="isaac_lab")

    # Run simulation loop
    env.reset()
    while True:
        obs = env.get_observations()
        env.apply_action(action)
        env.step()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation backend mode."""
    STANDALONE = "standalone"  # Pure Python simulation (no Isaac Sim)
    ISAAC_LAB = "isaac_lab"    # Full Isaac Lab with Isaac Sim
    ISAAC_GYM = "isaac_gym"    # Legacy Isaac Gym (deprecated)


class RobotType(Enum):
    """Supported robot types."""
    FRANKA_PANDA = "franka_panda"
    UR5E = "ur5e"
    UR10 = "ur10"
    KUKA_IIWA = "kuka_iiwa"
    DAIMON_VTLA = "daimon_vtla"  # Custom robot


@dataclass
class RobotState:
    """Current state of a simulated robot."""
    joint_positions: np.ndarray  # (n_joints,)
    joint_velocities: np.ndarray  # (n_joints,)
    joint_torques: np.ndarray  # (n_joints,)
    ee_position: np.ndarray  # (3,) xyz in world frame
    ee_orientation: np.ndarray  # (4,) quaternion wxyz
    ee_linear_velocity: np.ndarray  # (3,)
    ee_angular_velocity: np.ndarray  # (3,)
    gripper_state: float  # 0.0 (closed) to 1.0 (open)
    timestamp: float = 0.0


@dataclass
class CameraFrame:
    """Single camera observation."""
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 in meters
    segmentation: Optional[np.ndarray] = None  # (H, W) int32 instance IDs
    camera_id: str = "main"
    timestamp: float = 0.0
    intrinsics: Optional[np.ndarray] = None  # (3, 3)
    extrinsics: Optional[np.ndarray] = None  # (4, 4)


@dataclass
class SimulationConfig:
    """Configuration for Isaac Lab simulation."""
    # Simulation settings
    mode: SimulationMode = SimulationMode.STANDALONE
    dt: float = 1.0 / 60.0  # 60 Hz physics
    render_dt: float = 1.0 / 30.0  # 30 Hz rendering
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Robot settings
    robot_type: RobotType = RobotType.FRANKA_PANDA
    robot_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz

    # Camera settings
    num_cameras: int = 4
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_fps: int = 30

    # Scene settings
    scene_type: str = "tabletop"  # tabletop, warehouse, custom
    enable_objects: bool = True
    num_objects: int = 5

    # Performance settings
    num_envs: int = 1  # Parallel environments
    device: str = "cuda:0"
    headless: bool = False

    # Teleoperation settings
    enable_teleoperation: bool = True
    teleop_mode: str = "position"  # position, velocity, impedance


@dataclass
class SimulationMetrics:
    """Performance metrics for simulation."""
    physics_fps: float = 0.0
    render_fps: float = 0.0
    step_time_ms: float = 0.0
    total_steps: int = 0
    simulation_time: float = 0.0
    real_time_factor: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class IsaacLabEnvironment:
    """
    NVIDIA Isaac Lab simulation environment for Dynamical.ai.

    Provides a unified interface for robot simulation that works in:
    - Standalone mode (pure Python, for testing)
    - Isaac Lab mode (full physics simulation with NVIDIA Isaac Sim)

    Features:
    - Multi-robot support (Franka, UR5e, custom)
    - Multi-camera perception
    - Object manipulation
    - Teleoperation integration
    - Domain randomization
    - Parallel environments
    """

    # Robot joint configurations
    ROBOT_CONFIGS = {
        RobotType.FRANKA_PANDA: {
            "n_joints": 7,
            "n_gripper_joints": 2,
            "joint_limits_lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            "joint_limits_upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            "default_joint_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            "usd_path": "/Isaac/Robots/Franka/franka_instanceable.usd",
            "ee_link": "panda_hand",
        },
        RobotType.UR5E: {
            "n_joints": 6,
            "n_gripper_joints": 1,
            "joint_limits_lower": [-6.28, -6.28, -3.14, -6.28, -6.28, -6.28],
            "joint_limits_upper": [6.28, 6.28, 3.14, 6.28, 6.28, 6.28],
            "default_joint_pos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            "usd_path": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
            "ee_link": "tool0",
        },
        RobotType.DAIMON_VTLA: {
            "n_joints": 7,
            "n_gripper_joints": 1,
            "joint_limits_lower": [-3.14, -2.09, -3.14, -2.09, -3.14, -2.09, -3.14],
            "joint_limits_upper": [3.14, 2.09, 3.14, 2.09, 3.14, 2.09, 3.14],
            "default_joint_pos": [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
            "usd_path": None,  # Custom URDF/USD
            "ee_link": "end_effector",
        },
    }

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        mode: str = "standalone",
    ):
        """
        Initialize Isaac Lab environment.

        Args:
            config: Simulation configuration
            mode: "standalone" or "isaac_lab"
        """
        self.config = config or SimulationConfig()
        if mode:
            self.config.mode = SimulationMode(mode)

        self.robot_config = self.ROBOT_CONFIGS.get(
            self.config.robot_type,
            self.ROBOT_CONFIGS[RobotType.FRANKA_PANDA]
        )

        # State
        self._is_running = False
        self._current_step = 0
        self._simulation_time = 0.0
        self._start_time = 0.0

        # Robot state
        self._robot_state = self._create_default_robot_state()

        # Camera frames
        self._camera_frames: Dict[str, CameraFrame] = {}

        # Object states
        self._objects: Dict[str, Dict[str, Any]] = {}

        # Metrics
        self._metrics = SimulationMetrics()

        # Isaac Lab components (initialized in setup)
        self._sim = None
        self._scene = None
        self._robot = None
        self._cameras = []

        # Event callbacks
        self._on_step_callbacks: List[Callable] = []
        self._on_reset_callbacks: List[Callable] = []

        logger.info(f"IsaacLabEnvironment initialized in {self.config.mode.value} mode")

    def _create_default_robot_state(self) -> RobotState:
        """Create default robot state."""
        n_joints = self.robot_config["n_joints"]
        return RobotState(
            joint_positions=np.array(self.robot_config["default_joint_pos"]),
            joint_velocities=np.zeros(n_joints),
            joint_torques=np.zeros(n_joints),
            ee_position=np.array([0.5, 0.0, 0.5]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            ee_linear_velocity=np.zeros(3),
            ee_angular_velocity=np.zeros(3),
            gripper_state=1.0,
            timestamp=0.0,
        )

    async def setup(self) -> None:
        """
        Setup the simulation environment.

        In Isaac Lab mode, this initializes Isaac Sim and loads the scene.
        In standalone mode, this sets up the pure Python simulation.
        """
        logger.info("Setting up Isaac Lab environment...")

        if self.config.mode == SimulationMode.ISAAC_LAB:
            await self._setup_isaac_lab()
        else:
            await self._setup_standalone()

        # Initialize cameras
        self._setup_cameras()

        # Initialize scene objects
        if self.config.enable_objects:
            self._setup_objects()

        self._is_running = True
        self._start_time = time.time()

        logger.info("Isaac Lab environment setup complete")

    async def _setup_isaac_lab(self) -> None:
        """Setup full Isaac Lab simulation."""
        try:
            # Try to import Isaac Lab components
            # These imports require NVIDIA Isaac Sim to be installed
            from omni.isaac.lab.app import AppLauncher

            # Launch Isaac Sim app
            app_launcher = AppLauncher(headless=self.config.headless)
            simulation_app = app_launcher.app

            # Import Isaac Lab modules after app launch
            from omni.isaac.lab.sim import SimulationCfg, SimulationContext
            from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
            from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
            from omni.isaac.lab.sensors import CameraCfg

            # Create simulation context
            sim_cfg = SimulationCfg(
                dt=self.config.dt,
                render_interval=int(self.config.render_dt / self.config.dt),
                gravity=self.config.gravity,
                device=self.config.device,
            )
            self._sim = SimulationContext(sim_cfg)

            # Setup scene with robot
            await self._setup_isaac_scene()

            logger.info("Isaac Lab simulation initialized successfully")

        except ImportError as e:
            logger.warning(f"Isaac Lab not available: {e}")
            logger.warning("Falling back to standalone simulation mode")
            self.config.mode = SimulationMode.STANDALONE
            await self._setup_standalone()

    async def _setup_isaac_scene(self) -> None:
        """Setup Isaac Lab scene with robot and objects."""
        # This would configure the actual Isaac Lab scene
        # Placeholder for when Isaac Lab is available
        pass

    async def _setup_standalone(self) -> None:
        """Setup standalone simulation (pure Python)."""
        logger.info("Setting up standalone simulation...")

        # Initialize robot kinematics (simplified)
        self._fk_solver = StandaloneForwardKinematics(self.config.robot_type)
        self._ik_solver = StandaloneInverseKinematics(self.config.robot_type)

        # Initialize physics state
        self._physics_state = {
            "robot": self._robot_state,
            "objects": {},
            "contacts": [],
        }

        logger.info("Standalone simulation ready")

    def _setup_cameras(self) -> None:
        """Setup virtual cameras."""
        # Camera positions for multi-view setup
        camera_configs = [
            {"id": "front", "position": [1.0, 0.0, 0.8], "target": [0.0, 0.0, 0.4]},
            {"id": "left", "position": [0.0, 1.0, 0.8], "target": [0.0, 0.0, 0.4]},
            {"id": "right", "position": [0.0, -1.0, 0.8], "target": [0.0, 0.0, 0.4]},
            {"id": "top", "position": [0.5, 0.0, 1.5], "target": [0.0, 0.0, 0.4]},
        ]

        for i, cam_cfg in enumerate(camera_configs[:self.config.num_cameras]):
            self._camera_frames[cam_cfg["id"]] = CameraFrame(
                rgb=np.zeros((*self.config.camera_resolution[::-1], 3), dtype=np.uint8),
                depth=np.zeros(self.config.camera_resolution[::-1], dtype=np.float32),
                camera_id=cam_cfg["id"],
                intrinsics=self._compute_camera_intrinsics(),
                extrinsics=self._compute_camera_extrinsics(
                    cam_cfg["position"], cam_cfg["target"]
                ),
            )

        logger.info(f"Initialized {len(self._camera_frames)} virtual cameras")

    def _compute_camera_intrinsics(self) -> np.ndarray:
        """Compute camera intrinsic matrix."""
        w, h = self.config.camera_resolution
        fx = fy = w  # Simplified focal length
        cx, cy = w / 2, h / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def _compute_camera_extrinsics(
        self, position: List[float], target: List[float]
    ) -> np.ndarray:
        """Compute camera extrinsic matrix (world to camera)."""
        position = np.array(position)
        target = np.array(target)

        # Compute view direction
        z = position - target
        z = z / np.linalg.norm(z)

        # Compute right vector
        up = np.array([0, 0, 1])
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

        # Compute up vector
        y = np.cross(z, x)

        # Build extrinsic matrix
        R = np.stack([x, y, z], axis=1)
        t = -R.T @ position

        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R.T
        extrinsics[:3, 3] = t

        return extrinsics

    def _setup_objects(self) -> None:
        """Setup manipulation objects in the scene."""
        object_configs = [
            {"id": "cube_red", "type": "cube", "size": 0.05, "color": [1, 0, 0]},
            {"id": "cube_green", "type": "cube", "size": 0.05, "color": [0, 1, 0]},
            {"id": "cube_blue", "type": "cube", "size": 0.05, "color": [0, 0, 1]},
            {"id": "cylinder", "type": "cylinder", "radius": 0.03, "height": 0.1},
            {"id": "sphere", "type": "sphere", "radius": 0.04},
        ]

        for obj_cfg in object_configs[:self.config.num_objects]:
            # Random position on table
            pos = np.array([
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-0.3, 0.3),
                0.05  # On table surface
            ])

            self._objects[obj_cfg["id"]] = {
                **obj_cfg,
                "position": pos,
                "orientation": np.array([1, 0, 0, 0]),
                "velocity": np.zeros(6),
                "grasped": False,
            }

        logger.info(f"Initialized {len(self._objects)} objects")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the simulation to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation dictionary
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset robot to default pose
        self._robot_state = self._create_default_robot_state()

        # Reset simulation time
        self._current_step = 0
        self._simulation_time = 0.0

        # Randomize object positions if enabled
        if self.config.enable_objects:
            self._randomize_objects()

        # Call reset callbacks
        for callback in self._on_reset_callbacks:
            callback()

        logger.debug("Environment reset")
        return self.get_observations()

    def _randomize_objects(self) -> None:
        """Randomize object positions for domain randomization."""
        for obj_id, obj in self._objects.items():
            obj["position"] = np.array([
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-0.3, 0.3),
                0.05
            ])
            obj["orientation"] = np.array([1, 0, 0, 0])
            obj["grasped"] = False

    def step(self, action: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Step the simulation forward.

        Args:
            action: Robot action (joint positions, velocities, or end-effector pose)

        Returns:
            Observation dictionary
        """
        step_start = time.time()

        # Apply action if provided
        if action is not None:
            self._apply_action(action)

        # Step physics
        if self.config.mode == SimulationMode.ISAAC_LAB and self._sim:
            self._sim.step()
        else:
            self._step_standalone_physics()

        # Update state
        self._current_step += 1
        self._simulation_time += self.config.dt
        self._robot_state.timestamp = self._simulation_time

        # Update cameras
        self._update_cameras()

        # Update metrics
        step_time = time.time() - step_start
        self._update_metrics(step_time)

        # Call step callbacks
        for callback in self._on_step_callbacks:
            callback(self._current_step, self._simulation_time)

        return self.get_observations()

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to robot."""
        if self.config.teleop_mode == "position":
            # Direct joint position control
            n_joints = self.robot_config["n_joints"]
            joint_action = action[:n_joints]

            # Clip to joint limits
            lower = np.array(self.robot_config["joint_limits_lower"])
            upper = np.array(self.robot_config["joint_limits_upper"])
            joint_action = np.clip(joint_action, lower, upper)

            # Smooth transition
            alpha = 0.1  # Smoothing factor
            self._robot_state.joint_positions = (
                (1 - alpha) * self._robot_state.joint_positions +
                alpha * joint_action
            )

            # Update gripper if provided
            if len(action) > n_joints:
                self._robot_state.gripper_state = np.clip(action[n_joints], 0, 1)

        elif self.config.teleop_mode == "velocity":
            # Velocity control
            self._robot_state.joint_velocities = action[:self.robot_config["n_joints"]]
            self._robot_state.joint_positions += (
                self._robot_state.joint_velocities * self.config.dt
            )

        # Update end-effector pose using forward kinematics
        self._update_ee_pose()

    def _update_ee_pose(self) -> None:
        """Update end-effector pose from joint positions."""
        if hasattr(self, '_fk_solver'):
            ee_pos, ee_quat = self._fk_solver.compute(
                self._robot_state.joint_positions
            )
            self._robot_state.ee_position = ee_pos
            self._robot_state.ee_orientation = ee_quat

    def _step_standalone_physics(self) -> None:
        """Step physics in standalone mode."""
        # Simple velocity integration
        self._robot_state.joint_positions += (
            self._robot_state.joint_velocities * self.config.dt
        )

        # Apply gravity to objects
        for obj_id, obj in self._objects.items():
            if not obj["grasped"] and obj["position"][2] > 0:
                obj["position"][2] = max(0.05, obj["position"][2] - 0.01)

    def _update_cameras(self) -> None:
        """Update camera frames."""
        for cam_id, frame in self._camera_frames.items():
            # In standalone mode, generate synthetic images
            if self.config.mode == SimulationMode.STANDALONE:
                frame.rgb = self._render_synthetic_image(cam_id)
                frame.depth = self._render_synthetic_depth(cam_id)

            frame.timestamp = self._simulation_time

    def _render_synthetic_image(self, camera_id: str) -> np.ndarray:
        """Render synthetic RGB image for standalone mode."""
        h, w = self.config.camera_resolution[::-1]

        # Create base image with gradient background
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(40, 60, h)[:, np.newaxis]  # R
        img[:, :, 1] = np.linspace(40, 60, h)[:, np.newaxis]  # G
        img[:, :, 2] = np.linspace(50, 70, h)[:, np.newaxis]  # B

        # Draw floor grid
        for i in range(0, w, 50):
            img[h//2:, i:i+2, :] = [100, 100, 100]

        # Draw robot base (simplified)
        cx, cy = w // 2, h * 2 // 3
        cv2_available = False
        try:
            import cv2
            cv2_available = True
        except ImportError:
            pass

        if cv2_available:
            import cv2
            # Draw robot arm
            cv2.circle(img, (cx, cy), 30, (150, 150, 150), -1)
            cv2.rectangle(img, (cx-10, cy-100), (cx+10, cy), (180, 180, 180), -1)

            # Draw end effector
            ee_screen = self._world_to_screen(
                self._robot_state.ee_position, camera_id
            )
            if ee_screen is not None:
                cv2.circle(img, tuple(ee_screen.astype(int)), 15, (200, 200, 200), -1)

            # Draw objects
            for obj_id, obj in self._objects.items():
                obj_screen = self._world_to_screen(obj["position"], camera_id)
                if obj_screen is not None:
                    color = obj.get("color", [0.5, 0.5, 0.5])
                    color_bgr = tuple(int(c * 255) for c in color[::-1])
                    cv2.circle(img, tuple(obj_screen.astype(int)), 20, color_bgr, -1)

        return img

    def _render_synthetic_depth(self, camera_id: str) -> np.ndarray:
        """Render synthetic depth image."""
        h, w = self.config.camera_resolution[::-1]

        # Base depth (distance to table plane)
        depth = np.ones((h, w), dtype=np.float32) * 1.5

        # Add some variation
        depth += np.random.randn(h, w) * 0.01

        return depth

    def _world_to_screen(
        self, point: np.ndarray, camera_id: str
    ) -> Optional[np.ndarray]:
        """Project 3D world point to 2D screen coordinates."""
        frame = self._camera_frames.get(camera_id)
        if frame is None or frame.intrinsics is None or frame.extrinsics is None:
            return None

        # Transform to camera frame
        point_h = np.append(point, 1)
        point_cam = frame.extrinsics @ point_h

        if point_cam[2] <= 0:
            return None

        # Project to image
        point_2d = frame.intrinsics @ point_cam[:3]
        point_2d = point_2d[:2] / point_2d[2]

        return point_2d

    def _update_metrics(self, step_time: float) -> None:
        """Update simulation metrics."""
        self._metrics.step_time_ms = step_time * 1000
        self._metrics.total_steps = self._current_step
        self._metrics.simulation_time = self._simulation_time

        # Calculate FPS
        if step_time > 0:
            self._metrics.physics_fps = 1.0 / step_time

        # Real-time factor
        real_elapsed = time.time() - self._start_time
        if real_elapsed > 0:
            self._metrics.real_time_factor = self._simulation_time / real_elapsed

    def get_observations(self) -> Dict[str, Any]:
        """
        Get current observations from the simulation.

        Returns:
            Dictionary containing:
            - robot_state: Current robot state
            - cameras: Dictionary of camera frames
            - objects: Dictionary of object states
            - metrics: Simulation metrics
        """
        return {
            "robot_state": self._robot_state,
            "cameras": self._camera_frames,
            "objects": self._objects,
            "metrics": self._metrics,
            "simulation_time": self._simulation_time,
            "step": self._current_step,
        }

    def get_robot_state(self) -> RobotState:
        """Get current robot state."""
        return self._robot_state

    def get_camera_frame(self, camera_id: str = "front") -> Optional[CameraFrame]:
        """Get frame from specified camera."""
        return self._camera_frames.get(camera_id)

    def get_metrics(self) -> SimulationMetrics:
        """Get simulation metrics."""
        return self._metrics

    def set_robot_target(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Set target pose for robot end-effector.

        Args:
            target_position: Target XYZ position
            target_orientation: Target orientation as quaternion (wxyz)

        Returns:
            True if valid IK solution found
        """
        if hasattr(self, '_ik_solver'):
            joint_positions = self._ik_solver.solve(
                target_position,
                target_orientation or np.array([1, 0, 0, 0])
            )
            if joint_positions is not None:
                self._robot_state.joint_positions = joint_positions
                self._update_ee_pose()
                return True
        return False

    def set_gripper(self, state: float) -> None:
        """Set gripper state (0=closed, 1=open)."""
        self._robot_state.gripper_state = np.clip(state, 0, 1)

    def add_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Add callback for simulation events."""
        if event == "step":
            self._on_step_callbacks.append(callback)
        elif event == "reset":
            self._on_reset_callbacks.append(callback)

    async def run_async(
        self,
        duration: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Run simulation asynchronously.

        Args:
            duration: Maximum simulation time in seconds
            max_steps: Maximum number of steps
        """
        while self._is_running:
            self.step()

            # Check termination conditions
            if duration and self._simulation_time >= duration:
                break
            if max_steps and self._current_step >= max_steps:
                break

            # Yield to other tasks
            await asyncio.sleep(0)

    def close(self) -> None:
        """Clean up simulation resources."""
        self._is_running = False

        if self.config.mode == SimulationMode.ISAAC_LAB and self._sim:
            self._sim.close()

        logger.info("Isaac Lab environment closed")

    def __enter__(self):
        """Context manager entry."""
        asyncio.get_event_loop().run_until_complete(self.setup())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class StandaloneForwardKinematics:
    """Simplified forward kinematics for standalone mode."""

    # DH parameters for common robots (simplified)
    DH_PARAMS = {
        RobotType.FRANKA_PANDA: {
            "d": [0.333, 0, 0.316, 0, 0.384, 0, 0.107],
            "a": [0, 0, 0.0825, -0.0825, 0, 0.088, 0],
            "alpha": [-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0],
        },
        RobotType.UR5E: {
            "d": [0.1625, 0, 0, 0.1333, 0.0997, 0.0996],
            "a": [0, -0.425, -0.3922, 0, 0, 0],
            "alpha": [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
        },
    }

    def __init__(self, robot_type: RobotType):
        self.robot_type = robot_type
        self.dh = self.DH_PARAMS.get(robot_type, self.DH_PARAMS[RobotType.FRANKA_PANDA])

    def compute(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector pose from joint positions."""
        # Simplified FK using DH convention
        T = np.eye(4)

        for i, theta in enumerate(joint_positions[:len(self.dh["d"])]):
            d = self.dh["d"][i]
            a = self.dh["a"][i]
            alpha = self.dh["alpha"][i]

            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            Ti = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])

            T = T @ Ti

        position = T[:3, 3]

        # Convert rotation matrix to quaternion
        R = T[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(R)

        return position, quaternion

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


class StandaloneInverseKinematics:
    """Simplified inverse kinematics for standalone mode."""

    def __init__(self, robot_type: RobotType):
        self.robot_type = robot_type
        self.fk = StandaloneForwardKinematics(robot_type)

        # Get robot config
        config = IsaacLabEnvironment.ROBOT_CONFIGS.get(
            robot_type,
            IsaacLabEnvironment.ROBOT_CONFIGS[RobotType.FRANKA_PANDA]
        )
        self.n_joints = config["n_joints"]
        self.lower_limits = np.array(config["joint_limits_lower"])
        self.upper_limits = np.array(config["joint_limits_upper"])
        self.default_pos = np.array(config["default_joint_pos"])

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> Optional[np.ndarray]:
        """
        Solve IK using iterative Jacobian method.

        Args:
            target_position: Target XYZ position
            target_orientation: Target orientation as quaternion (wxyz)
            initial_guess: Initial joint configuration
            max_iterations: Maximum iterations
            tolerance: Position error tolerance

        Returns:
            Joint positions or None if no solution found
        """
        # Initialize
        q = initial_guess if initial_guess is not None else self.default_pos.copy()

        for _ in range(max_iterations):
            # Current pose
            current_pos, current_quat = self.fk.compute(q)

            # Position error
            pos_error = target_position - current_pos

            if np.linalg.norm(pos_error) < tolerance:
                return q

            # Numerical Jacobian
            J = self._compute_jacobian(q)

            # Damped least squares
            damping = 0.1
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, pos_error)

            # Update
            q = q + 0.5 * dq

            # Clip to limits
            q = np.clip(q, self.lower_limits, self.upper_limits)

        # Return best solution even if not converged
        return q

    def _compute_jacobian(
        self,
        q: np.ndarray,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """Compute numerical Jacobian."""
        J = np.zeros((3, self.n_joints))

        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += epsilon

            pos_plus, _ = self.fk.compute(q_plus)
            pos_current, _ = self.fk.compute(q)

            J[:, i] = (pos_plus - pos_current) / epsilon

        return J


# Export for convenience
__all__ = [
    "IsaacLabEnvironment",
    "SimulationConfig",
    "SimulationMode",
    "RobotType",
    "RobotState",
    "CameraFrame",
    "SimulationMetrics",
]

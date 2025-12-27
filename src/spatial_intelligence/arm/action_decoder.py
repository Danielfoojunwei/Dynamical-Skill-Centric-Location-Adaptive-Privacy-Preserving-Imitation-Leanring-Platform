"""
Action Decoder - Embodiment-Specific Action Generation

Converts image-space trajectory traces to robot-specific joint actions.
This enables CROSS-ROBOT TRANSFER by:
1. Taking universal image-space waypoints as input
2. Converting to 3D using camera parameters
3. Solving IK for the specific robot

This addresses MolmoAct Gap #3: "Dynamical's skills are trained per-robot"

Architecture:
    TrajectoryTrace [N, 2] (image coords)
            │
            ▼
    Unproject to 3D (using camera intrinsics + depth)
            │
            ▼
    Transform to Robot Base Frame (using camera extrinsics)
            │
            ▼
    Interpolate to Action Horizon
            │
            ▼
    IK Solver (robot-specific)
            │
            ▼
    Joint Actions [H, DOF]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Local imports
from .trajectory_trace import TrajectoryTrace
from .robot_registry import RobotConfig, RobotRegistry, CameraConfig

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    pin = None


@dataclass
class ActionDecoderConfig:
    """Configuration for action decoder."""
    # Output configuration
    action_horizon: int = 16
    action_frequency_hz: float = 10.0  # Actions per second

    # IK settings
    ik_max_iterations: int = 100
    ik_tolerance: float = 1e-4
    ik_damping: float = 0.1

    # Safety
    enforce_joint_limits: bool = True
    max_joint_velocity: float = 2.0  # rad/s
    smooth_trajectory: bool = True

    # Interpolation
    interpolation_method: str = "cubic"  # "linear", "cubic", "spline"

    # Fallback
    use_cartesian_fallback: bool = True  # If IK fails, output Cartesian


@dataclass
class DecodedActions:
    """Output from action decoder."""
    # Joint-space actions [H, DOF]
    joint_actions: np.ndarray

    # Cartesian-space trajectory [H, 7] (pos + quat)
    cartesian_trajectory: Optional[np.ndarray] = None

    # Gripper actions [H]
    gripper_actions: Optional[np.ndarray] = None

    # Validity mask [H] - True if IK succeeded
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    robot_id: str = ""
    source_trace: Optional[TrajectoryTrace] = None
    decoding_time_ms: float = 0.0

    @property
    def action_horizon(self) -> int:
        return len(self.joint_actions)

    @property
    def dof(self) -> int:
        return self.joint_actions.shape[1] if self.joint_actions.ndim == 2 else 0

    @property
    def success_rate(self) -> float:
        if len(self.valid_mask) == 0:
            return 1.0
        return float(self.valid_mask.mean())


class IKSolver:
    """
    Inverse Kinematics solver for trajectory conversion.

    Supports multiple backends:
    - Pinocchio (preferred)
    - Analytical (for simple robots)
    - Numerical (fallback)
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        config: ActionDecoderConfig,
    ):
        self.robot_config = robot_config
        self.config = config

        # Initialize IK backend
        self._model = None
        self._data = None
        self._ee_frame_id = None

        if HAS_PINOCCHIO and robot_config.urdf_path:
            self._init_pinocchio()
        else:
            logger.warning(
                f"Pinocchio not available for {robot_config.robot_id}, "
                "using numerical IK"
            )

    def _init_pinocchio(self):
        """Initialize Pinocchio model."""
        try:
            self._model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            self._data = self._model.createData()
            self._ee_frame_id = self._model.getFrameId(self.robot_config.ee_link)
            logger.info(f"Loaded Pinocchio model for {self.robot_config.robot_id}")
        except Exception as e:
            logger.warning(f"Failed to load URDF: {e}")

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_config: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Solve IK for a single target pose.

        Args:
            target_position: Target position [3]
            target_orientation: Target orientation as quaternion [4] (optional)
            initial_config: Initial joint configuration [DOF]

        Returns:
            (joint_config, success)
        """
        if initial_config is None:
            initial_config = self.robot_config.home_position

        if self._model is not None:
            return self._solve_pinocchio(
                target_position, target_orientation, initial_config
            )
        else:
            return self._solve_numerical(
                target_position, target_orientation, initial_config
            )

    def _solve_pinocchio(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        initial_config: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Solve IK using Pinocchio."""
        if not HAS_PINOCCHIO:
            return None, False

        # Create target SE3
        if target_orientation is not None:
            # Convert quaternion to rotation matrix
            quat = target_orientation  # [x, y, z, w]
            rot = self._quat_to_rotation(quat)
        else:
            rot = np.eye(3)

        target_se3 = pin.SE3(rot, target_position)

        # Solve IK
        q = initial_config.copy()

        for i in range(self.config.ik_max_iterations):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacement(self._model, self._data, self._ee_frame_id)

            current_se3 = self._data.oMf[self._ee_frame_id]
            error_se3 = target_se3.actInv(current_se3)
            error = pin.log6(error_se3).vector

            if np.linalg.norm(error) < self.config.ik_tolerance:
                # Enforce joint limits
                if self.config.enforce_joint_limits:
                    q = self.robot_config.clip_to_limits(q)
                return q, True

            # Compute Jacobian
            J = pin.computeFrameJacobian(
                self._model, self._data, q,
                self._ee_frame_id, pin.ReferenceFrame.LOCAL
            )

            # Damped least squares
            JtJ = J.T @ J
            damping = self.config.ik_damping * np.eye(self.robot_config.dof)
            dq = np.linalg.solve(JtJ + damping, J.T @ error)

            q = q + dq

        return None, False

    def _solve_numerical(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        initial_config: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Simple numerical IK fallback."""
        # This is a simplified placeholder
        # In production, use a proper IK library

        q = initial_config.copy()

        # Simple gradient descent
        for i in range(self.config.ik_max_iterations):
            # Compute current position (simplified - assumes first 3 joints control position)
            current_pos = self._forward_kinematics_simple(q)

            error = target_position - current_pos
            if np.linalg.norm(error) < self.config.ik_tolerance:
                if self.config.enforce_joint_limits:
                    q = self.robot_config.clip_to_limits(q)
                return q, True

            # Simple Jacobian approximation
            J = np.zeros((3, self.robot_config.dof))
            eps = 1e-6
            for j in range(min(3, self.robot_config.dof)):
                q_plus = q.copy()
                q_plus[j] += eps
                J[:, j] = (self._forward_kinematics_simple(q_plus) - current_pos) / eps

            # Update
            dq = np.linalg.lstsq(J, error, rcond=None)[0]
            q = q + 0.1 * dq

        return None, False

    def _forward_kinematics_simple(self, q: np.ndarray) -> np.ndarray:
        """Simple FK for testing (assumes planar robot)."""
        # This is a placeholder - real implementation uses Pinocchio
        x = np.cos(q[0]) * 0.5 + np.cos(q[0] + q[1]) * 0.4 if len(q) > 1 else np.cos(q[0]) * 0.5
        y = np.sin(q[0]) * 0.5 + np.sin(q[0] + q[1]) * 0.4 if len(q) > 1 else np.sin(q[0]) * 0.5
        z = 0.5 + (q[2] if len(q) > 2 else 0) * 0.1
        return np.array([x, y, z])

    @staticmethod
    def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [x, y, z, w] to rotation matrix."""
        x, y, z, w = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])


class ActionDecoder:
    """
    Convert trajectory traces to robot-specific actions.

    This is the key component for cross-robot transfer:
    - Takes universal image-space waypoints
    - Converts to 3D using camera calibration
    - Solves IK for the target robot
    """

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        robot_id: Optional[str] = None,
        config: Optional[ActionDecoderConfig] = None,
    ):
        """
        Initialize action decoder.

        Args:
            robot_config: Robot configuration (takes precedence)
            robot_id: Robot ID to look up in registry
            config: Decoder configuration
        """
        self.config = config or ActionDecoderConfig()

        # Get robot config
        if robot_config is not None:
            self.robot_config = robot_config
        elif robot_id is not None:
            self.robot_config = RobotRegistry.instance().get_or_raise(robot_id)
        else:
            raise ValueError("Must provide robot_config or robot_id")

        # Initialize IK solver
        self.ik_solver = IKSolver(self.robot_config, self.config)

    def decode(
        self,
        trace: TrajectoryTrace,
        current_joint_positions: Optional[np.ndarray] = None,
        camera_config: Optional[CameraConfig] = None,
        depth_map: Optional[np.ndarray] = None,
        gripper_target: Optional[float] = None,
    ) -> DecodedActions:
        """
        Decode trajectory trace to robot actions.

        Args:
            trace: Image-space trajectory trace
            current_joint_positions: Current robot configuration
            camera_config: Camera configuration (uses primary if None)
            depth_map: Depth image for 3D conversion
            gripper_target: Target gripper position [0, 1]

        Returns:
            DecodedActions with joint-space trajectory
        """
        start_time = time.time()

        # Get camera config
        if camera_config is None:
            camera_config = self.robot_config.primary_camera
        if camera_config is None:
            raise ValueError("No camera configuration available")

        # Get initial configuration
        if current_joint_positions is None:
            current_joint_positions = self.robot_config.home_position

        # Step 1: Convert trace to 3D points in robot frame
        points_3d = self._trace_to_3d(trace, camera_config, depth_map)

        # Step 2: Interpolate to action horizon
        target_positions = self._interpolate_trajectory(points_3d)

        # Step 3: Solve IK for each target
        joint_trajectory, valid_mask, cartesian_traj = self._solve_trajectory_ik(
            target_positions,
            current_joint_positions,
        )

        # Step 4: Apply smoothing
        if self.config.smooth_trajectory:
            joint_trajectory = self._smooth_trajectory(joint_trajectory)

        # Step 5: Enforce velocity limits
        joint_trajectory = self._enforce_velocity_limits(
            joint_trajectory, current_joint_positions
        )

        # Step 6: Generate gripper actions
        gripper_actions = None
        if self.robot_config.has_gripper:
            gripper_actions = self._generate_gripper_actions(trace, gripper_target)

        decoding_time = (time.time() - start_time) * 1000

        return DecodedActions(
            joint_actions=joint_trajectory,
            cartesian_trajectory=cartesian_traj,
            gripper_actions=gripper_actions,
            valid_mask=valid_mask,
            robot_id=self.robot_config.robot_id,
            source_trace=trace,
            decoding_time_ms=decoding_time,
        )

    def _trace_to_3d(
        self,
        trace: TrajectoryTrace,
        camera_config: CameraConfig,
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """Convert trace waypoints to 3D in robot base frame."""
        # Get depths
        if trace.waypoint_depths is not None:
            depths = trace.waypoint_depths
        elif depth_map is not None:
            depths = self._sample_depths(trace.waypoints, depth_map, camera_config)
        else:
            # Use default depth
            depths = np.ones(len(trace.waypoints)) * 0.5
            logger.warning("No depth available, using default 0.5m")

        # Unproject to 3D
        points_3d = []
        for wp, depth in zip(trace.waypoints, depths):
            # Scale waypoint from [0, 256) to image coords
            pixel = wp * np.array([camera_config.width, camera_config.height]) / 256.0
            point = camera_config.unproject_pixel(pixel, depth)
            points_3d.append(point)

        return np.array(points_3d)

    def _sample_depths(
        self,
        waypoints: np.ndarray,
        depth_map: np.ndarray,
        camera_config: CameraConfig,
    ) -> np.ndarray:
        """Sample depth map at waypoint locations."""
        depths = []
        for wp in waypoints:
            # Scale from [0, 256) to image coords
            u = int(wp[0] * camera_config.width / 256.0)
            v = int(wp[1] * camera_config.height / 256.0)

            # Clamp to image bounds
            u = np.clip(u, 0, depth_map.shape[1] - 1)
            v = np.clip(v, 0, depth_map.shape[0] - 1)

            depths.append(depth_map[v, u])

        return np.array(depths)

    def _interpolate_trajectory(
        self,
        points_3d: np.ndarray,
    ) -> np.ndarray:
        """Interpolate 3D points to action horizon."""
        if len(points_3d) == self.config.action_horizon:
            return points_3d

        t_original = np.linspace(0, 1, len(points_3d))
        t_target = np.linspace(0, 1, self.config.action_horizon)

        if self.config.interpolation_method == "linear":
            interpolated = np.zeros((self.config.action_horizon, 3))
            for dim in range(3):
                interpolated[:, dim] = np.interp(t_target, t_original, points_3d[:, dim])
        elif self.config.interpolation_method == "cubic":
            from scipy.interpolate import interp1d
            interpolated = np.zeros((self.config.action_horizon, 3))
            for dim in range(3):
                f = interp1d(t_original, points_3d[:, dim], kind='cubic', fill_value='extrapolate')
                interpolated[:, dim] = f(t_target)
        else:
            # Fallback to linear
            interpolated = np.zeros((self.config.action_horizon, 3))
            for dim in range(3):
                interpolated[:, dim] = np.interp(t_target, t_original, points_3d[:, dim])

        return interpolated

    def _solve_trajectory_ik(
        self,
        target_positions: np.ndarray,
        initial_config: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve IK for entire trajectory."""
        joint_trajectory = np.zeros((self.config.action_horizon, self.robot_config.dof))
        valid_mask = np.zeros(self.config.action_horizon, dtype=bool)
        cartesian_trajectory = np.zeros((self.config.action_horizon, 7))  # pos + quat

        current_config = initial_config.copy()

        for i, target_pos in enumerate(target_positions):
            # Solve IK
            solution, success = self.ik_solver.solve(
                target_pos,
                target_orientation=None,  # Position-only for now
                initial_config=current_config,
            )

            if success and solution is not None:
                joint_trajectory[i] = solution
                valid_mask[i] = True
                current_config = solution
            else:
                # Use previous config
                joint_trajectory[i] = current_config
                valid_mask[i] = False

            # Store Cartesian (even if IK failed)
            cartesian_trajectory[i, :3] = target_pos
            cartesian_trajectory[i, 3:] = [0, 0, 0, 1]  # Identity quaternion

        return joint_trajectory, valid_mask, cartesian_trajectory

    def _smooth_trajectory(
        self,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """Apply smoothing to trajectory."""
        # Simple moving average
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size

        smoothed = np.zeros_like(trajectory)
        for dim in range(trajectory.shape[1]):
            smoothed[:, dim] = np.convolve(
                trajectory[:, dim], kernel, mode='same'
            )

        return smoothed

    def _enforce_velocity_limits(
        self,
        trajectory: np.ndarray,
        initial_config: np.ndarray,
    ) -> np.ndarray:
        """Enforce velocity limits on trajectory."""
        dt = 1.0 / self.config.action_frequency_hz
        max_delta = self.config.max_joint_velocity * dt

        # Start from initial config
        limited = np.zeros_like(trajectory)
        limited[0] = trajectory[0]

        # Clamp velocities
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - limited[i-1]
            delta_clipped = np.clip(delta, -max_delta, max_delta)
            limited[i] = limited[i-1] + delta_clipped

        return limited

    def _generate_gripper_actions(
        self,
        trace: TrajectoryTrace,
        target: Optional[float],
    ) -> np.ndarray:
        """Generate gripper action trajectory."""
        actions = np.zeros(self.config.action_horizon)

        if target is not None:
            # Ramp to target
            actions[:] = np.linspace(0.5, target, self.config.action_horizon)
        else:
            # Infer from trace (close at terminal waypoint)
            terminal_idx = self.config.action_horizon - 1
            actions[:terminal_idx] = 1.0  # Open
            actions[terminal_idx:] = 0.0  # Close

        return actions


# ============================================================================
# Convenience Functions
# ============================================================================

def decode_trajectory_for_robot(
    trace: TrajectoryTrace,
    robot_id: str,
    current_joints: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
) -> DecodedActions:
    """
    Convenience function to decode a trajectory for a specific robot.

    Args:
        trace: Trajectory trace
        robot_id: Target robot ID
        current_joints: Current joint positions
        depth_map: Depth image

    Returns:
        DecodedActions
    """
    decoder = ActionDecoder(robot_id=robot_id)
    return decoder.decode(trace, current_joints, depth_map=depth_map)


def decode_for_all_robots(
    trace: TrajectoryTrace,
    depth_map: Optional[np.ndarray] = None,
) -> Dict[str, DecodedActions]:
    """
    Decode trajectory for all registered robots.

    This demonstrates cross-robot transfer:
    same trajectory trace → different robot actions.

    Args:
        trace: Trajectory trace
        depth_map: Depth image

    Returns:
        Dictionary mapping robot_id to DecodedActions
    """
    results = {}
    registry = RobotRegistry.instance()

    for robot_id in registry.list_robots():
        try:
            decoder = ActionDecoder(robot_id=robot_id)
            results[robot_id] = decoder.decode(trace, depth_map=depth_map)
        except Exception as e:
            logger.warning(f"Failed to decode for {robot_id}: {e}")

    return results

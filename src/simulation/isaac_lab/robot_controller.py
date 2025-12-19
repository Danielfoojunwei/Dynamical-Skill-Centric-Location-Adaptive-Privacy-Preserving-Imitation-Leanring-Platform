"""
Isaac Lab Robot Controller

Provides unified control interface for robots in Isaac Lab simulation,
supporting multiple control modes and teleoperation integration.
"""

import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """Robot control modes."""
    POSITION = "position"        # Joint position control
    VELOCITY = "velocity"        # Joint velocity control
    TORQUE = "torque"           # Joint torque control
    IMPEDANCE = "impedance"     # Impedance control
    CARTESIAN = "cartesian"     # Cartesian space control


@dataclass
class ControllerConfig:
    """Configuration for robot controller."""
    control_mode: ControlMode = ControlMode.POSITION
    control_frequency: float = 2.0  # Hz (500ms cycle for deliberate manipulation)

    # Position control gains
    position_kp: float = 100.0
    position_kd: float = 10.0

    # Velocity control gains
    velocity_kp: float = 10.0

    # Impedance control parameters
    impedance_stiffness: np.ndarray = None  # (6,) for Cartesian
    impedance_damping: np.ndarray = None

    # Safety limits
    max_joint_velocity: float = 1.0  # rad/s
    max_joint_acceleration: float = 5.0  # rad/s^2
    max_ee_velocity: float = 0.5  # m/s
    max_ee_force: float = 50.0  # N

    # Smoothing
    smoothing_alpha: float = 0.1
    action_scale: float = 0.1

    def __post_init__(self):
        if self.impedance_stiffness is None:
            self.impedance_stiffness = np.array([500, 500, 500, 50, 50, 50])
        if self.impedance_damping is None:
            self.impedance_damping = np.array([50, 50, 50, 5, 5, 5])


class IsaacRobotController:
    """
    Robot controller for Isaac Lab simulation.

    Supports:
    - Multiple control modes (position, velocity, torque, impedance)
    - Teleoperation input processing
    - Action chunking from policy outputs
    - Safety monitoring and limits
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        n_joints: int = 7,
    ):
        """
        Initialize robot controller.

        Args:
            config: Controller configuration
            n_joints: Number of robot joints
        """
        self.config = config or ControllerConfig()
        self.n_joints = n_joints

        # State tracking
        self._target_joint_pos = np.zeros(n_joints)
        self._target_joint_vel = np.zeros(n_joints)
        self._target_ee_pos = np.zeros(3)
        self._target_ee_quat = np.array([1, 0, 0, 0])
        self._target_gripper = 1.0  # Open

        # Previous state for smoothing
        self._prev_joint_pos = np.zeros(n_joints)
        self._prev_joint_vel = np.zeros(n_joints)

        # Action chunking buffer
        self._action_buffer: List[np.ndarray] = []
        self._action_index = 0

        # Safety monitoring
        self._safety_violations = []

        logger.info(f"Robot controller initialized with {n_joints} joints")

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set control mode."""
        self.config.control_mode = mode
        logger.info(f"Control mode set to {mode.value}")

    def set_target_joint_positions(
        self,
        positions: np.ndarray,
        gripper: Optional[float] = None,
    ) -> None:
        """
        Set target joint positions.

        Args:
            positions: Target joint positions (n_joints,)
            gripper: Gripper state 0-1
        """
        self._target_joint_pos = np.array(positions[:self.n_joints])
        if gripper is not None:
            self._target_gripper = np.clip(gripper, 0, 1)

    def set_target_ee_pose(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        gripper: Optional[float] = None,
    ) -> None:
        """
        Set target end-effector pose.

        Args:
            position: Target XYZ position (3,)
            orientation: Target quaternion wxyz (4,)
            gripper: Gripper state 0-1
        """
        self._target_ee_pos = np.array(position)
        if orientation is not None:
            self._target_ee_quat = np.array(orientation)
        if gripper is not None:
            self._target_gripper = np.clip(gripper, 0, 1)

    def set_action_chunk(self, actions: np.ndarray) -> None:
        """
        Set action chunk from policy (e.g., ACT, Diffusion Policy).

        Args:
            actions: Action chunk (horizon, action_dim)
        """
        self._action_buffer = list(actions)
        self._action_index = 0
        logger.debug(f"Loaded action chunk with {len(actions)} actions")

    def process_teleop_input(
        self,
        glove_data: Dict[str, Any],
        retarget_fn: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Process teleoperation input from DYGlove.

        Args:
            glove_data: Glove sensor data
            retarget_fn: Optional retargeting function

        Returns:
            Robot action
        """
        # Extract hand pose
        hand_position = np.array(glove_data.get("position", [0, 0, 0]))
        hand_orientation = np.array(glove_data.get("orientation", [1, 0, 0, 0]))
        finger_angles = np.array(glove_data.get("finger_angles", [0] * 21))

        # Apply retargeting if provided
        if retarget_fn:
            robot_action = retarget_fn(hand_position, hand_orientation, finger_angles)
        else:
            # Simple mapping
            robot_action = np.concatenate([
                hand_position * 0.5 + np.array([0.5, 0, 0.3]),  # Offset to workspace
                [1 - np.mean(finger_angles[:5]) / np.pi],  # Gripper from finger curl
            ])

        return robot_action

    def compute_control(
        self,
        current_joint_pos: np.ndarray,
        current_joint_vel: np.ndarray,
        current_ee_pos: Optional[np.ndarray] = None,
        current_ee_quat: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """
        Compute control output based on current mode.

        Args:
            current_joint_pos: Current joint positions
            current_joint_vel: Current joint velocities
            current_ee_pos: Current end-effector position
            current_ee_quat: Current end-effector orientation
            dt: Control timestep

        Returns:
            Dictionary with control outputs
        """
        # Get target from action buffer if available
        if self._action_buffer and self._action_index < len(self._action_buffer):
            action = self._action_buffer[self._action_index]
            self._action_index += 1

            if len(action) >= self.n_joints:
                self._target_joint_pos = action[:self.n_joints]
            if len(action) > self.n_joints:
                self._target_gripper = action[self.n_joints]

        # Compute control based on mode
        if self.config.control_mode == ControlMode.POSITION:
            output = self._compute_position_control(
                current_joint_pos, current_joint_vel, dt
            )
        elif self.config.control_mode == ControlMode.VELOCITY:
            output = self._compute_velocity_control(
                current_joint_pos, current_joint_vel, dt
            )
        elif self.config.control_mode == ControlMode.IMPEDANCE:
            output = self._compute_impedance_control(
                current_joint_pos, current_joint_vel,
                current_ee_pos, current_ee_quat, dt
            )
        else:
            output = self._compute_position_control(
                current_joint_pos, current_joint_vel, dt
            )

        # Apply safety limits
        output = self._apply_safety_limits(output, current_joint_pos, dt)

        # Update previous state
        self._prev_joint_pos = current_joint_pos.copy()
        self._prev_joint_vel = current_joint_vel.copy()

        return output

    def _compute_position_control(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """Compute position control with PD gains."""
        # Apply smoothing
        alpha = self.config.smoothing_alpha
        smoothed_target = (
            (1 - alpha) * self._prev_joint_pos +
            alpha * self._target_joint_pos
        )

        # PD control
        pos_error = smoothed_target - current_pos
        vel_error = -current_vel

        control = (
            self.config.position_kp * pos_error +
            self.config.position_kd * vel_error
        )

        return {
            "joint_positions": smoothed_target,
            "joint_velocities": pos_error / dt,
            "joint_torques": control,
            "gripper": self._target_gripper,
        }

    def _compute_velocity_control(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """Compute velocity control."""
        vel_error = self._target_joint_vel - current_vel
        control = self.config.velocity_kp * vel_error

        target_pos = current_pos + self._target_joint_vel * dt

        return {
            "joint_positions": target_pos,
            "joint_velocities": self._target_joint_vel,
            "joint_torques": control,
            "gripper": self._target_gripper,
        }

    def _compute_impedance_control(
        self,
        current_joint_pos: np.ndarray,
        current_joint_vel: np.ndarray,
        current_ee_pos: Optional[np.ndarray],
        current_ee_quat: Optional[np.ndarray],
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """Compute Cartesian impedance control."""
        if current_ee_pos is None:
            # Fallback to position control
            return self._compute_position_control(
                current_joint_pos, current_joint_vel, dt
            )

        # Cartesian error
        pos_error = self._target_ee_pos - current_ee_pos

        # Compute Cartesian force
        K = self.config.impedance_stiffness[:3]
        D = self.config.impedance_damping[:3]

        force = K * pos_error  # Simplified - no velocity damping

        # For now, return simple position offset
        # Full implementation would use Jacobian transpose
        target_joint_pos = current_joint_pos + pos_error.mean() * 0.1

        return {
            "joint_positions": target_joint_pos,
            "joint_velocities": np.zeros(self.n_joints),
            "joint_torques": np.zeros(self.n_joints),
            "gripper": self._target_gripper,
            "ee_force": force,
        }

    def _apply_safety_limits(
        self,
        output: Dict[str, np.ndarray],
        current_pos: np.ndarray,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """Apply safety limits to control output."""
        # Velocity limit
        if "joint_velocities" in output:
            vel = output["joint_velocities"]
            vel_magnitude = np.abs(vel)
            if np.any(vel_magnitude > self.config.max_joint_velocity):
                scale = self.config.max_joint_velocity / np.max(vel_magnitude)
                output["joint_velocities"] = vel * scale
                self._safety_violations.append("velocity_limit")

        # Position rate limit
        if "joint_positions" in output:
            pos_delta = output["joint_positions"] - current_pos
            max_delta = self.config.max_joint_velocity * dt

            if np.any(np.abs(pos_delta) > max_delta):
                pos_delta = np.clip(pos_delta, -max_delta, max_delta)
                output["joint_positions"] = current_pos + pos_delta

        return output

    def get_metrics(self) -> Dict[str, Any]:
        """Get controller metrics."""
        return {
            "control_mode": self.config.control_mode.value,
            "action_buffer_length": len(self._action_buffer),
            "action_index": self._action_index,
            "safety_violations": len(self._safety_violations),
            "target_gripper": self._target_gripper,
        }

    def reset(self) -> None:
        """Reset controller state."""
        self._target_joint_pos = np.zeros(self.n_joints)
        self._target_joint_vel = np.zeros(self.n_joints)
        self._prev_joint_pos = np.zeros(self.n_joints)
        self._prev_joint_vel = np.zeros(self.n_joints)
        self._action_buffer.clear()
        self._action_index = 0
        self._safety_violations.clear()
        self._target_gripper = 1.0

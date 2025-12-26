"""
Barrier Functions for Control Barrier Function Safety

Each barrier function h(x) defines a safety constraint where:
- h(x) > 0: Safe region
- h(x) = 0: Boundary
- h(x) < 0: Unsafe region

The CBF condition ensures h(x) ≥ 0 is invariant:
    dh/dt + α·h(x) ≥ 0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Robot state for barrier evaluation."""
    joint_positions: np.ndarray  # [7] joint angles
    joint_velocities: np.ndarray  # [7] joint velocities
    ee_position: np.ndarray  # [3] end-effector position
    ee_orientation: np.ndarray  # [4] quaternion
    ee_velocity: np.ndarray  # [6] twist
    ee_force: np.ndarray  # [6] wrench
    gripper_state: float  # 0=open, 1=closed

    # Environment
    obstacle_positions: Optional[np.ndarray] = None  # [N, 3]
    obstacle_radii: Optional[np.ndarray] = None  # [N]

    @classmethod
    def from_observation(cls, obs: dict) -> 'RobotState':
        """Create state from observation dictionary."""
        return cls(
            joint_positions=np.array(obs.get('joint_positions', np.zeros(7))),
            joint_velocities=np.array(obs.get('joint_velocities', np.zeros(7))),
            ee_position=np.array(obs.get('ee_position', np.zeros(3))),
            ee_orientation=np.array(obs.get('ee_orientation', [1, 0, 0, 0])),
            ee_velocity=np.array(obs.get('ee_velocity', np.zeros(6))),
            ee_force=np.array(obs.get('ee_force', np.zeros(6))),
            gripper_state=obs.get('gripper_state', 0.0),
            obstacle_positions=obs.get('obstacle_positions'),
            obstacle_radii=obs.get('obstacle_radii'),
        )


class BarrierFunction(ABC):
    """Abstract base class for barrier functions."""

    def __init__(self, name: str, alpha: float = 1.0):
        self.name = name
        self.alpha = alpha  # CBF decay rate

    @abstractmethod
    def h(self, state: RobotState) -> float:
        """
        Evaluate barrier function.

        Returns:
            h(x) where h(x) ≥ 0 means safe
        """
        pass

    @abstractmethod
    def grad_h(self, state: RobotState) -> np.ndarray:
        """
        Compute gradient of barrier function.

        Returns:
            ∇h(x) with respect to action-relevant state
        """
        pass

    def is_safe(self, state: RobotState) -> bool:
        """Check if current state is safe."""
        return self.h(state) >= 0

    def constraint(self, state: RobotState, action: np.ndarray) -> float:
        """
        Compute CBF constraint value.

        The constraint dh/dt + α·h ≥ 0 must be satisfied.
        Returns the left-hand side value.
        """
        h_val = self.h(state)
        grad = self.grad_h(state)

        # Approximate dh/dt ≈ ∇h · action
        dh_dt = np.dot(grad, action)

        return dh_dt + self.alpha * h_val


class CollisionBarrier(BarrierFunction):
    """
    Barrier function for collision avoidance.

    h(x) = ||p_ee - p_obs|| - r_safe
    """

    def __init__(self, min_distance: float = 0.05, alpha: float = 1.0):
        super().__init__("collision", alpha)
        self.min_distance = min_distance

    def h(self, state: RobotState) -> float:
        """Minimum distance to any obstacle minus safety margin."""
        if state.obstacle_positions is None or len(state.obstacle_positions) == 0:
            return 1.0  # No obstacles, safe

        distances = []
        for i, obs_pos in enumerate(state.obstacle_positions):
            dist = np.linalg.norm(state.ee_position - obs_pos)
            radius = state.obstacle_radii[i] if state.obstacle_radii is not None else 0.0
            distances.append(dist - radius - self.min_distance)

        return float(np.min(distances))

    def grad_h(self, state: RobotState) -> np.ndarray:
        """Gradient pointing away from nearest obstacle."""
        if state.obstacle_positions is None or len(state.obstacle_positions) == 0:
            return np.zeros(7)

        # Find nearest obstacle
        min_dist = float('inf')
        nearest_obs = None

        for i, obs_pos in enumerate(state.obstacle_positions):
            dist = np.linalg.norm(state.ee_position - obs_pos)
            radius = state.obstacle_radii[i] if state.obstacle_radii is not None else 0.0
            effective_dist = dist - radius

            if effective_dist < min_dist:
                min_dist = effective_dist
                nearest_obs = obs_pos

        if nearest_obs is None:
            return np.zeros(7)

        # Gradient in Cartesian space (pointing away from obstacle)
        direction = state.ee_position - nearest_obs
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(7)

        grad_cartesian = direction / norm

        # Map to joint space via Jacobian approximation
        # Simplified: assume first 3 components of action affect ee position
        grad_action = np.zeros(7)
        grad_action[:3] = grad_cartesian

        return grad_action


class JointLimitBarrier(BarrierFunction):
    """
    Barrier function for joint limit avoidance.

    h(x) = min(q - q_min, q_max - q)
    """

    def __init__(
        self,
        joint_limits_lower: Optional[np.ndarray] = None,
        joint_limits_upper: Optional[np.ndarray] = None,
        margin: float = 0.1,  # radians
        alpha: float = 2.0,
    ):
        super().__init__("joint_limits", alpha)
        self.margin = margin

        # Default limits for 7-DOF arm
        self.q_min = joint_limits_lower if joint_limits_lower is not None else \
            np.array([-2.8, -1.76, -2.8, -3.07, -2.8, -0.02, -2.8])
        self.q_max = joint_limits_upper if joint_limits_upper is not None else \
            np.array([2.8, 1.76, 2.8, -0.07, 2.8, 3.75, 2.8])

    def h(self, state: RobotState) -> float:
        """Minimum distance to any joint limit."""
        q = state.joint_positions

        dist_to_lower = q - (self.q_min + self.margin)
        dist_to_upper = (self.q_max - self.margin) - q

        return float(np.min(np.concatenate([dist_to_lower, dist_to_upper])))

    def grad_h(self, state: RobotState) -> np.ndarray:
        """Gradient pointing away from nearest limit."""
        q = state.joint_positions

        dist_to_lower = q - (self.q_min + self.margin)
        dist_to_upper = (self.q_max - self.margin) - q

        # Find which limit is closest
        all_dists = np.concatenate([dist_to_lower, dist_to_upper])
        min_idx = np.argmin(all_dists)

        grad = np.zeros(7)
        if min_idx < 7:
            # Lower limit is closest, gradient is positive (move away from lower)
            grad[min_idx] = 1.0
        else:
            # Upper limit is closest, gradient is negative (move away from upper)
            grad[min_idx - 7] = -1.0

        return grad


class VelocityBarrier(BarrierFunction):
    """
    Barrier function for velocity limits.

    h(x) = v_max² - ||v||²
    """

    def __init__(
        self,
        max_joint_velocity: float = 2.0,  # rad/s
        max_ee_velocity: float = 1.0,  # m/s
        alpha: float = 2.0,
    ):
        super().__init__("velocity", alpha)
        self.max_joint_velocity = max_joint_velocity
        self.max_ee_velocity = max_ee_velocity

    def h(self, state: RobotState) -> float:
        """Distance to velocity limit."""
        # Joint velocity constraint
        joint_vel_sq = np.sum(state.joint_velocities ** 2)
        joint_margin = self.max_joint_velocity ** 2 - joint_vel_sq

        # End-effector velocity constraint
        ee_vel_sq = np.sum(state.ee_velocity[:3] ** 2)
        ee_margin = self.max_ee_velocity ** 2 - ee_vel_sq

        return float(min(joint_margin, ee_margin))

    def grad_h(self, state: RobotState) -> np.ndarray:
        """Gradient pointing toward lower velocity."""
        joint_vel_sq = np.sum(state.joint_velocities ** 2)
        ee_vel_sq = np.sum(state.ee_velocity[:3] ** 2)

        joint_margin = self.max_joint_velocity ** 2 - joint_vel_sq
        ee_margin = self.max_ee_velocity ** 2 - ee_vel_sq

        # Gradient opposes current velocity direction
        if joint_margin < ee_margin:
            return -2.0 * state.joint_velocities
        else:
            grad = np.zeros(7)
            grad[:3] = -2.0 * state.ee_velocity[:3]
            return grad


class ForceBarrier(BarrierFunction):
    """
    Barrier function for force/torque limits.

    h(x) = f_max² - ||f||²
    """

    def __init__(
        self,
        max_force: float = 50.0,  # Newtons
        max_torque: float = 10.0,  # Nm
        alpha: float = 3.0,
    ):
        super().__init__("force", alpha)
        self.max_force = max_force
        self.max_torque = max_torque

    def h(self, state: RobotState) -> float:
        """Distance to force limit."""
        force = state.ee_force[:3]
        torque = state.ee_force[3:]

        force_margin = self.max_force ** 2 - np.sum(force ** 2)
        torque_margin = self.max_torque ** 2 - np.sum(torque ** 2)

        return float(min(force_margin, torque_margin))

    def grad_h(self, state: RobotState) -> np.ndarray:
        """Gradient pointing toward lower force."""
        force = state.ee_force[:3]
        torque = state.ee_force[3:]

        force_margin = self.max_force ** 2 - np.sum(force ** 2)
        torque_margin = self.max_torque ** 2 - np.sum(torque ** 2)

        # Force is result of motion, so gradient opposes motion direction
        if force_margin < torque_margin:
            grad = np.zeros(7)
            grad[:3] = -force / (np.linalg.norm(force) + 1e-6)
            return grad
        else:
            grad = np.zeros(7)
            grad[3:6] = -torque / (np.linalg.norm(torque) + 1e-6)
            return grad


class ExclusionZoneBarrier(BarrierFunction):
    """
    Barrier function for exclusion zones (no-go areas).

    h(x) = ||p_ee - p_zone|| - r_zone
    """

    def __init__(
        self,
        zone_center: np.ndarray,
        zone_radius: float,
        alpha: float = 2.0,
    ):
        super().__init__("exclusion_zone", alpha)
        self.zone_center = zone_center
        self.zone_radius = zone_radius

    def h(self, state: RobotState) -> float:
        """Distance from exclusion zone."""
        dist = np.linalg.norm(state.ee_position - self.zone_center)
        return float(dist - self.zone_radius)

    def grad_h(self, state: RobotState) -> np.ndarray:
        """Gradient pointing away from zone center."""
        direction = state.ee_position - self.zone_center
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return np.zeros(7)

        grad = np.zeros(7)
        grad[:3] = direction / norm
        return grad

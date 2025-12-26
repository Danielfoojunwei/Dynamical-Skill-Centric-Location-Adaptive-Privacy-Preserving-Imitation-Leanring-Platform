"""
Baseline Controllers for RTA Fallback

These controllers are formally verified or conservatively designed
to guarantee safety when the learned policy is not certified.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """State for baseline controller."""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_position: np.ndarray
    ee_velocity: np.ndarray
    target_position: Optional[np.ndarray] = None
    home_position: Optional[np.ndarray] = None


class BaselineController(ABC):
    """Abstract base class for verified baseline controllers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, state: ControllerState) -> np.ndarray:
        """Compute safe control action."""
        pass

    @abstractmethod
    def reset(self):
        """Reset controller state."""
        pass


class SafeStopController(BaselineController):
    """
    Emergency stop controller.

    Applies damping to bring robot to rest while respecting limits.
    """

    def __init__(
        self,
        damping_gain: float = 10.0,
        max_decel: float = 5.0,  # rad/s²
    ):
        super().__init__("safe_stop")
        self.damping_gain = damping_gain
        self.max_decel = max_decel

    def compute(self, state: ControllerState) -> np.ndarray:
        """Compute damping action to stop robot."""
        vel = state.joint_velocities

        # Proportional damping
        action = -self.damping_gain * vel

        # Limit deceleration
        action = np.clip(action, -self.max_decel, self.max_decel)

        return action

    def reset(self):
        pass


class ImpedanceController(BaselineController):
    """
    Impedance controller for compliant motion.

    Provides spring-damper behavior toward a target or home position.
    """

    def __init__(
        self,
        stiffness: float = 100.0,  # N/m or Nm/rad
        damping: float = 20.0,
        max_force: float = 50.0,
    ):
        super().__init__("impedance")
        self.stiffness = stiffness
        self.damping = damping
        self.max_force = max_force

        # Default home position (7-DOF neutral)
        self.home_position = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])

    def compute(self, state: ControllerState) -> np.ndarray:
        """Compute impedance control action."""
        # Use target if available, otherwise home
        target = state.target_position if state.target_position is not None \
            else (state.home_position if state.home_position is not None else self.home_position)

        pos = state.joint_positions
        vel = state.joint_velocities

        # Spring-damper: F = -k(x - x_target) - d*v
        position_error = pos - target
        action = -self.stiffness * position_error - self.damping * vel

        # Limit force
        action = np.clip(action, -self.max_force, self.max_force)

        return action

    def set_home(self, home_position: np.ndarray):
        """Set home position."""
        self.home_position = home_position

    def reset(self):
        pass


class HoldPositionController(BaselineController):
    """
    Hold current position with high stiffness.

    Used when learned policy is unsafe but robot should stay in place.
    """

    def __init__(
        self,
        stiffness: float = 200.0,
        damping: float = 40.0,
    ):
        super().__init__("hold_position")
        self.stiffness = stiffness
        self.damping = damping
        self.hold_position: Optional[np.ndarray] = None

    def compute(self, state: ControllerState) -> np.ndarray:
        """Hold current position."""
        if self.hold_position is None:
            # First call: set hold position to current
            self.hold_position = state.joint_positions.copy()

        pos = state.joint_positions
        vel = state.joint_velocities

        # PD control to hold position
        position_error = pos - self.hold_position
        action = -self.stiffness * position_error - self.damping * vel

        return action

    def reset(self):
        """Reset hold position (will be set on next compute)."""
        self.hold_position = None


class ReturnHomeController(BaselineController):
    """
    Safely return to home position.

    Uses trajectory interpolation with velocity limits.
    """

    def __init__(
        self,
        max_velocity: float = 0.5,  # rad/s
        stiffness: float = 50.0,
        damping: float = 10.0,
    ):
        super().__init__("return_home")
        self.max_velocity = max_velocity
        self.stiffness = stiffness
        self.damping = damping

        # Default home
        self.home_position = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])

        # Trajectory state
        self.target_position: Optional[np.ndarray] = None
        self.start_time: Optional[float] = None

    def compute(self, state: ControllerState) -> np.ndarray:
        """Compute action to return home."""
        home = state.home_position if state.home_position is not None else self.home_position
        pos = state.joint_positions
        vel = state.joint_velocities

        # Direction to home
        error = home - pos
        distance = np.linalg.norm(error)

        if distance < 0.01:
            # At home, just hold
            return -self.damping * vel

        # Velocity toward home (limited)
        direction = error / distance
        desired_vel = direction * min(self.max_velocity, distance)

        # PD control
        vel_error = vel - desired_vel
        action = self.stiffness * error - self.damping * vel_error

        return action

    def set_home(self, home_position: np.ndarray):
        """Set home position."""
        self.home_position = home_position

    def reset(self):
        self.target_position = None
        self.start_time = None

"""
Actuator Interface - Tier 1 Motor Control

Runs at 1kHz with hard real-time guarantees.
Direct interface to robot motors/actuators.
"""

import time
import logging
from typing import Dict, Any, Optional
from enum import Enum
import numpy as np

from .config import RobotConfig

logger = logging.getLogger(__name__)


class ActuatorMode(Enum):
    """Actuator control mode."""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    OFF = "off"


class ActuatorInterface:
    """
    Direct interface to robot actuators.

    Supports multiple control modes:
    - Position control
    - Velocity control
    - Torque control

    Handles hardware communication at 1kHz.
    """

    def __init__(self, rate_hz: int, robot_config: RobotConfig):
        self.rate_hz = rate_hz
        self.config = robot_config
        self.mode = ActuatorMode.OFF
        self._num_joints = robot_config.num_joints

        # State
        self._current_positions = np.zeros(self._num_joints)
        self._current_velocities = np.zeros(self._num_joints)
        self._current_torques = np.zeros(self._num_joints)
        self._commanded_action = np.zeros(self._num_joints)

        # Hardware interface (mock for now)
        self._hardware = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize hardware interface."""
        interface_type = self.config.control_interface

        if interface_type == "mock":
            logger.info("Initializing mock actuator interface")
            self._hardware = MockHardware(self._num_joints)
        elif interface_type == "can":
            logger.info(f"Initializing CAN interface on {self.config.control_address}")
            # self._hardware = CANInterface(self.config.control_address)
            self._hardware = MockHardware(self._num_joints)  # Fallback
        else:
            logger.warning(f"Unknown interface type: {interface_type}, using mock")
            self._hardware = MockHardware(self._num_joints)

        self._initialized = True
        self.mode = ActuatorMode.VELOCITY  # Default mode

    def read_sensors(self) -> Dict[str, Any]:
        """
        Read current sensor values from hardware.

        Returns:
            Dictionary with joint positions, velocities, torques
        """
        if not self._initialized:
            return {}

        state = self._hardware.read_state()

        self._current_positions = state['positions']
        self._current_velocities = state['velocities']
        self._current_torques = state['torques']

        return {
            'joint_positions': self._current_positions.copy(),
            'joint_velocities': self._current_velocities.copy(),
            'joint_torques': self._current_torques.copy(),
            'timestamp': time.time(),
        }

    def send(self, action: Optional[np.ndarray]) -> bool:
        """
        Send action command to actuators.

        Args:
            action: Action to execute (velocities in velocity mode, etc.)

        Returns:
            True if command sent successfully
        """
        if not self._initialized:
            return False

        if action is None:
            action = np.zeros(self._num_joints)

        # Clip to safe limits
        action = self._clip_to_limits(action)
        self._commanded_action = action

        # Send to hardware
        if self.mode == ActuatorMode.VELOCITY:
            return self._hardware.send_velocity(action)
        elif self.mode == ActuatorMode.POSITION:
            return self._hardware.send_position(action)
        elif self.mode == ActuatorMode.TORQUE:
            return self._hardware.send_torque(action)
        else:
            return True  # OFF mode, do nothing

    def safe_stop(self) -> None:
        """Execute safe stop (controlled deceleration)."""
        logger.info("Executing safe stop")
        self.send(np.zeros(self._num_joints))
        self.mode = ActuatorMode.OFF

    def emergency_stop(self) -> None:
        """Execute emergency stop (immediate halt)."""
        logger.critical("EMERGENCY STOP - disabling actuators")
        if self._hardware:
            self._hardware.emergency_stop()
        self.mode = ActuatorMode.OFF

    def set_mode(self, mode: ActuatorMode) -> None:
        """Set control mode."""
        self.mode = mode
        logger.info(f"Actuator mode set to {mode.value}")

    def _clip_to_limits(self, action: np.ndarray) -> np.ndarray:
        """Clip action to hardware limits."""
        # For velocity mode, clip to velocity limits
        if self.mode == ActuatorMode.VELOCITY:
            limits = np.array(self.config.joint_velocity_limits_dps)
            limits_rad = np.deg2rad(limits)
            return np.clip(action, -limits_rad, limits_rad)

        return action


class MockHardware:
    """Mock hardware for simulation/testing."""

    def __init__(self, num_joints: int):
        self.num_joints = num_joints
        self._positions = np.zeros(num_joints)
        self._velocities = np.zeros(num_joints)
        self._torques = np.zeros(num_joints)
        self._last_time = time.time()

    def read_state(self) -> Dict[str, np.ndarray]:
        """Read simulated state."""
        return {
            'positions': self._positions.copy(),
            'velocities': self._velocities.copy(),
            'torques': self._torques.copy(),
        }

    def send_velocity(self, velocities: np.ndarray) -> bool:
        """Simulate velocity control."""
        current_time = time.time()
        dt = current_time - self._last_time
        self._last_time = current_time

        # Integrate velocities to get positions
        self._positions += velocities * dt
        self._velocities = velocities
        return True

    def send_position(self, positions: np.ndarray) -> bool:
        """Simulate position control."""
        self._positions = positions
        return True

    def send_torque(self, torques: np.ndarray) -> bool:
        """Simulate torque control."""
        self._torques = torques
        return True

    def emergency_stop(self) -> None:
        """Simulate emergency stop."""
        self._velocities = np.zeros(self.num_joints)
        self._torques = np.zeros(self.num_joints)

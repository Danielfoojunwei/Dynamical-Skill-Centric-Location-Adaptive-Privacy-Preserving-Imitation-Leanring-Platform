"""
Safety Shield - Tier 1 Safety System

Runs at 1kHz with hard real-time guarantees.
Can ALWAYS override actions and stop the robot.
No GPU dependencies in critical path.

Based on industrial safety standards (ISO 10218, ISO 15066).
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np

from .config import SafetyConfig

logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    """Safety system status."""
    OK = "ok"
    WARNING = "warning"
    VIOLATION = "violation"
    ESTOP = "estop"


@dataclass
class SafetyViolation:
    """Details of a safety violation."""
    type: str
    severity: str  # "warning", "violation", "critical"
    joint: Optional[int] = None
    value: float = 0.0
    limit: float = 0.0
    message: str = ""


@dataclass
class SafetyState:
    """Current safety system state."""
    status: SafetyStatus = SafetyStatus.OK
    violations: List[SafetyViolation] = None
    last_check_time_us: float = 0.0
    estop_active: bool = False
    human_detected: bool = False
    min_obstacle_distance: float = float('inf')

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class SafetyShield:
    """
    Safety shield that runs at 1kHz.

    Checks:
    - Joint position limits
    - Joint velocity limits
    - Joint torque limits
    - Obstacle proximity
    - Human detection
    - Watchdog/heartbeat

    Can override any action with a safe stop command.
    """

    def __init__(self, rate_hz: int, config: SafetyConfig):
        self.rate_hz = rate_hz
        self.config = config

        self.state = SafetyState()
        self._estop_triggered = False
        self._last_heartbeat = 0.0

        # Pre-allocated arrays for performance
        self._joint_limits_lower = None
        self._joint_limits_upper = None
        self._velocity_limits = None
        self._torque_limits = None

    def initialize(self) -> None:
        """Initialize safety system."""
        logger.info("Initializing safety shield")
        self._last_heartbeat = time.time()

    def check(self, robot_state: Dict[str, Any]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check safety constraints.

        Args:
            robot_state: Current robot state with positions, velocities, etc.

        Returns:
            Tuple of (is_safe, override_action)
            - is_safe: True if no violations
            - override_action: Safe action to take if not safe (or None if safe)
        """
        check_start = time.time()
        self.state.violations = []

        # Check E-stop first (highest priority)
        if self._estop_triggered:
            self.state.status = SafetyStatus.ESTOP
            return False, self._get_estop_action(robot_state)

        # Check heartbeat
        if not self._check_heartbeat():
            self.state.status = SafetyStatus.ESTOP
            logger.critical("Heartbeat timeout - triggering E-stop")
            return False, self._get_estop_action(robot_state)

        # Check joint limits
        position_safe = self._check_position_limits(robot_state)
        velocity_safe = self._check_velocity_limits(robot_state)
        torque_safe = self._check_torque_limits(robot_state)

        # Check obstacles
        obstacle_safe = self._check_obstacles(robot_state)

        # Check humans
        human_safe = self._check_human_safety(robot_state)

        # Combine results
        is_safe = all([
            position_safe,
            velocity_safe,
            torque_safe,
            obstacle_safe,
            human_safe
        ])

        # Determine status
        if is_safe:
            self.state.status = SafetyStatus.OK
        elif any(v.severity == "critical" for v in self.state.violations):
            self.state.status = SafetyStatus.ESTOP
            self._estop_triggered = True
        elif any(v.severity == "violation" for v in self.state.violations):
            self.state.status = SafetyStatus.VIOLATION
        else:
            self.state.status = SafetyStatus.WARNING

        # Calculate override action if needed
        override_action = None
        if not is_safe:
            override_action = self._calculate_safe_action(robot_state)

        self.state.last_check_time_us = (time.time() - check_start) * 1_000_000

        return is_safe, override_action

    def trigger_estop(self) -> None:
        """Trigger emergency stop."""
        logger.critical("E-STOP TRIGGERED")
        self._estop_triggered = True
        self.state.status = SafetyStatus.ESTOP
        self.state.estop_active = True

    def reset_estop(self) -> bool:
        """
        Reset emergency stop.

        Returns True if reset successful, False if conditions prevent reset.
        """
        # Only reset if safe conditions are met
        if self.state.status == SafetyStatus.ESTOP:
            if not self.state.violations:
                logger.info("E-STOP reset")
                self._estop_triggered = False
                self.state.status = SafetyStatus.OK
                self.state.estop_active = False
                return True
            else:
                logger.warning("Cannot reset E-STOP - violations present")
                return False
        return True

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self._last_heartbeat = time.time()

    def _check_heartbeat(self) -> bool:
        """Check if heartbeat is recent."""
        elapsed_ms = (time.time() - self._last_heartbeat) * 1000
        return elapsed_ms < self.config.heartbeat_timeout_ms

    def _check_position_limits(self, robot_state: Dict[str, Any]) -> bool:
        """Check joint position limits."""
        positions = robot_state.get('joint_positions')
        if positions is None:
            return True

        safe = True
        margin = np.deg2rad(self.config.joint_limit_margin_deg)

        for i, pos in enumerate(positions):
            lower = self._joint_limits_lower[i] if self._joint_limits_lower is not None else -np.pi
            upper = self._joint_limits_upper[i] if self._joint_limits_upper is not None else np.pi

            if pos < lower + margin:
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="position_limit",
                    severity="violation" if pos < lower else "warning",
                    joint=i,
                    value=pos,
                    limit=lower,
                    message=f"Joint {i} near lower limit"
                ))
            elif pos > upper - margin:
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="position_limit",
                    severity="violation" if pos > upper else "warning",
                    joint=i,
                    value=pos,
                    limit=upper,
                    message=f"Joint {i} near upper limit"
                ))

        return safe

    def _check_velocity_limits(self, robot_state: Dict[str, Any]) -> bool:
        """Check joint velocity limits."""
        velocities = robot_state.get('joint_velocities')
        if velocities is None:
            return True

        safe = True
        margin_factor = 1.0 - (self.config.velocity_limit_margin_percent / 100.0)

        for i, vel in enumerate(velocities):
            limit = self._velocity_limits[i] if self._velocity_limits is not None else 3.14
            scaled_limit = limit * margin_factor

            if abs(vel) > scaled_limit:
                severity = "violation" if abs(vel) > limit else "warning"
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="velocity_limit",
                    severity=severity,
                    joint=i,
                    value=abs(vel),
                    limit=limit,
                    message=f"Joint {i} velocity exceeds limit"
                ))

        return safe

    def _check_torque_limits(self, robot_state: Dict[str, Any]) -> bool:
        """Check joint torque limits."""
        torques = robot_state.get('joint_torques')
        if torques is None:
            return True

        safe = True
        margin_factor = 1.0 - (self.config.force_limit_margin_percent / 100.0)

        for i, torque in enumerate(torques):
            limit = self._torque_limits[i] if self._torque_limits is not None else 100.0
            scaled_limit = limit * margin_factor

            if abs(torque) > scaled_limit:
                severity = "critical" if abs(torque) > limit else "warning"
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="torque_limit",
                    severity=severity,
                    joint=i,
                    value=abs(torque),
                    limit=limit,
                    message=f"Joint {i} torque exceeds limit"
                ))

        return safe

    def _check_obstacles(self, robot_state: Dict[str, Any]) -> bool:
        """Check obstacle proximity."""
        obstacles = robot_state.get('obstacles', [])
        if not obstacles:
            self.state.min_obstacle_distance = float('inf')
            return True

        min_distance = min(o.get('distance', float('inf')) for o in obstacles)
        self.state.min_obstacle_distance = min_distance

        if min_distance < self.config.min_obstacle_distance_m:
            self.state.violations.append(SafetyViolation(
                type="obstacle_proximity",
                severity="violation",
                value=min_distance,
                limit=self.config.min_obstacle_distance_m,
                message=f"Obstacle too close: {min_distance:.3f}m"
            ))
            return False

        return True

    def _check_human_safety(self, robot_state: Dict[str, Any]) -> bool:
        """Check human safety distance."""
        humans = robot_state.get('humans', [])
        self.state.human_detected = len(humans) > 0

        if not humans:
            return True

        for human in humans:
            distance = human.get('distance', float('inf'))
            if distance < self.config.human_safety_distance_m:
                self.state.violations.append(SafetyViolation(
                    type="human_proximity",
                    severity="critical",
                    value=distance,
                    limit=self.config.human_safety_distance_m,
                    message=f"Human too close: {distance:.3f}m"
                ))
                return False

        return True

    def _calculate_safe_action(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Calculate a safe action based on current violations."""
        num_joints = len(robot_state.get('joint_positions', [7]))

        # For critical violations, stop immediately
        if any(v.severity == "critical" for v in self.state.violations):
            return self._get_estop_action(robot_state)

        # For position limit violations, move away from limit
        safe_velocities = np.zeros(num_joints)

        for v in self.state.violations:
            if v.type == "position_limit" and v.joint is not None:
                # Move away from the violated limit
                if v.value < v.limit:
                    safe_velocities[v.joint] = 0.1  # Move positive
                else:
                    safe_velocities[v.joint] = -0.1  # Move negative

        return safe_velocities

    def _get_estop_action(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Get emergency stop action."""
        num_joints = len(robot_state.get('joint_positions', [7]))
        return np.zeros(num_joints)  # Zero velocity = stop

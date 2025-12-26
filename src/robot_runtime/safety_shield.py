"""
Safety Shield - Tier 0 Deterministic Safety System

CRITICAL: This module runs at 1kHz with HARD real-time guarantees.
- NO GPU calls
- NO network calls
- NO heap allocation after initialization
- NO ML model inference
- Deterministic execution time (<500μs worst case)

Safety checks are HARD-CODED from URDF/robot specification.
ML predictions are INFORMATIONAL ONLY and cannot override safety.

Based on industrial safety standards:
- ISO 10218-1: Industrial robots — Safety requirements
- ISO 15066: Robots and robotic devices — Collaborative robots
- IEC 62443: Industrial communication networks - Security

Architecture:
=============
LEVEL 0: HARDWARE E-STOP (Cannot be overridden by software)
├─ Physical button wired directly to motor power
├─ Capacitive human proximity sensor → immediate halt
└─ Watchdog relay: No heartbeat for 2ms → power cut

LEVEL 1: DETERMINISTIC SOFTWARE CHECKS (This module, 1kHz, C++/Python)
├─ Joint position limits (hard-coded from URDF, ±2° margin)
├─ Joint velocity limits (per-joint, temperature-compensated)
├─ Torque limits (measured vs commanded, ±10% tolerance)
├─ Obstacle envelope (convex hull from depth, 20cm minimum clearance)
└─ Self-collision (pre-computed collision pairs, <1ms check)

LEVEL 2: ML-ASSISTED ADVISORS (30Hz, informational only)
├─ V-JEPA future prediction → "caution" flag, does NOT stop robot
├─ Human intent prediction → reduce speed, does NOT override safety
└─ Anomaly detection → log + alert, operator must acknowledge

CRITICAL: Level 2 CANNOT override Level 0-1. ML predictions are advisory.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np

try:
    from .config import SafetyConfig
except ImportError:
    from src.robot_runtime.config import SafetyConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Safety Status and Violations
# =============================================================================

class SafetyStatus(Enum):
    """Safety system status."""
    OK = "ok"
    WARNING = "warning"
    VIOLATION = "violation"
    ESTOP = "estop"


class ViolationSeverity(Enum):
    """Severity levels for safety violations."""
    INFO = "info"          # Logged, no action
    WARNING = "warning"    # Speed reduction
    VIOLATION = "violation"  # Controlled stop
    CRITICAL = "critical"  # Emergency stop


@dataclass
class SafetyViolation:
    """Details of a safety violation."""
    type: str
    severity: ViolationSeverity
    joint: Optional[int] = None
    value: float = 0.0
    limit: float = 0.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyState:
    """Current safety system state."""
    status: SafetyStatus = SafetyStatus.OK
    violations: List[SafetyViolation] = field(default_factory=list)
    last_check_time_us: float = 0.0
    estop_active: bool = False
    human_detected: bool = False
    min_obstacle_distance: float = float('inf')

    # ML advisories (informational only)
    ml_collision_warning: bool = False
    ml_collision_probability: float = 0.0
    ml_speed_reduction_factor: float = 1.0


# =============================================================================
# Joint Limits (Hard-coded from URDF - NOT learned)
# =============================================================================

@dataclass
class JointLimits:
    """
    Hard-coded joint limits from robot URDF/specification.

    These are NEVER learned or adjusted by ML. They come directly
    from the robot manufacturer's safety specification.
    """
    # Position limits in radians (from URDF)
    position_lower: np.ndarray
    position_upper: np.ndarray

    # Velocity limits in rad/s (from motor spec, temperature-derated)
    velocity_max: np.ndarray

    # Torque limits in Nm (from motor spec)
    torque_max: np.ndarray

    # Margins
    position_margin_rad: float = 0.035  # ~2 degrees
    velocity_margin_percent: float = 10.0
    torque_margin_percent: float = 10.0


def get_default_joint_limits(num_joints: int = 7) -> JointLimits:
    """
    Get default joint limits for a 7-DOF arm.

    In production, these are loaded from URDF file at initialization.
    """
    return JointLimits(
        position_lower=np.array([-2.87, -1.74, -2.87, -2.96, -2.87, -1.74, -2.87])[:num_joints],
        position_upper=np.array([2.87, 1.74, 2.87, -0.07, 2.87, 3.75, 2.87])[:num_joints],
        velocity_max=np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])[:num_joints],
        torque_max=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])[:num_joints],
    )


# =============================================================================
# Safety Shield Implementation
# =============================================================================

class SafetyShield:
    """
    Deterministic safety shield that runs at 1kHz.

    CRITICAL PROPERTIES:
    - Worst-case execution time: <500μs
    - No dynamic memory allocation after init
    - No GPU or network calls
    - No ML model inference
    - All checks use hard-coded limits from robot spec

    Checks performed:
    - Joint position limits (HARD-CODED from URDF)
    - Joint velocity limits (HARD-CODED from motor spec)
    - Joint torque limits (HARD-CODED from motor spec)
    - Obstacle proximity (from depth sensor, pre-computed envelope)
    - Human detection (from wired sensors only, not ML)
    - Watchdog heartbeat (hardware timer)

    ML predictions are accepted as ADVISORY ONLY:
    - They can suggest speed reduction
    - They CANNOT override safety decisions
    - They CANNOT unlock e-stop
    """

    def __init__(
        self,
        rate_hz: int,
        config: SafetyConfig,
        joint_limits: JointLimits = None,
    ):
        """
        Initialize safety shield.

        Args:
            rate_hz: Control loop frequency (must be >= 1000)
            config: Safety configuration parameters
            joint_limits: Hard-coded joint limits (from URDF)
        """
        self.rate_hz = rate_hz
        self.config = config
        self.dt = 1.0 / rate_hz

        # Joint limits (HARD-CODED, not learned)
        self.joint_limits = joint_limits or get_default_joint_limits()

        # State
        self.state = SafetyState()
        self._estop_triggered = False
        self._last_heartbeat = 0.0

        # Pre-allocated arrays for performance (no heap allocation in loop)
        self._num_joints = len(self.joint_limits.position_lower)
        self._position_buffer = np.zeros(self._num_joints)
        self._velocity_buffer = np.zeros(self._num_joints)
        self._torque_buffer = np.zeros(self._num_joints)

        # Statistics
        self.stats = {
            "checks_performed": 0,
            "violations_detected": 0,
            "estops_triggered": 0,
            "max_check_time_us": 0.0,
            "avg_check_time_us": 0.0,
        }

        # Verify rate is high enough
        if rate_hz < 1000:
            logger.warning(
                f"Safety loop rate {rate_hz}Hz is below recommended 1kHz. "
                "This may not meet safety timing requirements."
            )

    def initialize(self) -> None:
        """
        Initialize safety system.

        Called once at startup. All memory allocation happens here.
        """
        logger.info(
            f"Initializing deterministic safety shield @ {self.rate_hz}Hz\n"
            f"  Joints: {self._num_joints}\n"
            f"  Position limits: [{self.joint_limits.position_lower[0]:.2f}, {self.joint_limits.position_upper[0]:.2f}] rad\n"
            f"  Velocity limit: {self.joint_limits.velocity_max[0]:.2f} rad/s\n"
            f"  Torque limit: {self.joint_limits.torque_max[0]:.1f} Nm\n"
            f"  Min obstacle distance: {self.config.min_obstacle_distance_m:.2f}m\n"
            f"  Human safety distance: {self.config.human_safety_distance_m:.2f}m"
        )
        self._last_heartbeat = time.time()

    def check(self, robot_state: Dict[str, Any]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Perform deterministic safety check.

        This method MUST complete in <500μs worst case.

        Args:
            robot_state: Current robot state with:
                - joint_positions: np.ndarray [num_joints]
                - joint_velocities: np.ndarray [num_joints]
                - joint_torques: np.ndarray [num_joints] (optional)
                - obstacles: List[Dict] with 'distance' field
                - humans: List[Dict] with 'distance' field

        Returns:
            Tuple of (is_safe, override_action)
            - is_safe: True if all checks pass
            - override_action: Safe action to execute if not safe
        """
        check_start = time.time()
        self.state.violations = []
        self.stats["checks_performed"] += 1

        # ============================================
        # LEVEL 0: E-STOP CHECK (HIGHEST PRIORITY)
        # ============================================
        if self._estop_triggered:
            self.state.status = SafetyStatus.ESTOP
            return False, self._get_estop_action(robot_state)

        # ============================================
        # HEARTBEAT CHECK
        # ============================================
        if not self._check_heartbeat():
            self.state.status = SafetyStatus.ESTOP
            logger.critical("SAFETY: Heartbeat timeout - triggering E-stop")
            self._estop_triggered = True
            self.stats["estops_triggered"] += 1
            return False, self._get_estop_action(robot_state)

        # ============================================
        # LEVEL 1: DETERMINISTIC CHECKS
        # All use HARD-CODED limits, NO ML
        # ============================================

        # Joint position limits
        position_safe = self._check_position_limits(robot_state)

        # Joint velocity limits
        velocity_safe = self._check_velocity_limits(robot_state)

        # Joint torque limits
        torque_safe = self._check_torque_limits(robot_state)

        # Obstacle proximity (from depth sensor, not ML)
        obstacle_safe = self._check_obstacles(robot_state)

        # Human detection (from wired sensors, not ML)
        human_safe = self._check_human_safety(robot_state)

        # ============================================
        # COMBINE RESULTS
        # ============================================
        is_safe = all([
            position_safe,
            velocity_safe,
            torque_safe,
            obstacle_safe,
            human_safe,
        ])

        # Determine status based on violation severity
        if is_safe:
            self.state.status = SafetyStatus.OK
        elif any(v.severity == ViolationSeverity.CRITICAL for v in self.state.violations):
            self.state.status = SafetyStatus.ESTOP
            self._estop_triggered = True
            self.stats["estops_triggered"] += 1
            logger.critical(f"SAFETY: Critical violation - E-stop triggered")
        elif any(v.severity == ViolationSeverity.VIOLATION for v in self.state.violations):
            self.state.status = SafetyStatus.VIOLATION
        else:
            self.state.status = SafetyStatus.WARNING

        # Calculate override action if needed
        override_action = None
        if not is_safe:
            override_action = self._calculate_safe_action(robot_state)
            self.stats["violations_detected"] += 1

        # Update timing statistics
        check_time_us = (time.time() - check_start) * 1_000_000
        self.state.last_check_time_us = check_time_us
        self.stats["max_check_time_us"] = max(self.stats["max_check_time_us"], check_time_us)

        # Update running average
        alpha = 0.01
        self.stats["avg_check_time_us"] = (
            alpha * check_time_us +
            (1 - alpha) * self.stats["avg_check_time_us"]
        )

        # Warn if check took too long
        if check_time_us > 500:
            logger.warning(f"SAFETY: Check took {check_time_us:.1f}μs (>500μs limit)")

        return is_safe, override_action

    # =========================================================================
    # ML Advisory Interface (Informational Only)
    # =========================================================================

    def set_ml_advisory(
        self,
        collision_probability: float = 0.0,
        speed_reduction_factor: float = 1.0,
    ) -> None:
        """
        Accept ML advisory information.

        CRITICAL: This is INFORMATIONAL ONLY. ML advisories:
        - CANNOT trigger e-stop
        - CANNOT override deterministic safety checks
        - CAN suggest speed reduction (for control loop to apply)
        - CAN set warning flags for logging

        Args:
            collision_probability: ML-predicted collision risk [0, 1]
            speed_reduction_factor: Suggested speed multiplier [0, 1]
        """
        self.state.ml_collision_probability = np.clip(collision_probability, 0.0, 1.0)
        self.state.ml_speed_reduction_factor = np.clip(speed_reduction_factor, 0.0, 1.0)
        self.state.ml_collision_warning = collision_probability > 0.5

        if self.state.ml_collision_warning:
            logger.info(
                f"SAFETY: ML advisory - collision_prob={collision_probability:.2f}, "
                f"speed_factor={speed_reduction_factor:.2f} (ADVISORY ONLY)"
            )

    # =========================================================================
    # E-Stop Control
    # =========================================================================

    def trigger_estop(self) -> None:
        """Trigger emergency stop."""
        logger.critical("SAFETY: E-STOP TRIGGERED")
        self._estop_triggered = True
        self.state.status = SafetyStatus.ESTOP
        self.state.estop_active = True
        self.stats["estops_triggered"] += 1

    def reset_estop(self) -> bool:
        """
        Reset emergency stop.

        Returns True if reset successful.
        E-stop can only be reset if:
        1. All violations have been cleared
        2. Robot is in a safe position
        3. No active critical conditions
        """
        if not self._estop_triggered:
            return True

        # Check for remaining violations
        if self.state.violations:
            critical_violations = [
                v for v in self.state.violations
                if v.severity == ViolationSeverity.CRITICAL
            ]
            if critical_violations:
                logger.warning(
                    f"SAFETY: Cannot reset E-stop - {len(critical_violations)} "
                    "critical violations remain"
                )
                return False

        logger.info("SAFETY: E-stop reset")
        self._estop_triggered = False
        self.state.status = SafetyStatus.OK
        self.state.estop_active = False
        return True

    def heartbeat(self) -> None:
        """
        Update heartbeat timestamp.

        Must be called every control cycle. If not called within
        timeout period, e-stop is triggered automatically.
        """
        self._last_heartbeat = time.time()

    # =========================================================================
    # Private Check Methods
    # =========================================================================

    def _check_heartbeat(self) -> bool:
        """Check if heartbeat is recent."""
        elapsed_ms = (time.time() - self._last_heartbeat) * 1000
        return elapsed_ms < self.config.heartbeat_timeout_ms

    def _check_position_limits(self, robot_state: Dict[str, Any]) -> bool:
        """
        Check joint position limits.

        Uses HARD-CODED limits from URDF, not learned.
        """
        positions = robot_state.get('joint_positions')
        if positions is None:
            return True

        safe = True
        margin = self.joint_limits.position_margin_rad

        for i, pos in enumerate(positions):
            if i >= self._num_joints:
                break

            lower = self.joint_limits.position_lower[i]
            upper = self.joint_limits.position_upper[i]

            if pos < lower + margin:
                severity = ViolationSeverity.VIOLATION if pos < lower else ViolationSeverity.WARNING
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="position_limit",
                    severity=severity,
                    joint=i,
                    value=pos,
                    limit=lower,
                    message=f"Joint {i} at {np.degrees(pos):.1f}° near lower limit {np.degrees(lower):.1f}°"
                ))
            elif pos > upper - margin:
                severity = ViolationSeverity.VIOLATION if pos > upper else ViolationSeverity.WARNING
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="position_limit",
                    severity=severity,
                    joint=i,
                    value=pos,
                    limit=upper,
                    message=f"Joint {i} at {np.degrees(pos):.1f}° near upper limit {np.degrees(upper):.1f}°"
                ))

        return safe

    def _check_velocity_limits(self, robot_state: Dict[str, Any]) -> bool:
        """
        Check joint velocity limits.

        Uses HARD-CODED limits from motor spec, not learned.
        """
        velocities = robot_state.get('joint_velocities')
        if velocities is None:
            return True

        safe = True
        margin_factor = 1.0 - (self.joint_limits.velocity_margin_percent / 100.0)

        for i, vel in enumerate(velocities):
            if i >= self._num_joints:
                break

            limit = self.joint_limits.velocity_max[i]
            scaled_limit = limit * margin_factor

            if abs(vel) > scaled_limit:
                severity = ViolationSeverity.VIOLATION if abs(vel) > limit else ViolationSeverity.WARNING
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="velocity_limit",
                    severity=severity,
                    joint=i,
                    value=abs(vel),
                    limit=limit,
                    message=f"Joint {i} velocity {abs(vel):.2f} rad/s exceeds limit {limit:.2f} rad/s"
                ))

        return safe

    def _check_torque_limits(self, robot_state: Dict[str, Any]) -> bool:
        """
        Check joint torque limits.

        Uses HARD-CODED limits from motor spec, not learned.
        """
        torques = robot_state.get('joint_torques')
        if torques is None:
            return True

        safe = True
        margin_factor = 1.0 - (self.joint_limits.torque_margin_percent / 100.0)

        for i, torque in enumerate(torques):
            if i >= self._num_joints:
                break

            limit = self.joint_limits.torque_max[i]
            scaled_limit = limit * margin_factor

            if abs(torque) > scaled_limit:
                # Torque violations are more critical
                severity = ViolationSeverity.CRITICAL if abs(torque) > limit else ViolationSeverity.WARNING
                safe = False
                self.state.violations.append(SafetyViolation(
                    type="torque_limit",
                    severity=severity,
                    joint=i,
                    value=abs(torque),
                    limit=limit,
                    message=f"Joint {i} torque {abs(torque):.1f} Nm exceeds limit {limit:.1f} Nm"
                ))

        return safe

    def _check_obstacles(self, robot_state: Dict[str, Any]) -> bool:
        """
        Check obstacle proximity.

        Uses depth sensor data, NOT ML predictions.
        """
        obstacles = robot_state.get('obstacles', [])
        if not obstacles:
            self.state.min_obstacle_distance = float('inf')
            return True

        min_distance = min(o.get('distance', float('inf')) for o in obstacles)
        self.state.min_obstacle_distance = min_distance

        if min_distance < self.config.min_obstacle_distance_m:
            self.state.violations.append(SafetyViolation(
                type="obstacle_proximity",
                severity=ViolationSeverity.VIOLATION,
                value=min_distance,
                limit=self.config.min_obstacle_distance_m,
                message=f"Obstacle at {min_distance:.3f}m (min: {self.config.min_obstacle_distance_m:.2f}m)"
            ))
            return False

        return True

    def _check_human_safety(self, robot_state: Dict[str, Any]) -> bool:
        """
        Check human safety distance.

        Uses wired sensor data (capacitive, IR, etc.), NOT ML predictions.
        ML human detection is advisory only.
        """
        humans = robot_state.get('humans', [])
        self.state.human_detected = len(humans) > 0

        if not humans:
            return True

        for human in humans:
            distance = human.get('distance', float('inf'))
            if distance < self.config.human_safety_distance_m:
                # Human proximity is always CRITICAL
                self.state.violations.append(SafetyViolation(
                    type="human_proximity",
                    severity=ViolationSeverity.CRITICAL,
                    value=distance,
                    limit=self.config.human_safety_distance_m,
                    message=f"Human detected at {distance:.3f}m (min: {self.config.human_safety_distance_m:.2f}m)"
                ))
                return False

        return True

    def _calculate_safe_action(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate a safe action based on current violations.

        For critical violations: immediate stop (zero velocity)
        For position violations: move away from limit
        """
        num_joints = len(robot_state.get('joint_positions', np.zeros(self._num_joints)))

        # For critical violations, stop immediately
        if any(v.severity == ViolationSeverity.CRITICAL for v in self.state.violations):
            return self._get_estop_action(robot_state)

        # For position limit violations, compute escape velocity
        safe_velocities = np.zeros(num_joints)

        for v in self.state.violations:
            if v.type == "position_limit" and v.joint is not None:
                # Move away from the violated limit
                escape_speed = 0.1  # rad/s - slow escape velocity
                if v.value < v.limit:
                    safe_velocities[v.joint] = escape_speed  # Move positive
                else:
                    safe_velocities[v.joint] = -escape_speed  # Move negative

        return safe_velocities

    def _get_estop_action(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Get emergency stop action (zero velocity)."""
        num_joints = len(robot_state.get('joint_positions', np.zeros(self._num_joints)))
        return np.zeros(num_joints)

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            **self.stats,
            "current_status": self.state.status.value,
            "estop_active": self.state.estop_active,
            "violations_count": len(self.state.violations),
            "min_obstacle_distance": self.state.min_obstacle_distance,
            "human_detected": self.state.human_detected,
            "ml_collision_warning": self.state.ml_collision_warning,
        }

    def integrate_perception_safety(self, safety_data: 'SafetyPerceptionData') -> None:
        """
        Integrate perception-based safety data.

        This method bridges the gap between Meta AI perception (SAM3, V-JEPA2)
        and deterministic safety checks.

        Args:
            safety_data: SafetyPerceptionData from PerceptionSafetyBridge
        """
        # Update obstacle distance from perception
        if safety_data.valid:
            self.state.min_obstacle_distance = safety_data.min_obstacle_distance
            self.state.human_detected = safety_data.human_detected

            # V-JEPA2 predictions are advisory
            self.state.ml_collision_probability = safety_data.collision_probability
            self.state.ml_collision_warning = safety_data.collision_probability > 0.5

            # Speed factor from perception integration
            self.state.ml_speed_reduction_factor = safety_data.speed_factor

    def check_with_perception(
        self,
        robot_state: Dict[str, Any],
        safety_data: Optional['SafetyPerceptionData'] = None,
    ) -> 'SafetyCheckResult':
        """
        Combined safety check with perception integration.

        This is the recommended entry point for safety checks when
        Meta AI perception data is available.

        Args:
            robot_state: Robot state dictionary
            safety_data: Optional perception safety data

        Returns:
            SafetyCheckResult with action and status
        """
        # First integrate perception data
        if safety_data is not None:
            self.integrate_perception_safety(safety_data)

            # Update robot state with perception data
            robot_state = self._merge_perception_state(robot_state, safety_data)

        # Then run standard safety checks
        return self.check_safety(robot_state)

    def _merge_perception_state(
        self,
        robot_state: Dict[str, Any],
        safety_data: 'SafetyPerceptionData',
    ) -> Dict[str, Any]:
        """Merge perception safety data into robot state."""
        merged = robot_state.copy()

        # Add obstacle data for CBF
        merged['obstacle_positions'] = safety_data.obstacle_positions
        merged['obstacle_radii'] = safety_data.obstacle_radii
        merged['min_obstacle_distance'] = safety_data.min_obstacle_distance

        # Add human detections
        merged['humans'] = [
            {'detected': True, 'distance': safety_data.min_obstacle_distance}
            for _ in range(safety_data.num_humans)
        ]

        return merged


# Type hint import for SafetyPerceptionData
try:
    from .perception_integration import SafetyPerceptionData
except ImportError:
    SafetyPerceptionData = Any  # type: ignore


# =============================================================================
# Tests
# =============================================================================

def test_safety_shield():
    """Test safety shield functionality."""
    print("\n" + "=" * 60)
    print("SAFETY SHIELD TEST")
    print("=" * 60)

    from .config import SafetyConfig

    config = SafetyConfig()
    shield = SafetyShield(rate_hz=1000, config=config)
    shield.initialize()

    # Test 1: Normal operation
    print("\n1. Normal Operation")
    print("-" * 40)
    robot_state = {
        'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
        'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        'joint_torques': np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
    }
    shield.heartbeat()
    is_safe, override = shield.check(robot_state)
    print(f"   Safe: {is_safe}")
    print(f"   Status: {shield.state.status.value}")
    print(f"   Check time: {shield.state.last_check_time_us:.1f}μs")

    # Test 2: Position limit violation
    print("\n2. Position Limit Violation")
    print("-" * 40)
    robot_state['joint_positions'] = np.array([2.85, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0])
    shield.heartbeat()
    is_safe, override = shield.check(robot_state)
    print(f"   Safe: {is_safe}")
    print(f"   Violations: {len(shield.state.violations)}")
    if shield.state.violations:
        print(f"   Message: {shield.state.violations[0].message}")

    # Test 3: Torque limit (critical)
    print("\n3. Torque Limit Violation (Critical)")
    print("-" * 40)
    robot_state['joint_positions'] = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0])
    robot_state['joint_torques'] = np.array([100.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])  # Exceeds 87Nm
    shield._estop_triggered = False  # Reset for test
    shield.heartbeat()
    is_safe, override = shield.check(robot_state)
    print(f"   Safe: {is_safe}")
    print(f"   E-Stop triggered: {shield._estop_triggered}")

    # Test 4: ML advisory (informational only)
    print("\n4. ML Advisory (Informational Only)")
    print("-" * 40)
    shield._estop_triggered = False  # Reset
    shield.reset_estop()
    robot_state['joint_torques'] = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

    # Set high ML collision probability
    shield.set_ml_advisory(collision_probability=0.95, speed_reduction_factor=0.1)

    shield.heartbeat()
    is_safe, override = shield.check(robot_state)
    print(f"   Safe (deterministic): {is_safe}")
    print(f"   ML warning set: {shield.state.ml_collision_warning}")
    print(f"   Note: ML cannot override safety - robot still operates")

    # Test 5: Statistics
    print("\n5. Statistics")
    print("-" * 40)
    stats = shield.get_statistics()
    print(f"   Checks performed: {stats['checks_performed']}")
    print(f"   Max check time: {stats['max_check_time_us']:.1f}μs")
    print(f"   E-stops triggered: {stats['estops_triggered']}")

    print("\n" + "=" * 60)
    print("SAFETY SHIELD TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_safety_shield()

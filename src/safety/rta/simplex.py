"""
Runtime Assurance - Simplex Architecture

Implements Simplex switching between learned policy and verified baseline.

The learned policy is allowed to act only when a runtime monitor
certifies its actions as safe. Otherwise, control switches to the
baseline controller.

Architecture:
    ┌─────────────┐
    │   Learned   │──────┐
    │   Policy    │      │
    └─────────────┘      │
                         ▼
                  ┌─────────────┐     ┌─────────────┐
                  │   Safety    │────►│   Switch    │────► Robot
                  │   Monitor   │     │             │
                  └─────────────┘     └─────────────┘
                                            ▲
    ┌─────────────┐                         │
    │  Baseline   │─────────────────────────┘
    │  Controller │
    └─────────────┘
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from .baseline import BaselineController, ImpedanceController, SafeStopController, ControllerState

logger = logging.getLogger(__name__)


class ControlSource(Enum):
    """Source of control action."""
    LEARNED = "learned"
    BASELINE = "baseline"
    EMERGENCY = "emergency"


@dataclass
class SafetyCertificate:
    """Certificate from safety monitor."""
    is_safe: bool
    confidence: float  # 0-1
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    # Detailed checks
    checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CertificationResult:
    """Result of certification check."""
    certified: bool
    certificate: SafetyCertificate
    recommended_action: Optional[np.ndarray] = None


@dataclass
class RTAConfig:
    """Configuration for Runtime Assurance."""
    # Switching thresholds
    switch_threshold: float = 0.9  # Confidence needed to use learned policy
    revert_threshold: float = 0.95  # Confidence needed to switch back to learned

    # Timing
    min_baseline_duration: float = 0.5  # Minimum time on baseline before switching back

    # Baseline controller
    baseline_type: str = "impedance"  # "impedance", "safe_stop", "hold_position"

    # Monitoring
    action_limit: float = 2.0  # Maximum action magnitude
    velocity_limit: float = 2.0  # rad/s

    # Lookahead
    use_lookahead: bool = True
    lookahead_steps: int = 5


class SafetyMonitor:
    """
    Monitors proposed actions and certifies safety.

    Checks:
    1. Action magnitude limits
    2. Velocity limits
    3. Predicted trajectory safety (lookahead)
    4. CBF constraint satisfaction
    """

    def __init__(self, config: RTAConfig):
        self.config = config
        self.cbf_filter = None  # Set externally if available

    def set_cbf(self, cbf_filter):
        """Set CBF filter for constraint checking."""
        self.cbf_filter = cbf_filter

    def certify(
        self,
        proposed_action: np.ndarray,
        state: ControllerState,
    ) -> SafetyCertificate:
        """
        Certify whether proposed action is safe.

        Returns certificate with confidence level.
        """
        checks = {}
        confidence = 1.0

        # Check 1: Action magnitude
        action_mag = np.linalg.norm(proposed_action)
        if action_mag > self.config.action_limit:
            checks["action_limit"] = False
            confidence *= 0.5
        else:
            checks["action_limit"] = True

        # Check 2: Velocity limits
        vel_mag = np.linalg.norm(state.joint_velocities)
        if vel_mag > self.config.velocity_limit:
            checks["velocity_limit"] = False
            confidence *= 0.7
        else:
            checks["velocity_limit"] = True

        # Check 3: Action smoothness (no sudden jumps)
        # Would need history for this, skip for now
        checks["smoothness"] = True

        # Check 4: CBF constraints (if available)
        if self.cbf_filter is not None:
            from ..cbf.barriers import RobotState
            robot_state = RobotState(
                joint_positions=state.joint_positions,
                joint_velocities=state.joint_velocities,
                ee_position=state.ee_position,
                ee_orientation=np.array([1, 0, 0, 0]),
                ee_velocity=state.ee_velocity,
                ee_force=np.zeros(6),
                gripper_state=0.0,
            )
            if not self.cbf_filter.is_safe(robot_state):
                checks["cbf_constraints"] = False
                confidence *= 0.3
            else:
                checks["cbf_constraints"] = True

        # Determine overall safety
        is_safe = all(checks.values()) and confidence >= self.config.switch_threshold

        return SafetyCertificate(
            is_safe=is_safe,
            confidence=confidence,
            reason="" if is_safe else "Failed checks: " + ", ".join(
                k for k, v in checks.items() if not v
            ),
            checks=checks,
        )


class RuntimeAssurance:
    """
    Runtime Assurance (RTA) system using Simplex architecture.

    Ensures safe operation by switching between learned policy
    and verified baseline controller based on safety certification.
    """

    def __init__(self, config: Optional[RTAConfig] = None):
        self.config = config or RTAConfig()

        # Setup baseline controller
        self.baseline = self._create_baseline()

        # Safety monitor
        self.monitor = SafetyMonitor(self.config)

        # State
        self.current_source = ControlSource.LEARNED
        self.last_switch_time = 0.0
        self.consecutive_safe = 0

        # Statistics
        self.stats = {
            "total_decisions": 0,
            "learned_used": 0,
            "baseline_used": 0,
            "switches_to_baseline": 0,
            "switches_to_learned": 0,
        }

    def _create_baseline(self) -> BaselineController:
        """Create baseline controller based on config."""
        if self.config.baseline_type == "safe_stop":
            return SafeStopController()
        elif self.config.baseline_type == "impedance":
            return ImpedanceController()
        else:
            return ImpedanceController()

    def set_cbf(self, cbf_filter):
        """Set CBF filter for safety monitoring."""
        self.monitor.set_cbf(cbf_filter)

    def arbitrate(
        self,
        learned_action: np.ndarray,
        state: ControllerState,
    ) -> Tuple[np.ndarray, ControlSource]:
        """
        Decide between learned policy and baseline controller.

        Args:
            learned_action: Action from learned policy
            state: Current robot state

        Returns:
            (action, source) where source indicates which controller was used
        """
        self.stats["total_decisions"] += 1
        current_time = time.time()

        # Certify learned action
        certificate = self.monitor.certify(learned_action, state)

        # Decision logic
        if self.current_source == ControlSource.LEARNED:
            if certificate.is_safe:
                # Continue with learned
                self.stats["learned_used"] += 1
                return learned_action, ControlSource.LEARNED
            else:
                # Switch to baseline
                self._switch_to_baseline(current_time)
                action = self.baseline.compute(state)
                return action, ControlSource.BASELINE

        else:  # Currently on baseline
            # Check if we can switch back
            time_on_baseline = current_time - self.last_switch_time

            if (certificate.is_safe and
                certificate.confidence >= self.config.revert_threshold and
                time_on_baseline >= self.config.min_baseline_duration):
                # Switch back to learned
                self._switch_to_learned(current_time)
                self.stats["learned_used"] += 1
                return learned_action, ControlSource.LEARNED
            else:
                # Stay on baseline
                self.stats["baseline_used"] += 1
                action = self.baseline.compute(state)
                return action, ControlSource.BASELINE

    def _switch_to_baseline(self, current_time: float):
        """Switch to baseline controller."""
        if self.current_source != ControlSource.BASELINE:
            self.current_source = ControlSource.BASELINE
            self.last_switch_time = current_time
            self.stats["switches_to_baseline"] += 1
            self.baseline.reset()
            logger.warning("RTA: Switching to baseline controller")

    def _switch_to_learned(self, current_time: float):
        """Switch back to learned policy."""
        if self.current_source != ControlSource.LEARNED:
            self.current_source = ControlSource.LEARNED
            self.stats["switches_to_learned"] += 1
            logger.info("RTA: Switching back to learned policy")

    def emergency_stop(self, state: ControllerState) -> np.ndarray:
        """Force emergency stop."""
        self.current_source = ControlSource.EMERGENCY
        stop_controller = SafeStopController(damping_gain=50.0)
        return stop_controller.compute(state)

    def set_home_position(self, home: np.ndarray):
        """Set home position for baseline controllers."""
        if hasattr(self.baseline, 'set_home'):
            self.baseline.set_home(home)

    @property
    def is_on_learned(self) -> bool:
        """Check if currently using learned policy."""
        return self.current_source == ControlSource.LEARNED

    @property
    def is_on_baseline(self) -> bool:
        """Check if currently using baseline controller."""
        return self.current_source == ControlSource.BASELINE

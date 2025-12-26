"""
Safe Policy Executor - Deterministic Safety Integration

Wraps learned policies with deterministic safety guarantees:
1. RTA switching between learned and baseline
2. CBF filtering for hard constraints
3. Runtime monitoring for temporal properties

This is the top-level safety layer that guarantees:
- No constraint violations (CBF)
- Verified fallback on uncertainty (RTA)
- Property monitoring and alerts (Monitor)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .cbf import CBFFilter, CBFConfig, CBFResult
from .cbf.barriers import RobotState
from .rta import RuntimeAssurance, RTAConfig, ControlSource
from .rta.baseline import ControllerState
from .runtime_monitor import RuntimeMonitor, MonitorConfig, MonitorResult

logger = logging.getLogger(__name__)


@dataclass
class SafeExecutorConfig:
    """Configuration for Safe Policy Executor."""
    # Enable/disable components
    use_cbf: bool = True
    use_rta: bool = True
    use_monitor: bool = True

    # CBF config
    cbf_config: Optional[CBFConfig] = None

    # RTA config
    rta_config: Optional[RTAConfig] = None

    # Monitor config
    monitor_config: Optional[MonitorConfig] = None

    # Loop timing
    safety_loop_hz: float = 1000.0
    control_loop_hz: float = 100.0

    @classmethod
    def for_jetson_thor(cls) -> 'SafeExecutorConfig':
        """Configuration optimized for Jetson Thor."""
        return cls(
            use_cbf=True,
            use_rta=True,
            use_monitor=True,
            cbf_config=CBFConfig(),
            rta_config=RTAConfig(),
            safety_loop_hz=1000.0,
        )

    @classmethod
    def minimal(cls) -> 'SafeExecutorConfig':
        """Minimal configuration (CBF only)."""
        return cls(
            use_cbf=True,
            use_rta=False,
            use_monitor=False,
        )


@dataclass
class SafeExecutionResult:
    """Result from safe policy execution."""
    # Action to execute
    action: np.ndarray

    # Source of action
    source: str  # "learned", "baseline", "emergency"
    was_modified: bool  # Action was modified by CBF

    # Component results
    cbf_result: Optional[CBFResult] = None
    rta_source: Optional[ControlSource] = None
    monitor_result: Optional[MonitorResult] = None

    # Timing
    total_time_ms: float = 0.0

    # Safety status
    is_safe: bool = True
    safety_margin: float = 1.0


class SafePolicyExecutor:
    """
    Safe Policy Executor with deterministic guarantees.

    Wraps learned policies (Pi0.5, Diffusion, etc.) with:
    1. CBF safety filter (hard constraint enforcement)
    2. RTA switching (verified fallback)
    3. Runtime monitor (property checking)

    Usage:
        executor = SafePolicyExecutor.for_jetson_thor()

        # In control loop:
        result = executor.execute(
            learned_action=policy.infer(observation),
            state=robot.get_state(),
        )

        robot.send_command(result.action)  # Guaranteed safe
    """

    def __init__(self, config: Optional[SafeExecutorConfig] = None):
        self.config = config or SafeExecutorConfig()

        # Initialize components
        self.cbf: Optional[CBFFilter] = None
        self.rta: Optional[RuntimeAssurance] = None
        self.monitor: Optional[RuntimeMonitor] = None

        self._setup_components()

        # Statistics
        self.stats = {
            "total_executions": 0,
            "cbf_modifications": 0,
            "rta_switches": 0,
            "monitor_violations": 0,
            "avg_time_ms": 0.0,
        }

    def _setup_components(self):
        """Setup safety components based on config."""
        if self.config.use_cbf:
            cbf_config = self.config.cbf_config or CBFConfig()
            self.cbf = CBFFilter(cbf_config)
            logger.info("CBF safety filter initialized")

        if self.config.use_rta:
            rta_config = self.config.rta_config or RTAConfig()
            self.rta = RuntimeAssurance(rta_config)

            # Connect CBF to RTA for certification
            if self.cbf is not None:
                self.rta.set_cbf(self.cbf)

            logger.info("RTA switching initialized")

        if self.config.use_monitor:
            monitor_config = self.config.monitor_config or MonitorConfig()
            self.monitor = RuntimeMonitor(monitor_config)
            self._setup_default_properties()
            logger.info("Runtime monitor initialized")

    def _setup_default_properties(self):
        """Setup default monitoring properties."""
        if self.monitor is None:
            return

        # Always no collision
        self.monitor.add_always_property(
            "no_collision",
            lambda s: getattr(s, 'min_obstacle_distance', 1.0) > 0.03,
        )

        # Velocity limits
        self.monitor.add_always_property(
            "velocity_limit",
            lambda s: np.linalg.norm(getattr(s, 'joint_velocities', [0])) < 2.5,
        )

    def execute(
        self,
        learned_action: np.ndarray,
        state: Dict[str, Any],
    ) -> SafeExecutionResult:
        """
        Execute with full safety stack.

        Args:
            learned_action: Action from learned policy
            state: Robot state dictionary

        Returns:
            SafeExecutionResult with guaranteed safe action
        """
        start_time = time.time()
        self.stats["total_executions"] += 1

        # Convert state formats
        robot_state = RobotState.from_observation(state)
        controller_state = ControllerState(
            joint_positions=robot_state.joint_positions,
            joint_velocities=robot_state.joint_velocities,
            ee_position=robot_state.ee_position,
            ee_velocity=robot_state.ee_velocity[:3] if len(robot_state.ee_velocity) >= 3 else np.zeros(3),
        )

        action = learned_action.copy()
        source = "learned"
        was_modified = False
        cbf_result = None
        rta_source = None
        monitor_result = None

        # Step 1: RTA arbitration (if enabled)
        if self.rta is not None:
            action, rta_source = self.rta.arbitrate(action, controller_state)
            source = rta_source.value

            if rta_source != ControlSource.LEARNED:
                self.stats["rta_switches"] += 1

        # Step 2: CBF filtering (always applied, if enabled)
        if self.cbf is not None:
            cbf_result = self.cbf.filter(action, robot_state)
            action = cbf_result.safe_action
            was_modified = cbf_result.was_modified

            if was_modified:
                self.stats["cbf_modifications"] += 1

        # Step 3: Runtime monitoring (if enabled)
        if self.monitor is not None:
            monitor_result = self.monitor.check(state)

            if monitor_result.any_violation:
                self.stats["monitor_violations"] += 1
                logger.warning(f"Property violations: {monitor_result.violated_properties}")

        # Compute safety margin
        safety_margin = self.cbf.get_safety_margin(robot_state) if self.cbf else 1.0

        # Timing
        total_time = (time.time() - start_time) * 1000
        self.stats["avg_time_ms"] = (
            self.stats["avg_time_ms"] * (self.stats["total_executions"] - 1) + total_time
        ) / self.stats["total_executions"]

        return SafeExecutionResult(
            action=action,
            source=source,
            was_modified=was_modified,
            cbf_result=cbf_result,
            rta_source=rta_source,
            monitor_result=monitor_result,
            total_time_ms=total_time,
            is_safe=safety_margin > 0,
            safety_margin=safety_margin,
        )

    def emergency_stop(self, state: Dict[str, Any]) -> np.ndarray:
        """Force emergency stop."""
        if self.rta is not None:
            controller_state = ControllerState(
                joint_positions=np.array(state.get('joint_positions', np.zeros(7))),
                joint_velocities=np.array(state.get('joint_velocities', np.zeros(7))),
                ee_position=np.array(state.get('ee_position', np.zeros(3))),
                ee_velocity=np.array(state.get('ee_velocity', np.zeros(3))),
            )
            return self.rta.emergency_stop(controller_state)
        else:
            # Return zero action
            return np.zeros(self.config.cbf_config.action_dim if self.config.cbf_config else 7)

    def set_home_position(self, home: np.ndarray):
        """Set home position for baseline controllers."""
        if self.rta is not None:
            self.rta.set_home_position(home)

    def add_exclusion_zone(self, center: np.ndarray, radius: float):
        """Add an exclusion zone to CBF."""
        if self.cbf is not None:
            from .cbf.barriers import ExclusionZoneBarrier
            barrier = ExclusionZoneBarrier(center, radius)
            self.cbf.add_barrier(barrier)

    @classmethod
    def for_jetson_thor(cls) -> 'SafePolicyExecutor':
        """Create executor optimized for Jetson Thor."""
        return cls(SafeExecutorConfig.for_jetson_thor())

    @classmethod
    def minimal(cls) -> 'SafePolicyExecutor':
        """Create minimal executor (CBF only)."""
        return cls(SafeExecutorConfig.minimal())

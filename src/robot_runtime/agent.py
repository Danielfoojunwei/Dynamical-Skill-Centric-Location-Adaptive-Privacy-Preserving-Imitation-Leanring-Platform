"""
Robot Runtime Agent - First-Class On-Robot Component

This agent owns all on-robot execution and can function WITHOUT
cloud connectivity. Everything else talks to it through stable APIs.

Based on real deployment patterns:
- ANYmal: GPU perception at 20Hz, CPU control deterministic
- Spot CORE I/O: Edge compute payload for in-field processing
- Isaac ROS: Modular accelerated perception for ROS 2
"""

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable
from queue import Queue, Empty
import signal

from .config import RobotRuntimeConfig
from .safety_shield import SafetyShield
from .state_estimator import StateEstimator
from .actuator_interface import ActuatorInterface
from .execution_state import ExecutionStateMachine, ExecutionState
from .perception_pipeline import PerceptionPipeline
from .policy_executor import PolicyExecutor
from .skill_cache import SkillCache
from .watchdog import Watchdog

logger = logging.getLogger(__name__)


class RuntimeMode(Enum):
    """Operating modes for the robot runtime."""
    NORMAL = "normal"           # Full capability
    OFFLINE = "offline"         # No cloud connectivity
    DEGRADED = "degraded"       # GPU overloaded, reduced perception
    CPU_ONLY = "cpu_only"       # Emergency mode, GPU failed
    STOPPED = "stopped"         # Safe stop state


@dataclass
class RuntimeTelemetry:
    """Telemetry data for monitoring."""
    tier1_loop_time_us: float = 0.0
    tier1_missed_deadlines: int = 0
    tier2_loop_time_us: float = 0.0
    perception_cascade_level: int = 1
    policy_inference_time_us: float = 0.0
    cloud_connected: bool = False
    mode: RuntimeMode = RuntimeMode.NORMAL
    uptime_seconds: float = 0.0


class RobotRuntimeAgent:
    """
    First-class component that runs ON THE ROBOT.

    Owns:
    - Hard scheduling (Tier 1-2)
    - Actuator interface
    - Skill execution state machine
    - Local feature cache + telemetry
    - Safety authority (can ALWAYS stop the robot)

    Everything else talks to it through stable APIs.
    """

    def __init__(self, config: RobotRuntimeConfig):
        self.config = config
        self.running = False
        self.mode = RuntimeMode.STOPPED
        self.start_time = 0.0

        # Tier 1: CPU-bound, deterministic (1kHz)
        self.safety_shield = SafetyShield(
            rate_hz=config.tier1.safety_rate_hz,
            config=config.safety
        )
        self.state_estimator = StateEstimator(
            rate_hz=config.tier1.state_estimation_rate_hz
        )
        self.actuator_interface = ActuatorInterface(
            rate_hz=config.tier1.actuator_rate_hz,
            robot_config=config.robot
        )
        self.watchdog = Watchdog(
            timeout_ms=config.tier1.watchdog_timeout_ms
        )

        # Tier 2: GPU-accelerated, bounded (10-100Hz)
        self.perception_pipeline = PerceptionPipeline(
            rate_hz=config.tier2.perception_rate_hz,
            config=config.perception
        )
        self.policy_executor = PolicyExecutor(
            rate_hz=config.tier2.policy_rate_hz,
            config=config.policy
        )

        # State machine for skill execution
        self.execution_state = ExecutionStateMachine()

        # Local caches (survive network outages)
        self.skill_cache = SkillCache(
            max_skills=config.tier2.max_cached_skills,
            cache_dir=config.cache_dir
        )

        # Telemetry buffer (async upload when connected)
        self.telemetry_buffer: Queue = Queue(maxsize=10000)
        self.telemetry = RuntimeTelemetry()

        # Threading
        self._tier1_thread: Optional[threading.Thread] = None
        self._tier2_thread: Optional[threading.Thread] = None
        self._telemetry_thread: Optional[threading.Thread] = None

        # Synchronization
        self._action_lock = threading.Lock()
        self._current_action = None
        self._tier2_ready = threading.Event()

        # Callbacks for cloud integration (optional)
        self._cloud_callbacks: Dict[str, Callable] = {}

        logger.info(f"RobotRuntimeAgent initialized with config: {config.robot.robot_id}")

    def start(self) -> bool:
        """Start the robot runtime agent."""
        if self.running:
            logger.warning("Runtime already running")
            return False

        self.running = True
        self.start_time = time.time()
        self.mode = RuntimeMode.NORMAL

        # Initialize components
        if not self._initialize_components():
            self.running = False
            self.mode = RuntimeMode.STOPPED
            return False

        # Start Tier 1 thread (highest priority)
        self._tier1_thread = threading.Thread(
            target=self._tier1_loop,
            name="tier1_control",
            daemon=True
        )

        # Start Tier 2 thread (lower priority)
        self._tier2_thread = threading.Thread(
            target=self._tier2_loop,
            name="tier2_perception",
            daemon=True
        )

        # Start telemetry thread (lowest priority)
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            name="telemetry",
            daemon=True
        )

        # Set thread priorities (if running with appropriate permissions)
        self._set_thread_priorities()

        self._tier1_thread.start()
        self._tier2_thread.start()
        self._telemetry_thread.start()

        logger.info("RobotRuntimeAgent started")
        return True

    def stop(self) -> None:
        """Stop the robot runtime agent safely."""
        logger.info("Stopping RobotRuntimeAgent...")
        self.running = False
        self.mode = RuntimeMode.STOPPED

        # Send safe stop command
        self.actuator_interface.safe_stop()

        # Wait for threads to finish
        if self._tier1_thread:
            self._tier1_thread.join(timeout=1.0)
        if self._tier2_thread:
            self._tier2_thread.join(timeout=1.0)
        if self._telemetry_thread:
            self._telemetry_thread.join(timeout=1.0)

        logger.info("RobotRuntimeAgent stopped")

    def execute_skill(self, skill_id: str, parameters: Dict[str, Any]) -> bool:
        """
        Request execution of a skill.

        The skill must be in the local cache. If not, returns False.
        """
        skill = self.skill_cache.get(skill_id)
        if skill is None:
            logger.warning(f"Skill {skill_id} not in cache")
            return False

        return self.execution_state.start_skill(skill, parameters)

    def emergency_stop(self) -> None:
        """Immediate emergency stop - safety critical."""
        logger.critical("EMERGENCY STOP triggered")
        self.safety_shield.trigger_estop()
        self.actuator_interface.emergency_stop()
        self.mode = RuntimeMode.STOPPED

    def set_cloud_callback(self, event: str, callback: Callable) -> None:
        """Register callback for cloud integration (optional)."""
        self._cloud_callbacks[event] = callback

    def get_telemetry(self) -> RuntimeTelemetry:
        """Get current telemetry snapshot."""
        self.telemetry.uptime_seconds = time.time() - self.start_time
        return self.telemetry

    # =========================================================================
    # TIER 1: 1kHz Deterministic Loop
    # =========================================================================

    def _tier1_loop(self) -> None:
        """
        1kHz deterministic loop - NEVER misses a deadline.

        This is the safety-critical control loop that:
        1. Reads sensor state
        2. Checks safety constraints
        3. Gets action from Tier 2 or fallback
        4. Commands actuators
        5. Pets the watchdog
        """
        period_ns = 1_000_000_000 // self.config.tier1.control_rate_hz  # 1ms at 1kHz
        next_time = time.time_ns()

        logger.info(f"Tier 1 loop started at {self.config.tier1.control_rate_hz} Hz")

        while self.running:
            loop_start = time.time_ns()

            try:
                # 1. Read sensors and update state estimate
                sensor_data = self.actuator_interface.read_sensors()
                state = self.state_estimator.update(sensor_data)

                # 2. Safety check (can override everything)
                safe, override_action = self.safety_shield.check(state)

                # 3. Get action
                if safe:
                    with self._action_lock:
                        action = self._current_action
                    if action is None:
                        action = self.execution_state.get_fallback_action()
                else:
                    action = override_action
                    logger.warning("Safety override active")

                # 4. Command actuators
                self.actuator_interface.send(action)

                # 5. Pet the watchdog
                self.watchdog.pet()

            except Exception as e:
                logger.error(f"Tier 1 loop error: {e}")
                self.actuator_interface.safe_stop()

            # Timing
            loop_end = time.time_ns()
            self.telemetry.tier1_loop_time_us = (loop_end - loop_start) / 1000

            # Check for missed deadline
            if loop_end > next_time + period_ns:
                self.telemetry.tier1_missed_deadlines += 1
                logger.warning(f"Tier 1 missed deadline: {(loop_end - next_time) / 1_000_000:.2f}ms late")

            # Sleep until next period
            next_time += period_ns
            sleep_ns = next_time - time.time_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000)

    # =========================================================================
    # TIER 2: 10-100Hz Perception and Policy Loop
    # =========================================================================

    def _tier2_loop(self) -> None:
        """
        10-100Hz perception and policy loop.

        Runs perception at 30Hz and policy inference at 100Hz.
        Uses cascaded models - heavier models only when needed.
        """
        base_rate_hz = self.config.tier2.policy_rate_hz
        period_ns = 1_000_000_000 // base_rate_hz
        next_time = time.time_ns()
        tick = 0

        perception_divisor = base_rate_hz // self.config.tier2.perception_rate_hz
        planning_divisor = base_rate_hz // self.config.tier2.planning_rate_hz

        logger.info(f"Tier 2 loop started at {base_rate_hz} Hz (perception: {self.config.tier2.perception_rate_hz} Hz)")

        self._tier2_ready.set()

        while self.running:
            loop_start = time.time_ns()

            try:
                # Perception at lower rate (e.g., 30Hz when base is 100Hz)
                if tick % perception_divisor == 0:
                    self.perception_pipeline.update()

                # Get current observation
                observation = self._build_observation()

                # Policy inference at base rate (100Hz)
                inference_start = time.time_ns()
                action = self.policy_executor.infer(observation)
                inference_time = (time.time_ns() - inference_start) / 1000
                self.telemetry.policy_inference_time_us = inference_time

                # Update action for Tier 1
                with self._action_lock:
                    self._current_action = action

                # Local planning at even lower rate (10Hz)
                if tick % planning_divisor == 0:
                    self._local_planning_step()

                tick += 1

            except Exception as e:
                logger.error(f"Tier 2 loop error: {e}")

            # Timing
            loop_end = time.time_ns()
            self.telemetry.tier2_loop_time_us = (loop_end - loop_start) / 1000

            # Sleep until next period
            next_time += period_ns
            sleep_ns = next_time - time.time_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000)

    def _build_observation(self) -> Dict[str, Any]:
        """Build observation for policy from current state and perception."""
        return {
            'state': self.state_estimator.get_state(),
            'perception': self.perception_pipeline.get_features(),
            'skill_context': self.execution_state.get_context(),
        }

    def _local_planning_step(self) -> None:
        """Local planning at 10Hz - reactive obstacle avoidance, trajectory updates."""
        # Get current perception
        perception = self.perception_pipeline.get_features()

        # Check for obstacles
        if perception.get('obstacles'):
            # Update local trajectory to avoid obstacles
            self.execution_state.update_trajectory_avoidance(perception['obstacles'])

    # =========================================================================
    # Telemetry and Cloud Integration
    # =========================================================================

    def _telemetry_loop(self) -> None:
        """Low-priority telemetry collection and upload."""
        while self.running:
            try:
                # Collect telemetry
                telemetry = self.get_telemetry()

                # Buffer for upload
                if not self.telemetry_buffer.full():
                    self.telemetry_buffer.put_nowait(telemetry)

                # Try to upload if cloud callback registered
                if 'upload_telemetry' in self._cloud_callbacks:
                    self._try_upload_telemetry()

            except Exception as e:
                logger.debug(f"Telemetry error: {e}")

            time.sleep(0.1)  # 10Hz telemetry collection

    def _try_upload_telemetry(self) -> None:
        """Attempt to upload buffered telemetry to cloud."""
        batch = []
        while len(batch) < 100:
            try:
                batch.append(self.telemetry_buffer.get_nowait())
            except Empty:
                break

        if batch:
            try:
                self._cloud_callbacks['upload_telemetry'](batch)
                self.telemetry.cloud_connected = True
            except Exception:
                # Re-queue on failure
                for item in batch:
                    if not self.telemetry_buffer.full():
                        self.telemetry_buffer.put_nowait(item)
                self.telemetry.cloud_connected = False

    # =========================================================================
    # Initialization and Configuration
    # =========================================================================

    def _initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            self.safety_shield.initialize()
            self.state_estimator.initialize()
            self.actuator_interface.initialize()
            self.perception_pipeline.initialize()
            self.policy_executor.initialize()
            self.skill_cache.load_cached_skills()
            self.watchdog.start()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _set_thread_priorities(self) -> None:
        """Set thread priorities for real-time scheduling."""
        try:
            # This requires running with appropriate permissions (e.g., CAP_SYS_NICE)
            # On Linux, you'd use SCHED_FIFO or SCHED_RR
            import ctypes
            libc = ctypes.CDLL('libc.so.6', use_errno=True)

            # SCHED_FIFO = 1
            SCHED_FIFO = 1

            class SchedParam(ctypes.Structure):
                _fields_ = [('sched_priority', ctypes.c_int)]

            # Set Tier 1 to highest priority (99)
            if self._tier1_thread and self._tier1_thread.native_id:
                param = SchedParam(99)
                result = libc.sched_setscheduler(
                    self._tier1_thread.native_id,
                    SCHED_FIFO,
                    ctypes.byref(param)
                )
                if result == 0:
                    logger.info("Tier 1 thread set to SCHED_FIFO priority 99")

            # Set Tier 2 to medium priority (50)
            if self._tier2_thread and self._tier2_thread.native_id:
                param = SchedParam(50)
                libc.sched_setscheduler(
                    self._tier2_thread.native_id,
                    SCHED_FIFO,
                    ctypes.byref(param)
                )

        except Exception as e:
            logger.warning(f"Could not set RT priorities (requires elevated permissions): {e}")

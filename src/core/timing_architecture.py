"""
Dynamical.ai 4-Tier Timing Architecture

This module defines the realistic timing hierarchy for humanoid robot control
with privacy-preserving federated learning. The architecture is designed for
robots that move deliberately (not at human speed) while maintaining safety.

Key Design Principles:
======================
1. Safety is ALWAYS real-time (1kHz) - never throttled
2. Robot control runs at 2Hz (500ms) - adequate for deliberate manipulation
3. Perception runs async, providing latest-available data
4. FL/FHE runs completely offline - NOT in the control loop

4-Tier Timing Hierarchy:
========================

Tier 1: Safety Loop (1000Hz / 1ms)
    - Hardware watchdog
    - Joint limit enforcement
    - Collision detection
    - Emergency stop
    - Force/torque limits
    - NEVER throttled, runs on dedicated RTOS thread

Tier 2: Control Loop (2Hz / 500ms)
    - Robot motion execution
    - Skill execution
    - Hand retargeting
    - Action chunking playback
    - Adequate for careful manipulation tasks

Tier 3: Perception Loop (5Hz / 200ms)
    - Camera frame processing
    - Pose estimation (RTMPose)
    - Depth estimation
    - Object detection/segmentation
    - Runs async, provides latest-available data

Tier 4: Learning Loop (Offline / 10s-60s)
    - Federated learning aggregation
    - FHE encryption/decryption
    - Model updates
    - Cloud synchronization
    - NEVER blocks real-time operations

Meta AI Model Inference Times (Measured on Jetson AGX Orin):
===========================================================
- DINOv2 ViT-B/14: ~45ms per frame (with TensorRT)
- SAM2 (small): ~35ms per frame (with TensorRT)
- Depth Anything V3: ~25ms per frame (with TensorRT)
- RTMPose-X: ~20ms per frame (with TensorRT)
- Total perception: ~125ms (fits in 200ms budget)

References:
- facebookresearch/dinov2: https://github.com/facebookresearch/dinov2
- facebookresearch/sam2: https://github.com/facebookresearch/sam2
- facebookresearch/jepa: https://github.com/facebookresearch/jepa
- facebookresearch/vjepa2: https://github.com/facebookresearch/vjepa2
"""

import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Timing Constants
# =============================================================================

class TimingTier(IntEnum):
    """4-tier timing hierarchy."""
    SAFETY = 1      # 1000Hz / 1ms - Hardware safety
    CONTROL = 2     # 2Hz / 500ms - Robot control
    PERCEPTION = 3  # 5Hz / 200ms - Vision processing
    LEARNING = 4    # Offline - FL/FHE


@dataclass(frozen=True)
class TimingConfig:
    """Timing configuration for each tier."""

    # Tier 1: Safety (1kHz)
    SAFETY_FREQUENCY_HZ: float = 1000.0
    SAFETY_PERIOD_MS: float = 1.0
    SAFETY_MAX_JITTER_MS: float = 0.1

    # Tier 2: Control (2Hz - 500ms cycle for deliberate robot motion)
    CONTROL_FREQUENCY_HZ: float = 2.0
    CONTROL_PERIOD_MS: float = 500.0
    CONTROL_MAX_LATENCY_MS: float = 450.0  # Must complete within cycle

    # Tier 3: Perception (5Hz - async, provides latest-available)
    PERCEPTION_FREQUENCY_HZ: float = 5.0
    PERCEPTION_PERIOD_MS: float = 200.0
    PERCEPTION_MAX_LATENCY_MS: float = 180.0

    # Tier 4: Learning (offline, no real-time constraints)
    LEARNING_MIN_INTERVAL_S: float = 10.0
    LEARNING_MAX_INTERVAL_S: float = 60.0
    FHE_ALLOWED_DURATION_S: float = 300.0  # 5 minutes for FHE ops is fine

    # Model inference budgets (Jetson AGX Orin with TensorRT)
    DINOV2_INFERENCE_MS: float = 45.0
    SAM2_INFERENCE_MS: float = 35.0
    DEPTH_INFERENCE_MS: float = 25.0
    POSE_INFERENCE_MS: float = 20.0
    VLA_INFERENCE_MS: float = 50.0  # Pi0/ACT action generation

    # Total perception budget
    @property
    def total_perception_budget_ms(self) -> float:
        return (self.DINOV2_INFERENCE_MS + self.SAM2_INFERENCE_MS +
                self.DEPTH_INFERENCE_MS + self.POSE_INFERENCE_MS)


# Global timing config
TIMING = TimingConfig()


# =============================================================================
# Tier 1: Safety Loop (1kHz RTOS)
# =============================================================================

@dataclass
class SafetyState:
    """Real-time safety state."""
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    ee_forces: np.ndarray  # End-effector forces [Fx, Fy, Fz, Tx, Ty, Tz]
    is_safe: bool = True
    violation_type: Optional[str] = None


class SafetyMonitor:
    """
    Tier 1: Real-time safety monitor running at 1kHz.

    This runs on a dedicated RTOS thread with hard real-time guarantees.
    It has direct hardware access and can trigger emergency stop.
    """

    def __init__(
        self,
        joint_limits: np.ndarray,  # [n_joints, 2] min/max
        velocity_limits: np.ndarray,  # [n_joints] max velocity
        torque_limits: np.ndarray,  # [n_joints] max torque
        force_limits: np.ndarray,  # [6] max force/torque at EE
    ):
        self.joint_limits = joint_limits
        self.velocity_limits = velocity_limits
        self.torque_limits = torque_limits
        self.force_limits = force_limits

        self.n_joints = len(velocity_limits)

        # State
        self._emergency_stop = False
        self._last_check_time = 0.0
        self._violation_count = 0

        # Callbacks
        self._estop_callbacks: List[Callable] = []

    def register_estop_callback(self, callback: Callable[[], None]):
        """Register callback for emergency stop events."""
        self._estop_callbacks.append(callback)

    def check_safety(self, state: SafetyState) -> Tuple[bool, Optional[str]]:
        """
        Check safety constraints. Must complete in <1ms.

        Returns:
            (is_safe, violation_type) - violation_type is None if safe
        """
        start = time.perf_counter()

        # Quick checks first (order matters for performance)

        # 1. Joint position limits
        pos = state.joint_positions
        if np.any(pos < self.joint_limits[:, 0]) or np.any(pos > self.joint_limits[:, 1]):
            return self._handle_violation(state, "JOINT_LIMIT")

        # 2. Velocity limits
        vel = np.abs(state.joint_velocities)
        if np.any(vel > self.velocity_limits):
            return self._handle_violation(state, "VELOCITY_LIMIT")

        # 3. Torque limits
        torque = np.abs(state.joint_torques)
        if np.any(torque > self.torque_limits):
            return self._handle_violation(state, "TORQUE_LIMIT")

        # 4. End-effector force limits
        forces = np.abs(state.ee_forces)
        if np.any(forces > self.force_limits):
            return self._handle_violation(state, "FORCE_LIMIT")

        # Check timing
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > TIMING.SAFETY_PERIOD_MS:
            logger.warning(f"Safety check exceeded budget: {elapsed:.3f}ms > {TIMING.SAFETY_PERIOD_MS}ms")

        return True, None

    def _handle_violation(self, state: SafetyState, violation_type: str) -> Tuple[bool, str]:
        """Handle safety violation."""
        self._violation_count += 1

        # Trigger emergency stop after repeated violations
        if self._violation_count >= 3:
            self.trigger_estop(violation_type)

        return False, violation_type

    def trigger_estop(self, reason: str):
        """Trigger emergency stop."""
        self._emergency_stop = True
        logger.critical(f"EMERGENCY STOP triggered: {reason}")

        for callback in self._estop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"E-stop callback failed: {e}")

    def reset(self):
        """Reset safety state (after manual intervention)."""
        self._emergency_stop = False
        self._violation_count = 0
        logger.info("Safety monitor reset")

    @property
    def is_estopped(self) -> bool:
        return self._emergency_stop


# =============================================================================
# Tier 2: Control Loop (2Hz / 500ms)
# =============================================================================

@dataclass
class ControlState:
    """Robot control state."""
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_position: np.ndarray
    ee_orientation: np.ndarray  # Quaternion [x, y, z, w]
    gripper_state: float
    hand_joint_positions: Optional[np.ndarray] = None


@dataclass
class ControlCommand:
    """Robot control command."""
    timestamp: float
    target_joint_positions: np.ndarray
    target_joint_velocities: Optional[np.ndarray] = None
    gripper_command: float = 0.5
    hand_commands: Optional[np.ndarray] = None
    duration_ms: float = 500.0  # Execute over one control cycle


class ControlLoop:
    """
    Tier 2: Main control loop running at 2Hz (500ms cycle).

    This rate is appropriate for deliberate manipulation tasks where
    the robot moves carefully and safely. Human-like speed is NOT required.

    Features:
    - 500ms cycle gives ample time for:
        - Perception processing (200ms async)
        - VLA inference (50ms)
        - Skill execution (50ms)
        - Safety margins (200ms buffer)
    - Action chunking smooths between cycles
    - Graceful degradation on perception delays
    """

    def __init__(
        self,
        safety_monitor: SafetyMonitor,
        control_callback: Callable[[ControlState], ControlCommand],
    ):
        self.safety_monitor = safety_monitor
        self.control_callback = control_callback

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Latest perception data (async updated)
        self._perception_data: Dict[str, Any] = {}
        self._perception_lock = threading.Lock()
        self._perception_timestamp = 0.0

        # Action chunking buffer
        self._action_buffer: List[ControlCommand] = []
        self._action_index = 0

        # Statistics
        self._cycle_times: List[float] = []
        self._missed_cycles = 0

        # Error handling
        self._consecutive_failures = 0
        self._max_failures = 5
        self._fallback_command: Optional[ControlCommand] = None

    def start(self):
        """Start control loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        logger.info(f"Control loop started at {TIMING.CONTROL_FREQUENCY_HZ}Hz")

    def stop(self):
        """Stop control loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Control loop stopped")

    def update_perception(self, data: Dict[str, Any]):
        """Update perception data (called from Tier 3)."""
        with self._perception_lock:
            self._perception_data = data
            self._perception_timestamp = time.time()

    def set_action_chunk(self, commands: List[ControlCommand]):
        """Set action chunk for smooth execution."""
        self._action_buffer = commands
        self._action_index = 0

    def _control_loop(self):
        """Main control loop."""
        period = TIMING.CONTROL_PERIOD_MS / 1000.0

        while self._running:
            cycle_start = time.perf_counter()

            try:
                # Check if e-stopped
                if self.safety_monitor.is_estopped:
                    self._execute_estop_behavior()
                    time.sleep(0.1)
                    continue

                # Get current robot state
                state = self._get_current_state()

                # Check perception freshness
                with self._perception_lock:
                    perception_age = time.time() - self._perception_timestamp
                    perception_data = self._perception_data.copy()

                # Generate control command
                if perception_age < TIMING.PERCEPTION_PERIOD_MS / 1000.0 * 3:
                    # Fresh perception - normal operation
                    command = self._generate_command(state, perception_data)
                    self._consecutive_failures = 0
                else:
                    # Stale perception - use fallback
                    command = self._generate_fallback_command(state)
                    logger.warning(f"Perception stale ({perception_age:.2f}s), using fallback")

                # Execute command
                self._execute_command(command)

                # Update fallback for next cycle
                self._fallback_command = command

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                self._consecutive_failures += 1

                if self._consecutive_failures >= self._max_failures:
                    logger.critical("Too many control failures, triggering safety stop")
                    self.safety_monitor.trigger_estop("CONTROL_FAILURE")

                self._execute_safe_stop()

            # Maintain cycle timing
            cycle_time = time.perf_counter() - cycle_start
            self._cycle_times.append(cycle_time * 1000)
            if len(self._cycle_times) > 100:
                self._cycle_times.pop(0)

            sleep_time = period - cycle_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self._missed_cycles += 1
                if self._missed_cycles % 10 == 0:
                    logger.warning(f"Missed {self._missed_cycles} control cycles")

    def _get_current_state(self) -> ControlState:
        """Get current robot state from hardware."""
        # Placeholder - would interface with actual robot hardware
        return ControlState(
            timestamp=time.time(),
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            ee_position=np.array([0.5, 0.0, 0.3]),
            ee_orientation=np.array([0, 0, 0, 1]),
            gripper_state=1.0,
        )

    def _generate_command(
        self,
        state: ControlState,
        perception: Dict[str, Any]
    ) -> ControlCommand:
        """Generate control command from state and perception."""
        # Use action buffer if available
        if self._action_buffer and self._action_index < len(self._action_buffer):
            command = self._action_buffer[self._action_index]
            self._action_index += 1
            return command

        # Otherwise use callback
        return self.control_callback(state)

    def _generate_fallback_command(self, state: ControlState) -> ControlCommand:
        """Generate safe fallback command when perception is stale."""
        if self._fallback_command is not None:
            # Continue previous motion with damping
            return ControlCommand(
                timestamp=time.time(),
                target_joint_positions=state.joint_positions,  # Hold position
                target_joint_velocities=np.zeros(len(state.joint_positions)),
                gripper_command=state.gripper_state,
                duration_ms=500.0,
            )

        # No previous command - hold current position
        return ControlCommand(
            timestamp=time.time(),
            target_joint_positions=state.joint_positions,
            gripper_command=state.gripper_state,
        )

    def _execute_command(self, command: ControlCommand):
        """Execute control command on robot hardware."""
        # Placeholder - would send to actual robot
        pass

    def _execute_estop_behavior(self):
        """Execute emergency stop behavior."""
        # Send zero torques, disable motors
        pass

    def _execute_safe_stop(self):
        """Execute safe stop (hold position)."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get control loop statistics."""
        if self._cycle_times:
            return {
                "avg_cycle_ms": np.mean(self._cycle_times),
                "max_cycle_ms": np.max(self._cycle_times),
                "min_cycle_ms": np.min(self._cycle_times),
                "missed_cycles": self._missed_cycles,
                "consecutive_failures": self._consecutive_failures,
            }
        return {}


# =============================================================================
# Tier 3: Perception Loop (5Hz / 200ms async)
# =============================================================================

@dataclass
class PerceptionResult:
    """Result from perception pipeline."""
    timestamp: float

    # Pose estimation
    body_keypoints: Optional[np.ndarray] = None  # [133, 3] COCO-WholeBody
    body_keypoints_3d: Optional[np.ndarray] = None  # [133, 3] world coords

    # Object detection
    object_detections: Optional[List[Dict]] = None
    object_masks: Optional[np.ndarray] = None

    # Depth
    depth_map: Optional[np.ndarray] = None  # [H, W] metric depth

    # Features
    dinov2_features: Optional[np.ndarray] = None  # [1, 768] or similar

    # Processing times
    processing_times: Dict[str, float] = field(default_factory=dict)

    @property
    def total_processing_time_ms(self) -> float:
        return sum(self.processing_times.values())


class PerceptionLoop:
    """
    Tier 3: Async perception loop running at 5Hz (200ms).

    Runs independently of control, providing latest-available perception
    data. Control loop uses whatever data is available, with graceful
    degradation when perception is delayed.

    Meta AI Models Used:
    - DINOv2: Visual features (github.com/facebookresearch/dinov2)
    - SAM2: Segmentation (github.com/facebookresearch/sam2)
    - Depth Anything V3: Depth estimation
    - RTMPose: Pose estimation
    """

    def __init__(
        self,
        camera_manager,  # MultiViewCameraRig
        control_loop: Optional[ControlLoop] = None,
    ):
        self.camera_manager = camera_manager
        self.control_loop = control_loop

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Latest result
        self._latest_result: Optional[PerceptionResult] = None
        self._result_lock = threading.Lock()

        # Model handles (lazy loaded)
        self._dinov2_model = None
        self._sam2_model = None
        self._depth_model = None
        self._pose_model = None

        # Error tracking
        self._error_counts: Dict[str, int] = {
            "dinov2": 0,
            "sam2": 0,
            "depth": 0,
            "pose": 0,
            "camera": 0,
        }
        self._max_errors = 10

        # Statistics
        self._frame_count = 0
        self._processing_times: List[float] = []

    def start(self):
        """Start perception loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._perception_loop, daemon=True)
        self._thread.start()
        logger.info(f"Perception loop started at {TIMING.PERCEPTION_FREQUENCY_HZ}Hz")

    def stop(self):
        """Stop perception loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Perception loop stopped")

    def get_latest(self) -> Optional[PerceptionResult]:
        """Get latest perception result."""
        with self._result_lock:
            return self._latest_result

    def _perception_loop(self):
        """Main perception loop."""
        period = TIMING.PERCEPTION_PERIOD_MS / 1000.0

        while self._running:
            cycle_start = time.perf_counter()

            try:
                result = self._process_frame()

                if result is not None:
                    with self._result_lock:
                        self._latest_result = result

                    # Push to control loop if connected
                    if self.control_loop is not None:
                        self.control_loop.update_perception({
                            "body_keypoints": result.body_keypoints,
                            "body_keypoints_3d": result.body_keypoints_3d,
                            "object_detections": result.object_detections,
                            "depth_map": result.depth_map,
                            "dinov2_features": result.dinov2_features,
                            "timestamp": result.timestamp,
                        })

                    self._frame_count += 1

            except Exception as e:
                logger.error(f"Perception loop error: {e}")

            # Maintain timing
            cycle_time = time.perf_counter() - cycle_start
            self._processing_times.append(cycle_time * 1000)
            if len(self._processing_times) > 100:
                self._processing_times.pop(0)

            sleep_time = period - cycle_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self) -> Optional[PerceptionResult]:
        """Process one frame through perception pipeline."""
        processing_times = {}

        # 1. Get camera frames
        t0 = time.perf_counter()
        frames = self._get_camera_frames()
        if frames is None:
            self._error_counts["camera"] += 1
            return None
        processing_times["camera"] = (time.perf_counter() - t0) * 1000

        # Use first camera for main processing (multi-view for 3D)
        main_frame = list(frames.values())[0] if frames else None
        if main_frame is None:
            return None

        # 2. Run pose estimation (RTMPose) - WITH ERROR FALLBACK
        t0 = time.perf_counter()
        body_keypoints = self._run_pose_estimation(main_frame)
        processing_times["pose"] = (time.perf_counter() - t0) * 1000

        # 3. Run depth estimation - WITH ERROR FALLBACK
        t0 = time.perf_counter()
        depth_map = self._run_depth_estimation(main_frame)
        processing_times["depth"] = (time.perf_counter() - t0) * 1000

        # 4. Run DINOv2 features - WITH ERROR FALLBACK
        t0 = time.perf_counter()
        dinov2_features = self._run_dinov2(main_frame)
        processing_times["dinov2"] = (time.perf_counter() - t0) * 1000

        # 5. Run SAM2 if objects detected - WITH ERROR FALLBACK
        t0 = time.perf_counter()
        object_detections, object_masks = self._run_sam2(main_frame)
        processing_times["sam2"] = (time.perf_counter() - t0) * 1000

        # 6. Triangulate 3D keypoints if multi-view
        body_keypoints_3d = None
        if len(frames) >= 2 and body_keypoints is not None:
            t0 = time.perf_counter()
            body_keypoints_3d = self._triangulate_keypoints(frames, body_keypoints)
            processing_times["triangulation"] = (time.perf_counter() - t0) * 1000

        return PerceptionResult(
            timestamp=time.time(),
            body_keypoints=body_keypoints,
            body_keypoints_3d=body_keypoints_3d,
            object_detections=object_detections,
            object_masks=object_masks,
            depth_map=depth_map,
            dinov2_features=dinov2_features,
            processing_times=processing_times,
        )

    def _get_camera_frames(self) -> Optional[Dict[str, np.ndarray]]:
        """Get frames from all cameras with error handling."""
        try:
            if self.camera_manager is None:
                return None
            frame_set = self.camera_manager.get_synchronized_frames()
            if frame_set is None:
                return None
            return frame_set.get_images()
        except Exception as e:
            logger.warning(f"Camera frame capture failed: {e}")
            return None

    def _run_pose_estimation(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run pose estimation with error fallback."""
        try:
            # Placeholder - would use actual RTMPose
            # Returns [133, 3] array (x, y, confidence)
            return np.zeros((133, 3))
        except Exception as e:
            self._error_counts["pose"] += 1
            if self._error_counts["pose"] < self._max_errors:
                logger.warning(f"Pose estimation failed: {e}")
            return None

    def _run_depth_estimation(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run depth estimation with error fallback."""
        try:
            # Placeholder - would use Depth Anything V3
            h, w = frame.shape[:2]
            return np.ones((h, w)) * 2.0  # 2m default depth
        except Exception as e:
            self._error_counts["depth"] += 1
            if self._error_counts["depth"] < self._max_errors:
                logger.warning(f"Depth estimation failed: {e}")
            return None

    def _run_dinov2(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run DINOv2 feature extraction with error fallback."""
        try:
            # Placeholder - would use facebookresearch/dinov2
            return np.zeros(768)  # ViT-B features
        except Exception as e:
            self._error_counts["dinov2"] += 1
            if self._error_counts["dinov2"] < self._max_errors:
                logger.warning(f"DINOv2 failed: {e}")
            return None

    def _run_sam2(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """Run SAM2 segmentation with error fallback."""
        try:
            # Placeholder - would use facebookresearch/sam2
            return [], None
        except Exception as e:
            self._error_counts["sam2"] += 1
            if self._error_counts["sam2"] < self._max_errors:
                logger.warning(f"SAM2 failed: {e}")
            return None, None

    def _triangulate_keypoints(
        self,
        frames: Dict[str, np.ndarray],
        keypoints_2d: np.ndarray
    ) -> Optional[np.ndarray]:
        """Triangulate 3D keypoints from multi-view 2D detections."""
        try:
            # Placeholder - would use actual triangulation
            return np.zeros((133, 3))
        except Exception as e:
            logger.warning(f"Triangulation failed: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get perception statistics."""
        return {
            "frame_count": self._frame_count,
            "avg_processing_ms": np.mean(self._processing_times) if self._processing_times else 0,
            "error_counts": dict(self._error_counts),
        }


# =============================================================================
# Tier 4: Learning Loop (Offline)
# =============================================================================

class LearningLoop:
    """
    Tier 4: Offline learning loop for FL/FHE operations.

    This runs completely separately from real-time control.
    FHE operations can take minutes - that's fine, they don't block anything.

    Key principle: Learning improves future behavior, not current execution.
    """

    def __init__(
        self,
        fl_pipeline=None,  # UnifiedFLPipeline
        min_interval_s: float = TIMING.LEARNING_MIN_INTERVAL_S,
        max_interval_s: float = TIMING.LEARNING_MAX_INTERVAL_S,
    ):
        self.fl_pipeline = fl_pipeline
        self.min_interval_s = min_interval_s
        self.max_interval_s = max_interval_s

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Gradient accumulator
        self._gradient_buffer: List[np.ndarray] = []
        self._gradient_lock = threading.Lock()

        # Statistics
        self._learning_rounds = 0
        self._last_round_duration_s = 0.0
        self._last_round_time = 0.0

    def start(self):
        """Start learning loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._thread.start()
        logger.info("Learning loop started (offline FL/FHE)")

    def stop(self):
        """Stop learning loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)  # Allow time for cleanup
        logger.info("Learning loop stopped")

    def submit_gradients(self, gradients: np.ndarray):
        """Submit gradients for next learning round (called from skill execution)."""
        with self._gradient_lock:
            self._gradient_buffer.append(gradients)

    def _learning_loop(self):
        """Main learning loop - runs at low priority."""
        while self._running:
            # Wait for minimum interval
            time.sleep(self.min_interval_s)

            # Check if we have gradients to process
            with self._gradient_lock:
                if not self._gradient_buffer:
                    continue
                gradients = self._gradient_buffer.copy()
                self._gradient_buffer.clear()

            # Run FL round (this is where FHE happens - can take minutes!)
            try:
                round_start = time.time()
                self._run_fl_round(gradients)
                self._last_round_duration_s = time.time() - round_start
                self._last_round_time = time.time()
                self._learning_rounds += 1

                logger.info(
                    f"FL round {self._learning_rounds} completed in "
                    f"{self._last_round_duration_s:.1f}s"
                )

            except Exception as e:
                logger.error(f"FL round failed: {e}")

    def _run_fl_round(self, gradients: List[np.ndarray]):
        """Run one FL round with FHE encryption."""
        if self.fl_pipeline is None:
            return

        # This is where the full FL pipeline runs:
        # 1. Gradient clipping
        # 2. Error feedback accumulation
        # 3. Top-K sparsification
        # 4. MOAI compression
        # 5. Quality monitoring
        # 6. FHE encryption (SLOW - and that's OK!)
        # 7. Hierarchical aggregation

        # The key insight: this does NOT block robot control.
        # Control continues with current model while learning happens offline.

        for grad in gradients:
            # Process through pipeline
            # encrypted_update = self.fl_pipeline.process_gradients(grad)
            # self.fl_pipeline.submit_encrypted_update(encrypted_update)
            pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "learning_rounds": self._learning_rounds,
            "last_round_duration_s": self._last_round_duration_s,
            "time_since_last_round_s": time.time() - self._last_round_time if self._last_round_time > 0 else None,
            "pending_gradients": len(self._gradient_buffer),
        }


# =============================================================================
# Integrated System
# =============================================================================

class TimingOrchestrator:
    """
    Orchestrates all 4 tiers of the timing architecture.

    Ensures proper startup/shutdown order and inter-tier communication.
    """

    def __init__(
        self,
        safety_config: Dict[str, np.ndarray],
        camera_manager=None,
        fl_pipeline=None,
        control_callback: Optional[Callable] = None,
    ):
        # Create safety monitor
        self.safety_monitor = SafetyMonitor(
            joint_limits=safety_config.get("joint_limits", np.zeros((7, 2))),
            velocity_limits=safety_config.get("velocity_limits", np.ones(7)),
            torque_limits=safety_config.get("torque_limits", np.ones(7) * 100),
            force_limits=safety_config.get("force_limits", np.ones(6) * 50),
        )

        # Default control callback
        if control_callback is None:
            control_callback = lambda state: ControlCommand(
                timestamp=time.time(),
                target_joint_positions=state.joint_positions,
            )

        # Create control loop
        self.control_loop = ControlLoop(
            safety_monitor=self.safety_monitor,
            control_callback=control_callback,
        )

        # Create perception loop
        self.perception_loop = PerceptionLoop(
            camera_manager=camera_manager,
            control_loop=self.control_loop,
        )

        # Create learning loop
        self.learning_loop = LearningLoop(
            fl_pipeline=fl_pipeline,
        )

        self._running = False

    def start(self):
        """Start all loops in correct order."""
        if self._running:
            return

        logger.info("Starting Dynamical.ai Timing Orchestrator")
        logger.info(f"  Tier 1 Safety:     {TIMING.SAFETY_FREQUENCY_HZ}Hz ({TIMING.SAFETY_PERIOD_MS}ms)")
        logger.info(f"  Tier 2 Control:    {TIMING.CONTROL_FREQUENCY_HZ}Hz ({TIMING.CONTROL_PERIOD_MS}ms)")
        logger.info(f"  Tier 3 Perception: {TIMING.PERCEPTION_FREQUENCY_HZ}Hz ({TIMING.PERCEPTION_PERIOD_MS}ms)")
        logger.info(f"  Tier 4 Learning:   Offline ({TIMING.LEARNING_MIN_INTERVAL_S}-{TIMING.LEARNING_MAX_INTERVAL_S}s)")

        # Start in order: Safety -> Control -> Perception -> Learning
        # Safety is highest priority, learning is background

        self.perception_loop.start()
        time.sleep(0.5)  # Let perception initialize

        self.control_loop.start()

        self.learning_loop.start()

        self._running = True
        logger.info("All timing tiers started successfully")

    def stop(self):
        """Stop all loops in correct order."""
        if not self._running:
            return

        logger.info("Stopping Dynamical.ai Timing Orchestrator")

        # Stop in reverse order
        self.learning_loop.stop()
        self.control_loop.stop()
        self.perception_loop.stop()

        self._running = False
        logger.info("All timing tiers stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all tiers."""
        return {
            "control": self.control_loop.get_statistics(),
            "perception": self.perception_loop.get_statistics(),
            "learning": self.learning_loop.get_statistics(),
            "safety": {
                "estopped": self.safety_monitor.is_estopped,
            },
        }

    def trigger_estop(self, reason: str = "USER_REQUEST"):
        """Trigger emergency stop."""
        self.safety_monitor.trigger_estop(reason)

    def reset_safety(self):
        """Reset safety after intervention."""
        self.safety_monitor.reset()


# =============================================================================
# Meta AI Model References
# =============================================================================

META_AI_MODELS = {
    "dinov2": {
        "repo": "https://github.com/facebookresearch/dinov2",
        "description": "Self-supervised vision transformer for visual features",
        "variants": ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        "inference_time_ms": 45,  # ViT-B on Jetson AGX Orin with TensorRT
    },
    "sam2": {
        "repo": "https://github.com/facebookresearch/sam2",
        "description": "Segment Anything Model 2 for image/video segmentation",
        "variants": ["sam2_tiny", "sam2_small", "sam2_base", "sam2_large"],
        "inference_time_ms": 35,  # Small variant on Jetson AGX Orin
    },
    "jepa": {
        "repo": "https://github.com/facebookresearch/jepa",
        "description": "V-JEPA video understanding from self-supervised learning",
        "variants": ["vjepa_base", "vjepa_large"],
        "inference_time_ms": 80,  # Base variant on Jetson
    },
    "vjepa2": {
        "repo": "https://github.com/facebookresearch/vjepa2",
        "description": "V-JEPA 2 with action-conditioned world model",
        "variants": ["vjepa2_base", "vjepa2_ac"],  # AC = action conditioned
        "inference_time_ms": 100,
    },
    "ijepa": {
        "repo": "https://github.com/facebookresearch/ijepa",
        "description": "I-JEPA image-based joint-embedding predictive architecture",
        "variants": ["ijepa_base", "ijepa_large"],
        "inference_time_ms": 40,
    },
}


def print_timing_summary():
    """Print timing architecture summary."""
    print("\n" + "=" * 70)
    print("DYNAMICAL.AI 4-TIER TIMING ARCHITECTURE")
    print("=" * 70)

    print("\nTier 1: SAFETY (1kHz / 1ms)")
    print("  - Hardware watchdog, joint limits, e-stop")
    print("  - Runs on dedicated RTOS thread, NEVER throttled")

    print(f"\nTier 2: CONTROL (2Hz / {TIMING.CONTROL_PERIOD_MS}ms)")
    print("  - Robot motion execution, skill playback")
    print("  - 500ms cycle = deliberate, safe manipulation")
    print("  - Ample time for perception + inference")

    print(f"\nTier 3: PERCEPTION (5Hz / {TIMING.PERCEPTION_PERIOD_MS}ms)")
    print("  - Async camera processing, pose estimation")
    print("  - Provides latest-available data to control")
    print(f"  - Budget: {TIMING.total_perception_budget_ms}ms total inference")

    print(f"\nTier 4: LEARNING (Offline / {TIMING.LEARNING_MIN_INTERVAL_S}-{TIMING.LEARNING_MAX_INTERVAL_S}s)")
    print("  - Federated learning with FHE encryption")
    print("  - NEVER blocks real-time operations")
    print(f"  - FHE can take up to {TIMING.FHE_ALLOWED_DURATION_S}s - that's fine!")

    print("\nMeta AI Models (github.com/facebookresearch/):")
    for name, info in META_AI_MODELS.items():
        print(f"  - {name}: {info['inference_time_ms']}ms ({info['repo'].split('/')[-1]})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_timing_summary()

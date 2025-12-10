"""
DYGlove Calibration Module

This module provides calibration routines for the DYGlove haptic glove:
- Encoder calibration (zero position, range)
- Force feedback motor calibration
- Finger tracking calibration
- Hand-camera registration

Calibration Process:
====================

1. ZERO POSITION: User holds hand flat, all joints at zero
2. FULL FIST: User makes a fist, captures max flexion
3. PINCH GRIP: Thumb-index pinch for abduction calibration
4. SPREAD FINGERS: Max abduction for all fingers
5. HAPTIC TEST: Test force feedback motors

The calibration data is stored and used to map raw encoder
values to calibrated joint angles.
"""

import os
import time
import json
import logging
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Import glove driver
try:
    from src.drivers.dyglove import (
        DYGloveDriver, DYGloveState, DYGloveJoint,
        DYGLOVE_JOINT_LIMITS, FINGER_JOINTS
    )
    HAS_GLOVE = True
except ImportError:
    HAS_GLOVE = False
    logger.warning("DYGlove driver not available")


class CalibrationStep(str, Enum):
    """Calibration procedure steps."""
    IDLE = "idle"
    FLAT_HAND = "flat_hand"
    FULL_FIST = "full_fist"
    PINCH_GRIP = "pinch_grip"
    SPREAD_FINGERS = "spread_fingers"
    HAPTIC_TEST = "haptic_test"
    COMPLETE = "complete"


@dataclass
class JointCalibration:
    """Calibration data for a single joint."""
    joint_index: int
    joint_name: str

    # Raw encoder values at calibration poses
    zero_raw: float = 0.0           # Encoder value at zero position
    min_raw: float = 0.0            # Encoder value at min angle
    max_raw: float = 0.0            # Encoder value at max angle

    # Calibrated angle limits (radians)
    angle_min: float = 0.0
    angle_max: float = 0.0

    # Scale factor (radians per raw unit)
    scale: float = 1.0

    # Quality metrics
    noise_std: float = 0.0          # Encoder noise level
    linearity_error: float = 0.0    # Non-linearity error

    def raw_to_angle(self, raw_value: float) -> float:
        """Convert raw encoder value to calibrated angle."""
        return (raw_value - self.zero_raw) * self.scale

    def angle_to_raw(self, angle: float) -> float:
        """Convert calibrated angle to raw encoder value."""
        return angle / self.scale + self.zero_raw


@dataclass
class HapticCalibration:
    """Calibration data for haptic feedback motors."""
    motor_index: int
    finger_name: str

    # Motor parameters
    zero_position: float = 0.0      # Servo position at zero force
    max_position: float = 1.0       # Servo position at max force
    max_force_n: float = 5.0        # Maximum force in Newtons

    # Response characteristics
    response_time_ms: float = 10.0  # Time to reach target position
    stiction_threshold: float = 0.1 # Force needed to overcome stiction

    # Calibration quality
    position_error: float = 0.0     # Position tracking error
    force_linearity: float = 1.0    # Force-position linearity (1.0=perfect)


@dataclass
class GloveCalibrationData:
    """Complete calibration data for one glove."""
    glove_id: str
    side: str                       # 'left' or 'right'
    calibrated_at: str              # ISO timestamp
    firmware_version: str = ""

    # Per-joint calibration
    joints: Dict[int, JointCalibration] = field(default_factory=dict)

    # Haptic motor calibration
    haptics: Dict[int, HapticCalibration] = field(default_factory=dict)

    # Hand geometry (optional)
    hand_length_mm: float = 180.0   # Middle finger tip to wrist
    palm_width_mm: float = 85.0     # Palm width at knuckles

    # Quality metrics
    overall_quality: float = 1.0    # 0-1, overall calibration quality

    def save(self, path: Path):
        """Save calibration to JSON file."""
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved calibration to {path}")

    @classmethod
    def load(cls, path: Path) -> 'GloveCalibrationData':
        """Load calibration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        joints = {
            int(k): JointCalibration(**v)
            for k, v in data.get('joints', {}).items()
        }
        haptics = {
            int(k): HapticCalibration(**v)
            for k, v in data.get('haptics', {}).items()
        }

        return cls(
            glove_id=data['glove_id'],
            side=data['side'],
            calibrated_at=data['calibrated_at'],
            firmware_version=data.get('firmware_version', ''),
            joints=joints,
            haptics=haptics,
            hand_length_mm=data.get('hand_length_mm', 180.0),
            palm_width_mm=data.get('palm_width_mm', 85.0),
            overall_quality=data.get('overall_quality', 1.0),
        )


class GloveCalibrator:
    """
    Calibration system for DYGlove.

    Usage:
        calibrator = GloveCalibrator(glove_driver)
        calibrator.start_calibration()

        # User follows prompts...
        calibrator.capture_pose(CalibrationStep.FLAT_HAND)
        calibrator.capture_pose(CalibrationStep.FULL_FIST)
        calibrator.capture_pose(CalibrationStep.PINCH_GRIP)
        calibrator.capture_pose(CalibrationStep.SPREAD_FINGERS)

        calibration = calibrator.finalize()
        calibration.save(Path("/var/lib/dynamical/glove_cal.json"))
    """

    # Number of samples to capture per pose
    SAMPLES_PER_POSE = 50

    def __init__(
        self,
        glove_driver: Optional[Any] = None,
        glove_id: str = "default",
        side: str = "right",
        calibration_dir: str = "/var/lib/dynamical/calibration",
    ):
        """
        Initialize calibrator.

        Args:
            glove_driver: DYGloveDriver instance (or None for mock)
            glove_id: Unique glove identifier
            side: 'left' or 'right'
            calibration_dir: Directory to save calibration files
        """
        self.glove_driver = glove_driver
        self.glove_id = glove_id
        self.side = side
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        self._current_step = CalibrationStep.IDLE
        self._pose_samples: Dict[CalibrationStep, List[np.ndarray]] = {}
        self._calibration: Optional[GloveCalibrationData] = None

        self._lock = threading.Lock()
        self._progress = 0.0
        self._status_message = "Not started"

        # Callbacks
        self._on_progress: Optional[Callable[[float, str], None]] = None

    @property
    def current_step(self) -> CalibrationStep:
        """Get current calibration step."""
        return self._current_step

    @property
    def progress(self) -> float:
        """Get calibration progress (0-1)."""
        return self._progress

    @property
    def status_message(self) -> str:
        """Get current status message."""
        return self._status_message

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates."""
        self._on_progress = callback

    def _update_progress(self, progress: float, message: str):
        """Update progress and notify callback."""
        self._progress = progress
        self._status_message = message
        if self._on_progress:
            try:
                self._on_progress(progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def start_calibration(self) -> bool:
        """
        Start the calibration process.

        Returns:
            True if calibration started successfully
        """
        with self._lock:
            self._pose_samples.clear()
            self._calibration = None
            self._current_step = CalibrationStep.FLAT_HAND
            self._update_progress(0.0, "Calibration started. Hold hand flat.")

        logger.info(f"Started calibration for glove {self.glove_id} ({self.side})")
        return True

    def get_next_instruction(self) -> Dict[str, Any]:
        """
        Get instruction for current calibration step.

        Returns:
            Dict with step info and instruction text
        """
        instructions = {
            CalibrationStep.IDLE: {
                "title": "Ready",
                "instruction": "Press 'Start' to begin calibration",
                "image": "idle.png",
            },
            CalibrationStep.FLAT_HAND: {
                "title": "Flat Hand",
                "instruction": "Hold your hand flat with fingers extended and together. Keep still for 2 seconds.",
                "image": "flat_hand.png",
            },
            CalibrationStep.FULL_FIST: {
                "title": "Full Fist",
                "instruction": "Make a tight fist with thumb wrapped around fingers. Keep still for 2 seconds.",
                "image": "full_fist.png",
            },
            CalibrationStep.PINCH_GRIP: {
                "title": "Pinch Grip",
                "instruction": "Touch your thumb tip to your index fingertip. Keep still for 2 seconds.",
                "image": "pinch_grip.png",
            },
            CalibrationStep.SPREAD_FINGERS: {
                "title": "Spread Fingers",
                "instruction": "Spread all fingers apart as far as possible. Keep still for 2 seconds.",
                "image": "spread_fingers.png",
            },
            CalibrationStep.HAPTIC_TEST: {
                "title": "Haptic Test",
                "instruction": "Relax your hand. You will feel vibration in each finger.",
                "image": "haptic_test.png",
            },
            CalibrationStep.COMPLETE: {
                "title": "Complete",
                "instruction": "Calibration complete! Your glove is ready to use.",
                "image": "complete.png",
            },
        }

        return {
            "step": self._current_step.value,
            **instructions.get(self._current_step, instructions[CalibrationStep.IDLE]),
        }

    def capture_pose(self, step: Optional[CalibrationStep] = None) -> Dict[str, Any]:
        """
        Capture the current pose for calibration.

        Args:
            step: Step to capture (uses current step if None)

        Returns:
            Dict with capture result
        """
        step = step or self._current_step

        if step == CalibrationStep.IDLE:
            return {"success": False, "error": "Calibration not started"}

        if step == CalibrationStep.COMPLETE:
            return {"success": False, "error": "Calibration already complete"}

        # Capture samples
        samples = []
        self._update_progress(self._progress, f"Capturing {step.value}...")

        for i in range(self.SAMPLES_PER_POSE):
            if self.glove_driver:
                state = self.glove_driver.get_state()
                if state:
                    samples.append(state.joint_angles.copy())
            else:
                # Mock data
                samples.append(self._generate_mock_sample(step))

            time.sleep(0.02)  # 50Hz sampling

        if len(samples) < self.SAMPLES_PER_POSE // 2:
            return {"success": False, "error": "Insufficient samples captured"}

        # Store samples
        with self._lock:
            self._pose_samples[step] = samples

        # Compute statistics
        samples_array = np.array(samples)
        mean_angles = np.mean(samples_array, axis=0)
        std_angles = np.std(samples_array, axis=0)

        # Advance to next step
        next_step = self._get_next_step(step)
        self._current_step = next_step

        # Update progress
        step_order = [
            CalibrationStep.FLAT_HAND,
            CalibrationStep.FULL_FIST,
            CalibrationStep.PINCH_GRIP,
            CalibrationStep.SPREAD_FINGERS,
            CalibrationStep.HAPTIC_TEST,
        ]
        try:
            step_idx = step_order.index(step)
            progress = (step_idx + 1) / len(step_order)
        except ValueError:
            progress = self._progress

        if next_step == CalibrationStep.COMPLETE:
            self._update_progress(1.0, "Calibration complete!")
            self._compute_calibration()
        else:
            self._update_progress(progress, f"Captured {step.value}. Next: {next_step.value}")

        return {
            "success": True,
            "step": step.value,
            "samples_captured": len(samples),
            "mean_angles": mean_angles.tolist(),
            "std_angles": std_angles.tolist(),
            "next_step": next_step.value,
        }

    def _get_next_step(self, current: CalibrationStep) -> CalibrationStep:
        """Get the next calibration step."""
        order = [
            CalibrationStep.FLAT_HAND,
            CalibrationStep.FULL_FIST,
            CalibrationStep.PINCH_GRIP,
            CalibrationStep.SPREAD_FINGERS,
            CalibrationStep.HAPTIC_TEST,
            CalibrationStep.COMPLETE,
        ]

        try:
            idx = order.index(current)
            return order[idx + 1] if idx < len(order) - 1 else CalibrationStep.COMPLETE
        except ValueError:
            return CalibrationStep.FLAT_HAND

    def _generate_mock_sample(self, step: CalibrationStep) -> np.ndarray:
        """Generate mock sample data for testing."""
        angles = np.zeros(21)

        if step == CalibrationStep.FLAT_HAND:
            angles = np.random.normal(0, 0.05, 21)  # Near zero

        elif step == CalibrationStep.FULL_FIST:
            # Max flexion for all fingers
            for finger in ['index', 'middle', 'ring', 'pinky']:
                joints = FINGER_JOINTS.get(finger, []) if HAS_GLOVE else []
                for j in joints:
                    if hasattr(j, 'value'):
                        angles[j.value] = 1.5 + np.random.normal(0, 0.1)

        elif step == CalibrationStep.PINCH_GRIP:
            # Thumb and index in pinch position
            angles[0] = 0.5  # Thumb TM flex
            angles[2] = 0.5  # Thumb MCP
            angles[5] = 0.8  # Index MCP flex

        elif step == CalibrationStep.SPREAD_FINGERS:
            # Max abduction
            for j in [1, 6, 10, 14, 18]:  # ABD joints
                if j < len(angles):
                    angles[j] = 0.3 + np.random.normal(0, 0.05)

        return angles + np.random.normal(0, 0.02, 21)  # Add noise

    def _compute_calibration(self):
        """Compute calibration from captured poses."""
        from datetime import datetime

        joints = {}

        # Get reference poses
        flat_samples = self._pose_samples.get(CalibrationStep.FLAT_HAND, [])
        fist_samples = self._pose_samples.get(CalibrationStep.FULL_FIST, [])
        spread_samples = self._pose_samples.get(CalibrationStep.SPREAD_FINGERS, [])

        if not flat_samples:
            logger.error("No flat hand samples captured")
            return

        flat_mean = np.mean(flat_samples, axis=0)
        fist_mean = np.mean(fist_samples, axis=0) if fist_samples else flat_mean
        spread_mean = np.mean(spread_samples, axis=0) if spread_samples else flat_mean

        flat_std = np.std(flat_samples, axis=0)

        # Compute per-joint calibration
        joint_names = [f"joint_{i}" for i in range(21)]
        if HAS_GLOVE:
            joint_names = [j.name for j in DYGloveJoint]

        for i in range(21):
            # Get limits from biomechanics
            limits = (-0.5, 1.5)  # Default
            if HAS_GLOVE:
                joint_enum = DYGloveJoint(i)
                limits = DYGLOVE_JOINT_LIMITS.get(joint_enum, limits)

            # Zero is flat hand position
            zero_raw = flat_mean[i]

            # Min/max from fist (flexion) and spread (abduction)
            min_raw = min(flat_mean[i], fist_mean[i])
            max_raw = max(flat_mean[i], fist_mean[i], spread_mean[i])

            # Scale factor
            raw_range = max_raw - min_raw
            angle_range = limits[1] - limits[0]
            scale = angle_range / raw_range if raw_range > 0.01 else 1.0

            joints[i] = JointCalibration(
                joint_index=i,
                joint_name=joint_names[i] if i < len(joint_names) else f"joint_{i}",
                zero_raw=zero_raw,
                min_raw=min_raw,
                max_raw=max_raw,
                angle_min=limits[0],
                angle_max=limits[1],
                scale=scale,
                noise_std=flat_std[i],
            )

        # Compute haptic calibration (simplified)
        haptics = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, name in enumerate(finger_names):
            haptics[i] = HapticCalibration(
                motor_index=i,
                finger_name=name,
                zero_position=0.0,
                max_position=1.0,
                max_force_n=5.0,
            )

        # Compute overall quality
        avg_noise = np.mean([j.noise_std for j in joints.values()])
        quality = max(0.0, 1.0 - avg_noise / 0.1)  # Quality decreases with noise

        self._calibration = GloveCalibrationData(
            glove_id=self.glove_id,
            side=self.side,
            calibrated_at=datetime.utcnow().isoformat(),
            joints=joints,
            haptics=haptics,
            overall_quality=quality,
        )

        logger.info(f"Computed calibration with quality {quality:.2f}")

    def finalize(self) -> Optional[GloveCalibrationData]:
        """
        Finalize and return calibration data.

        Returns:
            GloveCalibrationData if calibration complete, None otherwise
        """
        if self._current_step != CalibrationStep.COMPLETE:
            logger.warning("Calibration not complete")
            return None

        return self._calibration

    def save_calibration(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        Save calibration to file.

        Args:
            filename: Filename (uses default if None)

        Returns:
            Path to saved file, or None if failed
        """
        if not self._calibration:
            logger.error("No calibration to save")
            return None

        if filename is None:
            filename = f"glove_{self.glove_id}_{self.side}.json"

        path = self.calibration_dir / filename
        self._calibration.save(path)
        return path

    def load_calibration(self, filename: Optional[str] = None) -> Optional[GloveCalibrationData]:
        """
        Load calibration from file.

        Args:
            filename: Filename (uses default if None)

        Returns:
            GloveCalibrationData if loaded, None otherwise
        """
        if filename is None:
            filename = f"glove_{self.glove_id}_{self.side}.json"

        path = self.calibration_dir / filename

        if not path.exists():
            logger.warning(f"Calibration file not found: {path}")
            return None

        self._calibration = GloveCalibrationData.load(path)
        self._current_step = CalibrationStep.COMPLETE
        return self._calibration

    def test_haptics(self) -> Dict[str, Any]:
        """
        Test haptic feedback motors.

        Returns:
            Test results for each motor
        """
        results = {"success": True, "motors": {}}

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        for i, name in enumerate(finger_names):
            # Send test pulse to motor
            if self.glove_driver and hasattr(self.glove_driver, 'set_haptic'):
                # Pulse on
                self.glove_driver.set_haptic(i, 0.5)
                time.sleep(0.2)

                # Pulse off
                self.glove_driver.set_haptic(i, 0.0)
                time.sleep(0.1)

                results["motors"][name] = {"tested": True, "response": "ok"}
            else:
                results["motors"][name] = {"tested": True, "response": "mock"}

        logger.info("Haptic test complete")
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        return {
            "step": self._current_step.value,
            "progress": self._progress,
            "message": self._status_message,
            "poses_captured": list(self._pose_samples.keys()),
            "calibration_ready": self._calibration is not None,
        }


# =============================================================================
# Testing
# =============================================================================

def test_glove_calibration():
    """Test glove calibration system."""
    print("\n" + "=" * 60)
    print("GLOVE CALIBRATION TEST (Mock)")
    print("=" * 60)

    # Create calibrator without real driver
    calibrator = GloveCalibrator(
        glove_id="test_glove",
        side="right",
        calibration_dir="/tmp/glove_cal_test",
    )

    print("\n1. Start Calibration")
    print("-" * 40)
    calibrator.start_calibration()
    print(f"   Step: {calibrator.current_step}")

    print("\n2. Capture Poses")
    print("-" * 40)

    steps = [
        CalibrationStep.FLAT_HAND,
        CalibrationStep.FULL_FIST,
        CalibrationStep.PINCH_GRIP,
        CalibrationStep.SPREAD_FINGERS,
        CalibrationStep.HAPTIC_TEST,
    ]

    for step in steps:
        result = calibrator.capture_pose(step)
        print(f"   {step.value}: samples={result.get('samples_captured', 0)}, next={result.get('next_step')}")

    print("\n3. Finalize Calibration")
    print("-" * 40)

    calibration = calibrator.finalize()
    if calibration:
        print(f"   Glove ID: {calibration.glove_id}")
        print(f"   Side: {calibration.side}")
        print(f"   Joints calibrated: {len(calibration.joints)}")
        print(f"   Quality: {calibration.overall_quality:.2f}")

    print("\n4. Save and Load")
    print("-" * 40)

    path = calibrator.save_calibration()
    if path:
        print(f"   Saved to: {path}")

        loaded = calibrator.load_calibration()
        if loaded:
            print(f"   Loaded: {loaded.glove_id} ({loaded.calibrated_at})")

    print("\n" + "=" * 60)
    print("GLOVE CALIBRATION TESTS COMPLETE")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/glove_cal_test", ignore_errors=True)


if __name__ == "__main__":
    test_glove_calibration()

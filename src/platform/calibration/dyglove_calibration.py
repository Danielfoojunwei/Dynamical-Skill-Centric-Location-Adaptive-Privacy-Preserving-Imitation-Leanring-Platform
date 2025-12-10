"""
DOGlove Wireless Calibration Microservice

Comprehensive calibration service for converting DOGlove from wired to wireless
operation and calibrating all sensor systems.

Based on DOGlove specifications (https://do-glove.github.io/):
- 21-DOF motion capture (3 joints × 5 fingers + 6 wrist)
- 5-DOF haptic force feedback (fingertips)
- 5-DOF vibrotactile feedback (LRA actuators)
- MoCap frequency: 120 Hz max
- Haptic feedback frequency: 30 Hz max
- Joint angle accuracy: ±7.2° (reducible through calibration)
- Assembly cost: <$600 USD

Calibration Components:
1. IMU Calibration - Magnetometer, accelerometer, gyroscope
2. Joint Encoder Calibration - Per-joint min/max/offset values
3. Haptic Force Calibration - Force sensor thresholds and mapping
4. Wireless Link Calibration - BLE/WiFi latency and reliability
5. Cross-Calibration with MMPose - Align glove with camera-based tracking
"""

import os
import sys
import json
import math
import time
import struct
import logging
import threading
import statistics
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# =============================================================================
# Constants and Enums
# =============================================================================

class CalibrationPhase(str, Enum):
    """Calibration phases."""
    NOT_STARTED = "not_started"
    IMU_CALIBRATION = "imu_calibration"
    JOINT_CALIBRATION = "joint_calibration"
    HAPTIC_CALIBRATION = "haptic_calibration"
    WIRELESS_CALIBRATION = "wireless_calibration"
    CROSS_CALIBRATION = "cross_calibration"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


class WirelessProtocol(str, Enum):
    """Wireless protocols for DOGlove."""
    BLE = "ble"  # Bluetooth Low Energy
    BLE_CLASSIC = "ble_classic"  # Bluetooth Classic SPP
    WIFI = "wifi"  # WiFi UDP
    WIFI_TCP = "wifi_tcp"  # WiFi TCP


class Hand(str, Enum):
    """Hand type."""
    LEFT = "left"
    RIGHT = "right"


class Finger(int, Enum):
    """Finger indices."""
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class Joint(int, Enum):
    """Joint indices per finger."""
    MCP = 0  # Metacarpophalangeal
    PIP = 1  # Proximal interphalangeal
    DIP = 2  # Distal interphalangeal


# DOGlove specifications
DOGLOVE_SPECS = {
    "mocap_dof": 21,  # 15 finger + 6 wrist
    "haptic_dof": 5,  # 5 fingertips
    "vibrotactile_dof": 5,  # 5 LRA actuators
    "max_mocap_hz": 120,
    "max_haptic_hz": 30,
    "encoder_error_deg": 7.2,  # ±7.2° raw accuracy
    "calibrated_error_deg": 2.0,  # Target after calibration
    "force_threshold_1": 10,  # grams - detection threshold
    "force_threshold_2": 50,  # grams - feedback activation
    "force_threshold_3": 100,  # grams - force-only mode
}

# Joint ranges (typical human hand, degrees)
JOINT_RANGES = {
    Finger.THUMB: {
        Joint.MCP: (-10, 90),
        Joint.PIP: (-10, 80),
        Joint.DIP: (-5, 70),
    },
    Finger.INDEX: {
        Joint.MCP: (-20, 90),
        Joint.PIP: (0, 100),
        Joint.DIP: (0, 80),
    },
    Finger.MIDDLE: {
        Joint.MCP: (-20, 90),
        Joint.PIP: (0, 100),
        Joint.DIP: (0, 80),
    },
    Finger.RING: {
        Joint.MCP: (-20, 90),
        Joint.PIP: (0, 100),
        Joint.DIP: (0, 80),
    },
    Finger.PINKY: {
        Joint.MCP: (-20, 90),
        Joint.PIP: (0, 100),
        Joint.DIP: (0, 80),
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IMUCalibrationData:
    """IMU calibration parameters."""
    # Accelerometer bias (m/s²)
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0
    
    # Accelerometer scale
    accel_scale_x: float = 1.0
    accel_scale_y: float = 1.0
    accel_scale_z: float = 1.0
    
    # Gyroscope bias (deg/s)
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0
    
    # Magnetometer hard iron offset
    mag_offset_x: float = 0.0
    mag_offset_y: float = 0.0
    mag_offset_z: float = 0.0
    
    # Magnetometer soft iron matrix (flattened 3x3)
    mag_scale_matrix: List[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ])
    
    # Calibration quality
    accel_noise_std: float = 0.0
    gyro_noise_std: float = 0.0
    mag_noise_std: float = 0.0
    
    timestamp: str = ""


@dataclass
class JointCalibrationData:
    """Per-joint calibration parameters."""
    finger: int  # Finger index
    joint: int   # Joint index
    
    # Raw sensor range
    raw_min: int = 0
    raw_max: int = 4095  # 12-bit ADC
    
    # Calibrated angle range (degrees)
    angle_min: float = 0.0
    angle_max: float = 90.0
    
    # Offset and scale
    offset: float = 0.0
    scale: float = 1.0
    
    # Non-linearity correction (polynomial coefficients)
    poly_coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    
    # Quality metrics
    linearity_error: float = 0.0  # RMS error in degrees
    hysteresis: float = 0.0  # Degrees


@dataclass
class HapticCalibrationData:
    """Haptic actuator calibration parameters."""
    finger: int  # Finger index
    
    # Force sensor calibration
    force_sensor_offset: float = 0.0
    force_sensor_scale: float = 1.0  # grams per ADC unit
    
    # Servo motor calibration
    servo_min_pwm: int = 500
    servo_max_pwm: int = 2500
    servo_neutral_pwm: int = 1500
    
    # Force feedback mapping
    force_to_pwm_coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0])
    
    # LRA vibrotactile calibration
    lra_min_intensity: int = 0
    lra_max_intensity: int = 255
    lra_resonant_freq_hz: float = 170.0  # Typical LRA resonance
    
    # Thresholds (grams)
    threshold_detect: float = 10.0
    threshold_feedback: float = 50.0
    threshold_force_only: float = 100.0


@dataclass
class WirelessCalibrationData:
    """Wireless link calibration parameters."""
    protocol: WirelessProtocol = WirelessProtocol.BLE
    
    # Connection parameters
    device_address: str = ""
    service_uuid: str = ""
    tx_characteristic: str = ""
    rx_characteristic: str = ""
    
    # Performance metrics
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_max_ms: float = 0.0
    
    packet_loss_percent: float = 0.0
    throughput_kbps: float = 0.0
    
    # Optimal parameters
    mtu_size: int = 23  # BLE default
    connection_interval_ms: float = 7.5  # BLE connection interval
    tx_power_dbm: int = 0
    
    # Battery impact
    estimated_battery_hours: float = 8.0


@dataclass
class CrossCalibrationData:
    """Cross-calibration with MMPose data."""
    # Transform from glove frame to camera frame
    rotation_quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    translation_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Per-fingertip offsets (glove vs camera tracking)
    fingertip_offsets: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Quality metrics
    mean_position_error_mm: float = 0.0
    max_position_error_mm: float = 0.0
    correlation_coefficient: float = 0.0


@dataclass
class FullCalibrationResult:
    """Complete calibration result."""
    calibration_id: str
    glove_serial: str
    hand: Hand
    timestamp: str
    
    # Component calibrations
    imu: IMUCalibrationData = field(default_factory=IMUCalibrationData)
    joints: List[JointCalibrationData] = field(default_factory=list)
    haptics: List[HapticCalibrationData] = field(default_factory=list)
    wireless: WirelessCalibrationData = field(default_factory=WirelessCalibrationData)
    cross_calibration: Optional[CrossCalibrationData] = None
    
    # Overall quality
    overall_joint_accuracy_deg: float = 0.0
    overall_haptic_accuracy_percent: float = 0.0
    wireless_reliability_percent: float = 0.0
    
    # Status
    phase: CalibrationPhase = CalibrationPhase.NOT_STARTED
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# IMU Calibrator
# =============================================================================

class IMUCalibrator:
    """
    Calibrates the IMU (Inertial Measurement Unit) in DOGlove.
    
    The IMU provides wrist orientation tracking via:
    - 3-axis accelerometer
    - 3-axis gyroscope
    - 3-axis magnetometer
    
    Calibration procedures:
    1. Accelerometer: Static poses in 6 orientations
    2. Gyroscope: Stationary bias measurement
    3. Magnetometer: Figure-8 motion for hard/soft iron
    """
    
    # Gravity constant for accelerometer calibration
    GRAVITY = 9.81
    
    # Number of samples for each calibration step
    SAMPLES_PER_POSE = 100
    STATIONARY_SAMPLES = 500
    MAG_MOTION_SAMPLES = 1000
    
    def __init__(self):
        self._samples: Dict[str, List] = {
            "accel_poses": [],  # List of (pose_name, samples)
            "gyro_stationary": [],
            "mag_motion": [],
        }
        self._calibration: Optional[IMUCalibrationData] = None
    
    def collect_accelerometer_pose(
        self,
        pose_name: str,
        samples: List[Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """
        Collect accelerometer samples for a calibration pose.
        
        Standard 6-pose calibration:
        - pose_x_up: X-axis pointing up
        - pose_x_down: X-axis pointing down
        - pose_y_up: Y-axis pointing up
        - pose_y_down: Y-axis pointing down
        - pose_z_up: Z-axis pointing up (normal orientation)
        - pose_z_down: Z-axis pointing down
        
        Args:
            pose_name: Name of the calibration pose
            samples: List of (ax, ay, az) accelerometer readings
        
        Returns:
            Collection result with statistics
        """
        if len(samples) < self.SAMPLES_PER_POSE:
            return {
                "error": f"Need at least {self.SAMPLES_PER_POSE} samples, got {len(samples)}"
            }
        
        # Calculate mean and std for this pose
        if HAS_NUMPY:
            samples_arr = np.array(samples)
            mean = np.mean(samples_arr, axis=0)
            std = np.std(samples_arr, axis=0)
        else:
            mean = [statistics.mean(s[i] for s in samples) for i in range(3)]
            std = [statistics.stdev(s[i] for s in samples) for i in range(3)]
        
        self._samples["accel_poses"].append({
            "pose": pose_name,
            "mean": list(mean),
            "std": list(std),
            "n_samples": len(samples),
        })
        
        return {
            "pose": pose_name,
            "mean": list(mean),
            "std": list(std),
            "magnitude": math.sqrt(sum(m**2 for m in mean)),
            "status": "collected",
        }
    
    def collect_gyroscope_stationary(
        self,
        samples: List[Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """
        Collect gyroscope samples while glove is stationary.
        
        This measures the gyroscope bias (drift when not moving).
        
        Args:
            samples: List of (gx, gy, gz) gyroscope readings (deg/s)
        
        Returns:
            Collection result with bias estimate
        """
        if len(samples) < self.STATIONARY_SAMPLES:
            return {
                "error": f"Need at least {self.STATIONARY_SAMPLES} samples"
            }
        
        if HAS_NUMPY:
            samples_arr = np.array(samples)
            bias = np.mean(samples_arr, axis=0)
            noise_std = np.std(samples_arr, axis=0)
        else:
            bias = [statistics.mean(s[i] for s in samples) for i in range(3)]
            noise_std = [statistics.stdev(s[i] for s in samples) for i in range(3)]
        
        self._samples["gyro_stationary"] = {
            "bias": list(bias),
            "noise_std": list(noise_std),
            "n_samples": len(samples),
        }
        
        return {
            "bias_deg_s": list(bias),
            "noise_std_deg_s": list(noise_std),
            "status": "collected",
        }
    
    def collect_magnetometer_motion(
        self,
        samples: List[Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """
        Collect magnetometer samples during figure-8 motion.
        
        The figure-8 motion samples all orientations to calibrate
        hard iron (constant offset) and soft iron (axis-dependent scaling)
        distortions.
        
        Args:
            samples: List of (mx, my, mz) magnetometer readings
        
        Returns:
            Collection result
        """
        if len(samples) < self.MAG_MOTION_SAMPLES:
            return {
                "error": f"Need at least {self.MAG_MOTION_SAMPLES} samples"
            }
        
        self._samples["mag_motion"] = samples
        
        if HAS_NUMPY:
            samples_arr = np.array(samples)
            center = np.mean(samples_arr, axis=0)
        else:
            center = [statistics.mean(s[i] for s in samples) for i in range(3)]
        
        return {
            "n_samples": len(samples),
            "rough_center": list(center),
            "status": "collected",
        }
    
    def compute_calibration(self) -> IMUCalibrationData:
        """
        Compute full IMU calibration from collected samples.
        
        Returns:
            Complete IMU calibration data
        """
        cal = IMUCalibrationData(timestamp=datetime.utcnow().isoformat())
        
        # Calibrate accelerometer
        self._calibrate_accelerometer(cal)
        
        # Calibrate gyroscope
        self._calibrate_gyroscope(cal)
        
        # Calibrate magnetometer
        self._calibrate_magnetometer(cal)
        
        self._calibration = cal
        return cal
    
    def _calibrate_accelerometer(self, cal: IMUCalibrationData):
        """Compute accelerometer calibration parameters."""
        poses = self._samples.get("accel_poses", [])
        
        if len(poses) < 6:
            logger.warning("Insufficient accelerometer poses for full calibration")
            return
        
        # Find poses by orientation
        pose_map = {p["pose"]: p["mean"] for p in poses}
        
        # Calculate bias (average of opposing poses should equal zero)
        if "pose_x_up" in pose_map and "pose_x_down" in pose_map:
            cal.accel_bias_x = (pose_map["pose_x_up"][0] + pose_map["pose_x_down"][0]) / 2
            cal.accel_scale_x = (pose_map["pose_x_up"][0] - pose_map["pose_x_down"][0]) / (2 * self.GRAVITY)
        
        if "pose_y_up" in pose_map and "pose_y_down" in pose_map:
            cal.accel_bias_y = (pose_map["pose_y_up"][1] + pose_map["pose_y_down"][1]) / 2
            cal.accel_scale_y = (pose_map["pose_y_up"][1] - pose_map["pose_y_down"][1]) / (2 * self.GRAVITY)
        
        if "pose_z_up" in pose_map and "pose_z_down" in pose_map:
            cal.accel_bias_z = (pose_map["pose_z_up"][2] + pose_map["pose_z_down"][2]) / 2
            cal.accel_scale_z = (pose_map["pose_z_up"][2] - pose_map["pose_z_down"][2]) / (2 * self.GRAVITY)
        
        # Calculate noise from std values
        avg_std = statistics.mean(
            math.sqrt(sum(p["std"][i]**2 for i in range(3))) 
            for p in poses
        )
        cal.accel_noise_std = avg_std
    
    def _calibrate_gyroscope(self, cal: IMUCalibrationData):
        """Compute gyroscope calibration parameters."""
        gyro_data = self._samples.get("gyro_stationary", {})
        
        if not gyro_data:
            logger.warning("No gyroscope calibration data")
            return
        
        bias = gyro_data.get("bias", [0, 0, 0])
        noise_std = gyro_data.get("noise_std", [0, 0, 0])
        
        cal.gyro_bias_x = bias[0]
        cal.gyro_bias_y = bias[1]
        cal.gyro_bias_z = bias[2]
        cal.gyro_noise_std = math.sqrt(sum(n**2 for n in noise_std))
    
    def _calibrate_magnetometer(self, cal: IMUCalibrationData):
        """Compute magnetometer calibration using ellipsoid fitting."""
        mag_samples = self._samples.get("mag_motion", [])
        
        if len(mag_samples) < self.MAG_MOTION_SAMPLES:
            logger.warning("Insufficient magnetometer samples")
            return
        
        if not HAS_NUMPY:
            # Simple center calculation without numpy
            center = [statistics.mean(s[i] for s in mag_samples) for i in range(3)]
            cal.mag_offset_x = center[0]
            cal.mag_offset_y = center[1]
            cal.mag_offset_z = center[2]
            return
        
        samples_arr = np.array(mag_samples)
        
        # Simple hard iron calibration: find center of data
        center = np.mean(samples_arr, axis=0)
        cal.mag_offset_x = center[0]
        cal.mag_offset_y = center[1]
        cal.mag_offset_z = center[2]
        
        # Center the data
        centered = samples_arr - center
        
        # Soft iron: estimate axis scales from data spread
        stds = np.std(centered, axis=0)
        avg_std = np.mean(stds)
        
        # Create simple scaling matrix
        if avg_std > 0:
            scale_factors = avg_std / stds
            cal.mag_scale_matrix = [
                scale_factors[0], 0, 0,
                0, scale_factors[1], 0,
                0, 0, scale_factors[2]
            ]
        
        cal.mag_noise_std = float(np.mean(stds))
    
    def apply_calibration(
        self,
        accel: Tuple[float, float, float],
        gyro: Tuple[float, float, float],
        mag: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Apply calibration to raw IMU readings.
        
        Args:
            accel: Raw accelerometer (ax, ay, az)
            gyro: Raw gyroscope (gx, gy, gz)
            mag: Raw magnetometer (mx, my, mz)
        
        Returns:
            Tuple of calibrated (accel, gyro, mag)
        """
        if not self._calibration:
            return accel, gyro, mag
        
        cal = self._calibration
        
        # Calibrate accelerometer
        cal_accel = (
            (accel[0] - cal.accel_bias_x) / cal.accel_scale_x,
            (accel[1] - cal.accel_bias_y) / cal.accel_scale_y,
            (accel[2] - cal.accel_bias_z) / cal.accel_scale_z,
        )
        
        # Calibrate gyroscope
        cal_gyro = (
            gyro[0] - cal.gyro_bias_x,
            gyro[1] - cal.gyro_bias_y,
            gyro[2] - cal.gyro_bias_z,
        )
        
        # Calibrate magnetometer
        centered = (
            mag[0] - cal.mag_offset_x,
            mag[1] - cal.mag_offset_y,
            mag[2] - cal.mag_offset_z,
        )
        
        # Apply soft iron matrix
        m = cal.mag_scale_matrix
        cal_mag = (
            m[0]*centered[0] + m[1]*centered[1] + m[2]*centered[2],
            m[3]*centered[0] + m[4]*centered[1] + m[5]*centered[2],
            m[6]*centered[0] + m[7]*centered[1] + m[8]*centered[2],
        )
        
        return cal_accel, cal_gyro, cal_mag


# =============================================================================
# Joint Encoder Calibrator
# =============================================================================

class JointEncoderCalibrator:
    """
    Calibrates the joint angle encoders in DOGlove.
    
    DOGlove uses rotary encoders to measure 21 joint angles:
    - 5 fingers × 3 joints (MCP, PIP, DIP) = 15 DOF
    - Wrist (roll, pitch, yaw) = 3 DOF (from IMU)
    - Additional wrist joints = 3 DOF
    
    Raw encoder error is ±7.2°, reduced to ±2° through calibration.
    
    Calibration procedure:
    1. Full extension pose (all joints at minimum)
    2. Full flexion pose (all joints at maximum)
    3. Intermediate poses for linearity check
    4. Hysteresis measurement (flex then extend)
    """
    
    def __init__(self):
        self._calibrations: Dict[Tuple[int, int], JointCalibrationData] = {}
        self._samples: Dict[Tuple[int, int], List[Dict]] = {}
    
    def collect_pose_sample(
        self,
        pose_name: str,
        joint_readings: Dict[Tuple[int, int], int],
        reference_angles: Dict[Tuple[int, int], float] = None
    ) -> Dict[str, Any]:
        """
        Collect joint encoder readings for a calibration pose.
        
        Args:
            pose_name: Name of pose (e.g., "full_extension", "full_flexion")
            joint_readings: Dict of (finger, joint) -> raw ADC value
            reference_angles: Optional ground truth angles from external system
        
        Returns:
            Collection result
        """
        for (finger, joint), raw_value in joint_readings.items():
            key = (finger, joint)
            
            if key not in self._samples:
                self._samples[key] = []
            
            sample = {
                "pose": pose_name,
                "raw": raw_value,
                "reference": reference_angles.get(key) if reference_angles else None,
            }
            self._samples[key].append(sample)
        
        return {
            "pose": pose_name,
            "joints_collected": len(joint_readings),
            "status": "collected",
        }
    
    def compute_joint_calibration(
        self,
        finger: int,
        joint: int
    ) -> JointCalibrationData:
        """
        Compute calibration for a specific joint.
        
        Args:
            finger: Finger index (0-4)
            joint: Joint index (0-2)
        
        Returns:
            Joint calibration data
        """
        key = (finger, joint)
        samples = self._samples.get(key, [])
        
        cal = JointCalibrationData(finger=finger, joint=joint)
        
        if len(samples) < 2:
            logger.warning(f"Insufficient samples for finger {finger}, joint {joint}")
            return cal
        
        # Find min/max raw values
        raw_values = [s["raw"] for s in samples]
        cal.raw_min = min(raw_values)
        cal.raw_max = max(raw_values)
        
        # Get expected angle range for this joint
        finger_enum = Finger(finger)
        joint_enum = Joint(joint)
        angle_range = JOINT_RANGES.get(finger_enum, {}).get(joint_enum, (0, 90))
        cal.angle_min = angle_range[0]
        cal.angle_max = angle_range[1]
        
        # Calculate scale and offset for linear mapping
        raw_range = cal.raw_max - cal.raw_min
        angle_range_size = cal.angle_max - cal.angle_min
        
        if raw_range > 0:
            cal.scale = angle_range_size / raw_range
            cal.offset = cal.angle_min - cal.scale * cal.raw_min
        
        # If we have reference angles, compute linearity error
        ref_samples = [(s["raw"], s["reference"]) for s in samples if s["reference"] is not None]
        if len(ref_samples) >= 3:
            errors = []
            for raw, ref in ref_samples:
                predicted = cal.offset + cal.scale * raw
                errors.append(abs(predicted - ref))
            cal.linearity_error = statistics.mean(errors)
        
        # Compute polynomial correction if we have enough reference data
        if HAS_NUMPY and len(ref_samples) >= 5:
            raws = np.array([s[0] for s in ref_samples])
            refs = np.array([s[1] for s in ref_samples])
            
            # Fit quadratic polynomial
            coeffs = np.polyfit(raws, refs, 2)
            cal.poly_coeffs = list(coeffs)
            
            # Recalculate error with polynomial
            predicted = np.polyval(coeffs, raws)
            cal.linearity_error = float(np.sqrt(np.mean((predicted - refs) ** 2)))
        
        self._calibrations[key] = cal
        return cal
    
    def compute_all_calibrations(self) -> List[JointCalibrationData]:
        """Compute calibrations for all joints with data."""
        calibrations = []
        
        for key in self._samples.keys():
            cal = self.compute_joint_calibration(key[0], key[1])
            calibrations.append(cal)
        
        return calibrations
    
    def apply_calibration(
        self,
        finger: int,
        joint: int,
        raw_value: int
    ) -> float:
        """
        Apply calibration to convert raw reading to angle.
        
        Args:
            finger: Finger index
            joint: Joint index
            raw_value: Raw ADC reading
        
        Returns:
            Calibrated angle in degrees
        """
        key = (finger, joint)
        cal = self._calibrations.get(key)
        
        if not cal:
            # Default linear mapping
            return raw_value * 90.0 / 4095.0
        
        # Apply polynomial correction if available
        if len(cal.poly_coeffs) == 3 and HAS_NUMPY:
            return float(np.polyval(cal.poly_coeffs, raw_value))
        
        # Linear mapping
        return cal.offset + cal.scale * raw_value


# =============================================================================
# Haptic Calibrator
# =============================================================================

class HapticCalibrator:
    """
    Calibrates the haptic feedback system in DOGlove.
    
    DOGlove provides:
    - 5-DOF force feedback via cable-driven servo motors
    - 5-DOF vibrotactile feedback via Linear Resonant Actuators (LRA)
    
    Force feedback thresholds:
    - 10g: Detection threshold (haptic starts)
    - 50g: Feedback activation (user study mode)
    - 100g: Force-only mode (haptic stops, force continues)
    """
    
    def __init__(self):
        self._calibrations: Dict[int, HapticCalibrationData] = {}
        self._samples: Dict[int, List[Dict]] = {}
    
    def collect_force_sample(
        self,
        finger: int,
        applied_force_grams: float,
        sensor_reading: int,
        pwm_value: int = None
    ) -> Dict[str, Any]:
        """
        Collect force sensor calibration sample.
        
        Args:
            finger: Finger index (0-4)
            applied_force_grams: Known applied force in grams
            sensor_reading: Raw sensor ADC reading
            pwm_value: Servo PWM value if testing motor response
        
        Returns:
            Collection result
        """
        if finger not in self._samples:
            self._samples[finger] = []
        
        self._samples[finger].append({
            "force_grams": applied_force_grams,
            "sensor_raw": sensor_reading,
            "pwm": pwm_value,
        })
        
        return {
            "finger": finger,
            "samples_collected": len(self._samples[finger]),
            "status": "collected",
        }
    
    def compute_haptic_calibration(self, finger: int) -> HapticCalibrationData:
        """
        Compute haptic calibration for a finger.
        
        Args:
            finger: Finger index
        
        Returns:
            Haptic calibration data
        """
        samples = self._samples.get(finger, [])
        cal = HapticCalibrationData(finger=finger)
        
        if len(samples) < 3:
            logger.warning(f"Insufficient haptic samples for finger {finger}")
            return cal
        
        # Fit linear force sensor calibration
        forces = [s["force_grams"] for s in samples]
        readings = [s["sensor_raw"] for s in samples]
        
        if HAS_NUMPY:
            # Linear regression: force = offset + scale * reading
            coeffs = np.polyfit(readings, forces, 1)
            cal.force_sensor_scale = coeffs[0]
            cal.force_sensor_offset = coeffs[1]
        else:
            # Simple two-point calibration
            min_idx = forces.index(min(forces))
            max_idx = forces.index(max(forces))
            
            if readings[max_idx] != readings[min_idx]:
                cal.force_sensor_scale = (forces[max_idx] - forces[min_idx]) / (readings[max_idx] - readings[min_idx])
                cal.force_sensor_offset = forces[min_idx] - cal.force_sensor_scale * readings[min_idx]
        
        # Set standard thresholds (can be customized)
        cal.threshold_detect = DOGLOVE_SPECS["force_threshold_1"]
        cal.threshold_feedback = DOGLOVE_SPECS["force_threshold_2"]
        cal.threshold_force_only = DOGLOVE_SPECS["force_threshold_3"]
        
        # Compute force-to-PWM mapping if we have motor response data
        pwm_samples = [(s["force_grams"], s["pwm"]) for s in samples if s["pwm"] is not None]
        if len(pwm_samples) >= 2 and HAS_NUMPY:
            forces = np.array([s[0] for s in pwm_samples])
            pwms = np.array([s[1] for s in pwm_samples])
            cal.force_to_pwm_coeffs = list(np.polyfit(forces, pwms, 1))
        
        self._calibrations[finger] = cal
        return cal
    
    def compute_all_calibrations(self) -> List[HapticCalibrationData]:
        """Compute haptic calibrations for all fingers."""
        return [self.compute_haptic_calibration(i) for i in range(5)]
    
    def force_to_feedback(
        self,
        finger: int,
        measured_force_grams: float
    ) -> Dict[str, Any]:
        """
        Determine feedback based on measured force.
        
        Uses the three-threshold system from DOGlove paper:
        - Below threshold_detect: No feedback
        - threshold_detect to threshold_feedback: Haptic vibration
        - threshold_feedback to threshold_force_only: Force + haptic
        - Above threshold_force_only: Force only
        
        Args:
            finger: Finger index
            measured_force_grams: Measured force
        
        Returns:
            Feedback command
        """
        cal = self._calibrations.get(finger, HapticCalibrationData(finger=finger))
        
        if measured_force_grams < cal.threshold_detect:
            return {
                "force_feedback": False,
                "haptic_feedback": False,
                "intensity": 0.0,
            }
        elif measured_force_grams < cal.threshold_feedback:
            intensity = (measured_force_grams - cal.threshold_detect) / (cal.threshold_feedback - cal.threshold_detect)
            return {
                "force_feedback": False,
                "haptic_feedback": True,
                "intensity": min(1.0, intensity),
            }
        elif measured_force_grams < cal.threshold_force_only:
            intensity = (measured_force_grams - cal.threshold_feedback) / (cal.threshold_force_only - cal.threshold_feedback)
            return {
                "force_feedback": True,
                "haptic_feedback": True,
                "intensity": min(1.0, intensity),
            }
        else:
            return {
                "force_feedback": True,
                "haptic_feedback": False,
                "intensity": 1.0,
            }


# =============================================================================
# Wireless Calibrator
# =============================================================================

class WirelessCalibrator:
    """
    Calibrates wireless communication for DOGlove.
    
    Converts wired DOGlove to wireless using:
    - Bluetooth Low Energy (BLE) - default
    - Bluetooth Classic with SPP
    - WiFi UDP for lowest latency
    
    Calibration measures:
    - Round-trip latency
    - Packet loss
    - Throughput
    - Optimal connection parameters
    """
    
    # Test parameters
    LATENCY_TEST_PACKETS = 100
    THROUGHPUT_TEST_DURATION = 5.0  # seconds
    
    def __init__(self, protocol: WirelessProtocol = WirelessProtocol.BLE):
        self.protocol = protocol
        self._calibration = WirelessCalibrationData(protocol=protocol)
        self._connection = None
    
    async def test_latency(
        self,
        device_address: str,
        num_packets: int = None
    ) -> Dict[str, Any]:
        """
        Test round-trip latency.
        
        Args:
            device_address: BLE/WiFi address
            num_packets: Number of test packets
        
        Returns:
            Latency statistics
        """
        num_packets = num_packets or self.LATENCY_TEST_PACKETS
        latencies = []
        
        # Simulate latency measurement (actual implementation would use real connection)
        for i in range(num_packets):
            # Send ping packet and measure response time
            start = time.perf_counter()
            
            # Simulated response
            await self._send_ping()
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)
        
        self._calibration.latency_mean_ms = statistics.mean(latencies)
        self._calibration.latency_std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
        self._calibration.latency_max_ms = max(latencies)
        
        return {
            "mean_ms": self._calibration.latency_mean_ms,
            "std_ms": self._calibration.latency_std_ms,
            "max_ms": self._calibration.latency_max_ms,
            "min_ms": min(latencies),
            "samples": len(latencies),
        }
    
    async def test_packet_loss(
        self,
        device_address: str,
        num_packets: int = 1000
    ) -> Dict[str, Any]:
        """
        Test packet loss rate.
        
        Args:
            device_address: Device address
            num_packets: Number of test packets
        
        Returns:
            Packet loss statistics
        """
        sent = 0
        received = 0
        
        for i in range(num_packets):
            sent += 1
            if await self._send_test_packet(i):
                received += 1
        
        loss_rate = (sent - received) / sent * 100 if sent > 0 else 0
        self._calibration.packet_loss_percent = loss_rate
        
        return {
            "sent": sent,
            "received": received,
            "loss_percent": loss_rate,
        }
    
    async def test_throughput(
        self,
        device_address: str,
        duration: float = None
    ) -> Dict[str, Any]:
        """
        Test data throughput.
        
        Args:
            device_address: Device address
            duration: Test duration in seconds
        
        Returns:
            Throughput statistics
        """
        duration = duration or self.THROUGHPUT_TEST_DURATION
        
        bytes_sent = 0
        start = time.perf_counter()
        
        # Send maximum rate data
        packet_size = self._calibration.mtu_size
        
        while time.perf_counter() - start < duration:
            await self._send_data_packet(b'\x00' * packet_size)
            bytes_sent += packet_size
        
        elapsed = time.perf_counter() - start
        throughput_kbps = (bytes_sent * 8 / 1000) / elapsed
        
        self._calibration.throughput_kbps = throughput_kbps
        
        return {
            "bytes_sent": bytes_sent,
            "duration_s": elapsed,
            "throughput_kbps": throughput_kbps,
        }
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """
        Determine optimal wireless parameters based on test results.
        
        Returns:
            Recommended parameters
        """
        cal = self._calibration
        
        recommendations = {}
        
        # MTU optimization
        if cal.protocol == WirelessProtocol.BLE:
            # BLE can negotiate larger MTU
            if cal.latency_mean_ms < 10:
                recommendations["mtu"] = 247  # Max BLE MTU
            else:
                recommendations["mtu"] = 64  # Smaller for lower latency
        
        # Connection interval
        if cal.protocol == WirelessProtocol.BLE:
            if cal.latency_mean_ms < 15:
                recommendations["connection_interval_ms"] = 7.5  # Minimum
            else:
                recommendations["connection_interval_ms"] = 15.0  # Standard
        
        # TX power
        if cal.packet_loss_percent > 5:
            recommendations["tx_power_dbm"] = 4  # Increase power
        elif cal.packet_loss_percent < 1:
            recommendations["tx_power_dbm"] = -4  # Reduce for battery
        else:
            recommendations["tx_power_dbm"] = 0
        
        # Update calibration
        cal.mtu_size = recommendations.get("mtu", cal.mtu_size)
        cal.connection_interval_ms = recommendations.get("connection_interval_ms", cal.connection_interval_ms)
        cal.tx_power_dbm = recommendations.get("tx_power_dbm", cal.tx_power_dbm)
        
        # Estimate battery life
        base_hours = 8.0
        if cal.tx_power_dbm > 0:
            base_hours *= 0.7  # Higher power = less battery
        if cal.connection_interval_ms < 10:
            base_hours *= 0.8  # Faster polling = less battery
        
        cal.estimated_battery_hours = base_hours
        
        recommendations["estimated_battery_hours"] = base_hours
        
        return recommendations
    
    async def _send_ping(self):
        """Send ping packet (placeholder for actual implementation)."""
        await asyncio.sleep(0.008)  # Simulate ~8ms latency
    
    async def _send_test_packet(self, sequence: int) -> bool:
        """Send test packet and check response."""
        await asyncio.sleep(0.001)
        return True  # Simulate no loss
    
    async def _send_data_packet(self, data: bytes):
        """Send data packet."""
        await asyncio.sleep(0.001)
    
    def get_calibration(self) -> WirelessCalibrationData:
        """Get current calibration data."""
        return self._calibration


# Need asyncio for wireless calibrator
try:
    import asyncio
except ImportError:
    pass


# =============================================================================
# Cross-Calibration with MMPose
# =============================================================================

class CrossCalibrator:
    """
    Cross-calibrates DOGlove with MMPose camera-based tracking.
    
    This aligns the glove's coordinate frame with the camera system
    and validates tracking accuracy.
    """
    
    def __init__(self):
        self._samples: List[Dict] = []
        self._calibration: Optional[CrossCalibrationData] = None
    
    def collect_alignment_sample(
        self,
        glove_fingertips: Dict[int, Tuple[float, float, float]],
        camera_fingertips: Dict[int, Tuple[float, float, float]],
        glove_wrist_pose: Tuple[float, float, float, float, float, float, float] = None
    ):
        """
        Collect alignment sample with simultaneous glove and camera data.
        
        Args:
            glove_fingertips: Fingertip positions from glove FK (finger -> (x,y,z))
            camera_fingertips: Fingertip positions from MMPose (finger -> (x,y,z))
            glove_wrist_pose: Wrist pose from glove IMU (x, y, z, qw, qx, qy, qz)
        """
        self._samples.append({
            "glove": glove_fingertips,
            "camera": camera_fingertips,
            "wrist": glove_wrist_pose,
            "timestamp": time.time(),
        })
    
    def compute_cross_calibration(self) -> CrossCalibrationData:
        """
        Compute cross-calibration transform.
        
        Uses Procrustes analysis to find optimal rigid transform
        from glove frame to camera frame.
        
        Returns:
            Cross-calibration data
        """
        if len(self._samples) < 10:
            logger.warning("Insufficient cross-calibration samples")
            return CrossCalibrationData()
        
        if not HAS_NUMPY:
            logger.warning("numpy required for cross-calibration")
            return CrossCalibrationData()
        
        # Collect all point pairs
        glove_points = []
        camera_points = []
        
        for sample in self._samples:
            for finger in range(5):
                if finger in sample["glove"] and finger in sample["camera"]:
                    glove_points.append(sample["glove"][finger])
                    camera_points.append(sample["camera"][finger])
        
        glove_points = np.array(glove_points)
        camera_points = np.array(camera_points)
        
        # Compute centroids
        glove_centroid = np.mean(glove_points, axis=0)
        camera_centroid = np.mean(camera_points, axis=0)
        
        # Center the points
        glove_centered = glove_points - glove_centroid
        camera_centered = camera_points - camera_centroid
        
        # Compute rotation using SVD (Kabsch algorithm)
        H = glove_centered.T @ camera_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = camera_centroid - R @ glove_centroid
        
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
        quat = (quat[3], quat[0], quat[1], quat[2])  # (w, x, y, z)
        
        # Compute per-fingertip offsets
        fingertip_offsets = {}
        for finger in range(5):
            finger_glove = [s["glove"][finger] for s in self._samples if finger in s["glove"]]
            finger_camera = [s["camera"][finger] for s in self._samples if finger in s["camera"]]
            
            if finger_glove and finger_camera:
                # Transform glove points
                glove_arr = np.array(finger_glove)
                transformed = (R @ glove_arr.T).T + t
                
                camera_arr = np.array(finger_camera)
                offset = np.mean(camera_arr - transformed, axis=0)
                fingertip_offsets[finger] = tuple(offset)
        
        # Compute errors
        all_errors = []
        for sample in self._samples:
            for finger in range(5):
                if finger in sample["glove"] and finger in sample["camera"]:
                    glove_pt = np.array(sample["glove"][finger])
                    camera_pt = np.array(sample["camera"][finger])
                    
                    transformed = R @ glove_pt + t
                    if finger in fingertip_offsets:
                        transformed += np.array(fingertip_offsets[finger])
                    
                    error = np.linalg.norm(camera_pt - transformed) * 1000  # mm
                    all_errors.append(error)
        
        cal = CrossCalibrationData(
            rotation_quaternion=quat,
            translation_xyz=tuple(t),
            fingertip_offsets=fingertip_offsets,
            mean_position_error_mm=np.mean(all_errors),
            max_position_error_mm=np.max(all_errors),
            correlation_coefficient=float(np.corrcoef(
                glove_points.flatten(), camera_points.flatten()
            )[0, 1]),
        )
        
        self._calibration = cal
        return cal


# =============================================================================
# Main Calibration Service
# =============================================================================

class DOGloveCalibrationService:
    """
    Main calibration microservice for DOGlove wireless conversion.
    
    Provides complete calibration workflow:
    1. IMU calibration
    2. Joint encoder calibration
    3. Haptic system calibration
    4. Wireless link calibration
    5. Cross-calibration with MMPose
    6. Validation and reporting
    """
    
    def __init__(self, config_dir: str = "/etc/dynamical/doglove_calibration"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Calibrators
        self.imu_calibrator = IMUCalibrator()
        self.joint_calibrator = JointEncoderCalibrator()
        self.haptic_calibrator = HapticCalibrator()
        self.wireless_calibrator = WirelessCalibrator()
        self.cross_calibrator = CrossCalibrator()
        
        # State
        self._current_phase = CalibrationPhase.NOT_STARTED
        self._current_result: Optional[FullCalibrationResult] = None
        self._glove_serial: str = ""
        self._hand: Hand = Hand.RIGHT
    
    @property
    def phase(self) -> CalibrationPhase:
        """Get current calibration phase."""
        return self._current_phase
    
    def start_calibration(
        self,
        glove_serial: str,
        hand: Hand = Hand.RIGHT,
        wireless_protocol: WirelessProtocol = WirelessProtocol.BLE
    ) -> str:
        """
        Start a new calibration session.
        
        Args:
            glove_serial: DOGlove serial number
            hand: Left or right hand
            wireless_protocol: Target wireless protocol
        
        Returns:
            Calibration ID
        """
        self._glove_serial = glove_serial
        self._hand = hand
        self.wireless_calibrator = WirelessCalibrator(wireless_protocol)
        
        calibration_id = f"doglove_cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._current_result = FullCalibrationResult(
            calibration_id=calibration_id,
            glove_serial=glove_serial,
            hand=hand,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        self._current_phase = CalibrationPhase.IMU_CALIBRATION
        
        return calibration_id
    
    def get_calibration_instructions(self) -> Dict[str, Any]:
        """
        Get instructions for current calibration phase.
        
        Returns:
            Instructions and expected actions
        """
        instructions = {
            CalibrationPhase.NOT_STARTED: {
                "action": "Call start_calibration() to begin",
                "steps": [],
            },
            CalibrationPhase.IMU_CALIBRATION: {
                "action": "Calibrate IMU (accelerometer, gyroscope, magnetometer)",
                "steps": [
                    "1. Hold glove stationary for gyroscope bias (5 seconds)",
                    "2. Rotate to 6 orientations for accelerometer (X/Y/Z up/down)",
                    "3. Perform figure-8 motion for magnetometer (10 seconds)",
                ],
                "poses": [
                    {"name": "pose_x_up", "description": "Palm facing left, fingers up"},
                    {"name": "pose_x_down", "description": "Palm facing right, fingers up"},
                    {"name": "pose_y_up", "description": "Back of hand up, palm forward"},
                    {"name": "pose_y_down", "description": "Palm up, back of hand forward"},
                    {"name": "pose_z_up", "description": "Normal pose, palm down"},
                    {"name": "pose_z_down", "description": "Palm up"},
                ],
            },
            CalibrationPhase.JOINT_CALIBRATION: {
                "action": "Calibrate joint encoders",
                "steps": [
                    "1. Full hand extension (all fingers straight)",
                    "2. Full hand flexion (make tight fist)",
                    "3. Thumb-only flexion",
                    "4. Individual finger flexion (index, middle, ring, pinky)",
                    "5. Gradual open-close for hysteresis measurement",
                ],
                "target_accuracy": f"±{DOGLOVE_SPECS['calibrated_error_deg']}°",
            },
            CalibrationPhase.HAPTIC_CALIBRATION: {
                "action": "Calibrate haptic feedback system",
                "steps": [
                    "1. Zero-force baseline (no contact)",
                    "2. Apply known forces to each fingertip (10g, 50g, 100g, 200g)",
                    "3. Test servo motor response range",
                    "4. Calibrate LRA vibration intensity",
                ],
                "thresholds": {
                    "detection": f"{DOGLOVE_SPECS['force_threshold_1']}g",
                    "feedback": f"{DOGLOVE_SPECS['force_threshold_2']}g",
                    "force_only": f"{DOGLOVE_SPECS['force_threshold_3']}g",
                },
            },
            CalibrationPhase.WIRELESS_CALIBRATION: {
                "action": "Calibrate wireless communication",
                "steps": [
                    "1. Test latency (100 packets)",
                    "2. Test packet loss (1000 packets)",
                    "3. Test throughput (5 seconds)",
                    "4. Optimize connection parameters",
                ],
                "targets": {
                    "latency": "<20ms",
                    "packet_loss": "<1%",
                    "throughput": ">10 kbps",
                },
            },
            CalibrationPhase.CROSS_CALIBRATION: {
                "action": "Cross-calibrate with MMPose camera system",
                "steps": [
                    "1. Stand in camera view wearing calibrated glove",
                    "2. Perform 10 different hand poses",
                    "3. Record simultaneous glove and camera fingertip positions",
                    "4. Compute alignment transform",
                ],
                "requirements": [
                    "MMPose camera system must be calibrated",
                    "Good visibility of all fingertips",
                    "Stable lighting conditions",
                ],
            },
            CalibrationPhase.VALIDATION: {
                "action": "Validate calibration accuracy",
                "steps": [
                    "1. Compare glove angles with reference goniometer",
                    "2. Test haptic feedback response",
                    "3. Verify wireless reliability",
                    "4. Generate calibration report",
                ],
            },
            CalibrationPhase.COMPLETED: {
                "action": "Calibration complete",
                "steps": ["Review calibration report", "Save and export calibration"],
            },
            CalibrationPhase.FAILED: {
                "action": "Calibration failed",
                "steps": ["Review error messages", "Restart calibration"],
            },
        }
        
        return instructions.get(self._current_phase, {"action": "Unknown phase"})
    
    async def run_imu_calibration(
        self,
        accel_poses: Dict[str, List[Tuple[float, float, float]]],
        gyro_samples: List[Tuple[float, float, float]],
        mag_samples: List[Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """
        Run IMU calibration phase.
        
        Args:
            accel_poses: Dict of pose_name -> list of (ax, ay, az) samples
            gyro_samples: Stationary gyroscope samples
            mag_samples: Figure-8 motion magnetometer samples
        
        Returns:
            Calibration result
        """
        self._current_phase = CalibrationPhase.IMU_CALIBRATION
        
        # Collect accelerometer poses
        for pose_name, samples in accel_poses.items():
            self.imu_calibrator.collect_accelerometer_pose(pose_name, samples)
        
        # Collect gyroscope data
        self.imu_calibrator.collect_gyroscope_stationary(gyro_samples)
        
        # Collect magnetometer data
        self.imu_calibrator.collect_magnetometer_motion(mag_samples)
        
        # Compute calibration
        imu_cal = self.imu_calibrator.compute_calibration()
        self._current_result.imu = imu_cal
        
        # Advance to next phase
        self._current_phase = CalibrationPhase.JOINT_CALIBRATION
        
        return {
            "status": "completed",
            "phase": "imu",
            "next_phase": "joint",
            "results": asdict(imu_cal),
        }
    
    def run_joint_calibration(
        self,
        poses: Dict[str, Dict[Tuple[int, int], int]],
        reference_angles: Dict[str, Dict[Tuple[int, int], float]] = None
    ) -> Dict[str, Any]:
        """
        Run joint encoder calibration phase.
        
        Args:
            poses: Dict of pose_name -> {(finger, joint): raw_value}
            reference_angles: Optional ground truth angles
        
        Returns:
            Calibration result
        """
        self._current_phase = CalibrationPhase.JOINT_CALIBRATION
        
        # Collect samples for each pose
        for pose_name, readings in poses.items():
            refs = reference_angles.get(pose_name) if reference_angles else None
            self.joint_calibrator.collect_pose_sample(pose_name, readings, refs)
        
        # Compute all joint calibrations
        joint_cals = self.joint_calibrator.compute_all_calibrations()
        self._current_result.joints = joint_cals
        
        # Calculate overall accuracy
        errors = [j.linearity_error for j in joint_cals if j.linearity_error > 0]
        self._current_result.overall_joint_accuracy_deg = statistics.mean(errors) if errors else 0
        
        # Check if accuracy meets target
        if self._current_result.overall_joint_accuracy_deg > DOGLOVE_SPECS["calibrated_error_deg"]:
            self._current_result.issues.append(
                f"Joint accuracy {self._current_result.overall_joint_accuracy_deg:.2f}° exceeds target {DOGLOVE_SPECS['calibrated_error_deg']}°"
            )
        
        # Advance phase
        self._current_phase = CalibrationPhase.HAPTIC_CALIBRATION
        
        return {
            "status": "completed",
            "phase": "joint",
            "next_phase": "haptic",
            "num_joints_calibrated": len(joint_cals),
            "overall_accuracy_deg": self._current_result.overall_joint_accuracy_deg,
        }
    
    def run_haptic_calibration(
        self,
        force_samples: Dict[int, List[Tuple[float, int, int]]]
    ) -> Dict[str, Any]:
        """
        Run haptic system calibration.
        
        Args:
            force_samples: Dict of finger -> list of (force_grams, sensor_raw, pwm)
        
        Returns:
            Calibration result
        """
        self._current_phase = CalibrationPhase.HAPTIC_CALIBRATION
        
        # Collect samples
        for finger, samples in force_samples.items():
            for force, sensor, pwm in samples:
                self.haptic_calibrator.collect_force_sample(finger, force, sensor, pwm)
        
        # Compute calibrations
        haptic_cals = self.haptic_calibrator.compute_all_calibrations()
        self._current_result.haptics = haptic_cals
        
        # Advance phase
        self._current_phase = CalibrationPhase.WIRELESS_CALIBRATION
        
        return {
            "status": "completed",
            "phase": "haptic",
            "next_phase": "wireless",
            "num_fingers_calibrated": len(haptic_cals),
        }
    
    async def run_wireless_calibration(
        self,
        device_address: str
    ) -> Dict[str, Any]:
        """
        Run wireless link calibration.
        
        Args:
            device_address: BLE/WiFi address of glove
        
        Returns:
            Calibration result
        """
        self._current_phase = CalibrationPhase.WIRELESS_CALIBRATION
        
        # Run tests
        latency_result = await self.wireless_calibrator.test_latency(device_address)
        loss_result = await self.wireless_calibrator.test_packet_loss(device_address)
        throughput_result = await self.wireless_calibrator.test_throughput(device_address)
        
        # Optimize parameters
        optimization = self.wireless_calibrator.optimize_parameters()
        
        # Store calibration
        wireless_cal = self.wireless_calibrator.get_calibration()
        wireless_cal.device_address = device_address
        self._current_result.wireless = wireless_cal
        
        # Calculate reliability
        reliability = 100 - wireless_cal.packet_loss_percent
        self._current_result.wireless_reliability_percent = reliability
        
        # Check targets
        if wireless_cal.latency_mean_ms > 20:
            self._current_result.issues.append(
                f"Wireless latency {wireless_cal.latency_mean_ms:.1f}ms exceeds 20ms target"
            )
        
        if wireless_cal.packet_loss_percent > 1:
            self._current_result.issues.append(
                f"Packet loss {wireless_cal.packet_loss_percent:.2f}% exceeds 1% target"
            )
        
        # Advance phase
        self._current_phase = CalibrationPhase.CROSS_CALIBRATION
        
        return {
            "status": "completed",
            "phase": "wireless",
            "next_phase": "cross_calibration",
            "latency": latency_result,
            "packet_loss": loss_result,
            "throughput": throughput_result,
            "optimization": optimization,
        }
    
    def run_cross_calibration(
        self,
        samples: List[Dict[str, Dict[int, Tuple[float, float, float]]]]
    ) -> Dict[str, Any]:
        """
        Run cross-calibration with MMPose.
        
        Args:
            samples: List of {"glove": {finger: (x,y,z)}, "camera": {finger: (x,y,z)}}
        
        Returns:
            Calibration result
        """
        self._current_phase = CalibrationPhase.CROSS_CALIBRATION
        
        # Collect samples
        for sample in samples:
            self.cross_calibrator.collect_alignment_sample(
                sample.get("glove", {}),
                sample.get("camera", {})
            )
        
        # Compute calibration
        cross_cal = self.cross_calibrator.compute_cross_calibration()
        self._current_result.cross_calibration = cross_cal
        
        # Check accuracy
        if cross_cal.mean_position_error_mm > 10:
            self._current_result.issues.append(
                f"Cross-calibration error {cross_cal.mean_position_error_mm:.1f}mm exceeds 10mm target"
            )
        
        # Advance phase
        self._current_phase = CalibrationPhase.VALIDATION
        
        return {
            "status": "completed",
            "phase": "cross_calibration",
            "next_phase": "validation",
            "mean_error_mm": cross_cal.mean_position_error_mm,
            "max_error_mm": cross_cal.max_position_error_mm,
            "correlation": cross_cal.correlation_coefficient,
        }
    
    def finalize_calibration(self) -> FullCalibrationResult:
        """
        Finalize and save calibration.
        
        Returns:
            Complete calibration result
        """
        self._current_phase = CalibrationPhase.VALIDATION
        
        # Generate recommendations
        result = self._current_result
        
        if result.overall_joint_accuracy_deg > 2.0:
            result.recommendations.append(
                "Consider re-running joint calibration with reference angles"
            )
        
        if result.wireless.latency_mean_ms > 15:
            result.recommendations.append(
                "Consider switching to WiFi UDP for lower latency"
            )
        
        if result.wireless_reliability_percent < 99:
            result.recommendations.append(
                "Increase TX power or reduce distance for better reliability"
            )
        
        # Set final status
        if len(result.issues) == 0:
            result.phase = CalibrationPhase.COMPLETED
        else:
            result.phase = CalibrationPhase.COMPLETED  # Completed with issues
        
        self._current_phase = result.phase
        
        # Save calibration
        self._save_calibration(result)
        
        return result
    
    def _save_calibration(self, result: FullCalibrationResult):
        """Save calibration to file."""
        path = self.config_dir / f"{result.calibration_id}.json"
        
        # Convert to serializable format
        data = asdict(result)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved calibration to {path}")
    
    def load_calibration(self, calibration_id: str) -> Optional[FullCalibrationResult]:
        """Load calibration from file."""
        path = self.config_dir / f"{calibration_id}.json"
        
        if not path.exists():
            return None
        
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct result
        result = FullCalibrationResult(
            calibration_id=data["calibration_id"],
            glove_serial=data["glove_serial"],
            hand=Hand(data["hand"]),
            timestamp=data["timestamp"],
        )
        
        # Reconstruct sub-calibrations
        if "imu" in data:
            result.imu = IMUCalibrationData(**data["imu"])
        
        if "joints" in data:
            result.joints = [JointCalibrationData(**j) for j in data["joints"]]
        
        if "haptics" in data:
            result.haptics = [HapticCalibrationData(**h) for h in data["haptics"]]
        
        if "wireless" in data:
            result.wireless = WirelessCalibrationData(**data["wireless"])
        
        result.phase = CalibrationPhase(data.get("phase", "completed"))
        result.issues = data.get("issues", [])
        result.recommendations = data.get("recommendations", [])
        
        return result
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Generate human-readable calibration report."""
        if not self._current_result:
            return {"error": "No calibration performed"}
        
        result = self._current_result
        
        return {
            "calibration_id": result.calibration_id,
            "glove_serial": result.glove_serial,
            "hand": result.hand.value,
            "timestamp": result.timestamp,
            "status": result.phase.value,
            "summary": {
                "joint_accuracy": f"±{result.overall_joint_accuracy_deg:.2f}°",
                "target_accuracy": f"±{DOGLOVE_SPECS['calibrated_error_deg']}°",
                "wireless_latency": f"{result.wireless.latency_mean_ms:.1f}ms",
                "wireless_reliability": f"{result.wireless_reliability_percent:.1f}%",
                "cross_calibration_error": f"{result.cross_calibration.mean_position_error_mm:.1f}mm" if result.cross_calibration else "N/A",
            },
            "imu_quality": {
                "gyro_noise": f"{result.imu.gyro_noise_std:.4f}°/s",
                "accel_noise": f"{result.imu.accel_noise_std:.4f}m/s²",
                "mag_noise": f"{result.imu.mag_noise_std:.2f}",
            },
            "joints_calibrated": len(result.joints),
            "haptics_calibrated": len(result.haptics),
            "issues": result.issues,
            "recommendations": result.recommendations,
            "ready_for_wireless": len(result.issues) == 0,
        }


# =============================================================================
# FastAPI Service
# =============================================================================

def create_fastapi_app():
    """Create FastAPI application for DOGlove calibration service."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Dict
    
    app = FastAPI(
        title="DOGlove Wireless Calibration Service",
        description="Calibration microservice for converting DOGlove to wireless operation",
        version=__version__,
    )
    
    service = DOGloveCalibrationService()
    
    class StartCalibrationRequest(BaseModel):
        glove_serial: str
        hand: str = "right"
        wireless_protocol: str = "ble"
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": __version__}
    
    @app.get("/status")
    async def get_status():
        return {
            "phase": service.phase.value,
            "glove_serial": service._glove_serial,
            "hand": service._hand.value if service._hand else None,
        }
    
    @app.post("/calibration/start")
    async def start_calibration(request: StartCalibrationRequest):
        cal_id = service.start_calibration(
            glove_serial=request.glove_serial,
            hand=Hand(request.hand),
            wireless_protocol=WirelessProtocol(request.wireless_protocol),
        )
        return {"calibration_id": cal_id, "phase": service.phase.value}
    
    @app.get("/calibration/instructions")
    async def get_instructions():
        return service.get_calibration_instructions()
    
    @app.get("/calibration/report")
    async def get_report():
        return service.get_calibration_report()
    
    @app.post("/calibration/finalize")
    async def finalize():
        result = service.finalize_calibration()
        return {
            "calibration_id": result.calibration_id,
            "status": result.phase.value,
            "issues": result.issues,
            "recommendations": result.recommendations,
        }
    
    return app


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DOGlove Wireless Calibration Service")
    parser.add_argument("command", choices=["serve", "instructions", "report", "simulate"])
    parser.add_argument("--port", type=int, default=8092, help="Service port")
    parser.add_argument("--serial", default="DOGLOVE-001", help="Glove serial number")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        import uvicorn
        app = create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    elif args.command == "instructions":
        service = DOGloveCalibrationService()
        service.start_calibration(args.serial, Hand(args.hand))
        
        instructions = service.get_calibration_instructions()
        print(f"\nCalibration Phase: {service.phase.value}")
        print("=" * 60)
        print(f"Action: {instructions['action']}")
        print("\nSteps:")
        for step in instructions.get("steps", []):
            print(f"  {step}")
    
    elif args.command == "report":
        service = DOGloveCalibrationService()
        # Load most recent calibration
        import glob
        files = glob.glob(str(service.config_dir / "doglove_cal_*.json"))
        if files:
            latest = max(files)
            cal_id = Path(latest).stem
            result = service.load_calibration(cal_id)
            if result:
                service._current_result = result
                report = service.get_calibration_report()
                print(json.dumps(report, indent=2))
            else:
                print("No calibration found")
        else:
            print("No calibrations found")
    
    elif args.command == "simulate":
        print("Running simulated calibration...")
        service = DOGloveCalibrationService()
        
        # Start calibration
        cal_id = service.start_calibration(args.serial, Hand(args.hand))
        print(f"Started calibration: {cal_id}")
        
        # Simulate IMU calibration
        print("\nSimulating IMU calibration...")
        accel_poses = {
            f"pose_{axis}_{dir}": [(9.81 if axis == 'z' and dir == 'up' else 0, 
                                    9.81 if axis == 'y' and dir == 'up' else 0,
                                    9.81 if axis == 'x' and dir == 'up' else 0) + 
                                   tuple(np.random.normal(0, 0.1, 3)) for _ in range(100)]
            for axis in ['x', 'y', 'z'] for dir in ['up', 'down']
        }
        gyro_samples = [tuple(np.random.normal(0, 0.5, 3)) for _ in range(500)]
        mag_samples = [tuple(np.random.normal(0, 100, 3)) for _ in range(1000)]
        
        import asyncio
        asyncio.run(service.run_imu_calibration(accel_poses, gyro_samples, mag_samples))
        
        # Simulate joint calibration
        print("Simulating joint calibration...")
        poses = {
            "full_extension": {(f, j): 500 for f in range(5) for j in range(3)},
            "full_flexion": {(f, j): 3500 for f in range(5) for j in range(3)},
        }
        service.run_joint_calibration(poses)
        
        # Simulate haptic calibration
        print("Simulating haptic calibration...")
        force_samples = {
            f: [(g, int(g * 10), 1500 + int(g * 2)) for g in [0, 10, 50, 100, 200]]
            for f in range(5)
        }
        service.run_haptic_calibration(force_samples)
        
        # Simulate wireless calibration
        print("Simulating wireless calibration...")
        asyncio.run(service.run_wireless_calibration("AA:BB:CC:DD:EE:FF"))
        
        # Finalize
        result = service.finalize_calibration()
        
        print("\n" + "=" * 60)
        print("Calibration Complete!")
        print(json.dumps(service.get_calibration_report(), indent=2))


if __name__ == "__main__":
    main()

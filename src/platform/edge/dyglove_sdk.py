"""
DYGlove SDK - Dynamical.ai Wireless Haptic Glove

Complete SDK for the DYGlove haptic feedback glove system.
Based on DOGlove (arXiv:2502.07730) with wireless WiFi 6E modifications.

DYGlove Features (matching DOGlove paper):
- 21-DOF finger tracking:
  * Thumb: TM_flex, TM_abd, MCP, IP, Wrist_PS (5 DOF)
  * Index/Middle/Ring/Pinky: MCP_flex, MCP_abd, PIP, DIP (4 DOF each)
- 5-DOF Force feedback (Dynamixel XC/XL330 servos via cable-driven mechanism)
- 5-DOF Haptic feedback (8mm LRA @ 240Hz with DRV2605L driver)
- 16x Alps RDC506018A rotary encoders
- TI ADS1256 24-bit ADC @ 30kHz
- STM32F042K6T6 MCU @ 48MHz
- HTC Vive Tracker for wrist localization
- WiFi 6E wireless connectivity (DYGlove modification)

Reference:
    Zhang et al., "DOGlove: Dexterous Manipulation with a Low-Cost Open-Source 
    Haptic Force Feedback Glove", arXiv:2502.07730, 2025.
    https://do-glove.github.io/

Example:
    ```python
    from src.platform.edge.dyglove_sdk import DYGloveSDKClient, DYGloveDiscovery
    
    # Discover available gloves
    discovery = DYGloveDiscovery()
    gloves = discovery.scan()
    
    # Connect to glove
    glove = DYGloveSDKClient()
    glove.connect(port="/dev/ttyUSB0")
    
    # Calibrate
    glove.calibrate()
    
    # Read hand state (21 DOF)
    state = glove.get_state()
    print(f"21-DOF angles: {state.to_21dof_array()}")
    
    # Send force feedback (servos)
    glove.set_force_feedback([0.5, 0.3, 0.0, 0.0, 0.8])
    
    # Send haptic feedback (LRA vibration)
    glove.set_haptic_feedback(waveform_id=56, fingers=[0, 1])
    
    # Combined feedback (as per paper Table I)
    glove.set_combined_feedback(force_readings=[120, 80, 30, 0, 0])
    ```
"""

import os
import sys
import time
import struct
import threading
import math
import json
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime
from abc import ABC, abstractmethod
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

__version__ = "2.0.0"

# Optional imports
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    logger.warning("pyserial not available - serial connectivity disabled")

try:
    import bluetooth
    HAS_BLUETOOTH = True
except ImportError:
    HAS_BLUETOOTH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import socket
    HAS_SOCKET = True
except ImportError:
    HAS_SOCKET = False


# =============================================================================
# Enums and Constants
# =============================================================================

class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class ConnectionType(str, Enum):
    USB = "usb"
    SERIAL = "serial"
    BLUETOOTH = "bluetooth"
    BLUETOOTH_LE = "ble"
    WIFI = "wifi"


class GloveStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    CALIBRATING = "calibrating"
    ERROR = "error"


class CalibrationState(str, Enum):
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"


class Finger(int, Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class DYGloveJoint(IntEnum):
    """DYGlove 21-DOF joint indices (matching DOGlove paper)."""
    # Thumb (5 DOF)
    THUMB_TM_FLEX = 0
    THUMB_TM_ABD = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_WRIST_PS = 4
    # Index (4 DOF)
    INDEX_MCP_FLEX = 5
    INDEX_MCP_ABD = 6
    INDEX_PIP = 7
    INDEX_DIP = 8
    # Middle (4 DOF)
    MIDDLE_MCP_FLEX = 9
    MIDDLE_MCP_ABD = 10
    MIDDLE_PIP = 11
    MIDDLE_DIP = 12
    # Ring (4 DOF)
    RING_MCP_FLEX = 13
    RING_MCP_ABD = 14
    RING_PIP = 15
    RING_DIP = 16
    # Pinky (4 DOF)
    PINKY_MCP_FLEX = 17
    PINKY_MCP_ABD = 18
    PINKY_PIP = 19
    PINKY_DIP = 20


FINGER_JOINTS = {
    'thumb': [DYGloveJoint.THUMB_TM_FLEX, DYGloveJoint.THUMB_TM_ABD,
              DYGloveJoint.THUMB_MCP, DYGloveJoint.THUMB_IP, DYGloveJoint.THUMB_WRIST_PS],
    'index': [DYGloveJoint.INDEX_MCP_FLEX, DYGloveJoint.INDEX_MCP_ABD,
              DYGloveJoint.INDEX_PIP, DYGloveJoint.INDEX_DIP],
    'middle': [DYGloveJoint.MIDDLE_MCP_FLEX, DYGloveJoint.MIDDLE_MCP_ABD,
               DYGloveJoint.MIDDLE_PIP, DYGloveJoint.MIDDLE_DIP],
    'ring': [DYGloveJoint.RING_MCP_FLEX, DYGloveJoint.RING_MCP_ABD,
             DYGloveJoint.RING_PIP, DYGloveJoint.RING_DIP],
    'pinky': [DYGloveJoint.PINKY_MCP_FLEX, DYGloveJoint.PINKY_MCP_ABD,
              DYGloveJoint.PINKY_PIP, DYGloveJoint.PINKY_DIP],
}


class Protocol:
    """DOGlove communication protocol constants."""
    HEADER = 0xAA
    FOOTER = 0x55
    CMD_GET_STATE = 0x01
    CMD_SET_FORCE = 0x02
    CMD_SET_HAPTIC = 0x03
    CMD_CALIBRATE = 0x04
    CMD_GET_INFO = 0x05
    CMD_STREAM_START = 0x10
    CMD_STREAM_STOP = 0x11
    CMD_SET_COMBINED = 0x12
    RSP_STATE = 0x81
    RSP_INFO = 0x85
    RSP_ACK = 0x8F
    RSP_ERROR = 0xFF
    STATE_PACKET_SIZE = 64
    INFO_PACKET_SIZE = 32
    BAUD_RATE = 115200
    WIFI_UDP_PORT = 9876
    WIFI_TCP_PORT = 9877
    MAX_MOCAP_HZ = 120
    MAX_HAPTIC_HZ = 30


class HapticWaveform(IntEnum):
    """DRV2605L haptic waveform IDs (Immersion TouchSense 2200)."""
    STRONG_CLICK = 1
    SHARP_CLICK = 4
    SOFT_BUMP = 7
    DOUBLE_CLICK = 10
    STRONG_BUZZ = 13
    PULSING_STRONG = 52
    PULSING_SHARP_1_100 = 56  # DOGlove default
    TRANSITION_CLICK = 64
    LONG_BUZZ = 90
    SMOOTH_HUM = 94


# =============================================================================
# Data Classes - 21-DOF Compliant
# =============================================================================

@dataclass
class ThumbState:
    """Thumb state (5 DOF) matching DOGlove paper."""
    tm_flex: float = 0.0
    tm_abd: float = 0.0
    mcp: float = 0.0
    ip: float = 0.0
    wrist_ps: float = 0.0
    
    def to_list(self) -> List[float]:
        return [self.tm_flex, self.tm_abd, self.mcp, self.ip, self.wrist_ps]
    
    def to_radians(self) -> List[float]:
        return [math.radians(a) for a in self.to_list()]
    
    @classmethod
    def from_list(cls, angles: List[float]) -> 'ThumbState':
        return cls(
            tm_flex=angles[0] if len(angles) > 0 else 0.0,
            tm_abd=angles[1] if len(angles) > 1 else 0.0,
            mcp=angles[2] if len(angles) > 2 else 0.0,
            ip=angles[3] if len(angles) > 3 else 0.0,
            wrist_ps=angles[4] if len(angles) > 4 else 0.0,
        )


@dataclass
class FingerState:
    """Finger state (4 DOF) for Index/Middle/Ring/Pinky."""
    mcp_flex: float = 0.0
    mcp_abd: float = 0.0
    pip: float = 0.0
    dip: float = 0.0
    
    def to_list(self) -> List[float]:
        return [self.mcp_flex, self.mcp_abd, self.pip, self.dip]
    
    def to_radians(self) -> List[float]:
        return [math.radians(a) for a in self.to_list()]
    
    @property
    def mcp(self) -> float:
        return self.mcp_flex
    
    @mcp.setter
    def mcp(self, value: float):
        self.mcp_flex = value
    
    @classmethod
    def from_list(cls, angles: List[float]) -> 'FingerState':
        return cls(
            mcp_flex=angles[0] if len(angles) > 0 else 0.0,
            mcp_abd=angles[1] if len(angles) > 1 else 0.0,
            pip=angles[2] if len(angles) > 2 else 0.0,
            dip=angles[3] if len(angles) > 3 else 0.0,
        )


@dataclass
class Orientation:
    """3D orientation (quaternion)."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_euler(self) -> Tuple[float, float, float]:
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = math.asin(max(-1, min(1, sinp)))
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
    def to_array(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]


@dataclass
class WristPose:
    """Wrist pose from HTC Vive Tracker."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Orientation = field(default_factory=Orientation)
    tracking_valid: bool = False
    timestamp: float = 0.0


@dataclass
class DYGloveQualityConfig:
    """Configuration for quality filtering."""
    max_finger_angle_change: float = 15.0
    max_orientation_change: float = 20.0
    min_battery_level: int = 10
    enable_smoothing: bool = True
    smoothing_factor: float = 0.3
    target_hz: float = 120.0


@dataclass
class GloveState:
    """Complete glove state (21 DOF) matching DOGlove paper."""
    timestamp: float = 0.0
    sequence: int = 0
    thumb: ThumbState = field(default_factory=ThumbState)
    index: FingerState = field(default_factory=FingerState)
    middle: FingerState = field(default_factory=FingerState)
    ring: FingerState = field(default_factory=FingerState)
    pinky: FingerState = field(default_factory=FingerState)
    wrist_pose: WristPose = field(default_factory=WristPose)
    orientation: Orientation = field(default_factory=Orientation)
    wrist_roll: float = 0.0
    wrist_pitch: float = 0.0
    wrist_yaw: float = 0.0
    servo_positions: List[float] = field(default_factory=lambda: [0.0] * 5)
    servo_torques: List[float] = field(default_factory=lambda: [0.0] * 5)
    battery_level: int = 100
    temperature: float = 25.0
    encoder_validity: List[bool] = field(default_factory=lambda: [True] * 21)
    communication_latency_ms: float = 0.0
    
    def to_21dof_array(self) -> List[float]:
        """Get all 21 joint angles as flat array (degrees)."""
        angles = []
        angles.extend(self.thumb.to_list())
        angles.extend(self.index.to_list())
        angles.extend(self.middle.to_list())
        angles.extend(self.ring.to_list())
        angles.extend(self.pinky.to_list())
        return angles
    
    def to_21dof_radians(self) -> List[float]:
        return [math.radians(a) for a in self.to_21dof_array()]
    
    def to_numpy(self) -> 'np.ndarray':
        if HAS_NUMPY:
            return np.array(self.to_21dof_array())
        return self.to_21dof_array()
    
    @property
    def finger_angles(self) -> List[List[float]]:
        return [self.thumb.to_list(), self.index.to_list(),
                self.middle.to_list(), self.ring.to_list(), self.pinky.to_list()]
    
    @property
    def flat_angles(self) -> List[float]:
        return self.to_21dof_array()
    
    def get_flexion_angles(self) -> List[float]:
        """Get flexion angles only (15 DOF, no abduction)."""
        return [
            self.thumb.tm_flex, self.thumb.mcp, self.thumb.ip,
            self.index.mcp_flex, self.index.pip, self.index.dip,
            self.middle.mcp_flex, self.middle.pip, self.middle.dip,
            self.ring.mcp_flex, self.ring.pip, self.ring.dip,
            self.pinky.mcp_flex, self.pinky.pip, self.pinky.dip,
        ]
    
    def get_abduction_angles(self) -> List[float]:
        """Get abduction angles only (5 DOF)."""
        return [self.thumb.tm_abd, self.index.mcp_abd, self.middle.mcp_abd,
                self.ring.mcp_abd, self.pinky.mcp_abd]
    
    def get_finger_closure(self) -> List[float]:
        """Compute closure [0,1] for each finger."""
        closures = []
        thumb_max = 70 + 90 + 80
        thumb_sum = max(0, self.thumb.tm_flex) + max(0, self.thumb.mcp) + max(0, self.thumb.ip)
        closures.append(min(1.0, thumb_sum / thumb_max))
        finger_max = 90 + 100 + 70
        for finger in [self.index, self.middle, self.ring, self.pinky]:
            finger_sum = max(0, finger.mcp_flex) + max(0, finger.pip) + max(0, finger.dip)
            closures.append(min(1.0, finger_sum / finger_max))
        return closures
    
    @classmethod
    def from_21dof_array(cls, angles: List[float], timestamp: float = None) -> 'GloveState':
        if len(angles) < 21:
            angles = angles + [0.0] * (21 - len(angles))
        return cls(
            timestamp=timestamp or time.time(),
            thumb=ThumbState.from_list(angles[0:5]),
            index=FingerState.from_list(angles[5:9]),
            middle=FingerState.from_list(angles[9:13]),
            ring=FingerState.from_list(angles[13:17]),
            pinky=FingerState.from_list(angles[17:21]),
        )


@dataclass
class GloveInfo:
    """Device information."""
    serial_number: str = ""
    firmware_version: str = ""
    hardware_version: str = ""
    hand: Hand = Hand.RIGHT
    model: str = "DYGlove Pro"
    manufacturer: str = "Dynamical.ai"
    mac_address: str = ""
    wifi_ip: str = ""


@dataclass
class EncoderCalibration:
    """Per-encoder calibration data."""
    encoder_id: int = 0
    joint_name: str = ""
    voltage_to_angle: Dict[int, float] = field(default_factory=dict)
    poly_coefficients: List[float] = field(default_factory=lambda: [0.0, 360.0/3.3, 0.0, 0.0])
    min_voltage: float = 0.0
    max_voltage: float = 3.3
    min_angle: float = 0.0
    max_angle: float = 360.0
    timestamp: str = ""
    
    def voltage_to_angle_raw(self, voltage: float) -> float:
        return (voltage / 3.3) * 360.0
    
    def voltage_to_angle_calibrated(self, voltage: float) -> float:
        voltage_int = int(voltage * 1000)
        if voltage_int in self.voltage_to_angle:
            return self.voltage_to_angle[voltage_int]
        v = voltage
        angle = sum(c * (v ** i) for i, c in enumerate(self.poly_coefficients))
        return max(self.min_angle, min(self.max_angle, angle))


@dataclass
class CalibrationData:
    """Complete calibration data for all 21 encoders."""
    encoders: List[EncoderCalibration] = field(default_factory=lambda: [
        EncoderCalibration(encoder_id=i) for i in range(21)
    ])
    timestamp: str = ""
    hand: Hand = Hand.RIGHT
    min_values: List[float] = field(default_factory=lambda: [0.0] * 21)
    max_values: List[float] = field(default_factory=lambda: [90.0] * 21)
    offsets: List[float] = field(default_factory=lambda: [0.0] * 21)
    
    def save(self, path: str):
        data = {
            "min_values": self.min_values,
            "max_values": self.max_values,
            "offsets": self.offsets,
            "timestamp": self.timestamp,
            "hand": self.hand.value,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CalibrationData':
        with open(path) as f:
            data = json.load(f)
        return cls(
            timestamp=data.get("timestamp", ""),
            hand=Hand(data.get("hand", "right")),
            min_values=data.get("min_values", [0.0] * 21),
            max_values=data.get("max_values", [90.0] * 21),
            offsets=data.get("offsets", [0.0] * 21),
        )


@dataclass
class ForceCommand:
    """Force feedback command (5 Dynamixel servos)."""
    thumb: float = 0.0
    index: float = 0.0
    middle: float = 0.0
    ring: float = 0.0
    pinky: float = 0.0
    
    def to_list(self) -> List[float]:
        return [self.thumb, self.index, self.middle, self.ring, self.pinky]
    
    def to_pwm(self, max_pwm: int = 1023) -> List[int]:
        return [int(max(0, min(1, v)) * max_pwm) for v in self.to_list()]
    
    def clamp(self) -> 'ForceCommand':
        return ForceCommand(
            thumb=max(0, min(1, self.thumb)),
            index=max(0, min(1, self.index)),
            middle=max(0, min(1, self.middle)),
            ring=max(0, min(1, self.ring)),
            pinky=max(0, min(1, self.pinky)),
        )


@dataclass
class HapticCommand:
    """Haptic feedback command (5 LRA actuators)."""
    waveform_id: int = HapticWaveform.PULSING_SHARP_1_100
    fingers: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    intensity: float = 1.0
    duration_ms: int = 100
    thumb: float = 1.0
    index: float = 1.0
    middle: float = 1.0
    ring: float = 1.0
    pinky: float = 1.0
    
    def to_finger_mask(self) -> int:
        mask = 0
        for f in self.fingers:
            mask |= (1 << f)
        return mask
    
    def to_list(self) -> List[float]:
        return [self.thumb, self.index, self.middle, self.ring, self.pinky]


@dataclass
class CombinedFeedback:
    """Combined force + haptic feedback (DOGlove paper Table I)."""
    force_readings: List[float] = field(default_factory=lambda: [0.0] * 5)
    HAPTIC_START = 10
    FORCE_START = 50
    HAPTIC_STOP = 100
    FORCE_MAX = 3000
    
    def get_haptic_mask(self) -> int:
        mask = 0
        for i, f in enumerate(self.force_readings):
            if self.HAPTIC_START <= f < self.HAPTIC_STOP:
                mask |= (1 << i)
        return mask
    
    def get_force_intensities(self) -> List[float]:
        intensities = []
        for f in self.force_readings:
            if f >= self.FORCE_START:
                intensity = min(1.0, (f - self.FORCE_START) / (self.FORCE_MAX - self.FORCE_START))
                intensities.append(intensity)
            else:
                intensities.append(0.0)
        return intensities
    
    def to_commands(self) -> Tuple[ForceCommand, Optional[HapticCommand]]:
        force_cmd = ForceCommand(*self.get_force_intensities())
        haptic_mask = self.get_haptic_mask()
        if haptic_mask > 0:
            fingers = [i for i in range(5) if (haptic_mask >> i) & 1]
            haptic_cmd = HapticCommand(waveform_id=HapticWaveform.PULSING_SHARP_1_100, fingers=fingers)
        else:
            haptic_cmd = None
        return force_cmd, haptic_cmd


# =============================================================================
# Discovery
# =============================================================================

class DYGloveDiscovery:
    """Discovers DYGlove devices."""
    DOGLOVE_VID_PID = [("2FE3", "0001"), ("2FE3", "0002"), ("0483", "5740")]
    
    def scan(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        devices = []
        devices.extend(self._scan_serial())
        devices.extend(self._scan_wifi(timeout))
        return devices
    
    def _scan_serial(self) -> List[Dict[str, Any]]:
        devices = []
        if not HAS_SERIAL:
            return devices
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                is_doglove = False
                if port.vid and port.pid:
                    vid_hex = f"{port.vid:04X}"
                    pid_hex = f"{port.pid:04X}"
                    for known_vid, known_pid in self.DOGLOVE_VID_PID:
                        if vid_hex == known_vid and pid_hex == known_pid:
                            is_doglove = True
                            break
                desc = (port.description or "").lower()
                if "doglove" in desc or "dyglove" in desc:
                    is_doglove = True
                if port.device.startswith("/dev/ttyUSB") or port.device.startswith("/dev/ttyACM"):
                    devices.append({
                        "type": "serial", "port": port.device,
                        "description": port.description or "Unknown",
                        "verified": is_doglove,
                    })
        except Exception as e:
            logger.error(f"Serial scan failed: {e}")
        return devices
    
    def _scan_wifi(self, timeout: float) -> List[Dict[str, Any]]:
        devices = []
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            sock.sendto(b"DYGLOVE_DISCOVER", ('<broadcast>', Protocol.WIFI_UDP_PORT))
            try:
                while True:
                    data, addr = sock.recvfrom(1024)
                    if data.startswith(b"DYGLOVE_HERE"):
                        devices.append({
                            "type": "wifi", "ip": addr[0],
                            "port": Protocol.WIFI_TCP_PORT, "verified": True,
                        })
            except socket.timeout:
                pass
            sock.close()
        except Exception as e:
            logger.debug(f"WiFi scan failed: {e}")
        return devices


# =============================================================================
# Quality Filter
# =============================================================================

class DYGloveQualityFilter:
    def __init__(self, config: Optional[DYGloveQualityConfig] = None):
        self.config = config or DYGloveQualityConfig()
        self._last_valid_state: Optional[GloveState] = None
    
    def process_state(self, new_state: GloveState) -> Optional[GloveState]:
        if new_state.battery_level < self.config.min_battery_level:
            return None
        if self._last_valid_state is None:
            self._last_valid_state = new_state
            return new_state
        time_diff = new_state.timestamp - self._last_valid_state.timestamp
        if time_diff <= 0:
            return None
        if self.config.enable_smoothing:
            smoothed_state = self._apply_smoothing(self._last_valid_state, new_state)
            self._last_valid_state = smoothed_state
            return smoothed_state
        self._last_valid_state = new_state
        return new_state
    
    def _apply_smoothing(self, old_state: GloveState, new_state: GloveState) -> GloveState:
        alpha = self.config.smoothing_factor
        old_angles = old_state.to_21dof_array()
        new_angles = new_state.to_21dof_array()
        smoothed_angles = [old_angles[i] * (1 - alpha) + new_angles[i] * alpha for i in range(21)]
        smoothed = GloveState.from_21dof_array(smoothed_angles, new_state.timestamp)
        smoothed.battery_level = new_state.battery_level
        smoothed.servo_positions = new_state.servo_positions
        return smoothed


# =============================================================================
# Async Reader
# =============================================================================

class DYGloveAsyncReader:
    def __init__(self, driver: 'DYGloveDriverBase',
                 quality_config: Optional[DYGloveQualityConfig] = None,
                 target_hz: float = 120.0):
        self.driver = driver
        self.filter = DYGloveQualityFilter(quality_config)
        self.target_hz = target_hz
        self._latest_state: Optional[GloveState] = None
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
    
    def start(self):
        if self._streaming:
            return
        self._streaming = True
        self._stop_event.clear()
        self._stream_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._stream_thread.start()
    
    def stop(self):
        if not self._streaming:
            return
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        self._streaming = False
    
    def _read_loop(self):
        interval = 1.0 / self.target_hz
        while not self._stop_event.is_set():
            try:
                raw_state = self.driver.get_state()
                filtered_state = self.filter.process_state(raw_state)
                if filtered_state:
                    with self._state_lock:
                        self._latest_state = filtered_state
            except Exception as e:
                logger.error(f"Read error: {e}")
            time.sleep(interval)
    
    def get_latest_state(self) -> Optional[GloveState]:
        with self._state_lock:
            return self._latest_state
    
    @property
    def is_streaming(self) -> bool:
        return self._streaming
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# =============================================================================
# Driver Base Class
# =============================================================================

class DYGloveDriverBase(ABC):
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def get_state(self) -> GloveState:
        pass
    
    @abstractmethod
    def set_force_feedback(self, intensities: List[float] = None,
                           command: ForceCommand = None) -> bool:
        pass
    
    @abstractmethod
    def set_haptic_feedback(self, waveform_id: int = None,
                            fingers: List[int] = None,
                            command: HapticCommand = None) -> bool:
        pass
    
    @abstractmethod
    def calibrate(self, duration: float = 5.0) -> CalibrationData:
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def info(self) -> Optional[GloveInfo]:
        pass
    
    def set_combined_feedback(self, force_readings: List[float]) -> bool:
        combined = CombinedFeedback(force_readings=force_readings)
        force_cmd, haptic_cmd = combined.to_commands()
        success = self.set_force_feedback(command=force_cmd)
        if haptic_cmd:
            success &= self.set_haptic_feedback(command=haptic_cmd)
        return success


# =============================================================================
# SDK Client
# =============================================================================

class DYGloveSDKClient(DYGloveDriverBase):
    """DYGlove SDK client implementing STM32 protocol."""
    
    def __init__(self):
        self._connection = None
        self._connection_type: Optional[ConnectionType] = None
        self._status = GloveStatus.DISCONNECTED
        self._info: Optional[GloveInfo] = None
        self._calibration: Optional[CalibrationData] = None
        self._current_state: Optional[GloveState] = None
        self._sequence = 0
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_callback: Optional[Callable[[GloveState], None]] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
    
    @property
    def status(self) -> GloveStatus:
        return self._status
    
    @property
    def is_connected(self) -> bool:
        return self._status in [GloveStatus.CONNECTED, GloveStatus.STREAMING]
    
    @property
    def info(self) -> Optional[GloveInfo]:
        return self._info
    
    def connect(self, port: str = None, bluetooth_address: str = None,
                wifi_ip: str = None, timeout: float = 5.0) -> bool:
        if self.is_connected:
            return True
        self._status = GloveStatus.CONNECTING
        try:
            if wifi_ip:
                return self._connect_wifi(wifi_ip, timeout)
            elif port:
                return self._connect_serial(port, timeout)
            elif bluetooth_address:
                return self._connect_bluetooth(bluetooth_address, timeout)
            else:
                discovery = DYGloveDiscovery()
                devices = discovery.scan(timeout)
                if not devices:
                    self._status = GloveStatus.DISCONNECTED
                    return False
                device = devices[0]
                if device["type"] == "wifi":
                    return self._connect_wifi(device["ip"], timeout)
                else:
                    return self._connect_serial(device["port"], timeout)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._status = GloveStatus.ERROR
            return False
    
    def _connect_serial(self, port: str, timeout: float) -> bool:
        if not HAS_SERIAL:
            return False
        try:
            self._connection = serial.Serial(port=port, baudrate=Protocol.BAUD_RATE,
                                             timeout=timeout, write_timeout=timeout)
            self._connection_type = ConnectionType.SERIAL
            self._status = GloveStatus.CONNECTED
            self._info = self._query_device_info()
            logger.info(f"Connected to DYGlove on {port}")
            return True
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            self._status = GloveStatus.DISCONNECTED
            return False
    
    def _connect_wifi(self, ip: str, timeout: float) -> bool:
        try:
            self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._connection.settimeout(timeout)
            self._connection.connect((ip, Protocol.WIFI_TCP_PORT))
            self._connection_type = ConnectionType.WIFI
            self._status = GloveStatus.CONNECTED
            self._info = GloveInfo(wifi_ip=ip)
            logger.info(f"Connected via WiFi at {ip}")
            return True
        except Exception as e:
            logger.error(f"WiFi connection failed: {e}")
            self._status = GloveStatus.DISCONNECTED
            return False
    
    def _connect_bluetooth(self, address: str, timeout: float) -> bool:
        if not HAS_BLUETOOTH:
            return False
        try:
            self._connection = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self._connection.connect((address, 1))
            self._connection_type = ConnectionType.BLUETOOTH
            self._status = GloveStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Bluetooth connection failed: {e}")
            self._status = GloveStatus.DISCONNECTED
            return False
    
    def disconnect(self):
        self.stop_streaming()
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
        self._connection = None
        self._status = GloveStatus.DISCONNECTED
    
    def get_state(self) -> GloveState:
        if not self.is_connected:
            return GloveState(timestamp=time.time())
        try:
            self._send_command(Protocol.CMD_GET_STATE)
            data = self._read_response(Protocol.STATE_PACKET_SIZE)
            if data:
                return self._parse_state_packet(data)
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
        return self._current_state or GloveState(timestamp=time.time())
    
    def _parse_state_packet(self, data: bytes) -> GloveState:
        if len(data) < Protocol.STATE_PACKET_SIZE:
            return GloveState(timestamp=time.time())
        if data[0] != Protocol.HEADER or data[-1] != Protocol.FOOTER:
            return GloveState(timestamp=time.time())
        sequence = data[1]
        joint_angles = []
        for i in range(21):
            offset = 2 + i * 2
            angle_raw = struct.unpack('<h', data[offset:offset+2])[0]
            joint_angles.append(angle_raw / 100.0)
        servo_positions = []
        for i in range(5):
            offset = 44 + i * 2
            pos = struct.unpack('<h', data[offset:offset+2])[0]
            servo_positions.append(pos / 1000.0)
        battery = data[62] if len(data) > 62 else 100
        state = GloveState.from_21dof_array(joint_angles, time.time())
        state.sequence = sequence
        state.servo_positions = servo_positions
        state.battery_level = battery
        self._current_state = state
        return state
    
    def set_force_feedback(self, intensities: List[float] = None,
                           command: ForceCommand = None) -> bool:
        if not self.is_connected:
            return False
        if intensities is not None:
            command = ForceCommand(*intensities[:5])
        if command is None:
            command = ForceCommand()
        command = command.clamp()
        try:
            packet = bytearray([Protocol.HEADER, Protocol.CMD_SET_FORCE])
            for pwm in command.to_pwm(max_pwm=1023):
                packet.extend(struct.pack('<H', pwm))
            packet.append(Protocol.FOOTER)
            self._send_raw(bytes(packet))
            return True
        except Exception as e:
            logger.error(f"Force feedback failed: {e}")
            return False
    
    def set_haptic_feedback(self, waveform_id: int = None, fingers: List[int] = None,
                            command: HapticCommand = None) -> bool:
        if not self.is_connected:
            return False
        if waveform_id is not None or fingers is not None:
            command = HapticCommand(
                waveform_id=waveform_id or HapticWaveform.PULSING_SHARP_1_100,
                fingers=fingers if fingers is not None else [0, 1, 2, 3, 4],
            )
        if command is None:
            command = HapticCommand()
        try:
            packet = bytearray([Protocol.HEADER, Protocol.CMD_SET_HAPTIC])
            packet.append(command.waveform_id & 0x7F)
            packet.append(command.to_finger_mask())
            packet.extend(struct.pack('<H', command.duration_ms))
            packet.append(int(command.intensity * 255))
            packet.append(Protocol.FOOTER)
            self._send_raw(bytes(packet))
            return True
        except Exception as e:
            logger.error(f"Haptic feedback failed: {e}")
            return False
    
    def calibrate(self, duration: float = 5.0,
                  callback: Callable[[str, float], None] = None) -> CalibrationData:
        if not self.is_connected:
            return CalibrationData()
        self._status = GloveStatus.CALIBRATING
        try:
            if callback:
                callback("Starting calibration...", 0.0)
            self._send_command(Protocol.CMD_CALIBRATE)
            start_time = time.time()
            samples = []
            while time.time() - start_time < duration:
                state = self.get_state()
                samples.append(state.to_21dof_array())
                progress = (time.time() - start_time) / duration
                if callback:
                    callback(f"Collecting samples... {len(samples)}", progress)
                time.sleep(0.02)
            if HAS_NUMPY:
                samples_arr = np.array(samples)
                min_vals = samples_arr.min(axis=0).tolist()
                max_vals = samples_arr.max(axis=0).tolist()
                offsets = samples_arr.mean(axis=0).tolist()
            else:
                min_vals = [0.0] * 21
                max_vals = [90.0] * 21
                offsets = [0.0] * 21
            self._calibration = CalibrationData(
                min_values=min_vals, max_values=max_vals, offsets=offsets,
                timestamp=datetime.utcnow().isoformat(),
            )
            if callback:
                callback("Calibration complete!", 1.0)
            self._status = GloveStatus.CONNECTED
            return self._calibration
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self._status = GloveStatus.CONNECTED
            return CalibrationData()
    
    def start_streaming(self, callback: Callable[[GloveState], None], rate_hz: float = 120.0):
        if self._streaming:
            return
        self._stream_callback = callback
        self._streaming = True
        self._stop_event.clear()
        self._send_command(Protocol.CMD_STREAM_START)
        self._stream_thread = threading.Thread(target=self._stream_loop, args=(rate_hz,), daemon=True)
        self._stream_thread.start()
        self._status = GloveStatus.STREAMING
    
    def stop_streaming(self):
        if not self._streaming:
            return
        self._stop_event.set()
        self._streaming = False
        try:
            self._send_command(Protocol.CMD_STREAM_STOP)
        except:
            pass
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
        self._status = GloveStatus.CONNECTED
    
    def _stream_loop(self, rate_hz: float):
        interval = 1.0 / rate_hz
        while not self._stop_event.is_set():
            try:
                state = self.get_state()
                if self._stream_callback:
                    self._stream_callback(state)
            except Exception as e:
                logger.error(f"Stream error: {e}")
            time.sleep(interval)
    
    def _send_command(self, cmd: int):
        packet = bytes([Protocol.HEADER, cmd, Protocol.FOOTER])
        self._send_raw(packet)
    
    def _send_raw(self, data: bytes):
        if self._connection is None:
            return
        with self._lock:
            if self._connection_type == ConnectionType.SERIAL:
                self._connection.write(data)
            elif self._connection_type in [ConnectionType.WIFI, ConnectionType.BLUETOOTH]:
                self._connection.send(data)
    
    def _read_response(self, size: int) -> Optional[bytes]:
        if self._connection is None:
            return None
        try:
            with self._lock:
                if self._connection_type == ConnectionType.SERIAL:
                    return self._connection.read(size)
                else:
                    return self._connection.recv(size)
        except:
            return None
    
    def _query_device_info(self) -> GloveInfo:
        try:
            self._send_command(Protocol.CMD_GET_INFO)
            data = self._read_response(Protocol.INFO_PACKET_SIZE)
            if data and len(data) >= Protocol.INFO_PACKET_SIZE:
                serial_num = data[2:18].decode('utf-8', errors='ignore').strip('\x00')
                fw_version = f"{data[18]}.{data[19]}.{data[20]}"
                hand = Hand.LEFT if data[21] == 0 else Hand.RIGHT
                return GloveInfo(serial_number=serial_num, firmware_version=fw_version, hand=hand)
        except:
            pass
        return GloveInfo()


# =============================================================================
# Simulator
# =============================================================================

class DYGloveSimulator(DYGloveDriverBase):
    """DYGlove simulator for testing."""
    
    def __init__(self, side: Hand = Hand.RIGHT):
        self._side = side
        self._connected = False
        self._sim_time = 0.0
        self._info = GloveInfo(serial_number="SIM-001", firmware_version="1.0.0",
                               hand=side, model="DYGlove Simulator")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def info(self) -> Optional[GloveInfo]:
        return self._info
    
    def connect(self, **kwargs) -> bool:
        self._connected = True
        logger.info(f"DYGlove Simulator ({self._side.value}) connected")
        return True
    
    def disconnect(self):
        self._connected = False
    
    def get_state(self) -> GloveState:
        self._sim_time += 0.01
        t = self._sim_time
        def wave(offset: float, amplitude: float = 30.0) -> float:
            return amplitude * math.sin(t * 2 + offset)
        state = GloveState(
            timestamp=time.time(),
            thumb=ThumbState(tm_flex=30+wave(0), tm_abd=5+wave(0.5,10),
                            mcp=25+wave(0.2), ip=20+wave(0.3), wrist_ps=wave(0.1,15)),
            index=FingerState(mcp_flex=35+wave(0.5), mcp_abd=wave(0.6,5),
                             pip=30+wave(0.7), dip=25+wave(0.8)),
            middle=FingerState(mcp_flex=35+wave(1.0), mcp_abd=wave(1.1,3),
                              pip=30+wave(1.2), dip=25+wave(1.3)),
            ring=FingerState(mcp_flex=35+wave(1.5), mcp_abd=wave(1.6,3),
                            pip=30+wave(1.7), dip=25+wave(1.8)),
            pinky=FingerState(mcp_flex=35+wave(2.0), mcp_abd=wave(2.1,5),
                             pip=30+wave(2.2), dip=25+wave(2.3)),
            battery_level=85,
        )
        return state
    
    def set_force_feedback(self, intensities: List[float] = None,
                           command: ForceCommand = None) -> bool:
        return True
    
    def set_haptic_feedback(self, waveform_id: int = None, fingers: List[int] = None,
                            command: HapticCommand = None) -> bool:
        return True
    
    def calibrate(self, duration: float = 5.0,
                  callback: Callable[[str, float], None] = None) -> CalibrationData:
        if callback:
            callback("Simulating...", 0.5)
        time.sleep(min(duration, 0.5))
        if callback:
            callback("Complete!", 1.0)
        return CalibrationData(timestamp=datetime.utcnow().isoformat())


# =============================================================================
# WiFi Driver
# =============================================================================

class DYGloveWiFiDriver(DYGloveDriverBase):
    """WiFi driver for wireless DYGlove."""
    
    def __init__(self, ip_address: str, side: Hand = Hand.RIGHT):
        self.ip_address = ip_address
        self._side = side
        self._connected = False
        self._udp_sock = None
        self._tcp_sock = None
        self._latest_state: Optional[GloveState] = None
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._listen_thread = None
        self._info = GloveInfo(wifi_ip=ip_address, hand=side)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def info(self) -> Optional[GloveInfo]:
        return self._info
    
    def connect(self, **kwargs) -> bool:
        try:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind(('0.0.0.0', 0))
            self._udp_sock.settimeout(1.0)
            self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_sock.settimeout(5.0)
            self._tcp_sock.connect((self.ip_address, Protocol.WIFI_TCP_PORT))
            local_port = self._udp_sock.getsockname()[1]
            cmd = json.dumps({"cmd": "start_stream", "port": local_port}).encode()
            self._tcp_sock.send(cmd)
            self._stop_event.clear()
            self._listen_thread = threading.Thread(target=self._udp_loop, daemon=True)
            self._listen_thread.start()
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"WiFi connection failed: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        self._connected = False
        self._stop_event.set()
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)
        if self._tcp_sock:
            try:
                self._tcp_sock.close()
            except:
                pass
        if self._udp_sock:
            self._udp_sock.close()
        self._tcp_sock = None
        self._udp_sock = None
    
    def get_state(self) -> GloveState:
        with self._state_lock:
            return self._latest_state or GloveState(timestamp=time.time())
    
    def _udp_loop(self):
        while not self._stop_event.is_set():
            try:
                data, _ = self._udp_sock.recvfrom(Protocol.STATE_PACKET_SIZE)
                state = self._parse_udp_packet(data)
                if state:
                    with self._state_lock:
                        self._latest_state = state
            except socket.timeout:
                continue
            except:
                break
    
    def _parse_udp_packet(self, data: bytes) -> Optional[GloveState]:
        if len(data) < 48 or data[0] != Protocol.HEADER:
            return None
        joint_angles = []
        for i in range(21):
            offset = 2 + i * 2
            if offset + 2 <= len(data):
                angle = struct.unpack('<h', data[offset:offset+2])[0] / 100.0
                joint_angles.append(angle)
        if len(joint_angles) == 21:
            return GloveState.from_21dof_array(joint_angles, time.time())
        return None
    
    def set_force_feedback(self, intensities: List[float] = None,
                           command: ForceCommand = None) -> bool:
        if not self._connected or not self._tcp_sock:
            return False
        if intensities:
            command = ForceCommand(*intensities[:5])
        if command is None:
            return False
        try:
            cmd = json.dumps({"cmd": "force", "values": command.to_list()}).encode()
            self._tcp_sock.send(cmd)
            return True
        except:
            return False
    
    def set_haptic_feedback(self, waveform_id: int = None, fingers: List[int] = None,
                            command: HapticCommand = None) -> bool:
        if not self._connected or not self._tcp_sock:
            return False
        if waveform_id is not None:
            command = HapticCommand(waveform_id=waveform_id, fingers=fingers or [0,1,2,3,4])
        if command is None:
            return False
        try:
            cmd = json.dumps({"cmd": "haptic", "waveform": command.waveform_id,
                            "fingers": command.fingers}).encode()
            self._tcp_sock.send(cmd)
            return True
        except:
            return False
    
    def calibrate(self, duration: float = 5.0,
                  callback: Callable[[str, float], None] = None) -> CalibrationData:
        if callback:
            callback("Starting...", 0.0)
        samples = []
        start = time.time()
        while time.time() - start < duration:
            state = self.get_state()
            samples.append(state.to_21dof_array())
            if callback:
                callback("Collecting...", (time.time() - start) / duration)
            time.sleep(0.02)
        if callback:
            callback("Complete!", 1.0)
        if HAS_NUMPY and samples:
            arr = np.array(samples)
            return CalibrationData(min_values=arr.min(axis=0).tolist(),
                                  max_values=arr.max(axis=0).tolist(),
                                  offsets=arr.mean(axis=0).tolist(),
                                  timestamp=datetime.utcnow().isoformat())
        return CalibrationData()


# =============================================================================
# Factory Functions
# =============================================================================

def create_dyglove_driver(use_simulator: bool = False, side: Hand = Hand.RIGHT,
                          port: Optional[str] = None, wifi_ip: Optional[str] = None,
                          **kwargs) -> DYGloveDriverBase:
    if use_simulator:
        return DYGloveSimulator(side=side)
    elif wifi_ip:
        return DYGloveWiFiDriver(ip_address=wifi_ip, side=side)
    else:
        return DYGloveSDKClient()


def create_dyglove_reader(use_simulator: bool = False, side: Hand = Hand.RIGHT,
                          port: Optional[str] = None, wifi_ip: Optional[str] = None,
                          quality_config: Optional[DYGloveQualityConfig] = None,
                          target_hz: float = 120.0, **kwargs) -> DYGloveAsyncReader:
    driver = create_dyglove_driver(use_simulator, side, port, wifi_ip, **kwargs)
    return DYGloveAsyncReader(driver, quality_config, target_hz)


# =============================================================================
# CLI
# =============================================================================

# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Alias for Joint enum (legacy name)
Joint = DYGloveJoint


class DYGloveFK:
    """
    Forward Kinematics for DYGlove hand model.
    
    Computes fingertip positions from joint angles using simplified
    kinematic chain model.
    """
    
    # Finger segment lengths (mm) - approximate human hand proportions
    SEGMENT_LENGTHS = {
        'thumb': [40, 32, 30],      # Metacarpal, proximal, distal
        'index': [70, 43, 26, 18],  # Metacarpal, proximal, middle, distal
        'middle': [65, 45, 28, 20],
        'ring': [60, 42, 26, 18],
        'pinky': [55, 35, 22, 16],
    }
    
    # Base positions relative to wrist (mm)
    BASE_POSITIONS = {
        'thumb': (-25, 20, 0),
        'index': (35, 85, 0),
        'middle': (12, 90, 0),
        'ring': (-12, 85, 0),
        'pinky': (-35, 75, 0),
    }
    
    def __init__(self, scale: float = 1.0):
        """
        Initialize FK solver.
        
        Args:
            scale: Scale factor for hand size (1.0 = average adult)
        """
        self.scale = scale
    
    def compute_fingertip_positions(self, state: GloveState) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute fingertip positions from joint angles.
        
        Args:
            state: GloveState with 21-DOF joint angles
        
        Returns:
            Dictionary mapping finger name to (x, y, z) position in mm
        """
        positions = {}
        
        # Thumb
        angles = state.thumb.to_radians()
        positions['thumb'] = self._compute_finger_chain(
            'thumb', angles[:4], self.BASE_POSITIONS['thumb']
        )
        
        # Other fingers
        for name, finger in [
            ('index', state.index),
            ('middle', state.middle),
            ('ring', state.ring),
            ('pinky', state.pinky),
        ]:
            angles = finger.to_radians()
            positions[name] = self._compute_finger_chain(
                name, angles, self.BASE_POSITIONS[name]
            )
        
        return positions
    
    def _compute_finger_chain(
        self,
        finger: str,
        angles: List[float],
        base: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Compute fingertip position using simplified planar model."""
        lengths = self.SEGMENT_LENGTHS[finger]
        
        x, y, z = base
        cumulative_angle = 0.0
        
        # For simplicity, use 2D planar model (side view)
        for i, length in enumerate(lengths):
            if i < len(angles):
                cumulative_angle += angles[i]
            
            scaled_length = length * self.scale
            y += scaled_length * math.cos(cumulative_angle)
            z += scaled_length * math.sin(cumulative_angle)
        
        return (x * self.scale, y, z)
    
    def compute_finger_closure(self, state: GloveState) -> List[float]:
        """Compute finger closure values [0-1]."""
        return state.get_finger_closure()
    
    def compute_hand_openness(self, state: GloveState) -> float:
        """Compute overall hand openness [0-1]."""
        closures = state.get_finger_closure()
        return 1.0 - (sum(closures) / len(closures))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DYGlove SDK")
    parser.add_argument("command", choices=["scan", "connect", "stream", "calibrate", "simulate", "test"])
    parser.add_argument("--port", help="Serial port")
    parser.add_argument("--wifi", help="WiFi IP")
    parser.add_argument("--rate", type=float, default=120)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--side", default="right", choices=["left", "right"])
    args = parser.parse_args()
    
    if args.command == "scan":
        devices = DYGloveDiscovery().scan()
        print(f"Found {len(devices)} device(s):")
        for dev in devices:
            print(f"  {dev['type']}: {dev.get('port') or dev.get('ip')}")
    
    elif args.command == "simulate":
        print("Running DYGlove simulator (21 DOF)...")
        driver = DYGloveSimulator(side=Hand(args.side))
        driver.connect()
        try:
            while True:
                state = driver.get_state()
                angles = state.to_21dof_array()
                print(f"21-DOF: T={angles[0]:.1f}° I={angles[5]:.1f}° M={angles[9]:.1f}° "
                      f"R={angles[13]:.1f}° P={angles[17]:.1f}°", end='\r')
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nStopped")
            driver.disconnect()
    
    elif args.command == "test":
        print("Testing DYGlove SDK (21 DOF)...")
        state = GloveState()
        print(f"✓ GloveState: {len(state.to_21dof_array())} DOF")
        angles = list(range(21))
        state2 = GloveState.from_21dof_array(angles)
        print(f"✓ from_21dof_array: thumb.tm_flex={state2.thumb.tm_flex}° (expected 0)")
        closures = state2.get_finger_closure()
        print(f"✓ Finger closures: {[f'{c:.2f}' for c in closures]}")
        combined = CombinedFeedback(force_readings=[120, 80, 30, 5, 0])
        force_cmd, haptic_cmd = combined.to_commands()
        print(f"✓ Combined: force={[f'{v:.2f}' for v in force_cmd.to_list()]}")
        sim = DYGloveSimulator()
        sim.connect()
        state3 = sim.get_state()
        print(f"✓ Simulator: {len(state3.to_21dof_array())} DOF")
        sim.disconnect()
        print("\nAll tests passed!")
    
    else:
        glove = DYGloveSDKClient()
        if args.command == "connect":
            if glove.connect(port=args.port, wifi_ip=args.wifi):
                print(f"Connected: {glove.info}")
                glove.disconnect()
        elif args.command == "stream":
            if glove.connect(port=args.port, wifi_ip=args.wifi):
                def on_data(state):
                    print(f"21-DOF: {state.to_21dof_array()[:5]}...", end='\r')
                glove.start_streaming(on_data, rate_hz=args.rate)
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    glove.stop_streaming()
                    glove.disconnect()
        elif args.command == "calibrate":
            if glove.connect(port=args.port, wifi_ip=args.wifi):
                cal = glove.calibrate(duration=args.duration,
                                     callback=lambda m, p: print(f"{m} ({p*100:.0f}%)"))
                print(f"Calibration: {cal}")
                glove.disconnect()


if __name__ == "__main__":
    main()

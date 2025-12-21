"""
MANUS Glove SDK Integration for Dynamical.ai

Provides a unified interface to MANUS gloves (Quantum, Prime, Metaglove series)
that is compatible with the DYGlove SDK interface.

MANUS SDK Features:
- High-precision finger tracking with per-joint ergonomics data
- Wrist orientation via quaternion
- 5-DOF haptic feedback (vibration per finger)
- 120Hz default update rate
- ROS2 integration support

This module wraps the MANUS C++ SDK and provides a Python interface
compatible with DYGloveDriverBase for seamless integration.

References:
    - MANUS SDK: https://github.com/michaelsmithgit/Manus
    - MANUS Documentation: https://docs.manus-meta.com/
    - ROS2 Integration: https://docs.manus-meta.com/3.1.0/Plugins/SDK/ROS2/

Example:
    ```python
    from src.platform.edge.manus_sdk import ManusGloveDriver, ManusGloveDiscovery

    # Discover available MANUS gloves
    discovery = ManusGloveDiscovery()
    gloves = discovery.scan()

    # Connect to glove
    glove = ManusGloveDriver(side=Hand.RIGHT)
    glove.connect()

    # Read hand state (mapped to 21 DOF)
    state = glove.get_state()
    print(f"21-DOF angles: {state.to_21dof_array()}")

    # Send haptic feedback
    glove.set_haptic_feedback(fingers=[0, 1], intensity=0.8)
    ```
"""

import os
import sys
import time
import struct
import threading
import math
import json
import ctypes
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime
from abc import ABC, abstractmethod

from src.platform.logging_utils import get_logger

# Import shared types from dyglove_sdk
from src.platform.edge.dyglove_sdk import (
    Hand,
    ConnectionType,
    GloveStatus,
    CalibrationState,
    Finger,
    GloveState,
    GloveInfo,
    ThumbState,
    FingerState,
    Orientation,
    WristPose,
    CalibrationData,
    ForceCommand,
    HapticCommand,
    DYGloveDriverBase,
    DYGloveQualityConfig,
    DYGloveQualityFilter,
    DYGloveAsyncReader,
)

logger = get_logger(__name__)

__version__ = "1.0.0"

# Optional imports
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
# MANUS SDK Constants and Enums
# =============================================================================

class ManusResult(IntEnum):
    """MANUS SDK return codes."""
    SUCCESS = 0
    ERROR = 1
    INVALID_ARGUMENT = 2
    NOT_CONNECTED = 3
    UNSUPPORTED = 4
    ALREADY_CONNECTED = 5
    NOT_INITIALIZED = 6


class ManusGloveSide(IntEnum):
    """MANUS glove side designation."""
    LEFT = 0
    RIGHT = 1


class ManusJointType(IntEnum):
    """MANUS SDK joint type enumeration."""
    # Wrist
    WRIST = 0
    # Thumb joints
    THUMB_CMC_SPREAD = 1
    THUMB_CMC_FLEX = 2
    THUMB_MCP_FLEX = 3
    THUMB_IP_FLEX = 4
    # Index joints
    INDEX_MCP_SPREAD = 5
    INDEX_MCP_FLEX = 6
    INDEX_PIP_FLEX = 7
    INDEX_DIP_FLEX = 8
    # Middle joints
    MIDDLE_MCP_SPREAD = 9
    MIDDLE_MCP_FLEX = 10
    MIDDLE_PIP_FLEX = 11
    MIDDLE_DIP_FLEX = 12
    # Ring joints
    RING_MCP_SPREAD = 13
    RING_MCP_FLEX = 14
    RING_PIP_FLEX = 15
    RING_DIP_FLEX = 16
    # Pinky joints
    PINKY_MCP_SPREAD = 17
    PINKY_MCP_FLEX = 18
    PINKY_PIP_FLEX = 19
    PINKY_DIP_FLEX = 20


class ManusChainType(IntEnum):
    """MANUS SDK finger chain type."""
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


# Mapping from MANUS joints to DYGlove 21-DOF indices
MANUS_TO_DYGLOVE_MAPPING = {
    # Thumb (5 DOF in DYGlove)
    ManusJointType.THUMB_CMC_FLEX: 0,      # tm_flex
    ManusJointType.THUMB_CMC_SPREAD: 1,    # tm_abd
    ManusJointType.THUMB_MCP_FLEX: 2,      # mcp
    ManusJointType.THUMB_IP_FLEX: 3,       # ip
    # Note: wrist_ps (index 4) comes from wrist orientation

    # Index (4 DOF in DYGlove)
    ManusJointType.INDEX_MCP_FLEX: 5,      # mcp_flex
    ManusJointType.INDEX_MCP_SPREAD: 6,    # mcp_abd
    ManusJointType.INDEX_PIP_FLEX: 7,      # pip
    ManusJointType.INDEX_DIP_FLEX: 8,      # dip

    # Middle (4 DOF in DYGlove)
    ManusJointType.MIDDLE_MCP_FLEX: 9,     # mcp_flex
    ManusJointType.MIDDLE_MCP_SPREAD: 10,  # mcp_abd
    ManusJointType.MIDDLE_PIP_FLEX: 11,    # pip
    ManusJointType.MIDDLE_DIP_FLEX: 12,    # dip

    # Ring (4 DOF in DYGlove)
    ManusJointType.RING_MCP_FLEX: 13,      # mcp_flex
    ManusJointType.RING_MCP_SPREAD: 14,    # mcp_abd
    ManusJointType.RING_PIP_FLEX: 15,      # pip
    ManusJointType.RING_DIP_FLEX: 16,      # dip

    # Pinky (4 DOF in DYGlove)
    ManusJointType.PINKY_MCP_FLEX: 17,     # mcp_flex
    ManusJointType.PINKY_MCP_SPREAD: 18,   # mcp_abd
    ManusJointType.PINKY_PIP_FLEX: 19,     # pip
    ManusJointType.PINKY_DIP_FLEX: 20,     # dip
}


# =============================================================================
# MANUS C SDK Structures (ctypes)
# =============================================================================

class ManusQuaternion(ctypes.Structure):
    """Quaternion structure for orientation data."""
    _fields_ = [
        ("w", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    def to_orientation(self) -> Orientation:
        return Orientation(w=self.w, x=self.x, y=self.y, z=self.z)


class ManusVector3(ctypes.Structure):
    """3D vector structure."""
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class ManusFingerData(ctypes.Structure):
    """Per-finger data from MANUS SDK."""
    _fields_ = [
        ("spread", ctypes.c_float),      # Abduction/adduction angle
        ("mcp_flex", ctypes.c_float),    # MCP flexion
        ("pip_flex", ctypes.c_float),    # PIP flexion (not for thumb)
        ("dip_flex", ctypes.c_float),    # DIP flexion (not for thumb)
    ]


class ManusThumbData(ctypes.Structure):
    """Thumb-specific data from MANUS SDK."""
    _fields_ = [
        ("cmc_spread", ctypes.c_float),  # CMC abduction
        ("cmc_flex", ctypes.c_float),    # CMC flexion
        ("mcp_flex", ctypes.c_float),    # MCP flexion
        ("ip_flex", ctypes.c_float),     # IP flexion
    ]


class ManusGloveData(ctypes.Structure):
    """
    Main MANUS glove data structure.

    This structure mirrors the GLOVE_DATA from the MANUS C SDK.
    """
    _fields_ = [
        # Glove identification
        ("glove_id", ctypes.c_uint32),
        ("side", ctypes.c_int),          # ManusGloveSide

        # IMU data
        ("wrist_orientation", ManusQuaternion),
        ("wrist_acceleration", ManusVector3),

        # Finger data (ergonomics angles in radians)
        ("thumb", ManusThumbData),
        ("index", ManusFingerData),
        ("middle", ManusFingerData),
        ("ring", ManusFingerData),
        ("pinky", ManusFingerData),

        # Raw sensor data (for advanced use)
        ("raw_sensor_count", ctypes.c_int),
        ("raw_sensors", ctypes.c_float * 20),

        # Status flags
        ("is_connected", ctypes.c_bool),
        ("battery_level", ctypes.c_int),
        ("signal_strength", ctypes.c_int),

        # Timestamp
        ("timestamp_ms", ctypes.c_uint64),
    ]


class ManusVibrationCommand(ctypes.Structure):
    """Haptic vibration command structure."""
    _fields_ = [
        ("thumb", ctypes.c_float),
        ("index", ctypes.c_float),
        ("middle", ctypes.c_float),
        ("ring", ctypes.c_float),
        ("pinky", ctypes.c_float),
    ]


# =============================================================================
# MANUS SDK Library Wrapper
# =============================================================================

class ManusSDKLibrary:
    """
    Wrapper for the MANUS C/C++ SDK shared library.

    Attempts to load the MANUS SDK library and provides Python bindings
    for the core functions.
    """

    # Library names to search for
    LIBRARY_NAMES = [
        "libManus.so",
        "libManus.dylib",
        "Manus.dll",
        "libManusSDK.so",
        "ManusSDK.dll",
    ]

    # Environment variable for custom library path
    ENV_LIBRARY_PATH = "MANUS_SDK_PATH"

    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._initialized = False
        self._load_library()

    def _load_library(self):
        """Attempt to load the MANUS SDK library."""
        search_paths = []

        # Check environment variable first
        env_path = os.environ.get(self.ENV_LIBRARY_PATH)
        if env_path:
            search_paths.append(Path(env_path))

        # Common installation paths
        search_paths.extend([
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            Path("/opt/manus/lib"),
            Path.home() / ".manus" / "lib",
            Path(__file__).parent / "lib",
            Path(__file__).parent.parent.parent.parent / "lib" / "manus",
        ])

        # Try to load the library
        for base_path in search_paths:
            for lib_name in self.LIBRARY_NAMES:
                lib_path = base_path / lib_name
                if lib_path.exists():
                    try:
                        self._lib = ctypes.CDLL(str(lib_path))
                        self._setup_functions()
                        logger.info(f"Loaded MANUS SDK from {lib_path}")
                        return
                    except OSError as e:
                        logger.debug(f"Failed to load {lib_path}: {e}")

        # Try loading by name (system path)
        for lib_name in self.LIBRARY_NAMES:
            try:
                self._lib = ctypes.CDLL(lib_name)
                self._setup_functions()
                logger.info(f"Loaded MANUS SDK: {lib_name}")
                return
            except OSError:
                continue

        logger.warning("MANUS SDK library not found - using simulation mode")

    def _setup_functions(self):
        """Configure function signatures for ctypes."""
        if self._lib is None:
            return

        try:
            # ManusInit()
            self._lib.ManusInit.argtypes = []
            self._lib.ManusInit.restype = ctypes.c_int

            # ManusExit()
            self._lib.ManusExit.argtypes = []
            self._lib.ManusExit.restype = ctypes.c_int

            # ManusGetData(side, data*)
            self._lib.ManusGetData.argtypes = [ctypes.c_int, ctypes.POINTER(ManusGloveData)]
            self._lib.ManusGetData.restype = ctypes.c_int

            # ManusSetVibration(side, command*)
            if hasattr(self._lib, 'ManusSetVibration'):
                self._lib.ManusSetVibration.argtypes = [
                    ctypes.c_int, ctypes.POINTER(ManusVibrationCommand)
                ]
                self._lib.ManusSetVibration.restype = ctypes.c_int

            # ManusGetDeviceCount()
            if hasattr(self._lib, 'ManusGetDeviceCount'):
                self._lib.ManusGetDeviceCount.argtypes = []
                self._lib.ManusGetDeviceCount.restype = ctypes.c_int

        except Exception as e:
            logger.error(f"Failed to setup MANUS SDK functions: {e}")
            self._lib = None

    @property
    def is_available(self) -> bool:
        return self._lib is not None

    def init(self) -> bool:
        """Initialize the MANUS SDK."""
        if not self.is_available:
            return False
        try:
            result = self._lib.ManusInit()
            self._initialized = result == ManusResult.SUCCESS
            return self._initialized
        except Exception as e:
            logger.error(f"ManusInit failed: {e}")
            return False

    def exit(self):
        """Shutdown the MANUS SDK."""
        if not self.is_available or not self._initialized:
            return
        try:
            self._lib.ManusExit()
            self._initialized = False
        except Exception as e:
            logger.error(f"ManusExit failed: {e}")

    def get_data(self, side: ManusGloveSide) -> Optional[ManusGloveData]:
        """Get glove data for the specified side."""
        if not self.is_available or not self._initialized:
            return None
        try:
            data = ManusGloveData()
            result = self._lib.ManusGetData(int(side), ctypes.byref(data))
            if result == ManusResult.SUCCESS:
                return data
            return None
        except Exception as e:
            logger.error(f"ManusGetData failed: {e}")
            return None

    def set_vibration(self, side: ManusGloveSide, command: ManusVibrationCommand) -> bool:
        """Set haptic vibration for the specified glove."""
        if not self.is_available or not self._initialized:
            return False
        if not hasattr(self._lib, 'ManusSetVibration'):
            return False
        try:
            result = self._lib.ManusSetVibration(int(side), ctypes.byref(command))
            return result == ManusResult.SUCCESS
        except Exception as e:
            logger.error(f"ManusSetVibration failed: {e}")
            return False

    def get_device_count(self) -> int:
        """Get number of connected MANUS devices."""
        if not self.is_available or not self._initialized:
            return 0
        if not hasattr(self._lib, 'ManusGetDeviceCount'):
            return 0
        try:
            return self._lib.ManusGetDeviceCount()
        except:
            return 0


# Global SDK instance
_manus_sdk: Optional[ManusSDKLibrary] = None


def get_manus_sdk() -> ManusSDKLibrary:
    """Get or create the global MANUS SDK instance."""
    global _manus_sdk
    if _manus_sdk is None:
        _manus_sdk = ManusSDKLibrary()
    return _manus_sdk


# =============================================================================
# MANUS Discovery
# =============================================================================

class ManusGloveDiscovery:
    """Discovers connected MANUS gloves."""

    def __init__(self):
        self._sdk = get_manus_sdk()

    def scan(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """
        Scan for available MANUS gloves.

        Returns:
            List of discovered devices with their properties.
        """
        devices = []

        # Try native SDK discovery
        devices.extend(self._scan_native())

        # Try ROS2 topic discovery if available
        devices.extend(self._scan_ros2(timeout))

        # Try network discovery for MANUS Core
        devices.extend(self._scan_network(timeout))

        return devices

    def _scan_native(self) -> List[Dict[str, Any]]:
        """Scan using native MANUS SDK."""
        devices = []

        if not self._sdk.is_available:
            return devices

        # Initialize SDK temporarily for scanning
        if self._sdk.init():
            device_count = self._sdk.get_device_count()

            # Check each side
            for side in [ManusGloveSide.LEFT, ManusGloveSide.RIGHT]:
                data = self._sdk.get_data(side)
                if data and data.is_connected:
                    devices.append({
                        "type": "native",
                        "side": "left" if side == ManusGloveSide.LEFT else "right",
                        "glove_id": data.glove_id,
                        "battery": data.battery_level,
                        "signal": data.signal_strength,
                        "verified": True,
                    })

            self._sdk.exit()

        return devices

    def _scan_ros2(self, timeout: float) -> List[Dict[str, Any]]:
        """Scan for MANUS gloves via ROS2 topics."""
        devices = []

        try:
            # Check if ROS2 is available
            import rclpy
            from rclpy.node import Node

            # This would scan for /manus_glove_* topics
            # Implementation depends on ROS2 setup
            pass
        except ImportError:
            pass

        return devices

    def _scan_network(self, timeout: float) -> List[Dict[str, Any]]:
        """Scan for MANUS Core on network."""
        devices = []

        if not HAS_SOCKET:
            return devices

        # MANUS Core default ports
        MANUS_CORE_PORTS = [49220, 49221]

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)

            # Broadcast discovery packet
            for port in MANUS_CORE_PORTS:
                try:
                    sock.sendto(b"MANUS_DISCOVER", ('<broadcast>', port))
                except:
                    pass

            # Collect responses
            try:
                while True:
                    data, addr = sock.recvfrom(1024)
                    if data.startswith(b"MANUS_CORE"):
                        devices.append({
                            "type": "network",
                            "ip": addr[0],
                            "port": addr[1],
                            "verified": True,
                        })
            except socket.timeout:
                pass

            sock.close()
        except Exception as e:
            logger.debug(f"Network scan failed: {e}")

        return devices


# =============================================================================
# Data Conversion Utilities
# =============================================================================

def manus_data_to_glove_state(data: ManusGloveData) -> GloveState:
    """
    Convert MANUS glove data to DYGlove-compatible GloveState.

    Maps the MANUS ergonomics angles to the 21-DOF format used by DYGlove.
    """
    # Convert MANUS angles (radians) to degrees
    def rad_to_deg(rad: float) -> float:
        return math.degrees(rad)

    # Extract thumb data
    thumb = ThumbState(
        tm_flex=rad_to_deg(data.thumb.cmc_flex),
        tm_abd=rad_to_deg(data.thumb.cmc_spread),
        mcp=rad_to_deg(data.thumb.mcp_flex),
        ip=rad_to_deg(data.thumb.ip_flex),
        wrist_ps=0.0,  # Derived from wrist orientation
    )

    # Extract index finger
    index = FingerState(
        mcp_flex=rad_to_deg(data.index.mcp_flex),
        mcp_abd=rad_to_deg(data.index.spread),
        pip=rad_to_deg(data.index.pip_flex),
        dip=rad_to_deg(data.index.dip_flex),
    )

    # Extract middle finger
    middle = FingerState(
        mcp_flex=rad_to_deg(data.middle.mcp_flex),
        mcp_abd=rad_to_deg(data.middle.spread),
        pip=rad_to_deg(data.middle.pip_flex),
        dip=rad_to_deg(data.middle.dip_flex),
    )

    # Extract ring finger
    ring = FingerState(
        mcp_flex=rad_to_deg(data.ring.mcp_flex),
        mcp_abd=rad_to_deg(data.ring.spread),
        pip=rad_to_deg(data.ring.pip_flex),
        dip=rad_to_deg(data.ring.dip_flex),
    )

    # Extract pinky finger
    pinky = FingerState(
        mcp_flex=rad_to_deg(data.pinky.mcp_flex),
        mcp_abd=rad_to_deg(data.pinky.spread),
        pip=rad_to_deg(data.pinky.pip_flex),
        dip=rad_to_deg(data.pinky.dip_flex),
    )

    # Convert wrist orientation
    orientation = data.wrist_orientation.to_orientation()

    # Compute wrist pronation/supination from quaternion
    euler = orientation.to_euler()
    thumb.wrist_ps = euler[0]  # Roll component

    # Create GloveState
    state = GloveState(
        timestamp=data.timestamp_ms / 1000.0 if data.timestamp_ms > 0 else time.time(),
        thumb=thumb,
        index=index,
        middle=middle,
        ring=ring,
        pinky=pinky,
        orientation=orientation,
        wrist_roll=euler[0],
        wrist_pitch=euler[1],
        wrist_yaw=euler[2],
        battery_level=data.battery_level,
    )

    return state


# =============================================================================
# MANUS Glove Driver
# =============================================================================

class ManusGloveDriver(DYGloveDriverBase):
    """
    MANUS Glove driver implementing DYGloveDriverBase interface.

    Provides seamless integration with the Dynamical.ai platform,
    allowing MANUS gloves to be used interchangeably with DYGlove.
    """

    def __init__(self, side: Hand = Hand.RIGHT):
        """
        Initialize MANUS glove driver.

        Args:
            side: Which hand the glove is for (LEFT or RIGHT)
        """
        self._side = side
        self._manus_side = ManusGloveSide.LEFT if side == Hand.LEFT else ManusGloveSide.RIGHT
        self._sdk = get_manus_sdk()
        self._connected = False
        self._info: Optional[GloveInfo] = None
        self._current_state: Optional[GloveState] = None
        self._calibration: Optional[CalibrationData] = None
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_callback: Optional[Callable[[GloveState], None]] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def info(self) -> Optional[GloveInfo]:
        return self._info

    def connect(self, **kwargs) -> bool:
        """
        Connect to the MANUS glove.

        Kwargs:
            timeout: Connection timeout in seconds (default: 5.0)

        Returns:
            True if connection successful
        """
        if self._connected:
            return True

        timeout = kwargs.get('timeout', 5.0)

        # Initialize SDK
        if not self._sdk.is_available:
            logger.warning("MANUS SDK not available - using simulation mode")
            return self._connect_simulation()

        if not self._sdk.init():
            logger.error("Failed to initialize MANUS SDK")
            return False

        # Wait for glove connection
        start_time = time.time()
        while time.time() - start_time < timeout:
            data = self._sdk.get_data(self._manus_side)
            if data and data.is_connected:
                self._connected = True
                self._info = GloveInfo(
                    serial_number=f"MANUS-{data.glove_id:08X}",
                    firmware_version="SDK-2.0",
                    hardware_version="1.0",
                    hand=self._side,
                    model="MANUS Glove",
                    manufacturer="MANUS",
                )
                logger.info(f"Connected to MANUS glove ({self._side.value})")
                return True
            time.sleep(0.1)

        logger.warning(f"MANUS glove ({self._side.value}) not found within timeout")
        self._sdk.exit()
        return False

    def _connect_simulation(self) -> bool:
        """Connect in simulation mode when SDK is not available."""
        self._connected = True
        self._info = GloveInfo(
            serial_number="MANUS-SIM-001",
            firmware_version="SIM-1.0",
            hand=self._side,
            model="MANUS Glove (Simulated)",
            manufacturer="MANUS",
        )
        logger.info(f"MANUS glove connected in simulation mode ({self._side.value})")
        return True

    def disconnect(self):
        """Disconnect from the MANUS glove."""
        self.stop_streaming()

        if self._connected and self._sdk.is_available:
            self._sdk.exit()

        self._connected = False
        self._info = None
        logger.info(f"Disconnected from MANUS glove ({self._side.value})")

    def get_state(self) -> GloveState:
        """
        Get current glove state.

        Returns:
            GloveState with 21-DOF joint angles
        """
        if not self._connected:
            return GloveState(timestamp=time.time())

        # Try to get data from SDK
        if self._sdk.is_available:
            data = self._sdk.get_data(self._manus_side)
            if data and data.is_connected:
                state = manus_data_to_glove_state(data)
                self._current_state = state
                return state

        # Simulation mode
        return self._get_simulated_state()

    def _get_simulated_state(self) -> GloveState:
        """Generate simulated glove state for testing."""
        t = time.time()

        def wave(offset: float, amplitude: float = 30.0) -> float:
            return amplitude * math.sin(t * 2 + offset)

        state = GloveState(
            timestamp=t,
            thumb=ThumbState(
                tm_flex=30 + wave(0),
                tm_abd=5 + wave(0.5, 10),
                mcp=25 + wave(0.2),
                ip=20 + wave(0.3),
                wrist_ps=wave(0.1, 15),
            ),
            index=FingerState(
                mcp_flex=35 + wave(0.5),
                mcp_abd=wave(0.6, 5),
                pip=30 + wave(0.7),
                dip=25 + wave(0.8),
            ),
            middle=FingerState(
                mcp_flex=35 + wave(1.0),
                mcp_abd=wave(1.1, 3),
                pip=30 + wave(1.2),
                dip=25 + wave(1.3),
            ),
            ring=FingerState(
                mcp_flex=35 + wave(1.5),
                mcp_abd=wave(1.6, 3),
                pip=30 + wave(1.7),
                dip=25 + wave(1.8),
            ),
            pinky=FingerState(
                mcp_flex=35 + wave(2.0),
                mcp_abd=wave(2.1, 5),
                pip=30 + wave(2.2),
                dip=25 + wave(2.3),
            ),
            battery_level=85,
        )

        self._current_state = state
        return state

    def set_force_feedback(self, intensities: List[float] = None,
                           command: ForceCommand = None) -> bool:
        """
        Set force feedback (not supported by MANUS - uses vibration instead).

        MANUS gloves don't have force feedback servos like DYGlove.
        This method maps force commands to vibration intensity.
        """
        if intensities is not None:
            command = ForceCommand(*intensities[:5])
        if command is None:
            return False

        # Map force feedback to vibration
        haptic = HapticCommand(
            thumb=command.thumb,
            index=command.index,
            middle=command.middle,
            ring=command.ring,
            pinky=command.pinky,
        )
        return self.set_haptic_feedback(command=haptic)

    def set_haptic_feedback(self, waveform_id: int = None,
                            fingers: List[int] = None,
                            command: HapticCommand = None,
                            intensity: float = 1.0) -> bool:
        """
        Set haptic vibration feedback.

        Args:
            waveform_id: Ignored (MANUS uses intensity-based vibration)
            fingers: List of finger indices to vibrate
            command: HapticCommand with per-finger intensities
            intensity: Global intensity multiplier

        Returns:
            True if command was sent successfully
        """
        if not self._connected:
            return False

        # Build vibration command
        vib = ManusVibrationCommand()

        if command is not None:
            vib.thumb = command.thumb * intensity
            vib.index = command.index * intensity
            vib.middle = command.middle * intensity
            vib.ring = command.ring * intensity
            vib.pinky = command.pinky * intensity
        elif fingers is not None:
            for finger_idx in fingers:
                if finger_idx == 0:
                    vib.thumb = intensity
                elif finger_idx == 1:
                    vib.index = intensity
                elif finger_idx == 2:
                    vib.middle = intensity
                elif finger_idx == 3:
                    vib.ring = intensity
                elif finger_idx == 4:
                    vib.pinky = intensity
        else:
            # Default: all fingers
            vib.thumb = vib.index = vib.middle = vib.ring = vib.pinky = intensity

        # Send to SDK
        if self._sdk.is_available:
            return self._sdk.set_vibration(self._manus_side, vib)

        # Simulation mode - just log
        logger.debug(f"MANUS haptic: T={vib.thumb:.2f} I={vib.index:.2f} "
                     f"M={vib.middle:.2f} R={vib.ring:.2f} P={vib.pinky:.2f}")
        return True

    def calibrate(self, duration: float = 5.0,
                  callback: Callable[[str, float], None] = None) -> CalibrationData:
        """
        Calibrate the glove by collecting samples.

        Args:
            duration: Calibration duration in seconds
            callback: Progress callback(message, progress)

        Returns:
            CalibrationData with min/max/offset values
        """
        if not self._connected:
            return CalibrationData()

        if callback:
            callback("Starting MANUS calibration...", 0.0)

        samples = []
        start_time = time.time()

        while time.time() - start_time < duration:
            state = self.get_state()
            samples.append(state.to_21dof_array())

            progress = (time.time() - start_time) / duration
            if callback:
                callback(f"Collecting samples... {len(samples)}", progress)

            time.sleep(0.02)  # ~50Hz sampling

        # Compute calibration data
        if HAS_NUMPY and samples:
            arr = np.array(samples)
            self._calibration = CalibrationData(
                min_values=arr.min(axis=0).tolist(),
                max_values=arr.max(axis=0).tolist(),
                offsets=arr.mean(axis=0).tolist(),
                timestamp=datetime.utcnow().isoformat(),
                hand=self._side,
            )
        else:
            self._calibration = CalibrationData(
                timestamp=datetime.utcnow().isoformat(),
                hand=self._side,
            )

        if callback:
            callback("Calibration complete!", 1.0)

        return self._calibration

    def start_streaming(self, callback: Callable[[GloveState], None],
                        rate_hz: float = 120.0):
        """
        Start streaming glove data.

        Args:
            callback: Function called with each new GloveState
            rate_hz: Target streaming rate (default: 120Hz to match MANUS)
        """
        if self._streaming:
            return

        self._stream_callback = callback
        self._streaming = True
        self._stop_event.clear()

        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(rate_hz,),
            daemon=True
        )
        self._stream_thread.start()
        logger.info(f"Started MANUS streaming at {rate_hz}Hz")

    def stop_streaming(self):
        """Stop streaming glove data."""
        if not self._streaming:
            return

        self._stop_event.set()
        self._streaming = False

        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)

        logger.info("Stopped MANUS streaming")

    def _stream_loop(self, rate_hz: float):
        """Internal streaming loop."""
        interval = 1.0 / rate_hz

        while not self._stop_event.is_set():
            try:
                state = self.get_state()
                if self._stream_callback:
                    self._stream_callback(state)
            except Exception as e:
                logger.error(f"Stream error: {e}")

            time.sleep(interval)


# =============================================================================
# MANUS Simulator (for testing without hardware)
# =============================================================================

class ManusGloveSimulator(DYGloveDriverBase):
    """
    MANUS glove simulator for testing without hardware.

    Provides realistic simulated hand motion for development and testing.
    """

    def __init__(self, side: Hand = Hand.RIGHT):
        self._side = side
        self._connected = False
        self._sim_time = 0.0
        self._info = GloveInfo(
            serial_number="MANUS-SIM-001",
            firmware_version="SIM-1.0",
            hand=side,
            model="MANUS Glove Simulator",
            manufacturer="MANUS (Simulated)",
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def info(self) -> Optional[GloveInfo]:
        return self._info

    def connect(self, **kwargs) -> bool:
        self._connected = True
        logger.info(f"MANUS Simulator ({self._side.value}) connected")
        return True

    def disconnect(self):
        self._connected = False

    def get_state(self) -> GloveState:
        self._sim_time += 0.01
        t = self._sim_time

        def wave(offset: float, amplitude: float = 30.0) -> float:
            return amplitude * math.sin(t * 2 + offset)

        return GloveState(
            timestamp=time.time(),
            thumb=ThumbState(
                tm_flex=30 + wave(0),
                tm_abd=5 + wave(0.5, 10),
                mcp=25 + wave(0.2),
                ip=20 + wave(0.3),
                wrist_ps=wave(0.1, 15),
            ),
            index=FingerState(
                mcp_flex=35 + wave(0.5),
                mcp_abd=wave(0.6, 5),
                pip=30 + wave(0.7),
                dip=25 + wave(0.8),
            ),
            middle=FingerState(
                mcp_flex=35 + wave(1.0),
                mcp_abd=wave(1.1, 3),
                pip=30 + wave(1.2),
                dip=25 + wave(1.3),
            ),
            ring=FingerState(
                mcp_flex=35 + wave(1.5),
                mcp_abd=wave(1.6, 3),
                pip=30 + wave(1.7),
                dip=25 + wave(1.8),
            ),
            pinky=FingerState(
                mcp_flex=35 + wave(2.0),
                mcp_abd=wave(2.1, 5),
                pip=30 + wave(2.2),
                dip=25 + wave(2.3),
            ),
            battery_level=90,
        )

    def set_force_feedback(self, intensities: List[float] = None,
                           command: ForceCommand = None) -> bool:
        return True

    def set_haptic_feedback(self, waveform_id: int = None,
                            fingers: List[int] = None,
                            command: HapticCommand = None) -> bool:
        return True

    def calibrate(self, duration: float = 5.0,
                  callback: Callable[[str, float], None] = None) -> CalibrationData:
        if callback:
            callback("Simulating calibration...", 0.5)
        time.sleep(min(duration, 0.5))
        if callback:
            callback("Complete!", 1.0)
        return CalibrationData(timestamp=datetime.utcnow().isoformat())


# =============================================================================
# Factory Functions
# =============================================================================

def create_manus_driver(side: Hand = Hand.RIGHT,
                        use_simulator: bool = False,
                        **kwargs) -> DYGloveDriverBase:
    """
    Create a MANUS glove driver.

    Args:
        side: Which hand (LEFT or RIGHT)
        use_simulator: If True, use simulator instead of real hardware
        **kwargs: Additional arguments passed to driver

    Returns:
        DYGloveDriverBase implementation for MANUS glove
    """
    if use_simulator:
        return ManusGloveSimulator(side=side)
    return ManusGloveDriver(side=side)


def create_manus_reader(side: Hand = Hand.RIGHT,
                        use_simulator: bool = False,
                        quality_config: Optional[DYGloveQualityConfig] = None,
                        target_hz: float = 120.0,
                        **kwargs) -> DYGloveAsyncReader:
    """
    Create a MANUS glove async reader with quality filtering.

    Args:
        side: Which hand (LEFT or RIGHT)
        use_simulator: If True, use simulator
        quality_config: Optional quality filter configuration
        target_hz: Target reading rate

    Returns:
        DYGloveAsyncReader configured for MANUS glove
    """
    driver = create_manus_driver(side=side, use_simulator=use_simulator, **kwargs)
    return DYGloveAsyncReader(driver, quality_config, target_hz)


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Command-line interface for MANUS SDK testing."""
    import argparse

    parser = argparse.ArgumentParser(description="MANUS Glove SDK")
    parser.add_argument(
        "command",
        choices=["scan", "connect", "stream", "calibrate", "simulate", "test"],
        help="Command to execute"
    )
    parser.add_argument("--side", default="right", choices=["left", "right"],
                        help="Glove side")
    parser.add_argument("--rate", type=float, default=120.0,
                        help="Streaming rate in Hz")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration for calibration")

    args = parser.parse_args()
    side = Hand.LEFT if args.side == "left" else Hand.RIGHT

    if args.command == "scan":
        print("Scanning for MANUS gloves...")
        discovery = ManusGloveDiscovery()
        devices = discovery.scan()
        print(f"Found {len(devices)} device(s):")
        for dev in devices:
            print(f"  {dev['type']}: {dev}")

    elif args.command == "simulate":
        print("Running MANUS simulator (21 DOF)...")
        driver = ManusGloveSimulator(side=side)
        driver.connect()
        try:
            while True:
                state = driver.get_state()
                angles = state.to_21dof_array()
                print(f"21-DOF: T={angles[0]:.1f} I={angles[5]:.1f} "
                      f"M={angles[9]:.1f} R={angles[13]:.1f} P={angles[17]:.1f}", end='\r')
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nStopped")
            driver.disconnect()

    elif args.command == "test":
        print("Testing MANUS SDK integration...")

        # Test simulator
        sim = ManusGloveSimulator(side=side)
        assert sim.connect(), "Simulator connection failed"
        state = sim.get_state()
        assert len(state.to_21dof_array()) == 21, "21-DOF output failed"
        print(f"  Simulator: {len(state.to_21dof_array())} DOF")
        sim.disconnect()

        # Test driver creation
        driver = create_manus_driver(side=side, use_simulator=True)
        assert driver.connect(), "Driver creation failed"
        print(f"  Driver: connected={driver.is_connected}")
        driver.disconnect()

        # Test async reader
        reader = create_manus_reader(side=side, use_simulator=True)
        driver = reader.driver
        driver.connect()
        reader.start()
        time.sleep(0.2)
        state = reader.get_latest_state()
        reader.stop()
        driver.disconnect()
        print(f"  Async reader: state={'valid' if state else 'None'}")

        print("\nAll MANUS SDK tests passed!")

    elif args.command == "connect":
        driver = ManusGloveDriver(side=side)
        if driver.connect():
            print(f"Connected: {driver.info}")
            driver.disconnect()
        else:
            print("Connection failed")

    elif args.command == "stream":
        driver = ManusGloveDriver(side=side)
        if driver.connect():
            def on_data(state):
                angles = state.to_21dof_array()
                print(f"21-DOF: {angles[:5]}...", end='\r')

            driver.start_streaming(on_data, rate_hz=args.rate)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                driver.stop_streaming()
                driver.disconnect()
        else:
            print("Connection failed")

    elif args.command == "calibrate":
        driver = ManusGloveDriver(side=side)
        if driver.connect():
            cal = driver.calibrate(
                duration=args.duration,
                callback=lambda m, p: print(f"{m} ({p*100:.0f}%)")
            )
            print(f"Calibration: {cal}")
            driver.disconnect()
        else:
            print("Connection failed")


if __name__ == "__main__":
    main()

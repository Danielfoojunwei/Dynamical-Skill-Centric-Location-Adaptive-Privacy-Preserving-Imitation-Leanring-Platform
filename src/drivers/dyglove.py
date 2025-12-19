"""
DYGlove Integration Module for Dynamical.ai Edge Pipeline

Replaces DextaGlove with the Dynamical DYGlove - a proprietary wired
haptic force feedback glove with 21-DOF motion capture.

IMPORTANT: All DYGlove connections are now WIRED for reliability:
- Primary: USB 3.2 Gen 2 (10Gbps) for <1ms latency
- Secondary: Ethernet over USB-C for industrial deployments
- NO WiFi: Eliminated for deterministic timing and reliability

Reference: Internal Specs, "DYGlove: Wired Haptic Glove with USB/Ethernet"
https://dynamical.ai/hardware/dyglove

DYGlove Joint Configuration (21 DOF total):
==========================================

Index, Middle, Ring, Pinky fingers (4 DOF each = 16 DOF):
    - MCP_flex: Metacarpophalangeal flexion/extension (ball joint component 1)
    - MCP_abd:  Metacarpophalangeal abduction/adduction (ball joint component 2)  
    - PIP:      Proximal interphalangeal flexion/extension (hinge joint)
    - DIP:      Distal interphalangeal flexion/extension (hinge joint)

Thumb (5 DOF):
    - TM_flex:  Trapeziometacarpal flexion/extension (ball joint component 1)
    - TM_abd:   Trapeziometacarpal abduction/adduction (ball joint component 2)
    - MCP:      Metacarpophalangeal flexion/extension (hinge joint)
    - IP:       Interphalangeal flexion/extension (hinge joint)
    - Wrist_PS: Wrist pronation/supination

Hardware:
    - 16 Alps RDC506018A rotary encoders for joint angles
    - 5 Dynamixel XC/XL330 servo motors for force feedback
    - MCU for data aggregation and USB communication
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import IntEnum
import numpy as np
import time
import threading
import queue
import socket
import struct
import json
from abc import ABC, abstractmethod
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Joint Index Definitions
# =============================================================================

class DYGloveJoint(IntEnum):
    """
    DYGlove 21-DOF joint indices.
    
    Convention: joints are ordered by finger, then by joint within finger
    (proximal to distal), with MCP abduction following MCP flexion.
    """
    # Thumb (5 DOF) - indices 0-4
    THUMB_TM_FLEX = 0      # Trapeziometacarpal flexion
    THUMB_TM_ABD = 1       # Trapeziometacarpal abduction
    THUMB_MCP = 2          # Metacarpophalangeal flexion
    THUMB_IP = 3           # Interphalangeal flexion
    THUMB_WRIST_PS = 4     # Wrist pronation/supination
    
    # Index finger (4 DOF) - indices 5-8
    INDEX_MCP_FLEX = 5     # MCP flexion/extension
    INDEX_MCP_ABD = 6      # MCP abduction/adduction
    INDEX_PIP = 7          # PIP flexion
    INDEX_DIP = 8          # DIP flexion
    
    # Middle finger (4 DOF) - indices 9-12
    MIDDLE_MCP_FLEX = 9
    MIDDLE_MCP_ABD = 10
    MIDDLE_PIP = 11
    MIDDLE_DIP = 12
    
    # Ring finger (4 DOF) - indices 13-16
    RING_MCP_FLEX = 13
    RING_MCP_ABD = 14
    RING_PIP = 15
    RING_DIP = 16
    
    # Pinky finger (4 DOF) - indices 17-20
    PINKY_MCP_FLEX = 17
    PINKY_MCP_ABD = 18
    PINKY_PIP = 19
    PINKY_DIP = 20


# Joint angle limits in radians (based on human hand biomechanics)
DYGLOVE_JOINT_LIMITS = {
    # Thumb
    DYGloveJoint.THUMB_TM_FLEX: (-0.35, 1.22),      # ~-20° to 70°
    DYGloveJoint.THUMB_TM_ABD: (-0.52, 0.52),       # ~-30° to 30°
    DYGloveJoint.THUMB_MCP: (-0.26, 1.57),          # ~-15° to 90°
    DYGloveJoint.THUMB_IP: (-0.17, 1.40),           # ~-10° to 80°
    DYGloveJoint.THUMB_WRIST_PS: (-1.57, 1.57),     # ~-90° to 90°
    
    # Index finger
    DYGloveJoint.INDEX_MCP_FLEX: (-0.35, 1.57),     # ~-20° to 90°
    DYGloveJoint.INDEX_MCP_ABD: (-0.35, 0.35),      # ~-20° to 20°
    DYGloveJoint.INDEX_PIP: (0.0, 1.75),            # 0° to 100°
    DYGloveJoint.INDEX_DIP: (0.0, 1.22),            # 0° to 70°
    
    # Middle finger (similar to index)
    DYGloveJoint.MIDDLE_MCP_FLEX: (-0.35, 1.57),
    DYGloveJoint.MIDDLE_MCP_ABD: (-0.35, 0.35),
    DYGloveJoint.MIDDLE_PIP: (0.0, 1.75),
    DYGloveJoint.MIDDLE_DIP: (0.0, 1.22),
    
    # Ring finger
    DYGloveJoint.RING_MCP_FLEX: (-0.35, 1.57),
    DYGloveJoint.RING_MCP_ABD: (-0.35, 0.35),
    DYGloveJoint.RING_PIP: (0.0, 1.75),
    DYGloveJoint.RING_DIP: (0.0, 1.22),
    
    # Pinky finger
    DYGloveJoint.PINKY_MCP_FLEX: (-0.35, 1.57),
    DYGloveJoint.PINKY_MCP_ABD: (-0.35, 0.35),
    DYGloveJoint.PINKY_PIP: (0.0, 1.75),
    DYGloveJoint.PINKY_DIP: (0.0, 1.22),
}


# Finger groupings for convenience
FINGER_JOINTS = {
    'thumb': [
        DYGloveJoint.THUMB_TM_FLEX, DYGloveJoint.THUMB_TM_ABD,
        DYGloveJoint.THUMB_MCP, DYGloveJoint.THUMB_IP, DYGloveJoint.THUMB_WRIST_PS
    ],
    'index': [
        DYGloveJoint.INDEX_MCP_FLEX, DYGloveJoint.INDEX_MCP_ABD,
        DYGloveJoint.INDEX_PIP, DYGloveJoint.INDEX_DIP
    ],
    'middle': [
        DYGloveJoint.MIDDLE_MCP_FLEX, DYGloveJoint.MIDDLE_MCP_ABD,
        DYGloveJoint.MIDDLE_PIP, DYGloveJoint.MIDDLE_DIP
    ],
    'ring': [
        DYGloveJoint.RING_MCP_FLEX, DYGloveJoint.RING_MCP_ABD,
        DYGloveJoint.RING_PIP, DYGloveJoint.RING_DIP
    ],
    'pinky': [
        DYGloveJoint.PINKY_MCP_FLEX, DYGloveJoint.PINKY_MCP_ABD,
        DYGloveJoint.PINKY_PIP, DYGloveJoint.PINKY_DIP
    ],
}


# =============================================================================
# DYGlove State Representation
# =============================================================================

@dataclass
class DYGloveQuality:
    """
    Quality metrics specific to DYGlove data.
    
    These metrics help filter out unreliable glove readings during
    data collection for imitation learning.
    """
    encoder_validity: np.ndarray           # [21] boolean - encoder read success
    joint_in_range: np.ndarray             # [21] boolean - within biomechanical limits
    velocity_smoothness: float             # [0, 1] - temporal consistency
    force_feedback_active: bool            # servo motors responding
    communication_latency_ms: float        # USB communication delay
    
    @property
    def valid_joint_ratio(self) -> float:
        """Fraction of joints with valid readings."""
        return float(np.mean(self.encoder_validity & self.joint_in_range))
    
    @property
    def overall_score(self) -> float:
        """
        Overall quality score [0, 1].
        
        Weights:
        - 50% valid joints
        - 30% velocity smoothness
        - 20% communication quality
        """
        comm_quality = max(0, 1.0 - self.communication_latency_ms / 50.0)  # Penalty if > 50ms
        
        return (
            0.50 * self.valid_joint_ratio +
            0.30 * self.velocity_smoothness +
            0.20 * comm_quality
        )
    
    @classmethod
    def default_good(cls) -> 'DYGloveQuality':
        """Create default quality metrics (assume good)."""
        return cls(
            encoder_validity=np.ones(21, dtype=bool),
            joint_in_range=np.ones(21, dtype=bool),
            velocity_smoothness=1.0,
            force_feedback_active=True,
            communication_latency_ms=0.5  # USB 3.2 target < 1ms (wired connection)
        )


@dataclass
class DYGloveState:
    """
    Full state from DYGlove hardware.
    
    21-DOF motion capture with optional force feedback data.
    """
    timestamp: float                         # Unix timestamp
    side: str                                # 'left' or 'right'
    
    # Core data: 21 joint angles in radians
    joint_angles: np.ndarray                 # [21] - see DYGloveJoint for indexing
    
    # Optional: joint velocities (computed from encoder deltas)
    joint_velocities: Optional[np.ndarray] = None   # [21] rad/s
    
    # Optional: encoder raw values (for debugging)
    encoder_raw: Optional[np.ndarray] = None        # [21] raw ADC values
    
    # Force feedback servo positions (for closed-loop haptics)
    servo_positions: Optional[np.ndarray] = None    # [5] servo angles
    servo_torques: Optional[np.ndarray] = None      # [5] estimated torques
    
    # Wrist orientation from external tracker (e.g., HTC Vive Tracker)
    wrist_position: Optional[np.ndarray] = None     # [3] xyz in tracker frame
    wrist_orientation: Optional[np.ndarray] = None  # [4] quaternion xyzw
    
    # Quality metrics
    quality: DYGloveQuality = field(default_factory=DYGloveQuality.default_good)
    
    def __post_init__(self):
        assert self.joint_angles.shape == (21,), "joint_angles must be [21]"
        assert self.side in ('left', 'right'), "side must be 'left' or 'right'"
    
    def get_finger_angles(self, finger: str) -> np.ndarray:
        """Get all joint angles for a specific finger."""
        joints = FINGER_JOINTS[finger]
        return self.joint_angles[joints]
    
    def get_flexion_angles(self) -> np.ndarray:
        """
        Get flexion angles only (no abduction), in order:
        [thumb_TM, thumb_MCP, thumb_IP, index_MCP, index_PIP, index_DIP, 
         middle_MCP, middle_PIP, middle_DIP, ring_MCP, ring_PIP, ring_DIP,
         pinky_MCP, pinky_PIP, pinky_DIP]
        
        Returns [15] array.
        """
        flexion_indices = [
            DYGloveJoint.THUMB_TM_FLEX, DYGloveJoint.THUMB_MCP, DYGloveJoint.THUMB_IP,
            DYGloveJoint.INDEX_MCP_FLEX, DYGloveJoint.INDEX_PIP, DYGloveJoint.INDEX_DIP,
            DYGloveJoint.MIDDLE_MCP_FLEX, DYGloveJoint.MIDDLE_PIP, DYGloveJoint.MIDDLE_DIP,
            DYGloveJoint.RING_MCP_FLEX, DYGloveJoint.RING_PIP, DYGloveJoint.RING_DIP,
            DYGloveJoint.PINKY_MCP_FLEX, DYGloveJoint.PINKY_PIP, DYGloveJoint.PINKY_DIP,
        ]
        return self.joint_angles[flexion_indices]
    
    def get_abduction_angles(self) -> np.ndarray:
        """
        Get abduction/adduction angles:
        [thumb_TM_abd, index_MCP_abd, middle_MCP_abd, ring_MCP_abd, pinky_MCP_abd]
        
        Returns [5] array.
        """
        abd_indices = [
            DYGloveJoint.THUMB_TM_ABD,
            DYGloveJoint.INDEX_MCP_ABD,
            DYGloveJoint.MIDDLE_MCP_ABD,
            DYGloveJoint.RING_MCP_ABD,
            DYGloveJoint.PINKY_MCP_ABD,
        ]
        return self.joint_angles[abd_indices]
    
    def get_finger_closure(self) -> np.ndarray:
        """
        Compute closure [0,1] for each finger.
        0 = fully extended, 1 = fully flexed.
        
        Returns [5] array for [thumb, index, middle, ring, pinky].
        """
        closures = np.zeros(5)
        
        # Thumb: TM_flex + MCP + IP
        thumb_max = np.array([1.22, 1.57, 1.40])  # Max flexion
        thumb_angles = np.array([
            self.joint_angles[DYGloveJoint.THUMB_TM_FLEX],
            self.joint_angles[DYGloveJoint.THUMB_MCP],
            self.joint_angles[DYGloveJoint.THUMB_IP],
        ])
        closures[0] = np.clip(np.mean(thumb_angles / thumb_max), 0, 1)
        
        # Other fingers: MCP_flex + PIP + DIP
        for i, finger in enumerate(['index', 'middle', 'ring', 'pinky']):
            joints = FINGER_JOINTS[finger]
            flex_joints = [joints[0], joints[2], joints[3]]  # MCP_flex, PIP, DIP
            max_flex = np.array([1.57, 1.75, 1.22])
            angles = self.joint_angles[flex_joints]
            closures[i + 1] = np.clip(np.mean(angles / max_flex), 0, 1)
        
        return closures
    
    def get_grasp_aperture(self) -> float:
        """
        Overall grasp aperture [0, 1].
        0 = fully open, 1 = fully closed.
        """
        closures = self.get_finger_closure()
        weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])  # Thumb and index weighted more
        return float(np.dot(weights, closures))


# =============================================================================
# Mapping to Legacy DexterHandState Format
# =============================================================================


# Mapping to Legacy DexterHandState Format
# =============================================================================

def dyglove_to_dexterhand(
    dyglove_state: DYGloveState,
    T_world_glove: Optional[np.ndarray] = None
) -> 'DexterHandState':
    """
    Convert DYGlove 21-DOF state to legacy DexterHandState format.
    
    Returns:
        DexterHandState compatible with existing pipeline
    """
    # Import here to avoid circular dependency - use relative import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.human_state import DexterHandState
    
    # Extract finger_angles [5, 3]
    finger_angles = np.zeros((5, 3))
    
    # Thumb: TM_flex, MCP, IP
    finger_angles[0] = [
        dyglove_state.joint_angles[DYGloveJoint.THUMB_TM_FLEX],
        dyglove_state.joint_angles[DYGloveJoint.THUMB_MCP],
        dyglove_state.joint_angles[DYGloveJoint.THUMB_IP],
    ]
    
    # Index through Pinky: MCP_flex, PIP, DIP
    for i, finger in enumerate(['index', 'middle', 'ring', 'pinky']):
        joints = FINGER_JOINTS[finger]
        finger_angles[i + 1] = [
            dyglove_state.joint_angles[joints[0]],  # MCP_flex
            dyglove_state.joint_angles[joints[2]],  # PIP
            dyglove_state.joint_angles[joints[3]],  # DIP
        ]
    
    # Compute inter-finger abduction (spread between adjacent fingers)
    abd = dyglove_state.get_abduction_angles()  # [thumb, index, middle, ring, pinky]
    finger_abduction = np.array([
        abd[0] - abd[1],  # thumb-index spread
        abd[1] - abd[2],  # index-middle spread  
        abd[2] - abd[3],  # middle-ring spread
        abd[3] - abd[4],  # ring-pinky spread
    ])
    
    # Wrist quaternion
    if dyglove_state.wrist_orientation is not None:
        wrist_quat = dyglove_state.wrist_orientation
    else:
        wrist_quat = np.array([0, 0, 0, 1])  # Identity quaternion
    
    # Fingertip forces from servo torques if available
    fingertip_forces = None
    if dyglove_state.servo_torques is not None:
        fingertip_forces = dyglove_state.servo_torques
    
    return DexterHandState(
        timestamp=dyglove_state.timestamp,
        side=dyglove_state.side,
        finger_angles=finger_angles,
        finger_abduction=finger_abduction,
        wrist_quat_local=wrist_quat,
        fingertip_forces=fingertip_forces,
        T_world_glove_imu=T_world_glove
    )


# =============================================================================
# DYGlove Hardware Driver
# =============================================================================

class DYGloveDriverBase(ABC):
    """Abstract base class for DYGlove hardware communication."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the glove hardware."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the glove hardware."""
        pass
    
    @abstractmethod
    def read_state(self) -> Optional[DYGloveState]:
        """Read current state from the glove."""
        pass
    
    @abstractmethod
    def set_force_feedback(self, forces: np.ndarray) -> bool:
        """Set haptic force feedback on the servo motors."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if glove is connected."""
        pass



class DYGloveDriver(DYGloveDriverBase):
    """
    Real hardware driver for DYGlove (Wired USB/Ethernet).

    Connection Methods (in order of preference):
    1. USB 3.2 Gen 2: Direct USB connection via /dev/ttyUSB* or /dev/ttyACM*
    2. Ethernet over USB-C: For industrial deployments with deterministic timing

    NO WiFi support - removed for reliability and deterministic timing.
    Uses the DYGloveSDKClient for communication.
    """

    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        side: str = 'right',
        simulation_mode: bool = False,
        baudrate: int = 921600,
        timeout_ms: float = 10.0,
    ):
        """
        Initialize DYGlove driver with wired connection.

        Args:
            port: USB serial port (e.g., /dev/ttyUSB0, /dev/ttyACM0)
            side: 'left' or 'right' hand
            simulation_mode: If True, uses simulated data
            baudrate: Serial baudrate (default 921600 for USB 3.2)
            timeout_ms: Read timeout in milliseconds
        """
        self.port = port
        self.side = side
        self.simulation_mode = simulation_mode
        self.baudrate = baudrate
        self.timeout_ms = timeout_ms
        
        # Use the SDK Client
        from src.platform.edge.dyglove_sdk import DYGloveSDKClient
        self.client = DYGloveSDKClient()
        
        self._connected = False
        self._latest_state: Optional[DYGloveState] = None
        self._lock = threading.Lock()
        self._start_time = time.time()
        
    @property
    def is_connected(self) -> bool:
        return self.client.is_connected
    
    def connect(self) -> bool:
        """Connect to DYGlove hardware."""
        if self.simulation_mode:
            logger.info(f"DYGlove ({self.side}) connected in SIMULATION mode.")
            self._connected = True
            return True

        # For now, we assume the SDK handles the connection details.
        # If using WiFi, we might need to pass IP/Port to the SDK's connect method if it supports it,
        # or use discovery.
        # The SDK currently supports Serial/Bluetooth discovery.
        # If we stick to the SDK's discovery:
        success = self.client.connect()
        if success:
            logger.info(f"DYGlove connected via SDK.")
            self._connected = True
        else:
            logger.error("Failed to connect DYGlove via SDK.")
            self._connected = False
        return success
    
    def disconnect(self) -> None:
        """Disconnect from DYGlove hardware."""
        self.client.disconnect()
        self._connected = False
        print("DYGlove disconnected")
    
    def read_state(self) -> Optional[DYGloveState]:
        """
        Read current state from the glove.
        """
        if not self._connected:
            return None
        
        try:
            # Get state from SDK
            sdk_state = self.client.get_state()
            
            # Convert SDK state to driver DYGloveState
            # SDK State has: thumb, index, etc. (FingerState)
            # Driver DYGloveState needs: joint_angles [21]
            
            joint_angles = np.zeros(21)
            
            # Thumb
            joint_angles[DYGloveJoint.THUMB_TM_FLEX] = np.radians(sdk_state.thumb.mcp) # Mapping approx
            joint_angles[DYGloveJoint.THUMB_MCP] = np.radians(sdk_state.thumb.pip)
            joint_angles[DYGloveJoint.THUMB_IP] = np.radians(sdk_state.thumb.dip)
            
            # Fingers
            fingers = [sdk_state.index, sdk_state.middle, sdk_state.ring, sdk_state.pinky]
            finger_bases = [5, 9, 13, 17] # Start indices for index, middle, ring, pinky
            
            for i, finger in enumerate(fingers):
                base = finger_bases[i]
                # SDK gives MCP, PIP, DIP angles in degrees
                # Driver expects radians
                # Mapping:
                # SDK MCP -> Driver MCP_FLEX
                # SDK PIP -> Driver PIP
                # SDK DIP -> Driver DIP
                # Driver MCP_ABD is not in SDK simple FingerState, assume 0 for now
                
                joint_angles[base] = np.radians(finger.mcp)      # MCP_FLEX
                joint_angles[base + 1] = 0.0                     # MCP_ABD
                joint_angles[base + 2] = np.radians(finger.pip)  # PIP
                joint_angles[base + 3] = np.radians(finger.dip)  # DIP
            
            # Create state object
            state = DYGloveState(
                timestamp=time.time(),
                side=self.side,
                joint_angles=joint_angles,
                quality=DYGloveQuality.default_good() # Assume good for now
            )
            
            return state
            
        except Exception as e:
            # print(f"Error reading DYGlove: {e}") 
            return None
    
    def set_force_feedback(self, forces: np.ndarray) -> bool:
        """
        Set haptic force feedback.
        """
        if not self._connected:
            return False
            
        # Convert forces array to list for SDK
        return self.client.set_haptic_feedback(intensities=forces.tolist())
    
    def calibrate(self, neutral_pose: bool = True):
        """
        Calibrate encoder offsets.
        """
        if not self._connected:
            return
        
        self.client.calibrate()



class DYGloveEthernetDriver(DYGloveDriverBase):
    """
    Wired Ethernet driver for DYGlove (industrial deployment).

    Uses Ethernet over USB-C for deterministic timing in factory environments.
    Provides hardware-level timestamps via PTP (IEEE 1588) when available.

    Protocol:
    - UDP (Port 9876): High-frequency state streaming (Glove -> Host)
    - TCP (Port 9877): Reliable control/config (Host <-> Glove)
    """

    UDP_PORT = 9876
    TCP_PORT = 9877

    def __init__(self, ip_address: str, side: str = 'right'):
        """
        Initialize wired Ethernet driver.

        Args:
            ip_address: IP address of the glove (e.g., 192.168.100.10)
            side: 'left' or 'right' hand
        """
        self.ip_address = ip_address
        self.side = side
        
        self._connected = False
        self._udp_sock = None
        self._tcp_sock = None
        
        self._latest_state: Optional[DYGloveState] = None
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._listen_thread = None
        
        # Jitter buffer / stats
        self._last_packet_time = 0
        self._packet_count = 0
    
    @property
    def is_connected(self) -> bool:
        return self._connected
        
    def connect(self) -> bool:
        """Connect to DYGlove via wired Ethernet."""
        try:
            print(f"Connecting to DYGlove at {self.ip_address} via Ethernet...")
            
            # 1. Setup UDP Listener
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # QoS: Set IP_TOS to Low Delay (0x10) or DSCP EF (0xB8)
            # 0x10 = IPTOS_LOWDELAY
            # 0xB8 = DSCP EF (Expedited Forwarding) - often better for real-time
            try:
                self._udp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
            except Exception as e:
                print(f"Warning: Could not set QoS flags: {e}")

            self._udp_sock.bind(('0.0.0.0', 0)) # Bind to ephemeral port
            self._udp_sock.settimeout(1.0)
            
            # 2. Setup TCP Control Connection
            self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_sock.settimeout(3.0)
            self._tcp_sock.connect((self.ip_address, self.TCP_PORT))
            
            # 3. Send "Start Streaming" command via TCP
            # Tell glove where to send UDP packets
            local_ip = self._get_local_ip()
            local_port = self._udp_sock.getsockname()[1]
            
            cmd = {
                "cmd": "start_stream",
                "udp_ip": local_ip,
                "udp_port": local_port
            }
            self._send_tcp_json(cmd)
            
            # 4. Start Listener Thread
            self._stop_event.clear()
            self._listen_thread = threading.Thread(target=self._udp_loop, daemon=True)
            self._listen_thread.start()
            
            self._connected = True
            print(f"Connected to DYGlove ({self.side}) via Ethernet")
            return True
            
        except Exception as e:
            print(f"Failed to connect to DYGlove via Ethernet: {e}")
            self.disconnect()
            return False
            
    def disconnect(self) -> None:
        """Disconnect and stop streaming."""
        self._connected = False
        self._stop_event.set()
        
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)
            
        try:
            if self._tcp_sock:
                self._send_tcp_json({"cmd": "stop_stream"})
                self._tcp_sock.close()
        except:
            pass
            
        if self._udp_sock:
            self._udp_sock.close()
            
        self._tcp_sock = None
        self._udp_sock = None
        
    def read_state(self) -> Optional[DYGloveState]:
        """Return latest state received via UDP."""
        with self._state_lock:
            return self._latest_state
            
    def set_force_feedback(self, forces: np.ndarray) -> bool:
        """Send force feedback command via UDP (or TCP if reliable needed)."""
        # For low latency, we use UDP for haptics too, or TCP if packet loss is high.
        # Here we'll use UDP for speed.
        if not self._connected or not self._udp_sock:
            return False
            
        try:
            # Packet: [Header, F1, F2, F3, F4, F5, Checksum]
            # Simple mapping: 0-255 PWM value
            pwm_values = np.clip(forces * 50, 0, 255).astype(np.uint8)
            packet = struct.pack('<BBBBBBB', 0xAA, *pwm_values, 0x55)
            self._udp_sock.sendto(packet, (self.ip_address, self.UDP_PORT))
            return True
        except Exception as e:
            print(f"Error sending haptics: {e}")
            return False

    def _udp_loop(self):
        """Receive UDP state packets."""
        while not self._stop_event.is_set():
            try:
                data, _ = self._udp_sock.recvfrom(1024)
                state = self._parse_udp_packet(data)
                if state:
                    with self._state_lock:
                        # Jitter Buffer / Smoothing
                        # If we receive packets out of order, discard older ones
                        if self._latest_state and state.timestamp < self._latest_state.timestamp:
                            continue
                            
                        self._latest_state = state
                        
                        # Update stats
                        now = time.time()
                        if self._last_packet_time > 0:
                            jitter = abs((now - self._last_packet_time) - (1.0/120.0)) # Assuming 120Hz
                            # We could log jitter here
                        self._last_packet_time = now
                        self._packet_count += 1
                        
            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"UDP Receive Error: {e}")
                break
                
    def _parse_udp_packet(self, data: bytes) -> Optional[DYGloveState]:
        """
        Parse binary UDP packet.
        Format:
        - Header (1B): 0xBB
        - Seq (1B)
        - Timestamp (4B float)
        - Joints (21 * 2B int16, scaled by 1000)
        - IMU (4B Quat + 3B Accel) - Simplified for now
        - Battery (1B)
        - Footer (1B): 0xEE
        Total: ~50-60 bytes
        """
        if len(data) < 48: # Min size check
            return None
            
        try:
            # Unpack header
            if data[0] != 0xBB:
                return None
                
            # Parse joints (21 int16s)
            # Offset 6: Header(1)+Seq(1)+Time(4)
            joint_data = struct.unpack('<21h', data[6:48])
            joint_angles = np.array(joint_data, dtype=np.float32) / 1000.0
            
            # Construct state
            return DYGloveState(
                timestamp=time.time(),
                side=self.side,
                joint_angles=joint_angles,
                quality=DYGloveQuality.default_good() # TODO: Parse battery/link quality
            )
        except Exception:
            return None

    def _send_tcp_json(self, data: dict):
        """Send JSON command over TCP."""
        if not self._tcp_sock:
            return
        msg = json.dumps(data).encode('utf-8') + b'\n'
        self._tcp_sock.sendall(msg)
        
    def _get_local_ip(self):
        """Get local IP address."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    def read_state(self) -> Optional[DYGloveState]:
        if not self._connected:
            return None
        
        t = time.time() - self._start_time
        timestamp = time.time()
        
        # Generate sinusoidal motion for testing
        joint_angles = np.zeros(21)
        
        # Simulate a reaching-grasping motion
        grasp_phase = 0.5 * (1 + np.sin(2 * np.pi * 0.2 * t))  # 0.2 Hz grasp cycle
        
        for finger_idx, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
            joints = FINGER_JOINTS[finger]
            
            # Phase offset per finger for natural motion
            phase = grasp_phase * (1 + 0.1 * finger_idx)
            
            if finger == 'thumb':
                joint_angles[joints[0]] = 0.8 * phase  # TM flex
                joint_angles[joints[1]] = 0.2 * np.sin(2 * np.pi * 0.3 * t)  # TM abd
                joint_angles[joints[2]] = 1.2 * phase  # MCP
                joint_angles[joints[3]] = 1.0 * phase  # IP
                joint_angles[joints[4]] = 0.3 * np.sin(2 * np.pi * 0.1 * t)  # Wrist PS
            else:
                joint_angles[joints[0]] = 1.4 * phase  # MCP flex
                joint_angles[joints[1]] = 0.15 * np.sin(2 * np.pi * 0.15 * t + finger_idx)  # MCP abd
                joint_angles[joints[2]] = 1.6 * phase  # PIP
                joint_angles[joints[3]] = 1.0 * phase  # DIP
        
        # Add small noise
        joint_angles += np.random.normal(0, 0.02, 21)
        
        # Clip to limits
        for joint in DYGloveJoint:
            lo, hi = DYGLOVE_JOINT_LIMITS[joint]
            joint_angles[joint] = np.clip(joint_angles[joint], lo, hi)
        
        # Compute velocities
        joint_velocities = None
        if self._prev_angles is not None and self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 0:
                joint_velocities = (joint_angles - self._prev_angles) / dt
        
        self._prev_angles = joint_angles.copy()
        self._prev_timestamp = timestamp
        
        return DOGloveState(
            timestamp=timestamp,
            side=self.side,
            joint_angles=joint_angles,
            joint_velocities=joint_velocities,
            quality=DYGloveQuality.default_good()
        )
    
    def set_force_feedback(self, forces: np.ndarray) -> bool:
        # Simulated - just log
        return True


# =============================================================================
# DOGlove Quality Filter
# =============================================================================

@dataclass
class DYGloveQualityConfig:
    """Configuration for DYGlove quality filtering."""
    
    # Minimum overall quality score to accept frame
    min_quality_score: float = 0.6
    
    # Minimum fraction of valid joints
    min_valid_joint_ratio: float = 0.8
    
    # Maximum acceptable communication latency (ms)
    max_latency_ms: float = 50.0
    
    # Velocity smoothness threshold
    min_smoothness: float = 0.5
    
    # Joint velocity limit for spike detection (rad/s)
    max_joint_velocity: float = 15.0
    
    # Enable/disable specific checks
    check_encoder_validity: bool = True
    check_joint_limits: bool = True
    check_velocity: bool = True
    check_latency: bool = True


class DYGloveQualityFilter:
    """
    Real-time quality filter for DYGlove data.
    
    Filters out bad frames and optionally applies smoothing.
    """
    
    def __init__(self, config: Optional[DYGloveQualityConfig] = None):
        self.config = config or DYGloveQualityConfig()
        
        # History for temporal filtering
        self._history: List[DOGloveState] = []
        self._max_history = 10
        
        # Statistics
        self._total_frames = 0
        self._accepted_frames = 0
        self._rejection_reasons: Dict[str, int] = {
            'low_quality': 0,
            'invalid_joints': 0,
            'high_latency': 0,
            'velocity_spike': 0,
        }
    
    def filter(self, state: DYGloveState) -> Tuple[bool, Optional[DYGloveState]]:
        """
        Filter a DYGlove state for quality.
        
        Args:
            state: Raw DYGlove state
        
        Returns:
            (accepted, filtered_state) - if accepted=False, filtered_state is None
        """
        self._total_frames += 1
        quality = state.quality
        
        # Check overall quality
        if quality.overall_score < self.config.min_quality_score:
            self._rejection_reasons['low_quality'] += 1
            return False, None
        
        # Check valid joint ratio
        if self.config.check_encoder_validity:
            if quality.valid_joint_ratio < self.config.min_valid_joint_ratio:
                self._rejection_reasons['invalid_joints'] += 1
                return False, None
        
        # Check latency
        if self.config.check_latency:
            if quality.communication_latency_ms > self.config.max_latency_ms:
                self._rejection_reasons['high_latency'] += 1
                return False, None
        
        # Check velocity spikes
        if self.config.check_velocity and state.joint_velocities is not None:
            if np.any(np.abs(state.joint_velocities) > self.config.max_joint_velocity):
                self._rejection_reasons['velocity_spike'] += 1
                return False, None
        
        # Frame accepted - optionally apply smoothing
        self._accepted_frames += 1
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # Apply temporal smoothing
        smoothed_state = self._apply_smoothing(state)
        
        return True, smoothed_state
    
    def _apply_smoothing(self, state: DYGloveState) -> DYGloveState:
        """Apply exponential moving average smoothing."""
        if len(self._history) < 2:
            return state
        
        alpha = 0.7  # Smoothing factor (higher = less smoothing)
        
        prev_angles = self._history[-2].joint_angles
        smoothed_angles = alpha * state.joint_angles + (1 - alpha) * prev_angles
        
        # Create new state with smoothed angles
        return DOGloveState(
            timestamp=state.timestamp,
            side=state.side,
            joint_angles=smoothed_angles,
            joint_velocities=state.joint_velocities,
            encoder_raw=state.encoder_raw,
            servo_positions=state.servo_positions,
            servo_torques=state.servo_torques,
            wrist_position=state.wrist_position,
            wrist_orientation=state.wrist_orientation,
            quality=state.quality
        )
    
    def get_statistics(self) -> Dict:
        """Get filtering statistics."""
        acceptance_rate = self._accepted_frames / max(1, self._total_frames)
        return {
            'total_frames': self._total_frames,
            'accepted_frames': self._accepted_frames,
            'acceptance_rate': acceptance_rate,
            'rejection_reasons': dict(self._rejection_reasons),
        }
    
    def reset_statistics(self):
        """Reset filtering statistics."""
        self._total_frames = 0
        self._accepted_frames = 0
        for key in self._rejection_reasons:
            self._rejection_reasons[key] = 0


# =============================================================================
# Async Reader for Real-time Operation
# =============================================================================

class DYGloveAsyncReader:
    """
    Asynchronous DYGlove reader with quality filtering.
    
    Runs in a background thread, providing latest valid state on demand.
    """
    
    def __init__(
        self,
        driver: DYGloveDriverBase,
        quality_config: Optional[DYGloveQualityConfig] = None,
        target_hz: float = 100.0
    ):
        self.driver = driver
        self.filter = DOGloveQualityFilter(quality_config)
        self.target_hz = target_hz
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_queue: queue.Queue = queue.Queue(maxsize=10)
        self._latest_state: Optional[DOGloveState] = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start async reading."""
        if not self.driver.is_connected:
            if not self.driver.connect():
                return False
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True
    
    def stop(self):
        """Stop async reading."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.driver.disconnect()
    
    def _read_loop(self):
        """Background reading loop."""
        period = 1.0 / self.target_hz
        
        while self._running:
            t_start = time.time()
            
            state = self.driver.read_state()
            if state is not None:
                accepted, filtered_state = self.filter.filter(state)
                if accepted and filtered_state is not None:
                    with self._lock:
                        self._latest_state = filtered_state
                    
                    # Also add to queue for consumers that want all frames
                    try:
                        self._state_queue.put_nowait(filtered_state)
                    except queue.Full:
                        pass  # Drop oldest
            
            # Maintain target rate
            elapsed = time.time() - t_start
            if elapsed < period:
                time.sleep(period - elapsed)
    
    def get_latest_state(self) -> Optional[DYGloveState]:
        """Get the most recent valid state."""
        with self._lock:
            return self._latest_state
    
    def get_state_blocking(self, timeout: float = 1.0) -> Optional[DYGloveState]:
        """Get next state, blocking until available."""
        try:
            return self._state_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# =============================================================================
# Forward Kinematics for Visualization
# =============================================================================

class DYGloveFK:
    """
    Forward kinematics for DYGlove visualization.
    
    Computes fingertip positions given joint angles.
    Based on human hand anthropometry.
    """
    
    # Link lengths in meters (approximate human hand)
    LINK_LENGTHS = {
        'thumb': {
            'metacarpal': 0.040,
            'proximal': 0.030,
            'distal': 0.025,
        },
        'index': {
            'metacarpal': 0.065,
            'proximal': 0.040,
            'intermediate': 0.025,
            'distal': 0.020,
        },
        'middle': {
            'metacarpal': 0.060,
            'proximal': 0.045,
            'intermediate': 0.028,
            'distal': 0.022,
        },
        'ring': {
            'metacarpal': 0.055,
            'proximal': 0.040,
            'intermediate': 0.025,
            'distal': 0.020,
        },
        'pinky': {
            'metacarpal': 0.048,
            'proximal': 0.032,
            'intermediate': 0.020,
            'distal': 0.018,
        },
    }
    
    # MCP joint positions relative to wrist (palm layout)
    MCP_POSITIONS = {
        'thumb': np.array([0.02, 0.03, 0.0]),
        'index': np.array([0.03, 0.08, 0.0]),
        'middle': np.array([0.01, 0.085, 0.0]),
        'ring': np.array([-0.01, 0.08, 0.0]),
        'pinky': np.array([-0.03, 0.07, 0.0]),
    }
    
    def __init__(self):
        pass
    
    def compute_fingertip_positions(
        self,
        state: DYGloveState,
        wrist_pose: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions of all fingertips.
        
        Args:
            state: DOGlove joint angles
            wrist_pose: Optional 4x4 transform for wrist pose (default: identity)
        
        Returns:
            Dict mapping finger name to [3] position array
        """
        if wrist_pose is None:
            wrist_pose = np.eye(4)
        
        fingertips = {}
        
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            fingertips[finger] = self._compute_finger_fk(
                finger, state, wrist_pose
            )
        
        return fingertips
    
    def _compute_finger_fk(
        self,
        finger: str,
        state: DYGloveState,
        wrist_pose: np.ndarray
    ) -> np.ndarray:
        """Compute FK for a single finger."""
        joints = FINGER_JOINTS[finger]
        angles = state.joint_angles[joints]
        
        # Start at MCP position
        pos = self.MCP_POSITIONS[finger].copy()
        
        if finger == 'thumb':
            # Thumb: TM_flex, TM_abd, MCP, IP, wrist_PS
            # Simplified: just chain the rotations
            lengths = self.LINK_LENGTHS['thumb']
            
            # TM joint
            tm_flex, tm_abd = angles[0], angles[1]
            pos += lengths['metacarpal'] * np.array([
                np.sin(tm_abd) * np.cos(tm_flex),
                np.cos(tm_abd) * np.cos(tm_flex),
                np.sin(tm_flex)
            ])
            
            # MCP + IP
            mcp, ip = angles[2], angles[3]
            pos += lengths['proximal'] * np.array([0, np.cos(mcp), np.sin(mcp)])
            pos += lengths['distal'] * np.array([0, np.cos(mcp + ip), np.sin(mcp + ip)])
            
        else:
            # Other fingers: MCP_flex, MCP_abd, PIP, DIP
            lengths = self.LINK_LENGTHS[finger]
            
            mcp_flex, mcp_abd = angles[0], angles[1]
            pip, dip = angles[2], angles[3]
            
            # Proximal phalanx
            pos += lengths['proximal'] * np.array([
                np.sin(mcp_abd),
                np.cos(mcp_abd) * np.cos(mcp_flex),
                np.sin(mcp_flex)
            ])
            
            # Intermediate phalanx
            total_flex = mcp_flex + pip
            pos += lengths['intermediate'] * np.array([
                0,
                np.cos(total_flex),
                np.sin(total_flex)
            ])
            
            # Distal phalanx
            total_flex += dip
            pos += lengths['distal'] * np.array([
                0,
                np.cos(total_flex),
                np.sin(total_flex)
            ])
        
        # Transform to world frame
        pos_homogeneous = np.append(pos, 1)
        world_pos = wrist_pose @ pos_homogeneous
        
        return world_pos[:3]


# =============================================================================
# Integration with HumanState Pipeline
# =============================================================================

def integrate_dyglove_with_human_state(
    human3d_state: 'Human3DState',
    dyglove_right: Optional[DYGloveState],
    dyglove_left: Optional[DYGloveState],
    T_world_tracker_right: Optional[np.ndarray] = None,
    T_world_tracker_left: Optional[np.ndarray] = None,
) -> Tuple[Optional['DexterHandState'], Optional['DexterHandState']]:
    """
    Integrate DYGlove data with the existing HumanState pipeline.
    
    Converts DYGlove states to DexterHandState format and aligns with
    camera-observed wrist positions.
    
    Args:
        human3d_state: Body pose from camera
        dyglove_right: Right DYGlove state (or None)
        dyglove_left: Left DYGlove state (or None)
        T_world_tracker_right: Transform from right tracker to world
        T_world_tracker_left: Transform from left tracker to world
    
    Returns:
        (hand_right, hand_left) in DexterHandState format
    """
    hand_right = None
    hand_left = None
    
    if dyglove_right is not None:
        # Use camera wrist position to anchor glove data
        wrist_pos = human3d_state.wrist_right_position
        
        # If we have tracker data, use it for orientation
        if dyglove_right.wrist_orientation is not None and T_world_tracker_right is not None:
            T_glove = T_world_tracker_right
        else:
            # Estimate orientation from arm direction
            arm_dir = human3d_state.get_arm_direction('right')
            T_glove = _estimate_wrist_transform(wrist_pos, arm_dir)
        
        hand_right = dyglove_to_dexterhand(dyglove_right, T_glove)
    
    if dyglove_left is not None:
        wrist_pos = human3d_state.wrist_left_position
        
        if dyglove_left.wrist_orientation is not None and T_world_tracker_left is not None:
            T_glove = T_world_tracker_left
        else:
            arm_dir = human3d_state.get_arm_direction('left')
            T_glove = _estimate_wrist_transform(wrist_pos, arm_dir)
        
        hand_left = dyglove_to_dexterhand(dyglove_left, T_glove)
    
    return hand_right, hand_left


def _estimate_wrist_transform(wrist_pos: np.ndarray, arm_direction: np.ndarray) -> np.ndarray:
    """
    Estimate wrist transform from position and arm direction.
    
    Creates a 4x4 transform where:
    - Translation = wrist position
    - Z-axis points along arm direction (distal)
    - Y-axis points "up" (dorsal)
    """
    z_axis = arm_direction / np.linalg.norm(arm_direction)
    
    # Assume rough "up" direction
    up = np.array([0, 0, 1])
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 0.1:
        up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = wrist_pos
    
    return T


# =============================================================================
# Factory Functions
# =============================================================================

def create_dyglove_driver(
    use_simulator: bool = False,
    side: str = 'right',
    port: str = '/dev/ttyUSB0',
    **kwargs
) -> DYGloveDriverBase:
    """
    Factory function to create DYGlove driver.
    
    Args:
        use_simulator: If True, use simulator instead of real hardware
        side: 'left' or 'right'
        port: USB serial port (for real hardware)
    
    Returns:
        DYGlove driver instance
    """
    if use_simulator:
        # Assuming DYGloveSimulator is defined or imported, but for now we use DYGloveDriver with sim mode
        return DYGloveDriver(side=side, simulation_mode=True)
    else:
        return DYGloveDriver(port=port, side=side, **kwargs)


def create_dyglove_reader(
    use_simulator: bool = False,
    side: str = 'right',
    port: str = '/dev/ttyUSB0',
    quality_config: Optional[DYGloveQualityConfig] = None,
    target_hz: float = 100.0
) -> DYGloveAsyncReader:
    """
    Factory function to create async DYGlove reader with quality filtering.
    
    Args:
        use_simulator: If True, use simulator
        side: 'left' or 'right'
        port: USB serial port
        quality_config: Quality filtering configuration
        target_hz: Target reading rate
    
    Returns:
        Configured async reader
    """
    driver = create_dyglove_driver(use_simulator, side, port)
    return DYGloveAsyncReader(driver, quality_config, target_hz)

"""
Virtual DYGlove Simulator

Simulates the DYGlove haptic glove for teleoperation without physical hardware.
Supports keyboard/mouse input, scripted trajectories, and gamepad input.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """Input mode for virtual glove."""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GAMEPAD = "gamepad"
    SCRIPTED = "scripted"
    PLAYBACK = "playback"


@dataclass
class GloveConfig:
    """Configuration for virtual glove."""
    # Update rate
    update_rate: float = 1000.0  # Hz (matches real DYGlove)

    # Workspace limits (meters)
    workspace_min: Tuple[float, float, float] = (-0.5, -0.5, 0.0)
    workspace_max: Tuple[float, float, float] = (0.5, 0.5, 1.0)

    # Sensitivity
    position_sensitivity: float = 0.01
    rotation_sensitivity: float = 0.05
    finger_sensitivity: float = 0.1

    # Smoothing
    smoothing_alpha: float = 0.3

    # Noise (for realism)
    enable_noise: bool = True
    position_noise_std: float = 0.001
    rotation_noise_std: float = 0.01
    finger_noise_std: float = 0.02

    # Input mode
    input_mode: InputMode = InputMode.KEYBOARD


@dataclass
class GloveState:
    """Current state of virtual glove."""
    # Hand position (meters, in glove frame)
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))

    # Hand orientation (quaternion wxyz)
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    # Euler angles for convenience (roll, pitch, yaw in radians)
    euler_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Finger joint angles (21 DOF: 4 joints x 5 fingers + thumb rotation)
    finger_angles: np.ndarray = field(default_factory=lambda: np.zeros(21))

    # IMU data
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 9.81]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Grip state (0=open, 1=closed)
    grip_state: float = 0.0

    # Timestamp
    timestamp: float = 0.0

    # Connection status
    connected: bool = True


class VirtualDYGlove:
    """
    Virtual DYGlove simulator for teleoperation.

    Simulates:
    - Hand position and orientation tracking
    - 21-DOF finger joint tracking
    - IMU data (accelerometer, gyroscope)
    - Haptic feedback simulation

    Input modes:
    - Keyboard: WASD for position, QE for rotation, Space for grip
    - Mouse: Mouse movement for position, scroll for height
    - Gamepad: Analog sticks for position/rotation
    - Scripted: Pre-programmed trajectories
    - Playback: Replay recorded demonstrations
    """

    # Finger joint mapping
    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
    JOINTS_PER_FINGER = 4  # MCP, PIP, DIP, TIP (or equivalent)

    def __init__(self, config: Optional[GloveConfig] = None):
        """Initialize virtual glove."""
        self.config = config or GloveConfig()

        # State
        self._state = GloveState()
        self._target_state = GloveState()  # For smoothing

        # Timing
        self._last_update_time = time.time()
        self._update_interval = 1.0 / self.config.update_rate

        # Input state
        self._keyboard_state: Dict[str, bool] = {}
        self._mouse_position = np.array([0.0, 0.0])
        self._mouse_scroll = 0.0

        # Scripted trajectory
        self._trajectory: List[GloveState] = []
        self._trajectory_index = 0
        self._trajectory_start_time = 0.0

        # Playback buffer
        self._playback_buffer: List[Dict] = []
        self._playback_index = 0

        # Callbacks
        self._on_state_callbacks: List[Callable[[GloveState], None]] = []

        # Running state
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info("VirtualDYGlove initialized")

    async def start(self) -> None:
        """Start virtual glove."""
        if self._running:
            return

        self._running = True
        self._last_update_time = time.time()
        self._state.connected = True

        # Start update loop
        self._update_task = asyncio.create_task(self._update_loop())

        logger.info("Virtual glove started")

    async def stop(self) -> None:
        """Stop virtual glove."""
        self._running = False
        self._state.connected = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("Virtual glove stopped")

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._running:
            await self._update()
            await asyncio.sleep(self._update_interval)

    async def _update(self) -> None:
        """Update glove state."""
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        # Update based on input mode
        if self.config.input_mode == InputMode.KEYBOARD:
            self._update_from_keyboard(dt)
        elif self.config.input_mode == InputMode.MOUSE:
            self._update_from_mouse(dt)
        elif self.config.input_mode == InputMode.SCRIPTED:
            self._update_from_trajectory(now)
        elif self.config.input_mode == InputMode.PLAYBACK:
            self._update_from_playback(now)

        # Apply smoothing
        self._apply_smoothing()

        # Add noise if enabled
        if self.config.enable_noise:
            self._add_noise()

        # Update timestamp
        self._state.timestamp = now

        # Trigger callbacks
        for callback in self._on_state_callbacks:
            try:
                callback(self._state)
            except Exception as e:
                logger.error(f"Glove callback error: {e}")

    def _update_from_keyboard(self, dt: float) -> None:
        """Update state from keyboard input."""
        sens = self.config.position_sensitivity

        # Position (WASD + QE for up/down)
        delta = np.zeros(3)
        if self._keyboard_state.get('w'):
            delta[0] += sens
        if self._keyboard_state.get('s'):
            delta[0] -= sens
        if self._keyboard_state.get('a'):
            delta[1] += sens
        if self._keyboard_state.get('d'):
            delta[1] -= sens
        if self._keyboard_state.get('q'):
            delta[2] += sens
        if self._keyboard_state.get('e'):
            delta[2] -= sens

        self._target_state.position = np.clip(
            self._target_state.position + delta,
            self.config.workspace_min,
            self.config.workspace_max,
        )

        # Rotation (arrow keys)
        rot_sens = self.config.rotation_sensitivity
        euler = self._target_state.euler_angles.copy()

        if self._keyboard_state.get('up'):
            euler[0] += rot_sens  # Pitch
        if self._keyboard_state.get('down'):
            euler[0] -= rot_sens
        if self._keyboard_state.get('left'):
            euler[2] += rot_sens  # Yaw
        if self._keyboard_state.get('right'):
            euler[2] -= rot_sens

        self._target_state.euler_angles = euler
        self._target_state.orientation = self._euler_to_quaternion(euler)

        # Grip (space)
        if self._keyboard_state.get('space'):
            self._target_state.grip_state = min(1.0, self._target_state.grip_state + 0.1)
            self._update_finger_grip(self._target_state.grip_state)
        else:
            self._target_state.grip_state = max(0.0, self._target_state.grip_state - 0.1)
            self._update_finger_grip(self._target_state.grip_state)

    def _update_from_mouse(self, dt: float) -> None:
        """Update state from mouse input."""
        sens = self.config.position_sensitivity

        # Mouse position -> XY
        self._target_state.position[0] = np.clip(
            self._mouse_position[0] * sens,
            self.config.workspace_min[0],
            self.config.workspace_max[0],
        )
        self._target_state.position[1] = np.clip(
            self._mouse_position[1] * sens,
            self.config.workspace_min[1],
            self.config.workspace_max[1],
        )

        # Scroll -> Z
        self._target_state.position[2] = np.clip(
            0.5 + self._mouse_scroll * 0.1,
            self.config.workspace_min[2],
            self.config.workspace_max[2],
        )

    def _update_from_trajectory(self, now: float) -> None:
        """Update state from scripted trajectory."""
        if not self._trajectory:
            return

        if self._trajectory_index >= len(self._trajectory):
            self._trajectory_index = 0
            self._trajectory_start_time = now

        # Interpolate between waypoints
        elapsed = now - self._trajectory_start_time
        waypoint_duration = 1.0  # 1 second per waypoint

        current_idx = int(elapsed / waypoint_duration) % len(self._trajectory)
        next_idx = (current_idx + 1) % len(self._trajectory)
        t = (elapsed % waypoint_duration) / waypoint_duration

        current = self._trajectory[current_idx]
        next_wp = self._trajectory[next_idx]

        # Linear interpolation
        self._target_state.position = (1 - t) * current.position + t * next_wp.position
        self._target_state.grip_state = (1 - t) * current.grip_state + t * next_wp.grip_state
        self._target_state.finger_angles = (1 - t) * current.finger_angles + t * next_wp.finger_angles

        # SLERP for orientation
        self._target_state.orientation = self._slerp(
            current.orientation, next_wp.orientation, t
        )

    def _update_from_playback(self, now: float) -> None:
        """Update state from playback buffer."""
        if not self._playback_buffer:
            return

        if self._playback_index >= len(self._playback_buffer):
            self._playback_index = 0

        frame = self._playback_buffer[self._playback_index]
        self._playback_index += 1

        # Load state from frame
        self._target_state.position = np.array(frame.get("position", [0, 0, 0.5]))
        self._target_state.orientation = np.array(frame.get("orientation", [1, 0, 0, 0]))
        self._target_state.finger_angles = np.array(frame.get("finger_angles", np.zeros(21)))
        self._target_state.grip_state = frame.get("grip_state", 0.0)

    def _update_finger_grip(self, grip: float) -> None:
        """Update finger angles based on grip state."""
        # Simple grip model: all fingers curl proportionally
        for i in range(5):  # 5 fingers
            base_idx = i * 4
            # Each finger has 4 joints
            self._target_state.finger_angles[base_idx] = grip * 0.5  # MCP
            self._target_state.finger_angles[base_idx + 1] = grip * 1.2  # PIP
            self._target_state.finger_angles[base_idx + 2] = grip * 0.8  # DIP
            self._target_state.finger_angles[base_idx + 3] = grip * 0.3  # TIP

        # Thumb has different range
        self._target_state.finger_angles[0] = grip * 0.8  # Thumb rotation

    def _apply_smoothing(self) -> None:
        """Apply exponential smoothing to state."""
        alpha = self.config.smoothing_alpha

        self._state.position = (
            (1 - alpha) * self._state.position +
            alpha * self._target_state.position
        )
        self._state.euler_angles = (
            (1 - alpha) * self._state.euler_angles +
            alpha * self._target_state.euler_angles
        )
        self._state.finger_angles = (
            (1 - alpha) * self._state.finger_angles +
            alpha * self._target_state.finger_angles
        )
        self._state.grip_state = (
            (1 - alpha) * self._state.grip_state +
            alpha * self._target_state.grip_state
        )

        # SLERP for orientation
        self._state.orientation = self._slerp(
            self._state.orientation,
            self._target_state.orientation,
            alpha,
        )

    def _add_noise(self) -> None:
        """Add realistic noise to state."""
        self._state.position += np.random.normal(
            0, self.config.position_noise_std, 3
        )
        self._state.finger_angles += np.random.normal(
            0, self.config.finger_noise_std, 21
        )

        # Add small orientation noise
        euler_noise = np.random.normal(0, self.config.rotation_noise_std, 3)
        noisy_euler = self._state.euler_angles + euler_noise
        self._state.orientation = self._euler_to_quaternion(noisy_euler)

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (XYZ) to quaternion (wxyz)."""
        roll, pitch, yaw = euler

        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def _slerp(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)

        # If negative dot, negate one quaternion
        if dot < 0:
            q2 = -q2
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = math.acos(dot)
        theta = theta_0 * t

        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / np.linalg.norm(q2_perp)

        return q1 * math.cos(theta) + q2_perp * math.sin(theta)

    # === Public Interface ===

    def get_state(self) -> GloveState:
        """Get current glove state."""
        return self._state

    def get_position(self) -> np.ndarray:
        """Get hand position."""
        return self._state.position.copy()

    def get_orientation(self) -> np.ndarray:
        """Get hand orientation as quaternion."""
        return self._state.orientation.copy()

    def get_finger_angles(self) -> np.ndarray:
        """Get finger joint angles."""
        return self._state.finger_angles.copy()

    def get_grip_state(self) -> float:
        """Get grip state (0=open, 1=closed)."""
        return self._state.grip_state

    def is_connected(self) -> bool:
        """Check if glove is connected."""
        return self._state.connected

    # === Input Methods ===

    def set_key_state(self, key: str, pressed: bool) -> None:
        """Set keyboard key state."""
        self._keyboard_state[key.lower()] = pressed

    def set_mouse_position(self, x: float, y: float) -> None:
        """Set mouse position."""
        self._mouse_position = np.array([x, y])

    def set_mouse_scroll(self, delta: float) -> None:
        """Add to mouse scroll value."""
        self._mouse_scroll += delta

    def set_gamepad_input(
        self,
        left_stick: Tuple[float, float],
        right_stick: Tuple[float, float],
        triggers: Tuple[float, float],
    ) -> None:
        """Set gamepad input."""
        sens = self.config.position_sensitivity

        # Left stick -> XY position
        self._target_state.position[0] = np.clip(
            self._target_state.position[0] + left_stick[1] * sens,
            self.config.workspace_min[0],
            self.config.workspace_max[0],
        )
        self._target_state.position[1] = np.clip(
            self._target_state.position[1] + left_stick[0] * sens,
            self.config.workspace_min[1],
            self.config.workspace_max[1],
        )

        # Right stick -> rotation
        self._target_state.euler_angles[0] += right_stick[1] * self.config.rotation_sensitivity
        self._target_state.euler_angles[2] += right_stick[0] * self.config.rotation_sensitivity

        # Triggers -> Z position and grip
        z_input = triggers[1] - triggers[0]  # RT up, LT down
        self._target_state.position[2] = np.clip(
            self._target_state.position[2] + z_input * sens,
            self.config.workspace_min[2],
            self.config.workspace_max[2],
        )

    # === Trajectory Methods ===

    def set_trajectory(self, waypoints: List[Dict[str, Any]]) -> None:
        """Set scripted trajectory."""
        self._trajectory = []

        for wp in waypoints:
            state = GloveState(
                position=np.array(wp.get("position", [0, 0, 0.5])),
                orientation=np.array(wp.get("orientation", [1, 0, 0, 0])),
                finger_angles=np.array(wp.get("finger_angles", np.zeros(21))),
                grip_state=wp.get("grip", 0.0),
            )
            self._trajectory.append(state)

        self._trajectory_index = 0
        self._trajectory_start_time = time.time()

        logger.info(f"Trajectory set with {len(self._trajectory)} waypoints")

    def load_playback(self, recording: List[Dict]) -> None:
        """Load recording for playback."""
        self._playback_buffer = recording
        self._playback_index = 0
        logger.info(f"Loaded {len(recording)} frames for playback")

    # === Callbacks ===

    def on_state_update(self, callback: Callable[[GloveState], None]) -> None:
        """Register state update callback."""
        self._on_state_callbacks.append(callback)

    # === Demo Trajectories ===

    @classmethod
    def create_pick_trajectory(cls) -> List[Dict[str, Any]]:
        """Create a pick motion trajectory."""
        return [
            {"position": [0.0, 0.0, 0.5], "grip": 0.0},
            {"position": [0.3, 0.0, 0.5], "grip": 0.0},
            {"position": [0.3, 0.0, 0.2], "grip": 0.0},
            {"position": [0.3, 0.0, 0.1], "grip": 0.0},
            {"position": [0.3, 0.0, 0.1], "grip": 1.0},
            {"position": [0.3, 0.0, 0.3], "grip": 1.0},
        ]

    @classmethod
    def create_place_trajectory(cls) -> List[Dict[str, Any]]:
        """Create a place motion trajectory."""
        return [
            {"position": [0.3, 0.0, 0.3], "grip": 1.0},
            {"position": [0.3, 0.3, 0.3], "grip": 1.0},
            {"position": [0.3, 0.3, 0.1], "grip": 1.0},
            {"position": [0.3, 0.3, 0.1], "grip": 0.0},
            {"position": [0.3, 0.3, 0.3], "grip": 0.0},
            {"position": [0.0, 0.0, 0.5], "grip": 0.0},
        ]

    @classmethod
    def create_wave_trajectory(cls) -> List[Dict[str, Any]]:
        """Create a waving motion trajectory."""
        return [
            {"position": [0.0, 0.0, 0.5], "orientation": [1, 0, 0, 0], "grip": 0.0},
            {"position": [0.0, 0.2, 0.6], "orientation": [0.92, 0, 0.38, 0], "grip": 0.0},
            {"position": [0.0, -0.2, 0.6], "orientation": [0.92, 0, -0.38, 0], "grip": 0.0},
            {"position": [0.0, 0.2, 0.6], "orientation": [0.92, 0, 0.38, 0], "grip": 0.0},
            {"position": [0.0, -0.2, 0.6], "orientation": [0.92, 0, -0.38, 0], "grip": 0.0},
            {"position": [0.0, 0.0, 0.5], "orientation": [1, 0, 0, 0], "grip": 0.0},
        ]

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Get state as dictionary for telemetry."""
        return {
            "position": self._state.position.tolist(),
            "orientation": self._state.orientation.tolist(),
            "euler_angles": self._state.euler_angles.tolist(),
            "finger_angles": self._state.finger_angles.tolist(),
            "grip_state": self._state.grip_state,
            "linear_acceleration": self._state.linear_acceleration.tolist(),
            "angular_velocity": self._state.angular_velocity.tolist(),
            "timestamp": self._state.timestamp,
            "connected": self._state.connected,
        }

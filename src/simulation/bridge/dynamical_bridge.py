"""
Dynamical Simulation Bridge

Connects Isaac Lab simulation to the Dynamical.ai platform,
enabling real-time data streaming, control, and monitoring.

Architecture:
    Isaac Lab  <--ZeroMQ-->  Bridge  <--HTTP/WS-->  Dynamical API
                                |
                                v
                           Dashboard
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    ROBOT_STATE = "robot_state"
    CAMERA_FRAME = "camera_frame"
    SENSOR_DATA = "sensor_data"
    TASK_STATE = "task_state"
    METRICS = "metrics"
    CONTROL_COMMAND = "control_command"


@dataclass
class BridgeConfig:
    """Configuration for simulation bridge."""
    # API connection
    api_host: str = "localhost"
    api_port: int = 8000
    api_key: str = ""

    # ZeroMQ for high-performance streaming
    zmq_pub_port: int = 5555
    zmq_sub_port: int = 5556

    # Streaming rates (Hz)
    robot_state_rate: float = 100.0
    camera_rate: float = 30.0
    sensor_rate: float = 60.0
    metrics_rate: float = 1.0

    # Buffering
    command_buffer_size: int = 100
    telemetry_buffer_size: int = 1000

    # Features
    enable_zmq: bool = True
    enable_websocket: bool = True
    enable_recording: bool = False
    recording_path: str = "./recordings"


@dataclass
class RobotStateTelemetry:
    """Robot state telemetry packet."""
    timestamp: float
    joint_positions: List[float]
    joint_velocities: List[float]
    joint_torques: List[float]
    ee_position: List[float]
    ee_orientation: List[float]
    gripper_state: float
    control_mode: str = "position"


@dataclass
class CameraTelemetry:
    """Camera telemetry packet."""
    timestamp: float
    camera_id: str
    frame_number: int
    resolution: Tuple[int, int]
    # RGB and depth are base64 encoded for transmission
    rgb_b64: Optional[str] = None
    depth_b64: Optional[str] = None


@dataclass
class TaskTelemetry:
    """Task state telemetry packet."""
    timestamp: float
    task_type: str
    phase: str
    step: int
    reward: float
    success: bool = False
    objects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsTelemetry:
    """System metrics telemetry packet."""
    timestamp: float
    physics_fps: float
    render_fps: float
    step_time_ms: float
    real_time_factor: float
    gpu_memory_mb: float
    cpu_usage_percent: float
    total_steps: int
    simulation_time: float


class DynamicalSimBridge:
    """
    Bridge between Isaac Lab simulation and Dynamical platform.

    Features:
    - Real-time telemetry streaming (ZeroMQ + WebSocket)
    - Control command reception
    - Recording for demonstration collection
    - Metrics publishing
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        """Initialize bridge."""
        self.config = config or BridgeConfig()

        # State
        self._running = False
        self._connected = False

        # Queues for async communication
        self._telemetry_queue: Queue = Queue(maxsize=self.config.telemetry_buffer_size)
        self._command_queue: Queue = Queue(maxsize=self.config.command_buffer_size)

        # ZeroMQ sockets (lazy initialization)
        self._zmq_context = None
        self._zmq_publisher = None
        self._zmq_subscriber = None

        # WebSocket connections
        self._websocket_clients: List[Any] = []

        # Callbacks
        self._on_command_callbacks: List[Callable] = []
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []

        # Metrics tracking
        self._telemetry_count = 0
        self._command_count = 0
        self._last_metrics_time = 0.0
        self._metrics_interval = 1.0 / self.config.metrics_rate

        # Recording
        self._recording_buffer: List[Dict] = []
        self._is_recording = False

        # Background threads
        self._publisher_thread: Optional[threading.Thread] = None
        self._subscriber_thread: Optional[threading.Thread] = None

        logger.info("DynamicalSimBridge initialized")

    async def start(self) -> None:
        """Start the bridge."""
        if self._running:
            return

        self._running = True

        # Initialize ZeroMQ if enabled
        if self.config.enable_zmq:
            await self._init_zmq()

        # Start background threads
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop,
            daemon=True,
        )
        self._publisher_thread.start()

        self._subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True,
        )
        self._subscriber_thread.start()

        # Connect to Dynamical API
        await self._connect_to_api()

        logger.info("Bridge started")

    async def stop(self) -> None:
        """Stop the bridge."""
        self._running = False

        # Wait for threads
        if self._publisher_thread:
            self._publisher_thread.join(timeout=1.0)
        if self._subscriber_thread:
            self._subscriber_thread.join(timeout=1.0)

        # Cleanup ZeroMQ
        if self._zmq_publisher:
            self._zmq_publisher.close()
        if self._zmq_subscriber:
            self._zmq_subscriber.close()
        if self._zmq_context:
            self._zmq_context.term()

        logger.info("Bridge stopped")

    async def _init_zmq(self) -> None:
        """Initialize ZeroMQ sockets."""
        try:
            import zmq

            self._zmq_context = zmq.Context()

            # Publisher for telemetry
            self._zmq_publisher = self._zmq_context.socket(zmq.PUB)
            self._zmq_publisher.bind(f"tcp://*:{self.config.zmq_pub_port}")

            # Subscriber for commands
            self._zmq_subscriber = self._zmq_context.socket(zmq.SUB)
            self._zmq_subscriber.connect(f"tcp://localhost:{self.config.zmq_sub_port}")
            self._zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

            logger.info(f"ZeroMQ initialized - pub:{self.config.zmq_pub_port}, sub:{self.config.zmq_sub_port}")

        except ImportError:
            logger.warning("ZeroMQ not available, using HTTP only")
            self.config.enable_zmq = False

    async def _connect_to_api(self) -> None:
        """Connect to Dynamical API."""
        try:
            import aiohttp

            url = f"http://{self.config.api_host}:{self.config.api_port}/health"
            headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        self._connected = True
                        logger.info("Connected to Dynamical API")

                        for callback in self._on_connect_callbacks:
                            callback()
                    else:
                        logger.warning(f"API connection failed: {response.status}")

        except Exception as e:
            logger.warning(f"Could not connect to API: {e}")
            # Continue without API connection (useful for standalone testing)

    def _publisher_loop(self) -> None:
        """Background thread for publishing telemetry."""
        while self._running:
            try:
                # Get telemetry from queue
                telemetry = self._telemetry_queue.get(timeout=0.1)

                # Publish via ZeroMQ
                if self.config.enable_zmq and self._zmq_publisher:
                    topic = telemetry.get("type", "unknown")
                    data = json.dumps(telemetry, default=self._json_serializer)
                    self._zmq_publisher.send_string(f"{topic} {data}")

                self._telemetry_count += 1

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Publisher error: {e}")

    def _subscriber_loop(self) -> None:
        """Background thread for receiving commands."""
        while self._running:
            try:
                if self.config.enable_zmq and self._zmq_subscriber:
                    # Non-blocking receive
                    try:
                        import zmq
                        message = self._zmq_subscriber.recv_string(flags=zmq.NOBLOCK)
                        parts = message.split(" ", 1)
                        if len(parts) == 2:
                            topic, data = parts
                            command = json.loads(data)
                            self._handle_command(command)
                            self._command_count += 1
                    except:
                        pass

                time.sleep(0.001)  # 1ms polling

            except Exception as e:
                logger.error(f"Subscriber error: {e}")

    def _handle_command(self, command: Dict[str, Any]) -> None:
        """Handle received command."""
        self._command_queue.put(command)

        for callback in self._on_command_callbacks:
            try:
                callback(command)
            except Exception as e:
                logger.error(f"Command callback error: {e}")

    def _json_serializer(self, obj: Any) -> Any:
        """JSON serializer for numpy arrays and dataclasses."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # === Telemetry Publishing ===

    def publish_robot_state(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        gripper_state: float,
        control_mode: str = "position",
    ) -> None:
        """Publish robot state telemetry."""
        telemetry = {
            "type": StreamType.ROBOT_STATE.value,
            "timestamp": time.time(),
            "joint_positions": joint_positions.tolist(),
            "joint_velocities": joint_velocities.tolist(),
            "joint_torques": joint_torques.tolist(),
            "ee_position": ee_position.tolist(),
            "ee_orientation": ee_orientation.tolist(),
            "gripper_state": float(gripper_state),
            "control_mode": control_mode,
        }

        self._enqueue_telemetry(telemetry)

    def publish_camera_frame(
        self,
        camera_id: str,
        frame_number: int,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        compress: bool = True,
    ) -> None:
        """Publish camera frame telemetry."""
        import base64

        telemetry = {
            "type": StreamType.CAMERA_FRAME.value,
            "timestamp": time.time(),
            "camera_id": camera_id,
            "frame_number": frame_number,
        }

        if rgb is not None:
            if compress:
                # JPEG compression
                try:
                    import cv2
                    _, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    telemetry["rgb_b64"] = base64.b64encode(encoded).decode('utf-8')
                    telemetry["rgb_compressed"] = True
                except ImportError:
                    # Fallback: raw base64
                    telemetry["rgb_b64"] = base64.b64encode(rgb.tobytes()).decode('utf-8')
                    telemetry["rgb_shape"] = rgb.shape
            else:
                telemetry["rgb_b64"] = base64.b64encode(rgb.tobytes()).decode('utf-8')
                telemetry["rgb_shape"] = rgb.shape

        if depth is not None:
            # Depth as float16 for bandwidth
            depth_f16 = depth.astype(np.float16)
            telemetry["depth_b64"] = base64.b64encode(depth_f16.tobytes()).decode('utf-8')
            telemetry["depth_shape"] = depth.shape

        self._enqueue_telemetry(telemetry)

    def publish_task_state(
        self,
        task_type: str,
        phase: str,
        step: int,
        reward: float,
        success: bool = False,
        objects: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish task state telemetry."""
        telemetry = {
            "type": StreamType.TASK_STATE.value,
            "timestamp": time.time(),
            "task_type": task_type,
            "phase": phase,
            "step": step,
            "reward": float(reward),
            "success": success,
            "objects": objects or {},
        }

        self._enqueue_telemetry(telemetry)

    def publish_metrics(
        self,
        physics_fps: float,
        render_fps: float,
        step_time_ms: float,
        real_time_factor: float,
        gpu_memory_mb: float = 0.0,
        cpu_usage_percent: float = 0.0,
        total_steps: int = 0,
        simulation_time: float = 0.0,
    ) -> None:
        """Publish system metrics telemetry."""
        now = time.time()
        if now - self._last_metrics_time < self._metrics_interval:
            return

        self._last_metrics_time = now

        telemetry = {
            "type": StreamType.METRICS.value,
            "timestamp": now,
            "physics_fps": physics_fps,
            "render_fps": render_fps,
            "step_time_ms": step_time_ms,
            "real_time_factor": real_time_factor,
            "gpu_memory_mb": gpu_memory_mb,
            "cpu_usage_percent": cpu_usage_percent,
            "total_steps": total_steps,
            "simulation_time": simulation_time,
            "telemetry_count": self._telemetry_count,
            "command_count": self._command_count,
        }

        self._enqueue_telemetry(telemetry)

    def _enqueue_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Enqueue telemetry for publishing."""
        try:
            self._telemetry_queue.put_nowait(telemetry)

            # Recording
            if self._is_recording:
                self._recording_buffer.append(telemetry)

        except:
            # Queue full, drop oldest
            try:
                self._telemetry_queue.get_nowait()
                self._telemetry_queue.put_nowait(telemetry)
            except:
                pass

    # === Command Reception ===

    def get_command(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Get next command from queue."""
        try:
            if timeout > 0:
                return self._command_queue.get(timeout=timeout)
            else:
                return self._command_queue.get_nowait()
        except Empty:
            return None

    def on_command(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register command callback."""
        self._on_command_callbacks.append(callback)

    def on_connect(self, callback: Callable[[], None]) -> None:
        """Register connection callback."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register disconnection callback."""
        self._on_disconnect_callbacks.append(callback)

    # === Recording ===

    def start_recording(self) -> None:
        """Start recording telemetry."""
        self._is_recording = True
        self._recording_buffer.clear()
        logger.info("Started recording")

    def stop_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return buffer."""
        self._is_recording = False
        buffer = self._recording_buffer.copy()
        self._recording_buffer.clear()
        logger.info(f"Stopped recording, {len(buffer)} frames captured")
        return buffer

    def save_recording(self, filename: str) -> None:
        """Save recording buffer to file."""
        import os

        os.makedirs(self.config.recording_path, exist_ok=True)
        filepath = os.path.join(self.config.recording_path, filename)

        with open(filepath, 'w') as f:
            json.dump(self._recording_buffer, f, default=self._json_serializer)

        logger.info(f"Recording saved to {filepath}")

    # === HTTP API Integration ===

    async def post_telemetry_to_api(
        self,
        telemetry: Dict[str, Any],
        endpoint: str = "/api/simulation/telemetry",
    ) -> bool:
        """Post telemetry to Dynamical API."""
        if not self._connected:
            return False

        try:
            import aiohttp

            url = f"http://{self.config.api_host}:{self.config.api_port}{endpoint}"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=telemetry,
                    timeout=1.0,
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.debug(f"API post failed: {e}")
            return False

    async def get_command_from_api(
        self,
        endpoint: str = "/api/simulation/command",
    ) -> Optional[Dict[str, Any]]:
        """Get command from Dynamical API."""
        if not self._connected:
            return None

        try:
            import aiohttp

            url = f"http://{self.config.api_host}:{self.config.api_port}{endpoint}"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=1.0) as response:
                    if response.status == 200:
                        return await response.json()

        except Exception as e:
            logger.debug(f"API get failed: {e}")

        return None

    # === Status ===

    def is_running(self) -> bool:
        """Check if bridge is running."""
        return self._running

    def is_connected(self) -> bool:
        """Check if connected to API."""
        return self._connected

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "running": self._running,
            "connected": self._connected,
            "telemetry_count": self._telemetry_count,
            "command_count": self._command_count,
            "telemetry_queue_size": self._telemetry_queue.qsize(),
            "command_queue_size": self._command_queue.qsize(),
            "is_recording": self._is_recording,
            "recording_frames": len(self._recording_buffer),
        }

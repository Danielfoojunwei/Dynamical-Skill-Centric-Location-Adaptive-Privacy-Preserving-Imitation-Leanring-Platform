"""
Telemetry Publisher

Specialized publisher for streaming simulation telemetry to the dashboard.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PublisherConfig:
    """Telemetry publisher configuration."""
    # Update rates (Hz)
    robot_state_rate: float = 60.0
    camera_rate: float = 30.0
    task_rate: float = 30.0
    metrics_rate: float = 1.0
    fl_rate: float = 0.5  # Federated learning updates

    # Compression
    enable_compression: bool = True
    jpeg_quality: int = 80

    # Batching
    enable_batching: bool = False
    batch_size: int = 10
    batch_timeout_ms: float = 100.0


class TelemetryPublisher:
    """
    Publishes simulation telemetry to WebSocket clients.

    Features:
    - Rate-limited publishing per channel
    - Compression for camera frames
    - Batching for efficiency
    - Multiple subscriber support
    """

    def __init__(self, config: Optional[PublisherConfig] = None):
        """Initialize publisher."""
        self.config = config or PublisherConfig()

        # Subscribers (WebSocket connections)
        self._subscribers: Set[Any] = set()

        # Last publish times per channel
        self._last_publish: Dict[str, float] = {}

        # Publish intervals
        self._intervals = {
            "robot_state": 1.0 / self.config.robot_state_rate,
            "camera": 1.0 / self.config.camera_rate,
            "task": 1.0 / self.config.task_rate,
            "metrics": 1.0 / self.config.metrics_rate,
            "federated_learning": 1.0 / self.config.fl_rate,
        }

        # Batching buffer
        self._batch_buffer: List[Dict] = []
        self._batch_start_time = 0.0

        # Statistics
        self._publish_count = 0
        self._bytes_sent = 0

        logger.info("TelemetryPublisher initialized")

    def add_subscriber(self, websocket: Any) -> None:
        """Add WebSocket subscriber."""
        self._subscribers.add(websocket)
        logger.info(f"Subscriber added, total: {len(self._subscribers)}")

    def remove_subscriber(self, websocket: Any) -> None:
        """Remove WebSocket subscriber."""
        self._subscribers.discard(websocket)
        logger.info(f"Subscriber removed, total: {len(self._subscribers)}")

    def _should_publish(self, channel: str) -> bool:
        """Check if enough time has passed to publish on channel."""
        now = time.time()
        interval = self._intervals.get(channel, 0.1)
        last = self._last_publish.get(channel, 0)

        if now - last >= interval:
            self._last_publish[channel] = now
            return True
        return False

    async def publish(
        self,
        channel: str,
        data: Dict[str, Any],
        force: bool = False,
    ) -> int:
        """
        Publish data to subscribers.

        Args:
            channel: Channel name
            data: Data to publish
            force: Force publish regardless of rate limit

        Returns:
            Number of subscribers published to
        """
        if not force and not self._should_publish(channel):
            return 0

        if not self._subscribers:
            return 0

        # Prepare message
        message = {
            "channel": channel,
            "timestamp": time.time(),
            "data": data,
        }

        # Serialize
        try:
            json_data = json.dumps(message, default=self._json_serializer)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return 0

        # Batching
        if self.config.enable_batching:
            return await self._publish_batched(json_data)

        # Direct publish
        return await self._publish_direct(json_data)

    async def _publish_direct(self, json_data: str) -> int:
        """Publish directly to all subscribers."""
        sent = 0
        dead_sockets = []

        for ws in self._subscribers:
            try:
                await ws.send_text(json_data)
                sent += 1
                self._bytes_sent += len(json_data)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")
                dead_sockets.append(ws)

        # Remove dead connections
        for ws in dead_sockets:
            self._subscribers.discard(ws)

        self._publish_count += 1
        return sent

    async def _publish_batched(self, json_data: str) -> int:
        """Batch messages before publishing."""
        now = time.time()

        if not self._batch_buffer:
            self._batch_start_time = now

        self._batch_buffer.append(json_data)

        # Check if should flush
        batch_age_ms = (now - self._batch_start_time) * 1000
        should_flush = (
            len(self._batch_buffer) >= self.config.batch_size or
            batch_age_ms >= self.config.batch_timeout_ms
        )

        if should_flush:
            return await self._flush_batch()

        return 0

    async def _flush_batch(self) -> int:
        """Flush batched messages."""
        if not self._batch_buffer:
            return 0

        # Combine into batch message
        batch_message = json.dumps({
            "type": "batch",
            "messages": self._batch_buffer,
        })

        self._batch_buffer.clear()
        return await self._publish_direct(batch_message)

    def _json_serializer(self, obj: Any) -> Any:
        """JSON serializer for special types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # === Specialized Publishers ===

    async def publish_robot_state(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        gripper_state: float,
    ) -> int:
        """Publish robot state."""
        return await self.publish("robot_state", {
            "joint_positions": joint_positions.tolist(),
            "joint_velocities": joint_velocities.tolist(),
            "ee_position": ee_position.tolist(),
            "ee_orientation": ee_orientation.tolist(),
            "gripper_state": float(gripper_state),
        })

    async def publish_camera_frame(
        self,
        camera_id: str,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> int:
        """Publish camera frame (compressed)."""
        import base64

        data = {"camera_id": camera_id}

        if self.config.enable_compression:
            try:
                import cv2
                _, encoded = cv2.imencode(
                    '.jpg', rgb,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
                data["rgb_b64"] = base64.b64encode(encoded).decode('utf-8')
                data["compressed"] = True
            except ImportError:
                data["rgb_b64"] = base64.b64encode(rgb.tobytes()).decode('utf-8')
                data["shape"] = rgb.shape
        else:
            data["rgb_b64"] = base64.b64encode(rgb.tobytes()).decode('utf-8')
            data["shape"] = rgb.shape

        if depth is not None:
            depth_f16 = depth.astype(np.float16)
            data["depth_b64"] = base64.b64encode(depth_f16.tobytes()).decode('utf-8')
            data["depth_shape"] = depth.shape

        return await self.publish("camera", data)

    async def publish_task_state(
        self,
        task_type: str,
        phase: str,
        step: int,
        reward: float,
        success: bool,
        objects: Dict[str, Any],
    ) -> int:
        """Publish task state."""
        return await self.publish("task", {
            "task_type": task_type,
            "phase": phase,
            "step": step,
            "reward": reward,
            "success": success,
            "objects": objects,
        })

    async def publish_simulation_metrics(
        self,
        physics_fps: float,
        render_fps: float,
        step_time_ms: float,
        real_time_factor: float,
        total_steps: int,
        simulation_time: float,
    ) -> int:
        """Publish simulation metrics."""
        return await self.publish("metrics", {
            "physics_fps": physics_fps,
            "render_fps": render_fps,
            "step_time_ms": step_time_ms,
            "real_time_factor": real_time_factor,
            "total_steps": total_steps,
            "simulation_time": simulation_time,
        })

    async def publish_fl_update(
        self,
        round_number: int,
        global_loss: float,
        local_losses: Dict[str, float],
        num_clients: int,
        convergence_rate: float,
    ) -> int:
        """Publish federated learning update."""
        return await self.publish("federated_learning", {
            "round": round_number,
            "global_loss": global_loss,
            "local_losses": local_losses,
            "num_clients": num_clients,
            "convergence_rate": convergence_rate,
        }, force=True)

    async def publish_safety_status(
        self,
        safety_level: str,
        hazards: List[Dict],
        zone_violations: List[str],
    ) -> int:
        """Publish safety status."""
        return await self.publish("safety", {
            "level": safety_level,
            "hazards": hazards,
            "zone_violations": zone_violations,
        }, force=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        return {
            "subscriber_count": len(self._subscribers),
            "publish_count": self._publish_count,
            "bytes_sent": self._bytes_sent,
            "batch_buffer_size": len(self._batch_buffer),
        }

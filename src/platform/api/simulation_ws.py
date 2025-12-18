"""
WebSocket API for Real-Time Simulation Streaming

Provides WebSocket endpoints for streaming simulation data to the dashboard:
- Robot state (joint positions, end-effector pose)
- Camera frames
- Task progress
- Federated learning metrics
- Safety status
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


class ConnectionManager:
    """
    Manages WebSocket connections for real-time streaming.

    Features:
    - Channel-based subscriptions
    - Broadcast to multiple clients
    - Automatic cleanup on disconnect
    """

    def __init__(self):
        # Active connections per channel
        self.connections: Dict[str, Set[WebSocket]] = {
            "simulation": set(),
            "robot_state": set(),
            "camera": set(),
            "task": set(),
            "metrics": set(),
            "federated_learning": set(),
            "safety": set(),
            "all": set(),  # Receives everything
        }

        # Connection metadata
        self.connection_info: Dict[WebSocket, Dict] = {}

        # Statistics
        self.total_messages_sent = 0
        self.total_bytes_sent = 0

    async def connect(
        self,
        websocket: WebSocket,
        channels: List[str] = None,
        client_id: str = None,
    ) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()

        # Store connection info
        self.connection_info[websocket] = {
            "client_id": client_id or str(id(websocket)),
            "connected_at": time.time(),
            "channels": channels or ["all"],
        }

        # Add to channels
        for channel in (channels or ["all"]):
            if channel in self.connections:
                self.connections[channel].add(websocket)

        logger.info(f"WebSocket connected: {client_id}, channels: {channels}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        # Remove from all channels
        for channel_set in self.connections.values():
            channel_set.discard(websocket)

        # Remove metadata
        if websocket in self.connection_info:
            info = self.connection_info.pop(websocket)
            logger.info(f"WebSocket disconnected: {info['client_id']}")

    async def broadcast(
        self,
        channel: str,
        message: Dict[str, Any],
    ) -> int:
        """
        Broadcast message to all connections on a channel.

        Returns number of clients sent to.
        """
        # Get target connections
        targets = self.connections.get(channel, set()) | self.connections.get("all", set())

        if not targets:
            return 0

        # Serialize message
        try:
            json_data = json.dumps(message, default=self._json_serializer)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return 0

        # Send to all
        sent = 0
        dead_connections = []

        for websocket in targets:
            try:
                await websocket.send_text(json_data)
                sent += 1
                self.total_messages_sent += 1
                self.total_bytes_sent += len(json_data)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")
                dead_connections.append(websocket)

        # Cleanup dead connections
        for ws in dead_connections:
            self.disconnect(ws)

        return sent

    async def send_personal(
        self,
        websocket: WebSocket,
        message: Dict[str, Any],
    ) -> bool:
        """Send message to specific connection."""
        try:
            json_data = json.dumps(message, default=self._json_serializer)
            await websocket.send_text(json_data)
            self.total_messages_sent += 1
            return True
        except Exception as e:
            logger.debug(f"Personal send failed: {e}")
            return False

    def _json_serializer(self, obj: Any) -> Any:
        """JSON serializer for special types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": sum(len(s) for s in self.connections.values()),
            "connections_per_channel": {
                channel: len(conns) for channel, conns in self.connections.items()
            },
            "total_messages_sent": self.total_messages_sent,
            "total_bytes_sent": self.total_bytes_sent,
        }


# Global connection manager
manager = ConnectionManager()


# === Simulation State ===

class SimulationState:
    """Global simulation state for streaming."""

    def __init__(self):
        self.is_running = False
        self.environment = None
        self.bridge = None
        self.task = None

        # Cached state for polling endpoints
        self.robot_state: Dict = {}
        self.task_state: Dict = {}
        self.metrics: Dict = {}
        self.fl_state: Dict = {}
        self.safety_state: Dict = {}

        # Update timestamps
        self.last_robot_update = 0.0
        self.last_task_update = 0.0
        self.last_metrics_update = 0.0

    def update_robot_state(self, state: Dict) -> None:
        """Update robot state."""
        self.robot_state = state
        self.last_robot_update = time.time()

    def update_task_state(self, state: Dict) -> None:
        """Update task state."""
        self.task_state = state
        self.last_task_update = time.time()

    def update_metrics(self, metrics: Dict) -> None:
        """Update metrics."""
        self.metrics = metrics
        self.last_metrics_update = time.time()


sim_state = SimulationState()


# === WebSocket Endpoints ===

@router.websocket("/simulation")
async def simulation_websocket(
    websocket: WebSocket,
    channels: str = Query(default="all"),
):
    """
    Main WebSocket endpoint for simulation streaming.

    Query params:
        channels: Comma-separated list of channels to subscribe to
                  Options: robot_state, camera, task, metrics, federated_learning, safety, all

    Messages received:
        - {"type": "subscribe", "channels": ["robot_state", "task"]}
        - {"type": "unsubscribe", "channels": ["camera"]}
        - {"type": "command", "action": "reset"}
        - {"type": "teleop", "data": {...}}

    Messages sent:
        - {"channel": "robot_state", "timestamp": ..., "data": {...}}
        - {"channel": "task", "timestamp": ..., "data": {...}}
        - etc.
    """
    channel_list = channels.split(",") if channels else ["all"]

    await manager.connect(websocket, channels=channel_list)

    try:
        # Send initial state
        await manager.send_personal(websocket, {
            "type": "connected",
            "channels": channel_list,
            "simulation_running": sim_state.is_running,
        })

        while True:
            # Receive messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_client_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def handle_client_message(websocket: WebSocket, message: Dict) -> None:
    """Handle incoming WebSocket message from client."""
    msg_type = message.get("type", "")

    if msg_type == "subscribe":
        # Add to additional channels
        channels = message.get("channels", [])
        for channel in channels:
            if channel in manager.connections:
                manager.connections[channel].add(websocket)
        await manager.send_personal(websocket, {
            "type": "subscribed",
            "channels": channels,
        })

    elif msg_type == "unsubscribe":
        # Remove from channels
        channels = message.get("channels", [])
        for channel in channels:
            if channel in manager.connections:
                manager.connections[channel].discard(websocket)
        await manager.send_personal(websocket, {
            "type": "unsubscribed",
            "channels": channels,
        })

    elif msg_type == "command":
        # Handle simulation commands
        action = message.get("action")
        if action == "reset":
            if sim_state.environment:
                sim_state.environment.reset()
            await manager.broadcast("simulation", {
                "type": "reset",
                "timestamp": time.time(),
            })
        elif action == "start":
            sim_state.is_running = True
            await manager.broadcast("simulation", {
                "type": "started",
                "timestamp": time.time(),
            })
        elif action == "stop":
            sim_state.is_running = False
            await manager.broadcast("simulation", {
                "type": "stopped",
                "timestamp": time.time(),
            })

    elif msg_type == "teleop":
        # Forward teleoperation data
        teleop_data = message.get("data", {})
        # Process teleop (would connect to robot controller)
        await manager.broadcast("teleop", {
            "type": "teleop_command",
            "data": teleop_data,
            "timestamp": time.time(),
        })

    elif msg_type == "ping":
        # Respond to ping
        await manager.send_personal(websocket, {
            "type": "pong",
            "timestamp": time.time(),
        })


# === Streaming Functions (called from simulation loop) ===

async def stream_robot_state(state: Dict) -> None:
    """Stream robot state to connected clients."""
    sim_state.update_robot_state(state)

    await manager.broadcast("robot_state", {
        "channel": "robot_state",
        "timestamp": time.time(),
        "data": state,
    })


async def stream_camera_frame(
    camera_id: str,
    rgb_b64: str,
    depth_b64: Optional[str] = None,
) -> None:
    """Stream camera frame to connected clients."""
    await manager.broadcast("camera", {
        "channel": "camera",
        "timestamp": time.time(),
        "data": {
            "camera_id": camera_id,
            "rgb_b64": rgb_b64,
            "depth_b64": depth_b64,
        },
    })


async def stream_task_state(state: Dict) -> None:
    """Stream task state to connected clients."""
    sim_state.update_task_state(state)

    await manager.broadcast("task", {
        "channel": "task",
        "timestamp": time.time(),
        "data": state,
    })


async def stream_metrics(metrics: Dict) -> None:
    """Stream simulation metrics to connected clients."""
    sim_state.update_metrics(metrics)

    await manager.broadcast("metrics", {
        "channel": "metrics",
        "timestamp": time.time(),
        "data": metrics,
    })


async def stream_fl_update(update: Dict) -> None:
    """Stream federated learning update to connected clients."""
    sim_state.fl_state = update

    await manager.broadcast("federated_learning", {
        "channel": "federated_learning",
        "timestamp": time.time(),
        "data": update,
    })


async def stream_safety_status(status: Dict) -> None:
    """Stream safety status to connected clients."""
    sim_state.safety_state = status

    await manager.broadcast("safety", {
        "channel": "safety",
        "timestamp": time.time(),
        "data": status,
    })


# === REST Endpoints for Polling Fallback ===

@router.get("/state/robot")
async def get_robot_state():
    """Get current robot state (polling fallback)."""
    return {
        "state": sim_state.robot_state,
        "timestamp": sim_state.last_robot_update,
    }


@router.get("/state/task")
async def get_task_state():
    """Get current task state (polling fallback)."""
    return {
        "state": sim_state.task_state,
        "timestamp": sim_state.last_task_update,
    }


@router.get("/state/metrics")
async def get_simulation_metrics():
    """Get current metrics (polling fallback)."""
    return {
        "metrics": sim_state.metrics,
        "timestamp": sim_state.last_metrics_update,
    }


@router.get("/connections")
async def get_connection_stats():
    """Get WebSocket connection statistics."""
    return manager.get_stats()


# === Simulation Control REST Endpoints ===

class SimulationStartRequest(BaseModel):
    """Request to start simulation."""
    robot_type: str = "franka_panda"
    scene_type: str = "tabletop"
    task_type: str = "pick_place"
    enable_cameras: bool = True
    headless: bool = True


class TeleopCommand(BaseModel):
    """Teleoperation command."""
    position: Optional[List[float]] = None
    orientation: Optional[List[float]] = None
    gripper: Optional[float] = None


@router.post("/simulation/start")
async def start_simulation(request: SimulationStartRequest):
    """Start Isaac Lab simulation."""
    from src.simulation import IsaacLabEnvironment, SimulationConfig, RobotType

    if sim_state.is_running:
        return {"status": "already_running"}

    try:
        # Create simulation config
        robot_type = RobotType(request.robot_type) if hasattr(RobotType, request.robot_type.upper()) else RobotType.FRANKA_PANDA

        config = SimulationConfig(
            robot_type=robot_type,
            scene_type=request.scene_type,
            num_cameras=4 if request.enable_cameras else 0,
            headless=request.headless,
        )

        # Create environment
        sim_state.environment = IsaacLabEnvironment(config=config, mode="standalone")
        await sim_state.environment.setup()

        sim_state.is_running = True

        # Broadcast start event
        await manager.broadcast("simulation", {
            "type": "started",
            "config": {
                "robot_type": request.robot_type,
                "scene_type": request.scene_type,
                "task_type": request.task_type,
            },
            "timestamp": time.time(),
        })

        logger.info("Simulation started")
        return {"status": "started"}

    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/simulation/stop")
async def stop_simulation():
    """Stop Isaac Lab simulation."""
    if not sim_state.is_running:
        return {"status": "not_running"}

    if sim_state.environment:
        sim_state.environment.close()
        sim_state.environment = None

    sim_state.is_running = False

    await manager.broadcast("simulation", {
        "type": "stopped",
        "timestamp": time.time(),
    })

    logger.info("Simulation stopped")
    return {"status": "stopped"}


@router.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state."""
    if not sim_state.environment:
        return {"status": "no_simulation"}

    obs = sim_state.environment.reset()

    await manager.broadcast("simulation", {
        "type": "reset",
        "observation": obs,
        "timestamp": time.time(),
    })

    return {"status": "reset", "observation": obs}


@router.post("/simulation/teleop")
async def teleop_command(command: TeleopCommand):
    """Send teleoperation command to simulation."""
    if not sim_state.environment:
        return {"status": "no_simulation"}

    # Build action
    action = None
    if command.position:
        sim_state.environment.set_robot_target(
            target_position=np.array(command.position),
            target_orientation=np.array(command.orientation) if command.orientation else None,
        )

    if command.gripper is not None:
        sim_state.environment.set_gripper(command.gripper)

    return {"status": "command_sent"}


@router.get("/simulation/status")
async def get_simulation_status():
    """Get simulation status."""
    return {
        "is_running": sim_state.is_running,
        "has_environment": sim_state.environment is not None,
        "robot_state": sim_state.robot_state,
        "task_state": sim_state.task_state,
        "connection_stats": manager.get_stats(),
    }


# === Background Streaming Task ===

async def simulation_streaming_loop():
    """
    Background task that streams simulation data to WebSocket clients.

    This should be started when the simulation starts and stopped when it stops.
    """
    logger.info("Starting simulation streaming loop")

    frame_count = 0
    last_metrics_time = 0.0

    while sim_state.is_running and sim_state.environment:
        try:
            # Step simulation
            obs = sim_state.environment.step()

            # Stream robot state
            robot_state = sim_state.environment.get_robot_state()
            await stream_robot_state({
                "joint_positions": robot_state.joint_positions.tolist(),
                "joint_velocities": robot_state.joint_velocities.tolist(),
                "ee_position": robot_state.ee_position.tolist(),
                "ee_orientation": robot_state.ee_orientation.tolist(),
                "gripper_state": robot_state.gripper_state,
            })

            # Stream camera frames (at lower rate)
            if frame_count % 2 == 0:  # Every other frame
                for cam_id in sim_state.environment._camera_frames:
                    frame = sim_state.environment.get_camera_frame(cam_id)
                    if frame and frame.rgb is not None:
                        import base64
                        try:
                            import cv2
                            _, encoded = cv2.imencode('.jpg', frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            rgb_b64 = base64.b64encode(encoded).decode('utf-8')
                        except ImportError:
                            rgb_b64 = base64.b64encode(frame.rgb.tobytes()).decode('utf-8')

                        await stream_camera_frame(cam_id, rgb_b64)

            # Stream metrics (at lower rate)
            now = time.time()
            if now - last_metrics_time > 1.0:
                metrics = sim_state.environment.get_metrics()
                await stream_metrics({
                    "physics_fps": metrics.physics_fps,
                    "render_fps": metrics.render_fps,
                    "step_time_ms": metrics.step_time_ms,
                    "real_time_factor": metrics.real_time_factor,
                    "total_steps": metrics.total_steps,
                    "simulation_time": metrics.simulation_time,
                })
                last_metrics_time = now

            frame_count += 1

            # Control loop rate (~60 Hz)
            await asyncio.sleep(1/60)

        except Exception as e:
            logger.error(f"Streaming loop error: {e}")
            await asyncio.sleep(0.1)

    logger.info("Simulation streaming loop ended")

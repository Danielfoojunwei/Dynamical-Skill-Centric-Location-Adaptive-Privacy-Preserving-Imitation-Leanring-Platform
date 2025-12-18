"""
Action Subscriber

Receives control commands from dashboard and external sources.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from queue import Queue, Empty
import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of control actions."""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    GRIPPER = "gripper"
    TELEOP = "teleop"
    POLICY_ACTION = "policy_action"
    STOP = "stop"
    RESET = "reset"


@dataclass
class ActionCommand:
    """Action command packet."""
    action_type: ActionType
    data: Dict[str, Any]
    timestamp: float
    source: str = "unknown"
    priority: int = 0


@dataclass
class SubscriberConfig:
    """Action subscriber configuration."""
    queue_size: int = 100
    timeout_ms: float = 100.0
    enable_validation: bool = True
    max_joint_velocity: float = 1.0
    max_ee_velocity: float = 0.5


class ActionSubscriber:
    """
    Subscribes to control commands from various sources.

    Features:
    - Priority queue for commands
    - Action validation
    - Timeout handling
    - Multiple source support (WebSocket, ZeroMQ, API)
    """

    def __init__(self, config: Optional[SubscriberConfig] = None):
        """Initialize subscriber."""
        self.config = config or SubscriberConfig()

        # Command queue
        self._command_queue: Queue = Queue(maxsize=self.config.queue_size)

        # Callbacks
        self._on_action_callbacks: List[Callable[[ActionCommand], None]] = []
        self._on_stop_callbacks: List[Callable[[], None]] = []
        self._on_reset_callbacks: List[Callable[[], None]] = []

        # State
        self._last_command_time = 0.0
        self._command_count = 0
        self._validation_failures = 0

        # Joint limits for validation
        self._joint_limits = None

        logger.info("ActionSubscriber initialized")

    def set_joint_limits(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """Set joint limits for validation."""
        self._joint_limits = (lower, upper)

    async def process_message(self, message: str, source: str = "websocket") -> bool:
        """
        Process incoming WebSocket message.

        Args:
            message: JSON message string
            source: Message source identifier

        Returns:
            True if processed successfully
        """
        try:
            data = json.loads(message)
            return await self.process_command(data, source)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}")
            return False

    async def process_command(
        self,
        data: Dict[str, Any],
        source: str = "unknown",
    ) -> bool:
        """
        Process incoming command.

        Args:
            data: Command data
            source: Command source

        Returns:
            True if processed successfully
        """
        try:
            # Parse action type
            action_type_str = data.get("type", data.get("action_type", ""))
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning(f"Unknown action type: {action_type_str}")
                return False

            # Create command
            command = ActionCommand(
                action_type=action_type,
                data=data.get("data", data),
                timestamp=data.get("timestamp", time.time()),
                source=source,
                priority=data.get("priority", 0),
            )

            # Validate
            if self.config.enable_validation:
                if not self._validate_command(command):
                    self._validation_failures += 1
                    return False

            # Handle special commands
            if action_type == ActionType.STOP:
                for callback in self._on_stop_callbacks:
                    callback()
                return True

            if action_type == ActionType.RESET:
                for callback in self._on_reset_callbacks:
                    callback()
                return True

            # Enqueue
            self._enqueue_command(command)

            # Callbacks
            for callback in self._on_action_callbacks:
                try:
                    callback(command)
                except Exception as e:
                    logger.error(f"Action callback error: {e}")

            self._command_count += 1
            self._last_command_time = time.time()

            return True

        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return False

    def _validate_command(self, command: ActionCommand) -> bool:
        """Validate command against safety limits."""
        data = command.data

        if command.action_type == ActionType.JOINT_POSITION:
            positions = data.get("positions", data.get("joint_positions", []))
            if self._joint_limits:
                lower, upper = self._joint_limits
                positions = np.array(positions)
                if np.any(positions < lower) or np.any(positions > upper):
                    logger.warning("Joint position outside limits")
                    return False

        elif command.action_type == ActionType.JOINT_VELOCITY:
            velocities = data.get("velocities", data.get("joint_velocities", []))
            velocities = np.array(velocities)
            if np.any(np.abs(velocities) > self.config.max_joint_velocity):
                logger.warning("Joint velocity exceeds limit")
                return False

        elif command.action_type == ActionType.CARTESIAN_VELOCITY:
            velocity = data.get("velocity", data.get("linear_velocity", []))
            velocity = np.array(velocity)
            if np.linalg.norm(velocity) > self.config.max_ee_velocity:
                logger.warning("Cartesian velocity exceeds limit")
                return False

        return True

    def _enqueue_command(self, command: ActionCommand) -> None:
        """Enqueue command with priority handling."""
        try:
            self._command_queue.put_nowait(command)
        except:
            # Queue full, try to remove low priority items
            try:
                old_command = self._command_queue.get_nowait()
                if old_command.priority < command.priority:
                    self._command_queue.put_nowait(command)
                else:
                    # Put old command back
                    self._command_queue.put_nowait(old_command)
            except:
                pass

    def get_action(self, timeout_ms: Optional[float] = None) -> Optional[ActionCommand]:
        """
        Get next action from queue.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Next action command or None
        """
        timeout = (timeout_ms or self.config.timeout_ms) / 1000.0

        try:
            return self._command_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_latest_action(self, action_type: Optional[ActionType] = None) -> Optional[ActionCommand]:
        """
        Get latest action, discarding older ones.

        Args:
            action_type: Filter by action type

        Returns:
            Latest matching action or None
        """
        latest = None

        while True:
            try:
                command = self._command_queue.get_nowait()
                if action_type is None or command.action_type == action_type:
                    latest = command
            except Empty:
                break

        return latest

    def on_action(self, callback: Callable[[ActionCommand], None]) -> None:
        """Register action callback."""
        self._on_action_callbacks.append(callback)

    def on_stop(self, callback: Callable[[], None]) -> None:
        """Register stop callback."""
        self._on_stop_callbacks.append(callback)

    def on_reset(self, callback: Callable[[], None]) -> None:
        """Register reset callback."""
        self._on_reset_callbacks.append(callback)

    def has_pending_actions(self) -> bool:
        """Check if there are pending actions."""
        return not self._command_queue.empty()

    def clear_queue(self) -> int:
        """Clear action queue, return number of cleared items."""
        count = 0
        while True:
            try:
                self._command_queue.get_nowait()
                count += 1
            except Empty:
                break
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get subscriber statistics."""
        return {
            "command_count": self._command_count,
            "validation_failures": self._validation_failures,
            "queue_size": self._command_queue.qsize(),
            "last_command_age": time.time() - self._last_command_time if self._last_command_time > 0 else -1,
        }

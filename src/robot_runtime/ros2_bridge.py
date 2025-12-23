"""
ROS 2 Bridge for Robot Runtime

Provides seamless integration between the Python robot_runtime module
and the ROS 2 ecosystem. Automatically uses ROS 2 when available,
falls back to standalone mode otherwise.

Usage:
    from robot_runtime.ros2_bridge import ROS2Bridge, get_ros2_bridge

    # Get singleton bridge instance
    bridge = get_ros2_bridge()

    # Check if ROS 2 is available
    if bridge.is_ros2_available():
        # Subscribe to robot state from ROS 2
        bridge.subscribe_robot_state(callback)

        # Publish action to ROS 2
        bridge.publish_action(action)
    else:
        # Fallback to direct hardware access
        pass
"""

import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)

# Try to import ROS 2
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup

    from std_msgs.msg import Float64MultiArray, Bool
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Pose, PoseArray

    ROS2_AVAILABLE = True
    logger.info("ROS 2 available")
except ImportError:
    logger.info("ROS 2 not available, using standalone mode")


@dataclass
class ROS2Config:
    """Configuration for ROS 2 bridge."""
    node_name: str = "robot_runtime_bridge"
    namespace: str = "dynamical"
    use_sim_time: bool = False
    qos_depth: int = 10


class ROS2Bridge:
    """
    Bridge between robot_runtime and ROS 2.

    Provides:
    - Topic subscription/publishing
    - Service client/server
    - Action client/server
    - Automatic spin in background thread
    """

    _instance: Optional['ROS2Bridge'] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[ROS2Config] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[ROS2Config] = None):
        if self._initialized:
            return

        self.config = config or ROS2Config()
        self._node: Optional[Node] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks
        self._robot_state_callbacks: List[Callable] = []
        self._safety_status_callbacks: List[Callable] = []

        # Cached state
        self._last_robot_state: Optional[Dict] = None
        self._last_safety_status: Optional[Dict] = None

        self._initialized = True

    def is_ros2_available(self) -> bool:
        """Check if ROS 2 is available."""
        return ROS2_AVAILABLE

    def initialize(self) -> bool:
        """
        Initialize ROS 2 node and start spinning.

        Returns:
            True if initialized successfully
        """
        if not ROS2_AVAILABLE:
            logger.warning("ROS 2 not available, cannot initialize bridge")
            return False

        if self._node is not None:
            logger.warning("Bridge already initialized")
            return True

        try:
            # Initialize ROS 2
            if not rclpy.ok():
                rclpy.init()

            # Create node
            self._node = _BridgeNode(self.config, self)

            # Create executor
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self._node)

            # Start spin thread
            self._running = True
            self._spin_thread = threading.Thread(
                target=self._spin_loop,
                name="ros2_bridge_spin",
                daemon=True
            )
            self._spin_thread.start()

            logger.info(f"ROS 2 bridge initialized: {self.config.node_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ROS 2 bridge: {e}")
            return False

    def shutdown(self):
        """Shutdown ROS 2 bridge."""
        self._running = False

        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

        if self._executor:
            self._executor.shutdown()

        if self._node:
            self._node.destroy_node()
            self._node = None

        if rclpy.ok():
            rclpy.shutdown()

        logger.info("ROS 2 bridge shutdown")

    def _spin_loop(self):
        """Background spin loop."""
        while self._running and rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                logger.error(f"Spin error: {e}")
                time.sleep(0.1)

    # =========================================================================
    # Robot State
    # =========================================================================

    def subscribe_robot_state(self, callback: Callable[[Dict], None]):
        """Subscribe to robot state updates."""
        self._robot_state_callbacks.append(callback)

    def get_robot_state(self) -> Optional[Dict]:
        """Get last received robot state."""
        return self._last_robot_state

    def _on_robot_state(self, state: Dict):
        """Called when robot state received from ROS 2."""
        self._last_robot_state = state
        for callback in self._robot_state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Robot state callback error: {e}")

    # =========================================================================
    # Safety Status
    # =========================================================================

    def subscribe_safety_status(self, callback: Callable[[Dict], None]):
        """Subscribe to safety status updates."""
        self._safety_status_callbacks.append(callback)

    def get_safety_status(self) -> Optional[Dict]:
        """Get last received safety status."""
        return self._last_safety_status

    def _on_safety_status(self, status: Dict):
        """Called when safety status received from ROS 2."""
        self._last_safety_status = status
        for callback in self._safety_status_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Safety status callback error: {e}")

    # =========================================================================
    # Commands
    # =========================================================================

    def publish_action(self, action: List[float]) -> bool:
        """
        Publish action command to ROS 2.

        Args:
            action: Joint velocities or positions

        Returns:
            True if published successfully
        """
        if self._node is None:
            return False

        return self._node.publish_action(action)

    def trigger_estop(self, reason: str = "API request") -> bool:
        """Trigger emergency stop via ROS 2."""
        if self._node is None:
            return False

        return self._node.trigger_estop(reason)

    def execute_skill(self, skill_id: str, parameters: Dict) -> Optional[str]:
        """
        Execute skill via ROS 2.

        Args:
            skill_id: Skill to execute
            parameters: Skill parameters

        Returns:
            Execution ID or None if failed
        """
        if self._node is None:
            return None

        return self._node.execute_skill(skill_id, parameters)


class _BridgeNode(Node):
    """Internal ROS 2 node for the bridge."""

    def __init__(self, config: ROS2Config, bridge: ROS2Bridge):
        super().__init__(
            config.node_name,
            namespace=config.namespace,
            parameter_overrides=[{'use_sim_time': config.use_sim_time}]
        )

        self.bridge = bridge
        self.callback_group = ReentrantCallbackGroup()

        # Import message types
        from dynamical_msgs.msg import RobotState, SafetyStatus, SkillStatus
        from dynamical_msgs.srv import ExecuteSkill, TriggerEstop

        # Subscribers
        self.robot_state_sub = self.create_subscription(
            RobotState,
            'robot_state',
            self._robot_state_callback,
            config.qos_depth,
            callback_group=self.callback_group
        )

        self.safety_status_sub = self.create_subscription(
            SafetyStatus,
            'safety_status',
            self._safety_status_callback,
            config.qos_depth,
            callback_group=self.callback_group
        )

        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray,
            'action_command',
            config.qos_depth
        )

        # Service clients
        self.execute_skill_client = self.create_client(
            ExecuteSkill,
            'execute_skill',
            callback_group=self.callback_group
        )

        self.trigger_estop_client = self.create_client(
            TriggerEstop,
            'trigger_estop',
            callback_group=self.callback_group
        )

    def _robot_state_callback(self, msg):
        """Handle robot state from ROS 2."""
        state = {
            'joint_positions': list(msg.joint_positions),
            'joint_velocities': list(msg.joint_velocities),
            'joint_torques': list(msg.joint_torques),
            'ee_pose': {
                'position': [
                    msg.ee_pose.position.x,
                    msg.ee_pose.position.y,
                    msg.ee_pose.position.z,
                ],
                'orientation': [
                    msg.ee_pose.orientation.w,
                    msg.ee_pose.orientation.x,
                    msg.ee_pose.orientation.y,
                    msg.ee_pose.orientation.z,
                ],
            },
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }
        self.bridge._on_robot_state(state)

    def _safety_status_callback(self, msg):
        """Handle safety status from ROS 2."""
        status = {
            'status': msg.status,
            'estop_active': msg.estop_active,
            'violations_count': len(msg.violations),
            'human_detected': msg.human_detected,
            'min_obstacle_distance': msg.min_obstacle_distance,
        }
        self.bridge._on_safety_status(status)

    def publish_action(self, action: List[float]) -> bool:
        """Publish action command."""
        try:
            msg = Float64MultiArray()
            msg.data = action
            self.action_pub.publish(msg)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish action: {e}")
            return False

    def trigger_estop(self, reason: str) -> bool:
        """Trigger E-stop via service."""
        if not self.trigger_estop_client.wait_for_service(timeout_sec=1.0):
            return False

        try:
            from dynamical_msgs.srv import TriggerEstop
            request = TriggerEstop.Request()
            request.reason = reason

            future = self.trigger_estop_client.call_async(request)
            # Don't wait for response - E-stop should be immediate
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to trigger E-stop: {e}")
            return False

    def execute_skill(self, skill_id: str, parameters: Dict) -> Optional[str]:
        """Execute skill via service."""
        import json
        from dynamical_msgs.srv import ExecuteSkill

        if not self.execute_skill_client.wait_for_service(timeout_sec=1.0):
            return None

        try:
            request = ExecuteSkill.Request()
            request.skill_id = skill_id
            request.parameters_json = json.dumps(parameters)

            future = self.execute_skill_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.result() is not None:
                return future.result().execution_id
            return None
        except Exception as e:
            self.get_logger().error(f"Failed to execute skill: {e}")
            return None


# Singleton accessor
_bridge_instance: Optional[ROS2Bridge] = None


def get_ros2_bridge(config: Optional[ROS2Config] = None) -> ROS2Bridge:
    """Get the singleton ROS 2 bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ROS2Bridge(config)
    return _bridge_instance

#!/usr/bin/env python3
"""
API Bridge Node - Connects ROS 2 to FastAPI Backend

Bridges between:
- ROS 2 topics/services (robot runtime)
- FastAPI REST endpoints (platform API)
- WebSocket connections (UI)

Enables the existing platform API to work with ROS 2 robot control.
"""

import json
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from dynamical_msgs.msg import (
    RobotState,
    SafetyStatus,
    SkillStatus,
    SystemStatus,
    TelemetryData,
)
from dynamical_msgs.srv import (
    ExecuteSkill,
    StopSkill,
    GetRobotState,
    TriggerEstop,
    ResetEstop,
    GetSystemStatus,
)

import aiohttp


class APIBridgeNode(Node):
    """
    Bridge between ROS 2 and FastAPI platform API.

    Subscribes to ROS 2 topics and forwards to API.
    Receives API requests and translates to ROS 2 services.
    """

    def __init__(self):
        super().__init__('api_bridge')

        # Parameters
        self.declare_parameter('api_url', 'http://localhost:8000')
        self.declare_parameter('api_key', '')
        self.declare_parameter('sync_interval_sec', 1.0)
        self.declare_parameter('enable_websocket', True)

        self.api_url = self.get_parameter('api_url').value
        self.api_key = self.get_parameter('api_key').value
        self.sync_interval = self.get_parameter('sync_interval_sec').value

        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()

        # State cache (for API queries)
        self.current_robot_state: Optional[RobotState] = None
        self.current_safety_status: Optional[SafetyStatus] = None
        self.current_skill_status: Optional[SkillStatus] = None

        # Subscribers
        self.robot_state_sub = self.create_subscription(
            RobotState,
            '/dynamical/robot_state',
            self.robot_state_callback,
            10,
            callback_group=self.callback_group
        )
        self.safety_status_sub = self.create_subscription(
            SafetyStatus,
            '/dynamical/safety_status',
            self.safety_status_callback,
            10,
            callback_group=self.callback_group
        )
        self.skill_status_sub = self.create_subscription(
            SkillStatus,
            '/dynamical/skill_status',
            self.skill_status_callback,
            10,
            callback_group=self.callback_group
        )

        # Service clients (to call ROS 2 services from API requests)
        self.execute_skill_client = self.create_client(
            ExecuteSkill,
            '/dynamical/execute_skill',
            callback_group=self.callback_group
        )
        self.stop_skill_client = self.create_client(
            StopSkill,
            '/dynamical/stop_skill',
            callback_group=self.callback_group
        )
        self.trigger_estop_client = self.create_client(
            TriggerEstop,
            '/dynamical/trigger_estop',
            callback_group=self.callback_group
        )
        self.reset_estop_client = self.create_client(
            ResetEstop,
            '/dynamical/reset_estop',
            callback_group=self.callback_group
        )

        # Publisher for commands from API
        self.command_pub = self.create_publisher(
            String,
            '/dynamical/api_commands',
            10
        )

        # Sync timer
        self.sync_timer = self.create_timer(
            self.sync_interval,
            self.sync_with_api,
            callback_group=self.callback_group
        )

        # Thread pool for blocking HTTP calls
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.get_logger().info(f'API Bridge initialized, connecting to {self.api_url}')

    def robot_state_callback(self, msg: RobotState):
        """Cache robot state for API queries."""
        self.current_robot_state = msg

    def safety_status_callback(self, msg: SafetyStatus):
        """Cache safety status for API queries."""
        self.current_safety_status = msg

    def skill_status_callback(self, msg: SkillStatus):
        """Cache skill status for API queries."""
        self.current_skill_status = msg

    def sync_with_api(self):
        """Sync state with API backend."""
        if self.current_robot_state is None:
            return

        # Build status payload
        status = {
            'robot_id': 'dynamical_001',
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'joint_positions': list(self.current_robot_state.joint_positions),
            'joint_velocities': list(self.current_robot_state.joint_velocities),
            'ee_position': [
                self.current_robot_state.ee_pose.position.x,
                self.current_robot_state.ee_pose.position.y,
                self.current_robot_state.ee_pose.position.z,
            ],
        }

        if self.current_safety_status:
            status['safety_status'] = self.current_safety_status.status
            status['estop_active'] = self.current_safety_status.estop_active

        if self.current_skill_status:
            status['skill_id'] = self.current_skill_status.skill_id
            status['skill_state'] = self.current_skill_status.state
            status['skill_progress'] = self.current_skill_status.progress

        # Send to API (non-blocking)
        self.executor.submit(self._send_status_to_api, status)

    def _send_status_to_api(self, status: Dict[str, Any]):
        """Send status to API (runs in thread pool)."""
        try:
            import requests
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['X-API-Key'] = self.api_key

            response = requests.post(
                f'{self.api_url}/api/v1/robot/status',
                json=status,
                headers=headers,
                timeout=1.0
            )

            if response.status_code != 200:
                self.get_logger().warn(f'API sync failed: {response.status_code}')

        except Exception as e:
            self.get_logger().debug(f'API sync error: {e}')

    async def execute_skill_from_api(self, skill_id: str, parameters: Dict) -> Dict:
        """Execute skill from API request."""
        if not self.execute_skill_client.wait_for_service(timeout_sec=1.0):
            return {'success': False, 'message': 'Skill executor service not available'}

        request = ExecuteSkill.Request()
        request.skill_id = skill_id
        request.parameters_json = json.dumps(parameters)

        future = self.execute_skill_client.call_async(request)
        response = await future

        return {
            'success': response.success,
            'message': response.message,
            'execution_id': response.execution_id,
        }

    async def trigger_estop_from_api(self, reason: str) -> Dict:
        """Trigger E-stop from API request."""
        if not self.trigger_estop_client.wait_for_service(timeout_sec=1.0):
            return {'success': False, 'message': 'E-stop service not available'}

        request = TriggerEstop.Request()
        request.reason = reason

        future = self.trigger_estop_client.call_async(request)
        response = await future

        return {
            'success': response.success,
            'message': response.message,
        }

    def get_current_state(self) -> Dict:
        """Get current robot state for API."""
        if self.current_robot_state is None:
            return {'error': 'No robot state available'}

        state = {
            'joint_positions': list(self.current_robot_state.joint_positions),
            'joint_velocities': list(self.current_robot_state.joint_velocities),
            'ee_pose': {
                'position': {
                    'x': self.current_robot_state.ee_pose.position.x,
                    'y': self.current_robot_state.ee_pose.position.y,
                    'z': self.current_robot_state.ee_pose.position.z,
                },
                'orientation': {
                    'x': self.current_robot_state.ee_pose.orientation.x,
                    'y': self.current_robot_state.ee_pose.orientation.y,
                    'z': self.current_robot_state.ee_pose.orientation.z,
                    'w': self.current_robot_state.ee_pose.orientation.w,
                },
            },
        }

        if self.current_safety_status:
            state['safety'] = {
                'status': self.current_safety_status.status,
                'estop_active': self.current_safety_status.estop_active,
                'violations_count': len(self.current_safety_status.violations),
            }

        return state


def main(args=None):
    rclpy.init(args=args)
    node = APIBridgeNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.executor.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

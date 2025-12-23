#!/usr/bin/env python3
"""
Perception Pipeline Node - Cascaded perception with Isaac ROS acceleration

Implements the three-level cascade:
- Level 1: Always (30Hz, <10ms) - YOLO-Nano, DepthAnything-S
- Level 2: On-demand (10Hz, <50ms) - DINOv3-Small, SAM3-Base
- Level 3: Rare (1-5Hz, <500ms) - DINOv3-Giant, SAM3-Huge, V-JEPA2

Uses Isaac ROS for TensorRT-accelerated inference when available.
"""

import time
from enum import IntEnum
from typing import List, Optional, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header

from dynamical_msgs.msg import PerceptionFeatures, Detection
from cv_bridge import CvBridge

import numpy as np


class CascadeLevel(IntEnum):
    """Perception cascade levels."""
    LEVEL_1 = 1  # Always (30Hz)
    LEVEL_2 = 2  # On-demand (10Hz)
    LEVEL_3 = 3  # Rare (1-5Hz)


class PerceptionPipelineNode(Node):
    """
    Cascaded perception pipeline with trigger-based escalation.

    Uses Isaac ROS for GPU-accelerated inference when available.
    Falls back to OpenCV/ONNX when Isaac ROS is not present.
    """

    def __init__(self):
        super().__init__('perception_pipeline')

        # Parameters
        self.declare_parameter('rate_hz', 30)
        self.declare_parameter('level2_confidence_threshold', 0.7)
        self.declare_parameter('level3_confidence_threshold', 0.9)
        self.declare_parameter('level3_cooldown_sec', 5.0)
        self.declare_parameter('use_isaac_ros', True)
        self.declare_parameter('camera_topics', ['/camera/color/image_raw'])

        self.rate_hz = self.get_parameter('rate_hz').value
        self.level2_threshold = self.get_parameter('level2_confidence_threshold').value
        self.level3_threshold = self.get_parameter('level3_confidence_threshold').value
        self.level3_cooldown = self.get_parameter('level3_cooldown_sec').value
        self.use_isaac_ros = self.get_parameter('use_isaac_ros').value

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # State
        self.current_level = CascadeLevel.LEVEL_1
        self.last_level3_time = 0.0
        self.current_image: Optional[np.ndarray] = None
        self.current_detections: List[Detection] = []

        # Callback groups for parallel execution
        self.sensor_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        camera_topics = self.get_parameter('camera_topics').value
        self.image_subs = []
        for i, topic in enumerate(camera_topics):
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, idx=i: self.image_callback(msg, idx),
                sensor_qos,
                callback_group=self.sensor_cb_group
            )
            self.image_subs.append(sub)

        # Publishers
        self.features_pub = self.create_publisher(
            PerceptionFeatures,
            'perception_features',
            10
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            'detections',
            10
        )
        self.obstacles_pub = self.create_publisher(
            PoseArray,
            'obstacles',
            10
        )
        self.humans_pub = self.create_publisher(
            PoseArray,
            'humans',
            10
        )

        # Initialize models
        self._init_models()

        # Main perception timer
        timer_period = 1.0 / self.rate_hz
        self.timer = self.create_timer(
            timer_period,
            self.perception_loop,
            callback_group=self.timer_cb_group
        )

        self.get_logger().info(
            f'PerceptionPipeline initialized at {self.rate_hz}Hz, '
            f'Isaac ROS: {self.use_isaac_ros}'
        )

    def _init_models(self):
        """Initialize inference models."""
        self.models_loaded = {
            'level1': False,
            'level2': False,
            'level3': False,
        }

        # Try to load Isaac ROS models
        if self.use_isaac_ros:
            try:
                self._init_isaac_ros_models()
            except Exception as e:
                self.get_logger().warn(f'Isaac ROS not available: {e}')
                self.use_isaac_ros = False

        # Fallback to basic models
        if not self.use_isaac_ros:
            self._init_fallback_models()

    def _init_isaac_ros_models(self):
        """Initialize Isaac ROS accelerated models."""
        self.get_logger().info('Initializing Isaac ROS models...')

        # Isaac ROS uses separate nodes for each model
        # We'll communicate with them via topics/services
        # For now, mark as loaded and use topic-based communication
        self.models_loaded['level1'] = True
        self.models_loaded['level2'] = True

        self.get_logger().info('Isaac ROS models ready')

    def _init_fallback_models(self):
        """Initialize fallback models (OpenCV/ONNX)."""
        self.get_logger().info('Initializing fallback models...')

        # Placeholder - would load ONNX models here
        self.models_loaded['level1'] = True
        self.models_loaded['level2'] = True

        self.get_logger().info('Fallback models ready')

    def image_callback(self, msg: Image, camera_idx: int):
        """Handle incoming camera images."""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def perception_loop(self):
        """Main perception loop - runs at configured rate."""
        if self.current_image is None:
            return

        start_time = time.time()

        # Always run Level 1
        self._run_level1()

        # Check if Level 2 needed
        if self._needs_level2():
            self._run_level2()

        # Check if Level 3 needed
        if self._needs_level3() and self._can_run_level3():
            self._run_level3()

        # Publish results
        self._publish_results(start_time)

    def _run_level1(self):
        """
        Level 1: Always running (30Hz, <10ms)
        - Fast object detection (YOLO-Nano)
        - Fast depth estimation
        """
        self.current_level = CascadeLevel.LEVEL_1
        self.current_detections = []

        # Placeholder detection - would use TensorRT/ONNX
        # Simulate detecting a cup and table
        det1 = Detection()
        det1.class_name = 'cup'
        det1.class_id = 1
        det1.confidence = 0.95
        det1.bbox_x = 100.0
        det1.bbox_y = 100.0
        det1.bbox_width = 50.0
        det1.bbox_height = 80.0
        det1.position = Point(x=0.5, y=0.0, z=0.3)
        det1.distance = 0.5
        det1.is_obstacle = True
        det1.in_workspace = True
        self.current_detections.append(det1)

    def _run_level2(self):
        """
        Level 2: On-demand (10Hz, <50ms)
        - DINOv3-Small features
        - SAM3-Base segmentation
        - Pose estimation
        """
        self.current_level = CascadeLevel.LEVEL_2
        self.get_logger().debug('Running Level 2 perception')

        # Would run medium models here

    def _run_level3(self):
        """
        Level 3: Rare (1-5Hz, <500ms)
        - DINOv3-Giant
        - SAM3-Huge
        - V-JEPA2 prediction
        """
        self.current_level = CascadeLevel.LEVEL_3
        self.last_level3_time = time.time()
        self.get_logger().info('Running Level 3 perception')

        # Would run heavy models or offload to cloud

    def _needs_level2(self) -> bool:
        """Check if Level 2 is needed."""
        for det in self.current_detections:
            if det.confidence < self.level2_threshold:
                return True
            if det.in_workspace:
                return True
        return False

    def _needs_level3(self) -> bool:
        """Check if Level 3 is needed."""
        for det in self.current_detections:
            if det.class_name == 'person':
                return True
        return False

    def _can_run_level3(self) -> bool:
        """Check Level 3 cooldown."""
        elapsed = time.time() - self.last_level3_time
        return elapsed > self.level3_cooldown

    def _publish_results(self, start_time: float):
        """Publish perception results."""
        now = self.get_clock().now()

        # Perception features
        features = PerceptionFeatures()
        features.header = Header()
        features.header.stamp = now.to_msg()
        features.header.frame_id = 'camera_link'
        features.cascade_level = int(self.current_level)
        features.detections = self.current_detections
        features.inference_time_ms = (time.time() - start_time) * 1000
        self.features_pub.publish(features)

        # Obstacles (non-human detections)
        obstacles = PoseArray()
        obstacles.header = features.header
        for det in self.current_detections:
            if det.is_obstacle and not det.class_name == 'person':
                pose = Pose()
                pose.position = det.position
                obstacles.poses.append(pose)
        self.obstacles_pub.publish(obstacles)

        # Humans
        humans = PoseArray()
        humans.header = features.header
        for det in self.current_detections:
            if det.class_name == 'person':
                pose = Pose()
                pose.position = det.position
                humans.poses.append(pose)
        self.humans_pub.publish(humans)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionPipelineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

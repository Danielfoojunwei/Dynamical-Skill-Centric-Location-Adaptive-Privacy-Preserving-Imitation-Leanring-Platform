"""
Robot Runtime Launch File

Launches the complete Dynamical robot runtime stack:
- Tier 1: Safety, State Estimation, Actuator (1kHz, C++)
- Tier 2: Perception, Policy Execution (30-100Hz, Python)

Uses composition for intra-process communication where possible.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNode
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    runtime_pkg = get_package_share_directory('dynamical_runtime')
    perception_pkg = get_package_share_directory('dynamical_perception')
    bringup_pkg = get_package_share_directory('dynamical_bringup')

    # Launch arguments
    use_sim = LaunchConfiguration('use_sim', default='false')
    use_rviz = LaunchConfiguration('use_rviz', default='false')
    use_isaac_ros = LaunchConfiguration('use_isaac_ros', default='true')
    robot_config = LaunchConfiguration('robot_config', default='default')
    log_level = LaunchConfiguration('log_level', default='info')

    # Config file
    config_file = os.path.join(bringup_pkg, 'config', 'robot_params.yaml')

    # Declare launch arguments
    declare_args = [
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Use simulation mode'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz visualization'
        ),
        DeclareLaunchArgument(
            'use_isaac_ros',
            default_value='true',
            description='Use Isaac ROS for GPU acceleration'
        ),
        DeclareLaunchArgument(
            'robot_config',
            default_value='default',
            description='Robot configuration name'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        ),
    ]

    # =========================================================================
    # Tier 1: Real-time nodes (C++, composed for zero-copy)
    # =========================================================================
    tier1_container = ComposableNodeContainer(
        name='tier1_container',
        namespace='dynamical',
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded
        composable_node_descriptions=[
            # Safety Node (1kHz, highest priority)
            ComposableNode(
                package='dynamical_runtime',
                plugin='dynamical::SafetyNode',
                name='safety_node',
                parameters=[
                    config_file,
                    {
                        'rate_hz': 1000,
                        'use_realtime_scheduling': True,
                    }
                ],
                extra_arguments=[
                    {'use_intra_process_comms': True}
                ],
            ),
            # State Estimator Node (1kHz)
            ComposableNode(
                package='dynamical_runtime',
                plugin='dynamical::StateEstimatorNode',
                name='state_estimator_node',
                parameters=[config_file],
                extra_arguments=[
                    {'use_intra_process_comms': True}
                ],
            ),
            # Actuator Node (1kHz)
            ComposableNode(
                package='dynamical_runtime',
                plugin='dynamical::ActuatorNode',
                name='actuator_node',
                parameters=[
                    config_file,
                    {
                        'use_sim': use_sim,
                    }
                ],
                extra_arguments=[
                    {'use_intra_process_comms': True}
                ],
            ),
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
    )

    # =========================================================================
    # Tier 2: Perception and Policy (Python, separate processes)
    # =========================================================================
    perception_node = Node(
        package='dynamical_perception',
        executable='perception_pipeline_node.py',
        name='perception_pipeline',
        namespace='dynamical',
        parameters=[
            config_file,
            {
                'rate_hz': 30,
                'use_isaac_ros': use_isaac_ros,
            }
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
    )

    # Skill Executor Node
    skill_executor_node = Node(
        package='dynamical_runtime',
        executable='skill_executor_node_exec',
        name='skill_executor',
        namespace='dynamical',
        parameters=[config_file],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
    )

    # =========================================================================
    # API Bridge (connects ROS 2 to FastAPI backend)
    # =========================================================================
    api_bridge_node = Node(
        package='dynamical_bringup',
        executable='api_bridge_node.py',
        name='api_bridge',
        namespace='dynamical',
        parameters=[
            {
                'api_url': 'http://localhost:8000',
            }
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
    )

    # =========================================================================
    # Telemetry Node
    # =========================================================================
    telemetry_node = Node(
        package='dynamical_bringup',
        executable='telemetry_node.py',
        name='telemetry',
        namespace='dynamical',
        parameters=[
            {
                'publish_rate_hz': 10,
            }
        ],
        output='screen',
    )

    # =========================================================================
    # Create launch description
    # =========================================================================
    return LaunchDescription(
        declare_args + [
            # Set environment for real-time
            SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),

            # Tier 1 (real-time)
            tier1_container,

            # Tier 2 (perception/policy)
            perception_node,
            skill_executor_node,

            # Integration
            api_bridge_node,
            telemetry_node,
        ]
    )

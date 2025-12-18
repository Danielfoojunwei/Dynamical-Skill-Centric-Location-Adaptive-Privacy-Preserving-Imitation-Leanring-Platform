#!/usr/bin/env python3
"""
Dynamical.ai Isaac Lab Simulation Demo

This script demonstrates the complete Dynamical.ai system running with
NVIDIA Isaac Lab simulation, including:

1. Isaac Lab Environment Setup
2. Robot Teleoperation with Virtual DYGlove
3. Pick-and-Place Task Execution
4. Real-time Dashboard Streaming
5. Federated Learning Updates
6. Safety System Monitoring

Usage:
    # Basic demo
    python scripts/demo_isaac_lab_simulation.py

    # With specific robot
    python scripts/demo_isaac_lab_simulation.py --robot franka_panda

    # Headless mode (no visualization)
    python scripts/demo_isaac_lab_simulation.py --headless

    # Run specific task
    python scripts/demo_isaac_lab_simulation.py --task pick_place

Requirements:
    - Python 3.8+
    - NVIDIA Isaac Sim (optional, falls back to standalone mode)
    - Running FastAPI backend (port 8000)
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import IsaacLabEnvironment, DynamicalSimBridge
from src.simulation.isaac_lab import (
    SimulationConfig,
    SimulationMode,
    RobotType,
    IsaacRobotController,
    IsaacCameraManager,
    PickPlaceTask,
    ManipulationTask,
)
from src.simulation.scenes import VirtualDYGlove, TabletopScene
from src.simulation.bridge import TelemetryPublisher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("demo")


class DemoRunner:
    """
    Orchestrates the complete demo of the Dynamical.ai system with Isaac Lab.
    """

    def __init__(
        self,
        robot_type: str = "franka_panda",
        task_type: str = "pick_place",
        scene_type: str = "tabletop",
        headless: bool = False,
        duration: float = 60.0,
        enable_fl: bool = True,
        enable_teleop: bool = True,
    ):
        """
        Initialize demo runner.

        Args:
            robot_type: Robot type (franka_panda, ur5e, daimon_vtla)
            task_type: Task type (pick_place, reach, grasp)
            scene_type: Scene type (tabletop, warehouse)
            headless: Run without visualization
            duration: Demo duration in seconds
            enable_fl: Enable federated learning simulation
            enable_teleop: Enable teleoperation with virtual glove
        """
        self.robot_type = robot_type
        self.task_type = task_type
        self.scene_type = scene_type
        self.headless = headless
        self.duration = duration
        self.enable_fl = enable_fl
        self.enable_teleop = enable_teleop

        # Components (initialized in setup)
        self.environment: Optional[IsaacLabEnvironment] = None
        self.controller: Optional[IsaacRobotController] = None
        self.camera_manager: Optional[IsaacCameraManager] = None
        self.task: Optional[PickPlaceTask] = None
        self.bridge: Optional[DynamicalSimBridge] = None
        self.glove: Optional[VirtualDYGlove] = None
        self.scene: Optional[TabletopScene] = None

        # State
        self._running = False
        self._start_time = 0.0
        self._step_count = 0
        self._fl_round = 0

        # Metrics
        self._episode_rewards = []
        self._episode_successes = []
        self._fl_losses = []

    async def setup(self) -> None:
        """Setup all demo components."""
        logger.info("=" * 60)
        logger.info("Dynamical.ai Isaac Lab Simulation Demo")
        logger.info("=" * 60)
        logger.info(f"Robot: {self.robot_type}")
        logger.info(f"Task: {self.task_type}")
        logger.info(f"Scene: {self.scene_type}")
        logger.info(f"Duration: {self.duration}s")
        logger.info("=" * 60)

        # 1. Setup Isaac Lab Environment
        logger.info("\n[1/6] Setting up Isaac Lab environment...")
        await self._setup_environment()

        # 2. Setup Robot Controller
        logger.info("[2/6] Initializing robot controller...")
        self._setup_controller()

        # 3. Setup Camera Manager
        logger.info("[3/6] Setting up camera manager...")
        self._setup_cameras()

        # 4. Setup Task
        logger.info("[4/6] Creating task environment...")
        self._setup_task()

        # 5. Setup Bridge for Dashboard Streaming
        logger.info("[5/6] Connecting to dashboard...")
        await self._setup_bridge()

        # 6. Setup Virtual Glove (if enabled)
        if self.enable_teleop:
            logger.info("[6/6] Initializing virtual glove...")
            await self._setup_glove()
        else:
            logger.info("[6/6] Teleoperation disabled")

        logger.info("\nSetup complete! Starting demo...\n")

    async def _setup_environment(self) -> None:
        """Setup Isaac Lab environment."""
        try:
            robot_type = RobotType(self.robot_type)
        except ValueError:
            robot_type = RobotType.FRANKA_PANDA
            logger.warning(f"Unknown robot type '{self.robot_type}', using Franka Panda")

        config = SimulationConfig(
            mode=SimulationMode.STANDALONE,  # Use ISAAC_LAB when available
            robot_type=robot_type,
            scene_type=self.scene_type,
            num_cameras=4,
            headless=self.headless,
            dt=1.0 / 60.0,
        )

        self.environment = IsaacLabEnvironment(config=config, mode="standalone")
        await self.environment.setup()

        logger.info(f"  - Simulation mode: {config.mode.value}")
        logger.info(f"  - Physics rate: {1/config.dt:.0f} Hz")

    def _setup_controller(self) -> None:
        """Setup robot controller."""
        from src.simulation.isaac_lab.robot_controller import ControllerConfig, ControlMode

        config = ControllerConfig(
            control_mode=ControlMode.POSITION,
            control_frequency=100.0,
            smoothing_alpha=0.1,
        )

        robot_config = IsaacLabEnvironment.ROBOT_CONFIGS.get(
            RobotType(self.robot_type),
            IsaacLabEnvironment.ROBOT_CONFIGS[RobotType.FRANKA_PANDA]
        )

        self.controller = IsaacRobotController(
            config=config,
            n_joints=robot_config["n_joints"],
        )

        logger.info(f"  - Control mode: {config.control_mode.value}")
        logger.info(f"  - Joints: {robot_config['n_joints']}")

    def _setup_cameras(self) -> None:
        """Setup camera manager."""
        self.camera_manager = IsaacCameraManager(preset="manipulation_4cam")
        logger.info(f"  - Cameras: {len(self.camera_manager.cameras)}")

    def _setup_task(self) -> None:
        """Setup task environment."""
        from src.simulation.isaac_lab.task_environments import PickPlaceTaskConfig

        if self.task_type == "pick_place":
            config = PickPlaceTaskConfig(
                max_episode_length=500,
                success_threshold=0.03,
                randomize_initial=True,
                randomize_target=True,
            )
            self.task = PickPlaceTask(config=config)
        else:
            self.task = ManipulationTask(objective=self.task_type)

        # Connect task to robot state
        if self.environment:
            self.task.set_robot_state(self.environment.get_robot_state())

        logger.info(f"  - Task type: {self.task_type}")
        logger.info(f"  - Max steps: {self.task.config.max_episode_length}")

    async def _setup_bridge(self) -> None:
        """Setup bridge for dashboard streaming."""
        from src.simulation.bridge.dynamical_bridge import BridgeConfig

        config = BridgeConfig(
            api_host="localhost",
            api_port=8000,
            robot_state_rate=60.0,
            camera_rate=30.0,
            metrics_rate=1.0,
            enable_zmq=False,  # Use HTTP/WebSocket only
        )

        self.bridge = DynamicalSimBridge(config=config)
        await self.bridge.start()

        logger.info(f"  - API: http://{config.api_host}:{config.api_port}")
        logger.info(f"  - Robot state rate: {config.robot_state_rate} Hz")

    async def _setup_glove(self) -> None:
        """Setup virtual glove for teleoperation."""
        from src.simulation.scenes.virtual_glove import GloveConfig, InputMode

        config = GloveConfig(
            update_rate=100.0,
            input_mode=InputMode.SCRIPTED,  # Use scripted trajectory for demo
            smoothing_alpha=0.3,
        )

        self.glove = VirtualDYGlove(config=config)

        # Set demo trajectory
        trajectory = VirtualDYGlove.create_pick_trajectory() + VirtualDYGlove.create_place_trajectory()
        self.glove.set_trajectory(trajectory)

        await self.glove.start()

        logger.info(f"  - Update rate: {config.update_rate} Hz")
        logger.info(f"  - Mode: {config.input_mode.value}")

    async def run(self) -> None:
        """Run the demo."""
        self._running = True
        self._start_time = time.time()

        # Reset environment and task
        self.environment.reset()
        self.task.reset()

        logger.info("Demo started! Press Ctrl+C to stop.\n")

        episode = 0
        episode_reward = 0.0

        try:
            while self._running and (time.time() - self._start_time) < self.duration:
                # Get teleoperation input from virtual glove
                if self.enable_teleop and self.glove:
                    glove_state = self.glove.get_state()
                    action = self._glove_to_action(glove_state)
                else:
                    # Use task-based control
                    action = self._get_task_action()

                # Step simulation
                obs = self.environment.step(action)
                robot_state = self.environment.get_robot_state()

                # Update task
                self.task.set_robot_state(robot_state)
                task_obs, reward_info, done, info = self.task.step(action)
                episode_reward += reward_info.total

                # Stream data to dashboard
                await self._stream_to_dashboard(robot_state, task_obs, reward_info)

                # Check episode completion
                if done:
                    episode += 1
                    self._episode_rewards.append(episode_reward)
                    self._episode_successes.append(info.get("success", False))

                    logger.info(f"Episode {episode} completed: reward={episode_reward:.2f}, success={info.get('success')}")

                    # Reset for next episode
                    self.environment.reset()
                    self.task.reset()
                    episode_reward = 0.0

                # Simulate federated learning update
                if self.enable_fl and self._step_count % 1000 == 0:
                    await self._simulate_fl_round()

                self._step_count += 1

                # Control loop rate
                await asyncio.sleep(1/60)

        except asyncio.CancelledError:
            logger.info("Demo cancelled")

        self._running = False

    def _glove_to_action(self, glove_state) -> np.ndarray:
        """Convert glove state to robot action."""
        # Map glove position to end-effector target
        ee_target = glove_state.position * 0.5 + np.array([0.5, 0, 0.3])

        # Set target on environment
        self.environment.set_robot_target(
            target_position=ee_target,
            target_orientation=glove_state.orientation,
        )

        # Set gripper based on grip state
        self.environment.set_gripper(1.0 - glove_state.grip_state)

        # Return current joint positions (actual control happens internally)
        return self.environment.get_robot_state().joint_positions

    def _get_task_action(self) -> np.ndarray:
        """Get action from task waypoints."""
        # Get waypoints from task
        if hasattr(self.task, 'get_phase_waypoints'):
            waypoints = self.task.get_phase_waypoints()
            phase = self.task._phase.value if hasattr(self.task, '_phase') else 'approach'

            if phase in waypoints:
                target = waypoints[phase]
                self.environment.set_robot_target(target_position=target)

        # Return current joint positions
        return self.environment.get_robot_state().joint_positions

    async def _stream_to_dashboard(
        self,
        robot_state,
        task_obs: Dict,
        reward_info,
    ) -> None:
        """Stream data to dashboard via bridge."""
        if not self.bridge:
            return

        # Stream robot state
        self.bridge.publish_robot_state(
            joint_positions=robot_state.joint_positions,
            joint_velocities=robot_state.joint_velocities,
            joint_torques=robot_state.joint_torques,
            ee_position=robot_state.ee_position,
            ee_orientation=robot_state.ee_orientation,
            gripper_state=robot_state.gripper_state,
        )

        # Stream task state
        self.bridge.publish_task_state(
            task_type=self.task_type,
            phase=task_obs.get("phase", "unknown"),
            step=task_obs.get("step", self._step_count),
            reward=reward_info.total,
            success=task_obs.get("success", False),
            objects=task_obs.get("objects", {}),
        )

        # Stream metrics (at lower rate)
        if self._step_count % 60 == 0:
            metrics = self.environment.get_metrics()
            self.bridge.publish_metrics(
                physics_fps=metrics.physics_fps,
                render_fps=metrics.render_fps,
                step_time_ms=metrics.step_time_ms,
                real_time_factor=metrics.real_time_factor,
                total_steps=metrics.total_steps,
                simulation_time=metrics.simulation_time,
            )

    async def _simulate_fl_round(self) -> None:
        """Simulate a federated learning round."""
        self._fl_round += 1

        # Simulate loss decreasing over rounds
        base_loss = 1.0 / (1 + 0.1 * self._fl_round)
        noise = np.random.randn() * 0.05
        global_loss = max(0.01, base_loss + noise)

        self._fl_losses.append(global_loss)

        # Simulate local losses from multiple clients
        local_losses = {
            f"client_{i}": global_loss + np.random.randn() * 0.1
            for i in range(3)
        }

        logger.info(f"FL Round {self._fl_round}: global_loss={global_loss:.4f}")

        # Publish FL update
        if self.bridge:
            # Note: This would use stream_fl_update from WebSocket
            pass

    async def cleanup(self) -> None:
        """Cleanup demo resources."""
        logger.info("\nCleaning up...")

        if self.glove:
            await self.glove.stop()

        if self.bridge:
            await self.bridge.stop()

        if self.environment:
            self.environment.close()

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print demo summary."""
        elapsed = time.time() - self._start_time

        logger.info("\n" + "=" * 60)
        logger.info("Demo Summary")
        logger.info("=" * 60)
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Total steps: {self._step_count}")
        logger.info(f"Episodes: {len(self._episode_rewards)}")

        if self._episode_rewards:
            logger.info(f"Average reward: {np.mean(self._episode_rewards):.2f}")
            logger.info(f"Success rate: {np.mean(self._episode_successes)*100:.1f}%")

        if self._fl_losses:
            logger.info(f"FL rounds: {self._fl_round}")
            logger.info(f"Final loss: {self._fl_losses[-1]:.4f}")

        logger.info("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dynamical.ai Isaac Lab Simulation Demo")
    parser.add_argument("--robot", type=str, default="franka_panda",
                        help="Robot type (franka_panda, ur5e, daimon_vtla)")
    parser.add_argument("--task", type=str, default="pick_place",
                        help="Task type (pick_place, reach, grasp)")
    parser.add_argument("--scene", type=str, default="tabletop",
                        help="Scene type (tabletop, warehouse)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Demo duration in seconds")
    parser.add_argument("--no-fl", action="store_true",
                        help="Disable federated learning simulation")
    parser.add_argument("--no-teleop", action="store_true",
                        help="Disable teleoperation")

    args = parser.parse_args()

    # Create demo runner
    demo = DemoRunner(
        robot_type=args.robot,
        task_type=args.task,
        scene_type=args.scene,
        headless=args.headless,
        duration=args.duration,
        enable_fl=not args.no_fl,
        enable_teleop=not args.no_teleop,
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal...")
        demo._running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await demo.setup()
        await demo.run()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

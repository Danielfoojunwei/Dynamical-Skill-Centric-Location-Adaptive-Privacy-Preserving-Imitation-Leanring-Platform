"""
Comprehensive tests for Isaac Lab simulation module.

Tests cover:
- Environment creation and configuration
- Robot state management
- Camera observations
- Bridge connectivity
- Scene creation
- Virtual glove simulation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict


class TestIsaacLabEnvironment:
    """Tests for IsaacLabEnvironment."""

    def test_import_environment(self):
        """Test that environment can be imported."""
        from src.simulation.isaac_lab.environment import (
            IsaacLabEnvironment,
            SimulationMode,
            RobotType,
            SimulationConfig,
            RobotState,
            CameraFrame,
            SimulationMetrics,
        )
        assert IsaacLabEnvironment is not None
        assert SimulationMode.STANDALONE.value == "standalone"
        assert RobotType.FRANKA_PANDA.value == "franka_panda"

    def test_simulation_modes(self):
        """Test all simulation modes are defined."""
        from src.simulation.isaac_lab.environment import SimulationMode

        assert SimulationMode.STANDALONE.value == "standalone"
        assert SimulationMode.ISAAC_LAB.value == "isaac_lab"
        assert SimulationMode.ISAAC_GYM.value == "isaac_gym"

    def test_robot_types(self):
        """Test all robot types are defined."""
        from src.simulation.isaac_lab.environment import RobotType

        assert RobotType.FRANKA_PANDA.value == "franka_panda"
        assert RobotType.UR5E.value == "ur5e"
        assert RobotType.UR10.value == "ur10"
        assert RobotType.KUKA_IIWA.value == "kuka_iiwa"
        assert RobotType.DAIMON_VTLA.value == "daimon_vtla"

    def test_simulation_config_defaults(self):
        """Test SimulationConfig default values."""
        from src.simulation.isaac_lab.environment import SimulationConfig, SimulationMode, RobotType

        config = SimulationConfig()
        assert config.mode == SimulationMode.STANDALONE
        assert config.dt == pytest.approx(1.0 / 60.0)
        assert config.render_dt == pytest.approx(1.0 / 30.0)
        assert config.robot_type == RobotType.FRANKA_PANDA
        assert config.num_cameras == 4
        assert config.camera_resolution == (640, 480)
        assert config.num_envs == 1
        assert config.enable_teleoperation == True

    def test_robot_state_dataclass(self):
        """Test RobotState dataclass creation."""
        from src.simulation.isaac_lab.environment import RobotState

        state = RobotState(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_torques=np.zeros(7),
            ee_position=np.array([0.5, 0.0, 0.5]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            ee_linear_velocity=np.zeros(3),
            ee_angular_velocity=np.zeros(3),
            gripper_state=0.5,
            timestamp=0.0,
        )
        assert state.joint_positions.shape == (7,)
        assert state.ee_position.shape == (3,)
        assert state.gripper_state == 0.5

    def test_camera_frame_dataclass(self):
        """Test CameraFrame dataclass creation."""
        from src.simulation.isaac_lab.environment import CameraFrame

        frame = CameraFrame(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640), dtype=np.float32),
            camera_id="front",
            timestamp=0.1,
        )
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
        assert frame.camera_id == "front"

    def test_environment_creation_standalone(self):
        """Test environment creation in standalone mode."""
        from src.simulation.isaac_lab.environment import IsaacLabEnvironment, SimulationConfig, SimulationMode

        config = SimulationConfig(mode=SimulationMode.STANDALONE)
        env = IsaacLabEnvironment(config=config)
        assert env is not None
        assert env.config.mode == SimulationMode.STANDALONE

    def test_environment_reset(self):
        """Test environment reset."""
        from src.simulation.isaac_lab.environment import IsaacLabEnvironment, SimulationConfig, SimulationMode

        config = SimulationConfig(mode=SimulationMode.STANDALONE)
        env = IsaacLabEnvironment(config=config)
        obs = env.reset()
        assert obs is not None

    def test_environment_step(self):
        """Test environment step."""
        from src.simulation.isaac_lab.environment import IsaacLabEnvironment, SimulationConfig, SimulationMode

        config = SimulationConfig(mode=SimulationMode.STANDALONE)
        env = IsaacLabEnvironment(config=config)
        env.reset()

        # Create action (7 joints + 1 gripper)
        action = np.zeros(8)
        result = env.step(action)
        assert result is not None


class TestRobotController:
    """Tests for IsaacRobotController."""

    def test_import_controller(self):
        """Test that controller can be imported."""
        from src.simulation.isaac_lab.robot_controller import IsaacRobotController
        assert IsaacRobotController is not None

    def test_controller_creation(self):
        """Test controller creation."""
        from src.simulation.isaac_lab.robot_controller import IsaacRobotController, ControllerConfig

        config = ControllerConfig()
        controller = IsaacRobotController(config=config, n_joints=7)
        assert controller is not None


class TestCameraManager:
    """Tests for IsaacCameraManager."""

    def test_import_camera_manager(self):
        """Test that camera manager can be imported."""
        from src.simulation.isaac_lab.camera_manager import IsaacCameraManager
        assert IsaacCameraManager is not None


class TestTaskEnvironments:
    """Tests for task environments."""

    def test_import_tasks(self):
        """Test that tasks can be imported."""
        from src.simulation.isaac_lab.task_environments import ManipulationTask, PickPlaceTask
        assert ManipulationTask is not None
        assert PickPlaceTask is not None


class TestDynamicalBridge:
    """Tests for DynamicalSimBridge."""

    def test_import_bridge(self):
        """Test that bridge can be imported."""
        from src.simulation.bridge.dynamical_bridge import DynamicalSimBridge
        assert DynamicalSimBridge is not None

    def test_bridge_stream_types(self):
        """Test stream types are defined."""
        from src.simulation.bridge.dynamical_bridge import StreamType

        assert StreamType.ROBOT_STATE.value == "robot_state"
        assert StreamType.CAMERA_FRAME.value == "camera_frame"
        assert StreamType.SENSOR_DATA.value == "sensor_data"
        assert StreamType.TASK_STATE.value == "task_state"
        assert StreamType.METRICS.value == "metrics"


class TestTelemetryPublisher:
    """Tests for TelemetryPublisher."""

    def test_import_publisher(self):
        """Test that publisher can be imported."""
        from src.simulation.bridge.telemetry_publisher import TelemetryPublisher
        assert TelemetryPublisher is not None


class TestActionSubscriber:
    """Tests for ActionSubscriber."""

    def test_import_subscriber(self):
        """Test that subscriber can be imported."""
        from src.simulation.bridge.action_subscriber import ActionSubscriber
        assert ActionSubscriber is not None


class TestWarehouseScene:
    """Tests for WarehouseScene."""

    def test_import_scene(self):
        """Test that scene can be imported."""
        from src.simulation.scenes.warehouse_scene import WarehouseScene
        assert WarehouseScene is not None


class TestTabletopScene:
    """Tests for TabletopScene."""

    def test_import_scene(self):
        """Test that scene can be imported."""
        from src.simulation.scenes.tabletop_scene import TabletopScene
        assert TabletopScene is not None


class TestVirtualGlove:
    """Tests for VirtualDYGlove."""

    def test_import_glove(self):
        """Test that virtual glove can be imported."""
        from src.simulation.scenes.virtual_glove import VirtualDYGlove, GloveConfig, InputMode
        assert VirtualDYGlove is not None
        assert GloveConfig is not None

    def test_input_modes(self):
        """Test input modes are defined."""
        from src.simulation.scenes.virtual_glove import InputMode

        assert InputMode.KEYBOARD.value == "keyboard"
        assert InputMode.MOUSE.value == "mouse"
        assert InputMode.GAMEPAD.value == "gamepad"
        assert InputMode.SCRIPTED.value == "scripted"
        assert InputMode.PLAYBACK.value == "playback"

    def test_glove_config_defaults(self):
        """Test GloveConfig default values."""
        from src.simulation.scenes.virtual_glove import GloveConfig

        config = GloveConfig()
        assert config.update_rate == 1000.0
        assert config.smoothing_alpha == 0.3
        assert config.enable_noise == True

    def test_virtual_glove_creation(self):
        """Test virtual glove creation."""
        from src.simulation.scenes.virtual_glove import VirtualDYGlove, GloveConfig

        config = GloveConfig()
        glove = VirtualDYGlove(config=config)
        assert glove is not None


class TestSimulationIntegration:
    """Integration tests for simulation components."""

    def test_full_simulation_loop(self):
        """Test a complete simulation loop."""
        from src.simulation.isaac_lab.environment import (
            IsaacLabEnvironment,
            SimulationConfig,
            SimulationMode,
        )

        # Create environment
        config = SimulationConfig(
            mode=SimulationMode.STANDALONE,
            num_envs=1,
            headless=True,
        )
        env = IsaacLabEnvironment(config=config)

        # Reset
        obs = env.reset()
        assert obs is not None

        # Run 10 steps
        for i in range(10):
            action = np.zeros(8)  # 7 joints + gripper
            result = env.step(action)
            assert result is not None

    def test_multi_camera_observation(self):
        """Test multi-camera observation retrieval."""
        from src.simulation.isaac_lab.environment import (
            IsaacLabEnvironment,
            SimulationConfig,
            SimulationMode,
        )

        config = SimulationConfig(
            mode=SimulationMode.STANDALONE,
            num_cameras=4,
        )
        env = IsaacLabEnvironment(config=config)
        env.reset()

        # Get observations
        obs = env.get_observations()
        assert obs is not None


class TestSimulationWebSocket:
    """Tests for simulation WebSocket integration."""

    def test_import_simulation_ws(self):
        """Test that simulation WebSocket can be imported."""
        from src.platform.api.simulation_ws import ConnectionManager
        assert ConnectionManager is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

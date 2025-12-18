"""
Isaac Lab Task Environments

Defines manipulation task environments for training and evaluation.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task completion status."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class TaskConfig:
    """Base configuration for tasks."""
    max_episode_length: int = 500
    success_threshold: float = 0.02  # meters
    reward_scale: float = 1.0
    enable_dense_reward: bool = True
    randomize_initial: bool = True
    randomize_target: bool = True


@dataclass
class RewardInfo:
    """Reward breakdown information."""
    total: float = 0.0
    sparse: float = 0.0
    dense: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)


class BaseTask(ABC):
    """Base class for manipulation tasks."""

    def __init__(self, config: Optional[TaskConfig] = None):
        """Initialize task."""
        self.config = config or TaskConfig()
        self._step_count = 0
        self._status = TaskStatus.RUNNING
        self._episode_reward = 0.0

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset task to initial state."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], RewardInfo, bool, Dict]:
        """Execute one step of the task."""
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        pass

    @abstractmethod
    def compute_reward(self) -> RewardInfo:
        """Compute reward for current state."""
        pass

    @abstractmethod
    def check_success(self) -> bool:
        """Check if task is successfully completed."""
        pass

    def get_status(self) -> TaskStatus:
        """Get current task status."""
        return self._status


@dataclass
class ManipulationTaskConfig(TaskConfig):
    """Configuration for general manipulation tasks."""
    workspace_bounds: Tuple[np.ndarray, np.ndarray] = None
    object_types: List[str] = field(default_factory=lambda: ["cube", "cylinder", "sphere"])
    num_objects: int = 3
    enable_collision_penalty: bool = True
    enable_smooth_reward: bool = True

    def __post_init__(self):
        if self.workspace_bounds is None:
            self.workspace_bounds = (
                np.array([0.2, -0.4, 0.0]),  # min xyz
                np.array([0.8, 0.4, 0.5]),   # max xyz
            )


class ManipulationTask(BaseTask):
    """
    General manipulation task environment.

    Supports various manipulation objectives:
    - Reaching
    - Grasping
    - Placing
    - Stacking
    """

    def __init__(
        self,
        config: Optional[ManipulationTaskConfig] = None,
        objective: str = "reach",
    ):
        """
        Initialize manipulation task.

        Args:
            config: Task configuration
            objective: Task objective ("reach", "grasp", "place", "stack")
        """
        super().__init__(config or ManipulationTaskConfig())
        self.objective = objective

        # Object states
        self._objects: Dict[str, Dict[str, Any]] = {}
        self._target_position = np.zeros(3)
        self._target_object_id: Optional[str] = None

        # Robot state reference (set by environment)
        self._robot_state = None
        self._prev_ee_position = None

        # Initialize objects
        self._initialize_objects()

        logger.info(f"ManipulationTask initialized with objective: {objective}")

    def _initialize_objects(self) -> None:
        """Initialize manipulation objects."""
        config = self.config
        np.random.seed(None)

        for i in range(config.num_objects):
            obj_type = config.object_types[i % len(config.object_types)]
            obj_id = f"{obj_type}_{i}"

            # Random position within workspace
            pos = np.random.uniform(
                config.workspace_bounds[0],
                config.workspace_bounds[1],
            )
            pos[2] = 0.02  # On table

            self._objects[obj_id] = {
                "type": obj_type,
                "position": pos,
                "orientation": np.array([1, 0, 0, 0]),
                "grasped": False,
                "size": np.random.uniform(0.03, 0.06),
            }

        # Set target object
        if self._objects:
            self._target_object_id = list(self._objects.keys())[0]
            self._target_position = self._objects[self._target_object_id]["position"].copy()

    def set_robot_state(self, robot_state: Any) -> None:
        """Set reference to robot state."""
        self._robot_state = robot_state

    def reset(self) -> Dict[str, Any]:
        """Reset task."""
        self._step_count = 0
        self._status = TaskStatus.RUNNING
        self._episode_reward = 0.0
        self._prev_ee_position = None

        # Randomize if configured
        if self.config.randomize_initial:
            self._randomize_objects()

        if self.config.randomize_target:
            self._randomize_target()

        return self.get_observation()

    def _randomize_objects(self) -> None:
        """Randomize object positions."""
        for obj_id, obj in self._objects.items():
            pos = np.random.uniform(
                self.config.workspace_bounds[0],
                self.config.workspace_bounds[1],
            )
            pos[2] = 0.02
            obj["position"] = pos
            obj["grasped"] = False

        if self._target_object_id and self._target_object_id in self._objects:
            self._target_position = self._objects[self._target_object_id]["position"].copy()

    def _randomize_target(self) -> None:
        """Randomize target position."""
        if self.objective == "place":
            # Random place position
            self._target_position = np.random.uniform(
                self.config.workspace_bounds[0],
                self.config.workspace_bounds[1],
            )
            self._target_position[2] = 0.02
        elif self.objective == "stack":
            # Stack on another object
            if len(self._objects) > 1:
                other_objects = [k for k in self._objects.keys() if k != self._target_object_id]
                target_obj = np.random.choice(other_objects)
                self._target_position = self._objects[target_obj]["position"].copy()
                self._target_position[2] += self._objects[target_obj]["size"] + 0.01

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], RewardInfo, bool, Dict]:
        """
        Execute task step.

        Args:
            action: Robot action

        Returns:
            observation, reward_info, done, info
        """
        self._step_count += 1

        # Update object states based on grasping
        self._update_grasp_state()

        # Compute reward
        reward_info = self.compute_reward()
        self._episode_reward += reward_info.total

        # Check termination
        done = False
        info = {"status": self._status.value}

        if self.check_success():
            self._status = TaskStatus.SUCCESS
            done = True
            info["success"] = True
            reward_info.sparse = 10.0
            reward_info.total += 10.0
        elif self._step_count >= self.config.max_episode_length:
            self._status = TaskStatus.TIMEOUT
            done = True
            info["success"] = False

        # Store previous position
        if self._robot_state:
            self._prev_ee_position = self._robot_state.ee_position.copy()

        return self.get_observation(), reward_info, done, info

    def _update_grasp_state(self) -> None:
        """Update object grasp states based on gripper."""
        if self._robot_state is None:
            return

        ee_pos = self._robot_state.ee_position
        gripper = self._robot_state.gripper_state

        for obj_id, obj in self._objects.items():
            if obj["grasped"]:
                # Move object with gripper
                obj["position"] = ee_pos.copy()
                obj["position"][2] -= 0.02  # Offset below gripper

                # Release if gripper opens
                if gripper > 0.8:
                    obj["grasped"] = False
            else:
                # Check for grasp
                dist = np.linalg.norm(ee_pos - obj["position"])
                if dist < 0.05 and gripper < 0.3:
                    obj["grasped"] = True

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        obs = {
            "objects": self._objects,
            "target_position": self._target_position,
            "target_object_id": self._target_object_id,
            "objective": self.objective,
            "step": self._step_count,
        }

        if self._robot_state:
            obs["ee_position"] = self._robot_state.ee_position
            obs["gripper_state"] = self._robot_state.gripper_state

        return obs

    def compute_reward(self) -> RewardInfo:
        """Compute reward."""
        reward_info = RewardInfo()

        if self._robot_state is None:
            return reward_info

        ee_pos = self._robot_state.ee_position

        if self.objective == "reach":
            # Distance to target
            dist = np.linalg.norm(ee_pos - self._target_position)
            reward_info.components["distance"] = -dist
            reward_info.dense = -dist * self.config.reward_scale

        elif self.objective == "grasp":
            target_obj = self._objects.get(self._target_object_id)
            if target_obj:
                if target_obj["grasped"]:
                    reward_info.components["grasp"] = 1.0
                    reward_info.dense = 1.0
                else:
                    dist = np.linalg.norm(ee_pos - target_obj["position"])
                    reward_info.components["distance"] = -dist
                    reward_info.dense = -dist * 0.5

        elif self.objective == "place":
            target_obj = self._objects.get(self._target_object_id)
            if target_obj:
                obj_pos = target_obj["position"]
                place_dist = np.linalg.norm(obj_pos[:2] - self._target_position[:2])

                if target_obj["grasped"]:
                    # Reward for moving toward target
                    to_target = np.linalg.norm(ee_pos - self._target_position)
                    reward_info.components["to_target"] = -to_target
                    reward_info.dense = -to_target * 0.5
                else:
                    # Reward for object being near target
                    reward_info.components["place_distance"] = -place_dist
                    reward_info.dense = -place_dist

        # Smoothness reward
        if self.config.enable_smooth_reward and self._prev_ee_position is not None:
            motion = np.linalg.norm(ee_pos - self._prev_ee_position)
            reward_info.components["smoothness"] = -motion * 0.1

        reward_info.total = reward_info.dense + reward_info.sparse
        return reward_info

    def check_success(self) -> bool:
        """Check if task is complete."""
        if self._robot_state is None:
            return False

        threshold = self.config.success_threshold

        if self.objective == "reach":
            dist = np.linalg.norm(self._robot_state.ee_position - self._target_position)
            return dist < threshold

        elif self.objective == "grasp":
            target_obj = self._objects.get(self._target_object_id)
            return target_obj is not None and target_obj["grasped"]

        elif self.objective == "place":
            target_obj = self._objects.get(self._target_object_id)
            if target_obj and not target_obj["grasped"]:
                dist = np.linalg.norm(target_obj["position"][:2] - self._target_position[:2])
                return dist < threshold
            return False

        elif self.objective == "stack":
            target_obj = self._objects.get(self._target_object_id)
            if target_obj and not target_obj["grasped"]:
                dist = np.linalg.norm(target_obj["position"] - self._target_position)
                return dist < threshold * 2  # Looser threshold for stacking
            return False

        return False


@dataclass
class PickPlaceTaskConfig(ManipulationTaskConfig):
    """Configuration for pick-and-place task."""
    pick_height: float = 0.15
    place_height: float = 0.05
    approach_distance: float = 0.1


class PickPlaceTask(BaseTask):
    """
    Pick and place task environment.

    Phases:
    1. Approach: Move above object
    2. Descend: Lower to grasp height
    3. Grasp: Close gripper
    4. Lift: Raise object
    5. Transport: Move to target
    6. Lower: Lower to place height
    7. Release: Open gripper
    """

    class Phase(Enum):
        APPROACH = "approach"
        DESCEND = "descend"
        GRASP = "grasp"
        LIFT = "lift"
        TRANSPORT = "transport"
        LOWER = "lower"
        RELEASE = "release"
        DONE = "done"

    def __init__(self, config: Optional[PickPlaceTaskConfig] = None):
        """Initialize pick-and-place task."""
        super().__init__(config or PickPlaceTaskConfig())
        self.config: PickPlaceTaskConfig = self.config

        # Object to pick
        self._pick_object: Dict[str, Any] = {
            "position": np.array([0.5, 0.0, 0.02]),
            "orientation": np.array([1, 0, 0, 0]),
            "grasped": False,
            "size": 0.04,
        }

        # Place target
        self._place_target = np.array([0.5, 0.3, 0.02])

        # Current phase
        self._phase = self.Phase.APPROACH

        # Robot state
        self._robot_state = None

        logger.info("PickPlaceTask initialized")

    def set_robot_state(self, robot_state: Any) -> None:
        """Set robot state reference."""
        self._robot_state = robot_state

    def reset(self) -> Dict[str, Any]:
        """Reset task."""
        self._step_count = 0
        self._status = TaskStatus.RUNNING
        self._episode_reward = 0.0
        self._phase = self.Phase.APPROACH

        # Randomize positions
        if self.config.randomize_initial:
            self._pick_object["position"] = np.array([
                np.random.uniform(0.3, 0.6),
                np.random.uniform(-0.2, 0.2),
                0.02,
            ])
            self._pick_object["grasped"] = False

        if self.config.randomize_target:
            self._place_target = np.array([
                np.random.uniform(0.3, 0.6),
                np.random.uniform(-0.2, 0.2),
                0.02,
            ])
            # Ensure target is different from pick position
            while np.linalg.norm(self._place_target[:2] - self._pick_object["position"][:2]) < 0.1:
                self._place_target = np.array([
                    np.random.uniform(0.3, 0.6),
                    np.random.uniform(-0.2, 0.2),
                    0.02,
                ])

        return self.get_observation()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], RewardInfo, bool, Dict]:
        """Execute task step."""
        self._step_count += 1

        # Update phase
        self._update_phase()

        # Update object position if grasped
        if self._pick_object["grasped"] and self._robot_state:
            self._pick_object["position"] = self._robot_state.ee_position.copy()
            self._pick_object["position"][2] -= 0.02

        # Check grasp/release
        self._update_grasp()

        # Compute reward
        reward_info = self.compute_reward()
        self._episode_reward += reward_info.total

        # Check termination
        done = False
        info = {
            "status": self._status.value,
            "phase": self._phase.value,
        }

        if self.check_success():
            self._status = TaskStatus.SUCCESS
            done = True
            info["success"] = True
            reward_info.sparse = 20.0
            reward_info.total += 20.0
        elif self._step_count >= self.config.max_episode_length:
            self._status = TaskStatus.TIMEOUT
            done = True
            info["success"] = False

        return self.get_observation(), reward_info, done, info

    def _update_phase(self) -> None:
        """Update task phase based on state."""
        if self._robot_state is None:
            return

        ee_pos = self._robot_state.ee_position
        gripper = self._robot_state.gripper_state
        obj_pos = self._pick_object["position"]

        approach_pos = obj_pos.copy()
        approach_pos[2] = self.config.pick_height

        if self._phase == self.Phase.APPROACH:
            if np.linalg.norm(ee_pos - approach_pos) < 0.03:
                self._phase = self.Phase.DESCEND

        elif self._phase == self.Phase.DESCEND:
            if ee_pos[2] < obj_pos[2] + 0.03:
                self._phase = self.Phase.GRASP

        elif self._phase == self.Phase.GRASP:
            if self._pick_object["grasped"]:
                self._phase = self.Phase.LIFT

        elif self._phase == self.Phase.LIFT:
            if ee_pos[2] > self.config.pick_height:
                self._phase = self.Phase.TRANSPORT

        elif self._phase == self.Phase.TRANSPORT:
            transport_target = self._place_target.copy()
            transport_target[2] = self.config.pick_height
            if np.linalg.norm(ee_pos[:2] - self._place_target[:2]) < 0.03:
                self._phase = self.Phase.LOWER

        elif self._phase == self.Phase.LOWER:
            if ee_pos[2] < self.config.place_height + 0.03:
                self._phase = self.Phase.RELEASE

        elif self._phase == self.Phase.RELEASE:
            if not self._pick_object["grasped"] and gripper > 0.8:
                self._phase = self.Phase.DONE

    def _update_grasp(self) -> None:
        """Update grasp state."""
        if self._robot_state is None:
            return

        ee_pos = self._robot_state.ee_position
        gripper = self._robot_state.gripper_state
        obj_pos = self._pick_object["position"]

        if self._pick_object["grasped"]:
            if gripper > 0.7:  # Release
                self._pick_object["grasped"] = False
        else:
            dist = np.linalg.norm(ee_pos - obj_pos)
            if dist < 0.05 and gripper < 0.3:  # Grasp
                self._pick_object["grasped"] = True

    def get_observation(self) -> Dict[str, Any]:
        """Get observation."""
        obs = {
            "pick_object": self._pick_object,
            "place_target": self._place_target,
            "phase": self._phase.value,
            "step": self._step_count,
        }

        if self._robot_state:
            obs["ee_position"] = self._robot_state.ee_position
            obs["gripper_state"] = self._robot_state.gripper_state

        return obs

    def compute_reward(self) -> RewardInfo:
        """Compute phase-based reward."""
        reward_info = RewardInfo()

        if self._robot_state is None:
            return reward_info

        ee_pos = self._robot_state.ee_position
        obj_pos = self._pick_object["position"]

        # Phase-specific rewards
        if self._phase == self.Phase.APPROACH:
            approach_pos = obj_pos.copy()
            approach_pos[2] = self.config.pick_height
            dist = np.linalg.norm(ee_pos - approach_pos)
            reward_info.dense = -dist

        elif self._phase == self.Phase.DESCEND:
            target_z = obj_pos[2] + 0.02
            dist = abs(ee_pos[2] - target_z)
            reward_info.dense = -dist

        elif self._phase == self.Phase.GRASP:
            if self._pick_object["grasped"]:
                reward_info.dense = 2.0  # Bonus for grasping

        elif self._phase == self.Phase.LIFT:
            target_z = self.config.pick_height
            dist = abs(ee_pos[2] - target_z)
            reward_info.dense = -dist + (1.0 if self._pick_object["grasped"] else -2.0)

        elif self._phase == self.Phase.TRANSPORT:
            transport_target = self._place_target.copy()
            transport_target[2] = self.config.pick_height
            dist = np.linalg.norm(ee_pos - transport_target)
            reward_info.dense = -dist + (1.0 if self._pick_object["grasped"] else -2.0)

        elif self._phase == self.Phase.LOWER:
            target = self._place_target.copy()
            target[2] = self.config.place_height + 0.02
            dist = np.linalg.norm(ee_pos - target)
            reward_info.dense = -dist

        elif self._phase == self.Phase.RELEASE:
            reward_info.dense = 1.0 if not self._pick_object["grasped"] else 0.0

        elif self._phase == self.Phase.DONE:
            reward_info.dense = 5.0

        reward_info.components["phase"] = float(list(self.Phase).index(self._phase))
        reward_info.total = reward_info.dense * self.config.reward_scale

        return reward_info

    def check_success(self) -> bool:
        """Check if pick-and-place is complete."""
        if self._phase != self.Phase.DONE:
            return False

        if self._pick_object["grasped"]:
            return False

        # Check if object is near place target
        obj_pos = self._pick_object["position"]
        dist = np.linalg.norm(obj_pos[:2] - self._place_target[:2])

        return dist < self.config.success_threshold * 2

    def get_phase_waypoints(self) -> Dict[str, np.ndarray]:
        """Get waypoints for current phase (for visualization)."""
        obj_pos = self._pick_object["position"]

        waypoints = {
            "approach": np.array([obj_pos[0], obj_pos[1], self.config.pick_height]),
            "grasp": obj_pos.copy(),
            "lift": np.array([obj_pos[0], obj_pos[1], self.config.pick_height]),
            "transport": np.array([
                self._place_target[0],
                self._place_target[1],
                self.config.pick_height
            ]),
            "place": self._place_target.copy(),
        }

        return waypoints

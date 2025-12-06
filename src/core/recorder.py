"""
Robot state and demonstration step data structures.

These structures form the "output side" of the retargeting pipeline.
The Episode class collects DemoSteps and provides serialization.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path
import json


@dataclass
class RobotObs:
    """
    Robot observation at a single timestep.
    
    This captures the robot's proprioceptive state. For IL, we typically need:
    - Joint positions (always)
    - Joint velocities (for dynamics-aware policies)
    - End-effector pose (redundant but useful for task-space policies)
    - Gripper state (position and/or force)
    """
    timestamp: float
    
    # Joint state
    joint_positions: np.ndarray             # [n_joints] in radians
    joint_velocities: np.ndarray            # [n_joints] in rad/s
    joint_torques: Optional[np.ndarray] = None  # [n_joints] in Nm (if available)
    
    # End-effector state (derived from FK, but cached for convenience)
    ee_pose: np.ndarray = None              # [4, 4] transform in world/base frame
    ee_velocity: Optional[np.ndarray] = None  # [6] twist (linear, angular)
    
    # Gripper state
    gripper_position: float = 0.0           # Normalized [0, 1]: 0=open, 1=closed
    gripper_velocity: float = 0.0           # Opening/closing speed
    gripper_force: Optional[float] = None   # Gripping force in N (if available)
    
    @property
    def n_joints(self) -> int:
        return len(self.joint_positions)
    
    def to_vector(self, include_gripper: bool = True) -> np.ndarray:
        """
        Flatten observation to a vector for neural network input.
        
        Default layout: [joint_pos, joint_vel, gripper_pos]
        """
        components = [self.joint_positions, self.joint_velocities]
        if include_gripper:
            components.append(np.array([self.gripper_position]))
        return np.concatenate(components)
    
    @classmethod
    def vector_dim(cls, n_joints: int, include_gripper: bool = True) -> int:
        """Get the dimension of the flattened observation vector."""
        dim = n_joints * 2  # positions + velocities
        if include_gripper:
            dim += 1
        return dim


@dataclass
class RobotAction:
    """
    Robot action command at a single timestep.
    
    Actions can be specified in different spaces:
    - Joint space: target joint positions or velocities
    - Task space: target end-effector pose
    
    For ACT-style IL, we typically use joint position targets because:
    1. They're unambiguous (no IK needed at execution time)
    2. They compose well across different robot morphologies when normalized
    3. They capture the full arm configuration, not just EE pose
    """
    timestamp: float
    
    # Joint-space action
    joint_position_target: np.ndarray       # [n_joints] target positions in radians
    
    # Gripper action
    gripper_target: float                   # [0, 1]: 0=open, 1=closed
    
    # Optional: task-space specification (for debugging/visualization)
    ee_pose_target: Optional[np.ndarray] = None  # [4, 4] desired EE pose
    
    # Action metadata
    action_type: str = 'joint_position'     # 'joint_position', 'joint_velocity', 'ee_pose'
    
    @property
    def n_joints(self) -> int:
        return len(self.joint_position_target)
    
    def to_vector(self, include_gripper: bool = True) -> np.ndarray:
        """
        Flatten action to a vector for neural network output.
        
        Default layout: [joint_targets, gripper_target]
        """
        if include_gripper:
            return np.concatenate([self.joint_position_target, [self.gripper_target]])
        return self.joint_position_target
    
    @classmethod
    def vector_dim(cls, n_joints: int, include_gripper: bool = True) -> int:
        """Get the dimension of the flattened action vector."""
        dim = n_joints
        if include_gripper:
            dim += 1
        return dim


@dataclass
class DemoStep:
    """
    A single timestep of a demonstration.
    
    This is the fundamental unit of data collection. Each step pairs:
    - What the robot observed (RobotObs)
    - What action was commanded (RobotAction, from retargeting)
    - Metadata for organizing and filtering demos
    
    The step also retains the original HumanState for debugging,
    but this is NOT uploaded to the cloud (privacy).
    """
    step_index: int                         # Timestep within episode
    obs: RobotObs
    action: RobotAction
    
    # Metadata (included in cloud upload)
    episode_id: str
    task_id: str                            # What task is being demonstrated
    env_id: str                             # Which physical environment/site
    
    # Human state (kept locally for debugging, not uploaded)
    human_state: Optional[Any] = None       # HumanState, but typed as Any to avoid circular import
    
    # Quality flags (set during recording or post-processing)
    is_valid: bool = True                   # False if retargeting failed, IK unreachable, etc.
    quality_score: float = 1.0              # Heuristic quality (0-1), based on smoothness, reachability
    
    def to_dict(self, include_human: bool = False) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            'step_index': self.step_index,
            'timestamp': self.obs.timestamp,
            'obs': {
                'joint_positions': self.obs.joint_positions.tolist(),
                'joint_velocities': self.obs.joint_velocities.tolist(),
                'gripper_position': self.obs.gripper_position,
            },
            'action': {
                'joint_position_target': self.action.joint_position_target.tolist(),
                'gripper_target': self.action.gripper_target,
            },
            'meta': {
                'episode_id': self.episode_id,
                'task_id': self.task_id,
                'env_id': self.env_id,
                'is_valid': self.is_valid,
                'quality_score': self.quality_score,
            }
        }
        # Optionally include EE pose for visualization
        if self.action.ee_pose_target is not None:
            d['action']['ee_pose_target'] = self.action.ee_pose_target.tolist()
        
        return d


@dataclass
class Episode:
    """
    A complete demonstration episode.
    
    An episode is a sequence of DemoSteps representing one task execution.
    Episodes are the unit of storage and chunking.
    """
    episode_id: str
    task_id: str
    env_id: str
    
    steps: List[DemoStep] = field(default_factory=list)
    
    # Episode-level metadata
    robot_type: str = ''                    # URDF identifier or robot name
    demonstrator_id: str = ''               # Anonymized demonstrator identifier
    success: Optional[bool] = None          # Did the task succeed? (labeled post-hoc)
    
    # Recording info
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def append_step(self, step: DemoStep):
        """Add a step to the episode."""
        if not self.steps:
            self.start_time = step.obs.timestamp
        self.end_time = step.obs.timestamp
        self.steps.append(step)
    
    @property
    def duration(self) -> float:
        """Episode duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def n_valid_steps(self) -> int:
        """Count of valid (non-failed) steps."""
        return sum(1 for s in self.steps if s.is_valid)
    
    def get_obs_array(self, include_gripper: bool = True) -> np.ndarray:
        """
        Stack all observations into a [T, d_obs] array.
        Only includes valid steps.
        """
        valid_steps = [s for s in self.steps if s.is_valid]
        if not valid_steps:
            return np.array([])
        
        return np.stack([s.obs.to_vector(include_gripper) for s in valid_steps])
    
    def get_action_array(self, include_gripper: bool = True) -> np.ndarray:
        """
        Stack all actions into a [T, d_act] array.
        Only includes valid steps.
        """
        valid_steps = [s for s in self.steps if s.is_valid]
        if not valid_steps:
            return np.array([])
        
        return np.stack([s.action.to_vector(include_gripper) for s in valid_steps])
    
    def save(self, path: Path):
        """
        Save episode to disk.
        
        Format: .npz with arrays for obs, actions, and JSON metadata.
        This format is efficient for loading into PyTorch dataloaders.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        obs_array = self.get_obs_array()
        action_array = self.get_action_array()
        
        # Timestamps for valid steps
        valid_steps = [s for s in self.steps if s.is_valid]
        timestamps = np.array([s.obs.timestamp for s in valid_steps])
        
        # Metadata as JSON string
        meta = {
            'episode_id': self.episode_id,
            'task_id': self.task_id,
            'env_id': self.env_id,
            'robot_type': self.robot_type,
            'demonstrator_id': self.demonstrator_id,
            'success': self.success,
            'n_steps': len(valid_steps),
            'duration': self.duration,
        }
        
        np.savez_compressed(
            path,
            obs=obs_array,
            actions=action_array,
            timestamps=timestamps,
            meta=json.dumps(meta)
        )
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> 'Episode':
        """Load episode from disk."""
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data['meta']))
        
        episode = cls(
            episode_id=meta['episode_id'],
            task_id=meta['task_id'],
            env_id=meta['env_id'],
            robot_type=meta.get('robot_type', ''),
            demonstrator_id=meta.get('demonstrator_id', ''),
            success=meta.get('success'),
        )
        
        # Note: This reconstructs minimal DemoSteps without full robot state
        # For full reconstruction, you'd need to store more data
        obs_array = data['obs']
        action_array = data['actions']
        timestamps = data['timestamps']
        
        # Infer dimensions (assuming 7-DOF arm + gripper)
        n_steps = len(timestamps)
        if n_steps > 0:
            obs_dim = obs_array.shape[1]
            act_dim = action_array.shape[1]
            
            # Assume layout: obs = [joint_pos, joint_vel, gripper], act = [joint_target, gripper]
            n_joints = (obs_dim - 1) // 2
            
            for i in range(n_steps):
                obs = RobotObs(
                    timestamp=timestamps[i],
                    joint_positions=obs_array[i, :n_joints],
                    joint_velocities=obs_array[i, n_joints:2*n_joints],
                    gripper_position=obs_array[i, -1],
                )
                action = RobotAction(
                    timestamp=timestamps[i],
                    joint_position_target=action_array[i, :-1],
                    gripper_target=action_array[i, -1],
                )
                step = DemoStep(
                    step_index=i,
                    obs=obs,
                    action=action,
                    episode_id=episode.episode_id,
                    task_id=episode.task_id,
                    env_id=episode.env_id,
                )
                episode.steps.append(step)
        
        return episode

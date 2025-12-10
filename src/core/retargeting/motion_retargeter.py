"""
Motion Retargeting Implementation

Provides human-to-robot motion retargeting using:
- Joint position matching
- End-effector position matching (IK)
- Velocity-based retargeting

This replaces MockGMRRetargeter with real functionality.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .kinematics import RobotKinematics
from .ik_solver import IKSolver, IKSolverConfig, IKResult

logger = logging.getLogger(__name__)


class RetargetingMethod(Enum):
    """Retargeting method."""
    JOINT_POSITION = "joint"  # Direct joint mapping
    END_EFFECTOR = "ee"  # IK-based end-effector matching
    HYBRID = "hybrid"  # Combination of both


@dataclass
class RetargetingConfig:
    """Configuration for motion retargeting."""
    method: RetargetingMethod = RetargetingMethod.HYBRID

    # Joint mapping parameters
    joint_scale: np.ndarray = field(default_factory=lambda: np.ones(7))
    joint_offset: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # End-effector matching parameters
    ee_weight: float = 0.8
    joint_weight: float = 0.2

    # Filtering parameters
    smoothing_alpha: float = 0.3  # Exponential smoothing
    velocity_limit: float = 2.0  # rad/s per joint

    # Robot configuration
    robot_type: str = "generic_7dof"
    urdf_path: Optional[str] = None


@dataclass
class RetargetingResult:
    """Result of motion retargeting."""
    q: np.ndarray  # Robot joint configuration
    success: bool
    method_used: RetargetingMethod
    ee_error: float = 0.0
    joint_error: float = 0.0


class MotionRetargeter:
    """
    Human-to-Robot Motion Retargeter.

    This class provides real motion retargeting to replace MockGMRRetargeter.
    It supports multiple retargeting strategies:

    1. Joint Position: Direct mapping of human joint angles to robot
    2. End-Effector: Use IK to match hand positions
    3. Hybrid: Combine both methods

    Usage:
        retargeter = MotionRetargeter(
            urdf_path="robot.urdf",
            config=RetargetingConfig(method=RetargetingMethod.HYBRID)
        )

        # In teleoperation loop:
        result = retargeter.retarget(human_pose_3d)
        robot_q = result.q
    """

    # Human skeleton joint indices (COCO-WholeBody format)
    HUMAN_JOINTS = {
        'pelvis': 0,
        'right_hip': 12,
        'right_knee': 14,
        'right_ankle': 16,
        'left_hip': 11,
        'left_knee': 13,
        'left_ankle': 15,
        'spine': 0,  # Approximated
        'chest': 5,  # Approximated from shoulders
        'neck': 0,  # Approximated
        'head': 0,
        'left_shoulder': 5,
        'left_elbow': 7,
        'left_wrist': 9,
        'right_shoulder': 6,
        'right_elbow': 8,
        'right_wrist': 10,
    }

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        robot_type: str = "generic_7dof",
        config: Optional[RetargetingConfig] = None,
        n_joints: int = 7
    ):
        self.config = config or RetargetingConfig()
        self.n_joints = n_joints

        # Initialize kinematics
        self.kinematics = RobotKinematics(
            urdf_path=urdf_path or self.config.urdf_path,
            robot_type=robot_type or self.config.robot_type,
        )
        self.n_joints = self.kinematics.n_joints

        # Initialize IK solver
        self.ik_solver = IKSolver(
            self.kinematics,
            IKSolverConfig(max_iterations=50, tolerance=1e-3)
        )

        # State for filtering
        self.prev_q: Optional[np.ndarray] = None
        self.prev_time: float = 0.0

        # Arm calibration (human -> robot scale)
        self.human_arm_length: float = 0.6  # meters
        self.robot_arm_length: float = 0.8  # Will be computed from FK

        self._compute_robot_arm_length()

    def _compute_robot_arm_length(self):
        """Compute robot arm length from FK at zero config."""
        q_zero = np.zeros(self.n_joints)
        T_ee = self.kinematics.forward_kinematics(q_zero)
        self.robot_arm_length = np.linalg.norm(T_ee[:3, 3])

    def retarget(
        self,
        human_pose_3d: np.ndarray,
        timestamp: Optional[float] = None
    ) -> RetargetingResult:
        """
        Retarget human pose to robot configuration.

        Args:
            human_pose_3d: Human 3D pose [K, 3] in world frame
            timestamp: Current timestamp for velocity limiting

        Returns:
            RetargetingResult with robot joint configuration
        """
        method = self.config.method

        if method == RetargetingMethod.JOINT_POSITION:
            return self._retarget_joint_position(human_pose_3d)
        elif method == RetargetingMethod.END_EFFECTOR:
            return self._retarget_end_effector(human_pose_3d)
        else:
            return self._retarget_hybrid(human_pose_3d)

    def _retarget_joint_position(
        self,
        human_pose_3d: np.ndarray
    ) -> RetargetingResult:
        """Retarget using direct joint mapping."""
        q = np.zeros(self.n_joints)

        if len(human_pose_3d) < 17:
            return RetargetingResult(
                q=q,
                success=False,
                method_used=RetargetingMethod.JOINT_POSITION,
            )

        # Extract arm joint positions
        shoulder = human_pose_3d[6]  # Right shoulder
        elbow = human_pose_3d[8]  # Right elbow
        wrist = human_pose_3d[10]  # Right wrist

        # Compute joint angles from limb directions
        # Shoulder joint 1: rotation around vertical axis
        upper_arm = elbow - shoulder
        upper_arm_norm = upper_arm / (np.linalg.norm(upper_arm) + 1e-6)

        q[0] = np.arctan2(upper_arm_norm[1], upper_arm_norm[0])

        # Shoulder joint 2: elevation
        q[1] = np.arcsin(np.clip(-upper_arm_norm[2], -1, 1))

        # Shoulder joint 3: rotation (simplified)
        forearm = wrist - elbow
        forearm_norm = forearm / (np.linalg.norm(forearm) + 1e-6)

        # Cross product gives rotation axis
        cross = np.cross(upper_arm_norm, forearm_norm)
        q[2] = np.arctan2(cross[2], np.dot(upper_arm_norm, forearm_norm))

        # Elbow angle
        elbow_angle = np.arccos(np.clip(
            np.dot(upper_arm_norm, forearm_norm), -1, 1
        ))
        q[3] = np.pi - elbow_angle  # Elbow flexion

        # Wrist angles (simplified, would need hand orientation)
        q[4] = 0.0
        q[5] = 0.0
        q[6] = 0.0

        # Apply scaling and offset
        q = q * self.config.joint_scale[:len(q)] + self.config.joint_offset[:len(q)]

        # Apply smoothing
        q = self._apply_smoothing(q)

        # Clamp to limits
        if self.kinematics.joint_limits:
            q = self.kinematics.joint_limits.clamp(q)

        return RetargetingResult(
            q=q,
            success=True,
            method_used=RetargetingMethod.JOINT_POSITION,
        )

    def _retarget_end_effector(
        self,
        human_pose_3d: np.ndarray
    ) -> RetargetingResult:
        """Retarget using IK to match end-effector position."""
        if len(human_pose_3d) < 17:
            return RetargetingResult(
                q=np.zeros(self.n_joints),
                success=False,
                method_used=RetargetingMethod.END_EFFECTOR,
            )

        # Get hand position in human frame
        shoulder = human_pose_3d[6]
        wrist = human_pose_3d[10]

        # Compute relative position
        hand_rel = wrist - shoulder

        # Scale to robot workspace
        scale = self.robot_arm_length / (self.human_arm_length + 1e-6)
        hand_robot = hand_rel * scale

        # Create target transform
        T_target = np.eye(4)
        T_target[:3, 3] = hand_robot + np.array([0, 0, 0.3])  # Add base offset

        # Solve IK
        q_init = self.prev_q if self.prev_q is not None else np.zeros(self.n_joints)
        ik_result = self.ik_solver.solve(T_target, q_init, position_only=True)

        q = ik_result.q

        # Apply smoothing
        q = self._apply_smoothing(q)

        return RetargetingResult(
            q=q,
            success=ik_result.success,
            method_used=RetargetingMethod.END_EFFECTOR,
            ee_error=ik_result.position_error,
        )

    def _retarget_hybrid(
        self,
        human_pose_3d: np.ndarray
    ) -> RetargetingResult:
        """Hybrid retargeting combining joint and end-effector methods."""
        # Get results from both methods
        joint_result = self._retarget_joint_position(human_pose_3d)
        ee_result = self._retarget_end_effector(human_pose_3d)

        if not joint_result.success and not ee_result.success:
            return RetargetingResult(
                q=np.zeros(self.n_joints),
                success=False,
                method_used=RetargetingMethod.HYBRID,
            )

        # Weighted combination
        w_ee = self.config.ee_weight
        w_joint = self.config.joint_weight

        if not ee_result.success:
            q = joint_result.q
        elif not joint_result.success:
            q = ee_result.q
        else:
            # Blend based on weights
            total_weight = w_ee + w_joint
            q = (w_ee * ee_result.q + w_joint * joint_result.q) / total_weight

        # Clamp to limits
        if self.kinematics.joint_limits:
            q = self.kinematics.joint_limits.clamp(q)

        return RetargetingResult(
            q=q,
            success=True,
            method_used=RetargetingMethod.HYBRID,
            ee_error=ee_result.ee_error,
        )

    def _apply_smoothing(self, q: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to joint trajectory."""
        if self.prev_q is None:
            self.prev_q = q.copy()
            return q

        alpha = self.config.smoothing_alpha
        q_smooth = alpha * q + (1 - alpha) * self.prev_q
        self.prev_q = q_smooth.copy()

        return q_smooth

    def reset(self):
        """Reset retargeter state."""
        self.prev_q = None
        self.prev_time = 0.0


class WholeBodyRetargeter(MotionRetargeter):
    """
    Extended retargeter for whole-body motion.

    Handles:
    - Both arms
    - Head tracking
    - Torso orientation
    """

    def __init__(
        self,
        left_arm_kinematics: Optional[RobotKinematics] = None,
        right_arm_kinematics: Optional[RobotKinematics] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Separate kinematics for left/right arms
        self.left_arm_kinematics = left_arm_kinematics
        self.right_arm_kinematics = right_arm_kinematics or self.kinematics

    def retarget_wholebody(
        self,
        human_pose_3d: np.ndarray,
        include_head: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Retarget whole-body pose.

        Returns:
            Dict with 'left_arm', 'right_arm', 'head', 'torso' joint angles
        """
        result = {}

        # Right arm (main arm)
        if len(human_pose_3d) >= 11:
            right_result = self.retarget(human_pose_3d)
            result['right_arm'] = right_result.q

        # Left arm (mirror)
        if len(human_pose_3d) >= 10 and self.left_arm_kinematics:
            # Get left arm keypoints
            left_shoulder = human_pose_3d[5]
            left_elbow = human_pose_3d[7]
            left_wrist = human_pose_3d[9]

            # Create mirrored pose for left arm
            left_pose = human_pose_3d.copy()
            left_pose[6] = left_shoulder  # Swap to right arm indices
            left_pose[8] = left_elbow
            left_pose[10] = left_wrist

            left_result = self._retarget_joint_position(left_pose)
            result['left_arm'] = left_result.q

        # Head orientation (simplified)
        if include_head and len(human_pose_3d) >= 5:
            left_eye = human_pose_3d[1]
            right_eye = human_pose_3d[2]
            nose = human_pose_3d[0]

            # Compute head orientation
            eye_center = (left_eye + right_eye) / 2
            forward = nose - eye_center
            forward = forward / (np.linalg.norm(forward) + 1e-6)

            yaw = np.arctan2(forward[0], forward[1])
            pitch = np.arcsin(np.clip(-forward[2], -1, 1))

            result['head'] = np.array([yaw, pitch])

        return result

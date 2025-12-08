"""
Robot Kinematics Implementation

Provides forward kinematics and Jacobian computation using:
- Pinocchio (preferred, fast C++ library)
- urchin/yourdfpy (fallback, pure Python)
- NumPy-based analytical solutions (for specific robots)

This replaces the placeholder FK in retargeting.py
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import kinematics libraries
PINOCCHIO_AVAILABLE = False
URCHIN_AVAILABLE = False

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
    logger.info("Pinocchio available for kinematics")
except ImportError:
    pass

try:
    import urchin as ur
    URCHIN_AVAILABLE = True
    logger.info("urchin available for kinematics")
except ImportError:
    pass


@dataclass
class JointLimits:
    """Joint limits for a robot."""
    lower: np.ndarray
    upper: np.ndarray
    velocity: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits."""
        return np.clip(q, self.lower, self.upper)

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if joint configuration is within limits."""
        return np.all(q >= self.lower) and np.all(q <= self.upper)


@dataclass
class KinematicsConfig:
    """Configuration for robot kinematics."""
    urdf_path: str = ""
    base_link: str = "base_link"
    ee_link: str = "tool0"
    n_joints: int = 7
    dh_params: Optional[np.ndarray] = None  # For analytical FK


# Standard robot DH parameters (for common robots)
ROBOT_DH_PARAMS = {
    # Franka Emika Panda (modified DH)
    "panda": {
        "d": [0.333, 0, 0.316, 0, 0.384, 0, 0],
        "a": [0, 0, 0, 0.0825, -0.0825, 0, 0.088],
        "alpha": [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2],
        "theta_offset": [0, 0, 0, 0, 0, 0, 0],
        "joint_limits": {
            "lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            "upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
        }
    },
    # Universal Robots UR5e
    "ur5e": {
        "d": [0.1625, 0, 0, 0.1333, 0.0997, 0.0996],
        "a": [0, -0.425, -0.3922, 0, 0, 0],
        "alpha": [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
        "theta_offset": [0, 0, 0, 0, 0, 0],
        "joint_limits": {
            "lower": np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            "upper": np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
        }
    },
    # Generic 7-DOF arm
    "generic_7dof": {
        "d": [0.2, 0, 0.3, 0, 0.3, 0, 0.1],
        "a": [0, 0, 0, 0, 0, 0, 0],
        "alpha": [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0],
        "theta_offset": [0, 0, 0, 0, 0, 0, 0],
        "joint_limits": {
            "lower": np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]),
            "upper": np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]),
        }
    },
}


class RobotKinematics:
    """
    Robot kinematics using Pinocchio or fallback to DH parameters.

    This class provides:
    - Forward kinematics (FK): joint angles -> end-effector pose
    - Jacobian computation for IK
    - Joint limit enforcement

    Usage:
        # With URDF
        kinematics = RobotKinematics(urdf_path="robot.urdf")

        # With DH parameters
        kinematics = RobotKinematics(robot_type="panda")

        T_ee = kinematics.forward_kinematics(q)
        J = kinematics.jacobian(q)
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        robot_type: str = "generic_7dof",
        config: Optional[KinematicsConfig] = None,
    ):
        self.config = config or KinematicsConfig()

        if urdf_path:
            self.config.urdf_path = urdf_path

        self.n_joints = self.config.n_joints
        self._pinocchio_model = None
        self._pinocchio_data = None
        self._urchin_robot = None
        self._dh_params = None
        self.joint_limits = None

        # Try to load URDF with available libraries
        if self.config.urdf_path and Path(self.config.urdf_path).exists():
            self._load_urdf(self.config.urdf_path)
        else:
            # Use DH parameters for specified robot
            self._load_dh_params(robot_type)

    def _load_urdf(self, urdf_path: str):
        """Load URDF using available library."""
        if PINOCCHIO_AVAILABLE:
            try:
                self._pinocchio_model = pin.buildModelFromUrdf(urdf_path)
                self._pinocchio_data = self._pinocchio_model.createData()
                self.n_joints = self._pinocchio_model.nq

                # Extract joint limits
                self.joint_limits = JointLimits(
                    lower=self._pinocchio_model.lowerPositionLimit,
                    upper=self._pinocchio_model.upperPositionLimit,
                    velocity=self._pinocchio_model.velocityLimit,
                    effort=self._pinocchio_model.effortLimit,
                )

                logger.info(f"Loaded URDF with Pinocchio: {self.n_joints} joints")
                return

            except Exception as e:
                logger.warning(f"Pinocchio failed to load URDF: {e}")

        if URCHIN_AVAILABLE:
            try:
                self._urchin_robot = ur.URDF.load(urdf_path)
                self.n_joints = len(self._urchin_robot.actuated_joints)

                # Extract joint limits
                lower = []
                upper = []
                for joint in self._urchin_robot.actuated_joints:
                    if joint.limit:
                        lower.append(joint.limit.lower)
                        upper.append(joint.limit.upper)
                    else:
                        lower.append(-np.pi)
                        upper.append(np.pi)

                self.joint_limits = JointLimits(
                    lower=np.array(lower),
                    upper=np.array(upper),
                )

                logger.info(f"Loaded URDF with urchin: {self.n_joints} joints")
                return

            except Exception as e:
                logger.warning(f"urchin failed to load URDF: {e}")

        logger.warning("No URDF loader available, using DH fallback")
        self._load_dh_params("generic_7dof")

    def _load_dh_params(self, robot_type: str):
        """Load DH parameters for known robot types."""
        if robot_type not in ROBOT_DH_PARAMS:
            robot_type = "generic_7dof"

        params = ROBOT_DH_PARAMS[robot_type]
        self._dh_params = {
            "d": np.array(params["d"]),
            "a": np.array(params["a"]),
            "alpha": np.array(params["alpha"]),
            "theta_offset": np.array(params["theta_offset"]),
        }

        self.n_joints = len(params["d"])
        self.joint_limits = JointLimits(
            lower=params["joint_limits"]["lower"],
            upper=params["joint_limits"]["upper"],
        )

        logger.info(f"Using DH parameters for {robot_type}: {self.n_joints} joints")

    def forward_kinematics(
        self,
        q: np.ndarray,
        frame: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            q: Joint configuration [n_joints]
            frame: Target frame name (None = end-effector)

        Returns:
            T: 4x4 homogeneous transform
        """
        q = np.asarray(q).flatten()

        if len(q) < self.n_joints:
            q = np.pad(q, (0, self.n_joints - len(q)))
        elif len(q) > self.n_joints:
            q = q[:self.n_joints]

        # Pinocchio FK
        if self._pinocchio_model is not None:
            pin.forwardKinematics(self._pinocchio_model, self._pinocchio_data, q)
            pin.updateFramePlacements(self._pinocchio_model, self._pinocchio_data)

            if frame:
                frame_id = self._pinocchio_model.getFrameId(frame)
            else:
                frame_id = self._pinocchio_model.nframes - 1

            T = self._pinocchio_data.oMf[frame_id].homogeneous
            return T

        # urchin FK
        if self._urchin_robot is not None:
            cfg = {}
            for i, joint in enumerate(self._urchin_robot.actuated_joints):
                if i < len(q):
                    cfg[joint.name] = q[i]

            fk = self._urchin_robot.link_fk(cfg)

            if frame and frame in fk:
                return fk[frame]

            # Return last link transform
            last_link = list(fk.keys())[-1]
            return fk[last_link]

        # DH-based FK
        return self._dh_forward_kinematics(q)

    def _dh_forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute FK using DH parameters."""
        if self._dh_params is None:
            return np.eye(4)

        d = self._dh_params["d"]
        a = self._dh_params["a"]
        alpha = self._dh_params["alpha"]
        theta_offset = self._dh_params["theta_offset"]

        T = np.eye(4)

        for i in range(min(len(q), len(d))):
            theta = q[i] + theta_offset[i]

            # DH transformation matrix
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha[i]), np.sin(alpha[i])

            Ti = np.array([
                [ct, -st * ca, st * sa, a[i] * ct],
                [st, ct * ca, -ct * sa, a[i] * st],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])

            T = T @ Ti

        return T

    def jacobian(
        self,
        q: np.ndarray,
        frame: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute the geometric Jacobian.

        Args:
            q: Joint configuration [n_joints]
            frame: Target frame (None = end-effector)

        Returns:
            J: 6 x n_joints Jacobian matrix
        """
        q = np.asarray(q).flatten()

        if len(q) < self.n_joints:
            q = np.pad(q, (0, self.n_joints - len(q)))
        elif len(q) > self.n_joints:
            q = q[:self.n_joints]

        # Pinocchio Jacobian
        if self._pinocchio_model is not None:
            pin.computeJointJacobians(self._pinocchio_model, self._pinocchio_data, q)

            if frame:
                frame_id = self._pinocchio_model.getFrameId(frame)
            else:
                frame_id = self._pinocchio_model.nframes - 1

            J = pin.getFrameJacobian(
                self._pinocchio_model,
                self._pinocchio_data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            return J

        # Numerical Jacobian (fallback)
        return self._numerical_jacobian(q)

    def _numerical_jacobian(
        self,
        q: np.ndarray,
        delta: float = 1e-6
    ) -> np.ndarray:
        """Compute numerical Jacobian."""
        n = len(q)
        J = np.zeros((6, n))

        T_0 = self.forward_kinematics(q)
        pos_0 = T_0[:3, 3]
        R_0 = T_0[:3, :3]

        for i in range(n):
            q_p = q.copy()
            q_p[i] += delta

            T_p = self.forward_kinematics(q_p)
            pos_p = T_p[:3, 3]
            R_p = T_p[:3, :3]

            # Position derivative
            J[:3, i] = (pos_p - pos_0) / delta

            # Orientation derivative (using rotation vector)
            R_diff = R_p @ R_0.T
            trace = np.trace(R_diff)
            theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))

            if abs(theta) < 1e-6:
                omega = np.zeros(3)
            else:
                omega = theta / (2 * np.sin(theta)) * np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ])

            J[3:, i] = omega / delta

        return J

    def inverse_kinematics(
        self,
        T_target: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve inverse kinematics using damped least squares.

        Args:
            T_target: Target end-effector pose (4x4)
            q_init: Initial joint configuration
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            q: Joint configuration
            success: Whether IK converged
            error: Final position error
        """
        if q_init is None:
            q = np.zeros(self.n_joints)
        else:
            q = np.asarray(q_init).flatten().copy()

        damping = 0.01
        alpha = 0.5  # Step size

        for iteration in range(max_iterations):
            T_current = self.forward_kinematics(q)

            # Position error
            pos_err = T_target[:3, 3] - T_current[:3, 3]

            # Orientation error
            R_target = T_target[:3, :3]
            R_current = T_current[:3, :3]
            R_err = R_target @ R_current.T

            # Convert to axis-angle
            trace = np.trace(R_err)
            theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))

            if abs(theta) < 1e-6:
                rot_err = np.zeros(3)
            else:
                rot_err = theta / (2 * np.sin(theta)) * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1]
                ])

            # Combined error
            e = np.concatenate([pos_err, rot_err])
            error = np.linalg.norm(pos_err)

            if error < tolerance:
                return q, True, error

            # Compute Jacobian and update
            J = self.jacobian(q)

            # Damped least squares: dq = J^T (J J^T + λ²I)^-1 e
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), e)

            # Update with step size
            q = q + alpha * dq

            # Clamp to limits
            if self.joint_limits:
                q = self.joint_limits.clamp(q)

        return q, False, error

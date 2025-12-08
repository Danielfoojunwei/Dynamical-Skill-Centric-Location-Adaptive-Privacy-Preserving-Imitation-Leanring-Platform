"""
Inverse Kinematics Solver

Provides production-ready IK using:
- Damped Least Squares (DLS)
- Jacobian Transpose
- Null-space optimization

This replaces the placeholder IK in retargeting.py
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from .kinematics import RobotKinematics, JointLimits

logger = logging.getLogger(__name__)


class IKMethod(Enum):
    """IK solution method."""
    DAMPED_LEAST_SQUARES = "dls"
    JACOBIAN_TRANSPOSE = "jt"
    PSEUDOINVERSE = "pinv"


@dataclass
class IKSolverConfig:
    """Configuration for IK solver."""
    method: IKMethod = IKMethod.DAMPED_LEAST_SQUARES
    max_iterations: int = 100
    tolerance: float = 1e-4
    damping: float = 0.01
    step_size: float = 0.5
    position_weight: float = 1.0
    orientation_weight: float = 0.5
    joint_weight: Optional[np.ndarray] = None


@dataclass
class IKResult:
    """IK solution result."""
    q: np.ndarray
    success: bool
    position_error: float
    orientation_error: float
    iterations: int
    message: str = ""


class IKSolver:
    """
    Inverse Kinematics Solver.

    Provides multiple IK methods:
    - Damped Least Squares (DLS): Robust, handles singularities
    - Jacobian Transpose: Fast but may not converge
    - Pseudoinverse: Fast, may be unstable near singularities

    Usage:
        kinematics = RobotKinematics(urdf_path="robot.urdf")
        solver = IKSolver(kinematics)

        result = solver.solve(T_target, q_init)
        if result.success:
            q_solution = result.q
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        config: Optional[IKSolverConfig] = None
    ):
        self.kinematics = kinematics
        self.config = config or IKSolverConfig()
        self.n_joints = kinematics.n_joints

    def solve(
        self,
        T_target: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        position_only: bool = False
    ) -> IKResult:
        """
        Solve inverse kinematics for target pose.

        Args:
            T_target: Target end-effector pose (4x4)
            q_init: Initial joint configuration
            position_only: Only match position (ignore orientation)

        Returns:
            IKResult with solution and convergence info
        """
        if q_init is None:
            q = np.zeros(self.n_joints)
        else:
            q = np.asarray(q_init).flatten().copy()

        if len(q) != self.n_joints:
            q = np.zeros(self.n_joints)
            q[:min(len(q_init), self.n_joints)] = q_init[:self.n_joints]

        method = self.config.method

        if method == IKMethod.DAMPED_LEAST_SQUARES:
            return self._solve_dls(T_target, q, position_only)
        elif method == IKMethod.JACOBIAN_TRANSPOSE:
            return self._solve_jt(T_target, q, position_only)
        else:
            return self._solve_pinv(T_target, q, position_only)

    def _compute_error(
        self,
        T_current: np.ndarray,
        T_target: np.ndarray,
        position_only: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Compute task-space error."""
        # Position error
        pos_err = T_target[:3, 3] - T_current[:3, 3]
        pos_error_norm = np.linalg.norm(pos_err)

        if position_only:
            return pos_err, pos_error_norm, 0.0

        # Orientation error using rotation vector
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

        rot_error_norm = np.linalg.norm(rot_err)

        # Weighted error
        w_pos = self.config.position_weight
        w_rot = self.config.orientation_weight
        e = np.concatenate([w_pos * pos_err, w_rot * rot_err])

        return e, pos_error_norm, rot_error_norm

    def _solve_dls(
        self,
        T_target: np.ndarray,
        q: np.ndarray,
        position_only: bool
    ) -> IKResult:
        """Solve using Damped Least Squares."""
        damping = self.config.damping
        alpha = self.config.step_size

        for iteration in range(self.config.max_iterations):
            T_current = self.kinematics.forward_kinematics(q)
            e, pos_err, rot_err = self._compute_error(T_current, T_target, position_only)

            if pos_err < self.config.tolerance:
                return IKResult(
                    q=q,
                    success=True,
                    position_error=pos_err,
                    orientation_error=rot_err,
                    iterations=iteration + 1,
                    message="Converged"
                )

            # Get Jacobian
            J = self.kinematics.jacobian(q)
            if position_only:
                J = J[:3, :]
                e = e[:3]

            # Damped least squares
            # dq = J^T (J J^T + λ² I)^-1 e
            n_task = J.shape[0]
            JJT = J @ J.T + damping**2 * np.eye(n_task)

            try:
                dq = J.T @ np.linalg.solve(JJT, e)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.lstsq(JJT, e, rcond=None)[0]

            # Update
            q = q + alpha * dq

            # Clamp to limits
            if self.kinematics.joint_limits:
                q = self.kinematics.joint_limits.clamp(q)

        # Final check
        T_final = self.kinematics.forward_kinematics(q)
        _, pos_err, rot_err = self._compute_error(T_final, T_target, position_only)

        return IKResult(
            q=q,
            success=pos_err < self.config.tolerance * 10,
            position_error=pos_err,
            orientation_error=rot_err,
            iterations=self.config.max_iterations,
            message="Max iterations reached"
        )

    def _solve_jt(
        self,
        T_target: np.ndarray,
        q: np.ndarray,
        position_only: bool
    ) -> IKResult:
        """Solve using Jacobian Transpose."""
        alpha = self.config.step_size * 0.1  # JT needs smaller steps

        for iteration in range(self.config.max_iterations):
            T_current = self.kinematics.forward_kinematics(q)
            e, pos_err, rot_err = self._compute_error(T_current, T_target, position_only)

            if pos_err < self.config.tolerance:
                return IKResult(
                    q=q,
                    success=True,
                    position_error=pos_err,
                    orientation_error=rot_err,
                    iterations=iteration + 1,
                    message="Converged"
                )

            J = self.kinematics.jacobian(q)
            if position_only:
                J = J[:3, :]
                e = e[:3]

            # Jacobian transpose
            dq = alpha * J.T @ e
            q = q + dq

            if self.kinematics.joint_limits:
                q = self.kinematics.joint_limits.clamp(q)

        T_final = self.kinematics.forward_kinematics(q)
        _, pos_err, rot_err = self._compute_error(T_final, T_target, position_only)

        return IKResult(
            q=q,
            success=pos_err < self.config.tolerance * 10,
            position_error=pos_err,
            orientation_error=rot_err,
            iterations=self.config.max_iterations,
            message="Max iterations reached"
        )

    def _solve_pinv(
        self,
        T_target: np.ndarray,
        q: np.ndarray,
        position_only: bool
    ) -> IKResult:
        """Solve using Pseudoinverse."""
        alpha = self.config.step_size

        for iteration in range(self.config.max_iterations):
            T_current = self.kinematics.forward_kinematics(q)
            e, pos_err, rot_err = self._compute_error(T_current, T_target, position_only)

            if pos_err < self.config.tolerance:
                return IKResult(
                    q=q,
                    success=True,
                    position_error=pos_err,
                    orientation_error=rot_err,
                    iterations=iteration + 1,
                    message="Converged"
                )

            J = self.kinematics.jacobian(q)
            if position_only:
                J = J[:3, :]
                e = e[:3]

            # Pseudoinverse
            dq = np.linalg.pinv(J) @ e
            q = q + alpha * dq

            if self.kinematics.joint_limits:
                q = self.kinematics.joint_limits.clamp(q)

        T_final = self.kinematics.forward_kinematics(q)
        _, pos_err, rot_err = self._compute_error(T_final, T_target, position_only)

        return IKResult(
            q=q,
            success=pos_err < self.config.tolerance * 10,
            position_error=pos_err,
            orientation_error=rot_err,
            iterations=self.config.max_iterations,
            message="Max iterations reached"
        )

    def solve_with_nullspace(
        self,
        T_target: np.ndarray,
        q_init: np.ndarray,
        q_preferred: np.ndarray,
        nullspace_weight: float = 0.1
    ) -> IKResult:
        """
        Solve IK with null-space optimization.

        The null-space is used to optimize for a secondary objective
        (e.g., staying close to preferred configuration).

        Args:
            T_target: Target end-effector pose
            q_init: Initial configuration
            q_preferred: Preferred configuration for null-space
            nullspace_weight: Weight for null-space objective

        Returns:
            IKResult with solution
        """
        q = q_init.copy()
        damping = self.config.damping
        alpha = self.config.step_size

        for iteration in range(self.config.max_iterations):
            T_current = self.kinematics.forward_kinematics(q)
            e, pos_err, rot_err = self._compute_error(T_current, T_target)

            if pos_err < self.config.tolerance:
                return IKResult(
                    q=q,
                    success=True,
                    position_error=pos_err,
                    orientation_error=rot_err,
                    iterations=iteration + 1,
                    message="Converged"
                )

            J = self.kinematics.jacobian(q)
            n = J.shape[1]

            # Compute primary task solution
            JJT = J @ J.T + damping**2 * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(JJT)
            dq_primary = J_pinv @ e

            # Null-space projection
            N = np.eye(n) - J_pinv @ J

            # Secondary objective: move toward preferred config
            dq_secondary = N @ (q_preferred - q)

            # Combined update
            dq = dq_primary + nullspace_weight * dq_secondary
            q = q + alpha * dq

            if self.kinematics.joint_limits:
                q = self.kinematics.joint_limits.clamp(q)

        T_final = self.kinematics.forward_kinematics(q)
        _, pos_err, rot_err = self._compute_error(T_final, T_target)

        return IKResult(
            q=q,
            success=pos_err < self.config.tolerance * 10,
            position_error=pos_err,
            orientation_error=rot_err,
            iterations=self.config.max_iterations,
            message="Max iterations reached"
        )

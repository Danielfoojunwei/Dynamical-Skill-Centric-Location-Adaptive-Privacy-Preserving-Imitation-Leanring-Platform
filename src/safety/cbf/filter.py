"""
Control Barrier Function Filter

Filters proposed actions to satisfy safety constraints using QP optimization:

    min ||a - a_proposed||²
    s.t. ∇h(x)·a + α·h(x) ≥ 0  for all barriers h

This provides DETERMINISTIC safety guarantees.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .barriers import (
    BarrierFunction,
    CollisionBarrier,
    JointLimitBarrier,
    VelocityBarrier,
    ForceBarrier,
    RobotState,
)

logger = logging.getLogger(__name__)

# Optional QP solver
try:
    from scipy.optimize import minimize, LinearConstraint
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class CBFConfig:
    """Configuration for CBF Filter."""
    # Barrier parameters
    min_obstacle_distance: float = 0.05  # meters
    max_joint_velocity: float = 2.0  # rad/s
    max_ee_velocity: float = 1.0  # m/s
    max_force: float = 50.0  # Newtons
    max_torque: float = 10.0  # Nm

    # Joint limits (7-DOF arm defaults)
    joint_limits_lower: Optional[np.ndarray] = None
    joint_limits_upper: Optional[np.ndarray] = None
    joint_limit_margin: float = 0.1  # radians

    # CBF parameters
    default_alpha: float = 1.0  # CBF decay rate

    # Solver parameters
    solver_max_iter: int = 100
    solver_tolerance: float = 1e-6

    # Action space
    action_dim: int = 7
    action_limits: float = 1.0  # Max action magnitude

    @classmethod
    def for_franka(cls) -> 'CBFConfig':
        """Configuration for Franka Panda."""
        return cls(
            joint_limits_lower=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            joint_limits_upper=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
            max_joint_velocity=2.175,
            max_force=87.0,
        )


@dataclass
class CBFResult:
    """Result from CBF filtering."""
    safe_action: np.ndarray
    original_action: np.ndarray
    was_modified: bool
    modification_magnitude: float

    # Per-barrier info
    barrier_values: Dict[str, float] = field(default_factory=dict)
    constraint_margins: Dict[str, float] = field(default_factory=dict)
    active_constraints: List[str] = field(default_factory=list)


class CBFFilter:
    """
    Control Barrier Function safety filter.

    Provides DETERMINISTIC safety guarantees by solving:

        min ||a - a_proposed||²
        s.t. ∇h_i(x)·a + α_i·h_i(x) ≥ 0  for all i

    This ensures h(x) ≥ 0 is invariant (safety maintained forever).
    """

    def __init__(self, config: Optional[CBFConfig] = None):
        self.config = config or CBFConfig()
        self.barriers: List[BarrierFunction] = []

        self._setup_default_barriers()

        # Statistics
        self.stats = {
            "total_filters": 0,
            "actions_modified": 0,
            "constraints_activated": 0,
            "solver_failures": 0,
        }

    def _setup_default_barriers(self):
        """Setup default safety barriers."""
        self.barriers = [
            CollisionBarrier(
                min_distance=self.config.min_obstacle_distance,
                alpha=self.config.default_alpha,
            ),
            JointLimitBarrier(
                joint_limits_lower=self.config.joint_limits_lower,
                joint_limits_upper=self.config.joint_limits_upper,
                margin=self.config.joint_limit_margin,
                alpha=2.0,
            ),
            VelocityBarrier(
                max_joint_velocity=self.config.max_joint_velocity,
                max_ee_velocity=self.config.max_ee_velocity,
                alpha=2.0,
            ),
            ForceBarrier(
                max_force=self.config.max_force,
                max_torque=self.config.max_torque,
                alpha=3.0,
            ),
        ]

    def add_barrier(self, barrier: BarrierFunction):
        """Add a custom barrier function."""
        self.barriers.append(barrier)

    def filter(
        self,
        proposed_action: np.ndarray,
        state: RobotState,
    ) -> CBFResult:
        """
        Filter action to satisfy all CBF constraints.

        Args:
            proposed_action: Action from learned policy [action_dim]
            state: Current robot state

        Returns:
            CBFResult with safe action (GUARANTEED to satisfy constraints)
        """
        self.stats["total_filters"] += 1

        # Evaluate all barriers
        barrier_values = {}
        constraint_margins = {}
        active_constraints = []

        for barrier in self.barriers:
            h_val = barrier.h(state)
            barrier_values[barrier.name] = h_val

            # Check if constraint would be violated
            margin = barrier.constraint(state, proposed_action)
            constraint_margins[barrier.name] = margin

            if margin < 0:
                active_constraints.append(barrier.name)

        # If no constraints active, action is safe
        if not active_constraints:
            return CBFResult(
                safe_action=proposed_action.copy(),
                original_action=proposed_action.copy(),
                was_modified=False,
                modification_magnitude=0.0,
                barrier_values=barrier_values,
                constraint_margins=constraint_margins,
                active_constraints=[],
            )

        # Solve QP to find safe action
        self.stats["constraints_activated"] += 1
        safe_action = self._solve_cbf_qp(proposed_action, state)

        modification = np.linalg.norm(safe_action - proposed_action)
        was_modified = modification > 1e-6

        if was_modified:
            self.stats["actions_modified"] += 1

        return CBFResult(
            safe_action=safe_action,
            original_action=proposed_action.copy(),
            was_modified=was_modified,
            modification_magnitude=float(modification),
            barrier_values=barrier_values,
            constraint_margins=constraint_margins,
            active_constraints=active_constraints,
        )

    def _solve_cbf_qp(
        self,
        proposed_action: np.ndarray,
        state: RobotState,
    ) -> np.ndarray:
        """
        Solve CBF-QP to find minimally modified safe action.

        Solves:
            min ||a - a_proposed||²
            s.t. ∇h_i(x)·a + α_i·h_i(x) ≥ 0  for all i
        """
        if HAS_SCIPY:
            return self._solve_with_scipy(proposed_action, state)
        else:
            return self._solve_simple(proposed_action, state)

    def _solve_with_scipy(
        self,
        proposed_action: np.ndarray,
        state: RobotState,
    ) -> np.ndarray:
        """Solve using scipy optimizer."""
        n = len(proposed_action)

        # Objective: ||a - proposed||²
        def objective(a):
            return np.sum((a - proposed_action) ** 2)

        def objective_grad(a):
            return 2 * (a - proposed_action)

        # Constraints: ∇h·a + α·h ≥ 0  →  -∇h·a ≤ α·h
        A_ub = []
        b_ub = []

        for barrier in self.barriers:
            grad = barrier.grad_h(state)
            h_val = barrier.h(state)

            # -∇h·a ≤ α·h
            A_ub.append(-grad)
            b_ub.append(barrier.alpha * h_val)

        if A_ub:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            constraints = LinearConstraint(A_ub, -np.inf, b_ub)

            result = minimize(
                objective,
                proposed_action,
                method='SLSQP',
                jac=objective_grad,
                constraints={'type': 'ineq', 'fun': lambda a: b_ub - A_ub @ a},
                bounds=[(-self.config.action_limits, self.config.action_limits)] * n,
                options={'maxiter': self.config.solver_max_iter},
            )

            if result.success:
                return result.x
            else:
                self.stats["solver_failures"] += 1
                logger.warning(f"CBF QP solver failed: {result.message}")
                return self._solve_simple(proposed_action, state)
        else:
            return proposed_action

    def _solve_simple(
        self,
        proposed_action: np.ndarray,
        state: RobotState,
    ) -> np.ndarray:
        """
        Simple gradient projection when scipy not available.

        Projects action onto safe set using iterative gradient steps.
        """
        action = proposed_action.copy()
        max_iters = 50
        step_size = 0.1

        for _ in range(max_iters):
            # Check all constraints
            all_satisfied = True

            for barrier in self.barriers:
                margin = barrier.constraint(state, action)

                if margin < 0:
                    all_satisfied = False
                    # Project in gradient direction
                    grad = barrier.grad_h(state)
                    grad_norm = np.linalg.norm(grad)

                    if grad_norm > 1e-6:
                        # Move in direction that increases h
                        action += step_size * grad / grad_norm

            if all_satisfied:
                break

        # Clamp to action limits
        action = np.clip(action, -self.config.action_limits, self.config.action_limits)

        return action

    def is_safe(self, state: RobotState) -> bool:
        """Check if current state satisfies all barriers."""
        return all(barrier.is_safe(state) for barrier in self.barriers)

    def get_safety_margin(self, state: RobotState) -> float:
        """Get minimum safety margin across all barriers."""
        return min(barrier.h(state) for barrier in self.barriers)

    @classmethod
    def for_franka(cls) -> 'CBFFilter':
        """Create CBF filter configured for Franka Panda."""
        return cls(CBFConfig.for_franka())

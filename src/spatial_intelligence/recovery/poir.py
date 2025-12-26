"""
POIR Recovery - Planning to Return to Training Distribution

Implements recovery planning based on:
- POIR: "Planning to Practice" - learning to return to familiar states
- Uses world model (V-JEPA 2) to simulate paths back to distribution

When the robot drifts OOD, this module plans trajectories that return
the robot to states where imitation policies are reliable.

Key Features:
- Simulates recovery trajectories using world model
- Multiple recovery strategies (reverse, safe waypoint, home)
- Confidence-based strategy selection
- Integration with RIP for OOD detection

Architecture:
    OOD Detection (RIP) → POIR → Recovery Trajectory → Robot
                            ↑
              V-JEPA 2 World Model (simulation)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    REVERSE = "reverse"         # Reverse recent actions
    SAFE_WAYPOINT = "safe_waypoint"  # Navigate to known safe state
    HOME = "home"               # Return to home/initial position
    SLOW_CONTINUE = "slow_continue"  # Continue slowly with caution
    STOP = "stop"               # Emergency stop


class RecoveryStatus(Enum):
    """Status of recovery execution."""
    NOT_NEEDED = "not_needed"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RecoveryPlan:
    """A recovery plan with trajectory back to distribution."""
    # Strategy used
    strategy: RecoveryStrategy

    # Planned trajectory [H, A]
    trajectory: Any

    # Quality metrics
    estimated_steps: int
    confidence: float  # Confidence this will return to distribution
    simulated_ood_at_end: float  # Predicted OOD score after execution

    # Target state
    target_state: Optional[Any] = None
    target_description: Optional[str] = None

    # Execution tracking
    status: RecoveryStatus = RecoveryStatus.PLANNING
    steps_executed: int = 0


@dataclass
class POIRConfig:
    """Configuration for POIR Recovery."""
    # Trajectory parameters
    action_dim: int = 7
    max_recovery_steps: int = 50
    planning_horizon: int = 16

    # Strategy parameters
    reverse_window: int = 10  # How many steps back to consider for reverse
    num_simulation_samples: int = 8  # Trajectories to simulate

    # Thresholds
    recovery_success_threshold: float = 0.2  # OOD score to consider "in distribution"
    min_confidence: float = 0.5  # Minimum confidence to attempt recovery

    # World model integration
    use_world_model: bool = True
    world_model_horizon: int = 8

    # Safe waypoints (can be populated from demonstrations)
    safe_waypoints: List[Any] = field(default_factory=list)
    home_position: Optional[Any] = None

    # Device
    device: str = "cuda"


class WorldModelSimulator:
    """
    Simulates trajectories using V-JEPA 2 world model.

    This is a simplified interface - in production, this would
    connect to the actual V-JEPA 2 implementation.
    """

    def __init__(self, config: POIRConfig):
        self.config = config
        self._world_model = None

    def set_world_model(self, world_model: Any):
        """Set the V-JEPA 2 world model instance."""
        self._world_model = world_model

    def simulate_trajectory(
        self,
        current_state: Any,
        actions: Any,
    ) -> Tuple[List[Any], List[float]]:
        """
        Simulate executing actions from current state.

        Args:
            current_state: Current observation/embedding
            actions: Trajectory of actions [H, A]

        Returns:
            (predicted_states, predicted_ood_scores)
        """
        if self._world_model is not None and hasattr(self._world_model, 'predict_trajectory'):
            return self._world_model.predict_trajectory(current_state, actions)

        # Mock simulation
        if HAS_NUMPY:
            H = actions.shape[0] if hasattr(actions, 'shape') else len(actions)
            # Simulate states as accumulated actions (simplified)
            states = [current_state]
            ood_scores = []

            for i in range(H):
                # Mock: OOD score decreases as we execute recovery
                ood = max(0.1, 0.8 - i * 0.1)
                ood_scores.append(ood)
                states.append(current_state)  # Simplified

            return states, ood_scores

        return [], []


class POIRRecovery:
    """
    POIR (Planning to Return to Distribution) Recovery.

    When OOD is detected, plans a trajectory to return the robot
    to familiar states where imitation is reliable.

    Usage:
        config = POIRConfig(action_dim=7)
        poir = POIRRecovery(config)

        # When RIP detects OOD
        if safety_decision.should_trigger_recovery:
            plan = poir.plan_recovery(
                current_observation=observation,
                recent_actions=action_history,
                ood_score=safety_decision.ood_score,
            )

            # Execute recovery
            for action in plan.trajectory:
                robot.execute(action)
                if poir.check_in_distribution():
                    break
    """

    def __init__(self, config: Optional[POIRConfig] = None):
        self.config = config or POIRConfig()

        # World model for simulation
        self.simulator = WorldModelSimulator(self.config)

        # Action history buffer
        self.action_history: List[Any] = []
        self.state_history: List[Any] = []
        self.max_history = 100

        # Current recovery state
        self.current_plan: Optional[RecoveryPlan] = None
        self.recovery_status = RecoveryStatus.NOT_NEEDED

        # OOD evaluator callback
        self._ood_evaluator: Optional[Callable] = None

        # Stats
        self.stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "strategy_usage": {s.value: 0 for s in RecoveryStrategy},
            "avg_recovery_steps": 0.0,
        }

    def set_world_model(self, world_model: Any):
        """Set world model for trajectory simulation."""
        self.simulator.set_world_model(world_model)

    def set_ood_evaluator(self, evaluator: Callable[[Any], float]):
        """Set function to evaluate OOD score of a state."""
        self._ood_evaluator = evaluator

    def record_step(self, observation: Any, action: Any):
        """Record a step for history (used for reverse strategy)."""
        self.action_history.append(action)
        self.state_history.append(observation)

        # Limit history size
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
            self.state_history.pop(0)

    def plan_recovery(
        self,
        current_observation: Any,
        recent_actions: Optional[List[Any]] = None,
        ood_score: float = 0.5,
    ) -> RecoveryPlan:
        """
        Plan a recovery trajectory to return to distribution.

        Args:
            current_observation: Current scene observation
            recent_actions: Recent action history (or uses internal buffer)
            ood_score: Current OOD score from RIP

        Returns:
            RecoveryPlan with trajectory and metadata
        """
        self.recovery_status = RecoveryStatus.PLANNING
        self.stats["total_recoveries"] += 1

        # Use provided history or internal buffer
        if recent_actions is None:
            recent_actions = self.action_history[-self.config.reverse_window:]

        # Try different strategies and pick best
        candidates = []

        # Strategy 1: Reverse recent actions
        reverse_plan = self._plan_reverse(current_observation, recent_actions)
        if reverse_plan is not None:
            candidates.append(reverse_plan)

        # Strategy 2: Go to safe waypoint
        if self.config.safe_waypoints:
            waypoint_plan = self._plan_to_waypoint(current_observation)
            if waypoint_plan is not None:
                candidates.append(waypoint_plan)

        # Strategy 3: Go home
        if self.config.home_position is not None:
            home_plan = self._plan_to_home(current_observation)
            if home_plan is not None:
                candidates.append(home_plan)

        # Strategy 4: Slow continue (fallback)
        slow_plan = self._plan_slow_continue(current_observation)
        candidates.append(slow_plan)

        # Select best plan based on confidence and simulated OOD
        best_plan = self._select_best_plan(candidates)

        self.current_plan = best_plan
        self.recovery_status = RecoveryStatus.EXECUTING
        self.stats["strategy_usage"][best_plan.strategy.value] += 1

        return best_plan

    def _plan_reverse(
        self,
        current_obs: Any,
        recent_actions: List[Any],
    ) -> Optional[RecoveryPlan]:
        """Plan by reversing recent actions."""
        if not recent_actions:
            return None

        if not HAS_NUMPY:
            return None

        # Reverse and negate actions
        reversed_actions = []
        for action in reversed(recent_actions):
            if isinstance(action, np.ndarray):
                reversed_actions.append(-action)
            else:
                reversed_actions.append(action)

        trajectory = np.array(reversed_actions)

        # Simulate to estimate OOD at end
        _, ood_scores = self.simulator.simulate_trajectory(current_obs, trajectory)
        final_ood = ood_scores[-1] if ood_scores else 0.5

        return RecoveryPlan(
            strategy=RecoveryStrategy.REVERSE,
            trajectory=trajectory,
            estimated_steps=len(trajectory),
            confidence=0.7 if final_ood < self.config.recovery_success_threshold else 0.4,
            simulated_ood_at_end=final_ood,
            target_description="Reverse recent trajectory",
        )

    def _plan_to_waypoint(self, current_obs: Any) -> Optional[RecoveryPlan]:
        """Plan path to nearest safe waypoint."""
        if not self.config.safe_waypoints or not HAS_NUMPY:
            return None

        # Find nearest waypoint
        waypoints = self.config.safe_waypoints
        current_pos = current_obs[:3] if hasattr(current_obs, '__getitem__') else None

        if current_pos is None:
            # Use first waypoint as default
            target_waypoint = waypoints[0]
        else:
            # Find nearest
            distances = []
            for wp in waypoints:
                if isinstance(wp, np.ndarray) and len(wp) >= 3:
                    dist = np.linalg.norm(current_pos - wp[:3])
                    distances.append(dist)
                else:
                    distances.append(float('inf'))
            target_waypoint = waypoints[np.argmin(distances)]

        # Generate trajectory to waypoint (linear interpolation)
        trajectory = self._interpolate_to_target(
            current_obs, target_waypoint, self.config.planning_horizon
        )

        return RecoveryPlan(
            strategy=RecoveryStrategy.SAFE_WAYPOINT,
            trajectory=trajectory,
            estimated_steps=len(trajectory),
            confidence=0.8,
            simulated_ood_at_end=0.2,  # Waypoints are known safe
            target_state=target_waypoint,
            target_description="Navigate to safe waypoint",
        )

    def _plan_to_home(self, current_obs: Any) -> Optional[RecoveryPlan]:
        """Plan path to home position."""
        if self.config.home_position is None or not HAS_NUMPY:
            return None

        trajectory = self._interpolate_to_target(
            current_obs,
            self.config.home_position,
            self.config.max_recovery_steps,
        )

        return RecoveryPlan(
            strategy=RecoveryStrategy.HOME,
            trajectory=trajectory,
            estimated_steps=len(trajectory),
            confidence=0.9,  # Home is very reliable
            simulated_ood_at_end=0.1,
            target_state=self.config.home_position,
            target_description="Return to home position",
        )

    def _plan_slow_continue(self, current_obs: Any) -> RecoveryPlan:
        """Plan slow continuation (fallback strategy)."""
        if HAS_NUMPY:
            # Generate small random actions (exploration)
            trajectory = np.random.randn(
                self.config.planning_horizon,
                self.config.action_dim
            ).astype(np.float32) * 0.1
        else:
            trajectory = [[0.0] * self.config.action_dim] * self.config.planning_horizon

        return RecoveryPlan(
            strategy=RecoveryStrategy.SLOW_CONTINUE,
            trajectory=trajectory,
            estimated_steps=self.config.planning_horizon,
            confidence=0.3,
            simulated_ood_at_end=0.5,  # Uncertain
            target_description="Continue slowly with exploration",
        )

    def _interpolate_to_target(
        self,
        current: Any,
        target: Any,
        steps: int,
    ) -> Any:
        """Generate linear interpolation trajectory."""
        if not HAS_NUMPY:
            return [[0.0] * self.config.action_dim] * steps

        current = np.array(current) if not isinstance(current, np.ndarray) else current
        target = np.array(target) if not isinstance(target, np.ndarray) else target

        # Ensure same shape (use action_dim)
        if len(current) > self.config.action_dim:
            current = current[:self.config.action_dim]
        if len(target) > self.config.action_dim:
            target = target[:self.config.action_dim]

        # Pad if needed
        if len(current) < self.config.action_dim:
            current = np.pad(current, (0, self.config.action_dim - len(current)))
        if len(target) < self.config.action_dim:
            target = np.pad(target, (0, self.config.action_dim - len(target)))

        # Linear interpolation
        trajectory = np.zeros((steps, self.config.action_dim), dtype=np.float32)
        for i in range(steps):
            alpha = (i + 1) / steps
            trajectory[i] = current + alpha * (target - current)

        # Convert to velocity commands (differences)
        velocities = np.diff(trajectory, axis=0, prepend=current.reshape(1, -1))

        return velocities

    def _select_best_plan(self, candidates: List[RecoveryPlan]) -> RecoveryPlan:
        """Select best recovery plan from candidates."""
        if not candidates:
            # Emergency stop
            return RecoveryPlan(
                strategy=RecoveryStrategy.STOP,
                trajectory=np.zeros((1, self.config.action_dim)) if HAS_NUMPY else [[0] * self.config.action_dim],
                estimated_steps=1,
                confidence=1.0,
                simulated_ood_at_end=1.0,
                target_description="Emergency stop",
            )

        # Score each plan
        scores = []
        for plan in candidates:
            # Higher confidence is better
            score = plan.confidence * 2.0

            # Lower simulated OOD is better
            score -= plan.simulated_ood_at_end

            # Fewer steps is better (faster recovery)
            score -= plan.estimated_steps * 0.01

            scores.append(score)

        best_idx = scores.index(max(scores)) if not HAS_NUMPY else int(np.argmax(scores))
        return candidates[best_idx]

    def step_recovery(self, current_observation: Any) -> Tuple[Any, bool]:
        """
        Get next recovery action and check if complete.

        Args:
            current_observation: Current observation

        Returns:
            (next_action, is_complete)
        """
        if self.current_plan is None:
            return None, True

        plan = self.current_plan

        # Check if we've executed all steps
        if plan.steps_executed >= plan.estimated_steps:
            self._complete_recovery(True)
            return None, True

        # Check if we're back in distribution
        if self._ood_evaluator is not None:
            current_ood = self._ood_evaluator(current_observation)
            if current_ood < self.config.recovery_success_threshold:
                self._complete_recovery(True)
                return None, True

        # Get next action
        action = plan.trajectory[plan.steps_executed]
        plan.steps_executed += 1

        return action, False

    def _complete_recovery(self, success: bool):
        """Mark recovery as complete."""
        if self.current_plan:
            self.current_plan.status = RecoveryStatus.COMPLETED if success else RecoveryStatus.FAILED

            if success:
                self.stats["successful_recoveries"] += 1

            # Update average steps
            total = self.stats["total_recoveries"]
            avg = self.stats["avg_recovery_steps"]
            self.stats["avg_recovery_steps"] = (
                avg * (total - 1) + self.current_plan.steps_executed
            ) / total

        self.recovery_status = RecoveryStatus.COMPLETED if success else RecoveryStatus.FAILED
        self.current_plan = None

    def abort_recovery(self):
        """Abort current recovery."""
        if self.current_plan:
            self.current_plan.status = RecoveryStatus.FAILED
        self.recovery_status = RecoveryStatus.NOT_NEEDED
        self.current_plan = None

    def add_safe_waypoint(self, waypoint: Any, description: str = ""):
        """Add a safe waypoint for recovery planning."""
        self.config.safe_waypoints.append(waypoint)
        logger.info(f"Added safe waypoint: {description}")

    def set_home_position(self, position: Any):
        """Set home position for home recovery strategy."""
        self.config.home_position = position
        logger.info("Set home position for recovery")

    @property
    def is_recovering(self) -> bool:
        return self.recovery_status == RecoveryStatus.EXECUTING

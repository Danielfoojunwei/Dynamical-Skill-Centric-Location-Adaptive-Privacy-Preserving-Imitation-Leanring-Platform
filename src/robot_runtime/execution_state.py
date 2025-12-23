"""
Execution State Machine - Skill Execution Management

Manages the state of skill execution on the robot.
Runs at Tier 1 rate (1kHz) for state transitions.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """Skill execution states."""
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SkillContext:
    """Context for currently executing skill."""
    skill_id: str = ""
    skill_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    progress: float = 0.0  # 0.0 to 1.0
    state: ExecutionState = ExecutionState.IDLE
    error_message: str = ""


@dataclass
class Skill:
    """Skill definition."""
    id: str
    name: str
    policy_model: str
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionStateMachine:
    """
    State machine for skill execution.

    Manages:
    - Skill loading and initialization
    - Execution state transitions
    - Action generation
    - Progress tracking
    """

    def __init__(self):
        self._context = SkillContext()
        self._current_skill: Optional[Skill] = None
        self._action_buffer: Optional[np.ndarray] = None
        self._fallback_action: Optional[np.ndarray] = None
        self._trajectory: Optional[np.ndarray] = None
        self._trajectory_index: int = 0

    def start_skill(self, skill: Skill, parameters: Dict[str, Any]) -> bool:
        """
        Start executing a skill.

        Args:
            skill: Skill to execute
            parameters: Skill parameters

        Returns:
            True if skill started successfully
        """
        if self._context.state in [ExecutionState.RUNNING, ExecutionState.LOADING]:
            logger.warning(f"Cannot start skill - already executing {self._context.skill_id}")
            return False

        self._current_skill = skill
        self._context = SkillContext(
            skill_id=skill.id,
            skill_name=skill.name,
            parameters=parameters,
            start_time=time.time(),
            progress=0.0,
            state=ExecutionState.LOADING
        )

        logger.info(f"Starting skill: {skill.id}")

        # Transition to running
        self._context.state = ExecutionState.RUNNING
        return True

    def stop_skill(self) -> None:
        """Stop currently executing skill."""
        if self._context.state == ExecutionState.RUNNING:
            logger.info(f"Stopping skill: {self._context.skill_id}")
            self._context.state = ExecutionState.CANCELLED

        self._current_skill = None
        self._action_buffer = None

    def pause_skill(self) -> None:
        """Pause currently executing skill."""
        if self._context.state == ExecutionState.RUNNING:
            self._context.state = ExecutionState.PAUSED

    def resume_skill(self) -> None:
        """Resume paused skill."""
        if self._context.state == ExecutionState.PAUSED:
            self._context.state = ExecutionState.RUNNING

    def set_action(self, action: np.ndarray) -> None:
        """
        Set the action to be executed (called from Tier 2).

        Args:
            action: Action array (joint velocities, etc.)
        """
        self._action_buffer = action

    def get_current_action(self) -> Optional[np.ndarray]:
        """
        Get current action for Tier 1 control loop.

        Returns:
            Action array or None if no action
        """
        if self._context.state != ExecutionState.RUNNING:
            return None

        return self._action_buffer

    def get_fallback_action(self) -> np.ndarray:
        """
        Get fallback action when no policy output available.

        Returns:
            Zero velocity action (safe stop)
        """
        if self._fallback_action is not None:
            return self._fallback_action

        # Default: 7 DOF zero velocity
        return np.zeros(7)

    def set_fallback_action(self, action: np.ndarray) -> None:
        """Set the fallback action."""
        self._fallback_action = action

    def update_progress(self, progress: float) -> None:
        """Update skill progress (0.0 to 1.0)."""
        self._context.progress = np.clip(progress, 0.0, 1.0)

        if progress >= 1.0:
            self._context.state = ExecutionState.COMPLETED
            logger.info(f"Skill completed: {self._context.skill_id}")

    def mark_failed(self, error_message: str) -> None:
        """Mark skill as failed."""
        self._context.state = ExecutionState.FAILED
        self._context.error_message = error_message
        logger.error(f"Skill failed: {self._context.skill_id} - {error_message}")

    def get_context(self) -> Dict[str, Any]:
        """Get current execution context for policy."""
        return {
            'skill_id': self._context.skill_id,
            'parameters': self._context.parameters,
            'progress': self._context.progress,
            'state': self._context.state.value,
            'elapsed_time': time.time() - self._context.start_time,
        }

    def get_state(self) -> ExecutionState:
        """Get current execution state."""
        return self._context.state

    def update_trajectory_avoidance(self, obstacles: List[Dict[str, Any]]) -> None:
        """
        Update trajectory to avoid obstacles.

        Called from local planning (10Hz).
        """
        if not obstacles:
            return

        # Simple reactive avoidance - more sophisticated in production
        # Adjust trajectory to avoid closest obstacle
        closest = min(obstacles, key=lambda o: o.get('distance', float('inf')))

        if closest.get('distance', float('inf')) < 0.3:  # 30cm threshold
            logger.warning(f"Obstacle avoidance triggered: {closest}")
            # Would modify trajectory here

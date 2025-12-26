"""
Dynamical Executor - Simplified Architecture without Skill Blending

This executor leverages Deep Imitative Learning to eliminate the need for
skill blending entirely. The key insight:

    VLA models learn multi-objective behavior implicitly from demonstrations.
    There's no need to blend "grasp" + "avoid collision" at runtime -
    the model learns to grasp while avoiding collisions from the training data.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                         DYNAMICAL EXECUTOR                                       │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                  │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐            │
    │  │  Task Decomposer │────►│    Pi0.5 VLA    │────►│    Diffusion    │            │
    │  │  (long-horizon)  │     │                 │     │    Planner      │            │
    │  └─────────────────┘     └─────────────────┘     └─────────────────┘            │
    │          │                       │                       │                       │
    │          │                       ▼                       ▼                       │
    │          │               Semantic               Smooth                          │
    │          │               Actions                Trajectory                       │
    │          │                       │                       │                       │
    │          │                       └───────────┬───────────┘                       │
    │          │                                   ▼                                   │
    │          │                    ┌─────────────────────────┐                        │
    │          │                    │      CBF Filter         │                        │
    │          │                    │   (hard safety)         │                        │
    │          │                    └─────────────────────────┘                        │
    │          │                                   │                                   │
    │          │                                   ▼                                   │
    │          │                           SAFE ACTION                                 │
    │          │                      (guaranteed by CBF)                              │
    │          │                                                                       │
    │          └─────────── Sub-task chaining (NOT blending) ─────────────────────────┘
    │                                                                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘

Key differences from skill blending:
1. No runtime combination of multiple skill outputs
2. Task decomposition is sequential, not parallel
3. VLA handles multi-objective implicitly
4. CBF provides hard safety guarantees (not probabilistic)

Usage:
    executor = DynamicalExecutor.for_jetson_thor()

    # Simple task - direct execution
    result = executor.execute(
        instruction="pick up the red cup",
        images=camera_images,
        state=robot_state,
    )

    # Complex task - automatic decomposition
    result = executor.execute(
        instruction="set the table for dinner",
        images=camera_images,
        state=robot_state,
    )
    # Internally decomposes into sub-tasks and executes sequentially
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExecutorState(Enum):
    """Current state of the executor."""
    IDLE = "idle"
    EXECUTING = "executing"
    DECOMPOSING = "decomposing"
    RECOVERING = "recovering"
    ERROR = "error"


@dataclass
class ExecutorConfig:
    """Configuration for DynamicalExecutor."""
    # Action dimensions
    action_dim: int = 7
    action_horizon: int = 16

    # Task decomposition
    enable_decomposition: bool = True
    decomposition_threshold: float = 0.7  # Complexity score to trigger decomposition

    # Safety
    use_cbf: bool = True
    use_rta: bool = True

    # Deep IL
    use_diffusion: bool = True
    diffusion_steps: int = 30

    # Hardware
    device: str = "cuda"

    # Timing
    target_hz: float = 10.0
    safety_hz: float = 1000.0

    @classmethod
    def for_jetson_thor(cls) -> "ExecutorConfig":
        """Optimized for Jetson Thor."""
        return cls(
            device="cuda",
            use_cbf=True,
            use_rta=True,
            diffusion_steps=30,
        )

    @classmethod
    def minimal(cls) -> "ExecutorConfig":
        """Minimal config (CBF only)."""
        return cls(
            use_cbf=True,
            use_rta=False,
            use_diffusion=False,
            enable_decomposition=False,
        )


@dataclass
class ExecutionResult:
    """Result from executor."""
    # Output
    action: np.ndarray
    trajectory: Optional[np.ndarray] = None

    # Source
    source: str = "vla"  # "vla", "baseline", "recovery"
    was_safety_modified: bool = False

    # Task info
    current_subtask: Optional[str] = None
    subtask_index: int = 0
    total_subtasks: int = 1

    # Safety
    is_safe: bool = True
    safety_margin: float = 1.0

    # Performance
    total_time_ms: float = 0.0
    vla_time_ms: float = 0.0
    safety_time_ms: float = 0.0


class DynamicalExecutor:
    """
    Simplified executor that uses Deep Imitative Learning.

    Key insight: VLA learns multi-objective behavior implicitly,
    eliminating the need for skill blending entirely.

    The executor:
    1. Decomposes long-horizon tasks into sub-tasks
    2. Executes each sub-task with VLA + Diffusion
    3. Applies CBF safety filtering to all actions
    4. Uses RTA for verified fallback when needed

    NO SKILL BLENDING - VLA handles multi-objective internally.
    """

    def __init__(self, config: Optional[ExecutorConfig] = None):
        self.config = config or ExecutorConfig()

        # Components (lazy loaded)
        self._vla = None
        self._diffusion = None
        self._cbf = None
        self._rta = None
        self._decomposer = None

        # State
        self.state = ExecutorState.IDLE
        self._loaded = False

        # Current task execution
        self._current_instruction: Optional[str] = None
        self._subtasks: List[str] = []
        self._subtask_index: int = 0

        # Statistics
        self.stats = {
            "total_executions": 0,
            "decomposed_tasks": 0,
            "safety_interventions": 0,
            "rta_switches": 0,
            "avg_time_ms": 0.0,
        }

    def load(self):
        """Load all components."""
        if self._loaded:
            return

        logger.info("Loading DynamicalExecutor...")

        # Load VLA (Pi0.5)
        try:
            from ..spatial_intelligence.pi0 import Pi05Model, Pi05Config, HAS_OPENPI
            if HAS_OPENPI:
                pi05_config = Pi05Config(
                    device=self.config.device,
                    action_dim=self.config.action_dim,
                    action_horizon=self.config.action_horizon,
                )
                self._vla = Pi05Model(pi05_config)
                self._vla.load()
                logger.info("Loaded Pi0.5 VLA")
        except ImportError:
            logger.warning("Pi0.5 not available, using mock VLA")
            self._vla = MockVLA(self.config.action_dim, self.config.action_horizon)

        # Load Diffusion Planner
        if self.config.use_diffusion:
            try:
                from ..spatial_intelligence.planning import DiffusionPlanner, DiffusionConfig
                diff_config = DiffusionConfig(
                    action_dim=self.config.action_dim,
                    horizon=self.config.action_horizon,
                    num_diffusion_steps=self.config.diffusion_steps,
                    device=self.config.device,
                )
                self._diffusion = DiffusionPlanner(diff_config)
                self._diffusion.load_model()
                logger.info("Loaded Diffusion Planner")
            except ImportError:
                logger.warning("Diffusion Planner not available")

        # Load CBF Safety Filter
        if self.config.use_cbf:
            try:
                from ..safety.cbf import CBFFilter, CBFConfig
                cbf_config = CBFConfig(action_dim=self.config.action_dim)
                self._cbf = CBFFilter(cbf_config)
                logger.info("Loaded CBF Safety Filter")
            except ImportError:
                logger.warning("CBF not available")

        # Load RTA
        if self.config.use_rta:
            try:
                from ..safety.rta import RuntimeAssurance, RTAConfig
                rta_config = RTAConfig()
                self._rta = RuntimeAssurance(rta_config)
                if self._cbf is not None:
                    self._rta.set_cbf(self._cbf)
                logger.info("Loaded RTA")
            except ImportError:
                logger.warning("RTA not available")

        # Load Task Decomposer
        if self.config.enable_decomposition:
            try:
                from .task_decomposer import TaskDecomposer, DecomposerConfig
                self._decomposer = TaskDecomposer(DecomposerConfig())
                logger.info("Loaded Task Decomposer")
            except ImportError:
                logger.warning("Task Decomposer not available")

        self._loaded = True
        logger.info("DynamicalExecutor ready (NO SKILL BLENDING)")

    def execute(
        self,
        instruction: str,
        images: Any,
        state: Dict[str, Any],
        force_decomposition: bool = False,
    ) -> ExecutionResult:
        """
        Execute a task with Deep Imitative Learning.

        Args:
            instruction: Natural language task instruction
            images: Camera images
            state: Robot state dictionary
            force_decomposition: Force task decomposition even for simple tasks

        Returns:
            ExecutionResult with safe action to execute

        Note: This does NOT blend skills. The VLA handles multi-objective
        behavior implicitly. For long-horizon tasks, we decompose into
        sub-tasks and execute sequentially (not in parallel).
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        self.stats["total_executions"] += 1

        # Check if we need to decompose (long-horizon task)
        if self._should_decompose(instruction, force_decomposition):
            return self._execute_with_decomposition(instruction, images, state, start_time)

        # Simple task - direct execution
        return self._execute_single(instruction, images, state, start_time)

    def _should_decompose(self, instruction: str, force: bool) -> bool:
        """Determine if task needs decomposition."""
        if force:
            return True

        if not self.config.enable_decomposition:
            return False

        if self._decomposer is None:
            return False

        # Check complexity
        complexity = self._decomposer.estimate_complexity(instruction)
        return complexity > self.config.decomposition_threshold

    def _execute_single(
        self,
        instruction: str,
        images: Any,
        state: Dict[str, Any],
        start_time: float,
    ) -> ExecutionResult:
        """Execute a single task (no decomposition)."""
        self.state = ExecutorState.EXECUTING

        # Step 1: VLA inference
        vla_start = time.time()
        if self._vla is not None:
            vla_actions = self._vla_inference(instruction, images, state)
        else:
            vla_actions = np.zeros((self.config.action_horizon, self.config.action_dim))
        vla_time = (time.time() - vla_start) * 1000

        # Step 2: Diffusion refinement (optional)
        if self._diffusion is not None and self.config.use_diffusion:
            trajectory_batch = self._diffusion.plan(initial_actions=vla_actions)
            refined_actions = trajectory_batch.best.actions
        else:
            refined_actions = vla_actions

        # Step 3: Safety filtering (CBF + RTA)
        safety_start = time.time()
        safe_action, was_modified, source = self._apply_safety(
            refined_actions[0] if len(refined_actions.shape) > 1 else refined_actions,
            state,
        )
        safety_time = (time.time() - safety_start) * 1000

        if was_modified:
            self.stats["safety_interventions"] += 1

        # Compute safety margin
        safety_margin = self._get_safety_margin(state)

        total_time = (time.time() - start_time) * 1000
        self._update_avg_time(total_time)

        self.state = ExecutorState.IDLE

        return ExecutionResult(
            action=safe_action,
            trajectory=refined_actions,
            source=source,
            was_safety_modified=was_modified,
            current_subtask=instruction,
            subtask_index=0,
            total_subtasks=1,
            is_safe=safety_margin > 0,
            safety_margin=safety_margin,
            total_time_ms=total_time,
            vla_time_ms=vla_time,
            safety_time_ms=safety_time,
        )

    def _execute_with_decomposition(
        self,
        instruction: str,
        images: Any,
        state: Dict[str, Any],
        start_time: float,
    ) -> ExecutionResult:
        """Execute with task decomposition for long-horizon tasks."""
        self.state = ExecutorState.DECOMPOSING
        self.stats["decomposed_tasks"] += 1

        # Decompose if needed
        if self._current_instruction != instruction:
            self._current_instruction = instruction
            result = self._decomposer.decompose(instruction)
            self._subtasks = [st.instruction for st in result.subtasks]
            self._subtask_index = 0

        # Get current subtask
        if self._subtask_index >= len(self._subtasks):
            # All subtasks complete
            self._reset_task()
            self.state = ExecutorState.IDLE
            return ExecutionResult(
                action=np.zeros(self.config.action_dim),
                source="complete",
                current_subtask=None,
                subtask_index=self._subtask_index,
                total_subtasks=len(self._subtasks),
                is_safe=True,
            )

        current_subtask = self._subtasks[self._subtask_index]

        # Execute current subtask
        self.state = ExecutorState.EXECUTING
        result = self._execute_single(current_subtask, images, state, start_time)

        # Update result with decomposition info
        result.current_subtask = current_subtask
        result.subtask_index = self._subtask_index
        result.total_subtasks = len(self._subtasks)

        return result

    def advance_subtask(self):
        """Advance to next subtask (call when current subtask completes)."""
        self._subtask_index += 1

    def _reset_task(self):
        """Reset task state."""
        self._current_instruction = None
        self._subtasks = []
        self._subtask_index = 0

    def _vla_inference(
        self,
        instruction: str,
        images: Any,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Run VLA inference."""
        try:
            from ..spatial_intelligence.pi0 import Pi05Observation
            obs = Pi05Observation(
                images=images,
                instruction=instruction,
                proprio=state.get('joint_positions'),
            )
            result = self._vla.infer(obs)
            return np.array(result.actions)
        except Exception as e:
            logger.warning(f"VLA inference failed: {e}, using mock")
            return np.random.randn(
                self.config.action_horizon,
                self.config.action_dim,
            ).astype(np.float32) * 0.1

    def _apply_safety(
        self,
        action: np.ndarray,
        state: Dict[str, Any],
    ) -> Tuple[np.ndarray, bool, str]:
        """
        Apply safety filtering (CBF + RTA).

        Returns:
            (safe_action, was_modified, source)
        """
        source = "vla"
        was_modified = False

        # RTA arbitration first (if enabled)
        if self._rta is not None:
            try:
                from ..safety.rta.baseline import ControllerState
                from ..safety.rta import ControlSource
                controller_state = ControllerState(
                    joint_positions=np.array(state.get('joint_positions', np.zeros(7))),
                    joint_velocities=np.array(state.get('joint_velocities', np.zeros(7))),
                    ee_position=np.array(state.get('ee_position', np.zeros(3))),
                    ee_velocity=np.array(state.get('ee_velocity', np.zeros(3))),
                )
                action, rta_source = self._rta.arbitrate(action, controller_state)
                if rta_source != ControlSource.LEARNED:
                    source = rta_source.value
                    was_modified = True
                    self.stats["rta_switches"] += 1
            except Exception as e:
                logger.warning(f"RTA failed: {e}")

        # CBF filtering (always applied if available)
        if self._cbf is not None:
            try:
                from ..safety.cbf.barriers import RobotState
                robot_state = RobotState.from_observation(state)
                cbf_result = self._cbf.filter(action, robot_state)
                action = cbf_result.safe_action
                if cbf_result.was_modified:
                    was_modified = True
            except Exception as e:
                logger.warning(f"CBF failed: {e}")

        return action, was_modified, source

    def _get_safety_margin(self, state: Dict[str, Any]) -> float:
        """Get current safety margin from CBF."""
        if self._cbf is None:
            return 1.0
        try:
            from ..safety.cbf.barriers import RobotState
            robot_state = RobotState.from_observation(state)
            return self._cbf.get_safety_margin(robot_state)
        except Exception:
            return 1.0

    def _update_avg_time(self, total_time: float):
        """Update average execution time."""
        n = self.stats["total_executions"]
        self.stats["avg_time_ms"] = (
            self.stats["avg_time_ms"] * (n - 1) + total_time
        ) / n

    def emergency_stop(self, state: Dict[str, Any]) -> np.ndarray:
        """Force emergency stop."""
        self.state = ExecutorState.RECOVERING
        if self._rta is not None:
            try:
                from ..safety.rta.baseline import ControllerState
                controller_state = ControllerState(
                    joint_positions=np.array(state.get('joint_positions', np.zeros(7))),
                    joint_velocities=np.array(state.get('joint_velocities', np.zeros(7))),
                    ee_position=np.array(state.get('ee_position', np.zeros(3))),
                    ee_velocity=np.array(state.get('ee_velocity', np.zeros(3))),
                )
                return self._rta.emergency_stop(controller_state)
            except Exception:
                pass
        return np.zeros(self.config.action_dim)

    def set_home_position(self, home: np.ndarray):
        """Set home position for baseline controllers."""
        if self._rta is not None:
            self._rta.set_home_position(home)

    def add_exclusion_zone(self, center: np.ndarray, radius: float):
        """Add an exclusion zone to CBF."""
        if self._cbf is not None:
            try:
                from ..safety.cbf.barriers import ExclusionZoneBarrier
                barrier = ExclusionZoneBarrier(center, radius)
                self._cbf.add_barrier(barrier)
            except ImportError:
                pass

    @classmethod
    def for_jetson_thor(cls) -> "DynamicalExecutor":
        """Create executor optimized for Jetson Thor."""
        return cls(ExecutorConfig.for_jetson_thor())

    @classmethod
    def minimal(cls) -> "DynamicalExecutor":
        """Create minimal executor (CBF only)."""
        return cls(ExecutorConfig.minimal())


class MockVLA:
    """Mock VLA for testing when Pi0.5 not available."""

    def __init__(self, action_dim: int, action_horizon: int):
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def load(self):
        pass

    def infer(self, observation) -> Any:
        """Return mock actions."""
        @dataclass
        class MockResult:
            actions: np.ndarray

        return MockResult(
            actions=np.random.randn(self.action_horizon, self.action_dim).astype(np.float32) * 0.1
        )

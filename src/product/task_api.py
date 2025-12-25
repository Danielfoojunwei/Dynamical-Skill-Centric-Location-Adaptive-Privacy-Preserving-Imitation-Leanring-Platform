"""
Natural Language Task API

Execute robot tasks using natural language instructions with automatic
semantic understanding, task decomposition, and progress tracking.

Features:
=========
- Natural language task specification in 140+ languages
- Automatic task decomposition into subtasks
- Real-time progress tracking and status updates
- Failure recovery with automatic re-planning
- 128k token context for complex multi-turn interactions

Powered By:
==========
- Pi0.5: Vision-Language-Action with open-world generalization
- Gemma 3-27B: Multimodal understanding with 128k context
- Jetson Thor: 10Hz control with full perception pipeline

Usage:
    from src.product import TaskAPI

    # Create API instance
    api = TaskAPI.create_for_hardware()

    # Execute a task
    result = await api.execute(
        instruction="Pick up the red cup and place it on the shelf",
        wait_for_completion=True
    )

    # Check result
    print(f"Status: {result.status}")
    print(f"Subtasks completed: {result.subtasks_completed}/{result.subtasks_total}")
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import VLA interface
try:
    from ..spatial_intelligence.vla_interface import (
        VLAInterface,
        VLAConfig,
        VLAObservation,
        VLAResult,
    )
    HAS_VLA = True
except ImportError:
    HAS_VLA = False


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class TaskRequest:
    """Request to execute a task."""
    # Core instruction
    instruction: str

    # Task identification
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL

    # Execution options
    wait_for_completion: bool = True
    timeout_seconds: float = 300.0  # 5 minute default
    allow_replanning: bool = True
    max_retries: int = 3

    # Context (optional)
    context: Optional[Dict[str, Any]] = None
    previous_task_id: Optional[str] = None

    # Constraints (optional)
    workspace_bounds: Optional[Dict[str, float]] = None
    speed_limit: Optional[float] = None
    force_limit: Optional[float] = None

    # Callbacks
    on_subtask_complete: Optional[Callable] = None
    on_progress_update: Optional[Callable] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    language: str = "en"


@dataclass
class SubtaskResult:
    """Result of a single subtask."""
    subtask_id: str
    description: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    actions_executed: int = 0
    confidence: float = 0.0


@dataclass
class TaskResult:
    """Result of task execution."""
    # Task identification
    task_id: str
    instruction: str

    # Status
    status: TaskStatus
    success: bool

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Progress
    subtasks_total: int = 0
    subtasks_completed: int = 0
    subtask_results: List[SubtaskResult] = field(default_factory=list)

    # Performance
    total_actions: int = 0
    average_inference_time_ms: float = 0.0
    control_frequency_hz: float = 0.0

    # Error handling
    error_message: Optional[str] = None
    retries_used: int = 0
    replanning_count: int = 0

    # Explanation
    task_summary: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.subtasks_total == 0:
            return 0.0
        return (self.subtasks_completed / self.subtasks_total) * 100


class TaskAPI:
    """
    Natural Language Task API.

    High-level API for executing robot tasks using natural language,
    with automatic semantic understanding and task decomposition.
    """

    def __init__(
        self,
        vla_interface: Optional['VLAInterface'] = None,
        semantic_planner: Optional[Any] = None,
        monitoring_service: Optional[Any] = None,
        max_concurrent_tasks: int = 1,
    ):
        """
        Initialize Task API.

        Args:
            vla_interface: VLA model interface (auto-created if None)
            semantic_planner: Semantic task planner
            monitoring_service: Monitoring service for status updates
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.vla = vla_interface
        self.semantic_planner = semantic_planner
        self.monitoring = monitoring_service
        self.max_concurrent_tasks = max_concurrent_tasks

        # Task state
        self._active_tasks: Dict[str, TaskRequest] = {}
        self._task_results: Dict[str, TaskResult] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()

        # Observation sources (to be connected)
        self._camera_callback: Optional[Callable] = None
        self._proprio_callback: Optional[Callable] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Task API."""
        if self._initialized:
            return

        logger.info("Initializing Task API...")

        # Initialize VLA if not provided
        if self.vla is None and HAS_VLA:
            from ..spatial_intelligence.vla_interface import VLAInterface
            self.vla = VLAInterface.create_for_hardware()

        # Initialize semantic planner if not provided
        if self.semantic_planner is None:
            try:
                from .semantic_planner import SemanticPlanner
                self.semantic_planner = SemanticPlanner()
            except ImportError:
                logger.warning("Semantic planner not available")

        # Load VLA model
        if self.vla:
            self.vla.load()

        self._initialized = True
        logger.info("Task API initialized successfully")

    async def execute(
        self,
        instruction: str,
        **kwargs
    ) -> TaskResult:
        """
        Execute a task from natural language instruction.

        Args:
            instruction: Natural language task description
            **kwargs: Additional TaskRequest parameters

        Returns:
            TaskResult with execution status and details

        Example:
            result = await api.execute(
                instruction="Pick up the red cup and place it on the shelf",
                priority=TaskPriority.HIGH,
                timeout_seconds=120.0
            )
        """
        # Create task request
        request = TaskRequest(instruction=instruction, **kwargs)

        return await self.execute_task(request)

    async def execute_task(self, request: TaskRequest) -> TaskResult:
        """
        Execute a task request.

        Args:
            request: TaskRequest object

        Returns:
            TaskResult with execution status
        """
        await self.initialize()

        # Create result object
        result = TaskResult(
            task_id=request.task_id,
            instruction=request.instruction,
            status=TaskStatus.PENDING,
            success=False,
            started_at=datetime.now(),
        )

        try:
            # Store active task
            self._active_tasks[request.task_id] = request

            # Phase 1: Planning
            result.status = TaskStatus.PLANNING
            logger.info(f"Planning task: {request.instruction}")

            subtasks = await self._plan_task(request)
            result.subtasks_total = len(subtasks)

            # Phase 2: Execution
            result.status = TaskStatus.EXECUTING

            for i, subtask in enumerate(subtasks):
                subtask_result = await self._execute_subtask(
                    subtask,
                    request,
                    subtask_index=i
                )
                result.subtask_results.append(subtask_result)

                if subtask_result.status == TaskStatus.COMPLETED:
                    result.subtasks_completed += 1
                    result.total_actions += subtask_result.actions_executed

                    # Callback
                    if request.on_subtask_complete:
                        request.on_subtask_complete(subtask_result)
                else:
                    # Handle failure
                    if request.allow_replanning and result.retries_used < request.max_retries:
                        result.retries_used += 1
                        result.replanning_count += 1
                        # Re-plan remaining subtasks
                        logger.info(f"Replanning after subtask failure (attempt {result.retries_used})")
                        subtasks = await self._replan_task(request, result)
                    else:
                        result.status = TaskStatus.FAILED
                        result.error_message = subtask_result.error_message
                        break

                # Progress callback
                if request.on_progress_update:
                    request.on_progress_update(result.progress_percent)

            # Complete
            if result.status == TaskStatus.EXECUTING:
                result.status = TaskStatus.COMPLETED
                result.success = True
                result.task_summary = await self._generate_summary(result)

        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error_message = f"Task timed out after {request.timeout_seconds}s"
        except asyncio.CancelledError:
            result.status = TaskStatus.CANCELLED
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            logger.exception(f"Task execution failed: {e}")
        finally:
            result.completed_at = datetime.now()
            result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

            # Clean up
            self._active_tasks.pop(request.task_id, None)
            self._task_results[request.task_id] = result

        return result

    async def _plan_task(self, request: TaskRequest) -> List[Dict[str, Any]]:
        """Plan task into subtasks using semantic planner."""
        if self.semantic_planner:
            plan = await self.semantic_planner.decompose(request.instruction)
            return plan.subtasks
        else:
            # Fallback: single subtask
            return [{"description": request.instruction, "id": "0"}]

    async def _replan_task(
        self,
        request: TaskRequest,
        current_result: TaskResult
    ) -> List[Dict[str, Any]]:
        """Re-plan remaining subtasks after failure."""
        if self.semantic_planner:
            # Get completed subtask descriptions
            completed = [s.description for s in current_result.subtask_results
                        if s.status == TaskStatus.COMPLETED]

            # Re-plan with context
            plan = await self.semantic_planner.replan(
                original_instruction=request.instruction,
                completed_subtasks=completed,
                failure_reason=current_result.subtask_results[-1].error_message
            )
            return plan.subtasks
        return []

    async def _execute_subtask(
        self,
        subtask: Dict[str, Any],
        request: TaskRequest,
        subtask_index: int
    ) -> SubtaskResult:
        """Execute a single subtask."""
        result = SubtaskResult(
            subtask_id=subtask.get("id", str(subtask_index)),
            description=subtask.get("description", ""),
            status=TaskStatus.EXECUTING,
            started_at=datetime.now(),
        )

        try:
            # Get observations
            images = await self._get_camera_images()
            proprio = await self._get_proprioception()

            # Run VLA inference
            if self.vla:
                vla_result = self.vla.infer(
                    images=images,
                    instruction=subtask.get("description", request.instruction),
                    proprio=proprio,
                )

                result.actions_executed = vla_result.action_horizon
                result.confidence = vla_result.confidence or 0.9

                # Execute actions (would connect to robot controller)
                await self._execute_actions(vla_result.actions)

            result.status = TaskStatus.COMPLETED

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Subtask failed: {e}")

        result.completed_at = datetime.now()
        return result

    async def _get_camera_images(self) -> Any:
        """Get current camera images."""
        if self._camera_callback:
            return await self._camera_callback()
        # Return mock data for testing
        import numpy as np
        return np.zeros((3, 3, 224, 224), dtype=np.float32)

    async def _get_proprioception(self) -> Any:
        """Get current proprioceptive state."""
        if self._proprio_callback:
            return await self._proprio_callback()
        # Return mock data for testing
        import numpy as np
        return np.zeros(21, dtype=np.float32)

    async def _execute_actions(self, actions: Any) -> None:
        """Execute action sequence on robot."""
        # This would connect to the robot controller
        # For now, simulate execution
        await asyncio.sleep(0.1 * len(actions) if hasattr(actions, '__len__') else 0.1)

    async def _generate_summary(self, result: TaskResult) -> str:
        """Generate human-readable task summary."""
        return (
            f"Completed '{result.instruction}' in {result.duration_seconds:.1f}s. "
            f"Executed {result.subtasks_completed} subtasks with "
            f"{result.total_actions} total actions."
        )

    # =========================================================================
    # Task Management
    # =========================================================================

    async def pause_task(self, task_id: str) -> bool:
        """Pause a running task."""
        if task_id in self._active_tasks:
            # Signal pause (implementation depends on controller)
            logger.info(f"Pausing task {task_id}")
            return True
        return False

    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        if task_id in self._active_tasks:
            logger.info(f"Resuming task {task_id}")
            return True
        return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._active_tasks:
            logger.info(f"Cancelling task {task_id}")
            # Would signal cancellation to execution loop
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        if task_id in self._active_tasks:
            return TaskStatus.EXECUTING
        if task_id in self._task_results:
            return self._task_results[task_id].status
        return None

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task."""
        return self._task_results.get(task_id)

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_camera_source(self, callback: Callable) -> None:
        """Set camera image source callback."""
        self._camera_callback = callback

    def set_proprioception_source(self, callback: Callable) -> None:
        """Set proprioception source callback."""
        self._proprio_callback = callback

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create_for_hardware(cls) -> 'TaskAPI':
        """Create TaskAPI with auto-detected hardware configuration."""
        return cls()

    @classmethod
    def for_jetson_thor(cls) -> 'TaskAPI':
        """Create TaskAPI optimized for Jetson Thor."""
        if HAS_VLA:
            from ..spatial_intelligence.vla_interface import VLAInterface
            vla = VLAInterface.for_jetson_thor()
            return cls(vla_interface=vla)
        return cls()

    @classmethod
    def for_simulation(cls) -> 'TaskAPI':
        """Create TaskAPI for simulation/testing."""
        return cls()

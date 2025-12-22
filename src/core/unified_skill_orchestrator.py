"""
Dynamical.ai Unified Skill Orchestrator

This module unifies all skill-related logic into ONE coherent architecture:
- WHAT skill to use (MoE routing)
- WHICH robot executes (spatial routing)
- HOW to adapt (location context)
- WHEN to execute (timing tiers)

Replaces fragmented logic across:
- robot_skill_invoker.py (execution)
- moe_skill_router.py (skill selection)
- spatial_skill_router.py (robot assignment)
- skill_client.py (edge caching)

Architecture (Bottom-Up):
=========================

Layer 1: PERCEPTION (1000Hz safety, 100Hz control)
    ONVIF Cameras → DINOv3/SAM3 → Robot Detection + Object Detection
    └─► Updates RobotLocationTracker with robot positions

Layer 2: SPATIAL AWARENESS
    CameraWorkspaceMapper: Which cameras cover which physical zones
    RobotLocationTracker: Where each robot is (from Layer 1)
    └─► Provides location context for all decisions

Layer 3: TASK DECOMPOSITION (Cloud, async)
    Cloud LLM (GPT-4/Claude) → Decomposes natural language task
    "Pick up red cube and place on shelf" → [locate, approach, grasp, transport, place]
    └─► Returns skill sequence with dependencies

Layer 4: SKILL SELECTION (MoE, <10ms)
    MoESkillRouter: For each sub-task, select best skill(s)
    Task embedding → Gating network → Top-K skills + weights
    └─► Returns skill_ids and blend_weights

Layer 5: ROBOT ASSIGNMENT (Spatial, <1ms)
    TaskRouter: Which robot should execute?
    Strategies: NEAREST, LEAST_BUSY, CAPABILITY, ROUND_ROBIN
    └─► Returns robot_id based on location + capability

Layer 6: SKILL ADAPTATION (Location, <1ms)
    LocationAdaptiveSkillManager: Modify skill params for location
    Table height, shelf positions, obstacles, other robots
    └─► Returns location_params dict

Layer 7: EXECUTION (Edge, 200Hz real-time)
    EdgeSkillClient: Download skill if not cached, execute
    VLA model inference → Robot actions at 200Hz
    └─► Returns joint positions/velocities

Data Flow:
==========

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                              CLOUD                                        │
    │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐   │
    │  │  Task LLM  │    │   Skill    │    │    MoE     │    │  Location  │   │
    │  │ (Planner)  │───►│  Library   │───►│   Router   │    │  Database  │   │
    │  └────────────┘    └────────────┘    └─────┬──────┘    └──────┬─────┘   │
    │        │                                    │                  │         │
    │        │ skill_sequence                     │ skill_ids        │ configs │
    │        ▼                                    ▼                  ▼         │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │              UNIFIED SKILL ORCHESTRATOR (this module)            │   │
    │  │                                                                  │   │
    │  │  orchestrate_task(task, robot_id?) → SkillExecutionPlan          │   │
    │  │                                                                  │   │
    │  │  1. Decompose task → skill_sequence (from LLM or cache)          │   │
    │  │  2. For each skill:                                              │   │
    │  │     a. MoE route → skill_ids + weights                           │   │
    │  │     b. Assign robot → robot_id (if not specified)                │   │
    │  │     c. Adapt to location → location_params                       │   │
    │  │  3. Return execution plan                                        │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                    │                                     │
    └────────────────────────────────────┼─────────────────────────────────────┘
                                         │ gRPC (SkillExecutionPlan)
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                         EDGE (Jetson Thor)                                │
    │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐   │
    │  │   ONVIF    │    │   Skill    │    │    VLA     │    │   Robot    │   │
    │  │  Cameras   │───►│   Cache    │───►│  Executor  │───►│  Control   │   │
    │  └────────────┘    └────────────┘    └────────────┘    └────────────┘   │
    │       │                                     │                            │
    │       │ robot detections                    │ actions @ 200Hz            │
    │       ▼                                     ▼                            │
    │  ┌────────────┐                       ┌────────────┐                     │
    │  │  Location  │                       │   Safety   │                     │
    │  │  Tracker   │                       │   @ 1kHz   │                     │
    │  └────────────┘                       └────────────┘                     │
    └──────────────────────────────────────────────────────────────────────────┘

Key Decisions (ONE place for each):
===================================

1. WHAT SKILL? → MoESkillRouter.route()
   - Input: task_embedding
   - Output: skill_ids, weights
   - Logic: Learned gating network or cosine similarity fallback

2. WHICH ROBOT? → TaskRouter.assign_task()
   - Input: workspace_id, required_capabilities, target_position
   - Output: robot_id
   - Logic: NEAREST (default), LEAST_BUSY, CAPABILITY, ROUND_ROBIN

3. HOW TO ADAPT? → LocationAdaptiveSkillManager.get_adapted_params()
   - Input: skill_id, workspace_id
   - Output: location_params (table_height, obstacles, etc.)
   - Logic: Workspace config + auto-adaptation rules

4. WHEN TO EXECUTE? → Timing tier determines deadline
   - Real-time skills: 5ms deadline, 200Hz
   - Planning skills: 100ms-1s, async in cloud
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Import existing components (lazy import to avoid circular deps)
_moe_router = None
_skill_storage = None
_edge_client = None


def _get_moe_router():
    """Lazy import MoE router."""
    global _moe_router
    if _moe_router is None:
        try:
            from src.platform.cloud.moe_skill_router import MoESkillRouter
            _moe_router = MoESkillRouter(num_experts=16, embedding_dim=512)
        except ImportError:
            logger.warning("MoESkillRouter not available")
    return _moe_router


def _get_skill_storage():
    """Lazy import skill storage."""
    global _skill_storage
    if _skill_storage is None:
        try:
            from src.platform.cloud.moe_skill_router import EncryptedSkillStorage
            _skill_storage = EncryptedSkillStorage()
        except ImportError:
            logger.warning("EncryptedSkillStorage not available")
    return _skill_storage


def _get_edge_client():
    """Lazy import edge client."""
    global _edge_client
    if _edge_client is None:
        try:
            from src.platform.edge.skill_client import EdgeSkillClient
            _edge_client = EdgeSkillClient()
        except ImportError:
            logger.warning("EdgeSkillClient not available")
    return _edge_client


# =============================================================================
# Unified Data Models
# =============================================================================

class ExecutionTier(str, Enum):
    """Timing tier for skill execution."""
    REALTIME = "realtime"      # <5ms, 200Hz, on-device
    CONTROL = "control"        # <100ms, 10Hz, on-device
    PLANNING = "planning"      # <1s, async, cloud
    BACKGROUND = "background"  # >1s, async, cloud


class AssignmentStrategy(str, Enum):
    """Robot assignment strategy."""
    NEAREST = "nearest"
    LEAST_BUSY = "least_busy"
    CAPABILITY = "capability"
    ROUND_ROBIN = "round_robin"
    EXPLICIT = "explicit"  # Robot ID provided in request


@dataclass
class Workspace:
    """Physical workspace zone."""
    workspace_id: str
    name: str
    bounds_min: np.ndarray  # [x, y, z] meters
    bounds_max: np.ndarray  # [x, y, z] meters
    camera_ids: List[str] = field(default_factory=list)

    # Physical properties
    floor_height: float = 0.0
    ceiling_height: float = 3.0
    has_table: bool = False
    table_height: float = 0.75
    has_shelf: bool = False
    shelf_heights: List[float] = field(default_factory=list)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RobotState:
    """Current state of a robot."""
    robot_id: str
    workspace_id: Optional[str] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    visible_in_cameras: List[str] = field(default_factory=list)
    capabilities: set = field(default_factory=set)
    current_task_count: int = 0
    edge_device_ip: Optional[str] = None
    last_updated: float = 0.0


@dataclass
class SkillStep:
    """Single step in a skill execution plan."""
    step_id: int
    skill_ids: List[str]           # Skills to blend
    weights: List[float]           # MoE weights
    robot_id: str                  # Assigned robot
    workspace_id: Optional[str]    # Target workspace
    location_params: Dict[str, Any]  # Location-adapted parameters
    tier: ExecutionTier            # Timing tier
    depends_on: List[int] = field(default_factory=list)  # Step dependencies

    # Execution metadata
    estimated_duration_ms: float = 100.0
    deadline_ms: float = 100.0


@dataclass
class SkillExecutionPlan:
    """Complete execution plan for a task."""
    plan_id: str
    task_description: str
    steps: List[SkillStep]
    total_estimated_duration_ms: float
    created_at: float = field(default_factory=time.time)

    # Tracking
    status: str = "pending"  # pending, executing, completed, failed
    current_step: int = 0


@dataclass
class OrchestrationRequest:
    """Request to orchestrate a task."""
    task_description: str
    robot_id: Optional[str] = None  # Explicit robot, or None for auto-assign
    workspace_id: Optional[str] = None  # Target workspace
    target_position: Optional[np.ndarray] = None  # Target object position
    required_capabilities: Optional[set] = None
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.NEAREST
    tier: ExecutionTier = ExecutionTier.CONTROL
    max_skills_per_step: int = 3


@dataclass
class OrchestrationResult:
    """Result of task orchestration."""
    success: bool
    plan: Optional[SkillExecutionPlan] = None
    error_message: Optional[str] = None
    orchestration_time_ms: float = 0.0


# =============================================================================
# Unified Skill Orchestrator
# =============================================================================

class UnifiedSkillOrchestrator:
    """
    Single entry point for all skill orchestration.

    Combines:
    - MoE routing (what skill)
    - Spatial routing (which robot)
    - Location adaptation (how to execute)
    """

    def __init__(self):
        # Workspace registry
        self._workspaces: Dict[str, Workspace] = {}

        # Robot state tracking
        self._robots: Dict[str, RobotState] = {}

        # Camera to workspace mapping
        self._camera_to_workspaces: Dict[str, List[str]] = {}

        # Round-robin index for assignment
        self._rr_index = 0

        # Lock for thread safety
        self._lock = threading.RLock()

        # Skill embedding cache
        self._skill_embeddings: Dict[str, np.ndarray] = {}

        logger.info("UnifiedSkillOrchestrator initialized")

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure(self, config: Dict[str, Any]):
        """
        Configure orchestrator with site configuration.

        Args:
            config: Site config with workspaces, robots, cameras
        """
        with self._lock:
            # Register workspaces
            for ws_cfg in config.get("workspaces", []):
                workspace = Workspace(
                    workspace_id=ws_cfg["id"],
                    name=ws_cfg["name"],
                    bounds_min=np.array(ws_cfg.get("bounds_min", [0, 0, 0])),
                    bounds_max=np.array(ws_cfg.get("bounds_max", [1, 1, 1])),
                    camera_ids=ws_cfg.get("cameras", []),
                    floor_height=ws_cfg.get("floor_height", 0.0),
                    ceiling_height=ws_cfg.get("ceiling_height", 3.0),
                    has_table=ws_cfg.get("has_table", False),
                    table_height=ws_cfg.get("table_height", 0.75),
                    has_shelf=ws_cfg.get("has_shelf", False),
                    shelf_heights=ws_cfg.get("shelf_heights", []),
                )
                self._workspaces[workspace.workspace_id] = workspace

                # Update camera mapping
                for cam_id in workspace.camera_ids:
                    if cam_id not in self._camera_to_workspaces:
                        self._camera_to_workspaces[cam_id] = []
                    self._camera_to_workspaces[cam_id].append(workspace.workspace_id)

            # Register robots
            for robot_cfg in config.get("robots", []):
                robot = RobotState(
                    robot_id=robot_cfg["id"],
                    capabilities=set(robot_cfg.get("capabilities", [])),
                    edge_device_ip=robot_cfg.get("edge_ip"),
                )
                self._robots[robot.robot_id] = robot

        logger.info(f"Configured {len(self._workspaces)} workspaces, {len(self._robots)} robots")

    # =========================================================================
    # Location Tracking (from perception)
    # =========================================================================

    def update_robot_location(
        self,
        robot_id: str,
        position: np.ndarray,
        orientation: np.ndarray,
        camera_id: str,
    ):
        """
        Update robot location from perception.

        Called by perception pipeline when robot detected in camera.
        """
        with self._lock:
            if robot_id not in self._robots:
                self._robots[robot_id] = RobotState(robot_id=robot_id)

            robot = self._robots[robot_id]
            robot.position = position
            robot.orientation = orientation
            robot.last_updated = time.time()

            # Update visible cameras
            if camera_id not in robot.visible_in_cameras:
                robot.visible_in_cameras.append(camera_id)

            # Determine workspace from position
            robot.workspace_id = self._point_to_workspace(position)

    def _point_to_workspace(self, point: np.ndarray) -> Optional[str]:
        """Find workspace containing a 3D point."""
        for ws_id, ws in self._workspaces.items():
            if (np.all(point >= ws.bounds_min) and np.all(point <= ws.bounds_max)):
                return ws_id
        return None

    # =========================================================================
    # Main Orchestration Entry Point
    # =========================================================================

    def orchestrate(self, request: OrchestrationRequest) -> OrchestrationResult:
        """
        Main entry point: orchestrate a task into an execution plan.

        This is the ONE function that handles:
        1. Task decomposition (what needs to be done)
        2. Skill selection (which skills to use)
        3. Robot assignment (which robot executes)
        4. Location adaptation (how to adapt for location)

        Args:
            request: Orchestration request

        Returns:
            OrchestrationResult with execution plan
        """
        start_time = time.time()

        try:
            # Step 1: Decompose task into skill sequence
            skill_sequence = self._decompose_task(request.task_description)

            # Step 2: For each sub-task, create execution step
            steps = []
            for i, subtask in enumerate(skill_sequence):
                step = self._create_execution_step(
                    step_id=i,
                    subtask=subtask,
                    request=request,
                    previous_steps=steps,
                )
                steps.append(step)

            # Step 3: Build execution plan
            total_duration = sum(s.estimated_duration_ms for s in steps)

            plan = SkillExecutionPlan(
                plan_id=f"plan_{int(time.time() * 1000)}",
                task_description=request.task_description,
                steps=steps,
                total_estimated_duration_ms=total_duration,
            )

            orchestration_time = (time.time() - start_time) * 1000

            return OrchestrationResult(
                success=True,
                plan=plan,
                orchestration_time_ms=orchestration_time,
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return OrchestrationResult(
                success=False,
                error_message=str(e),
                orchestration_time_ms=(time.time() - start_time) * 1000,
            )

    def _decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose task into skill sequence.

        In production, this calls cloud LLM for complex decomposition.
        For real-time, uses cached decompositions or simple heuristics.
        """
        # Simple keyword-based decomposition (replace with LLM in production)
        subtasks = []

        task_lower = task_description.lower()

        # Detect common task patterns
        if "pick" in task_lower and "place" in task_lower:
            subtasks = [
                {"action": "locate", "object": self._extract_object(task_lower)},
                {"action": "approach", "target": "object"},
                {"action": "grasp", "object": self._extract_object(task_lower)},
                {"action": "transport", "destination": self._extract_destination(task_lower)},
                {"action": "place", "location": self._extract_destination(task_lower)},
            ]
        elif "pick" in task_lower or "grasp" in task_lower:
            subtasks = [
                {"action": "locate", "object": self._extract_object(task_lower)},
                {"action": "approach", "target": "object"},
                {"action": "grasp", "object": self._extract_object(task_lower)},
            ]
        elif "place" in task_lower or "put" in task_lower:
            subtasks = [
                {"action": "transport", "destination": self._extract_destination(task_lower)},
                {"action": "place", "location": self._extract_destination(task_lower)},
            ]
        elif "move" in task_lower or "go" in task_lower:
            subtasks = [
                {"action": "navigate", "destination": self._extract_destination(task_lower)},
            ]
        else:
            # Generic single-step task
            subtasks = [
                {"action": "execute", "task": task_description},
            ]

        return subtasks

    def _extract_object(self, task: str) -> str:
        """Extract object from task description."""
        # Simple extraction (replace with NER in production)
        words = task.split()
        for i, word in enumerate(words):
            if word in ["the", "a", "an"] and i + 1 < len(words):
                return words[i + 1]
        return "object"

    def _extract_destination(self, task: str) -> str:
        """Extract destination from task description."""
        if "table" in task:
            return "table"
        if "shelf" in task:
            return "shelf"
        if "box" in task:
            return "box"
        return "destination"

    def _create_execution_step(
        self,
        step_id: int,
        subtask: Dict[str, Any],
        request: OrchestrationRequest,
        previous_steps: List[SkillStep],
    ) -> SkillStep:
        """Create execution step for a subtask."""

        # 1. WHAT SKILL? - MoE routing
        skill_ids, weights = self._route_to_skills(
            subtask=subtask,
            max_skills=request.max_skills_per_step,
        )

        # 2. WHICH ROBOT? - Spatial routing
        robot_id = self._assign_robot(
            request=request,
            subtask=subtask,
        )

        # 3. WHERE? - Determine workspace
        workspace_id = self._determine_workspace(
            request=request,
            robot_id=robot_id,
        )

        # 4. HOW? - Location adaptation
        location_params = self._adapt_for_location(
            skill_ids=skill_ids,
            workspace_id=workspace_id,
        )

        # 5. WHEN? - Determine timing tier
        tier = self._determine_tier(subtask["action"])

        # Dependencies: each step depends on previous
        depends_on = [step_id - 1] if step_id > 0 else []

        return SkillStep(
            step_id=step_id,
            skill_ids=skill_ids,
            weights=weights,
            robot_id=robot_id,
            workspace_id=workspace_id,
            location_params=location_params,
            tier=tier,
            depends_on=depends_on,
            deadline_ms=5.0 if tier == ExecutionTier.REALTIME else 100.0,
        )

    # =========================================================================
    # Decision Functions (ONE place for each decision)
    # =========================================================================

    def _route_to_skills(
        self,
        subtask: Dict[str, Any],
        max_skills: int = 3,
    ) -> Tuple[List[str], List[float]]:
        """
        WHAT SKILL? - Route subtask to skills using MoE.

        This is the ONE place that decides which skills to use.
        """
        action = subtask.get("action", "")

        # Get MoE router
        router = _get_moe_router()
        storage = _get_skill_storage()

        if router and storage:
            # Generate task embedding
            task_text = f"{action} {subtask.get('object', '')} {subtask.get('destination', '')}"
            task_embedding = self._generate_embedding(task_text)

            # Get skill embeddings
            skill_embeddings = storage.get_skill_embeddings() if storage else {}

            if skill_embeddings:
                skill_ids, weights, _ = router.route(
                    task_embedding=task_embedding,
                    skill_embeddings=skill_embeddings,
                )
                return skill_ids[:max_skills], weights[:max_skills]

        # Fallback: map action to skill directly
        action_to_skill = {
            "locate": ["skill_detect_object"],
            "approach": ["skill_navigate_to"],
            "grasp": ["skill_grasp_object"],
            "transport": ["skill_carry_object"],
            "place": ["skill_place_object"],
            "navigate": ["skill_navigate_to"],
            "execute": ["skill_generic"],
        }

        skill_ids = action_to_skill.get(action, ["skill_generic"])
        weights = [1.0 / len(skill_ids)] * len(skill_ids)

        return skill_ids, weights

    def _assign_robot(
        self,
        request: OrchestrationRequest,
        subtask: Dict[str, Any],
    ) -> str:
        """
        WHICH ROBOT? - Assign robot based on strategy.

        This is the ONE place that decides which robot executes.
        """
        with self._lock:
            # Explicit assignment
            if request.robot_id:
                return request.robot_id

            # Get candidate robots
            candidates = list(self._robots.keys())

            # Filter by capabilities if required
            if request.required_capabilities:
                candidates = [
                    rid for rid in candidates
                    if request.required_capabilities.issubset(
                        self._robots[rid].capabilities
                    )
                ]

            # Filter by workspace if specified
            if request.workspace_id:
                candidates = [
                    rid for rid in candidates
                    if self._robots[rid].workspace_id == request.workspace_id
                ]

            if not candidates:
                # Fallback to any robot
                candidates = list(self._robots.keys())

            if not candidates:
                return "default_robot"

            # Apply strategy
            strategy = request.assignment_strategy

            if strategy == AssignmentStrategy.NEAREST and request.target_position is not None:
                return self._assign_nearest(candidates, request.target_position)

            elif strategy == AssignmentStrategy.LEAST_BUSY:
                return self._assign_least_busy(candidates)

            elif strategy == AssignmentStrategy.CAPABILITY:
                return self._assign_by_capability(candidates, request.required_capabilities)

            elif strategy == AssignmentStrategy.ROUND_ROBIN:
                return self._assign_round_robin(candidates)

            else:
                return candidates[0]

    def _assign_nearest(self, candidates: List[str], target: np.ndarray) -> str:
        """Assign to nearest robot."""
        min_dist = float('inf')
        nearest = candidates[0]

        for robot_id in candidates:
            robot = self._robots.get(robot_id)
            if robot:
                dist = np.linalg.norm(robot.position - target)
                if dist < min_dist:
                    min_dist = dist
                    nearest = robot_id

        return nearest

    def _assign_least_busy(self, candidates: List[str]) -> str:
        """Assign to robot with fewest tasks."""
        min_tasks = float('inf')
        least_busy = candidates[0]

        for robot_id in candidates:
            robot = self._robots.get(robot_id)
            if robot and robot.current_task_count < min_tasks:
                min_tasks = robot.current_task_count
                least_busy = robot_id

        return least_busy

    def _assign_by_capability(
        self,
        candidates: List[str],
        required: Optional[set],
    ) -> str:
        """Assign to robot with best capability match."""
        if not required:
            return candidates[0]

        best_score = -1
        best_robot = candidates[0]

        for robot_id in candidates:
            robot = self._robots.get(robot_id)
            if robot:
                score = len(robot.capabilities.intersection(required))
                if score > best_score:
                    best_score = score
                    best_robot = robot_id

        return best_robot

    def _assign_round_robin(self, candidates: List[str]) -> str:
        """Assign in round-robin fashion."""
        self._rr_index = (self._rr_index + 1) % len(candidates)
        return candidates[self._rr_index]

    def _determine_workspace(
        self,
        request: OrchestrationRequest,
        robot_id: str,
    ) -> Optional[str]:
        """Determine target workspace."""
        # Explicit workspace
        if request.workspace_id:
            return request.workspace_id

        # From target position
        if request.target_position is not None:
            return self._point_to_workspace(request.target_position)

        # From robot's current workspace
        robot = self._robots.get(robot_id)
        if robot:
            return robot.workspace_id

        return None

    def _adapt_for_location(
        self,
        skill_ids: List[str],
        workspace_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        HOW TO ADAPT? - Generate location-specific parameters.

        This is the ONE place that adapts skills for location.
        """
        params = {}

        if not workspace_id:
            return params

        workspace = self._workspaces.get(workspace_id)
        if not workspace:
            return params

        # Auto-adapt based on workspace properties
        params["floor_height"] = workspace.floor_height
        params["ceiling_height"] = workspace.ceiling_height
        params["max_reach_height"] = workspace.ceiling_height - 0.2

        if workspace.has_table:
            params["table_height"] = workspace.table_height
            params["default_pick_height"] = workspace.table_height

        if workspace.has_shelf:
            params["shelf_heights"] = workspace.shelf_heights

        if workspace.obstacles:
            params["obstacles"] = workspace.obstacles

        # Add other robots in workspace for collision avoidance
        other_robots = [
            rid for rid, r in self._robots.items()
            if r.workspace_id == workspace_id
        ]
        if other_robots:
            params["other_robots"] = other_robots

        return params

    def _determine_tier(self, action: str) -> ExecutionTier:
        """Determine timing tier for action."""
        realtime_actions = {"grasp", "place", "pour", "insert"}
        control_actions = {"approach", "transport", "navigate"}
        planning_actions = {"locate", "detect", "plan"}

        if action in realtime_actions:
            return ExecutionTier.REALTIME
        elif action in control_actions:
            return ExecutionTier.CONTROL
        elif action in planning_actions:
            return ExecutionTier.PLANNING
        else:
            return ExecutionTier.CONTROL

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding from text."""
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(512).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    # =========================================================================
    # Execution Interface
    # =========================================================================

    def execute_step(self, step: SkillStep, observation: np.ndarray) -> Dict[str, Any]:
        """
        Execute a single step of the plan.

        This sends the step to the appropriate edge device for execution.
        """
        edge_client = _get_edge_client()

        if not edge_client:
            logger.warning("Edge client not available - mock execution")
            return {
                "success": True,
                "action": np.zeros(23),
                "execution_time_ms": 5.0,
            }

        # Get robot's edge device
        robot = self._robots.get(step.robot_id)
        if not robot or not robot.edge_device_ip:
            return {"success": False, "error": "Robot edge device not found"}

        # Execute blended skills
        try:
            from src.platform.edge.skill_client import SkillBlendConfig

            blend_config = SkillBlendConfig(
                skill_ids=step.skill_ids,
                weights=step.weights,
                blend_mode="weighted_sum",
            )

            result = edge_client.execute_blended(blend_config, observation)

            return {
                "success": result.success,
                "action": result.output,
                "execution_time_ms": result.execution_time_ms,
            }

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {"success": False, "error": str(e)}

    def increment_task_count(self, robot_id: str):
        """Called when task assigned to robot."""
        with self._lock:
            if robot_id in self._robots:
                self._robots[robot_id].current_task_count += 1

    def decrement_task_count(self, robot_id: str):
        """Called when robot completes task."""
        with self._lock:
            if robot_id in self._robots:
                self._robots[robot_id].current_task_count = max(
                    0, self._robots[robot_id].current_task_count - 1
                )


# =============================================================================
# Global Instance
# =============================================================================

_orchestrator: Optional[UnifiedSkillOrchestrator] = None


def get_orchestrator() -> UnifiedSkillOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = UnifiedSkillOrchestrator()
    return _orchestrator


def configure_orchestrator(config: Dict[str, Any]) -> UnifiedSkillOrchestrator:
    """Configure global orchestrator."""
    orchestrator = get_orchestrator()
    orchestrator.configure(config)
    return orchestrator


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("UNIFIED SKILL ORCHESTRATOR")
    print("=" * 70)

    # Create and configure orchestrator
    orchestrator = get_orchestrator()

    config = {
        "workspaces": [
            {
                "id": "assembly_station_1",
                "name": "Assembly Station 1",
                "bounds_min": [0, 0, 0],
                "bounds_max": [2, 2, 2],
                "cameras": ["cam_01", "cam_02"],
                "has_table": True,
                "table_height": 0.8,
            },
            {
                "id": "storage_area",
                "name": "Storage Area",
                "bounds_min": [3, 0, 0],
                "bounds_max": [5, 2, 2],
                "cameras": ["cam_03"],
                "has_shelf": True,
                "shelf_heights": [0.5, 1.0, 1.5],
            },
        ],
        "robots": [
            {
                "id": "thor_001",
                "capabilities": ["manipulation", "locomotion"],
                "edge_ip": "192.168.1.10",
            },
            {
                "id": "thor_002",
                "capabilities": ["manipulation"],
                "edge_ip": "192.168.1.11",
            },
        ],
    }

    orchestrator.configure(config)

    # Simulate robot detection
    orchestrator.update_robot_location(
        robot_id="thor_001",
        position=np.array([1.0, 1.0, 0.0]),
        orientation=np.array([0, 0, 0, 1]),
        camera_id="cam_01",
    )

    print("\n1. Orchestrate Task")
    print("-" * 50)

    request = OrchestrationRequest(
        task_description="Pick up the red cube from the table and place it on the shelf",
        assignment_strategy=AssignmentStrategy.NEAREST,
        target_position=np.array([1.2, 0.8, 0.8]),
    )

    result = orchestrator.orchestrate(request)

    if result.success:
        print(f"   Plan ID: {result.plan.plan_id}")
        print(f"   Total duration: {result.plan.total_estimated_duration_ms:.0f}ms")
        print(f"   Orchestration time: {result.orchestration_time_ms:.2f}ms")
        print(f"\n   Steps ({len(result.plan.steps)}):")

        for step in result.plan.steps:
            print(f"      Step {step.step_id}:")
            print(f"         Skills: {step.skill_ids}")
            print(f"         Robot: {step.robot_id}")
            print(f"         Workspace: {step.workspace_id}")
            print(f"         Tier: {step.tier.value}")
            print(f"         Location params: {step.location_params}")
    else:
        print(f"   Failed: {result.error_message}")

    print("\n" + "=" * 70)
    print("UNIFIED ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
    ONE entry point:  orchestrator.orchestrate(request)

    ONE decision for each:
    - WHAT skill?   → _route_to_skills()  [MoE gating]
    - WHICH robot?  → _assign_robot()     [Spatial routing]
    - HOW to adapt? → _adapt_for_location() [Location params]
    - WHEN?         → _determine_tier()   [Timing tier]

    Data flow:
    Task → Decompose → [For each subtask: Route → Assign → Adapt] → Plan
    """)

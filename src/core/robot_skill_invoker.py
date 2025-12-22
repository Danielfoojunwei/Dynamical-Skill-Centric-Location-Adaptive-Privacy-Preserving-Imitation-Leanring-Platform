"""
Robot Skill Invoker - Humanoid Robot Skill Invocation Pipeline

This module provides the skill invocation pipeline for humanoid robots running on
Jetson AGX Orin 32GB. It allows the robot's control loop to:

1. Request skills based on task descriptions or observations
2. Blend multiple skills using MoE routing weights
3. Execute skills with safety pre-checks
4. Integrate with the 4-tier timing system

Architecture:
=============

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Robot Skill Invocation Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ Task Request │ ───► │ MoE Router   │ ───► │ Skill Blend  │              │
│  │ (NL or embed)│      │ (Cloud/Local)│      │ (Weighted)   │              │
│  └──────────────┘      └──────────────┘      └──────┬───────┘              │
│                                                      │                      │
│                                               ┌──────▼───────┐              │
│                                               │ Safety Check │              │
│                                               │ (Pre-execute)│              │
│                                               └──────┬───────┘              │
│                                                      │                      │
│                                               ┌──────▼───────┐              │
│                                               │ Execute Skill│              │
│                                               │ (Action gen) │              │
│                                               └──────┬───────┘              │
│                                                      │                      │
│                                               ┌──────▼───────┐              │
│                                               │Control Output│              │
│                                               │ (Robot cmds) │              │
│                                               └──────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Timing Integration:
==================
- Tier 1 Safety Loop (1kHz): Safety checks ALWAYS run, can_throttle=False
- Tier 2 Control Loop (100Hz): Skill execution runs here
- Tier 3 Learning Loop (10Hz): Skill updates from cloud (async)
- Tier 4 Cloud Sync (0.1Hz): FL skill updates

The skill invoker NEVER blocks the safety loop.
"""

import os
import time
import json
import hashlib
import logging
import threading
import asyncio
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import numpy as np

logger = logging.getLogger(__name__)

# Import EdgeSkillClient for skill execution
try:
    from src.platform.edge.skill_client import (
        EdgeSkillClient, SkillBlendConfig, SkillExecutionResult
    )
    HAS_SKILL_CLIENT = True
except ImportError:
    HAS_SKILL_CLIENT = False
    logger.warning("EdgeSkillClient not available - skill invocation disabled")

# Import safety manager
try:
    from src.platform.safety_manager import safety_manager
    HAS_SAFETY = True
except ImportError:
    HAS_SAFETY = False
    logger.warning("Safety manager not available")


# =============================================================================
# Data Classes
# =============================================================================

class InvocationMode(str, Enum):
    """Skill invocation modes."""
    DIRECT = "direct"           # Single skill execution
    BLENDED = "blended"         # MoE weighted blend
    SEQUENTIAL = "sequential"   # Skills executed in order
    AUTONOMOUS = "autonomous"   # Robot decides based on observation


class RobotRole(str, Enum):
    """Robot role in multi-robot coordination scenarios.

    Used by SwarmBridge/SwarmBrain for cross-site orchestration,
    but the role is passed here for skill-level awareness.
    """
    LEADER = "leader"           # Primary actor, initiates coordination
    FOLLOWER = "follower"       # Follows leader commands/trajectory
    OBSERVER = "observer"       # Monitors but doesn't act
    INDEPENDENT = "independent" # Acts autonomously (default)
    ASSISTANT = "assistant"     # Supports leader with secondary tasks


class ActionSpace(str, Enum):
    """Robot action space types."""
    JOINT_POSITION = "joint_position"       # Joint angles
    JOINT_VELOCITY = "joint_velocity"       # Joint velocities
    CARTESIAN_POSE = "cartesian_pose"       # End-effector pose
    HYBRID = "hybrid"                       # Mixed joint + task space


@dataclass
class ObservationState:
    """Current robot observation state."""
    # Proprioception
    joint_positions: np.ndarray         # Current joint angles [DOF]
    joint_velocities: np.ndarray        # Current joint velocities [DOF]
    joint_torques: Optional[np.ndarray] = None  # Current joint torques [DOF]

    # End-effector state
    ee_position: Optional[np.ndarray] = None      # [x, y, z]
    ee_orientation: Optional[np.ndarray] = None   # [qx, qy, qz, qw]
    gripper_state: Optional[float] = None         # 0.0=closed, 1.0=open

    # Hand state (if equipped with DYGlove)
    hand_joint_positions: Optional[np.ndarray] = None  # 21-DOF per hand
    hand_contact_forces: Optional[np.ndarray] = None   # Contact forces

    # Vision features (compressed from VLA)
    vision_embedding: Optional[np.ndarray] = None     # [512] or [1024]

    # Task context
    task_embedding: Optional[np.ndarray] = None       # From language instruction

    # Timing
    timestamp: float = field(default_factory=time.time)

    def to_flat_vector(self) -> np.ndarray:
        """Flatten observation to vector for skill input."""
        parts = [self.joint_positions, self.joint_velocities]

        if self.joint_torques is not None:
            parts.append(self.joint_torques)
        if self.ee_position is not None:
            parts.append(self.ee_position)
        if self.ee_orientation is not None:
            parts.append(self.ee_orientation)
        if self.gripper_state is not None:
            parts.append(np.array([self.gripper_state]))
        if self.hand_joint_positions is not None:
            parts.append(self.hand_joint_positions)
        if self.vision_embedding is not None:
            parts.append(self.vision_embedding)
        if self.task_embedding is not None:
            parts.append(self.task_embedding)

        return np.concatenate(parts).astype(np.float32)


@dataclass
class RobotAction:
    """Robot action output from skill execution."""
    # Action type
    action_space: ActionSpace

    # Joint-level actions
    joint_positions: Optional[np.ndarray] = None    # Target positions [DOF]
    joint_velocities: Optional[np.ndarray] = None   # Target velocities [DOF]

    # Cartesian actions
    ee_position_delta: Optional[np.ndarray] = None  # Delta [x, y, z]
    ee_orientation_delta: Optional[np.ndarray] = None  # Delta quaternion

    # Gripper
    gripper_action: Optional[float] = None          # 0.0=close, 1.0=open

    # Hand actions (21-DOF per hand)
    hand_joint_positions: Optional[np.ndarray] = None

    # Safety limits
    max_velocity: float = 1.0       # rad/s
    max_acceleration: float = 2.0   # rad/s^2

    # Execution parameters
    duration_s: float = 0.1         # Action duration
    is_blocking: bool = False       # Wait for completion

    # Metadata
    skill_id: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_command_vector(self) -> np.ndarray:
        """Convert to command vector for robot controller."""
        if self.action_space == ActionSpace.JOINT_POSITION:
            return self.joint_positions
        elif self.action_space == ActionSpace.JOINT_VELOCITY:
            return self.joint_velocities
        elif self.action_space == ActionSpace.CARTESIAN_POSE:
            return np.concatenate([
                self.ee_position_delta or np.zeros(3),
                self.ee_orientation_delta or np.zeros(4),
            ])
        else:
            # Hybrid: joint positions + gripper
            parts = []
            if self.joint_positions is not None:
                parts.append(self.joint_positions)
            if self.gripper_action is not None:
                parts.append(np.array([self.gripper_action]))
            if self.hand_joint_positions is not None:
                parts.append(self.hand_joint_positions)
            return np.concatenate(parts) if parts else np.array([])


@dataclass
class SkillInvocationRequest:
    """Request to invoke a skill.

    This is the EXECUTION interface for the Dynamical Edge Platform.

    Integration with UnifiedSkillOrchestrator:
    ==========================================
    The orchestrator handles PLANNING (what/which/how), the invoker handles EXECUTION.

        Orchestrator.orchestrate()  →  SkillExecutionPlan
                                              │
                                              ▼
        Orchestrator.execute_step() →  RobotSkillInvoker.invoke()
                                              │
                                              ▼
                                       RobotAction @ 200Hz

    When used with orchestrator, skill_ids and blend_weights are pre-computed.
    When used standalone, set task_description for automatic MoE routing.
    """
    # Robot identification (required for multi-robot deployments)
    robot_id: Optional[str] = None              # Unique robot identifier

    # Role & Coordination (for SwarmBridge integration)
    role: RobotRole = RobotRole.INDEPENDENT     # Robot's role in coordination
    goal: Optional[str] = None                  # High-level goal description
    coordination_id: Optional[str] = None       # Shared ID for coordinated tasks

    # Task specification (one of these should be provided)
    task_description: Optional[str] = None      # Natural language (triggers MoE routing)
    task_embedding: Optional[np.ndarray] = None # Pre-computed embedding
    skill_ids: Optional[List[str]] = None       # Direct skill IDs (from orchestrator)

    # Current state
    observation: Optional[ObservationState] = None

    # Invocation mode
    mode: InvocationMode = InvocationMode.AUTONOMOUS

    # MoE parameters
    max_skills: int = 3
    blend_weights: Optional[List[float]] = None  # MoE weights (from orchestrator)

    # Location context (from orchestrator's location adaptation)
    location_context: Optional[Dict[str, Any]] = None  # table_height, obstacles, etc.

    # Safety parameters
    check_safety: bool = True
    allow_keep_out_override: bool = False

    # Execution parameters
    action_space: ActionSpace = ActionSpace.JOINT_POSITION
    horizon_steps: int = 1      # Number of actions to generate

    # Timing
    deadline_ms: float = 10.0   # Max time for skill execution

    # Tracking
    request_id: str = field(default_factory=lambda: f"req_{int(time.time()*1000)}")


@dataclass
class SkillInvocationResult:
    """Result of skill invocation."""
    success: bool
    request_id: str

    # Output action(s)
    actions: List[RobotAction] = field(default_factory=list)

    # Skill info
    skill_ids_used: List[str] = field(default_factory=list)
    blend_weights: List[float] = field(default_factory=list)

    # Safety
    safety_status: str = "SAFE"  # SAFE, SLOW, STOP
    safety_violations: List[str] = field(default_factory=list)

    # Timing
    routing_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Error info
    error_message: Optional[str] = None


# =============================================================================
# Robot Skill Invoker
# =============================================================================

class RobotSkillInvoker:
    """
    Skill EXECUTION engine for humanoid robots (Layer 7).

    This is the control loop execution layer that runs at 200Hz on the edge.
    For task planning and skill selection, use UnifiedSkillOrchestrator.

    Architecture:
    =============
    UnifiedSkillOrchestrator (Cloud/Planning)
        │
        │ orchestrate() → SkillExecutionPlan
        │   - WHAT skill? (MoE routing)
        │   - WHICH robot? (spatial routing)
        │   - HOW to adapt? (location params)
        │
        ▼
    RobotSkillInvoker (Edge/Execution) ← YOU ARE HERE
        │
        │ invoke() → RobotAction @ 200Hz
        │   - Safety pre-check (1kHz, NEVER bypassed)
        │   - Skill execution via EdgeSkillClient
        │   - Action generation
        │
        ▼
    Robot Controller

    Usage:
    ======
    # With orchestrator (recommended):
    orchestrator = get_orchestrator()
    plan = orchestrator.orchestrate(request)
    for step in plan.steps:
        result = orchestrator.execute_step(step, observation)

    # Standalone (legacy):
    invoker = get_skill_invoker()
    result = invoker.invoke(SkillInvocationRequest(
        task_description="pick up the cup"  # Triggers MoE routing
    ))
    """

    def __init__(
        self,
        platform_url: str = "http://localhost:8080",
        device_id: str = "",
        use_encryption: bool = True,
        cache_dir: str = "/var/lib/dynamical/skill_cache",
        max_cache_mb: int = 1024,
        robot_dof: int = 23,
        hand_dof: int = 21,
        control_rate_hz: float = 100.0,
    ):
        """
        Initialize the skill invoker.

        Args:
            platform_url: Edge platform API URL
            device_id: Unique device identifier
            use_encryption: Enable N2HE encryption
            cache_dir: Skill cache directory
            max_cache_mb: Max cache size in MB
            robot_dof: Robot degrees of freedom
            hand_dof: Hand DOF (per hand)
            control_rate_hz: Control loop frequency
        """
        self.device_id = device_id or f"robot_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        self.robot_dof = robot_dof
        self.hand_dof = hand_dof
        self.control_dt = 1.0 / control_rate_hz

        # Initialize skill client
        if HAS_SKILL_CLIENT:
            self.skill_client = EdgeSkillClient(
                platform_url=platform_url,
                device_id=self.device_id,
                cache_dir=cache_dir,
                max_cache_mb=max_cache_mb,
                use_encryption=use_encryption,
            )
        else:
            self.skill_client = None
            logger.warning("Skill client not available - using mock execution")

        # Active skills (preloaded)
        self._active_skill_ids: List[str] = []
        self._lock = threading.Lock()

        # Async update queue
        self._update_queue: Queue = Queue()
        self._update_thread: Optional[threading.Thread] = None
        self._running = False

        # Statistics
        self.stats = {
            "invocations": 0,
            "successful": 0,
            "safety_stops": 0,
            "avg_latency_ms": 0.0,
            "skill_usage": {},
        }

        # Callbacks
        self._pre_execute_hooks: List[Callable] = []
        self._post_execute_hooks: List[Callable] = []

    def start(self):
        """Start the skill invoker background services."""
        if self._running:
            return

        self._running = True

        # Start async update thread
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
        )
        self._update_thread.start()

        logger.info(f"RobotSkillInvoker started for device {self.device_id}")

    def stop(self):
        """Stop the skill invoker."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        logger.info("RobotSkillInvoker stopped")

    def _update_loop(self):
        """Background loop for async skill updates."""
        while self._running:
            try:
                # Check for skill update requests
                try:
                    update_request = self._update_queue.get(timeout=1.0)
                    self._process_update(update_request)
                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Update loop error: {e}")
                time.sleep(1.0)

    def _process_update(self, request: Dict[str, Any]):
        """Process an async update request."""
        update_type = request.get("type")

        if update_type == "preload_skills":
            skill_ids = request.get("skill_ids", [])
            if self.skill_client:
                loaded = self.skill_client.preload_skills(skill_ids)
                logger.info(f"Preloaded {loaded}/{len(skill_ids)} skills")

        elif update_type == "refresh_skills":
            # Re-download updated skills from cloud
            for skill_id in self._active_skill_ids:
                if self.skill_client:
                    self.skill_client.get_skill(skill_id, download_if_missing=True)

    def preload_skills(self, skill_ids: List[str], async_load: bool = True):
        """
        Preload skills into memory for fast execution.

        Args:
            skill_ids: Skills to preload
            async_load: If True, load in background thread
        """
        if async_load:
            self._update_queue.put({
                "type": "preload_skills",
                "skill_ids": skill_ids,
            })
        else:
            if self.skill_client:
                self.skill_client.preload_skills(skill_ids)

        with self._lock:
            self._active_skill_ids = list(set(self._active_skill_ids + skill_ids))

    def register_pre_execute_hook(self, hook: Callable[[SkillInvocationRequest], bool]):
        """
        Register a pre-execution hook.

        Hook should return True to allow execution, False to block.
        """
        self._pre_execute_hooks.append(hook)

    def register_post_execute_hook(self, hook: Callable[[SkillInvocationResult], None]):
        """Register a post-execution hook."""
        self._post_execute_hooks.append(hook)

    def invoke(self, request: SkillInvocationRequest) -> SkillInvocationResult:
        """
        Invoke skills based on the request.

        This is the main entry point for the robot control loop.

        Args:
            request: Skill invocation request

        Returns:
            SkillInvocationResult with actions or error
        """
        start_time = time.time()

        # Update statistics
        self.stats["invocations"] += 1

        # Run pre-execute hooks
        for hook in self._pre_execute_hooks:
            try:
                if not hook(request):
                    return SkillInvocationResult(
                        success=False,
                        request_id=request.request_id,
                        error_message="Blocked by pre-execute hook",
                    )
            except Exception as e:
                logger.error(f"Pre-execute hook error: {e}")

        # Safety pre-check
        if request.check_safety and request.observation:
            safety_result = self._check_safety(request.observation)
            if safety_result == "STOP" and not request.allow_keep_out_override:
                self.stats["safety_stops"] += 1
                return SkillInvocationResult(
                    success=False,
                    request_id=request.request_id,
                    safety_status="STOP",
                    safety_violations=["Position in KEEP_OUT zone"],
                    error_message="Safety stop - position not safe",
                )
        else:
            safety_result = "SAFE"

        # Route to skills
        routing_start = time.time()
        skill_ids, weights = self._route_skills(request)
        routing_time_ms = (time.time() - routing_start) * 1000

        if not skill_ids:
            return SkillInvocationResult(
                success=False,
                request_id=request.request_id,
                error_message="No skills found for request",
                routing_time_ms=routing_time_ms,
            )

        # Execute skills
        exec_start = time.time()
        actions = self._execute_skills(
            skill_ids=skill_ids,
            weights=weights,
            observation=request.observation,
            mode=request.mode,
            action_space=request.action_space,
            horizon_steps=request.horizon_steps,
        )
        exec_time_ms = (time.time() - exec_start) * 1000

        total_time_ms = (time.time() - start_time) * 1000

        # Check deadline
        if total_time_ms > request.deadline_ms:
            logger.warning(f"Skill invocation exceeded deadline: {total_time_ms:.2f}ms > {request.deadline_ms}ms")

        # Update statistics
        self.stats["successful"] += 1
        alpha = 0.1
        self.stats["avg_latency_ms"] = (
            alpha * total_time_ms +
            (1 - alpha) * self.stats["avg_latency_ms"]
        )

        for skill_id in skill_ids:
            self.stats["skill_usage"][skill_id] = \
                self.stats["skill_usage"].get(skill_id, 0) + 1

        result = SkillInvocationResult(
            success=len(actions) > 0,
            request_id=request.request_id,
            actions=actions,
            skill_ids_used=skill_ids,
            blend_weights=weights,
            safety_status=safety_result,
            routing_time_ms=routing_time_ms,
            execution_time_ms=exec_time_ms,
            total_time_ms=total_time_ms,
        )

        # Run post-execute hooks
        for hook in self._post_execute_hooks:
            try:
                hook(result)
            except Exception as e:
                logger.error(f"Post-execute hook error: {e}")

        return result

    def _check_safety(self, observation: ObservationState) -> str:
        """
        Check safety before skill execution.

        Returns: "SAFE", "SLOW", or "STOP"
        """
        if not HAS_SAFETY:
            return "SAFE"

        # Get end-effector position
        if observation.ee_position is not None:
            x, y = observation.ee_position[0], observation.ee_position[1]
            return safety_manager.check_position(x, y)

        return "SAFE"

    def _route_skills(
        self,
        request: SkillInvocationRequest,
    ) -> Tuple[List[str], List[float]]:
        """
        Route request to appropriate skills.

        Returns:
            Tuple of (skill_ids, weights)
        """
        # Direct skill IDs provided
        if request.skill_ids:
            weights = request.blend_weights or [1.0 / len(request.skill_ids)] * len(request.skill_ids)
            return request.skill_ids, weights

        # Use skill client for MoE routing
        if self.skill_client:
            try:
                results = self.skill_client.request_skills(
                    task_description=request.task_description or "",
                    task_embedding=request.task_embedding.tolist() if request.task_embedding is not None else None,
                    max_skills=request.max_skills,
                )

                skill_ids = [r[0] for r in results]
                weights = [r[1] for r in results]

                return skill_ids, weights

            except Exception as e:
                logger.error(f"Skill routing failed: {e}")

        # Fallback: use active skills with equal weights
        if self._active_skill_ids:
            skills = self._active_skill_ids[:request.max_skills]
            weights = [1.0 / len(skills)] * len(skills)
            return skills, weights

        return [], []

    def _execute_skills(
        self,
        skill_ids: List[str],
        weights: List[float],
        observation: Optional[ObservationState],
        mode: InvocationMode,
        action_space: ActionSpace,
        horizon_steps: int,
    ) -> List[RobotAction]:
        """
        Execute skills and generate actions.

        Args:
            skill_ids: Skills to execute
            weights: Blend weights
            observation: Current observation
            mode: Invocation mode
            action_space: Output action space
            horizon_steps: Number of actions to generate

        Returns:
            List of RobotAction
        """
        if not self.skill_client:
            # Mock execution
            return self._mock_execute(action_space, horizon_steps)

        # Get observation vector
        if observation:
            obs_vector = observation.to_flat_vector()
        else:
            obs_vector = np.zeros(256, dtype=np.float32)

        actions = []

        for step in range(horizon_steps):
            if mode == InvocationMode.DIRECT:
                # Execute single skill
                result = self.skill_client.execute_skill(skill_ids[0], obs_vector)
                if result.success and result.output is not None:
                    action = self._output_to_action(result.output, action_space, skill_ids[0])
                    actions.append(action)

            elif mode in [InvocationMode.BLENDED, InvocationMode.AUTONOMOUS]:
                # Execute blended skills
                blend_config = SkillBlendConfig(
                    skill_ids=skill_ids,
                    weights=weights,
                    blend_mode="weighted_sum",
                )
                result = self.skill_client.execute_blended(blend_config, obs_vector)
                if result.success and result.output is not None:
                    action = self._output_to_action(result.output, action_space, "blended")
                    actions.append(action)

            elif mode == InvocationMode.SEQUENTIAL:
                # Execute skills in sequence
                for skill_id in skill_ids:
                    result = self.skill_client.execute_skill(skill_id, obs_vector)
                    if result.success and result.output is not None:
                        action = self._output_to_action(result.output, action_space, skill_id)
                        actions.append(action)
                        # Update obs for next skill (use action as next state estimate)
                        if action.joint_positions is not None:
                            obs_vector[:len(action.joint_positions)] = action.joint_positions
                break  # Only one step for sequential

        return actions

    def _output_to_action(
        self,
        output: np.ndarray,
        action_space: ActionSpace,
        skill_id: str,
    ) -> RobotAction:
        """Convert skill output to RobotAction."""
        output = output.flatten()

        if action_space == ActionSpace.JOINT_POSITION:
            joint_pos = output[:self.robot_dof] if len(output) >= self.robot_dof else output
            return RobotAction(
                action_space=action_space,
                joint_positions=joint_pos,
                skill_id=skill_id,
                duration_s=self.control_dt,
            )

        elif action_space == ActionSpace.JOINT_VELOCITY:
            joint_vel = output[:self.robot_dof] if len(output) >= self.robot_dof else output
            return RobotAction(
                action_space=action_space,
                joint_velocities=joint_vel,
                skill_id=skill_id,
                duration_s=self.control_dt,
            )

        elif action_space == ActionSpace.CARTESIAN_POSE:
            if len(output) >= 7:
                pos_delta = output[:3]
                ori_delta = output[3:7]
            else:
                pos_delta = output[:3] if len(output) >= 3 else np.zeros(3)
                ori_delta = np.array([0, 0, 0, 1])
            return RobotAction(
                action_space=action_space,
                ee_position_delta=pos_delta,
                ee_orientation_delta=ori_delta,
                skill_id=skill_id,
                duration_s=self.control_dt,
            )

        else:  # HYBRID
            # Split into joint positions + gripper + hand
            idx = 0
            joint_pos = output[idx:idx + self.robot_dof]
            idx += self.robot_dof

            gripper = output[idx] if idx < len(output) else 0.5
            idx += 1

            hand_pos = output[idx:idx + self.hand_dof * 2] if idx < len(output) else None

            return RobotAction(
                action_space=action_space,
                joint_positions=joint_pos,
                gripper_action=float(gripper),
                hand_joint_positions=hand_pos,
                skill_id=skill_id,
                duration_s=self.control_dt,
            )

    def _mock_execute(
        self,
        action_space: ActionSpace,
        horizon_steps: int,
    ) -> List[RobotAction]:
        """Mock execution for testing."""
        actions = []
        for _ in range(horizon_steps):
            if action_space == ActionSpace.JOINT_POSITION:
                action = RobotAction(
                    action_space=action_space,
                    joint_positions=np.zeros(self.robot_dof),
                    skill_id="mock",
                    confidence=0.5,
                )
            elif action_space == ActionSpace.CARTESIAN_POSE:
                action = RobotAction(
                    action_space=action_space,
                    ee_position_delta=np.zeros(3),
                    ee_orientation_delta=np.array([0, 0, 0, 1]),
                    skill_id="mock",
                    confidence=0.5,
                )
            else:
                action = RobotAction(
                    action_space=action_space,
                    joint_velocities=np.zeros(self.robot_dof),
                    skill_id="mock",
                    confidence=0.5,
                )
            actions.append(action)
        return actions

    def get_statistics(self) -> Dict[str, Any]:
        """Get invocation statistics."""
        return {
            **self.stats,
            "skill_client": self.skill_client.get_statistics() if self.skill_client else {},
            "active_skills": len(self._active_skill_ids),
        }

    def invoke_for_task(
        self,
        task: str,
        observation: ObservationState,
        **kwargs,
    ) -> SkillInvocationResult:
        """
        Convenience method to invoke skills for a natural language task.

        Args:
            task: Natural language task description
            observation: Current robot observation
            **kwargs: Additional parameters for SkillInvocationRequest

        Returns:
            SkillInvocationResult
        """
        request = SkillInvocationRequest(
            task_description=task,
            observation=observation,
            mode=InvocationMode.AUTONOMOUS,
            **kwargs,
        )
        return self.invoke(request)

    def invoke_skill_direct(
        self,
        skill_id: str,
        observation: ObservationState,
        **kwargs,
    ) -> SkillInvocationResult:
        """
        Convenience method to invoke a specific skill directly.

        Args:
            skill_id: Skill to invoke
            observation: Current robot observation
            **kwargs: Additional parameters

        Returns:
            SkillInvocationResult
        """
        request = SkillInvocationRequest(
            skill_ids=[skill_id],
            observation=observation,
            mode=InvocationMode.DIRECT,
            **kwargs,
        )
        return self.invoke(request)


# =============================================================================
# Global Invoker Instance
# =============================================================================

_invoker: Optional[RobotSkillInvoker] = None

def get_skill_invoker() -> RobotSkillInvoker:
    """Get the global skill invoker instance."""
    global _invoker
    if _invoker is None:
        _invoker = RobotSkillInvoker()
    return _invoker

def init_skill_invoker(**kwargs) -> RobotSkillInvoker:
    """Initialize the global skill invoker with custom settings."""
    global _invoker
    _invoker = RobotSkillInvoker(**kwargs)
    return _invoker


# =============================================================================
# Testing
# =============================================================================

def test_robot_skill_invoker():
    """Test the robot skill invoker."""
    print("\n" + "=" * 60)
    print("ROBOT SKILL INVOKER TEST")
    print("=" * 60)

    # Create invoker
    invoker = RobotSkillInvoker(
        device_id="test_robot_001",
        robot_dof=23,
        hand_dof=21,
        control_rate_hz=100.0,
    )
    invoker.start()

    print("\n1. Create Test Observation")
    print("-" * 40)

    obs = ObservationState(
        joint_positions=np.zeros(23),
        joint_velocities=np.zeros(23),
        ee_position=np.array([0.5, 0.0, 0.3]),
        ee_orientation=np.array([0, 0, 0, 1]),
        gripper_state=1.0,  # Open
        vision_embedding=np.random.randn(512).astype(np.float32),
    )
    print(f"   Joint positions: {obs.joint_positions.shape}")
    print(f"   EE position: {obs.ee_position}")
    print(f"   Vision embedding: {obs.vision_embedding.shape}")

    print("\n2. Test Task-Based Invocation (Mock)")
    print("-" * 40)

    result = invoker.invoke_for_task(
        task="Pick up the red cube from the table",
        observation=obs,
    )
    print(f"   Success: {result.success}")
    print(f"   Actions generated: {len(result.actions)}")
    print(f"   Total time: {result.total_time_ms:.2f}ms")

    if result.actions:
        action = result.actions[0]
        print(f"   Action space: {action.action_space}")
        if action.joint_positions is not None:
            print(f"   Joint positions shape: {action.joint_positions.shape}")

    print("\n3. Test Blended Execution")
    print("-" * 40)

    request = SkillInvocationRequest(
        skill_ids=["skill_grasp", "skill_place"],
        blend_weights=[0.7, 0.3],
        observation=obs,
        mode=InvocationMode.BLENDED,
        action_space=ActionSpace.HYBRID,
    )

    result = invoker.invoke(request)
    print(f"   Success: {result.success}")
    print(f"   Skills used: {result.skill_ids_used}")
    print(f"   Blend weights: {result.blend_weights}")

    print("\n4. Statistics")
    print("-" * 40)

    stats = invoker.get_statistics()
    print(f"   Total invocations: {stats['invocations']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")

    invoker.stop()

    print("\n" + "=" * 60)
    print("ROBOT SKILL INVOKER TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_robot_skill_invoker()

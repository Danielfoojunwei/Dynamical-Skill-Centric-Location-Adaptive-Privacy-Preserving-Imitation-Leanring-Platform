"""
Deep Imitative Learning Integration Layer

This module integrates the three core components of the Deep Imitative
Learning stack with Pi0.5:

1. **Diffusion Planner**: Refines VLA action proposals into smooth trajectories
2. **RIP Gating**: Epistemic uncertainty for safety decisions
3. **POIR Recovery**: Plans return to training distribution when OOD

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Pi0.5 VLA (OpenPI)                        │
    │         Semantic understanding + action proposals           │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   DIFFUSION PLANNER                          │
    │              Smooth trajectory generation                    │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      RIP GATING                              │
    │      Epistemic uncertainty → Safe to execute?                │
    └─────────────────────────────────────────────────────────────┘
                     │                        │
          ┌─────────┴─────────┐    ┌─────────┴─────────┐
          ▼ (Safe)             ▼ (OOD)
    ┌──────────────┐    ┌──────────────────────────────┐
    │   Execute    │    │        POIR RECOVERY          │
    │   Actions    │    │   Plan return to distribution │
    └──────────────┘    └──────────────────────────────┘

Usage:
    from src.spatial_intelligence.deep_imitative_learning import (
        DeepImitativeLearning,
        DILConfig,
    )

    # Create integrated pipeline
    dil = DeepImitativeLearning.create_for_hardware()

    # Execute with full safety pipeline
    result = await dil.execute(
        instruction="pick up the red cup",
        images=camera_images,
        proprio=robot_state,
    )

    if result.is_safe:
        robot.execute(result.actions)
    else:
        # Recovery is automatically planned
        await dil.execute_recovery()
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

# Import components
try:
    from .pi0 import Pi05Model, Pi05Config, Pi05Observation, Pi05Result, HAS_OPENPI
except ImportError:
    HAS_OPENPI = False
    Pi05Model = None
    Pi05Config = None
    Pi05Observation = None
    Pi05Result = None

try:
    from .planning import DiffusionPlanner, DiffusionConfig, TrajectoryBatch
except ImportError:
    DiffusionPlanner = None
    DiffusionConfig = None
    TrajectoryBatch = None

try:
    from .safety import RIPGating, RIPConfig, SafetyDecision, RiskLevel
except ImportError:
    RIPGating = None
    RIPConfig = None
    SafetyDecision = None
    RiskLevel = None

try:
    from .recovery import POIRRecovery, POIRConfig, RecoveryPlan, RecoveryStatus
except ImportError:
    POIRRecovery = None
    POIRConfig = None
    RecoveryPlan = None
    RecoveryStatus = None

try:
    from .visual_trace import (
        VisualTraceRenderer,
        VisualTraceConfig,
        VisualTrace,
        TraceModifier,
    )
    HAS_VISUAL_TRACE = True
except ImportError:
    HAS_VISUAL_TRACE = False
    VisualTraceRenderer = None
    VisualTraceConfig = None
    VisualTrace = None
    TraceModifier = None


class ExecutionMode(Enum):
    """Execution mode for the pipeline."""
    NORMAL = "normal"           # Full pipeline with safety
    FAST = "fast"               # Skip diffusion refinement
    SAFE_ONLY = "safe_only"     # Extra conservative, lower thresholds
    RECOVERY = "recovery"       # Currently in recovery mode


@dataclass
class DILConfig:
    """Configuration for Deep Imitative Learning pipeline."""
    # Component configs
    pi05_variant: str = "pi05_base"
    action_dim: int = 7
    action_horizon: int = 16

    # Diffusion planner
    use_diffusion: bool = True
    diffusion_steps: int = 50  # Reduced for speed
    num_trajectory_samples: int = 4

    # RIP safety
    use_safety_gating: bool = True
    ensemble_size: int = 5
    safe_threshold: float = 0.1
    critical_threshold: float = 0.6

    # POIR recovery
    use_recovery: bool = True
    max_recovery_steps: int = 50

    # Hardware
    device: str = "cuda"

    # Performance
    target_hz: float = 10.0  # Target control frequency

    @classmethod
    def for_jetson_thor(cls) -> "DILConfig":
        """Optimized config for Jetson Thor."""
        return cls(
            pi05_variant="pi05_base",
            device="cuda",
            diffusion_steps=30,  # Faster
            num_trajectory_samples=4,
        )

    @classmethod
    def for_development(cls) -> "DILConfig":
        """Development config (CPU, no heavy models)."""
        return cls(
            device="cpu",
            use_diffusion=False,  # Skip for speed
            ensemble_size=3,
        )


@dataclass
class DILResult:
    """Result from Deep Imitative Learning pipeline."""
    # Actions to execute
    actions: Any  # [H, A] action trajectory

    # Safety assessment
    is_safe: bool
    risk_level: Optional[Any] = None  # RiskLevel enum
    confidence: float = 1.0

    # Component outputs
    vla_actions: Optional[Any] = None  # Raw Pi0.5 output
    refined_trajectory: Optional[Any] = None  # Diffusion output
    safety_decision: Optional[Any] = None  # RIP output

    # Visual trace (MolmoAct-inspired steerability)
    visual_trace: Optional[Any] = None  # VisualTrace for operator preview

    # Recovery info
    needs_recovery: bool = False
    recovery_plan: Optional[Any] = None

    # Performance
    total_time_ms: float = 0.0
    vla_time_ms: float = 0.0
    diffusion_time_ms: float = 0.0
    safety_time_ms: float = 0.0
    trace_time_ms: float = 0.0  # Visual trace rendering time


class DeepImitativeLearning:
    """
    Deep Imitative Learning Pipeline.

    Integrates Pi0.5 VLA with:
    - Diffusion Planner for trajectory refinement
    - RIP Gating for safety assessment
    - POIR Recovery for OOD handling

    This provides a complete, safety-aware imitation learning system.
    """

    def __init__(self, config: Optional[DILConfig] = None):
        self.config = config or DILConfig()

        # Initialize components
        self.vla: Optional[Pi05Model] = None
        self.diffusion: Optional[DiffusionPlanner] = None
        self.safety: Optional[RIPGating] = None
        self.recovery: Optional[POIRRecovery] = None
        
        # Visual trace renderer (MolmoAct-inspired)
        self.trace_renderer: Optional[VisualTraceRenderer] = None
        self.trace_modifier: Optional[TraceModifier] = None
        self._current_trace: Optional[VisualTrace] = None

        # State
        self.mode = ExecutionMode.NORMAL
        self._loaded = False

        # History for recovery
        self.observation_history: List[Any] = []
        self.action_history: List[Any] = []

        # Stats
        self.stats = {
            "total_executions": 0,
            "safe_executions": 0,
            "recoveries_triggered": 0,
            "avg_total_time_ms": 0.0,
            "traces_generated": 0,
        }

    def load(self):
        """Load all components."""
        if self._loaded:
            return

        logger.info("Loading Deep Imitative Learning pipeline...")

        # Load Pi0.5
        if HAS_OPENPI and Pi05Model is not None:
            pi05_config = Pi05Config(
                variant=self.config.pi05_variant,
                device=self.config.device,
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
            )
            self.vla = Pi05Model(pi05_config)
            self.vla.load()
            logger.info("Loaded Pi0.5 VLA")
        else:
            logger.warning("Pi0.5 not available")

        # Load Diffusion Planner
        if self.config.use_diffusion and DiffusionPlanner is not None:
            diff_config = DiffusionConfig(
                action_dim=self.config.action_dim,
                horizon=self.config.action_horizon,
                num_diffusion_steps=self.config.diffusion_steps,
                num_samples=self.config.num_trajectory_samples,
                device=self.config.device,
            )
            self.diffusion = DiffusionPlanner(diff_config)
            self.diffusion.load_model()
            logger.info("Loaded Diffusion Planner")

        # Load RIP Safety
        if self.config.use_safety_gating and RIPGating is not None:
            rip_config = RIPConfig(
                ensemble_size=self.config.ensemble_size,
                action_dim=self.config.action_dim,
                safe_threshold=self.config.safe_threshold,
                warning_threshold=self.config.critical_threshold,
                device=self.config.device,
            )
            self.safety = RIPGating(rip_config)
            self.safety.load_model()
            logger.info("Loaded RIP Safety Gating")

        # Initialize POIR Recovery
        if self.config.use_recovery and POIRRecovery is not None:
            poir_config = POIRConfig(
                action_dim=self.config.action_dim,
                max_recovery_steps=self.config.max_recovery_steps,
                device=self.config.device,
            )
            self.recovery = POIRRecovery(poir_config)

            # Connect RIP to POIR for OOD evaluation
            if self.safety is not None:
                self.recovery.set_ood_evaluator(
                    lambda obs: self.safety.evaluate(obs).ood_score
                )
            logger.info("Initialized POIR Recovery")

        # Initialize Visual Trace Renderer (MolmoAct-inspired)
        if HAS_VISUAL_TRACE and VisualTraceRenderer is not None:
            self.trace_renderer = VisualTraceRenderer(VisualTraceConfig())
            self.trace_modifier = TraceModifier(self.trace_renderer)
            logger.info("Initialized Visual Trace Renderer")

        self._loaded = True
        logger.info("Deep Imitative Learning pipeline ready")

    def execute(
        self,
        instruction: str,
        images: Any,
        proprio: Optional[Any] = None,
        scene_features: Optional[Any] = None,
    ) -> DILResult:
        """
        Execute the full Deep Imitative Learning pipeline.

        Args:
            instruction: Natural language task instruction
            images: Camera images [N, H, W, C]
            proprio: Proprioceptive state [P]
            scene_features: Optional pre-computed DINOv3 features

        Returns:
            DILResult with actions and safety assessment
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        self.stats["total_executions"] += 1

        # Step 1: Pi0.5 VLA inference
        vla_start = time.time()
        vla_actions = self._vla_inference(instruction, images, proprio)
        vla_time = (time.time() - vla_start) * 1000

        # Step 2: Diffusion refinement
        diff_start = time.time()
        if self.config.use_diffusion and self.diffusion is not None:
            trajectory_batch = self.diffusion.plan(
                initial_actions=vla_actions,
                condition=scene_features,
            )
            refined_actions = trajectory_batch.best.actions
        else:
            refined_actions = vla_actions
            trajectory_batch = None
        diff_time = (time.time() - diff_start) * 1000

        # Step 3: Safety assessment
        safety_start = time.time()
        if self.config.use_safety_gating and self.safety is not None:
            # Use scene features if available, otherwise use flattened images
            obs_for_safety = scene_features if scene_features is not None else images
            safety_decision = self.safety.evaluate(
                observation=obs_for_safety,
                proprio=proprio,
                proposed_action=refined_actions,
            )
        else:
            safety_decision = None
        safety_time = (time.time() - safety_start) * 1000

        # Determine if safe and if recovery needed
        is_safe = True
        needs_recovery = False
        recovery_plan = None

        if safety_decision is not None:
            is_safe = safety_decision.is_safe
            needs_recovery = safety_decision.should_trigger_recovery

            if needs_recovery and self.recovery is not None:
                # Plan recovery
                recovery_plan = self.recovery.plan_recovery(
                    current_observation=obs_for_safety,
                    ood_score=safety_decision.ood_score,
                )
                self.stats["recoveries_triggered"] += 1
                self.mode = ExecutionMode.RECOVERY

        # Update history
        self._update_history(images, refined_actions)

        # Step 4: Generate visual trace (MolmoAct-inspired steerability)
        trace_start = time.time()
        visual_trace = None
        if self.trace_renderer is not None and trajectory_batch is not None:
            visual_trace = self.trace_renderer.render_trace(
                trajectory=trajectory_batch.best,
                image=images if HAS_NUMPY and hasattr(images, 'shape') else None,
            )
            self._current_trace = visual_trace
            self.stats["traces_generated"] += 1
        trace_time = (time.time() - trace_start) * 1000

        # Update stats
        total_time = (time.time() - start_time) * 1000
        if is_safe:
            self.stats["safe_executions"] += 1
        self.stats["avg_total_time_ms"] = (
            self.stats["avg_total_time_ms"] * (self.stats["total_executions"] - 1) +
            total_time
        ) / self.stats["total_executions"]

        return DILResult(
            actions=recovery_plan.trajectory if needs_recovery and recovery_plan else refined_actions,
            is_safe=is_safe,
            risk_level=safety_decision.risk_level if safety_decision else None,
            confidence=safety_decision.confidence if safety_decision else 1.0,
            vla_actions=vla_actions,
            refined_trajectory=trajectory_batch,
            safety_decision=safety_decision,
            visual_trace=visual_trace,
            needs_recovery=needs_recovery,
            recovery_plan=recovery_plan,
            total_time_ms=total_time,
            vla_time_ms=vla_time,
            diffusion_time_ms=diff_time,
            safety_time_ms=safety_time,
            trace_time_ms=trace_time,
        )

    def _vla_inference(
        self,
        instruction: str,
        images: Any,
        proprio: Optional[Any],
    ) -> Any:
        """Run Pi0.5 inference."""
        if self.vla is not None:
            obs = Pi05Observation(
                images=images,
                instruction=instruction,
                proprio=proprio,
            )
            result = self.vla.infer(obs)
            return result.actions

        # Mock if VLA not available
        if HAS_NUMPY:
            return np.random.randn(
                self.config.action_horizon,
                self.config.action_dim
            ).astype(np.float32) * 0.1
        return [[0.0] * self.config.action_dim] * self.config.action_horizon

    def _update_history(self, observation: Any, actions: Any):
        """Update observation and action history."""
        self.observation_history.append(observation)
        self.action_history.append(actions)

        # Limit history
        max_history = 100
        if len(self.observation_history) > max_history:
            self.observation_history.pop(0)
            self.action_history.pop(0)

        # Also update recovery module
        if self.recovery is not None:
            self.recovery.record_step(observation, actions)

    def step_recovery(self, current_observation: Any) -> Tuple[Optional[Any], bool]:
        """
        Execute one step of recovery.

        Returns:
            (action, is_complete)
        """
        if self.recovery is None or not self.recovery.is_recovering:
            self.mode = ExecutionMode.NORMAL
            return None, True

        action, is_complete = self.recovery.step_recovery(current_observation)

        if is_complete:
            self.mode = ExecutionMode.NORMAL

        return action, is_complete

    def abort_recovery(self):
        """Abort current recovery and return to normal mode."""
        if self.recovery is not None:
            self.recovery.abort_recovery()
        self.mode = ExecutionMode.NORMAL

    def set_safe_waypoints(self, waypoints: List[Any]):
        """Set safe waypoints for recovery planning."""
        if self.recovery is not None:
            for wp in waypoints:
                self.recovery.add_safe_waypoint(wp)

    def set_home_position(self, position: Any):
        """Set home position for recovery."""
        if self.recovery is not None:
            self.recovery.set_home_position(position)

    # =========================================================================
    # Visual Trace Methods (MolmoAct-Inspired Steerability)
    # =========================================================================

    def get_visual_trace(self) -> Optional[Any]:
        """
        Get the current visual trace for operator preview.
        
        Returns:
            VisualTrace with waypoints and overlay, or None if not available
        """
        return self._current_trace

    def modify_trace(self, command: str) -> Optional[Any]:
        """
        Modify the current trace using natural language.
        
        Args:
            command: Natural language command (e.g., "move middle waypoint left")
            
        Returns:
            Modified VisualTrace, or None if no trace available
        """
        if self._current_trace is None or self.trace_modifier is None:
            logger.warning("No trace available to modify")
            return None
        
        self._current_trace = self.trace_modifier.modify_trace(
            self._current_trace,
            command,
        )
        return self._current_trace

    def get_modified_trajectory(self) -> Optional[Any]:
        """
        Get the trajectory from modified visual trace.
        
        Use this to execute operator-corrected trajectory.
        
        Returns:
            Modified action trajectory [H, A], or None if no modifications
        """
        if self._current_trace is None:
            return None
        return self._current_trace.get_modified_trajectory()

    def reset_trace(self) -> Optional[Any]:
        """
        Reset visual trace modifications to original.
        
        Returns:
            Reset VisualTrace, or None if no trace available
        """
        if self._current_trace is None or self.trace_modifier is None:
            return None
        
        self._current_trace = self.trace_modifier.reset_trace(self._current_trace)
        return self._current_trace

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_recovering(self) -> bool:
        return self.mode == ExecutionMode.RECOVERY

    @classmethod
    def create_for_hardware(cls) -> "DeepImitativeLearning":
        """Create with auto-detected hardware config."""
        # Try to detect Jetson Thor
        try:
            with open("/proc/device-tree/model", "r") as f:
                if "thor" in f.read().lower():
                    return cls(DILConfig.for_jetson_thor())
        except FileNotFoundError:
            pass

        return cls(DILConfig())

    @classmethod
    def for_jetson_thor(cls) -> "DeepImitativeLearning":
        """Create optimized for Jetson Thor."""
        return cls(DILConfig.for_jetson_thor())

    @classmethod
    def for_development(cls) -> "DeepImitativeLearning":
        """Create for development/testing."""
        return cls(DILConfig.for_development())

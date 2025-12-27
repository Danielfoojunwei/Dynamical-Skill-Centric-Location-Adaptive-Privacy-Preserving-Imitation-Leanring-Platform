"""
Action Reasoning Model (ARM) Pipeline

Unified pipeline integrating all MolmoAct-inspired components:
1. Depth Tokenization (perception tokens)
2. Trajectory Prediction (spatial planning)
3. User Steerability (human-in-the-loop)
4. Action Decoding (embodiment-specific)
5. Action Reasoning (chain-of-thought)

This addresses ALL MolmoAct gaps:
- Interpretability: Trajectory visualization + reasoning traces
- Steerability: User guidance integration
- Cross-Robot Transfer: Embodiment-agnostic planning
- Performance: Structured 3-stage pipeline

Usage:
    pipeline = ARMPipeline.create_for_robot("ur10e")

    result = pipeline.execute(
        images=camera_images,
        instruction="pick up the red cup",
        depth_map=depth_image,
        user_guidance=optional_guidance,
    )

    # Visualize the plan
    visualization = result.visualize()

    # Get robot actions
    actions = result.decoded_actions.joint_actions
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Local imports
from .trajectory_trace import TrajectoryTrace, TrajectoryVisualizationConfig
from .depth_tokenizer import DepthVQVAE, DepthTokenizerConfig, DepthTokens
from .trajectory_predictor import TrajectoryPredictor, TrajectoryPredictorConfig
from .action_decoder import ActionDecoder, ActionDecoderConfig, DecodedActions
from .robot_registry import RobotConfig, RobotRegistry, CameraConfig
from .steerability import UserGuidance, GuidanceMode
from .action_reasoning import (
    ActionReasoningModule, ActionReasoningConfig, ReasoningOutput,
    TemplateReasoningGenerator, attach_reasoning_to_trace
)

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class ARMConfig:
    """Configuration for ARM pipeline."""
    # Robot configuration
    robot_id: str = "ur10e"

    # Component configs
    depth_tokenizer_config: DepthTokenizerConfig = field(
        default_factory=DepthTokenizerConfig
    )
    trajectory_predictor_config: TrajectoryPredictorConfig = field(
        default_factory=TrajectoryPredictorConfig
    )
    action_decoder_config: ActionDecoderConfig = field(
        default_factory=ActionDecoderConfig
    )
    visualization_config: TrajectoryVisualizationConfig = field(
        default_factory=TrajectoryVisualizationConfig
    )

    # Feature flags
    use_depth_tokenization: bool = True
    use_trajectory_prediction: bool = True
    use_action_reasoning: bool = True
    use_steerability: bool = True

    # Fallbacks
    fallback_to_template_reasoning: bool = True

    # Model paths
    depth_tokenizer_checkpoint: Optional[str] = None
    trajectory_predictor_checkpoint: Optional[str] = None
    reasoning_module_checkpoint: Optional[str] = None

    # Device
    device: str = "cuda"

    # Performance
    target_fps: float = 10.0


@dataclass
class ARMResult:
    """Complete result from ARM pipeline."""
    # Core outputs
    trajectory_trace: TrajectoryTrace
    decoded_actions: DecodedActions
    reasoning: ReasoningOutput

    # Input echoes
    instruction: str
    robot_id: str

    # Intermediate results
    depth_tokens: Optional[DepthTokens] = None
    user_guidance_applied: bool = False

    # Timing
    total_time_ms: float = 0.0
    depth_time_ms: float = 0.0
    trajectory_time_ms: float = 0.0
    decoding_time_ms: float = 0.0
    reasoning_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Check if pipeline succeeded."""
        return (
            self.trajectory_trace is not None and
            self.decoded_actions is not None and
            self.decoded_actions.success_rate > 0.5
        )

    @property
    def confidence(self) -> float:
        """Overall confidence in the result."""
        trace_conf = self.trajectory_trace.mean_confidence
        action_conf = self.decoded_actions.success_rate
        reasoning_conf = min(
            self.reasoning.perception_confidence,
            self.reasoning.spatial_confidence,
            self.reasoning.action_confidence,
        )
        return (trace_conf + action_conf + reasoning_conf) / 3

    def visualize(
        self,
        include_reasoning: bool = True,
        config: Optional[TrajectoryVisualizationConfig] = None,
    ) -> np.ndarray:
        """
        Create comprehensive visualization.

        Args:
            include_reasoning: Include reasoning panel
            config: Visualization config

        Returns:
            BGR image
        """
        config = config or TrajectoryVisualizationConfig()

        if include_reasoning:
            return self.trajectory_trace.visualize_with_reasoning(config)
        else:
            return self.trajectory_trace.visualize(config)

    def get_summary(self) -> str:
        """Get text summary of result."""
        return (
            f"ARM Result for '{self.instruction}' on {self.robot_id}\n"
            f"  Trajectory: {self.trajectory_trace.num_waypoints} waypoints, "
            f"confidence={self.trajectory_trace.mean_confidence:.2f}\n"
            f"  Actions: {self.decoded_actions.action_horizon} steps, "
            f"IK success={self.decoded_actions.success_rate:.1%}\n"
            f"  Reasoning: {self.reasoning.action_reasoning[:50]}...\n"
            f"  Total time: {self.total_time_ms:.1f}ms"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "instruction": self.instruction,
            "robot_id": self.robot_id,
            "trajectory_trace": self.trajectory_trace.to_dict(),
            "joint_actions": self.decoded_actions.joint_actions.tolist(),
            "reasoning": self.reasoning.to_dict(),
            "confidence": self.confidence,
            "timing": {
                "total_ms": self.total_time_ms,
                "depth_ms": self.depth_time_ms,
                "trajectory_ms": self.trajectory_time_ms,
                "decoding_ms": self.decoding_time_ms,
                "reasoning_ms": self.reasoning_time_ms,
            },
        }


class ARMPipeline:
    """
    Complete Action Reasoning Model Pipeline.

    Integrates all MolmoAct-inspired components into a unified pipeline
    for interpretable, steerable, cross-robot manipulation.
    """

    def __init__(self, config: Optional[ARMConfig] = None):
        self.config = config or ARMConfig()

        # Get robot configuration
        self.robot_config = RobotRegistry.instance().get_or_raise(
            self.config.robot_id
        )

        # Initialize components (lazy loading)
        self._depth_tokenizer: Optional[DepthVQVAE] = None
        self._trajectory_predictor: Optional[TrajectoryPredictor] = None
        self._action_decoder: Optional[ActionDecoder] = None
        self._reasoning_module: Optional[ActionReasoningModule] = None
        self._template_reasoning: Optional[TemplateReasoningGenerator] = None

        # State
        self._initialized = False

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "avg_trajectory_confidence": 0.0,
            "avg_ik_success_rate": 0.0,
            "avg_total_time_ms": 0.0,
        }

    def initialize(self) -> bool:
        """
        Initialize all pipeline components.

        Returns:
            True if successful
        """
        if self._initialized:
            return True

        logger.info(f"Initializing ARM pipeline for {self.config.robot_id}")

        try:
            # Initialize depth tokenizer
            if self.config.use_depth_tokenization and HAS_TORCH:
                logger.info("Loading depth tokenizer...")
                self._depth_tokenizer = DepthVQVAE(
                    self.config.depth_tokenizer_config
                )
                if self.config.depth_tokenizer_checkpoint:
                    self._depth_tokenizer.load_pretrained(
                        self.config.depth_tokenizer_checkpoint
                    )
                self._depth_tokenizer.to(self.config.device)
                self._depth_tokenizer.eval()

            # Initialize trajectory predictor
            if self.config.use_trajectory_prediction and HAS_TORCH:
                logger.info("Loading trajectory predictor...")
                self._trajectory_predictor = TrajectoryPredictor(
                    self.config.trajectory_predictor_config
                )
                if self.config.trajectory_predictor_checkpoint:
                    self._trajectory_predictor.load_pretrained(
                        self.config.trajectory_predictor_checkpoint
                    )
                self._trajectory_predictor.to(self.config.device)
                self._trajectory_predictor.eval()

            # Initialize action decoder
            logger.info("Initializing action decoder...")
            self._action_decoder = ActionDecoder(
                robot_config=self.robot_config,
                config=self.config.action_decoder_config,
            )

            # Initialize reasoning
            if self.config.use_action_reasoning:
                if HAS_TORCH and self.config.reasoning_module_checkpoint:
                    logger.info("Loading reasoning module...")
                    self._reasoning_module = ActionReasoningModule(
                        ActionReasoningConfig()
                    )
                    self._reasoning_module.load_pretrained(
                        self.config.reasoning_module_checkpoint
                    )
                    self._reasoning_module.to(self.config.device)
                    self._reasoning_module.eval()
                else:
                    logger.info("Using template-based reasoning")
                    self._template_reasoning = TemplateReasoningGenerator()

            self._initialized = True
            logger.info("ARM pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ARM pipeline: {e}")
            return False

    def execute(
        self,
        images: np.ndarray,
        instruction: str,
        depth_map: Optional[np.ndarray] = None,
        vision_features: Optional[np.ndarray] = None,
        instruction_embedding: Optional[np.ndarray] = None,
        current_joint_positions: Optional[np.ndarray] = None,
        user_guidance: Optional[UserGuidance] = None,
        camera_config: Optional[CameraConfig] = None,
    ) -> ARMResult:
        """
        Execute the complete ARM pipeline.

        Args:
            images: Camera images [H, W, 3] or [N, H, W, 3]
            instruction: Natural language task description
            depth_map: Optional depth image [H, W]
            vision_features: Pre-computed vision features
            instruction_embedding: Pre-computed instruction embedding
            current_joint_positions: Current robot configuration
            user_guidance: Optional user guidance
            camera_config: Camera configuration

        Returns:
            ARMResult with trajectory, actions, and reasoning
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        self.stats["total_executions"] += 1

        # Ensure image is 3D
        if images.ndim == 4:
            image = images[0]
        else:
            image = images

        # Stage 1: Depth Tokenization
        depth_start = time.time()
        depth_tokens = None

        if self._depth_tokenizer is not None and depth_map is not None:
            depth_tensor = torch.from_numpy(depth_map).float().to(self.config.device)
            depth_tokens = self._depth_tokenizer.encode(depth_tensor)

        depth_time = (time.time() - depth_start) * 1000

        # Stage 2: Trajectory Prediction
        trajectory_start = time.time()

        if self._trajectory_predictor is not None:
            # Prepare inputs
            if vision_features is None:
                # Use mock features if not provided
                vision_features = np.random.randn(1024).astype(np.float32)

            if instruction_embedding is None:
                # Use mock embedding if not provided
                instruction_embedding = np.random.randn(512).astype(np.float32)

            vision_tensor = torch.from_numpy(vision_features).float().to(self.config.device)
            if vision_tensor.ndim == 1:
                vision_tensor = vision_tensor.unsqueeze(0).unsqueeze(0)
            elif vision_tensor.ndim == 2:
                vision_tensor = vision_tensor.unsqueeze(0)

            instr_tensor = torch.from_numpy(instruction_embedding).float().to(self.config.device)
            if instr_tensor.ndim == 1:
                instr_tensor = instr_tensor.unsqueeze(0)

            depth_embeddings = None
            if depth_tokens is not None and depth_tokens.embeddings is not None:
                depth_embeddings = torch.from_numpy(
                    depth_tokens.embeddings
                ).float().to(self.config.device)
                if depth_embeddings.ndim == 3:
                    depth_embeddings = depth_embeddings.unsqueeze(0)

            trajectory_trace = self._trajectory_predictor.predict(
                vision_tensor,
                instr_tensor,
                depth_embeddings,
                source_image=image,
                instruction_text=instruction,
            )
        else:
            # Create mock trajectory for testing
            trajectory_trace = self._create_mock_trajectory(image, instruction)

        trajectory_time = (time.time() - trajectory_start) * 1000

        # Stage 2.5: Apply User Guidance (Steerability)
        guidance_applied = False
        if user_guidance is not None and self.config.use_steerability:
            trajectory_trace = user_guidance.apply_to_trace(trajectory_trace)
            guidance_applied = True

        # Stage 3: Action Decoding
        decoding_start = time.time()

        decoded_actions = self._action_decoder.decode(
            trace=trajectory_trace,
            current_joint_positions=current_joint_positions,
            camera_config=camera_config or self.robot_config.primary_camera,
            depth_map=depth_map,
        )

        decoding_time = (time.time() - decoding_start) * 1000

        # Stage 4: Action Reasoning
        reasoning_start = time.time()

        if self._reasoning_module is not None:
            # Use neural reasoning
            reasoning = self._reasoning_module.generate_reasoning(
                vision_tensor, instr_tensor,
                depth_embeddings, instruction,
            )
        elif self._template_reasoning is not None:
            # Use template reasoning
            reasoning = self._template_reasoning.generate(
                trajectory_trace,
                target_distance=(
                    float(trajectory_trace.waypoint_depths.mean())
                    if trajectory_trace.waypoint_depths is not None
                    else None
                ),
            )
        else:
            # Minimal reasoning
            reasoning = ReasoningOutput(
                perception_reasoning="Analyzing scene",
                spatial_reasoning="Computing spatial relationships",
                action_reasoning=f"Executing {trajectory_trace.num_waypoints}-waypoint trajectory",
            )

        reasoning_time = (time.time() - reasoning_start) * 1000

        # Attach reasoning to trace
        attach_reasoning_to_trace(trajectory_trace, reasoning)

        total_time = (time.time() - start_time) * 1000

        # Update statistics
        if decoded_actions.success_rate > 0.5:
            self.stats["successful_executions"] += 1

        alpha = 0.1
        self.stats["avg_trajectory_confidence"] = (
            alpha * trajectory_trace.mean_confidence +
            (1 - alpha) * self.stats["avg_trajectory_confidence"]
        )
        self.stats["avg_ik_success_rate"] = (
            alpha * decoded_actions.success_rate +
            (1 - alpha) * self.stats["avg_ik_success_rate"]
        )
        self.stats["avg_total_time_ms"] = (
            alpha * total_time +
            (1 - alpha) * self.stats["avg_total_time_ms"]
        )

        return ARMResult(
            trajectory_trace=trajectory_trace,
            decoded_actions=decoded_actions,
            reasoning=reasoning,
            instruction=instruction,
            robot_id=self.config.robot_id,
            depth_tokens=depth_tokens,
            user_guidance_applied=guidance_applied,
            total_time_ms=total_time,
            depth_time_ms=depth_time,
            trajectory_time_ms=trajectory_time,
            decoding_time_ms=decoding_time,
            reasoning_time_ms=reasoning_time,
        )

    def _create_mock_trajectory(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> TrajectoryTrace:
        """Create a mock trajectory for testing."""
        # Generate reasonable waypoints
        n_waypoints = 8
        waypoints = np.zeros((n_waypoints, 2), dtype=np.float32)

        # Start from center-left, move to center-right
        waypoints[:, 0] = np.linspace(80, 176, n_waypoints)
        waypoints[:, 1] = np.linspace(100, 156, n_waypoints)

        # Add some curve
        waypoints[:, 1] += 20 * np.sin(np.linspace(0, np.pi, n_waypoints))

        confidences = np.linspace(0.9, 0.95, n_waypoints).astype(np.float32)

        return TrajectoryTrace(
            waypoints=waypoints,
            confidences=confidences,
            source_image=image,
            instruction=instruction,
            model_name="mock_predictor",
        )

    def execute_for_robot(
        self,
        robot_id: str,
        **kwargs,
    ) -> ARMResult:
        """
        Execute pipeline for a different robot (cross-robot transfer).

        Args:
            robot_id: Target robot ID
            **kwargs: Arguments passed to execute()

        Returns:
            ARMResult for the specified robot
        """
        # Save current robot
        original_robot = self.config.robot_id
        original_decoder = self._action_decoder

        try:
            # Switch to new robot
            new_robot_config = RobotRegistry.instance().get_or_raise(robot_id)
            self._action_decoder = ActionDecoder(
                robot_config=new_robot_config,
                config=self.config.action_decoder_config,
            )
            self.config.robot_id = robot_id
            self.robot_config = new_robot_config

            # Execute
            result = self.execute(**kwargs)
            return result

        finally:
            # Restore original
            self.config.robot_id = original_robot
            self._action_decoder = original_decoder
            self.robot_config = RobotRegistry.instance().get_or_raise(original_robot)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_executions"] / max(self.stats["total_executions"], 1)
            ),
            "robot_id": self.config.robot_id,
            "initialized": self._initialized,
        }

    @classmethod
    def create_for_robot(
        cls,
        robot_id: str,
        device: str = "cuda",
    ) -> "ARMPipeline":
        """
        Create pipeline for a specific robot.

        Args:
            robot_id: Robot identifier
            device: Compute device

        Returns:
            Configured ARMPipeline
        """
        config = ARMConfig(
            robot_id=robot_id,
            device=device,
        )
        pipeline = cls(config)
        pipeline.initialize()
        return pipeline

    @classmethod
    def create_minimal(cls, robot_id: str = "ur10e") -> "ARMPipeline":
        """
        Create minimal pipeline (no neural models).

        Useful for testing and environments without GPU.
        """
        config = ARMConfig(
            robot_id=robot_id,
            use_depth_tokenization=False,
            use_trajectory_prediction=False,
            use_action_reasoning=True,
            fallback_to_template_reasoning=True,
            device="cpu",
        )
        return cls(config)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_arm_pipeline(
    robot_id: str = "ur10e",
    use_gpu: bool = True,
) -> ARMPipeline:
    """Create an ARM pipeline for the specified robot."""
    device = "cuda" if use_gpu and HAS_TORCH and torch.cuda.is_available() else "cpu"
    return ARMPipeline.create_for_robot(robot_id, device)


def execute_arm(
    images: np.ndarray,
    instruction: str,
    robot_id: str = "ur10e",
    depth_map: Optional[np.ndarray] = None,
    user_guidance: Optional[UserGuidance] = None,
) -> ARMResult:
    """
    One-shot ARM execution.

    Args:
        images: Camera images
        instruction: Task instruction
        robot_id: Target robot
        depth_map: Optional depth image
        user_guidance: Optional user guidance

    Returns:
        ARMResult
    """
    pipeline = ARMPipeline.create_minimal(robot_id)
    return pipeline.execute(
        images=images,
        instruction=instruction,
        depth_map=depth_map,
        user_guidance=user_guidance,
    )


def demonstrate_cross_robot_transfer(
    images: np.ndarray,
    instruction: str,
    depth_map: Optional[np.ndarray] = None,
) -> Dict[str, ARMResult]:
    """
    Demonstrate cross-robot transfer by executing on all registered robots.

    Args:
        images: Camera images
        instruction: Task instruction
        depth_map: Optional depth image

    Returns:
        Dictionary mapping robot_id to ARMResult
    """
    results = {}
    pipeline = ARMPipeline.create_minimal("ur10e")

    for robot_id in RobotRegistry.instance().list_robots():
        try:
            result = pipeline.execute_for_robot(
                robot_id,
                images=images,
                instruction=instruction,
                depth_map=depth_map,
            )
            results[robot_id] = result
            logger.info(f"{robot_id}: {result.decoded_actions.action_horizon} actions, "
                       f"IK success={result.decoded_actions.success_rate:.1%}")
        except Exception as e:
            logger.warning(f"Failed for {robot_id}: {e}")

    return results

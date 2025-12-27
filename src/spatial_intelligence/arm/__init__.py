"""
Action Reasoning Model (ARM) Module for Dynamical

Implements MolmoAct-inspired features:
1. Interpretability: Trajectory traces with visualization
2. Steerability: User guidance interface
3. Cross-Robot Transfer: Embodiment-agnostic planning
4. Chain-of-Thought: Action reasoning with explicit spatial planning

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    ARM PIPELINE                                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Stage 1: Perception Tokenization                                   │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  Image + Depth ──▶ DepthVQVAE ──▶ Perception Tokens         │   │
    │  │                                    (discrete, spatial-aware) │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                               │                                      │
    │                               ▼                                      │
    │  Stage 2: Spatial Planning                                          │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  Tokens + Instruction ──▶ TrajectoryPredictor               │   │
    │  │                           ──▶ TrajectoryTrace [N, 2]        │   │
    │  │                               (image-space waypoints)        │   │
    │  │                               + UserGuidance (optional)      │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                               │                                      │
    │                               ▼                                      │
    │  Stage 3: Action Decoding                                           │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  Trace + RobotConfig ──▶ ActionDecoder ──▶ Joint Actions    │   │
    │  │                          (embodiment-specific IK)            │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

References:
- MolmoAct: arXiv:2508.07917
- Physical Intelligence Pi0.5: arXiv:2504.16054
"""

from .trajectory_trace import (
    TrajectoryTrace,
    TrajectoryVisualizationConfig,
)

from .depth_tokenizer import (
    DepthVQVAE,
    DepthTokenizerConfig,
    DepthTokens,
)

from .trajectory_predictor import (
    TrajectoryPredictor,
    TrajectoryPredictorConfig,
)

from .action_decoder import (
    ActionDecoder,
    ActionDecoderConfig,
)

from .robot_registry import (
    RobotConfig,
    RobotRegistry,
    CameraConfig,
)

from .steerability import (
    UserGuidance,
    AvoidRegion,
    GuidanceMode,
)

from .action_reasoning import (
    ActionReasoningModule,
    ActionReasoningConfig,
    ReasoningOutput,
)

from .arm_pipeline import (
    ARMPipeline,
    ARMConfig,
    ARMResult,
)

__all__ = [
    # Core data structures
    "TrajectoryTrace",
    "TrajectoryVisualizationConfig",
    "DepthTokens",
    # Perception
    "DepthVQVAE",
    "DepthTokenizerConfig",
    # Planning
    "TrajectoryPredictor",
    "TrajectoryPredictorConfig",
    # Execution
    "ActionDecoder",
    "ActionDecoderConfig",
    # Robot configuration
    "RobotConfig",
    "RobotRegistry",
    "CameraConfig",
    # Steerability
    "UserGuidance",
    "AvoidRegion",
    "GuidanceMode",
    # Reasoning
    "ActionReasoningModule",
    "ActionReasoningConfig",
    "ReasoningOutput",
    # Pipeline
    "ARMPipeline",
    "ARMConfig",
    "ARMResult",
]

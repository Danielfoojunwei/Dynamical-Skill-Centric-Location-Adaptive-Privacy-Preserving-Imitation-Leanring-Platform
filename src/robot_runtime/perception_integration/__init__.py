"""
Perception-to-Safety Integration Bridge

Converts perception outputs (SAM3, Depth, V-JEPA2) to CBF-compatible
obstacle representations for deterministic safety.

Gap Addressed:
    - SAM3 segmentation → obstacle positions for CBF
    - Depth maps → obstacle positions for CBF
    - V-JEPA2 collision predictions → speed reduction factors

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  PERCEPTION → SAFETY BRIDGE                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  SAM3 Masks ──────┐                                                 │
    │                   │                                                  │
    │  Depth Maps ──────┼──▶ ObstacleExtractor ──▶ obstacle_positions     │
    │                   │                          obstacle_radii         │
    │  Camera Intrinsics┘                                                 │
    │                                                                      │
    │  V-JEPA2 ─────────────▶ CollisionIntegrator ──▶ speed_factor        │
    │  collision_probs                                  (deterministic)    │
    │                                                                      │
    │                              ↓                                       │
    │                         RobotState                                   │
    │                   (CBF-compatible format)                            │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from src.robot_runtime.perception_integration import (
        PerceptionSafetyBridge,
        ObstacleExtractor,
        CollisionIntegrator,
    )

    bridge = PerceptionSafetyBridge.from_config(config)

    # In perception loop:
    safety_data = bridge.process(
        sam3_result=segmentation,
        depth_map=depth,
        camera_intrinsics=K,
        vjepa_collision_probs=collision_probs,
    )

    # Update robot state for CBF
    robot_state.obstacle_positions = safety_data.obstacle_positions
    robot_state.obstacle_radii = safety_data.obstacle_radii
"""

from .obstacle_extractor import (
    ObstacleExtractor,
    ObstacleData,
    ExtractorConfig,
)

from .collision_integrator import (
    CollisionIntegrator,
    CollisionIntegrationResult,
    IntegratorConfig,
)

from .bridge import (
    PerceptionSafetyBridge,
    SafetyPerceptionData,
    BridgeConfig,
)

__all__ = [
    # Obstacle extraction
    'ObstacleExtractor',
    'ObstacleData',
    'ExtractorConfig',

    # Collision integration
    'CollisionIntegrator',
    'CollisionIntegrationResult',
    'IntegratorConfig',

    # Main bridge
    'PerceptionSafetyBridge',
    'SafetyPerceptionData',
    'BridgeConfig',
]

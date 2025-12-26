"""
Perception-Safety Bridge - Main Interface

Unified interface that converts ALL perception outputs to safety-compatible
format for CBF and SafetyShield.

This is the missing link between:
- Meta AI perception (SAM3, DINOv3, V-JEPA2)
- CBF safety filter (needs obstacle_positions, obstacle_radii)
- SafetyShield (needs min_obstacle_distance, human_detected)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .obstacle_extractor import ObstacleExtractor, ObstacleData, ExtractorConfig
from .collision_integrator import (
    CollisionIntegrator,
    CollisionIntegrationResult,
    IntegratorConfig,
    SafetyAction,
)

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for perception-safety bridge."""
    extractor_config: ExtractorConfig = field(default_factory=ExtractorConfig)
    integrator_config: IntegratorConfig = field(default_factory=IntegratorConfig)

    # Update rate limiting
    min_update_interval_ms: float = 10.0  # Max 100Hz

    # Fallback behavior when perception unavailable
    use_conservative_fallback: bool = True
    fallback_speed_factor: float = 0.5


@dataclass
class SafetyPerceptionData:
    """
    Complete safety data from perception.

    This struct is designed to be directly usable by:
    - CBF (obstacle_positions, obstacle_radii)
    - SafetyShield (min_obstacle_distance, human_detected)
    - RTA (speed_factor)
    """
    # For CBF collision barriers
    obstacle_positions: np.ndarray  # [N, 3]
    obstacle_radii: np.ndarray  # [N]

    # For SafetyShield
    min_obstacle_distance: float
    human_detected: bool
    num_humans: int = 0

    # For RTA speed control
    speed_factor: float = 1.0  # 0.0 to 1.0
    safety_action: SafetyAction = SafetyAction.CONTINUE

    # V-JEPA2 collision predictions (informational)
    collision_probability: float = 0.0
    collision_horizon_probs: Optional[np.ndarray] = None

    # Obstacle details
    obstacle_labels: List[str] = field(default_factory=list)
    obstacle_velocities: Optional[np.ndarray] = None

    # Metadata
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    valid: bool = True
    error_message: Optional[str] = None


class PerceptionSafetyBridge:
    """
    Bridge between perception and safety systems.

    Converts Meta AI perception outputs (SAM3, DINOv3, V-JEPA2)
    to CBF-compatible format.
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()

        # Components
        self.extractor = ObstacleExtractor(self.config.extractor_config)
        self.integrator = CollisionIntegrator(self.config.integrator_config)

        # State
        self._last_update_time: float = 0.0
        self._last_result: Optional[SafetyPerceptionData] = None
        self._depth_history: List[float] = []

        # Statistics
        self.stats = {
            "updates": 0,
            "rate_limited": 0,
            "errors": 0,
        }

    def set_camera_intrinsics(self, K: np.ndarray):
        """Set camera intrinsics for 3D projection."""
        self.extractor.set_camera_intrinsics(K)

    def set_camera_transform(self, T_world_camera: np.ndarray):
        """Set camera-to-world transform."""
        self.extractor.set_camera_transform(T_world_camera)

    def process(
        self,
        depth_map: Optional[np.ndarray] = None,
        sam3_result: Optional[Any] = None,
        human_detections: Optional[List[Dict]] = None,
        vjepa_collision_probs: Optional[np.ndarray] = None,
        unified_perception_result: Optional[Any] = None,
    ) -> SafetyPerceptionData:
        """
        Process perception data and return safety-compatible output.

        Args:
            depth_map: Depth image [H, W]
            sam3_result: SAM3 segmentation result
            human_detections: List of human detection dicts
            vjepa_collision_probs: V-JEPA2 collision probabilities [T]
            unified_perception_result: UnifiedPerceptionPipeline result (alternative)

        Returns:
            SafetyPerceptionData for CBF and SafetyShield
        """
        start_time = time.time()
        timestamp = start_time

        # Rate limiting
        if self._should_rate_limit():
            self.stats["rate_limited"] += 1
            if self._last_result is not None:
                return self._last_result
            else:
                return self._get_fallback_result(timestamp)

        self.stats["updates"] += 1

        try:
            # Extract from unified perception result if provided
            if unified_perception_result is not None:
                depth_map, sam3_result, vjepa_collision_probs, human_detections = (
                    self._extract_from_unified(unified_perception_result)
                )

            # Validate inputs
            if depth_map is None:
                return self._get_fallback_result(
                    timestamp,
                    error="No depth map provided",
                )

            # 1. Extract obstacles
            obstacles = self.extractor.extract(
                depth_map=depth_map,
                sam3_result=sam3_result,
                human_detections=human_detections,
            )

            # 2. Compute minimum distance
            min_distance = obstacles.min_distance
            if min_distance == float('inf'):
                # No obstacles detected - use depth minimum
                valid_depth = depth_map[depth_map > 0]
                if len(valid_depth) > 0:
                    min_distance = float(valid_depth.min()) / self.config.extractor_config.depth_scale
                else:
                    min_distance = float('inf')

            # 3. Compute depth velocity
            depth_velocity = self._compute_depth_velocity(min_distance)

            # 4. Integrate with V-JEPA2
            integration = self.integrator.integrate(
                min_depth_m=min_distance,
                vjepa_collision_probs=vjepa_collision_probs,
                depth_velocity=depth_velocity,
            )

            # 5. Build result
            result = SafetyPerceptionData(
                obstacle_positions=obstacles.positions,
                obstacle_radii=obstacles.radii,
                min_obstacle_distance=min_distance,
                human_detected=obstacles.has_humans,
                num_humans=sum(1 for l in obstacles.labels if l == "human"),
                speed_factor=integration.speed_factor,
                safety_action=integration.action,
                collision_probability=integration.vjepa_max_prob,
                collision_horizon_probs=integration.vjepa_horizon_probs,
                obstacle_labels=obstacles.labels,
                obstacle_velocities=obstacles.velocities,
                timestamp=timestamp,
                processing_time_ms=(time.time() - start_time) * 1000,
                valid=True,
            )

            self._last_result = result
            self._last_update_time = time.time()

            return result

        except Exception as e:
            logger.error(f"Perception bridge error: {e}")
            self.stats["errors"] += 1
            return self._get_fallback_result(timestamp, error=str(e))

    def _extract_from_unified(
        self,
        result: Any,
    ) -> tuple:
        """Extract components from UnifiedPerceptionPipeline result."""
        depth_map = None
        sam3_result = None
        vjepa_probs = None
        human_detections = None

        # Extract depth
        if hasattr(result, 'depth') and result.depth is not None:
            depth_map = result.depth

        # Extract SAM3 segmentation
        if hasattr(result, 'segmentation'):
            sam3_result = result.segmentation

        # Extract V-JEPA2 predictions
        if hasattr(result, 'world_prediction'):
            wp = result.world_prediction
            if hasattr(wp, 'collision_probabilities'):
                vjepa_probs = wp.collision_probabilities

        # Extract human detections from segmentation
        if sam3_result is not None and hasattr(sam3_result, 'masks'):
            # Filter for human masks
            human_detections = []
            for mask in sam3_result.masks:
                label = getattr(mask, 'label', 'unknown')
                if label.lower() in ['human', 'person', 'people']:
                    bbox = getattr(mask, 'bbox', None)
                    if bbox is not None:
                        human_detections.append({
                            'bbox': bbox,
                            'confidence': getattr(mask, 'confidence', 0.8),
                        })

        return depth_map, sam3_result, vjepa_probs, human_detections

    def _should_rate_limit(self) -> bool:
        """Check if update should be rate limited."""
        if self._last_update_time == 0:
            return False

        elapsed_ms = (time.time() - self._last_update_time) * 1000
        return elapsed_ms < self.config.min_update_interval_ms

    def _compute_depth_velocity(self, current_depth: float) -> Optional[float]:
        """Compute rate of change of minimum depth."""
        self._depth_history.append(current_depth)

        if len(self._depth_history) > 10:
            self._depth_history.pop(0)

        if len(self._depth_history) < 3:
            return None

        # Simple finite difference
        dt = self.config.min_update_interval_ms / 1000.0 * len(self._depth_history)
        velocity = (self._depth_history[-1] - self._depth_history[0]) / dt

        return velocity

    def _get_fallback_result(
        self,
        timestamp: float,
        error: Optional[str] = None,
    ) -> SafetyPerceptionData:
        """Return conservative fallback result."""
        return SafetyPerceptionData(
            obstacle_positions=np.zeros((0, 3)),
            obstacle_radii=np.zeros(0),
            min_obstacle_distance=0.5 if self.config.use_conservative_fallback else float('inf'),
            human_detected=self.config.use_conservative_fallback,  # Assume human present
            speed_factor=self.config.fallback_speed_factor if self.config.use_conservative_fallback else 1.0,
            safety_action=SafetyAction.SLOW if self.config.use_conservative_fallback else SafetyAction.CONTINUE,
            timestamp=timestamp,
            valid=False,
            error_message=error,
        )

    def update_robot_state(
        self,
        robot_state_dict: Dict[str, Any],
        safety_data: SafetyPerceptionData,
    ) -> Dict[str, Any]:
        """
        Update robot state dict with perception safety data.

        This is a convenience method to inject safety data into
        the robot state dict used by CBF and SafetyShield.
        """
        robot_state_dict['obstacle_positions'] = safety_data.obstacle_positions
        robot_state_dict['obstacle_radii'] = safety_data.obstacle_radii
        robot_state_dict['min_obstacle_distance'] = safety_data.min_obstacle_distance
        robot_state_dict['human_detected'] = safety_data.human_detected
        robot_state_dict['humans'] = [
            {'detected': True}
            for _ in range(safety_data.num_humans)
        ]

        # Add V-JEPA2 predictions as advisory
        robot_state_dict['ml_collision_probability'] = safety_data.collision_probability
        robot_state_dict['ml_speed_factor'] = safety_data.speed_factor

        return robot_state_dict

    @classmethod
    def from_config(cls, app_config: Any) -> 'PerceptionSafetyBridge':
        """Create bridge from application config."""
        bridge_config = BridgeConfig()

        # Extract settings if available
        if hasattr(app_config, 'safety'):
            safety_cfg = app_config.safety
            bridge_config.integrator_config.stop_distance_m = getattr(
                safety_cfg, 'stop_distance_m', 0.30
            )
            bridge_config.integrator_config.slow_distance_m = getattr(
                safety_cfg, 'slow_distance_m', 0.50
            )

        return cls(bridge_config)

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            **self.stats,
            "extractor": self.extractor.get_statistics(),
            "integrator": self.integrator.get_statistics(),
        }

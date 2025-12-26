"""
Collision Integrator - Fuse V-JEPA2 Predictions with Depth for Deterministic Safety

V-JEPA2 provides PROBABILISTIC collision predictions.
Depth sensor provides DETERMINISTIC distance measurements.

This module FUSES both to make deterministic safety decisions:
- If depth < threshold → STOP (deterministic)
- If V-JEPA2 predicts collision AND depth shows decreasing → SLOW DOWN
- If V-JEPA2 predicts collision BUT depth stable → CAUTION (log only)

Key Principle: ML predictions inform, physics/sensors decide.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SafetyAction(Enum):
    """Deterministic safety actions."""
    CONTINUE = "continue"  # Safe to proceed at full speed
    SLOW = "slow"  # Reduce speed by factor
    STOP = "stop"  # Immediate controlled stop
    EMERGENCY = "emergency"  # Emergency stop (hardware-level)


@dataclass
class IntegratorConfig:
    """Configuration for collision integration."""
    # Depth thresholds (HARD limits, not learned)
    emergency_stop_distance_m: float = 0.15  # Immediate stop
    stop_distance_m: float = 0.30  # Controlled stop
    slow_distance_m: float = 0.50  # Speed reduction

    # V-JEPA2 thresholds
    collision_prob_emergency: float = 0.95  # Very high confidence
    collision_prob_stop: float = 0.80  # High confidence
    collision_prob_slow: float = 0.50  # Moderate confidence

    # Speed reduction factors
    slow_speed_factor: float = 0.3  # 30% of normal speed
    caution_speed_factor: float = 0.7  # 70% of normal speed

    # Fusion weights
    depth_weight: float = 0.7  # Trust depth more
    vjepa_weight: float = 0.3  # V-JEPA2 is advisory

    # Temporal filtering
    collision_history_size: int = 5
    collision_persistence_threshold: int = 3  # Need 3 consecutive predictions


@dataclass
class CollisionIntegrationResult:
    """Result of collision integration."""
    # Primary output
    action: SafetyAction
    speed_factor: float  # 0.0 to 1.0

    # Source of decision
    triggered_by: str  # "depth", "vjepa", "fused", "none"

    # Details
    min_depth_m: float
    vjepa_max_prob: float
    vjepa_horizon_probs: Optional[np.ndarray] = None

    # For monitoring
    depth_safe: bool = True
    vjepa_safe: bool = True
    fused_safe: bool = True

    # Timing
    integration_time_ms: float = 0.0


class CollisionIntegrator:
    """
    Integrate V-JEPA2 collision predictions with depth measurements.

    Makes DETERMINISTIC safety decisions by fusing ML predictions
    with physical sensor measurements.
    """

    def __init__(self, config: Optional[IntegratorConfig] = None):
        self.config = config or IntegratorConfig()

        # History for temporal filtering
        self._collision_history: List[bool] = []
        self._vjepa_history: List[float] = []

        # Statistics
        self.stats = {
            "integrations": 0,
            "depth_triggered": 0,
            "vjepa_triggered": 0,
            "fused_triggered": 0,
            "emergency_stops": 0,
        }

    def integrate(
        self,
        min_depth_m: float,
        vjepa_collision_probs: Optional[np.ndarray] = None,
        depth_velocity: Optional[float] = None,  # Derivative of min depth
    ) -> CollisionIntegrationResult:
        """
        Integrate depth and V-JEPA2 for safety decision.

        Args:
            min_depth_m: Minimum depth to any obstacle
            vjepa_collision_probs: V-JEPA2 collision probabilities [T]
            depth_velocity: Rate of change of min depth (negative = approaching)

        Returns:
            CollisionIntegrationResult with action and speed factor
        """
        import time
        start_time = time.time()

        self.stats["integrations"] += 1

        # Default: safe
        action = SafetyAction.CONTINUE
        speed_factor = 1.0
        triggered_by = "none"
        depth_safe = True
        vjepa_safe = True
        fused_safe = True

        # 1. Depth-based decision (DETERMINISTIC, highest priority)
        depth_action, depth_factor = self._check_depth(min_depth_m)

        if depth_action == SafetyAction.EMERGENCY:
            self.stats["emergency_stops"] += 1
            return CollisionIntegrationResult(
                action=SafetyAction.EMERGENCY,
                speed_factor=0.0,
                triggered_by="depth",
                min_depth_m=min_depth_m,
                vjepa_max_prob=0.0 if vjepa_collision_probs is None else float(vjepa_collision_probs.max()),
                depth_safe=False,
                integration_time_ms=(time.time() - start_time) * 1000,
            )

        if depth_action != SafetyAction.CONTINUE:
            action = depth_action
            speed_factor = depth_factor
            triggered_by = "depth"
            depth_safe = False
            self.stats["depth_triggered"] += 1

        # 2. V-JEPA2 prediction (ADVISORY, can only make action more conservative)
        vjepa_max_prob = 0.0
        if vjepa_collision_probs is not None:
            vjepa_action, vjepa_factor, vjepa_max_prob = self._check_vjepa(
                vjepa_collision_probs
            )

            if vjepa_action != SafetyAction.CONTINUE:
                vjepa_safe = False
                self.stats["vjepa_triggered"] += 1

                # Only upgrade action (make more conservative), never downgrade
                if self._action_priority(vjepa_action) > self._action_priority(action):
                    # V-JEPA2 is more conservative, but only trust if depth supports it
                    if self._should_trust_vjepa(min_depth_m, depth_velocity, vjepa_max_prob):
                        action = vjepa_action
                        speed_factor = min(speed_factor, vjepa_factor)
                        triggered_by = "fused" if triggered_by != "none" else "vjepa"
                        fused_safe = False
                        self.stats["fused_triggered"] += 1

        # 3. Temporal filtering (reduce false positives)
        action, speed_factor = self._apply_temporal_filter(
            action, speed_factor, vjepa_max_prob
        )

        return CollisionIntegrationResult(
            action=action,
            speed_factor=speed_factor,
            triggered_by=triggered_by,
            min_depth_m=min_depth_m,
            vjepa_max_prob=vjepa_max_prob,
            vjepa_horizon_probs=vjepa_collision_probs,
            depth_safe=depth_safe,
            vjepa_safe=vjepa_safe,
            fused_safe=fused_safe,
            integration_time_ms=(time.time() - start_time) * 1000,
        )

    def _check_depth(self, min_depth_m: float) -> Tuple[SafetyAction, float]:
        """Check depth against hard-coded thresholds."""
        if min_depth_m <= self.config.emergency_stop_distance_m:
            return SafetyAction.EMERGENCY, 0.0

        if min_depth_m <= self.config.stop_distance_m:
            return SafetyAction.STOP, 0.0

        if min_depth_m <= self.config.slow_distance_m:
            # Gradual slowdown
            factor = (min_depth_m - self.config.stop_distance_m) / (
                self.config.slow_distance_m - self.config.stop_distance_m
            )
            factor = max(self.config.slow_speed_factor, factor)
            return SafetyAction.SLOW, factor

        return SafetyAction.CONTINUE, 1.0

    def _check_vjepa(
        self,
        collision_probs: np.ndarray,
    ) -> Tuple[SafetyAction, float, float]:
        """Check V-JEPA2 collision predictions."""
        if len(collision_probs) == 0:
            return SafetyAction.CONTINUE, 1.0, 0.0

        max_prob = float(collision_probs.max())

        # Check immediate collision (first timestep)
        immediate_prob = float(collision_probs[0])

        if immediate_prob >= self.config.collision_prob_emergency:
            return SafetyAction.STOP, 0.0, max_prob

        if immediate_prob >= self.config.collision_prob_stop:
            return SafetyAction.STOP, 0.0, max_prob

        if max_prob >= self.config.collision_prob_slow:
            return SafetyAction.SLOW, self.config.slow_speed_factor, max_prob

        if max_prob >= 0.3:  # Caution threshold
            return SafetyAction.SLOW, self.config.caution_speed_factor, max_prob

        return SafetyAction.CONTINUE, 1.0, max_prob

    def _should_trust_vjepa(
        self,
        min_depth_m: float,
        depth_velocity: Optional[float],
        vjepa_prob: float,
    ) -> bool:
        """
        Determine if V-JEPA2 prediction should be trusted.

        Only trust V-JEPA2 if physical evidence supports it:
        - Obstacle is within relevant range
        - Obstacle is getting closer (negative velocity)
        """
        # If obstacle is far, V-JEPA2 prediction less reliable
        if min_depth_m > 2.0:
            return False

        # If V-JEPA2 confidence is very high, trust it
        if vjepa_prob > 0.9:
            return True

        # If approaching obstacle (negative velocity), trust V-JEPA2
        if depth_velocity is not None and depth_velocity < -0.1:
            return True

        # If obstacle is close, trust V-JEPA2
        if min_depth_m < 1.0:
            return True

        return False

    def _action_priority(self, action: SafetyAction) -> int:
        """Get priority of action (higher = more conservative)."""
        priorities = {
            SafetyAction.CONTINUE: 0,
            SafetyAction.SLOW: 1,
            SafetyAction.STOP: 2,
            SafetyAction.EMERGENCY: 3,
        }
        return priorities.get(action, 0)

    def _apply_temporal_filter(
        self,
        action: SafetyAction,
        speed_factor: float,
        vjepa_prob: float,
    ) -> Tuple[SafetyAction, float]:
        """Apply temporal filtering to reduce false positives."""
        # Track collision predictions
        is_collision = action != SafetyAction.CONTINUE

        self._collision_history.append(is_collision)
        self._vjepa_history.append(vjepa_prob)

        # Keep history limited
        if len(self._collision_history) > self.config.collision_history_size:
            self._collision_history.pop(0)
            self._vjepa_history.pop(0)

        # Require persistence for non-depth triggers
        if action in [SafetyAction.SLOW] and len(self._collision_history) >= 3:
            recent_collisions = sum(self._collision_history[-3:])
            if recent_collisions < self.config.collision_persistence_threshold:
                # Not persistent enough, downgrade to continue
                return SafetyAction.CONTINUE, 1.0

        return action, speed_factor

    def reset(self):
        """Reset collision history."""
        self._collision_history.clear()
        self._vjepa_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return dict(self.stats)

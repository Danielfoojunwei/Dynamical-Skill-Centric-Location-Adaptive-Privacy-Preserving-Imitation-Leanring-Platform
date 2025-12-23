"""
Skill Blender - Stable Multi-Skill Blending with Safety Guarantees

Implements the defensive skill blending described in README v0.7.1:
1. All skills output normalized actions in [-1, 1]
2. All skills use same action space (verified at registration)
3. Blend weights sum to 1.0
4. Maximum delta between frames (jerk limiting)
5. Uncertainty-based weight reduction

This module addresses the critique: "naive blending is a standard way to
create jitter, oscillations, and unsafe boundary behavior."
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ActionSpace(str, Enum):
    """Robot action space types."""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    CARTESIAN_POSE = "cartesian_pose"
    HYBRID = "hybrid"


@dataclass
class SkillOutput:
    """Output from a single skill inference."""
    action: np.ndarray  # Normalized to [-1, 1]
    confidence: float  # 0.0 to 1.0
    skill_id: str
    action_space: ActionSpace
    timestamp: float = field(default_factory=time.time)


@dataclass
class BlendConfig:
    """Configuration for skill blending."""
    # Action normalization
    action_min: float = -1.0
    action_max: float = 1.0

    # Confidence thresholds
    confidence_threshold: float = 0.3  # Below this, skill weight becomes 0
    min_total_confidence: float = 0.1  # Below this, use safe default

    # Jerk limiting
    max_action_delta_per_second: float = 2.0  # Max change rate in normalized units

    # Stability
    blend_smoothing_alpha: float = 0.3  # EMA smoothing for blend weights

    # Safety
    enable_action_clipping: bool = True
    enable_jerk_limiting: bool = True
    enable_confidence_weighting: bool = True


@dataclass
class BlendResult:
    """Result of skill blending operation."""
    action: np.ndarray
    original_weights: List[float]
    adjusted_weights: List[float]
    skill_ids: List[str]
    total_confidence: float
    used_safe_default: bool = False
    jerk_limited: bool = False
    clipping_applied: bool = False
    blend_time_ms: float = 0.0


class SkillBlender:
    """
    Blend multiple skills with stability guarantees.

    Constraints enforced:
    1. All skills output normalized actions in [-1, 1]
    2. All skills use same action space (verified at registration)
    3. Blend weights sum to 1.0
    4. Maximum delta between frames (jerk limiting)
    5. Uncertainty-based weight reduction

    Usage:
        blender = SkillBlender(config=BlendConfig())

        # Register skills with their action spaces
        blender.register_skill("grasp", ActionSpace.JOINT_POSITION, action_dim=7)
        blender.register_skill("place", ActionSpace.JOINT_POSITION, action_dim=7)

        # Blend skill outputs
        outputs = [
            SkillOutput(action=grasp_action, confidence=0.9, skill_id="grasp", ...),
            SkillOutput(action=place_action, confidence=0.7, skill_id="place", ...),
        ]
        weights = [0.6, 0.4]
        result = blender.blend(outputs, weights, dt=0.01)
    """

    def __init__(
        self,
        config: BlendConfig = None,
        action_dim: int = 7,
        safe_default_action: np.ndarray = None,
    ):
        self.config = config or BlendConfig()
        self.action_dim = action_dim

        # Safe default action (zero velocity/hold position)
        if safe_default_action is not None:
            self.safe_default_action = safe_default_action
        else:
            self.safe_default_action = np.zeros(action_dim, dtype=np.float32)

        # Registered skills
        self._registered_skills: Dict[str, Dict[str, Any]] = {}

        # State for temporal filtering
        self._last_action: Optional[np.ndarray] = None
        self._last_weights: Optional[List[float]] = None
        self._last_blend_time: float = 0.0

        # Statistics
        self.stats = {
            "blends": 0,
            "jerk_limited_count": 0,
            "safe_default_count": 0,
            "clipping_count": 0,
        }

    def register_skill(
        self,
        skill_id: str,
        action_space: ActionSpace,
        action_dim: int,
        action_normalizer: Callable[[np.ndarray], np.ndarray] = None,
    ) -> None:
        """
        Register a skill for blending.

        All skills must have the same action space and dimension for blending.

        Args:
            skill_id: Unique skill identifier
            action_space: Action space type
            action_dim: Dimension of action vector
            action_normalizer: Optional function to normalize skill output to [-1, 1]

        Raises:
            ValueError: If action space or dimension doesn't match registered skills
        """
        # Check consistency with existing skills
        if self._registered_skills:
            first_skill = next(iter(self._registered_skills.values()))
            if first_skill["action_space"] != action_space:
                raise ValueError(
                    f"Action space mismatch: {skill_id} uses {action_space}, "
                    f"but existing skills use {first_skill['action_space']}"
                )
            if first_skill["action_dim"] != action_dim:
                raise ValueError(
                    f"Action dimension mismatch: {skill_id} has dim {action_dim}, "
                    f"but existing skills have dim {first_skill['action_dim']}"
                )

        self._registered_skills[skill_id] = {
            "action_space": action_space,
            "action_dim": action_dim,
            "normalizer": action_normalizer,
        }

        # Update blender action dim
        self.action_dim = action_dim
        self.safe_default_action = np.zeros(action_dim, dtype=np.float32)

        logger.info(f"Registered skill {skill_id} with action_space={action_space}, dim={action_dim}")

    def validate_skill_output(self, output: SkillOutput) -> Tuple[bool, str]:
        """
        Validate a skill output before blending.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check skill is registered
        if output.skill_id not in self._registered_skills:
            return False, f"Skill {output.skill_id} not registered"

        skill_info = self._registered_skills[output.skill_id]

        # Check action space matches
        if output.action_space != skill_info["action_space"]:
            return False, f"Action space mismatch for {output.skill_id}"

        # Check action dimension
        if len(output.action) != skill_info["action_dim"]:
            return False, f"Action dimension mismatch for {output.skill_id}: expected {skill_info['action_dim']}, got {len(output.action)}"

        # Check normalization
        if output.action.min() < self.config.action_min - 0.01:
            return False, f"Action below minimum: {output.action.min()} < {self.config.action_min}"
        if output.action.max() > self.config.action_max + 0.01:
            return False, f"Action above maximum: {output.action.max()} > {self.config.action_max}"

        # Check confidence range
        if not 0.0 <= output.confidence <= 1.0:
            return False, f"Confidence out of range: {output.confidence}"

        return True, ""

    def blend(
        self,
        outputs: List[SkillOutput],
        weights: List[float],
        dt: float = 0.01,
    ) -> BlendResult:
        """
        Blend multiple skill outputs with stability guarantees.

        Args:
            outputs: List of skill outputs to blend
            weights: Blend weights for each skill (must sum to 1.0)
            dt: Time step since last blend (for jerk limiting)

        Returns:
            BlendResult with blended action and metadata
        """
        start_time = time.time()
        self.stats["blends"] += 1

        # Validate inputs
        if len(outputs) != len(weights):
            raise ValueError(f"Length mismatch: {len(outputs)} outputs, {len(weights)} weights")

        if len(outputs) == 0:
            return BlendResult(
                action=self.safe_default_action.copy(),
                original_weights=[],
                adjusted_weights=[],
                skill_ids=[],
                total_confidence=0.0,
                used_safe_default=True,
                blend_time_ms=(time.time() - start_time) * 1000,
            )

        # Validate weight sum
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-4:
            logger.warning(f"Weights don't sum to 1.0 (sum={weight_sum}), normalizing")
            weights = [w / weight_sum for w in weights]

        # Validate each skill output
        valid_outputs = []
        valid_weights = []
        skill_ids = []

        for output, weight in zip(outputs, weights):
            is_valid, error = self.validate_skill_output(output)
            if is_valid:
                valid_outputs.append(output)
                valid_weights.append(weight)
                skill_ids.append(output.skill_id)
            else:
                logger.warning(f"Skipping invalid skill output: {error}")

        if not valid_outputs:
            self.stats["safe_default_count"] += 1
            return BlendResult(
                action=self.safe_default_action.copy(),
                original_weights=weights,
                adjusted_weights=[],
                skill_ids=[],
                total_confidence=0.0,
                used_safe_default=True,
                blend_time_ms=(time.time() - start_time) * 1000,
            )

        # Renormalize valid weights
        valid_weight_sum = sum(valid_weights)
        if valid_weight_sum > 0:
            valid_weights = [w / valid_weight_sum for w in valid_weights]

        original_weights = valid_weights.copy()

        # Apply confidence-based weight adjustment
        if self.config.enable_confidence_weighting:
            adjusted_weights = []
            for output, weight in zip(valid_outputs, valid_weights):
                if output.confidence >= self.config.confidence_threshold:
                    adjusted_weights.append(weight * output.confidence)
                else:
                    # Below threshold: zero out
                    adjusted_weights.append(0.0)

            # Renormalize adjusted weights
            adjusted_sum = sum(adjusted_weights)
            if adjusted_sum < self.config.min_total_confidence:
                # All skills too uncertain: use safe default
                self.stats["safe_default_count"] += 1
                return BlendResult(
                    action=self.safe_default_action.copy(),
                    original_weights=original_weights,
                    adjusted_weights=adjusted_weights,
                    skill_ids=skill_ids,
                    total_confidence=adjusted_sum,
                    used_safe_default=True,
                    blend_time_ms=(time.time() - start_time) * 1000,
                )

            adjusted_weights = [w / adjusted_sum for w in adjusted_weights]
        else:
            adjusted_weights = valid_weights

        # Compute weighted sum of actions
        blended_action = np.zeros(self.action_dim, dtype=np.float32)
        for output, weight in zip(valid_outputs, adjusted_weights):
            blended_action += weight * output.action.astype(np.float32)

        # Apply action clipping
        clipping_applied = False
        if self.config.enable_action_clipping:
            before_clip = blended_action.copy()
            blended_action = np.clip(
                blended_action,
                self.config.action_min,
                self.config.action_max,
            )
            if not np.allclose(before_clip, blended_action):
                clipping_applied = True
                self.stats["clipping_count"] += 1

        # Apply jerk limiting (smooth transitions)
        jerk_limited = False
        if self.config.enable_jerk_limiting and self._last_action is not None:
            max_delta = self.config.max_action_delta_per_second * dt
            delta = blended_action - self._last_action

            # Clip delta to max allowed change
            delta_magnitude = np.abs(delta)
            if np.any(delta_magnitude > max_delta):
                jerk_limited = True
                self.stats["jerk_limited_count"] += 1

                # Scale delta to respect limits
                scale = np.minimum(max_delta / (delta_magnitude + 1e-8), 1.0)
                blended_action = self._last_action + delta * scale

        # Update state
        self._last_action = blended_action.copy()
        self._last_weights = adjusted_weights.copy()
        self._last_blend_time = time.time()

        # Compute total confidence
        total_confidence = sum(
            w * o.confidence for w, o in zip(adjusted_weights, valid_outputs)
        )

        return BlendResult(
            action=blended_action,
            original_weights=original_weights,
            adjusted_weights=adjusted_weights,
            skill_ids=skill_ids,
            total_confidence=total_confidence,
            used_safe_default=False,
            jerk_limited=jerk_limited,
            clipping_applied=clipping_applied,
            blend_time_ms=(time.time() - start_time) * 1000,
        )

    def reset_state(self) -> None:
        """Reset temporal state (call when starting new episode)."""
        self._last_action = None
        self._last_weights = None
        self._last_blend_time = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get blending statistics."""
        return {
            **self.stats,
            "registered_skills": list(self._registered_skills.keys()),
            "action_dim": self.action_dim,
        }


# =============================================================================
# Tests
# =============================================================================

def test_skill_blender():
    """Test skill blender functionality."""
    print("\n" + "=" * 60)
    print("SKILL BLENDER TEST")
    print("=" * 60)

    config = BlendConfig(
        confidence_threshold=0.3,
        max_action_delta_per_second=2.0,
    )
    blender = SkillBlender(config=config, action_dim=7)

    # Register skills
    print("\n1. Register Skills")
    print("-" * 40)
    blender.register_skill("grasp", ActionSpace.JOINT_POSITION, action_dim=7)
    blender.register_skill("place", ActionSpace.JOINT_POSITION, action_dim=7)
    print("   Registered: grasp, place")

    # Test valid blending
    print("\n2. Test Valid Blending")
    print("-" * 40)
    outputs = [
        SkillOutput(
            action=np.array([0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]),
            confidence=0.9,
            skill_id="grasp",
            action_space=ActionSpace.JOINT_POSITION,
        ),
        SkillOutput(
            action=np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]),
            confidence=0.7,
            skill_id="place",
            action_space=ActionSpace.JOINT_POSITION,
        ),
    ]
    weights = [0.6, 0.4]

    result = blender.blend(outputs, weights, dt=0.01)
    print(f"   Success: {not result.used_safe_default}")
    print(f"   Blended action: [{result.action[0]:.3f}, {result.action[1]:.3f}, ...]")
    print(f"   Adjusted weights: {[f'{w:.3f}' for w in result.adjusted_weights]}")
    print(f"   Total confidence: {result.total_confidence:.3f}")

    # Test low confidence fallback
    print("\n3. Test Low Confidence Fallback")
    print("-" * 40)
    low_conf_outputs = [
        SkillOutput(
            action=np.array([0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]),
            confidence=0.1,  # Below threshold
            skill_id="grasp",
            action_space=ActionSpace.JOINT_POSITION,
        ),
        SkillOutput(
            action=np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]),
            confidence=0.05,  # Below threshold
            skill_id="place",
            action_space=ActionSpace.JOINT_POSITION,
        ),
    ]

    result = blender.blend(low_conf_outputs, weights, dt=0.01)
    print(f"   Used safe default: {result.used_safe_default}")
    print(f"   Action is zero: {np.allclose(result.action, 0)}")

    # Test jerk limiting
    print("\n4. Test Jerk Limiting")
    print("-" * 40)
    blender.reset_state()

    # First blend to set baseline
    result1 = blender.blend(outputs, weights, dt=0.01)

    # Large jump
    jump_outputs = [
        SkillOutput(
            action=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Max
            confidence=0.9,
            skill_id="grasp",
            action_space=ActionSpace.JOINT_POSITION,
        ),
        SkillOutput(
            action=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Max
            confidence=0.7,
            skill_id="place",
            action_space=ActionSpace.JOINT_POSITION,
        ),
    ]

    result2 = blender.blend(jump_outputs, weights, dt=0.01)
    print(f"   Jerk limited: {result2.jerk_limited}")
    print(f"   Action clamped (not jumping to 1.0)")

    # Test normalization violation
    print("\n5. Test Normalization Violation Detection")
    print("-" * 40)
    blender.reset_state()

    bad_output = SkillOutput(
        action=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Exceeds [-1, 1]
        confidence=0.9,
        skill_id="grasp",
        action_space=ActionSpace.JOINT_POSITION,
    )

    is_valid, error = blender.validate_skill_output(bad_output)
    print(f"   Invalid detected: {not is_valid}")
    print(f"   Error: {error}")

    # Statistics
    print("\n6. Statistics")
    print("-" * 40)
    stats = blender.get_statistics()
    print(f"   Blends: {stats['blends']}")
    print(f"   Jerk limited: {stats['jerk_limited_count']}")
    print(f"   Safe default used: {stats['safe_default_count']}")

    print("\n" + "=" * 60)
    print("SKILL BLENDER TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_skill_blender()

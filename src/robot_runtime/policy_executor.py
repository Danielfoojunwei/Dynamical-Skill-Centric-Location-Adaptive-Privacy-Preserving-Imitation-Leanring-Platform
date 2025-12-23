"""
Policy Executor - Tier 2 Policy Inference

Runs at 100Hz with TensorRT-compiled models.
Generates actions from observations.
"""

import time
import logging
from typing import Dict, Any, Optional
import numpy as np

from .config import PolicyConfig

logger = logging.getLogger(__name__)


class PolicyExecutor:
    """
    Policy executor for action generation.

    Runs VLA/ACT models at 100Hz using TensorRT.
    Supports skill-conditioned policies.
    """

    def __init__(self, rate_hz: int, config: PolicyConfig):
        self.rate_hz = rate_hz
        self.config = config

        self._model = None
        self._action_dim = config.action_dim
        self._last_action = np.zeros(self._action_dim)
        self._initialized = False

        # Inference stats
        self._inference_count = 0
        self._total_inference_time = 0.0

    def initialize(self) -> None:
        """Initialize policy model."""
        logger.info(f"Initializing policy executor with {self.config.default_model}")

        # In production, load TensorRT engine
        # self._model = load_tensorrt(f'{self.config.default_model}.engine')

        self._initialized = True

    def infer(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Run policy inference.

        Args:
            observation: Current observation with state, perception, context

        Returns:
            Action array (joint velocities, etc.)
        """
        if not self._initialized:
            return np.zeros(self._action_dim)

        start_time = time.time()

        try:
            # Extract relevant features
            state = observation.get('state')
            perception = observation.get('perception', {})
            context = observation.get('skill_context', {})

            # Build model input
            model_input = self._build_input(state, perception, context)

            # Run inference
            action = self._run_model(model_input)

            # Post-process action
            action = self._post_process(action)

            self._last_action = action

            # Track stats
            inference_time = time.time() - start_time
            self._inference_count += 1
            self._total_inference_time += inference_time

            return action

        except Exception as e:
            logger.error(f"Policy inference error: {e}")
            return self._last_action  # Return last valid action

    def load_skill_policy(self, skill_id: str) -> bool:
        """
        Load skill-specific policy weights.

        Args:
            skill_id: ID of skill to load

        Returns:
            True if loaded successfully
        """
        logger.info(f"Loading policy for skill: {skill_id}")
        # In production, load skill-specific LoRA weights or full model
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = 0.0
        if self._inference_count > 0:
            avg_time = self._total_inference_time / self._inference_count

        return {
            'inference_count': self._inference_count,
            'avg_inference_time_ms': avg_time * 1000,
            'target_rate_hz': self.rate_hz,
        }

    def _build_input(
        self,
        state: Any,
        perception: Dict[str, Any],
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Build model input tensor."""
        # Simplified: concatenate relevant features
        features = []

        # Joint state
        if state is not None and hasattr(state, 'joint_positions'):
            features.extend(state.joint_positions.tolist())
            features.extend(state.joint_velocities.tolist())

        # Perception features
        if 'dense_features' in perception and perception['dense_features'] is not None:
            features.extend(perception['dense_features'][:64].tolist())

        # Skill context
        features.append(context.get('progress', 0.0))

        return np.array(features, dtype=np.float32)

    def _run_model(self, model_input: np.ndarray) -> np.ndarray:
        """Run model inference."""
        # Mock inference - would use TensorRT in production
        # return self._model.infer(model_input)

        # Simple reactive policy for testing
        action = np.zeros(self._action_dim)

        # Small random exploration
        action += np.random.randn(self._action_dim) * 0.01

        return action

    def _post_process(self, action: np.ndarray) -> np.ndarray:
        """Post-process action (clipping, smoothing)."""
        # Clip to max action norm
        norm = np.linalg.norm(action)
        if norm > self.config.max_action_norm:
            action = action * (self.config.max_action_norm / norm)

        # Simple exponential smoothing
        alpha = 0.8
        action = alpha * action + (1 - alpha) * self._last_action

        return action

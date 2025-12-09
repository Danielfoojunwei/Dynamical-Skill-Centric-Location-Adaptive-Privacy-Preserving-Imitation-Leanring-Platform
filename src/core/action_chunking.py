"""
Action Chunking Module

Provides temporal action chunking for robot control policies.
Action chunking groups multiple timesteps of actions into a single prediction,
which improves:
- Temporal coherence (smoother motion)
- Execution efficiency (fewer model calls)
- Learning from demonstrations (captures motion patterns)

Reference:
- ACT (Action Chunking Transformer): https://arxiv.org/abs/2304.13705
- Pi0: Physical Intelligence policy architecture
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class ActionChunkConfig:
    """Configuration for action chunking."""
    chunk_size: int = 16  # Number of actions per chunk (action_horizon)
    action_dim: int = 7   # Dimension of action space (e.g., 7 DOF arm)
    overlap: int = 8      # Overlap between consecutive chunks for smoothing
    execution_horizon: int = 8  # How many actions to execute before re-predicting
    temporal_smoothing: float = 0.3  # Exponential smoothing factor
    use_temporal_ensemble: bool = True  # Blend overlapping predictions


@dataclass
class ActionChunk:
    """A chunk of temporally coherent actions."""
    actions: np.ndarray  # Shape: [chunk_size, action_dim]
    timestamp: float
    confidence: float = 1.0
    chunk_id: int = 0

    @property
    def size(self) -> int:
        return len(self.actions)

    def get_action(self, step: int) -> np.ndarray:
        """Get action at specific step within chunk."""
        if step >= self.size:
            return self.actions[-1]  # Hold last action
        return self.actions[step]


class ActionChunkExecutor:
    """
    Executes action chunks with temporal smoothing and ensemble blending.

    This class handles:
    - Tracking position within current chunk
    - Blending overlapping predictions
    - Temporal smoothing for smooth motion
    - Re-triggering prediction when needed

    Usage:
        config = ActionChunkConfig(chunk_size=16, action_dim=7)
        executor = ActionChunkExecutor(config)

        # When receiving new prediction from model
        executor.set_chunk(chunk)

        # In control loop
        while running:
            action = executor.step()
            robot.send_action(action)

            if executor.needs_new_chunk():
                new_chunk = model.predict(observation)
                executor.set_chunk(new_chunk)
    """

    def __init__(self, config: Optional[ActionChunkConfig] = None):
        self.config = config or ActionChunkConfig()

        # Current chunk state
        self.current_chunk: Optional[ActionChunk] = None
        self.chunk_step: int = 0
        self.total_steps: int = 0

        # Temporal ensemble buffer
        self._prediction_buffer: deque = deque(maxlen=4)
        self._previous_action: Optional[np.ndarray] = None

        # Statistics
        self.stats = {
            'chunks_received': 0,
            'actions_executed': 0,
            'predictions_blended': 0,
        }

    def set_chunk(self, chunk: ActionChunk):
        """
        Set new action chunk from model prediction.

        If temporal ensemble is enabled, the new chunk is blended with
        previous overlapping predictions.
        """
        if self.config.use_temporal_ensemble and self.current_chunk is not None:
            # Store for blending
            self._prediction_buffer.append({
                'chunk': self.current_chunk,
                'start_step': self.total_steps - self.chunk_step,
                'chunk_step': self.chunk_step,
            })

        self.current_chunk = chunk
        self.chunk_step = 0
        self.stats['chunks_received'] += 1

        logger.debug(f"ActionChunkExecutor: New chunk {chunk.chunk_id}, size={chunk.size}")

    def step(self) -> np.ndarray:
        """
        Get next action to execute.

        Returns:
            action: np.ndarray of shape [action_dim]
        """
        if self.current_chunk is None:
            return np.zeros(self.config.action_dim)

        # Get base action from current chunk
        action = self.current_chunk.get_action(self.chunk_step)

        # Apply temporal ensemble if enabled
        if self.config.use_temporal_ensemble:
            action = self._apply_temporal_ensemble(action)

        # Apply temporal smoothing
        if self._previous_action is not None:
            alpha = self.config.temporal_smoothing
            action = alpha * action + (1 - alpha) * self._previous_action

        self._previous_action = action.copy()
        self.chunk_step += 1
        self.total_steps += 1
        self.stats['actions_executed'] += 1

        return action

    def _apply_temporal_ensemble(self, action: np.ndarray) -> np.ndarray:
        """Blend overlapping predictions using weighted average."""
        if len(self._prediction_buffer) == 0:
            return action

        blended = action.copy()
        total_weight = 1.0

        for pred in self._prediction_buffer:
            # Calculate overlap with current step
            pred_start = pred['start_step']
            pred_end = pred_start + pred['chunk'].size

            if pred_start <= self.total_steps < pred_end:
                # This prediction covers current step
                idx = self.total_steps - pred_start
                if idx < pred['chunk'].size:
                    pred_action = pred['chunk'].get_action(idx)

                    # Weight by recency (newer predictions get higher weight)
                    age = self.total_steps - pred_start
                    weight = np.exp(-0.1 * age)

                    blended = blended + weight * pred_action
                    total_weight += weight
                    self.stats['predictions_blended'] += 1

        return blended / total_weight

    def needs_new_chunk(self) -> bool:
        """Check if a new chunk should be predicted."""
        if self.current_chunk is None:
            return True

        # Re-predict when we've executed enough of current chunk
        return self.chunk_step >= self.config.execution_horizon

    def reset(self):
        """Reset executor state."""
        self.current_chunk = None
        self.chunk_step = 0
        self.total_steps = 0
        self._prediction_buffer.clear()
        self._previous_action = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            **self.stats,
            'current_chunk_step': self.chunk_step,
            'total_steps': self.total_steps,
            'buffer_size': len(self._prediction_buffer),
        }


class ActionChunkPredictor:
    """
    Wrapper for model that produces action chunks.

    Handles:
    - Preparing observations for model
    - Running inference
    - Post-processing predictions into chunks
    """

    def __init__(
        self,
        model,  # VLA model (Pi0, OpenVLA, ACT, etc.)
        config: Optional[ActionChunkConfig] = None
    ):
        self.model = model
        self.config = config or ActionChunkConfig()
        self._chunk_counter = 0

    def predict(
        self,
        observation: Dict[str, Any],
        instruction: str = ""
    ) -> ActionChunk:
        """
        Predict action chunk from observation.

        Args:
            observation: Dict with 'image', 'proprio', etc.
            instruction: Language instruction (for VLA models)

        Returns:
            ActionChunk with predicted actions
        """
        start_time = time.time()

        # Run model inference
        if hasattr(self.model, 'sample_actions'):
            # Pi0-style interface
            actions = self.model.sample_actions(
                images=observation.get('image'),
                instruction=instruction,
                proprio=observation.get('proprio'),
            )
            if hasattr(actions, 'cpu'):
                actions = actions.cpu().numpy()
        elif hasattr(self.model, 'predict'):
            # VendorAdapter interface
            result = self.model.predict(observation)
            actions = result.get('action', np.zeros(self.config.action_dim))
            # Expand single action to chunk
            if actions.ndim == 1:
                actions = np.tile(actions, (self.config.chunk_size, 1))
        else:
            # Fallback
            actions = np.zeros((self.config.chunk_size, self.config.action_dim))

        # Ensure correct shape
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[0] < self.config.chunk_size:
            # Pad with last action
            padding = np.tile(actions[-1:], (self.config.chunk_size - actions.shape[0], 1))
            actions = np.vstack([actions, padding])

        self._chunk_counter += 1

        inference_time = time.time() - start_time
        logger.debug(f"ActionChunkPredictor: Predicted chunk in {inference_time*1000:.1f}ms")

        return ActionChunk(
            actions=actions[:self.config.chunk_size],
            timestamp=start_time,
            confidence=1.0,
            chunk_id=self._chunk_counter,
        )


class ActionChunkBuffer:
    """
    Buffer for storing and retrieving action chunks.

    Useful for:
    - Caching predictions during high-latency inference
    - Replaying demonstrations
    - Temporal analysis
    """

    def __init__(self, max_size: int = 100):
        self.buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, chunk: ActionChunk):
        """Add chunk to buffer."""
        self.buffer.append(chunk)

    def get_latest(self) -> Optional[ActionChunk]:
        """Get most recent chunk."""
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1]

    def get_sequence(self, n: int) -> List[ActionChunk]:
        """Get last n chunks."""
        return list(self.buffer)[-n:]

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Testing
# =============================================================================

def test_action_chunking():
    """Test action chunking module."""
    print("\n" + "=" * 60)
    print("ACTION CHUNKING TEST")
    print("=" * 60)

    config = ActionChunkConfig(
        chunk_size=16,
        action_dim=7,
        execution_horizon=8,
        temporal_smoothing=0.3,
        use_temporal_ensemble=True,
    )

    executor = ActionChunkExecutor(config)

    # Test 1: Basic chunk execution
    print("\n1. Basic Chunk Execution")
    print("-" * 40)

    chunk = ActionChunk(
        actions=np.random.randn(16, 7),
        timestamp=time.time(),
        chunk_id=1,
    )
    executor.set_chunk(chunk)

    actions = []
    for i in range(10):
        action = executor.step()
        actions.append(action)
        print(f"   Step {i}: action[0]={action[0]:.4f}, needs_new={executor.needs_new_chunk()}")

    print(f"   Executed {len(actions)} actions")

    # Test 2: Temporal smoothing
    print("\n2. Temporal Smoothing")
    print("-" * 40)

    executor.reset()
    chunk1 = ActionChunk(actions=np.ones((16, 7)) * 0.0, timestamp=time.time(), chunk_id=1)
    executor.set_chunk(chunk1)

    # Execute half
    for _ in range(8):
        executor.step()

    # New chunk with different values
    chunk2 = ActionChunk(actions=np.ones((16, 7)) * 1.0, timestamp=time.time(), chunk_id=2)
    executor.set_chunk(chunk2)

    action = executor.step()
    print(f"   After switch: action[0]={action[0]:.4f} (should be smoothed)")

    # Test 3: Statistics
    print("\n3. Statistics")
    print("-" * 40)
    stats = executor.get_statistics()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("ACTION CHUNKING TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_action_chunking()

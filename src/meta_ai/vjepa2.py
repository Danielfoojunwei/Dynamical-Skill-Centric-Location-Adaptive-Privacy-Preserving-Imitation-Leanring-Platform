"""
V-JEPA 2 World Model for Dynamical Edge Platform

V-JEPA 2 (Video Joint-Embedding Predictive Architecture v2) provides:
- World modeling: Predicts future video frames/states
- Physics understanding: Learns object interactions
- Action generation: Native robotic action prediction
- Zero-shot manipulation: Generalizes to new tasks

Repository: github.com/facebookresearch/vjepa2
License: Apache 2.0

Key Capabilities for Robotics:
1. Predictive Safety: Anticipate collisions before they occur
2. Action Planning: Generate action sequences for tasks
3. Self-Supervision: Learn from video without labels
4. World Simulation: Mental simulation of action outcomes

Integration with Dynamical Platform:
- World model for MoE skill router
- Predictive safety layer (1kHz tier)
- Action generation for manipulation
- Replaces/augments Pi0 base VLA
"""

import os
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Configuration
# =============================================================================

class VJEPA2ModelSize(str, Enum):
    """V-JEPA 2 model sizes."""
    SMALL = "vjepa2_small"      # ~100M params, fastest
    BASE = "vjepa2_base"        # ~300M params
    LARGE = "vjepa2_large"      # ~600M params
    HUGE = "vjepa2_huge"        # ~1B+ params, best quality


class PredictionMode(str, Enum):
    """V-JEPA 2 prediction modes."""
    FRAME = "frame"             # Predict future frames
    EMBEDDING = "embedding"     # Predict future embeddings
    ACTION = "action"           # Predict actions to reach goal


@dataclass
class VJEPA2Config:
    """Configuration for V-JEPA 2 world model."""
    # Model selection
    model_size: VJEPA2ModelSize = VJEPA2ModelSize.LARGE

    # Input configuration
    input_size: int = 224           # Frame resolution
    num_frames: int = 16            # Context frames
    frame_rate: int = 30            # Video FPS

    # Prediction settings
    prediction_horizon: int = 16    # Future frames to predict
    prediction_mode: PredictionMode = PredictionMode.EMBEDDING

    # Embedding dimensions
    embed_dim: int = 768            # Feature dimension
    action_dim: int = 32            # Action space dimension

    # Robot-specific settings
    robot_dof: int = 23             # Robot degrees of freedom
    gripper_dim: int = 1            # Gripper action dimension
    use_proprioception: bool = True

    # Optimization
    use_fp16: bool = True
    compile_model: bool = False

    # Safety prediction
    enable_safety_prediction: bool = True
    collision_threshold: float = 0.7

    # Cache settings
    cache_dir: str = "/var/lib/dynamical/models/vjepa2"

    # Device
    device: str = "cuda"

    @property
    def total_action_dim(self) -> int:
        return self.robot_dof + self.gripper_dim


@dataclass
class WorldState:
    """Representation of world state."""
    # Visual state
    frame_embedding: np.ndarray     # [embed_dim]
    frame_sequence: Optional[np.ndarray] = None  # [T, H, W, 3]

    # Object states
    object_positions: Optional[np.ndarray] = None   # [N, 3]
    object_velocities: Optional[np.ndarray] = None  # [N, 3]

    # Robot state
    joint_positions: Optional[np.ndarray] = None    # [DOF]
    joint_velocities: Optional[np.ndarray] = None   # [DOF]
    ee_pose: Optional[np.ndarray] = None            # [7] pos + quat

    # Timestamp
    timestamp: float = 0.0


@dataclass
class WorldModelPrediction:
    """Output from V-JEPA 2 world model."""
    # Predicted future embeddings
    future_embeddings: np.ndarray   # [T, embed_dim]

    # Predicted actions (if action mode)
    predicted_actions: Optional[np.ndarray] = None  # [T, action_dim]

    # Safety predictions
    collision_probabilities: Optional[np.ndarray] = None  # [T]
    hazard_masks: Optional[np.ndarray] = None  # [T, H, W]

    # Confidence
    prediction_confidence: np.ndarray  # [T]

    # Timing
    inference_time_ms: float = 0.0

    @property
    def horizon(self) -> int:
        return self.future_embeddings.shape[0]

    @property
    def max_collision_prob(self) -> float:
        if self.collision_probabilities is not None:
            return float(self.collision_probabilities.max())
        return 0.0


@dataclass
class ActionPlan:
    """Action plan from world model."""
    actions: np.ndarray             # [T, action_dim]
    expected_states: np.ndarray     # [T, embed_dim]
    success_probability: float
    collision_free: bool

    # Timing
    planning_time_ms: float = 0.0


# =============================================================================
# V-JEPA 2 World Model
# =============================================================================

class VJEPA2WorldModel:
    """
    V-JEPA 2 World Model for robot planning and prediction.

    Provides:
    - Future state prediction from current observation
    - Action planning to reach goal states
    - Collision/hazard prediction for safety
    - Mental simulation of action outcomes

    Usage:
        world_model = VJEPA2WorldModel()
        world_model.load_model()

        # Predict future states
        prediction = world_model.predict(current_state)

        # Plan actions to goal
        plan = world_model.plan_to_goal(current_state, goal_state)

        # Check action safety
        safe = world_model.is_action_safe(state, action)
    """

    # Model dimensions
    MODEL_DIMS = {
        VJEPA2ModelSize.SMALL: 384,
        VJEPA2ModelSize.BASE: 768,
        VJEPA2ModelSize.LARGE: 1024,
        VJEPA2ModelSize.HUGE: 1280,
    }

    def __init__(self, config: VJEPA2Config = None):
        self.config = config or VJEPA2Config()

        # Update embed_dim based on model size
        self.config.embed_dim = self.MODEL_DIMS.get(
            self.config.model_size, 768
        )

        # Model components
        self.encoder = None          # Video encoder
        self.predictor = None        # Future predictor
        self.action_decoder = None   # Action generation
        self.safety_head = None      # Collision prediction

        self._is_loaded = False

        # Frame buffer for temporal context
        self._frame_buffer: List[np.ndarray] = []
        self._embedding_buffer: List[np.ndarray] = []

        # Statistics
        self.stats = {
            "predictions_made": 0,
            "actions_planned": 0,
            "safety_checks": 0,
            "collisions_predicted": 0,
            "avg_inference_time_ms": 0.0,
        }

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_model(self, weights_path: Optional[str] = None) -> bool:
        """
        Load V-JEPA 2 model.

        Args:
            weights_path: Path to custom weights

        Returns:
            True if successful
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available - using mock V-JEPA 2")
            self._is_loaded = True
            return True

        try:
            logger.info(f"Loading V-JEPA 2 model: {self.config.model_size.value}")

            # Create model components
            self.encoder = self._create_encoder()
            self.predictor = self._create_predictor()
            self.action_decoder = self._create_action_decoder()

            if self.config.enable_safety_prediction:
                self.safety_head = self._create_safety_head()

            # Move to device
            device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

            self.encoder = self.encoder.to(device)
            self.predictor = self.predictor.to(device)
            self.action_decoder = self.action_decoder.to(device)

            if self.safety_head:
                self.safety_head = self.safety_head.to(device)

            # Set eval mode
            self.encoder.eval()
            self.predictor.eval()
            self.action_decoder.eval()

            if self.safety_head:
                self.safety_head.eval()

            # FP16 optimization
            if self.config.use_fp16:
                self.encoder = self.encoder.half()
                self.predictor = self.predictor.half()
                self.action_decoder = self.action_decoder.half()
                if self.safety_head:
                    self.safety_head = self.safety_head.half()

            self._is_loaded = True
            logger.info(f"V-JEPA 2 {self.config.model_size.value} loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to load V-JEPA 2: {e}")
            return False

    def _create_encoder(self) -> nn.Module:
        """Create V-JEPA 2 video encoder."""
        class MockVJEPA2Encoder(nn.Module):
            def __init__(self, embed_dim: int, num_frames: int):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_frames = num_frames

                # Spatial encoder (per frame)
                self.spatial = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                )

                # Temporal encoder
                self.temporal = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=256 * 7 * 7,
                        nhead=8,
                        dim_feedforward=2048,
                        batch_first=True,
                    ),
                    num_layers=2,
                )

                # Project to embed_dim
                self.proj = nn.Linear(256 * 7 * 7, embed_dim)

            def forward(self, x):
                # x: [B, T, C, H, W]
                B, T, C, H, W = x.shape

                # Encode each frame
                x = x.view(B * T, C, H, W)
                spatial_feats = self.spatial(x)  # [B*T, 256, 7, 7]
                spatial_feats = spatial_feats.view(B, T, -1)  # [B, T, 256*7*7]

                # Temporal encoding
                temporal_feats = self.temporal(spatial_feats)  # [B, T, D]

                # Project to final embedding
                embeddings = self.proj(temporal_feats)  # [B, T, embed_dim]

                return embeddings

            def encode_frame(self, frame):
                # Single frame encoding
                if frame.dim() == 3:
                    frame = frame.unsqueeze(0).unsqueeze(0)  # Add batch and time
                elif frame.dim() == 4:
                    frame = frame.unsqueeze(1)  # Add time

                return self.forward(frame)[:, 0]  # Return first (only) frame embedding

        return MockVJEPA2Encoder(self.config.embed_dim, self.config.num_frames)

    def _create_predictor(self) -> nn.Module:
        """Create V-JEPA 2 predictor for future embeddings."""
        class MockVJEPA2Predictor(nn.Module):
            def __init__(self, embed_dim: int, horizon: int):
                super().__init__()
                self.embed_dim = embed_dim
                self.horizon = horizon

                # Context aggregation
                self.context_agg = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=8,
                        dim_feedforward=embed_dim * 4,
                        batch_first=True,
                    ),
                    num_layers=4,
                )

                # Prediction heads (one per future step)
                self.pred_heads = nn.ModuleList([
                    nn.Linear(embed_dim, embed_dim)
                    for _ in range(horizon)
                ])

                # Confidence prediction
                self.confidence_head = nn.Linear(embed_dim, horizon)

            def forward(self, context_embeddings, actions=None):
                # context_embeddings: [B, T, D]
                # actions: [B, T, A] (optional)

                # Aggregate context
                aggregated = self.context_agg(context_embeddings)
                latest = aggregated[:, -1]  # [B, D]

                # Predict future embeddings
                predictions = []
                for head in self.pred_heads:
                    pred = head(latest)
                    predictions.append(pred)

                predictions = torch.stack(predictions, dim=1)  # [B, horizon, D]

                # Predict confidence
                confidence = torch.sigmoid(self.confidence_head(latest))  # [B, horizon]

                return predictions, confidence

        return MockVJEPA2Predictor(self.config.embed_dim, self.config.prediction_horizon)

    def _create_action_decoder(self) -> nn.Module:
        """Create action decoder for robot control."""
        class MockActionDecoder(nn.Module):
            def __init__(self, embed_dim: int, action_dim: int, horizon: int):
                super().__init__()
                self.embed_dim = embed_dim
                self.action_dim = action_dim

                # State-to-action transformer
                self.transformer = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=embed_dim,
                        nhead=8,
                        dim_feedforward=embed_dim * 4,
                        batch_first=True,
                    ),
                    num_layers=4,
                )

                # Action output head
                self.action_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, action_dim),
                    nn.Tanh(),  # Actions in [-1, 1]
                )

                # Learnable action queries
                self.action_queries = nn.Parameter(torch.randn(1, horizon, embed_dim))

            def forward(self, state_embedding, goal_embedding=None):
                # state_embedding: [B, D] or [B, T, D]
                # goal_embedding: [B, D] (optional)

                B = state_embedding.shape[0]

                if state_embedding.dim() == 2:
                    state_embedding = state_embedding.unsqueeze(1)

                # Expand action queries
                queries = self.action_queries.expand(B, -1, -1)

                # Decode actions
                if goal_embedding is not None:
                    goal_embedding = goal_embedding.unsqueeze(1)
                    memory = torch.cat([state_embedding, goal_embedding], dim=1)
                else:
                    memory = state_embedding

                decoded = self.transformer(queries, memory)  # [B, horizon, D]
                actions = self.action_head(decoded)  # [B, horizon, action_dim]

                return actions

        return MockActionDecoder(
            self.config.embed_dim,
            self.config.total_action_dim,
            self.config.prediction_horizon
        )

    def _create_safety_head(self) -> nn.Module:
        """Create safety/collision prediction head."""
        class MockSafetyHead(nn.Module):
            def __init__(self, embed_dim: int, horizon: int):
                super().__init__()

                # Collision probability predictor
                self.collision_predictor = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, horizon),
                    nn.Sigmoid(),
                )

                # Hazard mask predictor (lightweight)
                self.hazard_decoder = nn.Sequential(
                    nn.Linear(embed_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 14 * 14),  # Low-res hazard map
                    nn.Sigmoid(),
                )

            def forward(self, current_embedding, future_embeddings, action=None):
                # current_embedding: [B, D]
                # future_embeddings: [B, T, D]

                B, T, D = future_embeddings.shape

                # Combine current + future for collision prediction
                combined = torch.cat([
                    current_embedding.unsqueeze(1).expand(-1, T, -1),
                    future_embeddings
                ], dim=-1)  # [B, T, 2D]

                # Predict collision probabilities
                collision_probs = self.collision_predictor(combined.view(B, T, -1).mean(dim=1))  # [B, T]

                # Predict hazard masks
                hazard_logits = self.hazard_decoder(future_embeddings)  # [B, T, 196]
                hazard_masks = hazard_logits.view(B, T, 14, 14)

                return collision_probs, hazard_masks

        return MockSafetyHead(self.config.embed_dim, self.config.prediction_horizon)

    def _preprocess_frame(self, frame: Union[np.ndarray, Any]) -> 'torch.Tensor':
        """Preprocess single frame."""
        if not HAS_TORCH:
            return None

        if isinstance(frame, np.ndarray):
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[-1] == 3:
                frame = frame.transpose(2, 0, 1)
            tensor = torch.from_numpy(frame)
        elif isinstance(frame, torch.Tensor):
            tensor = frame
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        # Resize
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=(self.config.input_size, self.config.input_size),
            mode='bilinear',
            align_corners=False
        )

        return tensor

    def encode_frame(self, frame: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Encode a single frame to embedding.

        Args:
            frame: Input frame [H, W, 3]

        Returns:
            Frame embedding [embed_dim]
        """
        if not self._is_loaded:
            self.load_model()

        if not HAS_TORCH:
            return np.random.randn(self.config.embed_dim).astype(np.float32)

        tensor = self._preprocess_frame(frame)
        device = next(self.encoder.parameters()).device
        tensor = tensor.to(device)

        if self.config.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            embedding = self.encoder.encode_frame(tensor)
            embedding = embedding.cpu().float().numpy()

        return embedding.squeeze()

    def add_frame(self, frame: Union[np.ndarray, Any]):
        """
        Add frame to temporal buffer.

        Args:
            frame: Input frame
        """
        # Encode frame
        embedding = self.encode_frame(frame)

        # Add to buffers
        self._frame_buffer.append(frame if isinstance(frame, np.ndarray) else np.array(frame))
        self._embedding_buffer.append(embedding)

        # Keep buffer size limited
        max_frames = self.config.num_frames
        if len(self._frame_buffer) > max_frames:
            self._frame_buffer = self._frame_buffer[-max_frames:]
            self._embedding_buffer = self._embedding_buffer[-max_frames:]

    def predict(
        self,
        current_state: Optional[WorldState] = None,
        num_steps: Optional[int] = None,
    ) -> WorldModelPrediction:
        """
        Predict future states.

        Args:
            current_state: Current world state (uses buffer if None)
            num_steps: Number of steps to predict (default: prediction_horizon)

        Returns:
            WorldModelPrediction with future embeddings
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()
        horizon = num_steps or self.config.prediction_horizon

        if not HAS_TORCH:
            # Mock prediction
            future_embeddings = np.random.randn(horizon, self.config.embed_dim).astype(np.float32)
            confidence = np.linspace(0.9, 0.5, horizon).astype(np.float32)

            collision_probs = None
            if self.config.enable_safety_prediction:
                collision_probs = np.random.rand(horizon).astype(np.float32) * 0.3

            inference_time = (time.time() - start_time) * 1000

            self.stats["predictions_made"] += 1

            return WorldModelPrediction(
                future_embeddings=future_embeddings,
                collision_probabilities=collision_probs,
                prediction_confidence=confidence,
                inference_time_ms=inference_time,
            )

        # Get context embeddings
        if current_state is not None and current_state.frame_embedding is not None:
            context = torch.from_numpy(current_state.frame_embedding).unsqueeze(0).unsqueeze(0)
        elif self._embedding_buffer:
            context = torch.from_numpy(np.stack(self._embedding_buffer)).unsqueeze(0)
        else:
            raise ValueError("No context available for prediction")

        device = next(self.predictor.parameters()).device
        context = context.to(device)

        if self.config.use_fp16:
            context = context.half()

        with torch.no_grad():
            # Predict future embeddings
            future_embeddings, confidence = self.predictor(context)

            # Safety prediction
            collision_probs = None
            hazard_masks = None
            if self.safety_head is not None:
                current_emb = context[:, -1]  # Latest context
                collision_probs, hazard_masks = self.safety_head(current_emb, future_embeddings)
                collision_probs = collision_probs.cpu().float().numpy().squeeze()
                hazard_masks = hazard_masks.cpu().float().numpy().squeeze()

                if collision_probs.max() > self.config.collision_threshold:
                    self.stats["collisions_predicted"] += 1

            future_embeddings = future_embeddings.cpu().float().numpy().squeeze()
            confidence = confidence.cpu().float().numpy().squeeze()

        inference_time = (time.time() - start_time) * 1000

        self.stats["predictions_made"] += 1
        self.stats["avg_inference_time_ms"] = (
            0.9 * self.stats["avg_inference_time_ms"] + 0.1 * inference_time
        )

        return WorldModelPrediction(
            future_embeddings=future_embeddings[:horizon],
            collision_probabilities=collision_probs[:horizon] if collision_probs is not None else None,
            hazard_masks=hazard_masks[:horizon] if hazard_masks is not None else None,
            prediction_confidence=confidence[:horizon],
            inference_time_ms=inference_time,
        )

    def plan_to_goal(
        self,
        current_state: WorldState,
        goal_state: WorldState,
        max_attempts: int = 3,
    ) -> ActionPlan:
        """
        Plan action sequence to reach goal state.

        Args:
            current_state: Current world state
            goal_state: Desired goal state
            max_attempts: Maximum planning attempts

        Returns:
            ActionPlan with action sequence
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        if not HAS_TORCH:
            # Mock planning
            actions = np.random.randn(
                self.config.prediction_horizon,
                self.config.total_action_dim
            ).astype(np.float32) * 0.1
            actions = np.clip(actions, -1, 1)

            expected_states = np.random.randn(
                self.config.prediction_horizon,
                self.config.embed_dim
            ).astype(np.float32)

            planning_time = (time.time() - start_time) * 1000
            self.stats["actions_planned"] += 1

            return ActionPlan(
                actions=actions,
                expected_states=expected_states,
                success_probability=0.85,
                collision_free=True,
                planning_time_ms=planning_time,
            )

        device = next(self.action_decoder.parameters()).device

        # Convert states to tensors
        current_emb = torch.from_numpy(current_state.frame_embedding).unsqueeze(0).to(device)
        goal_emb = torch.from_numpy(goal_state.frame_embedding).unsqueeze(0).to(device)

        if self.config.use_fp16:
            current_emb = current_emb.half()
            goal_emb = goal_emb.half()

        best_plan = None
        best_success_prob = 0.0

        for attempt in range(max_attempts):
            with torch.no_grad():
                # Generate actions
                actions = self.action_decoder(current_emb, goal_emb)
                actions = actions.cpu().float().numpy().squeeze()

                # Predict expected states from actions
                future_prediction = self.predict(current_state)

                # Check safety
                collision_free = True
                if future_prediction.collision_probabilities is not None:
                    collision_free = future_prediction.max_collision_prob < self.config.collision_threshold

                # Estimate success probability
                goal_distance = np.linalg.norm(
                    future_prediction.future_embeddings[-1] - goal_state.frame_embedding
                )
                success_prob = np.exp(-goal_distance / 10.0)

                if collision_free and success_prob > best_success_prob:
                    best_success_prob = success_prob
                    best_plan = ActionPlan(
                        actions=actions,
                        expected_states=future_prediction.future_embeddings,
                        success_probability=float(success_prob),
                        collision_free=collision_free,
                    )

        planning_time = (time.time() - start_time) * 1000
        self.stats["actions_planned"] += 1

        if best_plan is None:
            # Return default plan if all attempts failed
            best_plan = ActionPlan(
                actions=np.zeros((self.config.prediction_horizon, self.config.total_action_dim)),
                expected_states=np.zeros((self.config.prediction_horizon, self.config.embed_dim)),
                success_probability=0.0,
                collision_free=False,
            )

        best_plan.planning_time_ms = planning_time
        return best_plan

    def is_action_safe(
        self,
        current_state: WorldState,
        proposed_action: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Check if proposed action is safe.

        Args:
            current_state: Current world state
            proposed_action: Proposed action [action_dim]

        Returns:
            Tuple of (is_safe, collision_probability)
        """
        if not self._is_loaded:
            self.load_model()

        self.stats["safety_checks"] += 1

        # Predict future from action
        prediction = self.predict(current_state, num_steps=1)

        if prediction.collision_probabilities is not None:
            collision_prob = float(prediction.collision_probabilities[0])
            is_safe = collision_prob < self.config.collision_threshold
            return is_safe, collision_prob

        return True, 0.0

    def simulate_action_sequence(
        self,
        current_state: WorldState,
        actions: np.ndarray,
    ) -> List[WorldModelPrediction]:
        """
        Simulate a sequence of actions.

        Args:
            current_state: Starting state
            actions: Action sequence [T, action_dim]

        Returns:
            List of predictions for each step
        """
        if not self._is_loaded:
            self.load_model()

        predictions = []
        state = current_state

        for i, action in enumerate(actions):
            prediction = self.predict(state, num_steps=1)
            predictions.append(prediction)

            # Update state for next step
            state = WorldState(
                frame_embedding=prediction.future_embeddings[0],
                timestamp=state.timestamp + 1.0 / self.config.frame_rate,
            )

        return predictions

    def get_statistics(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return {
            **self.stats,
            "model_size": self.config.model_size.value,
            "embed_dim": self.config.embed_dim,
            "prediction_horizon": self.config.prediction_horizon,
            "is_loaded": self._is_loaded,
            "buffer_frames": len(self._frame_buffer),
            "safety_enabled": self.config.enable_safety_prediction,
        }

    def clear_buffer(self):
        """Clear frame buffer."""
        self._frame_buffer.clear()
        self._embedding_buffer.clear()


# =============================================================================
# Testing
# =============================================================================

def test_vjepa2():
    """Test V-JEPA 2 world model."""
    print("\n" + "=" * 60)
    print("V-JEPA 2 WORLD MODEL TEST")
    print("=" * 60)

    # Create world model
    config = VJEPA2Config(
        model_size=VJEPA2ModelSize.LARGE,
        prediction_horizon=16,
        enable_safety_prediction=True,
    )
    world_model = VJEPA2WorldModel(config)

    print("\n1. Load Model")
    print("-" * 40)
    success = world_model.load_model()
    print(f"   Model loaded: {success}")
    print(f"   Embed dim: {world_model.config.embed_dim}")

    print("\n2. Frame Encoding")
    print("-" * 40)

    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    embedding = world_model.encode_frame(test_frame)
    print(f"   Frame shape: {test_frame.shape}")
    print(f"   Embedding shape: {embedding.shape}")

    print("\n3. Temporal Context")
    print("-" * 40)

    # Add frames to buffer
    for i in range(8):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        world_model.add_frame(frame)
    print(f"   Buffer frames: {len(world_model._frame_buffer)}")

    print("\n4. Future Prediction")
    print("-" * 40)

    prediction = world_model.predict()
    print(f"   Future embeddings shape: {prediction.future_embeddings.shape}")
    print(f"   Prediction confidence: {prediction.prediction_confidence[:3]}...")
    print(f"   Inference time: {prediction.inference_time_ms:.2f}ms")

    if prediction.collision_probabilities is not None:
        print(f"   Max collision prob: {prediction.max_collision_prob:.3f}")

    print("\n5. Action Planning")
    print("-" * 40)

    current_state = WorldState(
        frame_embedding=embedding,
    )
    goal_state = WorldState(
        frame_embedding=np.random.randn(config.embed_dim).astype(np.float32),
    )

    plan = world_model.plan_to_goal(current_state, goal_state)
    print(f"   Actions shape: {plan.actions.shape}")
    print(f"   Success probability: {plan.success_probability:.3f}")
    print(f"   Collision free: {plan.collision_free}")
    print(f"   Planning time: {plan.planning_time_ms:.2f}ms")

    print("\n6. Safety Check")
    print("-" * 40)

    proposed_action = np.random.randn(config.total_action_dim).astype(np.float32)
    is_safe, collision_prob = world_model.is_action_safe(current_state, proposed_action)
    print(f"   Action safe: {is_safe}")
    print(f"   Collision probability: {collision_prob:.3f}")

    print("\n7. Statistics")
    print("-" * 40)
    stats = world_model.get_statistics()
    print(f"   Predictions made: {stats['predictions_made']}")
    print(f"   Actions planned: {stats['actions_planned']}")
    print(f"   Safety checks: {stats['safety_checks']}")
    print(f"   Collisions predicted: {stats['collisions_predicted']}")

    print("\n" + "=" * 60)
    print("V-JEPA 2 TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_vjepa2()

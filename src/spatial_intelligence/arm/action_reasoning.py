"""
Action Reasoning Module - Chain-of-Thought for Robot Actions

Generates explicit reasoning traces for robot actions:
1. Perception reasoning: "I see a red cup on the left"
2. Spatial reasoning: "The cup is 0.3m away, within reach"
3. Action reasoning: "I will move my gripper to grasp it"

This provides INTERPRETABILITY beyond just trajectory visualization
and enables better generalization through explicit reasoning.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Local imports
from .trajectory_trace import TrajectoryTrace


@dataclass
class ActionReasoningConfig:
    """Configuration for action reasoning module."""
    # Input dimensions
    vision_dim: int = 1024
    depth_dim: int = 256
    instruction_dim: int = 512

    # Model architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4

    # Reasoning generation
    max_perception_tokens: int = 64
    max_spatial_tokens: int = 48
    max_action_tokens: int = 48

    # Vocabulary (simplified for demo - real impl uses tokenizer)
    vocab_size: int = 10000

    # Device
    device: str = "cuda"


@dataclass
class ReasoningOutput:
    """Output from action reasoning module."""
    # Reasoning text
    perception_reasoning: str
    spatial_reasoning: str
    action_reasoning: str

    # Confidence scores
    perception_confidence: float = 1.0
    spatial_confidence: float = 1.0
    action_confidence: float = 1.0

    # Detected objects (from perception reasoning)
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)

    # Spatial relations
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)

    # Planned actions
    planned_actions: List[str] = field(default_factory=list)

    # Timing
    inference_time_ms: float = 0.0

    def full_reasoning(self) -> str:
        """Get complete reasoning trace."""
        return (
            f"Perception: {self.perception_reasoning}\n"
            f"Spatial: {self.spatial_reasoning}\n"
            f"Action: {self.action_reasoning}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "perception_reasoning": self.perception_reasoning,
            "spatial_reasoning": self.spatial_reasoning,
            "action_reasoning": self.action_reasoning,
            "perception_confidence": self.perception_confidence,
            "spatial_confidence": self.spatial_confidence,
            "action_confidence": self.action_confidence,
            "detected_objects": self.detected_objects,
            "spatial_relations": self.spatial_relations,
            "planned_actions": self.planned_actions,
            "inference_time_ms": self.inference_time_ms,
        }


class ReasoningEncoder(nn.Module):
    """Encode visual and depth features for reasoning."""

    def __init__(self, config: ActionReasoningConfig):
        super().__init__()
        self.config = config

        # Vision projection
        self.vision_proj = nn.Linear(config.vision_dim, config.hidden_dim)

        # Depth projection
        self.depth_proj = nn.Linear(config.depth_dim, config.hidden_dim)

        # Instruction projection
        self.instruction_proj = nn.Linear(config.instruction_dim, config.hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        depth_features: Optional[torch.Tensor],
        instruction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode multimodal features.

        Args:
            vision_features: [B, N_v, D_v] or [B, D_v]
            depth_features: [B, N_d, D_d] or None
            instruction: [B, D_i]

        Returns:
            Encoded features [B, N, H]
        """
        # Project vision
        if vision_features.ndim == 2:
            vision_features = vision_features.unsqueeze(1)
        vision = self.vision_proj(vision_features)

        # Project instruction
        instr = self.instruction_proj(instruction).unsqueeze(1)

        # Combine features
        if depth_features is not None:
            if depth_features.ndim == 2:
                depth_features = depth_features.unsqueeze(1)
            depth = self.depth_proj(depth_features)
            features = torch.cat([vision, depth, instr], dim=1)
        else:
            features = torch.cat([vision, instr], dim=1)

        # Self-attention
        attended, _ = self.cross_attention(features, features, features)
        output = self.norm(features + attended)

        return output


class ReasoningDecoder(nn.Module):
    """Generate reasoning text from encoded features."""

    def __init__(self, config: ActionReasoningConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            config.hidden_dim, config.num_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        max_length: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reasoning tokens.

        Args:
            memory: Encoded features [B, N, H]
            target_ids: Teacher forcing targets [B, T] (training only)
            max_length: Maximum generation length

        Returns:
            logits: [B, T, V]
            hidden: [B, T, H]
        """
        B = memory.size(0)
        device = memory.device

        if target_ids is not None:
            # Teacher forcing
            embedded = self.embedding(target_ids)
            decoded = self.decoder(embedded, memory)
            logits = self.output_proj(decoded)
            return logits, decoded
        else:
            # Autoregressive generation
            current_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
            all_logits = []

            for _ in range(max_length):
                embedded = self.embedding(current_ids)
                decoded = self.decoder(embedded, memory)
                logits = self.output_proj(decoded[:, -1:, :])
                all_logits.append(logits)

                # Greedy decoding
                next_token = logits.argmax(dim=-1)
                current_ids = torch.cat([current_ids, next_token], dim=1)

            return torch.cat(all_logits, dim=1), decoded


class ActionReasoningModule(nn.Module):
    """
    Complete action reasoning module.

    Generates three types of reasoning:
    1. Perception: What the robot sees
    2. Spatial: Geometric understanding
    3. Action: What the robot will do
    """

    def __init__(self, config: Optional[ActionReasoningConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ActionReasoningModule")

        super().__init__()
        self.config = config or ActionReasoningConfig()

        # Shared encoder
        self.encoder = ReasoningEncoder(self.config)

        # Three separate decoders for different reasoning types
        self.perception_decoder = ReasoningDecoder(self.config)
        self.spatial_decoder = ReasoningDecoder(self.config)
        self.action_decoder = ReasoningDecoder(self.config)

        # Confidence heads
        self.perception_conf = nn.Linear(self.config.hidden_dim, 1)
        self.spatial_conf = nn.Linear(self.config.hidden_dim, 1)
        self.action_conf = nn.Linear(self.config.hidden_dim, 1)

        # Simple word list for demo (real impl uses proper tokenizer)
        self._init_vocab()

    def _init_vocab(self):
        """Initialize simple vocabulary for demo."""
        self.vocab = {
            0: "<pad>", 1: "<sos>", 2: "<eos>",
            3: "I", 4: "see", 5: "a", 6: "the",
            7: "red", 8: "blue", 9: "green", 10: "object",
            11: "cup", 12: "box", 13: "table", 14: "on",
            15: "left", 16: "right", 17: "front", 18: "is",
            19: "near", 20: "far", 21: "meters", 22: "away",
            23: "will", 24: "move", 25: "grasp", 26: "place",
            27: "gripper", 28: "to", 29: "towards", 30: "reach",
            31: "pick", 32: "up", 33: "put", 34: "down",
            35: "approximately", 36: "about", 37: "within", 38: "range",
        }
        self.vocab_inv = {v: k for k, v in self.vocab.items()}

    def forward(
        self,
        vision_features: torch.Tensor,
        instruction: torch.Tensor,
        depth_features: Optional[torch.Tensor] = None,
        target_perception: Optional[torch.Tensor] = None,
        target_spatial: Optional[torch.Tensor] = None,
        target_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Returns:
            perception_logits, spatial_logits, action_logits
        """
        # Encode
        memory = self.encoder(vision_features, depth_features, instruction)

        # Decode each type
        perception_logits, _ = self.perception_decoder(
            memory, target_perception, self.config.max_perception_tokens
        )
        spatial_logits, _ = self.spatial_decoder(
            memory, target_spatial, self.config.max_spatial_tokens
        )
        action_logits, _ = self.action_decoder(
            memory, target_action, self.config.max_action_tokens
        )

        return perception_logits, spatial_logits, action_logits

    @torch.no_grad()
    def generate_reasoning(
        self,
        vision_features: torch.Tensor,
        instruction: torch.Tensor,
        depth_features: Optional[torch.Tensor] = None,
        instruction_text: Optional[str] = None,
    ) -> ReasoningOutput:
        """
        Generate reasoning output.

        Args:
            vision_features: Vision embeddings
            instruction: Instruction embedding
            depth_features: Optional depth embeddings
            instruction_text: Text instruction for context

        Returns:
            ReasoningOutput with all reasoning traces
        """
        self.eval()
        start_time = time.time()

        # Encode
        memory = self.encoder(vision_features, depth_features, instruction)

        # Generate each reasoning type
        perception_logits, perception_hidden = self.perception_decoder(
            memory, max_length=self.config.max_perception_tokens
        )
        spatial_logits, spatial_hidden = self.spatial_decoder(
            memory, max_length=self.config.max_spatial_tokens
        )
        action_logits, action_hidden = self.action_decoder(
            memory, max_length=self.config.max_action_tokens
        )

        # Decode tokens to text
        perception_text = self._decode_tokens(perception_logits.argmax(dim=-1)[0])
        spatial_text = self._decode_tokens(spatial_logits.argmax(dim=-1)[0])
        action_text = self._decode_tokens(action_logits.argmax(dim=-1)[0])

        # Get confidences
        perception_conf = torch.sigmoid(
            self.perception_conf(perception_hidden.mean(dim=1))
        ).item()
        spatial_conf = torch.sigmoid(
            self.spatial_conf(spatial_hidden.mean(dim=1))
        ).item()
        action_conf = torch.sigmoid(
            self.action_conf(action_hidden.mean(dim=1))
        ).item()

        inference_time = (time.time() - start_time) * 1000

        return ReasoningOutput(
            perception_reasoning=perception_text,
            spatial_reasoning=spatial_text,
            action_reasoning=action_text,
            perception_confidence=perception_conf,
            spatial_confidence=spatial_conf,
            action_confidence=action_conf,
            inference_time_ms=inference_time,
        )

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        words = []
        for tid in token_ids.cpu().numpy():
            if tid == 2:  # <eos>
                break
            if tid in self.vocab:
                word = self.vocab[tid]
                if word not in ["<pad>", "<sos>", "<eos>"]:
                    words.append(word)
        return " ".join(words) if words else "Unable to generate reasoning"

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded ActionReasoningModule from {checkpoint_path}")

    def save(self, checkpoint_path: str) -> None:
        """Save model weights."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, checkpoint_path)
        logger.info(f"Saved ActionReasoningModule to {checkpoint_path}")


# ============================================================================
# Template-Based Reasoning (Fallback)
# ============================================================================

class TemplateReasoningGenerator:
    """
    Generate reasoning using templates when model is not available.

    This provides interpretability without requiring a trained reasoning model.
    """

    def __init__(self):
        self.perception_templates = [
            "I observe {n_objects} objects in the scene",
            "The target object is located in the {location} of the frame",
            "I detect: {object_list}",
        ]

        self.spatial_templates = [
            "The target is approximately {distance:.2f} meters away",
            "The object is {relation} the robot's current position",
            "Workspace is {clearance}",
        ]

        self.action_templates = [
            "I will {action} the {object}",
            "Moving gripper towards target at ({x:.0f}, {y:.0f})",
            "Executing {n_waypoints}-step trajectory",
        ]

    def generate(
        self,
        trace: TrajectoryTrace,
        detected_objects: Optional[List[Dict]] = None,
        target_distance: Optional[float] = None,
    ) -> ReasoningOutput:
        """Generate template-based reasoning."""

        # Perception reasoning
        if detected_objects:
            obj_list = ", ".join([o.get("label", "object") for o in detected_objects])
            perception = f"I observe {len(detected_objects)} objects: {obj_list}"
        else:
            perception = "Analyzing scene for manipulation targets"

        # Spatial reasoning
        if target_distance is not None:
            relation = "within reach" if target_distance < 0.8 else "at extended range"
            spatial = f"Target is approximately {target_distance:.2f}m away, {relation}"
        elif trace.waypoint_depths is not None:
            avg_depth = np.mean(trace.waypoint_depths)
            spatial = f"Target depth approximately {avg_depth:.2f} meters"
        else:
            spatial = "Estimating spatial layout from visual features"

        # Action reasoning
        if trace.num_waypoints > 0:
            start = trace.waypoints[0]
            end = trace.waypoints[-1]
            action = (
                f"Executing {trace.num_waypoints}-step trajectory "
                f"from ({start[0]:.0f}, {start[1]:.0f}) to ({end[0]:.0f}, {end[1]:.0f})"
            )
        else:
            action = "Planning approach trajectory"

        return ReasoningOutput(
            perception_reasoning=perception,
            spatial_reasoning=spatial,
            action_reasoning=action,
            perception_confidence=0.8,
            spatial_confidence=0.7,
            action_confidence=trace.mean_confidence,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_reasoning_for_trace(
    trace: TrajectoryTrace,
    use_template: bool = True,
    reasoning_module: Optional[ActionReasoningModule] = None,
) -> ReasoningOutput:
    """
    Generate reasoning for a trajectory trace.

    Args:
        trace: Trajectory trace
        use_template: Use template-based reasoning
        reasoning_module: Optional trained reasoning module

    Returns:
        ReasoningOutput
    """
    if use_template or reasoning_module is None:
        generator = TemplateReasoningGenerator()
        return generator.generate(trace)
    else:
        # Would use neural module here
        raise NotImplementedError("Neural reasoning requires trained model")


def attach_reasoning_to_trace(
    trace: TrajectoryTrace,
    reasoning: ReasoningOutput,
) -> TrajectoryTrace:
    """Attach reasoning output to a trajectory trace."""
    trace.perception_reasoning = reasoning.perception_reasoning
    trace.spatial_reasoning = reasoning.spatial_reasoning
    trace.action_reasoning = reasoning.action_reasoning
    return trace

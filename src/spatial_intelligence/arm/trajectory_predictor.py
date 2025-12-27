"""
Trajectory Predictor - Image-Space Waypoint Generation

Predicts a sequence of 2D waypoints in image coordinates that represent
the robot's intended motion path. This enables:
1. Interpretable planning (waypoints can be visualized)
2. Embodiment-agnostic planning (image coords are universal)
3. User steerability (waypoints can be modified)

Architecture:
    Perception Tokens [N, D] + Instruction [1, D]
                    │
                    ▼
            Cross-Attention
                    │
                    ▼
        Transformer Decoder (autoregressive)
                    │
                    ▼
        Waypoint Head ──▶ (x, y, confidence, is_terminal)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

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
    F = None

# Local imports
from .trajectory_trace import TrajectoryTrace, WaypointType


@dataclass
class TrajectoryPredictorConfig:
    """Configuration for TrajectoryPredictor."""
    # Input dimensions
    perception_dim: int = 1024  # DINOv3 output dimension
    depth_token_dim: int = 256  # DepthVQVAE embedding dimension
    instruction_dim: int = 512  # Language embedding dimension

    # Model architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 6
    dropout: float = 0.1
    feedforward_dim: int = 2048

    # Output configuration
    max_waypoints: int = 20
    output_dim: int = 4  # (x, y, confidence, is_terminal)
    waypoint_range: int = 256  # Waypoints in [0, 256)

    # Depth integration
    use_depth_tokens: bool = True
    depth_grid_size: Tuple[int, int] = (35, 63)

    # Training
    teacher_forcing_ratio: float = 0.5

    # Device
    device: str = "cuda"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SpatialPositionalEncoding(nn.Module):
    """2D positional encoding for spatial features."""

    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()

        pe = torch.zeros(height * width, d_model)

        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width).reshape(-1)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1).reshape(-1)

        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * (-np.log(10000.0) / (d_model // 2))
        )

        pe[:, 0:d_model // 4:2] = torch.sin(y_pos.unsqueeze(1) * div_term[:d_model // 8])
        pe[:, 1:d_model // 4:2] = torch.cos(y_pos.unsqueeze(1) * div_term[:d_model // 8])
        pe[:, d_model // 4:d_model // 2:2] = torch.sin(x_pos.unsqueeze(1) * div_term[:d_model // 8])
        pe[:, d_model // 4 + 1:d_model // 2:2] = torch.cos(x_pos.unsqueeze(1) * div_term[:d_model // 8])

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PerceptionEncoder(nn.Module):
    """Encode perception features (vision + depth tokens)."""

    def __init__(self, config: TrajectoryPredictorConfig):
        super().__init__()
        self.config = config

        # Project vision features
        self.vision_proj = nn.Linear(config.perception_dim, config.hidden_dim)

        # Project depth tokens if used
        if config.use_depth_tokens:
            self.depth_proj = nn.Linear(config.depth_token_dim, config.hidden_dim)
            self.spatial_pos_enc = SpatialPositionalEncoding(
                config.hidden_dim,
                config.depth_grid_size[0],
                config.depth_grid_size[1]
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        depth_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode perception features.

        Args:
            vision_features: [B, N_v, D_vision] or [B, D_vision]
            depth_embeddings: [B, H, W, D_depth] or [B, N_d, D_depth]

        Returns:
            Encoded features [B, N, D_hidden]
        """
        # Handle vision features
        if vision_features.ndim == 2:
            vision_features = vision_features.unsqueeze(1)

        vision_enc = self.vision_proj(vision_features)  # [B, N_v, H]

        # Handle depth tokens
        if self.config.use_depth_tokens and depth_embeddings is not None:
            if depth_embeddings.ndim == 4:
                # [B, H, W, D] -> [B, H*W, D]
                B, H, W, D = depth_embeddings.shape
                depth_flat = depth_embeddings.reshape(B, H * W, D)
            else:
                depth_flat = depth_embeddings

            depth_enc = self.depth_proj(depth_flat)
            depth_enc = self.spatial_pos_enc(depth_enc)

            # Concatenate vision and depth
            features = torch.cat([vision_enc, depth_enc], dim=1)
        else:
            features = vision_enc

        # Encode
        encoded = self.encoder(features)

        return encoded


class WaypointDecoder(nn.Module):
    """Autoregressive decoder for waypoint prediction."""

    def __init__(self, config: TrajectoryPredictorConfig):
        super().__init__()
        self.config = config

        # Waypoint embedding (for autoregressive input)
        self.waypoint_embed = nn.Linear(config.output_dim, config.hidden_dim)

        # Positional encoding for sequence
        self.pos_enc = PositionalEncoding(config.hidden_dim, config.max_waypoints)

        # Instruction projection
        self.instruction_proj = nn.Linear(config.instruction_dim, config.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Learned start token
        self.start_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

    def forward(
        self,
        memory: torch.Tensor,
        instruction: torch.Tensor,
        target_waypoints: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode waypoints.

        Args:
            memory: Encoded perception [B, N, H]
            instruction: Instruction embedding [B, D_instr]
            target_waypoints: Ground truth for teacher forcing [B, T, 4]
            teacher_forcing: Whether to use teacher forcing

        Returns:
            waypoint_outputs: Predicted waypoints [B, T, 4]
            hidden_states: Hidden states for analysis
        """
        B = memory.size(0)
        device = memory.device

        # Project instruction and add to memory
        instr_enc = self.instruction_proj(instruction).unsqueeze(1)  # [B, 1, H]
        memory = torch.cat([instr_enc, memory], dim=1)

        # Initialize with start token
        current_input = self.start_token.expand(B, -1, -1)  # [B, 1, H]

        outputs = []
        hidden_states = []

        for t in range(self.config.max_waypoints):
            # Add positional encoding
            current_input_pe = self.pos_enc(current_input)

            # Generate causal mask
            tgt_mask = self._generate_square_subsequent_mask(
                current_input.size(1)
            ).to(device)

            # Decode
            decoded = self.decoder(
                current_input_pe,
                memory,
                tgt_mask=tgt_mask,
            )
            hidden_states.append(decoded[:, -1:, :])

            # Predict waypoint
            output = self.output_head(decoded[:, -1:, :])  # [B, 1, 4]
            outputs.append(output)

            # Check termination
            is_terminal = torch.sigmoid(output[:, :, 3]).mean() > 0.5
            if is_terminal and not self.training:
                break

            # Prepare next input
            if teacher_forcing and target_waypoints is not None and t < target_waypoints.size(1) - 1:
                next_wp = target_waypoints[:, t + 1:t + 2, :]
                next_embed = self.waypoint_embed(next_wp)
            else:
                next_embed = self.waypoint_embed(output)

            current_input = torch.cat([current_input, next_embed], dim=1)

        waypoint_outputs = torch.cat(outputs, dim=1)  # [B, T, 4]
        hidden_states = torch.cat(hidden_states, dim=1)  # [B, T, H]

        return waypoint_outputs, hidden_states

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class TrajectoryPredictor(nn.Module):
    """
    Full trajectory predictor model.

    Predicts image-space waypoints from perception tokens and instructions.
    """

    def __init__(self, config: Optional[TrajectoryPredictorConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TrajectoryPredictor")

        super().__init__()
        self.config = config or TrajectoryPredictorConfig()

        # Build components
        self.perception_encoder = PerceptionEncoder(self.config)
        self.waypoint_decoder = WaypointDecoder(self.config)

        self._initialized = False

    def forward(
        self,
        vision_features: torch.Tensor,
        instruction_embedding: torch.Tensor,
        depth_embeddings: Optional[torch.Tensor] = None,
        target_waypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            vision_features: Vision embeddings [B, N, D] or [B, D]
            instruction_embedding: Instruction [B, D_instr]
            depth_embeddings: Optional depth tokens [B, H, W, D]
            target_waypoints: Optional ground truth [B, T, 4]

        Returns:
            waypoint_outputs: [B, T, 4] (x, y, conf, terminal)
            hidden_states: [B, T, H]
        """
        # Encode perception
        memory = self.perception_encoder(vision_features, depth_embeddings)

        # Decode waypoints
        teacher_forcing = (
            self.training and
            target_waypoints is not None and
            np.random.random() < self.config.teacher_forcing_ratio
        )

        waypoint_outputs, hidden_states = self.waypoint_decoder(
            memory,
            instruction_embedding,
            target_waypoints=target_waypoints,
            teacher_forcing=teacher_forcing,
        )

        return waypoint_outputs, hidden_states

    @torch.no_grad()
    def predict(
        self,
        vision_features: torch.Tensor,
        instruction_embedding: torch.Tensor,
        depth_embeddings: Optional[torch.Tensor] = None,
        source_image: Optional[np.ndarray] = None,
        instruction_text: Optional[str] = None,
    ) -> TrajectoryTrace:
        """
        Predict trajectory trace from inputs.

        Args:
            vision_features: Vision embeddings [B, N, D] or [B, D]
            instruction_embedding: Instruction [B, D_instr]
            depth_embeddings: Optional depth tokens
            source_image: Original image for visualization
            instruction_text: Text instruction for metadata

        Returns:
            TrajectoryTrace with predicted waypoints
        """
        self.eval()
        start_time = time.time()

        # Forward pass
        waypoint_outputs, _ = self.forward(
            vision_features,
            instruction_embedding,
            depth_embeddings,
        )

        # Process outputs (take first batch element)
        wp_out = waypoint_outputs[0].cpu().numpy()  # [T, 4]

        # Extract waypoints
        waypoints = np.zeros((len(wp_out), 2), dtype=np.float32)
        waypoints[:, 0] = np.clip(
            torch.sigmoid(torch.tensor(wp_out[:, 0])).numpy() * self.config.waypoint_range,
            0, self.config.waypoint_range - 1
        )
        waypoints[:, 1] = np.clip(
            torch.sigmoid(torch.tensor(wp_out[:, 1])).numpy() * self.config.waypoint_range,
            0, self.config.waypoint_range - 1
        )

        # Extract confidences
        confidences = torch.sigmoid(torch.tensor(wp_out[:, 2])).numpy()

        # Extract terminal flags
        terminal_probs = torch.sigmoid(torch.tensor(wp_out[:, 3])).numpy()

        # Truncate at first terminal waypoint
        terminal_idx = np.where(terminal_probs > 0.5)[0]
        if len(terminal_idx) > 0:
            end_idx = terminal_idx[0] + 1
            waypoints = waypoints[:end_idx]
            confidences = confidences[:end_idx]

        # Determine waypoint types
        waypoint_types = [WaypointType.PREDICTED] * len(waypoints)
        if len(waypoints) > 0:
            waypoint_types[-1] = WaypointType.TERMINAL

        inference_time = (time.time() - start_time) * 1000

        # Create trajectory trace
        trace = TrajectoryTrace(
            waypoints=waypoints,
            confidences=confidences,
            waypoint_types=waypoint_types,
            source_image=source_image,
            instruction=instruction_text,
            model_name="trajectory_predictor",
            inference_time_ms=inference_time,
        )

        return trace

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            predictions: Predicted waypoints [B, T, 4]
            targets: Ground truth waypoints [B, T, 4]
            mask: Valid waypoint mask [B, T]

        Returns:
            Dictionary of losses
        """
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)
            predictions = predictions * mask
            targets = targets * mask
            num_valid = mask.sum()
        else:
            num_valid = predictions.numel()

        # Coordinate loss (x, y)
        coord_loss = F.mse_loss(
            predictions[:, :, :2],
            targets[:, :, :2],
            reduction='sum'
        ) / num_valid

        # Confidence loss (BCE)
        conf_loss = F.binary_cross_entropy_with_logits(
            predictions[:, :, 2],
            targets[:, :, 2],
            reduction='sum'
        ) / num_valid

        # Terminal loss (BCE)
        terminal_loss = F.binary_cross_entropy_with_logits(
            predictions[:, :, 3],
            targets[:, :, 3],
            reduction='sum'
        ) / num_valid

        total_loss = coord_loss + 0.5 * conf_loss + 0.5 * terminal_loss

        return {
            "total": total_loss,
            "coordinate": coord_loss,
            "confidence": conf_loss,
            "terminal": terminal_loss,
        }

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self._initialized = True
        logger.info(f"Loaded TrajectoryPredictor from {checkpoint_path}")

    def save(self, checkpoint_path: str) -> None:
        """Save model weights."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, checkpoint_path)
        logger.info(f"Saved TrajectoryPredictor to {checkpoint_path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mock_trajectory_predictor(
    device: str = "cpu"
) -> TrajectoryPredictor:
    """Create a mock predictor for testing."""
    config = TrajectoryPredictorConfig(
        hidden_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        feedforward_dim=256,
        device=device,
    )
    model = TrajectoryPredictor(config)
    model.to(device)
    return model


def predict_trajectory_from_features(
    vision_features: np.ndarray,
    instruction_embedding: np.ndarray,
    predictor: Optional[TrajectoryPredictor] = None,
    depth_embeddings: Optional[np.ndarray] = None,
    source_image: Optional[np.ndarray] = None,
    instruction_text: Optional[str] = None,
    device: str = "cuda",
) -> TrajectoryTrace:
    """
    Convenience function to predict trajectory from numpy arrays.

    Args:
        vision_features: Vision embeddings [N, D] or [D]
        instruction_embedding: Instruction [D_instr]
        predictor: Optional pre-loaded predictor
        depth_embeddings: Optional depth tokens
        source_image: Original image
        instruction_text: Text instruction
        device: Device for inference

    Returns:
        TrajectoryTrace
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for trajectory prediction")

    if predictor is None:
        predictor = create_mock_trajectory_predictor(device)

    # Convert to tensors
    vision_tensor = torch.from_numpy(vision_features).float().to(device)
    if vision_tensor.ndim == 1:
        vision_tensor = vision_tensor.unsqueeze(0).unsqueeze(0)
    elif vision_tensor.ndim == 2:
        vision_tensor = vision_tensor.unsqueeze(0)

    instr_tensor = torch.from_numpy(instruction_embedding).float().to(device)
    if instr_tensor.ndim == 1:
        instr_tensor = instr_tensor.unsqueeze(0)

    depth_tensor = None
    if depth_embeddings is not None:
        depth_tensor = torch.from_numpy(depth_embeddings).float().to(device)
        if depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(0)

    # Predict
    trace = predictor.predict(
        vision_tensor,
        instr_tensor,
        depth_tensor,
        source_image=source_image,
        instruction_text=instruction_text,
    )

    return trace

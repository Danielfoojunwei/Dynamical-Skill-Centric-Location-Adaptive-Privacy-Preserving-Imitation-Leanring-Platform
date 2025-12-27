"""
Depth VQVAE Tokenizer - Discrete Depth Representations

Converts continuous depth maps into discrete tokens that can be processed
by language models for spatial reasoning.

Key Innovation from MolmoAct:
- Depth as discrete tokens enables explicit 3D reasoning
- Tokens integrate with transformer-based planners
- Enables distance estimation between objects

Architecture:
    Depth Map [H, W] ──▶ Encoder ──▶ Latent [H', W', D] ──▶ Quantize ──▶ Tokens [H', W']
                                                              │
                                                              ▼
                                                         Codebook [K, D]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

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


@dataclass
class DepthTokenizerConfig:
    """Configuration for Depth VQVAE."""
    # Input/output dimensions
    input_height: int = 280
    input_width: int = 504
    output_height: int = 35  # input_height / 8
    output_width: int = 63   # input_width / 8

    # VQ-VAE parameters
    codebook_size: int = 1024  # Number of discrete tokens
    embedding_dim: int = 256   # Dimension of each token

    # Encoder architecture
    encoder_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256, 256]
    )

    # Training parameters
    commitment_cost: float = 0.25  # VQ commitment loss weight
    decay: float = 0.99  # EMA decay for codebook updates

    # Quantization
    use_ema: bool = True  # Use EMA codebook updates (more stable)

    # Device
    device: str = "cuda"


@dataclass
class DepthTokens:
    """Output from depth tokenization."""
    # Discrete token indices [H', W']
    tokens: np.ndarray

    # Token embeddings [H', W', D] (for feeding to transformers)
    embeddings: Optional[np.ndarray] = None

    # Original depth map shape
    original_shape: Tuple[int, int] = (280, 504)

    # Reconstruction (if decoder was run)
    reconstructed_depth: Optional[np.ndarray] = None

    # Quantization loss (for training)
    vq_loss: float = 0.0

    # Timing
    encoding_time_ms: float = 0.0

    @property
    def token_grid_shape(self) -> Tuple[int, int]:
        """Shape of token grid."""
        return self.tokens.shape

    @property
    def num_tokens(self) -> int:
        """Total number of tokens."""
        return self.tokens.size

    @property
    def unique_tokens(self) -> int:
        """Number of unique tokens used."""
        return len(np.unique(self.tokens))

    def flatten(self) -> np.ndarray:
        """Flatten tokens for sequence models."""
        return self.tokens.flatten()

    def to_sequence(self) -> np.ndarray:
        """Convert to sequence with positional info [N, 3] (token_id, row, col)."""
        h, w = self.tokens.shape
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        return np.stack([
            self.tokens.flatten(),
            rows.flatten(),
            cols.flatten()
        ], axis=1).astype(np.int32)


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates."""

    def __init__(self, config: DepthTokenizerConfig):
        super().__init__()
        self.config = config

        # Codebook
        self.embedding = nn.Embedding(config.codebook_size, config.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / config.codebook_size,
            1.0 / config.codebook_size
        )

        # EMA tracking
        if config.use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(config.codebook_size))
            self.register_buffer('ema_w', torch.zeros(config.codebook_size, config.embedding_dim))
            self.ema_w.data.copy_(self.embedding.weight.data)

    def forward(
        self,
        z: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize latent vectors.

        Args:
            z: Latent vectors [B, D, H, W]
            training: Whether in training mode

        Returns:
            z_q: Quantized vectors [B, D, H, W]
            indices: Token indices [B, H, W]
            vq_loss: VQ loss scalar
        """
        # Reshape for distance computation
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]

        # Compute distances to codebook entries
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)  # [B*H*W]
        indices_reshaped = indices.view(B, H, W)

        # Get quantized vectors
        z_q_flat = self.embedding(indices)  # [B*H*W, D]
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        # Compute loss
        if training:
            # Commitment loss
            commitment_loss = F.mse_loss(z_q.detach(), z)

            # Codebook loss (or EMA update)
            if self.config.use_ema:
                with torch.no_grad():
                    # Compute cluster assignments
                    encodings = F.one_hot(indices, self.config.codebook_size).float()

                    # Update cluster sizes
                    self.ema_cluster_size.data.mul_(self.config.decay).add_(
                        encodings.sum(0), alpha=1 - self.config.decay
                    )

                    # Update cluster centroids
                    dw = torch.matmul(encodings.t(), z_flat)
                    self.ema_w.data.mul_(self.config.decay).add_(
                        dw, alpha=1 - self.config.decay
                    )

                    # Normalize
                    n = self.ema_cluster_size.sum()
                    cluster_size = (
                        (self.ema_cluster_size + 1e-5)
                        / (n + self.config.codebook_size * 1e-5) * n
                    )
                    self.embedding.weight.data.copy_(
                        self.ema_w / cluster_size.unsqueeze(1)
                    )

                codebook_loss = torch.tensor(0.0, device=z.device)
            else:
                codebook_loss = F.mse_loss(z_q, z.detach())

            vq_loss = codebook_loss + self.config.commitment_cost * commitment_loss

            # Straight-through estimator
            z_q = z + (z_q - z).detach()
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        return z_q, indices_reshaped, vq_loss


class DepthEncoder(nn.Module):
    """Encoder network for depth maps."""

    def __init__(self, config: DepthTokenizerConfig):
        super().__init__()

        channels = [1] + config.encoder_channels + [config.embedding_dim]
        layers = []

        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(
                channels[i], channels[i + 1],
                kernel_size=4, stride=2, padding=1
            ))
            if i < len(channels) - 2:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
                layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DepthDecoder(nn.Module):
    """Decoder network for depth reconstruction."""

    def __init__(self, config: DepthTokenizerConfig):
        super().__init__()

        channels = [config.embedding_dim] + config.encoder_channels[::-1] + [1]
        layers = []

        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(
                channels[i], channels[i + 1],
                kernel_size=4, stride=2, padding=1
            ))
            if i < len(channels) - 2:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
                layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DepthVQVAE(nn.Module):
    """
    Depth VQVAE for tokenizing depth maps.

    Converts continuous depth maps to discrete tokens that can be
    processed by transformer-based models for spatial reasoning.
    """

    def __init__(self, config: Optional[DepthTokenizerConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for DepthVQVAE")

        super().__init__()
        self.config = config or DepthTokenizerConfig()

        # Build networks
        self.encoder = DepthEncoder(self.config)
        self.quantizer = VectorQuantizer(self.config)
        self.decoder = DepthDecoder(self.config)

        # Pre/post processing
        self.depth_mean = 1.5  # Average depth in meters
        self.depth_std = 1.0

        self._initialized = False

    def encode(
        self,
        depth_map: torch.Tensor,
        return_embeddings: bool = True,
    ) -> DepthTokens:
        """
        Encode depth map to discrete tokens.

        Args:
            depth_map: Depth map [B, 1, H, W] or [B, H, W] in meters
            return_embeddings: Whether to return token embeddings

        Returns:
            DepthTokens with token indices and optionally embeddings
        """
        start_time = time.time()

        # Handle input shape
        if depth_map.ndim == 3:
            depth_map = depth_map.unsqueeze(1)
        elif depth_map.ndim == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)

        # Normalize depth
        depth_norm = (depth_map - self.depth_mean) / self.depth_std

        # Resize to expected input size if needed
        if depth_norm.shape[2:] != (self.config.input_height, self.config.input_width):
            depth_norm = F.interpolate(
                depth_norm,
                size=(self.config.input_height, self.config.input_width),
                mode='bilinear',
                align_corners=False
            )

        # Encode
        z = self.encoder(depth_norm)

        # Quantize
        z_q, indices, vq_loss = self.quantizer(z, training=self.training)

        # Get embeddings if requested
        embeddings = None
        if return_embeddings:
            # [B, D, H', W'] -> [B, H', W', D]
            embeddings = z_q.permute(0, 2, 3, 1).detach().cpu().numpy()
            if embeddings.shape[0] == 1:
                embeddings = embeddings[0]

        encoding_time = (time.time() - start_time) * 1000

        # Convert to numpy
        tokens = indices.detach().cpu().numpy()
        if tokens.shape[0] == 1:
            tokens = tokens[0]

        return DepthTokens(
            tokens=tokens,
            embeddings=embeddings,
            original_shape=depth_map.shape[2:],
            vq_loss=float(vq_loss.item()),
            encoding_time_ms=encoding_time,
        )

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to depth map.

        Args:
            tokens: Token indices [B, H', W']

        Returns:
            Reconstructed depth [B, 1, H, W]
        """
        # Get embeddings
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)

        B, H, W = tokens.shape
        z_q = self.quantizer.embedding(tokens)  # [B, H, W, D]
        z_q = z_q.permute(0, 3, 1, 2)  # [B, D, H, W]

        # Decode
        depth_norm = self.decoder(z_q)

        # Denormalize
        depth = depth_norm * self.depth_std + self.depth_mean

        return depth

    def forward(
        self,
        depth_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, DepthTokens]:
        """
        Full forward pass: encode and decode.

        Args:
            depth_map: Input depth [B, 1, H, W]

        Returns:
            reconstructed: Reconstructed depth [B, 1, H, W]
            tokens: DepthTokens with encoding info
        """
        # Encode
        tokens = self.encode(depth_map)

        # Decode
        tokens_tensor = torch.from_numpy(tokens.tokens).to(depth_map.device)
        if tokens_tensor.ndim == 2:
            tokens_tensor = tokens_tensor.unsqueeze(0)
        reconstructed = self.decode(tokens_tensor)

        # Resize to original size
        if reconstructed.shape[2:] != depth_map.shape[2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=depth_map.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        tokens.reconstructed_depth = reconstructed.detach().cpu().numpy()

        return reconstructed, tokens

    def compute_loss(
        self,
        depth_map: torch.Tensor,
        reconstructed: torch.Tensor,
        vq_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            depth_map: Original depth
            reconstructed: Reconstructed depth
            vq_loss: VQ loss from quantizer

        Returns:
            Dictionary of losses
        """
        recon_loss = F.mse_loss(reconstructed, depth_map)
        total_loss = recon_loss + vq_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "vq": vq_loss,
        }

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "depth_mean" in checkpoint:
            self.depth_mean = checkpoint["depth_mean"]
            self.depth_std = checkpoint["depth_std"]
        self._initialized = True
        logger.info(f"Loaded DepthVQVAE from {checkpoint_path}")

    def save(self, checkpoint_path: str) -> None:
        """Save model weights."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "depth_mean": self.depth_mean,
            "depth_std": self.depth_std,
        }, checkpoint_path)
        logger.info(f"Saved DepthVQVAE to {checkpoint_path}")

    @property
    def codebook_usage(self) -> float:
        """Fraction of codebook being used."""
        if hasattr(self.quantizer, 'ema_cluster_size'):
            used = (self.quantizer.ema_cluster_size > 1).sum().item()
            return used / self.config.codebook_size
        return 1.0

    def get_codebook_embeddings(self) -> np.ndarray:
        """Get codebook embeddings for analysis."""
        return self.quantizer.embedding.weight.detach().cpu().numpy()


# ============================================================================
# Convenience Functions
# ============================================================================

def tokenize_depth(
    depth_map: np.ndarray,
    tokenizer: Optional[DepthVQVAE] = None,
    device: str = "cuda",
) -> DepthTokens:
    """
    Convenience function to tokenize a depth map.

    Args:
        depth_map: Depth map [H, W] in meters
        tokenizer: Optional pre-loaded tokenizer
        device: Device for inference

    Returns:
        DepthTokens
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for depth tokenization")

    if tokenizer is None:
        tokenizer = DepthVQVAE()
        tokenizer.to(device)
        tokenizer.eval()

    depth_tensor = torch.from_numpy(depth_map).float().to(device)

    with torch.no_grad():
        tokens = tokenizer.encode(depth_tensor)

    return tokens


def create_mock_depth_tokens(
    height: int = 35,
    width: int = 63,
    num_unique: int = 100,
) -> DepthTokens:
    """Create mock depth tokens for testing."""
    tokens = np.random.randint(0, num_unique, (height, width), dtype=np.int64)
    embeddings = np.random.randn(height, width, 256).astype(np.float32)

    return DepthTokens(
        tokens=tokens,
        embeddings=embeddings,
        original_shape=(280, 504),
    )

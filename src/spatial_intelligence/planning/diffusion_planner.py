"""
Diffusion Planner - Score-based Trajectory Generation

Implements diffusion-based trajectory planning inspired by:
- Diffuser (Janner et al., 2022): Planning with Diffusion for Flexible Behavior Synthesis
- Decision Diffuser: Conditional diffusion for decision-making

This planner refines VLA action proposals into smooth, multi-modal trajectories
using iterative denoising with learned score functions.

Key Features:
- Generates smooth action trajectories from noisy proposals
- Handles multi-modal action distributions
- Supports goal-conditioned planning
- Integrates with Pi0.5 for semantic guidance

Architecture:
    Pi0.5 Action Proposal → Diffusion Denoising → Smooth Trajectory
                                    ↑
                        Score Network (learned)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class DenoisingSchedule(Enum):
    """Denoising schedule types for diffusion."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"
    SIGMOID = "sigmoid"


@dataclass
class DiffusionConfig:
    """Configuration for Diffusion Planner."""
    # Trajectory dimensions
    action_dim: int = 7
    horizon: int = 16

    # Diffusion parameters
    num_diffusion_steps: int = 100
    schedule: DenoisingSchedule = DenoisingSchedule.COSINE
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # Conditioning
    use_goal_conditioning: bool = True
    use_state_conditioning: bool = True
    condition_dim: int = 512  # DINOv3 feature dim

    # Inference
    device: str = "cuda"
    dtype: str = "float32"
    num_samples: int = 8  # Number of trajectory samples
    guidance_scale: float = 1.0  # Classifier-free guidance


@dataclass
class Trajectory:
    """A single trajectory of actions."""
    actions: Any  # [H, A] array of actions
    horizon: int
    action_dim: int

    # Quality metrics
    smoothness: Optional[float] = None
    likelihood: Optional[float] = None
    goal_distance: Optional[float] = None

    def to_numpy(self) -> Any:
        """Convert to numpy array."""
        if HAS_TORCH and torch is not None and isinstance(self.actions, torch.Tensor):
            return self.actions.cpu().numpy()
        return self.actions


@dataclass
class TrajectoryBatch:
    """Batch of trajectory samples from diffusion."""
    trajectories: List[Trajectory]
    best_idx: int = 0

    # Aggregated metrics
    mean_smoothness: Optional[float] = None
    diversity: Optional[float] = None

    @property
    def best(self) -> Trajectory:
        """Get the best trajectory."""
        return self.trajectories[self.best_idx]

    @property
    def num_samples(self) -> int:
        return len(self.trajectories)


class ScoreNetwork(nn.Module if HAS_TORCH else object):
    """
    Score network for diffusion trajectory generation.

    Predicts the score (gradient of log probability) for denoising.
    Uses a Transformer architecture for temporal modeling.
    """

    def __init__(self, config: DiffusionConfig):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ScoreNetwork")
        super().__init__()

        self.config = config

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Action embedding
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)

        # Condition embedding (for goal/state conditioning)
        if config.use_goal_conditioning or config.use_state_conditioning:
            self.condition_embed = nn.Linear(config.condition_dim, config.hidden_dim)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.horizon, config.hidden_dim) * 0.02
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(
        self,
        x: Any,  # [B, H, A] noisy actions (torch.Tensor)
        t: Any,  # [B] timesteps (torch.Tensor)
        condition: Optional[Any] = None,  # [B, C] conditioning (torch.Tensor)
    ) -> Any:
        """Predict score for denoising."""
        B, H, A = x.shape

        # Embed time
        t_embed = self.time_embed(t.float().unsqueeze(-1))  # [B, D]

        # Embed actions
        x_embed = self.action_embed(x)  # [B, H, D]

        # Add positional encoding
        x_embed = x_embed + self.pos_embed

        # Add time embedding to all positions
        x_embed = x_embed + t_embed.unsqueeze(1)

        # Add conditioning if available
        if condition is not None and hasattr(self, 'condition_embed'):
            cond_embed = self.condition_embed(condition)  # [B, D]
            x_embed = x_embed + cond_embed.unsqueeze(1)

        # Transformer
        h = self.transformer(x_embed)  # [B, H, D]

        # Project to action space (score prediction)
        score = self.output_proj(h)  # [B, H, A]

        return score


class DiffusionPlanner:
    """
    Diffusion-based trajectory planner.

    Generates smooth, multi-modal action trajectories using
    iterative denoising with a learned score network.

    Usage:
        config = DiffusionConfig(action_dim=7, horizon=16)
        planner = DiffusionPlanner(config)
        planner.load_model()

        # Generate trajectory from VLA proposal
        trajectory = planner.plan(
            initial_actions=vla_actions,
            condition=scene_features,
        )
    """

    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.model: Optional[ScoreNetwork] = None
        self._loaded = False

        # Precompute noise schedule
        self._setup_noise_schedule()

        # Stats
        self.stats = {
            "backend": "mock" if not HAS_TORCH else "torch",
            "total_plans": 0,
            "avg_inference_ms": 0.0,
        }

    def _setup_noise_schedule(self):
        """Setup beta schedule for diffusion."""
        T = self.config.num_diffusion_steps

        if self.config.schedule == DenoisingSchedule.LINEAR:
            betas = self._linear_schedule(T)
        elif self.config.schedule == DenoisingSchedule.COSINE:
            betas = self._cosine_schedule(T)
        elif self.config.schedule == DenoisingSchedule.QUADRATIC:
            betas = self._quadratic_schedule(T)
        else:
            betas = self._linear_schedule(T)

        if HAS_NUMPY:
            self.betas = betas
            self.alphas = 1.0 - betas
            self.alphas_cumprod = np.cumprod(self.alphas)
            self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def _linear_schedule(self, T: int):
        """Linear beta schedule."""
        if HAS_NUMPY:
            return np.linspace(self.config.beta_start, self.config.beta_end, T)
        return [self.config.beta_start + i * (self.config.beta_end - self.config.beta_start) / T for i in range(T)]

    def _cosine_schedule(self, T: int, s: float = 0.008):
        """Cosine beta schedule (improved)."""
        if HAS_NUMPY:
            steps = np.arange(T + 1)
            alphas_cumprod = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return np.clip(betas, 0.0001, 0.999)
        return self._linear_schedule(T)  # Fallback

    def _quadratic_schedule(self, T: int):
        """Quadratic beta schedule."""
        if HAS_NUMPY:
            return np.linspace(
                self.config.beta_start ** 0.5,
                self.config.beta_end ** 0.5,
                T
            ) ** 2
        return self._linear_schedule(T)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load the score network."""
        if self._loaded:
            return

        if HAS_TORCH:
            self.model = ScoreNetwork(self.config)

            if checkpoint_path:
                state_dict = torch.load(checkpoint_path, map_location=self.config.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded diffusion model from {checkpoint_path}")
            else:
                logger.info("Initialized diffusion model with random weights (no checkpoint)")

            self.model.to(self.config.device)
            self.model.eval()
            self.stats["backend"] = "torch"
        else:
            logger.warning("PyTorch not available, using mock mode")
            self.stats["backend"] = "mock"

        self._loaded = True

    def plan(
        self,
        initial_actions: Optional[Any] = None,
        condition: Optional[Any] = None,
        goal: Optional[Any] = None,
        num_samples: Optional[int] = None,
    ) -> TrajectoryBatch:
        """
        Generate trajectory samples using diffusion.

        Args:
            initial_actions: Optional VLA action proposal [H, A] for guidance
            condition: Scene features from DINOv3/perception [C]
            goal: Goal specification (position, features, etc.)
            num_samples: Number of trajectories to sample

        Returns:
            TrajectoryBatch with multiple trajectory samples
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()
        num_samples = num_samples or self.config.num_samples

        if HAS_TORCH and self.model is not None:
            trajectories = self._diffusion_sample(
                initial_actions=initial_actions,
                condition=condition,
                num_samples=num_samples,
            )
        else:
            trajectories = self._mock_sample(
                initial_actions=initial_actions,
                num_samples=num_samples,
            )

        # Score trajectories
        best_idx = self._select_best_trajectory(trajectories, goal)

        # Calculate metrics
        smoothness_scores = [self._compute_smoothness(t) for t in trajectories]
        mean_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0

        for i, traj in enumerate(trajectories):
            traj.smoothness = smoothness_scores[i]

        batch = TrajectoryBatch(
            trajectories=trajectories,
            best_idx=best_idx,
            mean_smoothness=mean_smoothness,
            diversity=self._compute_diversity(trajectories),
        )

        # Update stats
        inference_time = (time.time() - start_time) * 1000
        self.stats["total_plans"] += 1
        self.stats["avg_inference_ms"] = (
            self.stats["avg_inference_ms"] * (self.stats["total_plans"] - 1) + inference_time
        ) / self.stats["total_plans"]

        return batch

    def _diffusion_sample(
        self,
        initial_actions: Optional[Any],
        condition: Optional[Any],
        num_samples: int,
    ) -> List[Trajectory]:
        """Sample trajectories using diffusion denoising."""
        device = self.config.device
        H = self.config.horizon
        A = self.config.action_dim

        # Start from noise
        x = torch.randn(num_samples, H, A, device=device)

        # Prepare condition
        cond = None
        if condition is not None:
            if isinstance(condition, np.ndarray):
                condition = torch.from_numpy(condition).float()
            cond = condition.to(device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(num_samples, -1)

        # Reverse diffusion
        with torch.no_grad():
            for t in reversed(range(self.config.num_diffusion_steps)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

                # Predict score
                score = self.model(x, t_batch, cond)

                # Apply guidance from initial actions if provided
                if initial_actions is not None and t < self.config.num_diffusion_steps // 2:
                    if isinstance(initial_actions, np.ndarray):
                        initial_actions = torch.from_numpy(initial_actions).float()
                    guidance = initial_actions.to(device)
                    if guidance.dim() == 2:
                        guidance = guidance.unsqueeze(0)
                    score = score + self.config.guidance_scale * (guidance - x)

                # Denoising step
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]

                # Predict x0
                sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
                x0_pred = (x - sqrt_one_minus_alpha_cumprod * score) / self.sqrt_alphas_cumprod[t]

                # Compute mean
                if t > 0:
                    alpha_cumprod_prev = self.alphas_cumprod[t - 1]
                    posterior_mean = (
                        np.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0_pred +
                        np.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * x
                    )
                    posterior_var = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                    noise = torch.randn_like(x)
                    x = posterior_mean + np.sqrt(posterior_var) * noise
                else:
                    x = x0_pred

        # Convert to trajectories
        trajectories = []
        for i in range(num_samples):
            traj = Trajectory(
                actions=x[i].cpu().numpy(),
                horizon=H,
                action_dim=A,
            )
            trajectories.append(traj)

        return trajectories

    def _mock_sample(
        self,
        initial_actions: Optional[Any],
        num_samples: int,
    ) -> List[Trajectory]:
        """Mock sampling without PyTorch."""
        if not HAS_NUMPY:
            raise ImportError("NumPy required for mock sampling")

        H = self.config.horizon
        A = self.config.action_dim

        trajectories = []
        for i in range(num_samples):
            if initial_actions is not None:
                # Add noise to initial actions
                actions = initial_actions + np.random.randn(H, A) * 0.1
            else:
                # Random smooth trajectory
                actions = np.cumsum(np.random.randn(H, A) * 0.05, axis=0)

            traj = Trajectory(
                actions=actions.astype(np.float32),
                horizon=H,
                action_dim=A,
            )
            trajectories.append(traj)

        return trajectories

    def _compute_smoothness(self, trajectory: Trajectory) -> float:
        """Compute trajectory smoothness (lower is smoother)."""
        actions = trajectory.to_numpy()
        if not HAS_NUMPY or actions is None:
            return 0.0

        # Compute second derivative (acceleration)
        velocity = np.diff(actions, axis=0)
        acceleration = np.diff(velocity, axis=0)

        # L2 norm of acceleration
        smoothness = np.mean(np.linalg.norm(acceleration, axis=1))
        return float(smoothness)

    def _compute_diversity(self, trajectories: List[Trajectory]) -> float:
        """Compute diversity among trajectory samples."""
        if not HAS_NUMPY or len(trajectories) < 2:
            return 0.0

        # Pairwise distances
        arrays = [t.to_numpy() for t in trajectories]
        distances = []
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                dist = np.mean(np.linalg.norm(arrays[i] - arrays[j], axis=1))
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def _select_best_trajectory(
        self,
        trajectories: List[Trajectory],
        goal: Optional[Any] = None,
    ) -> int:
        """Select the best trajectory based on quality metrics."""
        if not trajectories:
            return 0

        scores = []
        for i, traj in enumerate(trajectories):
            # Smoothness score (lower is better, so negate)
            smooth = self._compute_smoothness(traj)
            score = -smooth

            # Goal distance if provided
            if goal is not None and HAS_NUMPY:
                final_action = traj.to_numpy()[-1]
                if isinstance(goal, np.ndarray) and goal.shape == final_action.shape:
                    goal_dist = np.linalg.norm(final_action - goal)
                    traj.goal_distance = float(goal_dist)
                    score -= goal_dist * 0.5

            scores.append(score)

        return int(np.argmax(scores)) if HAS_NUMPY else 0

    def refine_trajectory(
        self,
        trajectory: Trajectory,
        num_steps: int = 10,
    ) -> Trajectory:
        """Refine a trajectory with additional denoising steps."""
        if not HAS_TORCH or self.model is None:
            return trajectory

        device = self.config.device
        actions = torch.from_numpy(trajectory.to_numpy()).float().unsqueeze(0).to(device)

        # Add small noise and denoise
        noise_scale = 0.1
        noisy = actions + torch.randn_like(actions) * noise_scale

        with torch.no_grad():
            for t in range(num_steps - 1, -1, -1):
                t_batch = torch.tensor([t], device=device)
                score = self.model(noisy, t_batch, None)
                noisy = noisy - score * 0.01  # Small step

        return Trajectory(
            actions=noisy.squeeze(0).cpu().numpy(),
            horizon=trajectory.horizon,
            action_dim=trajectory.action_dim,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

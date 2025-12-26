"""
RIP Gating - Robust Imitative Planning with Epistemic Uncertainty

Implements safety gating based on:
- RIP (Filos et al., 2020): Epistemic uncertainty for safe imitation
- Ensemble disagreement for OOD detection
- Risk-aware action selection

This module detects when the robot is in an unfamiliar state and
gates actions to prevent unsafe behavior.

Key Features:
- Ensemble-based epistemic uncertainty estimation
- Out-of-distribution (OOD) detection
- Risk level classification
- Action gating with fallback strategies

Architecture:
    Observations → Ensemble Models → Disagreement → Risk Level
                                          ↓
                           Safe to Execute or Trigger Recovery
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

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


class RiskLevel(Enum):
    """Risk level classification."""
    SAFE = "safe"           # Low uncertainty, execute normally
    CAUTION = "caution"     # Moderate uncertainty, proceed carefully
    WARNING = "warning"     # High uncertainty, consider alternatives
    CRITICAL = "critical"   # Very high uncertainty, trigger recovery


@dataclass
class UncertaintyEstimate:
    """Epistemic uncertainty estimate from ensemble."""
    # Core metrics
    epistemic: float        # Ensemble disagreement (epistemic uncertainty)
    aleatoric: float        # Within-model variance (aleatoric uncertainty)
    total: float            # Combined uncertainty

    # Ensemble details
    ensemble_size: int
    mean_prediction: Any    # Mean action across ensemble
    std_prediction: Any     # Std of predictions

    # Per-action uncertainties
    action_uncertainties: Optional[Any] = None  # [A] per-dimension uncertainty


@dataclass
class SafetyDecision:
    """Safety decision from RIP gating."""
    # Decision
    risk_level: RiskLevel
    is_safe: bool
    confidence: float  # Confidence in the decision (0-1)

    # Uncertainty details
    uncertainty: UncertaintyEstimate

    # Recommendations
    should_execute: bool
    should_slow_down: bool
    should_trigger_recovery: bool

    # Metrics
    ood_score: float  # Out-of-distribution score (higher = more OOD)
    distribution_distance: float  # Distance from training distribution

    # Optional: suggested safe action
    safe_action: Optional[Any] = None


@dataclass
class RIPConfig:
    """Configuration for RIP Gating."""
    # Ensemble configuration
    ensemble_size: int = 5
    hidden_dim: int = 256
    num_layers: int = 3

    # Input/output dimensions
    observation_dim: int = 512  # DINOv3 features
    action_dim: int = 7
    proprio_dim: int = 7

    # Thresholds
    safe_threshold: float = 0.1       # Below: SAFE
    caution_threshold: float = 0.3    # Below: CAUTION
    warning_threshold: float = 0.6    # Below: WARNING, Above: CRITICAL

    # OOD detection
    ood_threshold: float = 0.5
    use_mahalanobis: bool = True  # Use Mahalanobis distance for OOD

    # Device
    device: str = "cuda"


class EnsembleMember(nn.Module if HAS_TORCH else object):
    """Single ensemble member for uncertainty estimation."""

    def __init__(self, config: RIPConfig):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for EnsembleMember")
        super().__init__()

        self.config = config
        input_dim = config.observation_dim + config.proprio_dim

        # MLP encoder
        layers = []
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        self.encoder = nn.Sequential(*layers)

        # Action prediction head
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Uncertainty head (predicts aleatoric uncertainty)
        self.uncertainty_head = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(
        self,
        observation: Any,  # torch.Tensor
        proprio: Optional[Any] = None,  # torch.Tensor
    ) -> Tuple[Any, Any]:
        """Forward pass returning action and uncertainty."""
        if proprio is not None:
            x = torch.cat([observation, proprio], dim=-1)
        else:
            # Pad with zeros if no proprio
            batch_size = observation.shape[0]
            proprio_zeros = torch.zeros(
                batch_size, self.config.proprio_dim,
                device=observation.device
            )
            x = torch.cat([observation, proprio_zeros], dim=-1)

        # Encode
        h = self.encoder(x)

        # Predict action and uncertainty
        action = self.action_head(h)
        log_var = self.uncertainty_head(h)

        return action, log_var


class RIPGating:
    """
    Robust Imitative Planning (RIP) Gating.

    Uses ensemble disagreement to detect out-of-distribution states
    and gate unsafe actions.

    Usage:
        config = RIPConfig(ensemble_size=5)
        rip = RIPGating(config)
        rip.load_model()

        # Check safety of an action
        decision = rip.evaluate(
            observation=scene_features,
            proprio=robot_state,
            proposed_action=vla_action,
        )

        if decision.should_execute:
            robot.execute(proposed_action)
        elif decision.should_trigger_recovery:
            recovery.activate()
    """

    def __init__(self, config: Optional[RIPConfig] = None):
        self.config = config or RIPConfig()
        self.ensemble: List[EnsembleMember] = []
        self._loaded = False

        # Training distribution statistics (for OOD detection)
        self.train_mean: Optional[Any] = None
        self.train_cov_inv: Optional[Any] = None  # Inverse covariance for Mahalanobis

        # Stats
        self.stats = {
            "backend": "mock" if not HAS_TORCH else "torch",
            "total_evaluations": 0,
            "ood_detections": 0,
            "avg_uncertainty": 0.0,
        }

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load the ensemble models."""
        if self._loaded:
            return

        if HAS_TORCH:
            self.ensemble = []
            for i in range(self.config.ensemble_size):
                member = EnsembleMember(self.config)
                self.ensemble.append(member)

            if checkpoint_path:
                state_dict = torch.load(checkpoint_path, map_location=self.config.device)
                for i, member in enumerate(self.ensemble):
                    member.load_state_dict(state_dict[f"ensemble_{i}"])
                if "train_mean" in state_dict:
                    self.train_mean = state_dict["train_mean"]
                if "train_cov_inv" in state_dict:
                    self.train_cov_inv = state_dict["train_cov_inv"]
                logger.info(f"Loaded RIP ensemble from {checkpoint_path}")
            else:
                logger.info("Initialized RIP ensemble with random weights")

            for member in self.ensemble:
                member.to(self.config.device)
                member.eval()

            self.stats["backend"] = "torch"
        else:
            logger.warning("PyTorch not available, using mock mode")
            self.stats["backend"] = "mock"

        self._loaded = True

    def evaluate(
        self,
        observation: Any,
        proprio: Optional[Any] = None,
        proposed_action: Optional[Any] = None,
    ) -> SafetyDecision:
        """
        Evaluate safety of current state and proposed action.

        Args:
            observation: Scene features from DINOv3 [C]
            proprio: Proprioceptive state [P]
            proposed_action: Proposed action from VLA [A]

        Returns:
            SafetyDecision with risk level and recommendations
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()

        if HAS_TORCH and self.ensemble:
            uncertainty = self._ensemble_uncertainty(observation, proprio)
            ood_score = self._compute_ood_score(observation, proprio)
        else:
            uncertainty = self._mock_uncertainty()
            ood_score = 0.1

        # Classify risk level
        risk_level = self._classify_risk(uncertainty.epistemic, ood_score)

        # Make safety decision
        decision = SafetyDecision(
            risk_level=risk_level,
            is_safe=risk_level in [RiskLevel.SAFE, RiskLevel.CAUTION],
            confidence=1.0 - uncertainty.epistemic,
            uncertainty=uncertainty,
            should_execute=risk_level == RiskLevel.SAFE,
            should_slow_down=risk_level == RiskLevel.CAUTION,
            should_trigger_recovery=risk_level == RiskLevel.CRITICAL,
            ood_score=ood_score,
            distribution_distance=ood_score,
        )

        # If warning level, compute safe action
        if risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]:
            decision.safe_action = uncertainty.mean_prediction

        # Update stats
        self.stats["total_evaluations"] += 1
        if ood_score > self.config.ood_threshold:
            self.stats["ood_detections"] += 1
        self.stats["avg_uncertainty"] = (
            self.stats["avg_uncertainty"] * (self.stats["total_evaluations"] - 1) +
            uncertainty.epistemic
        ) / self.stats["total_evaluations"]

        return decision

    def _ensemble_uncertainty(
        self,
        observation: Any,
        proprio: Optional[Any],
    ) -> UncertaintyEstimate:
        """Compute uncertainty from ensemble disagreement."""
        device = self.config.device

        # Convert to tensors
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        obs = observation.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        prop = None
        if proprio is not None:
            if isinstance(proprio, np.ndarray):
                proprio = torch.from_numpy(proprio).float()
            prop = proprio.to(device)
            if prop.dim() == 1:
                prop = prop.unsqueeze(0)

        # Get predictions from all ensemble members
        predictions = []
        aleatoric_vars = []

        with torch.no_grad():
            for member in self.ensemble:
                action, log_var = member(obs, prop)
                predictions.append(action)
                aleatoric_vars.append(torch.exp(log_var))

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [E, B, A]
        aleatoric_vars = torch.stack(aleatoric_vars, dim=0)  # [E, B, A]

        # Compute statistics
        mean_pred = predictions.mean(dim=0)  # [B, A]
        std_pred = predictions.std(dim=0)    # [B, A]

        # Epistemic uncertainty: variance across ensemble
        epistemic = std_pred.mean().item()

        # Aleatoric uncertainty: mean of predicted variances
        aleatoric = aleatoric_vars.mean().item()

        # Total uncertainty
        total = np.sqrt(epistemic**2 + aleatoric**2) if HAS_NUMPY else epistemic + aleatoric

        return UncertaintyEstimate(
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            ensemble_size=len(self.ensemble),
            mean_prediction=mean_pred.squeeze(0).cpu().numpy(),
            std_prediction=std_pred.squeeze(0).cpu().numpy(),
            action_uncertainties=std_pred.squeeze(0).cpu().numpy(),
        )

    def _mock_uncertainty(self) -> UncertaintyEstimate:
        """Mock uncertainty for testing without PyTorch."""
        if HAS_NUMPY:
            mean_pred = np.zeros(self.config.action_dim)
            std_pred = np.ones(self.config.action_dim) * 0.1
        else:
            mean_pred = [0.0] * self.config.action_dim
            std_pred = [0.1] * self.config.action_dim

        return UncertaintyEstimate(
            epistemic=0.1,
            aleatoric=0.05,
            total=0.11,
            ensemble_size=self.config.ensemble_size,
            mean_prediction=mean_pred,
            std_prediction=std_pred,
        )

    def _compute_ood_score(
        self,
        observation: Any,
        proprio: Optional[Any],
    ) -> float:
        """Compute out-of-distribution score."""
        if not HAS_TORCH:
            return 0.1

        # Convert to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        obs = observation.to(self.config.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # If we have training distribution stats, use Mahalanobis distance
        if self.train_mean is not None and self.train_cov_inv is not None:
            diff = obs - self.train_mean
            mahal = torch.sqrt(torch.sum(diff @ self.train_cov_inv * diff, dim=-1))
            return mahal.mean().item()

        # Otherwise use simple heuristic based on feature magnitude
        feature_norm = torch.norm(obs, dim=-1).mean().item()

        # Assume training features have norm around 1-10
        expected_norm = 5.0
        ood_score = abs(feature_norm - expected_norm) / expected_norm

        return min(ood_score, 1.0)

    def _classify_risk(self, epistemic: float, ood_score: float) -> RiskLevel:
        """Classify risk level based on uncertainty and OOD score."""
        # Combine epistemic uncertainty and OOD score
        combined = 0.7 * epistemic + 0.3 * ood_score

        if combined < self.config.safe_threshold:
            return RiskLevel.SAFE
        elif combined < self.config.caution_threshold:
            return RiskLevel.CAUTION
        elif combined < self.config.warning_threshold:
            return RiskLevel.WARNING
        else:
            return RiskLevel.CRITICAL

    def set_training_distribution(
        self,
        observations: Any,
    ):
        """
        Set training distribution statistics for OOD detection.

        Args:
            observations: Training observations [N, C]
        """
        if not HAS_TORCH:
            return

        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()

        observations = observations.to(self.config.device)

        self.train_mean = observations.mean(dim=0, keepdim=True)

        # Compute covariance
        centered = observations - self.train_mean
        cov = (centered.T @ centered) / (observations.shape[0] - 1)

        # Add regularization for numerical stability
        cov = cov + torch.eye(cov.shape[0], device=self.config.device) * 1e-6

        self.train_cov_inv = torch.inverse(cov)

        logger.info(f"Set training distribution from {observations.shape[0]} samples")

    def register_callback(
        self,
        risk_level: RiskLevel,
        callback: Callable[[SafetyDecision], None],
    ):
        """Register callback for specific risk level."""
        if not hasattr(self, '_callbacks'):
            self._callbacks = {}
        self._callbacks[risk_level] = callback

    @property
    def is_loaded(self) -> bool:
        return self._loaded

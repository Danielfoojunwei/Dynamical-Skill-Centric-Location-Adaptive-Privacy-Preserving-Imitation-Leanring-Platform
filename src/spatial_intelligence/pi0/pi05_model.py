"""
Pi0.5 VLA Model - Official Physical Intelligence Implementation

This module provides the official Pi0.5 integration from Physical Intelligence's
openpi library. Pi0.5 is a Vision-Language-Action model pre-trained on 10k+ hours
of robot data with open-world generalization.

IMPORTANT: This uses Pi0.5 AS-IS from Physical Intelligence.
We do NOT swap out the backbone - it uses their pre-trained weights.

Pi0.5 Capabilities:
==================
- Open-world generalization to unseen environments
- Pre-trained on 10k+ hours of diverse robot data
- Semantic subtask prediction
- Multi-robot skill transfer

Installation:
============
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && pip install -e .

Available Checkpoints:
=====================
- pi0_base: Original Pi0 base model
- pi0_fast_base: Fast inference variant
- pi05_base: Pi0.5 with open-world generalization
- pi05_libero: Fine-tuned for LIBERO benchmark (SOTA)
- pi05_droid: Fine-tuned on DROID dataset

References:
- Paper: https://arxiv.org/abs/2504.16054
- Code: https://github.com/Physical-Intelligence/openpi
- Blog: https://www.physicalintelligence.company/blog/pi05
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for openpi installation
try:
    from openpi.training import config as openpi_config
    from openpi.policies import policy_config as openpi_policy_config
    from openpi.shared import download as openpi_download
    HAS_OPENPI = True
    logger.info("OpenPI library found - Pi0.5 available")
except ImportError:
    HAS_OPENPI = False
    logger.warning(
        "OpenPI not installed. Pi0.5 unavailable. Install with:\n"
        "  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
        "  cd openpi && pip install -e ."
    )

# Check for torch
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Pi05Variant(Enum):
    """Available Pi0.5 model variants from Physical Intelligence."""
    # Base models (pre-trained on 10k+ hours)
    PI0_BASE = "pi0_base"
    PI0_FAST_BASE = "pi0_fast_base"
    PI05_BASE = "pi05_base"

    # Fine-tuned variants
    PI05_LIBERO = "pi05_libero"   # SOTA on LIBERO benchmark
    PI05_DROID = "pi05_droid"     # DROID dataset fine-tuned
    PI0_ALOHA = "pi0_aloha"       # ALOHA robot fine-tuned


# Official checkpoint URLs from Physical Intelligence
CHECKPOINT_URLS = {
    Pi05Variant.PI0_BASE: "gs://openpi-assets/checkpoints/pi0_base",
    Pi05Variant.PI0_FAST_BASE: "gs://openpi-assets/checkpoints/pi0_fast_base",
    Pi05Variant.PI05_BASE: "gs://openpi-assets/checkpoints/pi05_base",
    Pi05Variant.PI05_LIBERO: "gs://openpi-assets/checkpoints/pi05_libero",
    Pi05Variant.PI05_DROID: "gs://openpi-assets/checkpoints/pi05_droid",
}


@dataclass
class Pi05Config:
    """Configuration for Pi0.5 model."""
    # Model variant
    variant: Pi05Variant = Pi05Variant.PI05_BASE

    # Checkpoint configuration
    checkpoint_dir: Optional[str] = None  # Local path or None to download
    cache_dir: str = "~/.cache/openpi"

    # Inference configuration
    device: str = "cuda"

    # Action configuration (from Pi0.5 defaults)
    action_horizon: int = 16
    action_dim: int = 7  # 6-DOF + gripper

    # Optimization
    use_torch: bool = True  # Use PyTorch backend


@dataclass
class Pi05Observation:
    """Observation input for Pi0.5 inference."""
    # Image observations - required
    # Shape: [num_cameras, H, W, C] or [H, W, C] for single camera
    images: Any

    # Language instruction - required
    instruction: str

    # Proprioceptive state - optional
    # Shape: [state_dim]
    proprio: Optional[Any] = None

    # Gripper state - optional
    gripper: Optional[float] = None


@dataclass
class Pi05Result:
    """Result from Pi0.5 inference."""
    # Predicted actions [action_horizon, action_dim]
    actions: Any

    # Action metadata
    action_horizon: int = 16
    action_dim: int = 7

    # Optional semantic prediction (Pi0.5 feature)
    subtask_prediction: Optional[str] = None

    # Inference metadata
    inference_time_ms: Optional[float] = None


class Pi05Model:
    """
    Pi0.5 VLA Model from Physical Intelligence.

    This is a wrapper around the official openpi implementation.
    It uses Pi0.5 exactly as released - no backbone modifications.
    """

    def __init__(self, config: Optional[Pi05Config] = None):
        """
        Initialize Pi0.5 model.

        Args:
            config: Model configuration. If None, uses defaults.
        """
        if not HAS_OPENPI:
            raise ImportError(
                "OpenPI library required for Pi0.5. Install with:\n"
                "  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
                "  cd openpi && pip install -e ."
            )

        self.config = config or Pi05Config()
        self.policy = None
        self._loaded = False

    def load(self) -> None:
        """Load the Pi0.5 model."""
        if self._loaded:
            logger.info("Pi0.5 model already loaded")
            return

        variant = self.config.variant
        logger.info(f"Loading Pi0.5 variant: {variant.value}")

        # Get openpi config for this variant
        openpi_cfg = openpi_config.get_config(variant.value)

        # Get checkpoint directory
        if self.config.checkpoint_dir:
            checkpoint_dir = Path(self.config.checkpoint_dir).expanduser()
            logger.info(f"Using local checkpoint: {checkpoint_dir}")
        else:
            # Download from GCS
            checkpoint_url = CHECKPOINT_URLS.get(variant)
            if not checkpoint_url:
                raise ValueError(f"No checkpoint URL for variant: {variant}")

            logger.info(f"Downloading checkpoint from: {checkpoint_url}")
            cache_dir = Path(self.config.cache_dir).expanduser()
            checkpoint_dir = openpi_download.maybe_download(
                checkpoint_url,
                cache_dir=cache_dir
            )

        # Create policy
        logger.info("Creating Pi0.5 policy...")
        self.policy = openpi_policy_config.create_trained_policy(
            openpi_cfg,
            str(checkpoint_dir)
        )

        self._loaded = True
        logger.info(f"Pi0.5 model loaded successfully: {variant.value}")

    def infer(self, observation: Pi05Observation) -> Pi05Result:
        """
        Run Pi0.5 inference to generate actions.

        Args:
            observation: Input observation with images and instruction

        Returns:
            Pi05Result with predicted actions
        """
        if not self._loaded:
            self.load()

        import time
        start_time = time.perf_counter()

        # Convert observation to openpi format
        obs_dict = {
            "image": observation.images,
            "instruction": observation.instruction,
        }

        if observation.proprio is not None:
            obs_dict["proprio"] = observation.proprio

        if observation.gripper is not None:
            obs_dict["gripper"] = observation.gripper

        # Run inference
        result = self.policy.infer(obs_dict)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Extract actions
        actions = result.get("actions", result.get("action"))

        return Pi05Result(
            actions=actions,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            subtask_prediction=result.get("subtask"),
            inference_time_ms=inference_time,
        )

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.policy is not None:
            del self.policy
            self.policy = None

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._loaded = False
        logger.info("Pi0.5 model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def create(cls, variant: str = "pi05_base") -> "Pi05Model":
        """
        Create Pi0.5 model with specified variant.

        Args:
            variant: Model variant name

        Returns:
            Pi05Model instance
        """
        try:
            variant_enum = Pi05Variant(variant)
        except ValueError:
            logger.warning(f"Unknown variant '{variant}', using pi05_base")
            variant_enum = Pi05Variant.PI05_BASE

        config = Pi05Config(variant=variant_enum)
        return cls(config)

    @classmethod
    def for_jetson_thor(cls) -> "Pi05Model":
        """Create Pi0.5 model optimized for Jetson Thor."""
        config = Pi05Config(
            variant=Pi05Variant.PI05_BASE,
            device="cuda",
        )
        return cls(config)


# Convenience function
def load_pi05(variant: str = "pi05_base") -> Pi05Model:
    """
    Load Pi0.5 model.

    Args:
        variant: Model variant (pi0_base, pi05_base, pi05_libero, etc.)

    Returns:
        Loaded Pi05Model
    """
    model = Pi05Model.create(variant)
    model.load()
    return model


def list_variants() -> List[str]:
    """List available Pi0.5 variants."""
    return [v.value for v in Pi05Variant]


def check_installation() -> Dict[str, bool]:
    """Check Pi0.5 installation status."""
    return {
        "openpi_installed": HAS_OPENPI,
        "torch_installed": HAS_TORCH,
        "cuda_available": HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False,
    }

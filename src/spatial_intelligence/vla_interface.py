"""
VLA (Vision-Language-Action) Interface

This module provides a unified interface for Pi0.5 VLA model inference
from Physical Intelligence.

Pi0.5 Features:
- Pre-trained on 10k+ hours of robot data
- Open-world generalization to unseen environments
- Semantic task understanding
- Multi-camera support
- Proprioceptive conditioning

Jetson Thor Optimization:
========================
- 128GB memory supports full Pi0.5 model
- 2070 TFLOPS enables 10Hz control with perception
- Native FP8 support for faster inference
- MIG partitioning for workload isolation

Usage:
    from src.spatial_intelligence.vla_interface import VLAInterface, VLAConfig

    # Auto-detect hardware and create optimal config
    interface = VLAInterface.create_for_hardware()

    # Or specify explicitly
    config = VLAConfig.for_jetson_thor()
    interface = VLAInterface(config)

    # Run inference
    actions = interface.infer(
        images=images,
        instruction="pick up the red cup",
        proprio=proprio_state,
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from .pi0 import (
        Pi05Model, Pi05Config, Pi05Variant, Pi05Observation, Pi05Result,
        HAS_OPENPI,
    )
except ImportError:
    HAS_OPENPI = False
    Pi05Model = None
    Pi05Config = None
    Pi05Variant = None
    Pi05Observation = None
    Pi05Result = None


class HardwareTarget(Enum):
    """Hardware targets for optimization."""
    JETSON_THOR = "jetson_thor"     # 128GB, 2070 TFLOPS
    JETSON_ORIN = "jetson_orin"     # 64GB, 275 TFLOPS
    GENERIC_GPU = "generic_gpu"     # Generic CUDA GPU
    CPU = "cpu"                     # CPU fallback


@dataclass
class VLAConfig:
    """VLA configuration for Pi0.5."""
    # Hardware target
    hardware: HardwareTarget = HardwareTarget.JETSON_THOR

    # Pi0.5 configuration
    variant: str = "pi05_base"
    checkpoint_dir: Optional[str] = None

    # Inference configuration
    device: str = "cuda"
    dtype: str = "float16"
    use_tensorrt: bool = True
    use_fp8: bool = True
    use_flash_attention: bool = True

    # Action configuration
    action_horizon: int = 16
    action_dim: int = 7

    # Performance tuning
    batch_size: int = 1

    @classmethod
    def for_jetson_thor(cls) -> "VLAConfig":
        """Optimal configuration for Jetson Thor."""
        return cls(
            hardware=HardwareTarget.JETSON_THOR,
            variant="pi05_base",
            device="cuda",
            dtype="float16",
            use_tensorrt=True,
            use_fp8=True,  # Thor has native FP8
            use_flash_attention=True,
        )

    @classmethod
    def for_jetson_orin(cls) -> "VLAConfig":
        """Configuration for Jetson AGX Orin."""
        return cls(
            hardware=HardwareTarget.JETSON_ORIN,
            variant="pi05_base",
            device="cuda",
            dtype="float16",
            use_tensorrt=True,
            use_fp8=False,  # Orin doesn't have native FP8
            use_flash_attention=True,
        )

    @classmethod
    def for_development(cls) -> "VLAConfig":
        """Lightweight configuration for development."""
        return cls(
            hardware=HardwareTarget.GENERIC_GPU,
            variant="pi05_base",
            device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
            dtype="float32",
            use_tensorrt=False,
            use_fp8=False,
            use_flash_attention=False,
        )


@dataclass
class VLAObservation:
    """Standardized observation for VLA inference."""
    # Image observations [N, C, H, W] or [B, N, C, H, W]
    images: Any

    # Language instruction
    instruction: str

    # Proprioceptive state [DOF] or [B, DOF]
    proprio: Optional[Any] = None

    # Optional: gripper state
    gripper: Optional[float] = None

    # Optional: depth images
    depth: Optional[Any] = None


@dataclass
class VLAResult:
    """Result from VLA inference."""
    # Predicted actions [H, A] or [B, H, A]
    actions: Any

    # Action horizon
    action_horizon: int

    # Action dimension
    action_dim: int

    # Confidence/quality score (optional)
    confidence: Optional[float] = None

    # Semantic subtask prediction (Pi0.5 feature)
    subtask: Optional[str] = None

    # Inference time in milliseconds
    inference_time_ms: Optional[float] = None


class VLAInterface:
    """
    Unified interface for Pi0.5 VLA model inference.

    Uses the official Physical Intelligence Pi0.5 implementation.
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        """
        Initialize VLA interface.

        Args:
            config: VLA configuration. If None, auto-detects hardware.
        """
        if not HAS_OPENPI:
            raise ImportError(
                "Pi0.5 not available. Install with:\n"
                "  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
                "  cd openpi && pip install -e ."
            )

        self.config = config or self._auto_detect_config()
        self.model: Optional[Pi05Model] = None
        self._loaded = False

    def _auto_detect_config(self) -> VLAConfig:
        """Auto-detect hardware and create optimal config."""
        # Try to detect Jetson model
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                if "thor" in model:
                    logger.info("Detected Jetson Thor")
                    return VLAConfig.for_jetson_thor()
                elif "orin" in model:
                    logger.info("Detected Jetson Orin")
                    return VLAConfig.for_jetson_orin()
        except FileNotFoundError:
            pass

        # Check CUDA availability
        if HAS_TORCH and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)

            if memory_gb >= 100:  # Likely Thor
                logger.info(f"Detected GPU with {memory_gb:.1f}GB, using Thor config")
                return VLAConfig.for_jetson_thor()
            elif memory_gb >= 32:  # Likely Orin or high-end GPU
                logger.info(f"Detected GPU with {memory_gb:.1f}GB, using Orin config")
                return VLAConfig.for_jetson_orin()

        logger.info("Using development config")
        return VLAConfig.for_development()

    def load(self) -> None:
        """Load the Pi0.5 model."""
        if self._loaded:
            return

        logger.info(f"Loading Pi0.5 variant: {self.config.variant}")

        # Map variant string to enum
        variant_map = {
            "pi0_base": Pi05Variant.PI0_BASE,
            "pi0_fast_base": Pi05Variant.PI0_FAST_BASE,
            "pi05_base": Pi05Variant.PI05_BASE,
            "pi05_libero": Pi05Variant.PI05_LIBERO,
            "pi05_droid": Pi05Variant.PI05_DROID,
        }
        variant = variant_map.get(self.config.variant, Pi05Variant.PI05_BASE)

        # Create Pi05Config
        pi05_config = Pi05Config(
            variant=variant,
            device=self.config.device,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
        )

        self.model = Pi05Model(pi05_config)
        self.model.load()

        self._loaded = True
        logger.info("Pi0.5 model loaded successfully")

    def infer(
        self,
        images: Any,
        instruction: str,
        proprio: Optional[Any] = None,
        gripper: Optional[float] = None,
        **kwargs
    ) -> VLAResult:
        """
        Run VLA inference.

        Args:
            images: Image observations [N, H, W, C] or [B, N, H, W, C]
            instruction: Language instruction
            proprio: Proprioceptive state [DOF] or [B, DOF]
            gripper: Gripper state (optional)
            **kwargs: Additional observation fields

        Returns:
            VLAResult with predicted actions
        """
        if not self._loaded:
            self.load()

        # Convert to Pi05Observation
        obs = Pi05Observation(
            images=images,
            instruction=instruction,
            proprio=proprio,
            gripper=gripper,
        )

        # Run inference
        result = self.model.infer(obs)

        return VLAResult(
            actions=result.actions,
            action_horizon=result.action_horizon if result.action_horizon else self.config.action_horizon,
            action_dim=result.actions.shape[-1] if hasattr(result.actions, 'shape') else self.config.action_dim,
            confidence=result.confidence,
            inference_time_ms=result.inference_time_ms,
        )

    def unload(self) -> None:
        """Unload the model."""
        if self.model is not None:
            self.model.unload()
            self.model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @classmethod
    def create_for_hardware(cls) -> "VLAInterface":
        """Create interface with auto-detected hardware config."""
        return cls(config=None)

    @classmethod
    def for_jetson_thor(cls) -> "VLAInterface":
        """Create interface optimized for Jetson Thor."""
        return cls(config=VLAConfig.for_jetson_thor())

    @classmethod
    def for_jetson_orin(cls) -> "VLAInterface":
        """Create interface optimized for Jetson Orin."""
        return cls(config=VLAConfig.for_jetson_orin())

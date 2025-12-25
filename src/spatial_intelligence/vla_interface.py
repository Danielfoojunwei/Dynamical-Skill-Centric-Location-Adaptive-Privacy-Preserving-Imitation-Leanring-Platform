"""
Unified VLA (Vision-Language-Action) Interface

This module provides a unified interface for VLA model inference,
abstracting over different backends:

1. **Pi0 Custom**: Gemma 3 backbone with custom MoE (local implementation)
2. **Pi0.5 OpenPI**: Official Physical Intelligence implementation

Jetson Thor Optimization:
========================
- 128GB memory supports Gemma 3-27B or full Pi0.5
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .pi0 import Pi0, Pi0Config, VLMBackbone
    HAS_PI0 = True
except ImportError:
    HAS_PI0 = False

try:
    from .pi0 import Pi05Backend, Pi05Config, Pi05Variant, Pi05Observation
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False


class VLABackendType(Enum):
    """Available VLA backends."""
    PI0_CUSTOM = "pi0_custom"       # Custom Pi0 with Gemma 3
    PI05_OPENPI = "pi05_openpi"     # Official OpenPI Pi0.5
    AUTO = "auto"                   # Auto-select based on availability


class HardwareTarget(Enum):
    """Hardware targets for optimization."""
    JETSON_THOR = "jetson_thor"     # 128GB, 2070 TFLOPS
    JETSON_ORIN = "jetson_orin"     # 64GB, 275 TFLOPS
    GENERIC_GPU = "generic_gpu"     # Generic CUDA GPU
    CPU = "cpu"                     # CPU fallback


@dataclass
class VLAConfig:
    """Unified VLA configuration."""
    # Backend selection
    backend: VLABackendType = VLABackendType.AUTO

    # Hardware target
    hardware: HardwareTarget = HardwareTarget.JETSON_THOR

    # Pi0 Custom configuration
    pi0_vlm_backbone: str = "google/gemma-3-27b-it"
    pi0_action_dim: int = 7
    pi0_action_horizon: int = 16
    pi0_moe_depth: int = 24

    # Pi0.5 OpenPI configuration
    pi05_variant: str = "pi05_base"
    pi05_checkpoint_dir: Optional[str] = None

    # Inference configuration
    device: str = "cuda"
    dtype: str = "float16"
    use_tensorrt: bool = True
    use_fp8: bool = True
    use_flash_attention: bool = True

    # Performance tuning
    batch_size: int = 1
    num_inference_steps: int = 10

    @classmethod
    def for_jetson_thor(cls) -> "VLAConfig":
        """Optimal configuration for Jetson Thor."""
        return cls(
            backend=VLABackendType.PI05_OPENPI,  # Use official Pi0.5
            hardware=HardwareTarget.JETSON_THOR,
            pi0_vlm_backbone="google/gemma-3-27b-it",  # Thor can run 27B
            pi0_moe_depth=24,
            pi05_variant="pi05_base",
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
            backend=VLABackendType.PI0_CUSTOM,
            hardware=HardwareTarget.JETSON_ORIN,
            pi0_vlm_backbone="google/gemma-3-4b-it",  # Fits in Orin memory
            pi0_moe_depth=18,
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
            backend=VLABackendType.PI0_CUSTOM,
            hardware=HardwareTarget.GENERIC_GPU,
            pi0_vlm_backbone="google/paligemma-3b-pt-224",
            pi0_moe_depth=12,
            num_inference_steps=5,
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

    # Optional: image masks for multi-camera setup
    image_masks: Optional[Any] = None

    # Optional: depth images
    depth: Optional[Any] = None

    # Optional: object detections for semantic grounding
    detections: Optional[List[Dict]] = None


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


class VLABackend(ABC):
    """Abstract base class for VLA backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def infer(self, observation: VLAObservation) -> VLAResult:
        """Run inference."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class Pi0CustomBackend(VLABackend):
    """Backend using custom Pi0 with Gemma 3."""

    def __init__(self, config: VLAConfig):
        self.config = config
        self.model = None
        self._loaded = False

        if not HAS_PI0:
            raise ImportError("Pi0 module not available")

    def load(self) -> None:
        if self._loaded:
            return

        logger.info(f"Loading Pi0 with backbone: {self.config.pi0_vlm_backbone}")

        # Map string backbone to VLMBackbone enum
        backbone_map = {
            "google/paligemma-3b-pt-224": VLMBackbone.PALIGEMMA_3B,
            "google/gemma-3-4b-it": VLMBackbone.GEMMA3_4B,
            "google/gemma-3-12b-it": VLMBackbone.GEMMA3_12B,
            "google/gemma-3-27b-it": VLMBackbone.GEMMA3_27B,
        }
        vlm_backbone = backbone_map.get(
            self.config.pi0_vlm_backbone,
            VLMBackbone.GEMMA3_12B
        )

        # Create Pi0 config
        pi0_config = Pi0Config(
            vlm_backbone=vlm_backbone,
            action_dim=self.config.pi0_action_dim,
            action_horizon=self.config.pi0_action_horizon,
            moe_depth=self.config.pi0_moe_depth,
            num_inference_steps=self.config.num_inference_steps,
            dtype=self.config.dtype,
            device=self.config.device,
            use_flash_attention=self.config.use_flash_attention,
        )

        self.model = Pi0.from_config(pi0_config)
        self.model.to(self.config.device)
        self.model.eval()

        self._loaded = True
        logger.info("Pi0 model loaded successfully")

    def infer(self, observation: VLAObservation) -> VLAResult:
        if not self._loaded:
            self.load()

        import time
        start_time = time.perf_counter()

        # Prepare inputs
        images = observation.images
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)

        # Ensure correct shape [B, N, C, H, W]
        if images.dim() == 4:
            images = images.unsqueeze(0)

        # Process language instruction
        language_tokens = self.model.vlm_processor(
            text=observation.instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.model.vlm_max_text_tokens,
        )["input_ids"]

        # Prepare masks
        batch_size = images.shape[0]
        num_cams = images.shape[1]
        image_masks = observation.image_masks
        if image_masks is None:
            image_masks = torch.ones(batch_size, num_cams, device=self.config.device)

        language_masks = torch.ones_like(language_tokens)

        # Prepare proprio
        proprio = observation.proprio
        if proprio is None:
            proprio = torch.zeros(batch_size, 21, device=self.config.device)
        elif not isinstance(proprio, torch.Tensor):
            proprio = torch.tensor(proprio)
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        # Move to device
        images = images.to(self.config.device)
        image_masks = image_masks.to(self.config.device)
        language_tokens = language_tokens.to(self.config.device)
        language_masks = language_masks.to(self.config.device)
        proprio = proprio.to(self.config.device)

        # Run inference
        with torch.inference_mode():
            actions = self.model(
                images=images,
                image_masks=image_masks,
                language_tokens=language_tokens,
                language_masks=language_masks,
                proprio_states=proprio,
            )

        inference_time = (time.perf_counter() - start_time) * 1000

        return VLAResult(
            actions=actions.cpu().numpy() if hasattr(actions, 'cpu') else actions,
            action_horizon=self.config.pi0_action_horizon,
            action_dim=self.config.pi0_action_dim,
            inference_time_ms=inference_time,
        )

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class Pi05OpenPIBackend(VLABackend):
    """Backend using official Physical Intelligence Pi0.5."""

    def __init__(self, config: VLAConfig):
        self.config = config
        self.backend = None
        self._loaded = False

        if not HAS_OPENPI:
            raise ImportError(
                "OpenPI not available. Install with:\n"
                "pip install git+https://github.com/Physical-Intelligence/openpi.git"
            )

    def load(self) -> None:
        if self._loaded:
            return

        logger.info(f"Loading Pi0.5 variant: {self.config.pi05_variant}")

        # Map variant string to enum
        variant_map = {
            "pi0_base": Pi05Variant.PI0_BASE,
            "pi0_fast_base": Pi05Variant.PI0_FAST_BASE,
            "pi05_base": Pi05Variant.PI05_BASE,
            "pi05_libero": Pi05Variant.PI05_LIBERO,
            "pi05_droid": Pi05Variant.PI05_DROID,
        }
        variant = variant_map.get(self.config.pi05_variant, Pi05Variant.PI05_BASE)

        # Create Pi05Config
        pi05_config = Pi05Config(
            variant=variant,
            checkpoint_dir=self.config.pi05_checkpoint_dir,
            use_pytorch=True,
            device=self.config.device,
            dtype=self.config.dtype,
            use_tensorrt=self.config.use_tensorrt,
            use_fp8=self.config.use_fp8,
        )

        self.backend = Pi05Backend(pi05_config)
        self.backend.load()

        self._loaded = True
        logger.info("Pi0.5 model loaded successfully")

    def infer(self, observation: VLAObservation) -> VLAResult:
        if not self._loaded:
            self.load()

        import time
        start_time = time.perf_counter()

        # Convert to Pi05Observation
        pi05_obs = Pi05Observation(
            images=observation.images,
            instruction=observation.instruction,
            proprio=observation.proprio,
            detections=observation.detections,
            depth=observation.depth,
        )

        # Run inference
        result = self.backend.infer(pi05_obs, return_intermediate=True)

        inference_time = (time.perf_counter() - start_time) * 1000

        return VLAResult(
            actions=result["actions"],
            action_horizon=result.get("action_horizon", 16),
            action_dim=result["actions"].shape[-1] if hasattr(result["actions"], 'shape') else 7,
            confidence=result.get("confidence"),
            subtask=result.get("subtask_prediction"),
            inference_time_ms=inference_time,
        )

    def unload(self) -> None:
        if self.backend is not None:
            self.backend.unload()
            self.backend = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class VLAInterface:
    """
    Unified interface for VLA model inference.

    Automatically selects the best available backend based on
    configuration and available dependencies.
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        """
        Initialize VLA interface.

        Args:
            config: VLA configuration. If None, auto-detects hardware.
        """
        self.config = config or self._auto_detect_config()
        self.backend = self._create_backend()

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

    def _create_backend(self) -> VLABackend:
        """Create the appropriate backend based on config."""
        backend_type = self.config.backend

        # Auto-select backend
        if backend_type == VLABackendType.AUTO:
            if HAS_OPENPI:
                backend_type = VLABackendType.PI05_OPENPI
                logger.info("Auto-selected Pi0.5 OpenPI backend")
            elif HAS_PI0:
                backend_type = VLABackendType.PI0_CUSTOM
                logger.info("Auto-selected Pi0 Custom backend")
            else:
                raise ImportError("No VLA backend available")

        # Create backend
        if backend_type == VLABackendType.PI05_OPENPI:
            return Pi05OpenPIBackend(self.config)
        elif backend_type == VLABackendType.PI0_CUSTOM:
            return Pi0CustomBackend(self.config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def load(self) -> None:
        """Load the model."""
        self.backend.load()

    def infer(
        self,
        images: Any,
        instruction: str,
        proprio: Optional[Any] = None,
        **kwargs
    ) -> VLAResult:
        """
        Run VLA inference.

        Args:
            images: Image observations [N, C, H, W] or [B, N, C, H, W]
            instruction: Language instruction
            proprio: Proprioceptive state [DOF] or [B, DOF]
            **kwargs: Additional observation fields

        Returns:
            VLAResult with predicted actions
        """
        observation = VLAObservation(
            images=images,
            instruction=instruction,
            proprio=proprio,
            **kwargs
        )
        return self.backend.infer(observation)

    def unload(self) -> None:
        """Unload the model."""
        self.backend.unload()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.backend.is_loaded

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

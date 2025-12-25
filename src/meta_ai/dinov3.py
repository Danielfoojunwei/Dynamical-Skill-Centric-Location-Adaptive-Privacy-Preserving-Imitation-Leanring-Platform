"""
DINOv3 Vision Encoder for Dynamical Edge Platform

DINOv3 (Self-DIstillation with NO labels v3) provides state-of-the-art
self-supervised visual features for downstream tasks.

Features:
- Self-supervised learning (no labels required)
- Strong transfer to robotics tasks
- Multiple model sizes (ViT-S, ViT-B, ViT-L, ViT-H+, ViT-7B)
- Dense feature extraction for pixel-level tasks
- ConvNeXt variants for efficient deployment

Repository: github.com/facebookresearch/dinov3
License: Apache 2.0

Integration with Dynamical Platform:
- Replaces/augments base VLA vision encoder
- Features fed to MoE skill router
- Supports N2HE encryption for privacy

Model Loading:
- Primary: HuggingFace Transformers (v4.56.0+)
- Fallback: PyTorch Hub with local repo
- Mock: For development without GPU/models
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
    torch = None
    nn = None
    F = None
    logger.warning("PyTorch not available - using mock DINOv3")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# HuggingFace Transformers support
try:
    from transformers import AutoImageProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoImageProcessor = None
    AutoModel = None


# =============================================================================
# Configuration
# =============================================================================

class DINOv3ModelSize(str, Enum):
    """DINOv3 model sizes matching facebookresearch/dinov3 variants."""
    SMALL = "vits16"         # ViT-S/16, fastest
    BASE = "vitb16"          # ViT-B/16, balanced
    LARGE = "vitl16"         # ViT-L/16, high quality
    HUGE_PLUS = "vith16plus" # ViT-H+/16, very high quality
    GIANT_7B = "vit7b16"     # ViT-7B/16, best quality (~7B params)
    # ConvNeXt variants for edge deployment
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_BASE = "convnext_base"
    CONVNEXT_LARGE = "convnext_large"


class DINOv3Dataset(str, Enum):
    """Training dataset for normalization selection."""
    LVD = "lvd1689m"    # Web images (1.689B images)
    SAT = "sat493m"     # Satellite imagery (493M images)


@dataclass
class DINOv3Config:
    """Configuration for DINOv3 encoder."""
    # Model selection
    model_size: DINOv3ModelSize = DINOv3ModelSize.LARGE
    dataset: DINOv3Dataset = DINOv3Dataset.LVD  # Training dataset for normalization

    # Input configuration
    input_size: int = 518           # DINOv3 native resolution
    patch_size: int = 14            # ViT patch size

    # Feature extraction
    output_dim: int = 1024          # Feature dimension (model-dependent)
    use_cls_token: bool = True      # Use [CLS] token for global features
    use_patch_tokens: bool = True   # Use patch tokens for dense features
    num_register_tokens: int = 4    # Register tokens (DINOv3 feature)

    # Loading options
    use_huggingface: bool = True    # Prefer HuggingFace over torch.hub
    local_repo_path: Optional[str] = None  # Path to cloned dinov3 repo

    # Optimization for edge deployment
    use_fp16: bool = True           # Half precision for speed
    compile_model: bool = False     # torch.compile for optimization

    # Cache settings
    cache_dir: str = "/var/lib/dynamical/models/dinov3"

    # Device
    device: str = "cuda"

    @property
    def num_patches(self) -> int:
        """Number of patches in the image."""
        return (self.input_size // self.patch_size) ** 2

    @property
    def feature_map_size(self) -> int:
        """Spatial size of the feature map."""
        return self.input_size // self.patch_size

    @property
    def huggingface_model_id(self) -> str:
        """Get HuggingFace model ID for this configuration."""
        # Map model size to HuggingFace model ID
        size_map = {
            DINOv3ModelSize.CONVNEXT_TINY: "convnext-tiny",
            DINOv3ModelSize.CONVNEXT_BASE: "convnext-base",
            DINOv3ModelSize.CONVNEXT_LARGE: "convnext-large",
            DINOv3ModelSize.SMALL: "vits16",
            DINOv3ModelSize.BASE: "vitb16",
            DINOv3ModelSize.LARGE: "vitl16",
            DINOv3ModelSize.HUGE_PLUS: "vith16plus",
            DINOv3ModelSize.GIANT_7B: "vit7b16",
        }
        size_str = size_map.get(self.model_size, "vitl16")
        dataset_str = self.dataset.value
        return f"facebook/dinov3-{size_str}-pretrain-{dataset_str}"

    @property
    def torch_hub_model_name(self) -> str:
        """Get torch.hub model name."""
        return f"dinov3_{self.model_size.value}"


@dataclass
class DINOv3Features:
    """Output features from DINOv3."""
    # Global feature (from [CLS] token)
    global_features: np.ndarray     # Shape: [batch, output_dim]

    # Dense features (from patch tokens)
    dense_features: Optional[np.ndarray] = None  # Shape: [batch, H, W, output_dim]

    # Intermediate features for multi-scale
    intermediate_features: Optional[List[np.ndarray]] = None

    # Metadata
    model_size: str = ""
    inference_time_ms: float = 0.0

    @property
    def feature_dim(self) -> int:
        return self.global_features.shape[-1]


# =============================================================================
# DINOv3 Model Wrapper
# =============================================================================

class DINOv3Encoder:
    """
    DINOv3 Vision Encoder for robot perception.

    Provides self-supervised visual features optimized for:
    - Object recognition and localization
    - Scene understanding
    - Visual feature extraction for skill learning

    Usage:
        encoder = DINOv3Encoder()
        encoder.load_model()

        # Single image
        features = encoder.encode(image)

        # Batch of images
        features = encoder.encode_batch(images)

        # Dense features for segmentation
        dense = encoder.get_dense_features(image)
    """

    # Model dimension mapping (from DINOv3 paper/repo)
    MODEL_DIMS = {
        DINOv3ModelSize.SMALL: 384,
        DINOv3ModelSize.BASE: 768,
        DINOv3ModelSize.LARGE: 1024,
        DINOv3ModelSize.HUGE_PLUS: 1280,
        DINOv3ModelSize.GIANT_7B: 1536,
        DINOv3ModelSize.CONVNEXT_TINY: 768,
        DINOv3ModelSize.CONVNEXT_BASE: 1024,
        DINOv3ModelSize.CONVNEXT_LARGE: 1536,
    }

    # HuggingFace model IDs
    HF_MODEL_IDS = {
        DINOv3ModelSize.CONVNEXT_TINY: "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        DINOv3ModelSize.CONVNEXT_BASE: "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        DINOv3ModelSize.CONVNEXT_LARGE: "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    }

    def __init__(self, config: DINOv3Config = None):
        self.config = config or DINOv3Config()

        # Update output_dim based on model size
        self.config.output_dim = self.MODEL_DIMS.get(
            self.config.model_size, 1024
        )

        # Model state
        self.model = None
        self.transform = None
        self.processor = None  # HuggingFace processor
        self._is_loaded = False
        self._using_huggingface = False
        self._using_mock = False

        # Statistics
        self.stats = {
            "images_processed": 0,
            "total_inference_time_ms": 0.0,
            "avg_inference_time_ms": 0.0,
            "backend": "not_loaded",
        }

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_model(self, weights_path: Optional[str] = None) -> bool:
        """
        Load DINOv3 model weights.

        Loading priority:
        1. Local weights file (if provided)
        2. HuggingFace Transformers (if available and configured)
        3. PyTorch Hub (with local or remote repo)
        4. Mock model (for development)

        Args:
            weights_path: Path to custom weights. If None, downloads from hub.

        Returns:
            True if successful
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available - using mock model")
            self._is_loaded = True
            self._using_mock = True
            self.stats["backend"] = "mock"
            return True

        try:
            # Option 1: Load from local weights file
            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading DINOv3 from local weights: {weights_path}")
                self.model = self._load_local_weights(weights_path)
                self.stats["backend"] = "local_weights"

            # Option 2: Try HuggingFace Transformers
            elif self.config.use_huggingface and HAS_TRANSFORMERS:
                loaded = self._load_from_huggingface()
                if loaded:
                    self.stats["backend"] = "huggingface"
                else:
                    # Fall back to torch.hub
                    loaded = self._load_from_torch_hub()
                    if loaded:
                        self.stats["backend"] = "torch_hub"
                    else:
                        self._create_and_set_mock_model()

            # Option 3: Try PyTorch Hub
            else:
                loaded = self._load_from_torch_hub()
                if loaded:
                    self.stats["backend"] = "torch_hub"
                else:
                    self._create_and_set_mock_model()

            # Move to device
            if self.model is not None and torch.cuda.is_available() and self.config.device == "cuda":
                self.model = self.model.cuda()

            # Set eval mode
            if self.model is not None and hasattr(self.model, 'eval'):
                self.model.eval()

            # Use FP16 if configured
            if self.config.use_fp16 and torch.cuda.is_available() and self.model is not None:
                if hasattr(self.model, 'half'):
                    self.model = self.model.half()

            # Compile model for optimization
            if self.config.compile_model and hasattr(torch, 'compile') and self.model is not None:
                self.model = torch.compile(self.model)

            # Setup transform (only if not using HuggingFace processor)
            if not self._using_huggingface:
                self._setup_transform()

            self._is_loaded = True
            logger.info(f"DINOv3 {self.config.model_size.value} loaded successfully via {self.stats['backend']}")

            return True

        except Exception as e:
            logger.error(f"Failed to load DINOv3: {e}")
            return False

    def _load_from_huggingface(self) -> bool:
        """Load model from HuggingFace Transformers."""
        if not HAS_TRANSFORMERS:
            return False

        try:
            model_id = self.config.huggingface_model_id
            logger.info(f"Loading DINOv3 from HuggingFace: {model_id}")

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
            self._using_huggingface = True

            logger.info(f"Successfully loaded DINOv3 from HuggingFace: {model_id}")
            return True

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            return False

    def _load_from_torch_hub(self) -> bool:
        """Load model from PyTorch Hub."""
        try:
            model_name = self.config.torch_hub_model_name
            logger.info(f"Loading DINOv3 from torch.hub: {model_name}")

            # Try with local repo first if specified
            if self.config.local_repo_path and os.path.exists(self.config.local_repo_path):
                self.model = torch.hub.load(
                    self.config.local_repo_path,
                    model_name,
                    source='local',
                    trust_repo=True,
                )
            else:
                # Try loading from GitHub
                self.model = torch.hub.load(
                    'facebookresearch/dinov3',
                    model_name,
                    trust_repo=True,
                )

            logger.info(f"Successfully loaded DINOv3 from torch.hub: {model_name}")
            return True

        except Exception as e:
            logger.warning(f"Could not load from torch.hub: {e}")
            return False

    def _create_and_set_mock_model(self):
        """Create and set mock model as fallback."""
        logger.info("Creating mock DINOv3 model for development")
        self.model = self._create_mock_model()
        self._using_mock = True
        self.stats["backend"] = "mock"

    def _create_mock_model(self) -> Any:
        """Create a mock DINOv3 model for testing."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for mock model")
        class MockDINOv3(nn.Module):
            def __init__(self, output_dim: int):
                super().__init__()
                self.output_dim = output_dim
                self.proj = nn.Linear(3 * 14 * 14, output_dim)  # Simple projection

            def forward(self, x):
                batch_size = x.shape[0]
                # Downsample and flatten
                x = F.adaptive_avg_pool2d(x, (14, 14))
                x = x.view(batch_size, -1)
                # Project to feature dim
                features = self.proj(x)
                return features

            def forward_features(self, x):
                batch_size = x.shape[0]
                # Create patch tokens
                x = F.adaptive_avg_pool2d(x, (37, 37))  # 518/14 = 37
                patches = x.unfold(2, 1, 1).unfold(3, 1, 1)
                patches = patches.reshape(batch_size, -1, 3)
                # Project each patch
                patch_features = torch.randn(batch_size, 37*37, self.output_dim, device=x.device)
                cls_token = torch.randn(batch_size, 1, self.output_dim, device=x.device)
                return torch.cat([cls_token, patch_features], dim=1)

        return MockDINOv3(self.config.output_dim)

    def _load_local_weights(self, path: str) -> Any:
        """Load model from local weights file."""
        # Create base model architecture
        model = self._create_mock_model()

        # Load state dict
        state_dict = torch.load(path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # Try to load weights
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Could not load all weights: {e}")

        return model

    def _setup_transform(self):
        """Setup image preprocessing transform."""
        if not HAS_TORCH:
            return

        # DINOv3 normalization depends on training dataset
        if self.config.dataset == DINOv3Dataset.SAT:
            # Satellite imagery normalization
            mean = [0.430, 0.411, 0.296]
            std = [0.213, 0.156, 0.143]
        else:
            # LVD (web images) - ImageNet-style normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        class DINOv3Transform:
            def __init__(self, size: int, mean: List[float], std: List[float]):
                self.size = size
                self.mean = torch.tensor(mean).view(3, 1, 1)
                self.std = torch.tensor(std).view(3, 1, 1)

            def __call__(self, image: Union[np.ndarray, 'Image.Image', torch.Tensor]) -> torch.Tensor:
                # Convert to tensor
                if isinstance(image, np.ndarray):
                    if image.dtype == np.uint8:
                        image = image.astype(np.float32) / 255.0
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)
                    if image.shape[-1] == 3:
                        image = image.transpose(2, 0, 1)
                    tensor = torch.from_numpy(image)
                elif HAS_PIL and isinstance(image, Image.Image):
                    import torchvision.transforms.functional as TF
                    tensor = TF.to_tensor(image)
                elif isinstance(image, torch.Tensor):
                    tensor = image
                else:
                    raise TypeError(f"Unsupported image type: {type(image)}")

                # Ensure 4D tensor [B, C, H, W]
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)

                # Resize
                tensor = F.interpolate(
                    tensor,
                    size=(self.size, self.size),
                    mode='bicubic',
                    align_corners=False
                )

                # Normalize
                mean = self.mean.to(tensor.device)
                std = self.std.to(tensor.device)
                tensor = (tensor - mean) / std

                return tensor

        self.transform = DINOv3Transform(self.config.input_size, mean, std)

    def encode(
        self,
        image: Union[np.ndarray, Any],
        return_dense: bool = False,
    ) -> DINOv3Features:
        """
        Encode a single image to features.

        Args:
            image: Input image [H, W, 3] or PIL Image
            return_dense: Also return dense (patch-level) features

        Returns:
            DINOv3Features with global and optionally dense features
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        if not HAS_TORCH or self._using_mock:
            # Mock encoding
            global_features = np.random.randn(1, self.config.output_dim).astype(np.float32)
            dense_features = None
            if return_dense:
                h = w = self.config.feature_map_size
                dense_features = np.random.randn(1, h, w, self.config.output_dim).astype(np.float32)

            inference_time = (time.time() - start_time) * 1000
            return DINOv3Features(
                global_features=global_features,
                dense_features=dense_features,
                model_size=self.config.model_size.value,
                inference_time_ms=inference_time,
            )

        # Use HuggingFace processor if available
        if self._using_huggingface and self.processor is not None:
            return self._encode_huggingface(image, return_dense, start_time)

        # Preprocess with custom transform
        tensor = self.transform(image)

        # Move to device
        if torch.cuda.is_available() and self.config.device == "cuda":
            tensor = tensor.cuda()
            if self.config.use_fp16:
                tensor = tensor.half()

        # Forward pass
        with torch.no_grad():
            if return_dense and hasattr(self.model, 'forward_features'):
                # Get all tokens including patches
                all_tokens = self.model.forward_features(tensor)

                # Split CLS and patch tokens
                cls_token = all_tokens[:, 0]  # [B, D]
                patch_tokens = all_tokens[:, 1:]  # [B, N, D]

                # Reshape patch tokens to spatial grid
                h = w = self.config.feature_map_size
                dense = patch_tokens[:, :h*w].reshape(-1, h, w, self.config.output_dim)

                global_features = cls_token.cpu().float().numpy()
                dense_features = dense.cpu().float().numpy()
            else:
                # Just get global features
                global_features = self.model(tensor)
                if isinstance(global_features, tuple):
                    global_features = global_features[0]
                global_features = global_features.cpu().float().numpy()
                dense_features = None

        inference_time = (time.time() - start_time) * 1000

        # Update stats
        self.stats["images_processed"] += 1
        self.stats["total_inference_time_ms"] += inference_time
        self.stats["avg_inference_time_ms"] = (
            self.stats["total_inference_time_ms"] / self.stats["images_processed"]
        )

        return DINOv3Features(
            global_features=global_features,
            dense_features=dense_features,
            model_size=self.config.model_size.value,
            inference_time_ms=inference_time,
        )

    def _encode_huggingface(
        self,
        image: Union[np.ndarray, Any],
        return_dense: bool,
        start_time: float,
    ) -> DINOv3Features:
        """Encode using HuggingFace model and processor."""
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            if HAS_PIL:
                if image.dtype == np.uint8:
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                raise RuntimeError("PIL required for HuggingFace processing")
        else:
            pil_image = image

        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if self.config.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Extract features based on model type
            if hasattr(outputs, 'last_hidden_state'):
                # Transformer-style output
                hidden_state = outputs.last_hidden_state  # [B, N, D]

                # CLS token is usually first
                global_features = hidden_state[:, 0].cpu().float().numpy()

                dense_features = None
                if return_dense:
                    # Patch tokens (excluding CLS)
                    patch_tokens = hidden_state[:, 1:]
                    h = w = int(np.sqrt(patch_tokens.shape[1]))
                    if h * w == patch_tokens.shape[1]:
                        dense = patch_tokens.reshape(-1, h, w, patch_tokens.shape[-1])
                        dense_features = dense.cpu().float().numpy()

            elif hasattr(outputs, 'pooler_output'):
                global_features = outputs.pooler_output.cpu().float().numpy()
                dense_features = None
            else:
                # Fallback: use first output
                global_features = outputs[0].cpu().float().numpy()
                if global_features.ndim == 3:
                    global_features = global_features[:, 0]
                dense_features = None

        inference_time = (time.time() - start_time) * 1000

        # Update stats
        self.stats["images_processed"] += 1
        self.stats["total_inference_time_ms"] += inference_time
        self.stats["avg_inference_time_ms"] = (
            self.stats["total_inference_time_ms"] / self.stats["images_processed"]
        )

        return DINOv3Features(
            global_features=global_features,
            dense_features=dense_features,
            model_size=self.config.model_size.value,
            inference_time_ms=inference_time,
        )

    def encode_batch(
        self,
        images: List[Union[np.ndarray, Any]],
        return_dense: bool = False,
    ) -> DINOv3Features:
        """
        Encode a batch of images.

        Args:
            images: List of input images
            return_dense: Also return dense features

        Returns:
            DINOv3Features with batched outputs
        """
        if not self._is_loaded:
            self.load_model()

        if not HAS_TORCH:
            # Mock batch encoding
            batch_size = len(images)
            global_features = np.random.randn(batch_size, self.config.output_dim).astype(np.float32)
            dense_features = None
            if return_dense:
                h = w = self.config.feature_map_size
                dense_features = np.random.randn(batch_size, h, w, self.config.output_dim).astype(np.float32)

            return DINOv3Features(
                global_features=global_features,
                dense_features=dense_features,
                model_size=self.config.model_size.value,
            )

        start_time = time.time()

        # Preprocess all images
        tensors = [self.transform(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        # Move to device
        if torch.cuda.is_available() and self.config.device == "cuda":
            batch = batch.cuda()
            if self.config.use_fp16:
                batch = batch.half()

        # Forward pass
        with torch.no_grad():
            if return_dense and hasattr(self.model, 'forward_features'):
                all_tokens = self.model.forward_features(batch)
                cls_token = all_tokens[:, 0]
                patch_tokens = all_tokens[:, 1:]

                h = w = self.config.feature_map_size
                dense = patch_tokens[:, :h*w].reshape(-1, h, w, self.config.output_dim)

                global_features = cls_token.cpu().float().numpy()
                dense_features = dense.cpu().float().numpy()
            else:
                global_features = self.model(batch)
                if isinstance(global_features, tuple):
                    global_features = global_features[0]
                global_features = global_features.cpu().float().numpy()
                dense_features = None

        inference_time = (time.time() - start_time) * 1000

        # Update stats
        self.stats["images_processed"] += len(images)
        self.stats["total_inference_time_ms"] += inference_time

        return DINOv3Features(
            global_features=global_features,
            dense_features=dense_features,
            model_size=self.config.model_size.value,
            inference_time_ms=inference_time,
        )

    def get_dense_features(
        self,
        image: Union[np.ndarray, Any],
    ) -> np.ndarray:
        """
        Get dense (patch-level) features for segmentation tasks.

        Args:
            image: Input image

        Returns:
            Dense features [H, W, D]
        """
        features = self.encode(image, return_dense=True)
        if features.dense_features is not None:
            return features.dense_features[0]  # Remove batch dim
        return None

    def compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Cosine similarity score
        """
        f1 = features1.flatten()
        f2 = features2.flatten()

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(f1, f2) / (norm1 * norm2))

    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            **self.stats,
            "model_size": self.config.model_size.value,
            "output_dim": self.config.output_dim,
            "is_loaded": self._is_loaded,
            "using_huggingface": self._using_huggingface,
            "using_mock": self._using_mock,
            "dataset": self.config.dataset.value,
        }


# =============================================================================
# Testing
# =============================================================================

def test_dinov3():
    """Test DINOv3 encoder."""
    print("\n" + "=" * 60)
    print("DINOV3 VISION ENCODER TEST")
    print("=" * 60)

    # Create encoder with HuggingFace preference
    config = DINOv3Config(
        model_size=DINOv3ModelSize.LARGE,
        dataset=DINOv3Dataset.LVD,
        input_size=518,
        use_huggingface=True,
    )
    encoder = DINOv3Encoder(config)

    print("\n1. Load Model")
    print("-" * 40)
    success = encoder.load_model()
    print(f"   Model loaded: {success}")
    print(f"   Output dim: {encoder.config.output_dim}")
    print(f"   Backend: {encoder.stats['backend']}")
    print(f"   HuggingFace model ID: {config.huggingface_model_id}")

    print("\n2. Encode Single Image")
    print("-" * 40)

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    features = encoder.encode(test_image)
    print(f"   Global features shape: {features.global_features.shape}")
    print(f"   Inference time: {features.inference_time_ms:.2f}ms")

    print("\n3. Encode with Dense Features")
    print("-" * 40)

    features = encoder.encode(test_image, return_dense=True)
    print(f"   Global features shape: {features.global_features.shape}")
    if features.dense_features is not None:
        print(f"   Dense features shape: {features.dense_features.shape}")

    print("\n4. Batch Encoding")
    print("-" * 40)

    batch_images = [test_image] * 4
    features = encoder.encode_batch(batch_images)
    print(f"   Batch features shape: {features.global_features.shape}")

    print("\n5. Feature Similarity")
    print("-" * 40)

    features1 = encoder.encode(test_image)
    features2 = encoder.encode(test_image)  # Same image

    similarity = encoder.compute_similarity(
        features1.global_features,
        features2.global_features
    )
    print(f"   Self-similarity: {similarity:.4f}")

    print("\n6. Statistics")
    print("-" * 40)
    stats = encoder.get_statistics()
    print(f"   Images processed: {stats['images_processed']}")
    print(f"   Avg inference time: {stats['avg_inference_time_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("DINOV3 TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_dinov3()

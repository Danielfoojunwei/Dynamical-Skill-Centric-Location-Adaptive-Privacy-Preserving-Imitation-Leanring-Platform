"""
Pi0: A Vision-Language-Action Flow Model for General Robot Control.

This module implements the Pi0 model from Physical Intelligence for embodied AI,
with support for multiple VLM backbones including Gemma 3.

Supported Backbones:
===================
- PaliGemma 3B (legacy): google/paligemma-3b-pt-224
- Gemma 3 4B (multimodal): google/gemma-3-4b-it
- Gemma 3 12B (multimodal): google/gemma-3-12b-it
- Gemma 3 27B (multimodal): google/gemma-3-27b-it

Jetson Thor Optimizations:
=========================
With 128GB memory and 2070 TFLOPS, Jetson Thor can run:
- Gemma 3-27B: Full precision, ~50GB memory
- Gemma 3-12B: Full precision, ~24GB memory
- Gemma 3-4B: Full precision, ~8GB memory

IMPORTANT: This module requires PyTorch and Transformers to be installed.
Install with: pip install torch torchvision transformers>=4.56.0

References:
- Physical Intelligence: https://www.physicalintelligence.company/
- Gemma 3: https://huggingface.co/blog/gemma3
- PaliGemma: https://huggingface.co/google/paligemma-3b-pt-224
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union

# Require PyTorch - no mock fallback for production
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    raise ImportError(
        "PyTorch is required for Pi0 model. "
        "Install with: pip install torch torchvision"
    )

# Require Transformers for VLM backbone
try:
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        PaliGemmaForConditionalGeneration,
    )
    # Try to import Gemma 3 specific classes
    try:
        from transformers import Gemma3ForConditionalGeneration
        HAS_GEMMA3 = True
    except ImportError:
        HAS_GEMMA3 = False

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    HAS_GEMMA3 = False
    raise ImportError(
        "Transformers is required for Pi0 model. "
        "Install with: pip install transformers>=4.56.0"
    )

from .modules import ActionEncoder, GemmaMoE, MoeExpertConfig, SinusoidalPosEmb

logger = logging.getLogger(__name__)


class VLMBackbone(Enum):
    """Supported VLM backbone models."""
    # Legacy PaliGemma
    PALIGEMMA_3B = "google/paligemma-3b-pt-224"

    # Gemma 3 multimodal variants (March 2025)
    GEMMA3_4B = "google/gemma-3-4b-it"
    GEMMA3_12B = "google/gemma-3-12b-it"
    GEMMA3_27B = "google/gemma-3-27b-it"

    # Gemma 3 text-only variants
    GEMMA3_1B = "google/gemma-3-1b-it"
    GEMMA3_270M = "google/gemma-3-270m-it"


# VLM hidden sizes for each backbone
VLM_HIDDEN_SIZES = {
    VLMBackbone.PALIGEMMA_3B: 2048,
    VLMBackbone.GEMMA3_270M: 1536,
    VLMBackbone.GEMMA3_1B: 2048,
    VLMBackbone.GEMMA3_4B: 3072,
    VLMBackbone.GEMMA3_12B: 4608,
    VLMBackbone.GEMMA3_27B: 5120,
}

# Image resolutions for each backbone
VLM_IMAGE_SIZES = {
    VLMBackbone.PALIGEMMA_3B: 224,
    VLMBackbone.GEMMA3_4B: 448,
    VLMBackbone.GEMMA3_12B: 448,
    VLMBackbone.GEMMA3_27B: 448,
}

# Memory requirements (GB) for each backbone
VLM_MEMORY_GB = {
    VLMBackbone.PALIGEMMA_3B: 6,
    VLMBackbone.GEMMA3_270M: 1,
    VLMBackbone.GEMMA3_1B: 2,
    VLMBackbone.GEMMA3_4B: 8,
    VLMBackbone.GEMMA3_12B: 24,
    VLMBackbone.GEMMA3_27B: 54,
}


@dataclass
class Pi0Config:
    """Configuration for Pi0 model."""
    # VLM backbone selection
    vlm_backbone: VLMBackbone = VLMBackbone.GEMMA3_12B  # Default for Thor

    # Action configuration
    action_dim: int = 7
    action_horizon: int = 16

    # VLM expert configuration
    vlm_expert_intermediate_size: int = 16384
    vlm_expert_num_heads: int = 8
    vlm_expert_num_kv_heads: int = 1
    vlm_expert_head_dim: int = 256
    vlm_max_text_tokens: int = 128

    # Action expert configuration
    action_expert_width: int = 1024
    action_expert_intermediate_size: int = 4096
    action_expert_num_heads: int = 8
    action_expert_num_kv_heads: int = 1
    action_expert_head_dim: int = 256

    # MoE configuration
    moe_depth: int = 18

    # Flow matching configuration
    num_inference_steps: int = 10
    flow_sig_min: float = 0.001
    flow_alpha: float = 1.5
    flow_beta: float = 1.0

    # Hardware configuration
    dtype: str = "float16"  # float16, bfloat16, float32
    device: str = "cuda"

    # Jetson Thor optimizations
    use_tensorrt: bool = False
    use_fp8: bool = False  # Thor has native FP8 support
    use_flash_attention: bool = True

    @property
    def vlm_expert_width(self) -> int:
        """Get VLM expert width based on backbone."""
        return VLM_HIDDEN_SIZES.get(self.vlm_backbone, 2048)

    @property
    def image_size(self) -> int:
        """Get image size for backbone."""
        return VLM_IMAGE_SIZES.get(self.vlm_backbone, 224)

    @property
    def memory_required_gb(self) -> float:
        """Estimate memory required for this configuration."""
        vlm_mem = VLM_MEMORY_GB.get(self.vlm_backbone, 8)
        # Add ~2GB for MoE and action components
        return vlm_mem + 2.0

    @classmethod
    def for_jetson_thor(cls) -> "Pi0Config":
        """Optimal configuration for Jetson Thor (128GB, 2070 TFLOPS)."""
        return cls(
            vlm_backbone=VLMBackbone.GEMMA3_27B,  # Can run largest model
            action_dim=7,
            action_horizon=16,
            moe_depth=24,  # Deeper MoE with Thor's compute
            num_inference_steps=10,
            dtype="float16",
            use_tensorrt=True,
            use_fp8=True,
            use_flash_attention=True,
        )

    @classmethod
    def for_jetson_orin(cls) -> "Pi0Config":
        """Configuration for Jetson AGX Orin (64GB, 275 TFLOPS)."""
        return cls(
            vlm_backbone=VLMBackbone.GEMMA3_4B,  # Fits in memory
            action_dim=7,
            action_horizon=16,
            moe_depth=18,
            num_inference_steps=10,
            dtype="float16",
            use_tensorrt=True,
            use_fp8=False,
            use_flash_attention=True,
        )

    @classmethod
    def for_development(cls) -> "Pi0Config":
        """Lightweight configuration for development/testing."""
        return cls(
            vlm_backbone=VLMBackbone.PALIGEMMA_3B,
            action_dim=7,
            action_horizon=16,
            moe_depth=12,
            num_inference_steps=5,
            dtype="float32",
            use_tensorrt=False,
            use_fp8=False,
            use_flash_attention=False,
        )


# Global tokenizer for static method
_tokenizer = None

# Default backbone (updated to Gemma 3-12B for Thor)
LANGUAGE_MODEL_NAME = VLMBackbone.GEMMA3_12B.value
VLM_BACKBONE = VLMBackbone.GEMMA3_12B.value
VLM_EXPERT_WIDTH = VLM_HIDDEN_SIZES[VLMBackbone.GEMMA3_12B]


class Pi0(nn.Module):
    """
    Implementation of Pi0 model from the Physical Intelligence paper.

    Supports multiple VLM backbones:
    - PaliGemma 3B (legacy)
    - Gemma 3 4B/12B/27B (multimodal, recommended for Thor)

    Optimized for NVIDIA Jetson Thor with 128GB memory and 2070 TFLOPS.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        vlm_backbone: Union[VLMBackbone, str] = VLMBackbone.GEMMA3_12B,
        vlm_expert_intermediate_size: int = 16384,
        vlm_expert_num_heads: int = 8,
        vlm_expert_num_kv_heads: int = 1,
        vlm_expert_head_dim: int = 256,
        vlm_max_text_tokens: int = 128,
        action_expert_width: int = 1024,
        action_expert_intermediate_size: int = 4096,
        action_expert_num_heads: int = 8,
        action_expert_num_kv_heads: int = 1,
        action_expert_head_dim: int = 256,
        moe_depth: int = 18,
        num_inference_steps: int = 10,
        flow_sig_min: float = 0.001,
        flow_alpha: float = 1.5,
        flow_beta: float = 1.0,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_flash_attention: bool = True,
        config: Optional[Pi0Config] = None,
    ):
        super().__init__()

        # Use config if provided
        if config is not None:
            vlm_backbone = config.vlm_backbone
            action_dim = config.action_dim
            action_horizon = config.action_horizon
            vlm_expert_intermediate_size = config.vlm_expert_intermediate_size
            vlm_expert_num_heads = config.vlm_expert_num_heads
            vlm_expert_num_kv_heads = config.vlm_expert_num_kv_heads
            vlm_expert_head_dim = config.vlm_expert_head_dim
            vlm_max_text_tokens = config.vlm_max_text_tokens
            action_expert_width = config.action_expert_width
            action_expert_intermediate_size = config.action_expert_intermediate_size
            action_expert_num_heads = config.action_expert_num_heads
            action_expert_num_kv_heads = config.action_expert_num_kv_heads
            action_expert_head_dim = config.action_expert_head_dim
            moe_depth = config.moe_depth
            num_inference_steps = config.num_inference_steps
            flow_sig_min = config.flow_sig_min
            flow_alpha = config.flow_alpha
            flow_beta = config.flow_beta
            device = config.device
            use_flash_attention = config.use_flash_attention

            # Convert dtype string to torch dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(config.dtype, torch.float16)

        # Handle string backbone specification
        if isinstance(vlm_backbone, str):
            vlm_backbone = VLMBackbone(vlm_backbone)

        self.vlm_backbone_type = vlm_backbone
        self.vlm_backbone_name = vlm_backbone.value
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.vlm_max_text_tokens = vlm_max_text_tokens

        # Get VLM-specific dimensions
        self.vlm_hidden_size = VLM_HIDDEN_SIZES.get(vlm_backbone, 2048)
        self.image_size = VLM_IMAGE_SIZES.get(vlm_backbone, 224)

        # Assuming 3 cameras for now as per AGX config
        num_rgbs = 3
        # Image tokens depend on backbone
        if vlm_backbone in [VLMBackbone.GEMMA3_4B, VLMBackbone.GEMMA3_12B, VLMBackbone.GEMMA3_27B]:
            # Gemma 3 uses different image tokenization
            self.tokens_per_image = 576  # 24x24 patches
        else:
            self.tokens_per_image = 256  # PaliGemma default

        self.vlm_max_tokens = num_rgbs * self.tokens_per_image + self.vlm_max_text_tokens
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        self.dtype = dtype
        self.device = device

        # Proprioception dimension (joints + vel + torque)
        # Assuming 7 joints -> 7 + 7 + 7 = 21
        proprio_dim = 21

        # Determine attention implementation
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"

        # Load VLM backbone based on type
        logger.info(f"Loading VLM Backbone: {self.vlm_backbone_name}")
        logger.info(f"  Hidden size: {self.vlm_hidden_size}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Memory required: ~{VLM_MEMORY_GB.get(vlm_backbone, 8)}GB")

        if vlm_backbone == VLMBackbone.PALIGEMMA_3B:
            # Legacy PaliGemma loading
            self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
                self.vlm_backbone_name,
                torch_dtype=self.dtype,
                attn_implementation="eager",  # PaliGemma doesn't support flash
            )
            self.is_gemma3 = False
        elif vlm_backbone in [VLMBackbone.GEMMA3_4B, VLMBackbone.GEMMA3_12B, VLMBackbone.GEMMA3_27B]:
            # Gemma 3 multimodal loading
            if HAS_GEMMA3:
                self.vlm = Gemma3ForConditionalGeneration.from_pretrained(
                    self.vlm_backbone_name,
                    torch_dtype=self.dtype,
                    attn_implementation=attn_impl,
                )
            else:
                # Fallback to AutoModel
                self.vlm = AutoModel.from_pretrained(
                    self.vlm_backbone_name,
                    torch_dtype=self.dtype,
                    attn_implementation=attn_impl,
                    trust_remote_code=True,
                )
            self.is_gemma3 = True
        else:
            # Text-only Gemma 3 variants
            self.vlm = AutoModelForCausalLM.from_pretrained(
                self.vlm_backbone_name,
                torch_dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            self.is_gemma3 = True

        self.vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_backbone_name, padding_side="right"
        )
        self.vlm_embedding_module = self.vlm.get_input_embeddings()

        # Disable finetuning of the VLM
        for param in self.vlm.parameters():
            param.requires_grad = False

        # Create a mixture of experts (MoE) model
        expert_configs = {
            "vlm": MoeExpertConfig(
                hidden_size=self.vlm_hidden_size,  # Use backbone-specific size
                intermediate_size=vlm_expert_intermediate_size,
                head_dim=vlm_expert_head_dim,
                num_attention_heads=vlm_expert_num_heads,
                num_key_value_heads=vlm_expert_num_kv_heads,
            ),
            "action": MoeExpertConfig(
                hidden_size=action_expert_width,
                intermediate_size=action_expert_intermediate_size,
                head_dim=action_expert_head_dim,
                num_attention_heads=action_expert_num_heads,
                num_key_value_heads=action_expert_num_kv_heads,
            ),
        }
        self.moe = GemmaMoE(moe_depth, expert_configs)
        self.action_encoder = ActionEncoder(self.action_dim, action_expert_width)
        self.time_embedding = SinusoidalPosEmb(action_expert_width)
        self.proprio_encoder = nn.Linear(proprio_dim, action_expert_width)
        self.action_decoder = nn.Linear(
            action_expert_width,
            self.action_dim,
        )

        # Check if we can use pretrained weights
        try:
            gemma_config = self.vlm.config.text_config
            self.using_pretrained_vlm = (
                gemma_config.intermediate_size == vlm_expert_intermediate_size
                and gemma_config.hidden_size == self.vlm_hidden_size
            )
        except AttributeError:
            self.using_pretrained_vlm = False

        # Load pretrained weights into VLM expert
        if self.using_pretrained_vlm:
            self._load_pretrained_vlm_weights()

        # Delete the language model to save memory (keep only embeddings)
        if hasattr(self.vlm, 'model') and hasattr(self.vlm.model, 'language_model'):
            del self.vlm.model.language_model
        elif hasattr(self.vlm, 'language_model'):
            del self.vlm.language_model

        # Resize the images to appropriate size for backbone
        self.image_normalizer = torch.nn.Sequential(
            T.Resize((self.image_size, self.image_size)),
        )

        logger.info(f"Pi0 model initialized with {vlm_backbone.value}")
        logger.info(f"  Using pretrained VLM weights: {self.using_pretrained_vlm}")
        logger.info(f"  Is Gemma 3: {self.is_gemma3}")

    @classmethod
    def from_config(cls, config: Pi0Config) -> "Pi0":
        """Create Pi0 model from configuration."""
        return cls(config=config)

    @classmethod
    def for_jetson_thor(cls) -> "Pi0":
        """Create Pi0 model optimized for Jetson Thor."""
        config = Pi0Config.for_jetson_thor()
        return cls.from_config(config)

    @classmethod
    def for_jetson_orin(cls) -> "Pi0":
        """Create Pi0 model optimized for Jetson AGX Orin."""
        config = Pi0Config.for_jetson_orin()
        return cls.from_config(config)

    def _load_pretrained_vlm_weights(self) -> None:
        """Load pretrained PaliGemma weights into the VLM expert of the MoE."""
        logger.info("Loading pretrained PaliGemma weights into VLM expert...")
        vlm_state_dict = self.vlm.model.language_model.state_dict()
        moe_state_dict = self.moe.state_dict()
        new_state_dict = {}
        for moe_key, moe_param in moe_state_dict.items():
            if "experts.vlm" in moe_key:
                vlm_key = moe_key.replace("experts.vlm.", "")
                if vlm_key in vlm_state_dict:
                    new_state_dict[moe_key] = vlm_state_dict[vlm_key]
                else:
                     logger.warning(f"VLM key not found: {vlm_key}")
            else:
                new_state_dict[moe_key] = moe_param

        self.moe.load_state_dict(new_state_dict, strict=False)
        logger.info("Successfully loaded pretrained PaliGemma weights into VLM expert.")

    def _create_expert_attention_masks(
        self, batch_size: int, pad_masks: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """Create attention masks for the experts."""
        vlm_mask = pad_masks.unsqueeze(1) * pad_masks.unsqueeze(2)
        vlm_mask = torch.where(vlm_mask == 1, 0.0, torch.finfo(self.dtype).min).to(
            self.dtype
        )
        state_len = 1
        action_len = self.action_horizon

        stat_act_len = state_len + action_len
        state_action_mask = torch.zeros(
            (stat_act_len, stat_act_len), device=self.device, dtype=self.dtype
        )

        state_action_mask[0, 0] = 1
        for i in range(1, stat_act_len):
            state_action_mask[i, 0] = 1
            state_action_mask[i, 1 : i + 1] = 1

        state_action_mask = torch.where(
            state_action_mask == 1, 0.0, torch.finfo(self.dtype).min
        ).to(self.dtype)

        vlm_mask = vlm_mask.unsqueeze(1)
        state_action_mask = (
            state_action_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)
        )

        return {"vlm": vlm_mask, "action": state_action_mask}

    def _create_pi0_mix_attention_mask(
        self, batch_size: int, vlm_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Create the mixed attention mask for the Pi0 model."""
        vlm_len = vlm_seq_len if vlm_seq_len is not None else self.vlm_max_tokens
        state_len = 1
        action_len = self.action_horizon
        total_seq_len = vlm_len + state_len + action_len

        mask = torch.zeros(
            (total_seq_len, total_seq_len), device=self.device, dtype=self.dtype
        )

        mask[:vlm_len, :vlm_len] = 1
        mask[vlm_len:, :vlm_len] = 1
        mask[vlm_len : vlm_len + state_len, : vlm_len + state_len] = 1

        action_start = vlm_len + state_len
        for i in range(0, action_len):
            mask[action_start + i, : action_start + i + 1] = 1

        mask = mask.unsqueeze(0).unsqueeze(1)
        mask = mask.expand(batch_size, 1, -1, -1)
        attention_mask = torch.where(mask == 1, 0.0, torch.finfo(self.dtype).min).to(
            self.dtype
        )
        return attention_mask

    def _create_pi0_position_ids(
        self, batch_size: int, vlm_seq_len: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        """Create position IDs for the Pi0 model."""
        vlm_len = vlm_seq_len if vlm_seq_len is not None else self.vlm_max_tokens
        vlm_pos = torch.arange(1, vlm_len + 1, device=self.device).type(self.dtype)
        vlm_pos = vlm_pos.unsqueeze(0).expand(batch_size, -1)

        state_action_pos = torch.arange(
            1, 1 + self.action_horizon + 1, device=self.device
        ).type(self.dtype)
        state_action_pos = state_action_pos.unsqueeze(0).expand(batch_size, -1)

        position_ids = {"vlm": vlm_pos, "action": state_action_pos}
        return position_ids

    def _forward_vlm_merged_text_images(
        self,
        images: torch.Tensor,
        image_masks: torch.Tensor,
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for merging text and images in the VLM."""
        embs = []
        pad_masks = []

        # iterate over num_cam images
        # images shape: [B, num_cams, C, H, W]
        num_cams = images.shape[1]
        for i in range(num_cams):
            img = images[:, i]
            img_mask = image_masks[:, i]
            
            img_emb = self.vlm.model.get_image_features(img)
            img_emb = img_emb.to(dtype=self.dtype, device=self.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = (
                img_mask[:, None].expand(bsize, num_img_embs).to(device=self.device)
            )

            embs.append(img_emb)
            pad_masks.append(img_mask)

        language_embeddings = self.vlm_embedding_module(language_tokens)
        embs.append(language_embeddings)
        pad_masks.append(language_masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        return embs, pad_masks

    def _predict_action(
        self,
        merged_text_images: torch.Tensor,
        proprio_embeds: torch.Tensor,
        action: torch.Tensor,
        t: torch.Tensor,
        vlm_seq_len: Optional[int] = None,
        pad_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict action sequence from observations."""
        batch_size = proprio_embeds.size(0)
        time_cond = self.time_embedding(t)
        action_embeds = self.action_encoder(action, time_cond)
        proprio_embeds = proprio_embeds.unsqueeze(1)
        proprio_action_tokens = torch.cat([proprio_embeds, action_embeds], dim=1)
        
        proprio_action_embeds = self.moe(
            hidden_states={
                "vlm": merged_text_images,
                "action": proprio_action_tokens,
            },
            expert_attention_masks=self._create_expert_attention_masks(
                batch_size, pad_masks
            ),
            mix_attention_mask=self._create_pi0_mix_attention_mask(
                batch_size, vlm_seq_len
            ),
            position_ids=self._create_pi0_position_ids(batch_size, vlm_seq_len),
        )["action"]
        
        action_embeds = proprio_action_embeds[:, 1:]
        return self.action_decoder(action_embeds)

    def forward(
        self, 
        images: torch.Tensor, 
        image_masks: torch.Tensor,
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
        proprio_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for generating actions."""
        batch_size = images.shape[0]
        
        # Preprocess images
        # Expect images to be [B, num_cams, C, H, W]
        # Normalize to [-1, 1]
        processed_images = []
        for i in range(images.shape[1]):
            img = self.image_normalizer(images[:, i])
            img = img * 2.0 - 1.0
            processed_images.append(img)
        processed_images = torch.stack(processed_images, dim=1)

        merged_text_images, pad_masks = self._forward_vlm_merged_text_images(
            processed_images, image_masks, language_tokens, language_masks
        )
        
        proprio_embeds = self.proprio_encoder(proprio_states)

        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(
            batch_size, device=self.device, dtype=proprio_embeds.dtype
        )
        action = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=proprio_embeds.dtype,
        )
        
        actual_seq_len = merged_text_images.shape[1]

        for _ in range(self.num_inference_steps):
            action_vel = self._predict_action(
                merged_text_images, proprio_embeds, action, t, actual_seq_len, pad_masks
            )
            action += delta_t * action_vel
            t += delta_t
            
        return action

"""
Pi0: A Vision-Language-Action Flow Model for General Robot Control (AGX Orin Port).

This module implements the Pi0 model from Physical Intelligence for embodied AI.

IMPORTANT: This module requires PyTorch and Transformers to be installed.
Install with: pip install torch torchvision transformers

References:
- Physical Intelligence: https://www.physicalintelligence.company/
- PaliGemma: https://huggingface.co/google/paligemma-3b-pt-224
"""

import logging
import os
import time
from typing import Optional, List, Dict, Any

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
    from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    raise ImportError(
        "Transformers is required for Pi0 model. "
        "Install with: pip install transformers"
    )

from .modules import ActionEncoder, GemmaMoE, MoeExpertConfig, SinusoidalPosEmb

logger = logging.getLogger(__name__)

# Global tokenizer for static method
_tokenizer = None
LANGUAGE_MODEL_NAME = "google/paligemma-3b-pt-224"
VLM_BACKBONE = "google/paligemma-3b-pt-224"
VLM_EXPERT_WIDTH = 2048  # Width of the VLM expert, matches PaliGemma's hidden size


class Pi0(nn.Module):
    """Implementation of Pi0 model from the Physical Intelligence paper.
    
    Ported for AGX Orin 32GB.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
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
        dtype: torch.dtype = torch.float32,
        device: str = "cuda"
    ):
        super().__init__()
        
        # if not HAS_TRANSFORMERS:
        #     raise ImportError("Transformers library is required for Pi0")

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.vlm_max_text_tokens = vlm_max_text_tokens
        # Assuming 3 cameras for now as per AGX config
        num_rgbs = 3 
        self.vlm_max_tokens = num_rgbs * 256 + self.vlm_max_text_tokens
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        self.dtype = dtype
        self.device = device
        
        # Proprioception dimension (joints + vel + torque)
        # Assuming 7 joints -> 7 + 7 + 7 = 21
        proprio_dim = 21 

        logger.info(f"Loading VLM Backbone: {VLM_BACKBONE}")
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            VLM_BACKBONE, dtype=self.dtype, attn_implementation="eager"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            VLM_BACKBONE, padding_side="right"
        )
        self.vlm_embedding_module = self.vlm.get_input_embeddings()
        
        # Disable finetuning of the VLM
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        # Create a mixture of experts (MoE) model
        expert_configs = {
            "vlm": MoeExpertConfig(
                hidden_size=VLM_EXPERT_WIDTH,
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

        gemma_config = self.vlm.config.text_config
        self.using_pretrained_paligemma = (
            gemma_config.intermediate_size == vlm_expert_intermediate_size
            and gemma_config.hidden_size == VLM_EXPERT_WIDTH
        )
        
        # Load PaliGemma weights into VLM expert
        if self.using_pretrained_paligemma:
            self._load_pretrained_vlm_weights()
        
        # Delete the language model to save memory (keep only embeddings)
        del self.vlm.model.language_model
        
        # Resize the images to 224x224
        self.image_normalizer = torch.nn.Sequential(
            T.Resize((224, 224)),
        )

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

# MolmoAct Integration Mapping for Dynamical

## Component-by-Component Implementation Guide

**Purpose:** Map existing Dynamical components to MolmoAct features and identify specific changes needed.

---

## Quick Reference: Integration Summary

| MolmoAct Feature | Dynamical Status | Action Required | Effort |
|------------------|------------------|-----------------|--------|
| Depth VQVAE Tokenization | **Partial** - DepthAnythingV3 exists | Add VQVAE layer | Medium |
| Perception Tokens | **Missing** - uses raw features | Add tokenization | High |
| Trajectory Traces | **Missing** - direct joint actions | Add trajectory predictor | High |
| User Steerability | **Missing** | Add guidance interface | Medium |
| Action Chain-of-Thought | **Missing** - uses BC | Extend training pipeline | Medium |
| Embodiment-Agnostic Planning | **Partial** - IK exists | Refactor skill routing | Medium |

---

## 1. DEPTH TOKENIZATION

### Current State in Dynamical

**Existing Component:** DepthAnythingV3
- **File:** `src/core/depth_estimation/depth_anything_v3.py:204`
- **Class:** `DepthAnythingV3`
- **Method:** `infer(image, focal_length) → DepthResult` (line 442)
- **Output:** Continuous depth map [H, W] in meters

**Integration Point:** Unified Perception Pipeline
- **File:** `src/meta_ai/unified_perception.py:315`
- **Method:** `process_frame()`
- **Current Flow:** DINOv3 → SAM3 → V-JEPA2 → Feature Fusion
- **Gap:** Depth is NOT integrated into feature fusion

### What to Add: Depth VQVAE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DEPTH TOKENIZATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXISTING:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Camera [H, W, 3] ──▶ DepthAnythingV3.infer() ──▶ depth [H, W]      │    │
│  │                       (src/core/depth_estimation/depth_anything_v3.py)   │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  NEW - TO IMPLEMENT:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  depth [H, W] ──▶ DepthVQVAE.encode() ──▶ depth_tokens [H', W']     │    │
│  │                   (NEW: src/spatial_intelligence/depth_tokenizer.py)     │
│  │                                                                      │    │
│  │  Architecture:                                                       │    │
│  │  - Encoder: Conv2d(1→64→128→256→embedding_dim)                      │    │
│  │  - Codebook: [1024, 256] learnable embeddings                       │    │
│  │  - Decoder: TransposeConv2d for reconstruction                     │    │
│  │  - Loss: L_recon + L_vq_commitment + L_codebook_ema                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  MODIFY - unified_perception.py:                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  _fuse_features() (line 449):                                       │    │
│  │  - ADD: if self.depth_tokenizer:                                    │    │
│  │           depth_tokens = self.depth_tokenizer.encode(depth_map)     │    │
│  │           features.append(depth_tokens.flatten())                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

**Step 1: Create DepthVQVAE class**
```python
# NEW FILE: src/spatial_intelligence/depth_tokenizer.py

@dataclass
class DepthTokenizerConfig:
    codebook_size: int = 1024
    embedding_dim: int = 256
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    input_size: Tuple[int, int] = (280, 504)  # Match DepthAnythingV3 output
    output_size: Tuple[int, int] = (35, 63)   # Downsampled token grid

class DepthVQVAE(nn.Module):
    def __init__(self, config: DepthTokenizerConfig):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, config.embedding_dim, 1),
        )

        # Vector Quantization
        self.codebook = nn.Embedding(config.codebook_size, config.embedding_dim)

        # Decoder (for training)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.embedding_dim, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def encode(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Encode depth map to discrete tokens."""
        z = self.encoder(depth_map)  # [B, D, H', W']
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.config.embedding_dim)

        # Find nearest codebook entries
        distances = torch.cdist(z_flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)  # [B*H'*W']

        return indices.reshape(depth_map.shape[0], *self.config.output_size)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct depth from tokens."""
        z_q = self.codebook(tokens)  # [B, H', W', D]
        z_q = z_q.permute(0, 3, 1, 2)  # [B, D, H', W']
        return self.decoder(z_q)
```

**Step 2: Modify unified_perception.py**
```python
# MODIFY: src/meta_ai/unified_perception.py

# Line ~60: Add import
from src.spatial_intelligence.depth_tokenizer import DepthVQVAE, DepthTokenizerConfig

# Line ~110: Add config field
@dataclass
class PerceptionConfig:
    # ... existing fields ...
    enable_depth_tokenization: bool = True  # NEW
    depth_tokenizer_config: DepthTokenizerConfig = field(
        default_factory=DepthTokenizerConfig
    )

# Line ~212: Add to __init__
class UnifiedPerceptionPipeline:
    def __init__(self, config):
        # ... existing init ...
        self.depth_tokenizer: Optional[DepthVQVAE] = None  # NEW
        self.depth_estimator: Optional[DepthAnythingV3] = None  # NEW

# Line ~290: Add to initialize()
def initialize(self):
    # ... existing initialization ...

    # NEW: Initialize depth tokenizer
    if self.config.enable_depth_tokenization:
        logger.info("Loading Depth Tokenizer...")
        self.depth_estimator = DepthAnythingV3()
        self.depth_tokenizer = DepthVQVAE(self.config.depth_tokenizer_config)
        self.depth_tokenizer.load_state_dict(
            torch.load("checkpoints/depth_vqvae.pth")
        )

# Line ~370: Add depth processing in process_frame()
def process_frame(self, frame, ...):
    # ... existing processing ...

    # NEW: Depth tokenization
    if self.depth_tokenizer and self.depth_estimator:
        depth_result = self.depth_estimator.infer(frame)
        depth_map = torch.from_numpy(depth_result.depth_map).unsqueeze(0).unsqueeze(0)
        result.depth_tokens = self.depth_tokenizer.encode(depth_map)

# Line ~449: Modify _fuse_features()
def _fuse_features(self, result):
    features = []

    # ... existing feature collection ...

    # NEW: Add depth tokens
    if result.depth_tokens is not None:
        depth_flat = result.depth_tokens.flatten().numpy()
        features.append(depth_flat)

    # ... rest of fusion ...
```

**Step 3: Add PerceptionResult field**
```python
# MODIFY: src/meta_ai/unified_perception.py:151

@dataclass
class PerceptionResult:
    # ... existing fields ...

    # NEW: Depth tokenization
    depth_tokens: Optional[np.ndarray] = None  # [H', W'] discrete tokens
    depth_token_embeddings: Optional[np.ndarray] = None  # [H', W', D] embeddings
```

---

## 2. TRAJECTORY TRACES

### Current State in Dynamical

**VLA Integration (Pi0.5)**
- **File:** `src/spatial_intelligence/pi0/pi05_model.py:213`
- **Method:** `infer(observation) → Pi05Result`
- **Output:** Joint actions [16, 7] directly

**Deep Imitative Learning**
- **File:** `src/spatial_intelligence/deep_imitative_learning.py:296`
- **Method:** `execute(instruction, images, proprio)`
- **Flow:** Pi0.5 → Diffusion → RIP → Actions
- **Gap:** No intermediate trajectory representation

**Diffusion Planner**
- **File:** `src/spatial_intelligence/planning/diffusion_planner.py:313`
- **Method:** `plan(initial_actions, condition)`
- **Works in:** Action space (joint positions), NOT image space

### What to Add: Trajectory Trace Predictor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY TRACE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT FLOW (to preserve):                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Perception ──▶ Pi0.5 VLA ──▶ [16, 7] actions ──▶ Diffusion ──▶ ...│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  NEW PARALLEL PATH (MolmoAct-style):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Perception Tokens ──┐                                              │    │
│  │  (DINOv3 + Depth)    │                                              │    │
│  │                      ▼                                              │    │
│  │  ┌───────────────────────────────────────────────────────────┐     │    │
│  │  │  TrajectoryPredictor (NEW)                                 │     │    │
│  │  │  - Input: perception_tokens [N, D], instruction_emb [512] │     │    │
│  │  │  - Architecture: Cross-attention + Transformer decoder    │     │    │
│  │  │  - Output: waypoints [max_wp, 2] in image coords [0, 256) │     │    │
│  │  └───────────────────────────────────────────────────────────┘     │    │
│  │                      │                                              │    │
│  │                      ▼                                              │    │
│  │  ┌───────────────────────────────────────────────────────────┐     │    │
│  │  │  TrajectoryTrace (data structure)                          │     │    │
│  │  │  - waypoints: np.ndarray [N, 2]                            │     │    │
│  │  │  - confidences: np.ndarray [N]                             │     │    │
│  │  │  - waypoint_depths: np.ndarray [N]                         │     │    │
│  │  │  - visualize(): overlay on source image                   │     │    │
│  │  └───────────────────────────────────────────────────────────┘     │    │
│  │                      │                                              │    │
│  │                      ▼                                              │    │
│  │  ┌───────────────────────────────────────────────────────────┐     │    │
│  │  │  ActionDecoder (per-robot)                                 │     │    │
│  │  │  - Input: trace + camera_intrinsics + robot_config        │     │    │
│  │  │  - Projects image coords → 3D → IK → joint actions        │     │    │
│  │  │  - Enables embodiment-agnostic planning                   │     │    │
│  │  └───────────────────────────────────────────────────────────┘     │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  INTEGRATION POINT (deep_imitative_learning.py:296-397):                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  def execute(...):                                                   │    │
│  │      # Existing VLA path                                            │    │
│  │      vla_actions = self._vla_inference(...)                         │    │
│  │                                                                      │    │
│  │      # NEW: Trajectory trace path (parallel)                        │    │
│  │      if self.trajectory_predictor:                                  │    │
│  │          trace = self.trajectory_predictor.predict(                 │    │
│  │              perception_tokens=scene_features,                      │    │
│  │              instruction=instruction,                               │    │
│  │          )                                                          │    │
│  │          # Convert trace → actions for this robot                   │    │
│  │          trace_actions = self.action_decoder.decode(trace, proprio) │    │
│  │          # Blend or select based on confidence                      │    │
│  │          final_actions = self._blend_actions(vla_actions, trace_actions)│ │
│  │                                                                      │    │
│  │      # Rest of pipeline (diffusion, safety, etc.)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

**Step 1: Create TrajectoryTrace data structure**
```python
# NEW FILE: src/spatial_intelligence/trajectory_trace.py

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class TrajectoryTrace:
    """Image-space trajectory representation (MolmoAct-style)."""

    # Waypoints in image coordinates [N, 2] values in [0, 256)
    waypoints: np.ndarray

    # Confidence per waypoint [N]
    confidences: np.ndarray

    # Depth at each waypoint [N] (from depth tokenizer)
    waypoint_depths: Optional[np.ndarray] = None

    # Source image for visualization
    source_image: Optional[np.ndarray] = None

    # Metadata
    instruction: Optional[str] = None
    timestamp: float = 0.0

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    def visualize(self, scale_to: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Overlay trajectory on source image."""
        if self.source_image is None:
            # Create blank canvas
            vis = np.ones((scale_to[1], scale_to[0], 3), dtype=np.uint8) * 255
        else:
            vis = cv2.resize(self.source_image.copy(), scale_to)

        # Scale waypoints from [0, 256) to image size
        scale_x = scale_to[0] / 256.0
        scale_y = scale_to[1] / 256.0
        scaled_wp = self.waypoints.copy()
        scaled_wp[:, 0] *= scale_x
        scaled_wp[:, 1] *= scale_y

        # Draw trajectory path
        for i in range(len(scaled_wp) - 1):
            pt1 = tuple(scaled_wp[i].astype(int))
            pt2 = tuple(scaled_wp[i + 1].astype(int))
            # Color intensity based on confidence
            intensity = int(255 * self.confidences[i])
            cv2.line(vis, pt1, pt2, (0, intensity, 0), 2)

        # Draw waypoints as circles
        for i, (wp, conf) in enumerate(zip(scaled_wp, self.confidences)):
            color = (0, int(255 * conf), 0)
            radius = 5 + int(conf * 5)  # Larger = more confident
            cv2.circle(vis, tuple(wp.astype(int)), radius, color, -1)
            # Number the waypoints
            cv2.putText(vis, str(i), tuple(wp.astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis

    def to_3d(
        self,
        camera_intrinsics: np.ndarray,  # [3, 3]
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert image waypoints to 3D coordinates using camera model."""
        if self.waypoint_depths is None and depth_map is None:
            raise ValueError("Need depth information for 3D conversion")

        depths = self.waypoint_depths if self.waypoint_depths is not None else \
                 np.array([depth_map[int(wp[1]), int(wp[0])] for wp in self.waypoints])

        # Unproject: pixel → camera frame
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        points_3d = np.zeros((len(self.waypoints), 3))
        for i, (wp, d) in enumerate(zip(self.waypoints, depths)):
            # Scale waypoint from [0, 256) to actual image coords
            u, v = wp[0] * (cx * 2 / 256), wp[1] * (cy * 2 / 256)
            points_3d[i] = [
                (u - cx) * d / fx,
                (v - cy) * d / fy,
                d
            ]

        return points_3d
```

**Step 2: Create TrajectoryPredictor**
```python
# NEW FILE: src/spatial_intelligence/trajectory_predictor.py

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrajectoryPredictorConfig:
    perception_dim: int = 1024  # DINOv3 output
    instruction_dim: int = 512  # Language embedding
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_waypoints: int = 20
    output_dim: int = 4  # (x, y, confidence, is_terminal)

class TrajectoryPredictor(nn.Module):
    """Predict image-space waypoints from perception tokens."""

    def __init__(self, config: TrajectoryPredictorConfig):
        super().__init__()
        self.config = config

        # Project perception to hidden dim
        self.perception_proj = nn.Linear(config.perception_dim, config.hidden_dim)

        # Project instruction to hidden dim
        self.instruction_proj = nn.Linear(config.instruction_dim, config.hidden_dim)

        # Cross-attention: waypoints attend to perception + instruction
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, batch_first=True
        )

        # Transformer decoder for autoregressive waypoint prediction
        decoder_layer = nn.TransformerDecoderLayer(
            config.hidden_dim, config.num_heads, dim_feedforward=config.hidden_dim * 4
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )

        # Waypoint output head
        self.waypoint_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Learned start token
        self.start_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

    def forward(
        self,
        perception_tokens: torch.Tensor,  # [B, N_tokens, D_perception]
        instruction_embedding: torch.Tensor,  # [B, D_instruction]
    ) -> TrajectoryTrace:
        """Generate trajectory trace autoregressively."""
        batch_size = perception_tokens.shape[0]

        # Project inputs
        perception = self.perception_proj(perception_tokens)  # [B, N, H]
        instruction = self.instruction_proj(instruction_embedding).unsqueeze(1)  # [B, 1, H]

        # Concatenate context
        context = torch.cat([perception, instruction], dim=1)  # [B, N+1, H]

        # Initialize with start token
        waypoints = []
        confidences = []
        current_tokens = self.start_token.expand(batch_size, -1, -1)  # [B, 1, H]

        for step in range(self.config.max_waypoints):
            # Cross-attend to context
            attended, _ = self.cross_attention(
                current_tokens, context, context
            )

            # Decode
            decoded = self.transformer_decoder(attended, context)

            # Predict waypoint
            output = self.waypoint_head(decoded[:, -1, :])  # [B, 4]

            x = torch.sigmoid(output[:, 0]) * 256  # Scale to [0, 256)
            y = torch.sigmoid(output[:, 1]) * 256
            conf = torch.sigmoid(output[:, 2])
            is_terminal = torch.sigmoid(output[:, 3])

            waypoints.append(torch.stack([x, y], dim=-1))
            confidences.append(conf)

            # Check termination
            if is_terminal.mean() > 0.5:
                break

            # Update for next step
            current_tokens = torch.cat([
                current_tokens,
                self.waypoint_head.inverse(output).unsqueeze(1)  # Simplified
            ], dim=1)

        # Build TrajectoryTrace (single sample, not batched)
        return TrajectoryTrace(
            waypoints=torch.stack(waypoints, dim=1)[0].detach().cpu().numpy(),
            confidences=torch.stack(confidences, dim=1)[0].detach().cpu().numpy(),
        )
```

**Step 3: Create ActionDecoder (embodiment-specific)**
```python
# NEW FILE: src/spatial_intelligence/action_decoder.py

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .trajectory_trace import TrajectoryTrace

@dataclass
class ActionDecoderConfig:
    robot_dof: int = 7
    action_horizon: int = 16
    use_ik: bool = True

class ActionDecoder:
    """Convert trajectory traces to robot-specific actions."""

    def __init__(
        self,
        config: ActionDecoderConfig,
        robot_kinematics: Optional["RobotKinematics"] = None,
    ):
        self.config = config
        self.kinematics = robot_kinematics

    def decode(
        self,
        trace: TrajectoryTrace,
        current_state: np.ndarray,  # Current joint positions
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
    ) -> np.ndarray:
        """Convert image-space trace to joint actions."""

        # Step 1: Convert trace to 3D points in camera frame
        points_3d_camera = trace.to_3d(camera_intrinsics)

        # Step 2: Transform to world frame
        T_camera_world = np.linalg.inv(camera_extrinsics)
        points_3d_world = (T_camera_world[:3, :3] @ points_3d_camera.T +
                          T_camera_world[:3, 3:4]).T

        # Step 3: Interpolate to action horizon
        t_original = np.linspace(0, 1, len(points_3d_world))
        t_horizon = np.linspace(0, 1, self.config.action_horizon)

        target_positions = np.zeros((self.config.action_horizon, 3))
        for dim in range(3):
            target_positions[:, dim] = np.interp(
                t_horizon, t_original, points_3d_world[:, dim]
            )

        # Step 4: Convert to joint positions via IK
        if self.kinematics and self.config.use_ik:
            actions = self._solve_ik_trajectory(target_positions, current_state)
        else:
            # Fallback: use as Cartesian targets (for robots with Cartesian control)
            actions = target_positions

        return actions

    def _solve_ik_trajectory(
        self,
        target_positions: np.ndarray,
        initial_config: np.ndarray,
    ) -> np.ndarray:
        """Solve IK for each target position in the trajectory."""
        actions = np.zeros((self.config.action_horizon, self.config.robot_dof))
        current_config = initial_config.copy()

        for i, target_pos in enumerate(target_positions):
            # Solve IK for this target
            target_pose = np.eye(4)
            target_pose[:3, 3] = target_pos

            solution = self.kinematics.inverse_kinematics(
                target_pose, current_config
            )

            if solution is not None:
                actions[i] = solution
                current_config = solution
            else:
                # Use last valid config
                actions[i] = current_config

        return actions
```

**Step 4: Integrate into DeepImitativeLearning**
```python
# MODIFY: src/spatial_intelligence/deep_imitative_learning.py

# Line ~65: Add imports
from .trajectory_trace import TrajectoryTrace
from .trajectory_predictor import TrajectoryPredictor, TrajectoryPredictorConfig
from .action_decoder import ActionDecoder, ActionDecoderConfig

# Line ~120: Add to DILConfig
@dataclass
class DILConfig:
    # ... existing fields ...

    # NEW: Trajectory prediction
    use_trajectory_prediction: bool = True
    trajectory_predictor_checkpoint: str = "checkpoints/trajectory_predictor.pth"
    blend_mode: str = "confidence"  # "confidence", "average", "trajectory_only"

# Line ~165: Add to DILResult
@dataclass
class DILResult:
    # ... existing fields ...

    # NEW: Trajectory trace output
    trajectory_trace: Optional[TrajectoryTrace] = None
    trace_visualization: Optional[np.ndarray] = None

# Line ~210: Add to __init__
class DeepImitativeLearning:
    def __init__(self, config):
        # ... existing init ...

        # NEW: Trajectory prediction
        self.trajectory_predictor: Optional[TrajectoryPredictor] = None
        self.action_decoder: Optional[ActionDecoder] = None

# Line ~280: Add to load()
def load(self):
    # ... existing loading ...

    # NEW: Load trajectory predictor
    if self.config.use_trajectory_prediction:
        self.trajectory_predictor = TrajectoryPredictor(
            TrajectoryPredictorConfig()
        )
        self.trajectory_predictor.load_state_dict(
            torch.load(self.config.trajectory_predictor_checkpoint)
        )

        self.action_decoder = ActionDecoder(ActionDecoderConfig())

# Line ~320: Modify execute()
def execute(self, instruction, images, proprio, scene_features):
    # ... existing VLA inference ...
    vla_actions = self._vla_inference(instruction, images, proprio)

    # NEW: Parallel trajectory prediction
    trajectory_trace = None
    trace_actions = None

    if self.trajectory_predictor and scene_features is not None:
        # Predict trajectory
        trajectory_trace = self.trajectory_predictor(
            perception_tokens=torch.from_numpy(scene_features).unsqueeze(0),
            instruction_embedding=self._encode_instruction(instruction),
        )
        trajectory_trace.source_image = images[0]

        # Convert to robot actions
        trace_actions = self.action_decoder.decode(
            trajectory_trace,
            current_state=proprio,
            camera_intrinsics=self.camera_intrinsics,
            camera_extrinsics=self.camera_extrinsics,
        )

    # Blend VLA and trace actions
    if trace_actions is not None:
        if self.config.blend_mode == "trajectory_only":
            final_actions = trace_actions
        elif self.config.blend_mode == "average":
            final_actions = (vla_actions + trace_actions) / 2
        else:  # confidence-based
            trace_conf = trajectory_trace.confidences.mean()
            final_actions = trace_conf * trace_actions + (1 - trace_conf) * vla_actions
    else:
        final_actions = vla_actions

    # ... rest of pipeline (diffusion, safety, etc.) ...

    return DILResult(
        # ... existing fields ...
        trajectory_trace=trajectory_trace,
        trace_visualization=trajectory_trace.visualize() if trajectory_trace else None,
    )
```

---

## 3. EMBODIMENT-AGNOSTIC PLANNING

### Current State in Dynamical

**Skill Routing (MoE)**
- **File:** `src/platform/cloud/moe_skill_router.py:199`
- **Method:** `route(task_embedding, skill_embeddings)`
- **Issue:** Routes to robot-specific skills

**Skill Execution**
- **File:** `src/core/robot_skill_invoker.py:330`
- **Method:** `invoke(SkillInvocationRequest)`
- **Issue:** Skills produce joint-space actions directly

**IK Solver**
- **File:** `src/core/retargeting/` (exists but separate from skills)
- **Can leverage:** Transform task-space → joint-space

### What to Change: Decouple Planning from Embodiment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBODIMENT-AGNOSTIC SKILL SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT (Embodiment-Coupled):                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Task ──▶ MoE Router ──▶ Robot-Specific Skill ──▶ Joint Actions    │    │
│  │           (routes to UR10e-skill or Franka-skill)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  PROPOSED (Embodiment-Agnostic):                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Task ──▶ MoE Router ──▶ Universal Skill ──▶ Trajectory Trace      │    │
│  │           (routes to task-space skill)       (image-space plan)     │    │
│  │                                    │                                 │    │
│  │                                    ▼                                 │    │
│  │           ┌─────────────────────────────────────────────────┐       │    │
│  │           │  Robot Registry                                  │       │    │
│  │           │  ├─ UR10e: ActionDecoder(dof=6, urdf=...)       │       │    │
│  │           │  ├─ Franka: ActionDecoder(dof=7, urdf=...)      │       │    │
│  │           │  ├─ Humanoid: ActionDecoder(dof=23, urdf=...)   │       │    │
│  │           │  └─ Custom: ActionDecoder(dof=N, urdf=...)      │       │    │
│  │           └─────────────────────────────────────────────────┘       │    │
│  │                                    │                                 │    │
│  │                                    ▼                                 │    │
│  │           Robot-Specific Joint Actions (via IK)                     │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

**Step 1: Create Robot Registry**
```python
# NEW FILE: src/core/robot_registry.py

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

@dataclass
class RobotConfig:
    robot_id: str
    name: str
    dof: int
    urdf_path: str
    ee_link: str
    base_link: str

    # Limits
    joint_limits_lower: List[float]
    joint_limits_upper: List[float]
    velocity_limits: List[float]

    # Camera configuration (for trace → action conversion)
    camera_intrinsics: np.ndarray  # [3, 3]
    camera_extrinsics: np.ndarray  # [4, 4] T_camera_base

class RobotRegistry:
    """Central registry of supported robots."""

    _robots: Dict[str, RobotConfig] = {}

    @classmethod
    def register(cls, config: RobotConfig):
        cls._robots[config.robot_id] = config

    @classmethod
    def get(cls, robot_id: str) -> Optional[RobotConfig]:
        return cls._robots.get(robot_id)

    @classmethod
    def list_robots(cls) -> List[str]:
        return list(cls._robots.keys())

# Pre-register common robots
RobotRegistry.register(RobotConfig(
    robot_id="ur10e",
    name="Universal Robots UR10e",
    dof=6,
    urdf_path="urdfs/ur10e.urdf",
    ee_link="tool0",
    base_link="base_link",
    joint_limits_lower=[-2*np.pi]*6,
    joint_limits_upper=[2*np.pi]*6,
    velocity_limits=[2.0]*6,
    camera_intrinsics=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
    camera_extrinsics=np.eye(4),
))

RobotRegistry.register(RobotConfig(
    robot_id="humanoid_23dof",
    name="Humanoid 23-DOF",
    dof=23,
    urdf_path="urdfs/humanoid.urdf",
    ee_link="right_hand_link",
    base_link="torso_link",
    joint_limits_lower=[-np.pi]*23,
    joint_limits_upper=[np.pi]*23,
    velocity_limits=[1.5]*23,
    camera_intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
    camera_extrinsics=np.eye(4),
))
```

**Step 2: Modify Skill Invocation**
```python
# MODIFY: src/core/robot_skill_invoker.py

# Line ~250: Add to SkillInvocationRequest
@dataclass
class SkillInvocationRequest:
    # ... existing fields ...

    # NEW: Trajectory-based execution
    trajectory_trace: Optional[TrajectoryTrace] = None
    use_embodiment_agnostic: bool = False

# Line ~330: Modify invoke()
def invoke(self, request: SkillInvocationRequest) -> SkillInvocationResult:
    # ... existing safety checks ...

    # NEW: Embodiment-agnostic path
    if request.use_embodiment_agnostic and request.trajectory_trace:
        robot_config = RobotRegistry.get(request.robot_id)
        if robot_config is None:
            raise ValueError(f"Unknown robot: {request.robot_id}")

        # Create action decoder for this robot
        decoder = ActionDecoder(
            config=ActionDecoderConfig(robot_dof=robot_config.dof),
            robot_kinematics=self._get_kinematics(robot_config),
        )

        # Convert trace to robot-specific actions
        actions = decoder.decode(
            request.trajectory_trace,
            request.observation.joint_positions,
            robot_config.camera_intrinsics,
            robot_config.camera_extrinsics,
        )

        return SkillInvocationResult(
            success=True,
            actions=[RobotAction(
                action_space=ActionSpace.JOINT_POSITION,
                joint_positions=actions[i],
                duration_s=0.1,
            ) for i in range(len(actions))],
            skill_ids_used=["trajectory_trace"],
        )

    # ... existing skill-based execution ...
```

---

## 4. USER STEERABILITY

### Current State in Dynamical

**No existing steerability interface.** Robot behavior is entirely model-driven.

### What to Add: User Guidance Interface

```python
# NEW FILE: src/spatial_intelligence/steerability.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class UserGuidance:
    """User-provided guidance for trajectory planning."""

    # User-drawn waypoints [N, 2] in image coords
    forced_waypoints: Optional[List[Tuple[int, int]]] = None

    # Regions to avoid (polygons in image coords)
    avoid_regions: Optional[List[np.ndarray]] = None

    # Preferred path direction
    path_bias: Optional[str] = None  # "left", "right", "above", "below"

    # Speed modifier (0.5 = half speed, 2.0 = double)
    speed_modifier: float = 1.0

    # Timestamp of guidance
    timestamp: float = 0.0

    def apply_to_trace(self, trace: TrajectoryTrace) -> TrajectoryTrace:
        """Apply user guidance constraints to a trajectory trace."""
        new_waypoints = trace.waypoints.copy()
        new_confidences = trace.confidences.copy()

        # Insert forced waypoints
        if self.forced_waypoints:
            forced = np.array(self.forced_waypoints)
            # Interpolate original waypoints to pass through forced ones
            new_waypoints = self._interpolate_through(new_waypoints, forced)
            new_confidences = np.ones(len(new_waypoints))  # High confidence for forced

        # Avoid regions
        if self.avoid_regions:
            new_waypoints = self._reroute_around(new_waypoints, self.avoid_regions)

        return TrajectoryTrace(
            waypoints=new_waypoints,
            confidences=new_confidences,
            source_image=trace.source_image,
        )

    def _interpolate_through(
        self,
        original: np.ndarray,
        forced: np.ndarray,
    ) -> np.ndarray:
        """Interpolate trajectory to pass through forced waypoints."""
        # Simple insertion - more sophisticated methods possible
        combined = np.vstack([original, forced])
        # Sort by x-coordinate (assuming left-to-right motion)
        sorted_idx = np.argsort(combined[:, 0])
        return combined[sorted_idx]

    def _reroute_around(
        self,
        waypoints: np.ndarray,
        avoid_regions: List[np.ndarray],
    ) -> np.ndarray:
        """Reroute trajectory to avoid specified regions."""
        for region in avoid_regions:
            # Check which waypoints are inside the region
            for i, wp in enumerate(waypoints):
                if self._point_in_polygon(wp, region):
                    # Move waypoint to nearest edge of region
                    waypoints[i] = self._nearest_edge_point(wp, region)
        return waypoints

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting."""
        # Implementation omitted for brevity
        pass
```

**Integration into DIL:**
```python
# MODIFY: src/spatial_intelligence/deep_imitative_learning.py

# Line ~300: Add guidance parameter
def execute(
    self,
    instruction: str,
    images: Any,
    proprio: Optional[Any] = None,
    scene_features: Optional[Any] = None,
    user_guidance: Optional[UserGuidance] = None,  # NEW
) -> DILResult:
    # ... existing code ...

    # NEW: Apply user guidance
    if trajectory_trace and user_guidance:
        trajectory_trace = user_guidance.apply_to_trace(trajectory_trace)

    # ... rest of pipeline ...
```

---

## 5. ACTION CHAIN-OF-THOUGHT

### Current State in Dynamical

**Training Data Format:**
- **File:** `src/training/compositional/compositional_trainer.py:50`
- **Class:** `DemoCapture` - stores frames, actions, proprio
- **Gap:** No reasoning annotations

**Skill Discovery:**
- **File:** `src/training/compositional/skill_discovery.py`
- **Segments demos into primitives based on visual/temporal cues
- **Gap:** No semantic labels or reasoning

### What to Add: Reasoning Annotations

```python
# MODIFY: src/training/compositional/compositional_trainer.py

# Line ~50: Extend DemoCapture
@dataclass
class DemoCapture:
    # ... existing fields ...

    # NEW: Reasoning annotations
    task_reasoning: Optional[str] = None  # Why this task?
    subtask_labels: Optional[List[str]] = None  # Per-segment labels
    object_interactions: Optional[List[Dict]] = None  # Per-frame object info

    # NEW: Chain-of-thought structure
    perception_reasoning: Optional[str] = None  # "I see a red cup on left"
    spatial_reasoning: Optional[str] = None  # "Cup is 0.3m away"
    action_reasoning: Optional[str] = None  # "Move gripper to grasp"

# Line ~100: Extend DemoSegment
@dataclass
class DemoSegment:
    # ... existing fields ...

    # NEW: Semantic annotations
    skill_name: Optional[str] = None  # "grasp", "place", etc.
    preconditions: Optional[Dict] = None  # Required state before
    postconditions: Optional[Dict] = None  # Expected state after
    reasoning: Optional[str] = None  # Why this action sequence?

# NEW: Training with reasoning loss
class ReasoningAugmentedTrainer:
    """Train with chain-of-thought reasoning supervision."""

    def compute_loss(
        self,
        predicted_actions: torch.Tensor,
        expert_actions: torch.Tensor,
        predicted_trace: TrajectoryTrace,
        expert_trace: TrajectoryTrace,
        predicted_reasoning: str,
        expert_reasoning: str,
    ) -> torch.Tensor:
        # Action loss (existing BC)
        L_action = F.mse_loss(predicted_actions, expert_actions)

        # NEW: Trajectory trace loss
        L_trace = F.mse_loss(
            torch.from_numpy(predicted_trace.waypoints),
            torch.from_numpy(expert_trace.waypoints),
        )

        # NEW: Reasoning similarity loss
        pred_emb = self.text_encoder(predicted_reasoning)
        expert_emb = self.text_encoder(expert_reasoning)
        L_reasoning = 1 - F.cosine_similarity(pred_emb, expert_emb, dim=-1)

        return L_action + 0.5 * L_trace + 0.1 * L_reasoning
```

---

## 6. COMPLETE FILE CHANGES SUMMARY

### New Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `src/spatial_intelligence/depth_tokenizer.py` | DepthVQVAE implementation | P1 |
| `src/spatial_intelligence/trajectory_trace.py` | TrajectoryTrace data structure | P0 |
| `src/spatial_intelligence/trajectory_predictor.py` | Image-space waypoint prediction | P0 |
| `src/spatial_intelligence/action_decoder.py` | Trace → robot actions | P0 |
| `src/spatial_intelligence/steerability.py` | User guidance interface | P1 |
| `src/core/robot_registry.py` | Multi-robot configuration | P1 |

### Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/meta_ai/unified_perception.py` | Add depth tokenization to fusion | P1 |
| `src/spatial_intelligence/deep_imitative_learning.py` | Add trajectory prediction path | P0 |
| `src/core/robot_skill_invoker.py` | Add embodiment-agnostic execution | P1 |
| `src/training/compositional/compositional_trainer.py` | Add reasoning annotations | P2 |
| `src/platform/cloud/moe_skill_router.py` | Route to universal skills | P1 |

### Training Data to Collect

| Data | Source | Usage |
|------|--------|-------|
| Depth VQVAE training | Existing depth maps | Train tokenizer |
| Trajectory annotations | Convert joint actions → image traces | Train predictor |
| Reasoning annotations | Manual + LLM-assisted | Train CoT |

---

## 7. TESTING STRATEGY

### Unit Tests
```python
# tests/test_trajectory_trace.py
def test_trajectory_visualization():
    trace = TrajectoryTrace(
        waypoints=np.array([[10, 20], [50, 60], [100, 120]]),
        confidences=np.array([0.9, 0.8, 0.95]),
    )
    vis = trace.visualize()
    assert vis.shape == (480, 640, 3)

def test_trajectory_to_3d():
    trace = TrajectoryTrace(
        waypoints=np.array([[128, 128]]),
        confidences=np.array([1.0]),
        waypoint_depths=np.array([0.5]),
    )
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    points_3d = trace.to_3d(K)
    assert points_3d.shape == (1, 3)
```

### Integration Tests
```python
# tests/test_dil_with_trajectory.py
def test_execute_with_trajectory_prediction():
    dil = DeepImitativeLearning(DILConfig(use_trajectory_prediction=True))
    dil.load()

    result = dil.execute(
        instruction="pick up the cup",
        images=np.random.rand(1, 480, 640, 3),
    )

    assert result.trajectory_trace is not None
    assert result.trace_visualization is not None
```

---

*Document generated for MolmoAct integration into Dynamical Edge Platform v0.9.0*

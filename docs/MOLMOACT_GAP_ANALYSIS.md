# MolmoAct Integration Gap Analysis for Dynamical

## Dynamical Edge Platform v0.9.0

**Analysis Date:** December 2025
**Source:** MolmoAct: Action Reasoning Models (AI2, August 2025)
**Paper:** arXiv:2508.07917
**GitHub:** https://github.com/allenai/molmoact

---

## Executive Summary

MolmoAct introduces **Action Reasoning Models (ARMs)** - a new paradigm that structures robot control into three explicit stages: perception tokenization, spatial planning via trajectory traces, and action decoding. This contrasts with Dynamical's current approach which directly maps perception to actions using Pi0.5 VLA.

**Key Finding:** Dynamical lacks explicit spatial reasoning between perception and action, resulting in:
- Opaque decision-making (no interpretability)
- No user steerability (cannot guide robot behavior)
- Embodiment-coupled skills (limited cross-robot transfer)
- No depth-aware reasoning tokens (3D understanding is implicit)

| Gap Category | Current Dynamical | MolmoAct Approach | Priority |
|--------------|------------------|-------------------|----------|
| Spatial Reasoning | Implicit (black-box) | Explicit 3-stage pipeline | **P0** |
| Depth Tokenization | Raw depth features | VQVAE depth tokens | **P1** |
| Trajectory Traces | None | Image-space waypoints | **P0** |
| User Steerability | None | Editable trajectories | **P1** |
| Action Chain-of-Thought | None | Reasoning annotations | **P2** |
| Embodiment Agnosticism | Robot-specific | Pixel-space planning | **P1** |

---

## 1. The Action Reasoning Model (ARM) Paradigm

### What is an ARM?

MolmoAct defines ARMs as models that integrate perception, planning, and control through a **structured three-stage pipeline** rather than direct end-to-end mapping:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MOLMOACT ARM PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: PERCEPTION TOKENS                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Image + Depth ──▶ VQVAE ──▶ Spatially-Grounded Perception Tokens   │    │
│  │                              (encode geometric structure + depth)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  STAGE 2: SPATIAL PLANNING (Image-Space Waypoints)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Perception Tokens + Instruction ──▶ 2D Waypoint Sequence           │    │
│  │                                      [x,y] in [0,256) image coords   │    │
│  │                                      (interpretable, editable)       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  STAGE 3: ACTION DECODING                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Waypoints + Robot Config ──▶ Low-Level Motor Commands               │    │
│  │                               (denormalized for specific robot)      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Current Dynamical Architecture (For Comparison)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMICAL CURRENT PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Camera + Instruction ──▶ Pi0.5 VLA ──▶ Joint Actions [H=16, A=7]   │    │
│  │                           (black-box)     (direct output)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Diffusion Planner ──▶ Smooth Trajectory                             │    │
│  │  (refinement only, no spatial reasoning)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RIP Gating ──▶ Safety Check ──▶ CBF Filter ──▶ Motor Commands       │    │
│  │  (uncertainty)    (is OOD?)      (hard constraints)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Difference:** Dynamical jumps directly from perception to joint-space actions without explicit spatial reasoning. The "what" and "where" are entangled in the VLA output.

---

## 2. Gap Analysis: Detailed Breakdown

### Gap #1: No Depth-Aware Perception Tokens (P1)

**Current State:**
- Dynamical uses DINOv3, SAM3, V-JEPA2 for perception
- Features are concatenated/fused as continuous vectors
- Depth is used by V-JEPA2 but not tokenized for language model integration

**MolmoAct Approach:**
- Uses VQVAE to compress depth information into discrete tokens
- Depth tokens integrate with the language model backbone (7B params)
- Enables explicit 3D spatial reasoning within the LLM

**Location in Dynamical:**
```
src/meta_ai/unified_perception.py:449-490
└── _fuse_features() - concatenates features without depth tokenization
```

**Why This Matters:**
- Tokenized depth allows the model to *reason* about distances
- "Object A is 0.5m away, object B is 1.2m away, therefore pick A first"
- Current approach: depth is implicit in the VLA, not explicit reasoning

**Proposed Implementation:**
```python
# NEW: src/spatial_intelligence/depth_tokenizer.py

class DepthVQVAE:
    """VQVAE for depth tokenization (inspired by MolmoAct)."""

    def __init__(self, codebook_size: int = 1024, embedding_dim: int = 256):
        self.encoder = DepthEncoder()  # Conv layers
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.decoder = DepthDecoder()  # For reconstruction loss

    def encode(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Encode depth to discrete tokens."""
        z = self.encoder(depth_map)  # [B, D, H', W']
        # Find nearest codebook entries
        indices = self._quantize(z)   # [B, H', W']
        return indices  # Discrete tokens for LLM

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct depth from tokens (for training)."""
        z = self.codebook(tokens)
        return self.decoder(z)
```

---

### Gap #2: No Trajectory Trace Visualization (P0)

**Current State:**
- Pi0.5 outputs joint-space actions directly
- Actions are opaque - no visualization of planned motion
- Diffusion planner operates in action space, not image space

**MolmoAct Approach:**
- Outputs 2D waypoints overlaid on input images
- Trajectory traces are visible and interpretable
- Coordinates bounded in [0, 256) integer space

**Location in Dynamical:**
```
src/spatial_intelligence/pi0/pi05_model.py:213-255
└── infer() - returns actions directly, no intermediate representation

src/spatial_intelligence/deep_imitative_learning.py:296-397
└── execute() - no visualization of planned trajectory
```

**Why This Matters:**
- **Debugging:** Engineers can see what the robot "intends" to do
- **Trust:** Operators can verify plans before execution
- **Failure Analysis:** When things go wrong, traces show where reasoning failed
- **Cross-robot transfer:** Pixel-space plans transfer across embodiments

**Proposed Implementation:**
```python
# NEW: src/spatial_intelligence/trajectory_trace.py

@dataclass
class TrajectoryTrace:
    """Image-space trajectory representation (MolmoAct-style)."""

    # Waypoints in image coordinates [N, 2] where values in [0, 256)
    waypoints: np.ndarray

    # Confidence per waypoint
    confidences: np.ndarray

    # Source image for overlay
    source_image: Optional[np.ndarray] = None

    # Depth at each waypoint (from depth tokenizer)
    waypoint_depths: Optional[np.ndarray] = None

    def visualize(self) -> np.ndarray:
        """Overlay trajectory on source image."""
        if self.source_image is None:
            raise ValueError("No source image for visualization")

        vis = self.source_image.copy()

        # Draw trajectory path
        for i in range(len(self.waypoints) - 1):
            pt1 = tuple(self.waypoints[i].astype(int))
            pt2 = tuple(self.waypoints[i + 1].astype(int))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        # Draw waypoints
        for i, wp in enumerate(self.waypoints):
            color = (0, int(255 * self.confidences[i]), 0)
            cv2.circle(vis, tuple(wp.astype(int)), 5, color, -1)

        return vis

    def to_robot_actions(
        self,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        robot_kinematics: Any,
    ) -> np.ndarray:
        """Convert image-space waypoints to robot actions."""
        # This decouples planning from embodiment!
        pass


class TrajectoryPredictor(nn.Module):
    """Predict image-space trajectory from perception tokens."""

    def forward(
        self,
        perception_tokens: torch.Tensor,
        instruction_embedding: torch.Tensor,
    ) -> TrajectoryTrace:
        """Generate trajectory trace from tokens + instruction."""
        # Autoregressive waypoint prediction
        waypoints = []

        for step in range(self.max_waypoints):
            next_wp = self.predict_next_waypoint(...)
            waypoints.append(next_wp)

            if self.is_terminal(next_wp):
                break

        return TrajectoryTrace(waypoints=np.stack(waypoints))
```

---

### Gap #3: No User Steerability (P1)

**Current State:**
- Robot behavior is entirely model-driven
- No mechanism for operator intervention during planning
- Only safety stops (CBF, RTA) can modify actions

**MolmoAct Approach:**
- Users can sketch target poses or paths on images
- Annotations are integrated in real-time
- Model respects user-provided waypoints as constraints

**Location in Dynamical:**
```
src/execution/dynamical_executor.py - no steerability interface
src/core/unified_skill_orchestrator.py - no user guidance input
```

**Why This Matters:**
- **Human-Robot Collaboration:** Operators guide ambiguous situations
- **Error Correction:** Fix mistakes without full re-planning
- **Preference Injection:** "Go around the left side of the obstacle"
- **Safety Override:** Direct the robot to safer paths

**Proposed Implementation:**
```python
# NEW: src/spatial_intelligence/steerability.py

@dataclass
class UserGuidance:
    """User-provided guidance for trajectory planning."""

    # User-drawn waypoints (from tablet/phone interface)
    forced_waypoints: Optional[List[Tuple[int, int]]] = None

    # Avoid regions (drawn polygons)
    avoid_regions: Optional[List[np.ndarray]] = None

    # Preferred path direction
    path_bias: Optional[str] = None  # "left", "right", "above", "below"

    # Speed preference
    speed_modifier: float = 1.0  # 0.5 = half speed, 2.0 = double

    # Precision requirement
    require_precision: bool = False


class SteerableTrajectoryPredictor(TrajectoryPredictor):
    """Trajectory predictor that respects user guidance."""

    def forward(
        self,
        perception_tokens: torch.Tensor,
        instruction_embedding: torch.Tensor,
        user_guidance: Optional[UserGuidance] = None,
    ) -> TrajectoryTrace:
        """Generate trajectory with user constraints."""

        if user_guidance and user_guidance.forced_waypoints:
            # Condition model on user waypoints
            guidance_tokens = self.encode_waypoints(
                user_guidance.forced_waypoints
            )
            perception_tokens = torch.cat([
                perception_tokens,
                guidance_tokens
            ], dim=-1)

        trace = super().forward(perception_tokens, instruction_embedding)

        # Post-process to respect avoid regions
        if user_guidance and user_guidance.avoid_regions:
            trace = self.reroute_around(trace, user_guidance.avoid_regions)

        return trace
```

---

### Gap #4: No Action Chain-of-Thought (P2)

**Current State:**
- Training is direct imitation (observation → action)
- No explicit reasoning traces in training data
- Model learns implicit correlations, not causal reasoning

**MolmoAct Approach:**
- Training data includes "Action Chain-of-Thought" annotations
- Model learns to reason: "I see X, therefore I should do Y because Z"
- Enables better generalization to novel situations

**Location in Dynamical:**
```
src/training/compositional/skill_discovery.py - no reasoning annotations
src/training/compositional/compositional_trainer.py - direct BC/IL
```

**Why This Matters:**
- **Generalization:** Reasoning transfers to new scenarios
- **Debugging:** Can inspect why model made a decision
- **Correction:** Can fix reasoning errors, not just action errors
- **Few-shot Learning:** Reasoning enables better few-shot adaptation

**Proposed Implementation:**
```python
# Enhanced training data format with reasoning

@dataclass
class ActionReasoningExample:
    """Training example with chain-of-thought reasoning."""

    # Standard fields
    images: List[np.ndarray]
    instruction: str
    actions: np.ndarray

    # NEW: Chain-of-thought reasoning
    perception_reasoning: str  # "I see a red cup on the left side of the table"
    spatial_reasoning: str     # "The cup is approximately 0.3m away, within reach"
    action_reasoning: str      # "I will move my gripper left and down to grasp it"

    # NEW: Intermediate representations
    depth_tokens: np.ndarray
    trajectory_trace: TrajectoryTrace


class ActionReasoningTrainer:
    """Train models with explicit reasoning supervision."""

    def compute_loss(self, batch: ActionReasoningExample) -> torch.Tensor:
        # Standard action loss
        action_loss = F.mse_loss(pred_actions, batch.actions)

        # NEW: Reasoning losses
        perception_loss = self.perception_decoder_loss(
            pred_perception_tokens, batch.perception_reasoning
        )
        trajectory_loss = F.mse_loss(
            pred_trajectory, batch.trajectory_trace.waypoints
        )

        return action_loss + 0.1 * perception_loss + 0.5 * trajectory_loss
```

---

### Gap #5: Embodiment-Coupled Skills (P1)

**Current State:**
- Skills are trained for specific robots (joint dimensions)
- MoE routes to robot-specific skill policies
- Cross-robot transfer requires retraining

**MolmoAct Approach:**
- Planning happens in image space (embodiment-agnostic)
- Trajectory traces are universal across robots
- Only final action decoding is robot-specific

**Location in Dynamical:**
```
src/platform/cloud/moe_skill_router.py - routes to robot-specific skills
src/core/robot_skill_invoker.py - assumes specific joint configuration
```

**Why This Matters:**
- **Scalability:** One model serves multiple robot types
- **Data Efficiency:** Pool data from all robots for training
- **Sim-to-Real:** Pixel-space plans transfer from simulation

**Proposed Architecture Change:**
```
CURRENT (Embodiment-Coupled):
Image → VLA → Joint Actions (Robot A: 23 DOF)
                           (Robot B: 7 DOF)
                           (Robot C: 12 DOF)

PROPOSED (Embodiment-Agnostic Planning):
Image → Perception Tokens → Trajectory Trace (universal)
                                    │
                            ┌───────┴───────┐
                            │               │
                            ▼               ▼
                    Action Decoder A   Action Decoder B
                    (23 DOF)           (7 DOF)
```

---

## 3. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

1. **Depth Tokenizer (VQVAE)**
   - Implement `DepthVQVAE` class
   - Train on Dynamical depth data
   - Integrate with unified perception pipeline

2. **Trajectory Trace Data Structure**
   - Define `TrajectoryTrace` dataclass
   - Implement visualization utilities
   - Add to `DILResult` output

### Phase 2: Trajectory Prediction (Weeks 5-8)

3. **Image-Space Trajectory Predictor**
   - Implement `TrajectoryPredictor` module
   - Train on converted action sequences (project joints to image space)
   - Replace direct VLA → action with VLA → trace → action

4. **Action Decoder (Embodiment-Specific)**
   - Implement per-robot action decoders
   - Take trajectory traces as input
   - Output robot-specific joint commands

### Phase 3: Steerability & Reasoning (Weeks 9-12)

5. **User Steerability Interface**
   - Implement `UserGuidance` protocol
   - Add tablet/phone annotation support
   - Integrate with `SteerableTrajectoryPredictor`

6. **Action Chain-of-Thought Training**
   - Annotate subset of demonstrations with reasoning
   - Implement `ActionReasoningTrainer`
   - Fine-tune with reasoning supervision

---

## 4. Technical Specifications

### Depth VQVAE Architecture
```
Input: Depth map [H, W, 1]
Encoder:
  - Conv2d(1, 64, 4, 2, 1)  + ReLU
  - Conv2d(64, 128, 4, 2, 1) + ReLU
  - Conv2d(128, 256, 4, 2, 1) + ReLU
  - Conv2d(256, embedding_dim, 1)
Codebook: [1024, 256] (1024 discrete tokens, 256-dim each)
Decoder: Symmetric transpose convolutions
Loss: Reconstruction + VQ commitment + codebook EMA
```

### Trajectory Predictor Architecture
```
Input:
  - Perception tokens [B, N_tokens, D_token]
  - Instruction embedding [B, D_instruction]
Output:
  - Waypoints [B, max_waypoints, 2] in [0, 256)
  - Confidences [B, max_waypoints]

Architecture:
  - Cross-attention: perception ← instruction
  - Transformer decoder: autoregressive waypoint prediction
  - MLP head: token → (x, y, confidence, is_terminal)
```

### Action Decoder Architecture (Per-Robot)
```
Input:
  - Trajectory trace [B, N_waypoints, 2]
  - Camera intrinsics [3, 3]
  - Current joint positions [B, DOF]
Output:
  - Joint actions [B, horizon, DOF]

Architecture:
  - MLP: trace → latent
  - LSTM: sequence modeling
  - Per-joint heads: latent → action
  - Robot-specific normalization
```

---

## 5. Expected Improvements

Based on MolmoAct's published results:

| Metric | Current Dynamical* | With ARM Integration |
|--------|-------------------|---------------------|
| SimplerEnv Zero-Shot | ~55% | **70%+** |
| OOD Generalization | Baseline | **+23%** |
| Interpretability | None | **Full trajectory viz** |
| User Steerability | None | **Real-time guidance** |
| Cross-Robot Transfer | Retrain required | **Shared planning** |

*Estimated based on Pi0.5 baseline without ARM structure

---

## 6. Risk Assessment

### Technical Risks

1. **Training Data Requirements**
   - MolmoAct uses 10k+ trajectory annotations
   - Dynamical may need to collect/annotate similar data
   - Mitigation: Start with auto-generated traces from existing demos

2. **Latency Impact**
   - Three-stage pipeline may be slower than direct VLA
   - Mitigation: Optimize trajectory predictor, cache perception tokens

3. **Integration Complexity**
   - Requires changes to multiple core modules
   - Mitigation: Phased rollout, feature flags for gradual adoption

### Operational Risks

1. **User Interface Development**
   - Steerability requires mobile/tablet app
   - Mitigation: Start with web-based annotation interface

2. **Safety During Steering**
   - User-guided trajectories may be unsafe
   - Mitigation: CBF filter remains active, validate user traces

---

## 7. References

1. **MolmoAct Paper:** Lee et al., "MolmoAct: Action Reasoning Models that can Reason in Space," arXiv:2508.07917, 2025
2. **MolmoAct Blog:** https://allenai.org/blog/molmoact
3. **MolmoAct Code:** https://github.com/allenai/molmoact
4. **Pi0.5 (Current Baseline):** https://arxiv.org/abs/2504.16054

---

## Appendix A: File Locations for Modifications

| Component | Current File | Modification Type |
|-----------|-------------|-------------------|
| Depth Tokenization | `src/meta_ai/unified_perception.py` | Add VQVAE integration |
| Trajectory Traces | `src/spatial_intelligence/deep_imitative_learning.py` | Add trace output |
| VLA Integration | `src/spatial_intelligence/pi0/pi05_model.py` | Add perception token output |
| Skill Routing | `src/platform/cloud/moe_skill_router.py` | Route by trace, not action |
| Skill Invocation | `src/core/robot_skill_invoker.py` | Accept trace input |
| Safety Filtering | `src/safety/cbf/filter.py` | Validate traces before actions |

---

## Appendix B: MolmoAct Training Configuration Reference

From the official MolmoAct repository:

```yaml
# Pre-training (100k steps)
batch_size: 512
learning_rate: 1e-5 to 2e-5
gpus: 256 (32 nodes × 8)
data: OXE + web multimodal

# Mid-training (50k steps)
batch_size: 256
learning_rate: reduced
gpus: 128 (16 nodes × 8)
data: Robot action reasoning

# Post-training (40k-80k steps)
batch_size: 128
method: LoRA (rank=32, alpha=16)
gpus: 64 (8 nodes × 8)
data: Task-specific (LIBERO, etc.)
```

---

*Document generated by gap analysis of MolmoAct vs Dynamical Edge Platform v0.9.0*

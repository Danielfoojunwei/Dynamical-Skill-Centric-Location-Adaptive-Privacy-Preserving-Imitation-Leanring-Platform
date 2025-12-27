# MolmoAct: Deep Dive Analysis & Implementation Plan

## Executive Summary

This document provides a comprehensive analysis of **MolmoAct** (Action Reasoning Models) from Allen Institute for AI and compares it with our current Dynamical Edge Platform architecture. We identify key innovations that can significantly enhance our system and propose a detailed implementation roadmap.

**Key Insight**: MolmoAct introduces a revolutionary **three-stage pipeline** (Perception → Planning → Action) with **depth-aware perception tokens** and **visual waypoint traces**. While we already have **Depth Anything V3** for metric depth estimation, MolmoAct's key innovation is using depth to create **spatial tokens** that are fed directly into the action model—something our current Pi0.5 pipeline does not do.

---

## Table of Contents

1. [MolmoAct Technical Deep Dive](#1-molmoact-technical-deep-dive)
2. [Current System Analysis](#2-current-system-analysis)
3. [Feature-by-Feature Comparison](#3-feature-by-feature-comparison)
4. [Gap Analysis](#4-gap-analysis)
5. [Key Innovations to Integrate](#5-key-innovations-to-integrate)
6. [Implementation Plan](#6-implementation-plan)
7. [Risk Assessment](#7-risk-assessment)
8. [Metrics & Success Criteria](#8-metrics--success-criteria)

---

## 1. MolmoAct Technical Deep Dive

### 1.1 What is MolmoAct?

MolmoAct is the first **Action Reasoning Model (ARM)**—a new class of robotic foundation models that explicitly integrate perception, planning, and control through a structured three-stage pipeline. Unlike traditional Vision-Language-Action (VLA) models that map directly from observation to action, MolmoAct introduces intermediate representations that enable:

- **Explainability**: Visual reasoning traces show what the robot plans to do
- **Steerability**: Users can edit waypoints to modify behavior in real-time
- **Spatial Reasoning**: Depth-aware tokens enable 3D understanding
- **Generalization**: Decoupled stages transfer better across embodiments

### 1.2 Three-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: PERCEPTION                               │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │ RGB Image   │ →  │ VQVAE Encoder    │ →  │ Depth-Aware Perception │  │
│  │ + Depth     │    │ (Pre-trained)    │    │ Tokens [P₁, P₂, ..Pₙ]  │  │
│  └─────────────┘    └──────────────────┘    └────────────────────────┘  │
│                                                        │                 │
│  Key Innovation: Tokens encode geometric structure     │                 │
│  + positional embeddings → estimate object distances   │                 │
└────────────────────────────────────────────────────────┼─────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: PLANNING                                 │
│  ┌────────────────────────┐    ┌─────────────────────────────────────┐  │
│  │ Perception Tokens      │ →  │ Image-Space Waypoint Sequence       │  │
│  │ + Language Instruction │    │ [(x₁,y₁), (x₂,y₂), ... (xₖ,yₖ)]    │  │
│  └────────────────────────┘    └─────────────────────────────────────┘  │
│                                               │                          │
│  Key Innovation: Waypoints are EMBODIMENT-INDEPENDENT                    │
│  → Same plan works for different robots                                  │
│  → Users can EDIT waypoints to steer behavior                            │
└───────────────────────────────────────────────┼──────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: ACTION DECODING                          │
│  ┌─────────────────────────┐    ┌────────────────────────────────────┐  │
│  │ Waypoints               │ →  │ Low-Level Motor Commands           │  │
│  │ + Robot Kinematics      │    │ [θ₁, θ₂, ... θₙ, gripper]          │  │
│  └─────────────────────────┘    └────────────────────────────────────┘  │
│                                                                          │
│  Key Innovation: Denormalization uses robot's kinematic config           │
│  → Enables multi-robot transfer                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Technical Innovations

#### A. Depth-Aware Perception Tokens

Unlike standard VLAs that use text tokens (which struggle with spatial reasoning), MolmoAct uses:

```
Perception Token = VQVAE_Encode(RGB + Depth) + PositionalEmbedding
```

**Benefits**:
- Encodes geometric structure, not just semantics
- Enables distance estimation between objects
- Provides grounding for 3D reasoning
- Pre-trained VQVAE ensures consistent tokenization

#### B. Visual Waypoint Traces

MolmoAct generates **image-space waypoints** as intermediate planning representations:

```python
# Example waypoint sequence for "pick up the red cup"
waypoints = [
    (320, 240),  # Current end-effector position
    (380, 220),  # Move toward cup
    (420, 200),  # Approach from above
    (420, 280),  # Descend to grasp
    (420, 200),  # Lift
]
```

**Benefits**:
- **Explainable**: Users see the planned trajectory overlaid on camera view
- **Editable**: Users can drag waypoints to modify behavior
- **Embodiment-independent**: Same waypoints work across robots
- **Debuggable**: Easy to identify where planning goes wrong

#### C. Chain-of-Thought Action Reasoning

Training data includes explicit reasoning chains:

```
Observation: [RGB image with red cup on table]
Instruction: "Pick up the red cup"

Perception: <token_1><token_2>...<token_n>
           → "Cup at (420, 280), table height 0.75m"

Plan: "Move to (380, 220) → approach (420, 200) → descend (420, 280) → grasp"

Action: [joint_angles] + [gripper_close]
```

### 1.4 Training Details

| Aspect | MolmoAct-7B |
|--------|-------------|
| **Pre-training Data** | Open-X Embodiment (curated) + Multimodal Reasoning |
| **Post-training Data** | ~12,000 robot episodes with CoT annotations |
| **Hardware** | 256× H100 GPUs (pre-train: ~1 day, fine-tune: ~2 hours on 64× H100) |
| **Model Size** | 7B parameters |
| **Inference Speed** | Real-time capable with optimization |

### 1.5 Benchmark Performance

| Benchmark | MolmoAct-7B-D | Pi-0 | OpenVLA | GR00T N1 |
|-----------|---------------|------|---------|----------|
| **SimplerEnv (Zero-shot)** | **70.5%** | 67.2% | 55.4% | 68.1% |
| **SimplerEnv (OOD)** | **72.1%** | 48.8% | 42.1% | 51.3% |
| **LIBERO (Fine-tuned)** | **86.6%** | 82.4% | 71.2% | 78.5% |
| **Real-world (Single-arm)** | +10% over π0-FAST |
| **Real-world (Bimanual)** | +22.7% over π0-FAST |

---

## 2. Current System Analysis

### 2.1 Our Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CURRENT SYSTEM: Pi0.5 + DIL                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐    ┌────────────────┐    ┌─────────────────────┐    │
│  │ RGB Images     │ →  │ Pi0.5 VLA      │ →  │ Direct Action       │    │
│  │ + Instruction  │    │ (7B params)    │    │ Prediction          │    │
│  └────────────────┘    └────────────────┘    └──────────┬──────────┘    │
│                                                         │               │
│                                                         ▼               │
│                        ┌─────────────────────────────────────────┐      │
│                        │ DIFFUSION PLANNER                        │      │
│                        │ Score-based trajectory refinement        │      │
│                        └────────────────────┬────────────────────┘      │
│                                             │                            │
│                                             ▼                            │
│                        ┌─────────────────────────────────────────┐      │
│                        │ RIP GATING                               │      │
│                        │ Epistemic uncertainty → Safe/OOD?        │      │
│                        └────────────────────┬────────────────────┘      │
│                              ┌──────────────┴──────────────┐            │
│                              │                             │            │
│                              ▼ (Safe)                      ▼ (OOD)      │
│                        ┌──────────┐              ┌────────────────┐     │
│                        │ Execute  │              │ POIR Recovery  │     │
│                        └──────────┘              └────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Current Strengths

| Strength | Description | Location |
|----------|-------------|----------|
| **Safety-First Architecture** | 1kHz deterministic safety loop with CBF/RTA | `src/robot_runtime/safety_shield.py` |
| **Privacy Preservation** | 7-stage FL pipeline with N2HE encryption | `src/moai/n2he.py`, `src/platform/cloud/federated_learning.py` |
| **Skill-Centric Design** | MoE routing, compositional verification | `src/platform/cloud/moe_skill_router.py` |
| **Multi-Tier Control** | Clear separation (1kHz→200Hz→10Hz→1Hz→async) | `src/robot_runtime/agent.py` |
| **Depth Anything V3** | TensorRT-accelerated metric depth estimation with depth-pose fusion | `src/core/depth_estimation/depth_anything_v3.py`, `src/core/depth_estimation/depth_pose_fusion.py` |
| **Diffusion Refinement** | Trajectory smoothing via score networks | `src/spatial_intelligence/planning/diffusion_planner.py` |
| **OOD Detection + Recovery** | RIP gating + POIR recovery | `src/spatial_intelligence/safety/rip_gating.py` |
| **Offline Operation** | Full functionality with cached skills | Architecture design |

### 2.3 Current Limitations

| Limitation | Impact | Root Cause |
|------------|--------|------------|
| **Depth not integrated into VLA** | Depth available but not used for action prediction | Pi0.5 takes RGB only, depth used separately for pose fusion |
| **No explicit planning stage** | Black-box action prediction | VLA end-to-end design |
| **No visual trace explanations** | Hard to debug failures | No intermediate representations |
| **No user-steerable plans** | Can't adjust behavior in real-time | No waypoint interface |
| **Limited OOD generalization** | Struggles with novel scenes | No spatial reasoning grounding in action model |
| **Single embodiment training** | Poor transfer to new robots | No embodiment decoupling |

---

## 3. Feature-by-Feature Comparison

### 3.1 Architecture Comparison

| Feature | MolmoAct | Our System | Gap |
|---------|----------|------------|-----|
| **Input Modalities** | RGB + Depth → unified tokens | RGB + Depth (separate paths) | **INTEGRATION** |
| **Vision Encoder** | VQVAE (depth-aware tokens) | DINOv3 + Depth Anything V3 (not fused for actions) | MODERATE |
| **Planning Representation** | Image-space waypoints | None (implicit) | **CRITICAL** |
| **Action Head** | Embodiment-specific decoder | Pi0.5 direct output | MODERATE |
| **Reasoning Chain** | Explicit CoT annotations | None | **HIGH** |
| **Explainability** | Visual traces | RIP uncertainty only | **HIGH** |
| **Steerability** | Waypoint editing | None | **HIGH** |

### 3.2 Capability Comparison

| Capability | MolmoAct | Our System | Notes |
|------------|----------|------------|-------|
| **Zero-shot generalization** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | MolmoAct: +15-20% on OOD |
| **Spatial reasoning** | ⭐⭐⭐⭐⭐ | ⭐⭐ | We lack depth tokens |
| **Multi-robot transfer** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Our actions are embodiment-specific |
| **Safety guarantees** | ⭐⭐ | ⭐⭐⭐⭐⭐ | We have CBF/RTA stack |
| **Privacy preservation** | ⭐ | ⭐⭐⭐⭐⭐ | We have N2HE + FL |
| **Skill composition** | ⭐⭐ | ⭐⭐⭐⭐ | We have MoE + verification |
| **Edge deployment** | ⭐⭐⭐ | ⭐⭐⭐⭐ | We're optimized for Jetson Thor |
| **Real-time performance** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 200Hz control loop |

### 3.3 Training & Data Comparison

| Aspect | MolmoAct | Our System |
|--------|----------|------------|
| **Training Data** | Open-X + curated 12k episodes | Federated from fleet |
| **CoT Annotations** | Yes (action reasoning chains) | No |
| **Data Privacy** | Open (model weights public) | Privacy-preserved (FHE) |
| **Continuous Learning** | Batch re-training | Online FL updates |

---

## 4. Gap Analysis

### 4.1 Critical Gaps (Must Address)

#### Gap 1: Depth Not Integrated into Action Model
- **Current**: RGB → Pi0.5 VLA (separate path: RGB → Depth Anything V3 → pose fusion only)
- **Problem**: Depth Anything V3 exists but is only used for pose fusion, NOT for action prediction
- **Impact**: VLA lacks 3D spatial reasoning for manipulation tasks
- **MolmoAct Solution**: VQVAE creates spatial tokens from RGBD that are fed INTO the action model
- **Our Advantage**: We already have Depth Anything V3 running at 30Hz on Jetson—just need to integrate it

#### Gap 2: No Explicit Planning Stage
- **Current**: Direct observation → action mapping (black-box)
- **Problem**: Cannot explain or modify planned behavior
- **Impact**: Hard to debug, impossible to steer
- **MolmoAct Solution**: Image-space waypoint sequence generation

#### Gap 3: No Chain-of-Thought Reasoning
- **Current**: Training data is (obs, action) pairs only
- **Problem**: Model doesn't learn to reason about actions
- **Impact**: Limited generalization to novel scenarios
- **MolmoAct Solution**: CoT-annotated training data with reasoning traces

### 4.2 High-Priority Gaps

#### Gap 4: Limited Explainability
- **Current**: Only RIP uncertainty scores
- **MolmoAct**: Full visual trace overlay + natural language explanation

#### Gap 5: No User Steering Interface
- **Current**: Accept or reject actions only
- **MolmoAct**: Drag waypoints to modify trajectory

#### Gap 6: Single Embodiment Coupling
- **Current**: Pi0.5 actions tied to training robot
- **MolmoAct**: Decoupled waypoints + per-robot decoder

### 4.3 Nice-to-Have Improvements

- VQVAE tokenization for perception (vs DINOv3)
- Autoregressive multi-stage generation
- Action reasoning dataset creation pipeline

---

## 5. Key Innovations to Integrate

Based on the gap analysis, here are the prioritized innovations from MolmoAct to integrate:

### 5.1 Priority 1: Spatial Tokenizer for VLA Integration

**What**: Create spatial tokens from existing Depth Anything V3 output and integrate with action model

**We Already Have**:
- ✅ `src/core/depth_estimation/depth_anything_v3.py` - TensorRT-accelerated metric depth
- ✅ `src/core/depth_estimation/depth_pose_fusion.py` - Depth-pose fusion for 3D reconstruction
- ✅ Point cloud generation capability
- ✅ Bilinear depth sampling at keypoint locations

**What's Missing**: Spatial tokens that feed INTO the VLA for action prediction

**Implementation Approach**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEW: Spatial Token Integration                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐                             │
│  │ DINOv3 Features  │   │ Depth Anything   │  ← ALREADY HAVE BOTH       │
│  │ (semantic)       │   │ V3 (metric depth)│                             │
│  └────────┬─────────┘   └────────┬─────────┘                             │
│           │                      │                                       │
│           └──────────┬───────────┘                                       │
│                      │                                                   │
│                      ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               NEW: SPATIAL TOKEN ENCODER                         │    │
│  │  ┌─────────────────────────────────────────────────────────────┐│    │
│  │  │ Option A: DINOv3 features + depth channel concatenation     ││    │
│  │  │ Option B: Learned fusion MLP (semantic + depth → tokens)    ││    │
│  │  │ Option C: VQVAE tokenizer (like MolmoAct) for discretization││    │
│  │  └─────────────────────────────────────────────────────────────┘│    │
│  │                                                                  │    │
│  │  Output: Spatial Perception Tokens [S₁, S₂, ... Sₙ]             │    │
│  │          Each token has: (semantic_features, x, y, depth, normal)│    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                      │                                                   │
│                      ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               MODIFIED: VLA with Spatial Conditioning            │    │
│  │  Pi0.5 or custom ARM that accepts spatial tokens as input        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Files to Create/Modify**:
- `src/spatial_intelligence/spatial_tokenizer.py` - NEW: Fuse DINOv3 + Depth → spatial tokens
- `src/spatial_intelligence/pi0/pi05_model.py` - MODIFY: Accept spatial tokens as additional input
- `src/core/depth_estimation/depth_anything_v3.py` - REUSE: Already implemented!

### 5.2 Priority 2: Visual Waypoint Planning Stage

**What**: Generate intermediate image-space waypoints before action decoding

**Implementation Approach**:
```python
@dataclass
class WaypointPlan:
    """Image-space waypoint sequence."""
    waypoints: List[Tuple[int, int]]  # [(x1,y1), (x2,y2), ...]
    confidence: List[float]            # Per-waypoint confidence
    timestamps: List[float]            # Estimated timing

    def to_overlay(self, image: np.ndarray) -> np.ndarray:
        """Render waypoints on image for visualization."""
        ...

    def edit(self, idx: int, new_pos: Tuple[int, int]):
        """User edits waypoint position."""
        ...
```

**Architecture**:
```
┌────────────────────────────────────────────────────────────────────────┐
│                    NEW: Waypoint Planning Stage                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐    ┌────────────────────────────────────────┐   │
│  │ Spatial Tokens    │ →  │ Waypoint Predictor                      │   │
│  │ + Instruction     │    │ (Transformer / Diffusion)               │   │
│  └───────────────────┘    │                                         │   │
│                           │ Output: [(x₁,y₁,t₁), (x₂,y₂,t₂), ...]  │   │
│                           └────────────────────────────────────────┘   │
│                                           │                            │
│                                           ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    USER STEERING INTERFACE                       │   │
│  │  ┌─────────────────┐                                             │   │
│  │  │ Camera View +   │  • Drag waypoints to modify                 │   │
│  │  │ Waypoint Overlay│  • Add/remove intermediate points           │   │
│  │  │ [●→●→●→●→●]     │  • Approve or request new plan              │   │
│  │  └─────────────────┘                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                           │                            │
│                                           ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ACTION DECODER                                │   │
│  │  Waypoints + Robot Kinematics → Joint Commands                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Files to Create**:
- `src/spatial_intelligence/planning/waypoint_planner.py` - Waypoint predictor
- `src/spatial_intelligence/planning/action_decoder.py` - Waypoint → action
- `src/platform/ui/components/WaypointEditor.tsx` - UI for steering

### 5.3 Priority 3: Chain-of-Thought Training Pipeline

**What**: Annotate training data with reasoning chains for better generalization

**Implementation Approach**:
```python
@dataclass
class ReasoningAnnotation:
    """Chain-of-thought annotation for training."""
    observation: np.ndarray
    instruction: str

    # NEW: Reasoning chain
    perception_description: str  # "Red cup at (320, 240), 0.5m away"
    plan_description: str        # "Approach from above, descend to grasp"
    waypoints: List[Tuple[int, int]]

    # Final action
    action: np.ndarray
```

**Data Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│              NEW: Chain-of-Thought Annotation Pipeline                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                     │
│  │ Raw Episode     │                                                     │
│  │ (obs, action)   │                                                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: Perception Annotation                                    │    │
│  │ • Run DINOv3 + SAM3 to segment objects                           │    │
│  │ • Use depth to estimate 3D positions                             │    │
│  │ • Generate: "Red cup at pixel (320,240), 0.5m from camera"       │    │
│  └────────┬────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: Waypoint Extraction                                      │    │
│  │ • Track end-effector in image space across episode               │    │
│  │ • Subsample to key waypoints (direction changes)                 │    │
│  │ • Generate: [(280,220), (320,200), (320,280)]                    │    │
│  └────────┬────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: Plan Description (Optional LLM)                          │    │
│  │ • Use VLM to describe high-level plan                            │    │
│  │ • Generate: "Move right and up, then descend to grasp"           │    │
│  └────────┬────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ Annotated Data  │                                                     │
│  │ {obs, percep,   │ → Used for ARM training                             │
│  │  waypoints,     │                                                     │
│  │  plan, action}  │                                                     │
│  └─────────────────┘                                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Files to Create**:
- `src/training/cot/perception_annotator.py` - Auto-annotate perception
- `src/training/cot/waypoint_extractor.py` - Extract waypoints from demos
- `src/training/cot/plan_annotator.py` - Optional LLM plan description
- `src/training/cot/arm_trainer.py` - Train with CoT data

### 5.4 Priority 4: Embodiment-Decoupled Action Decoder

**What**: Separate embodiment-independent planning from robot-specific actions

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│              NEW: Embodiment-Decoupled Action System                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ SHARED: Embodiment-Independent Waypoint Planner                  │    │
│  │ (Same model works for all robots)                                │    │
│  │                                                                  │    │
│  │ Input: Spatial tokens + instruction                              │    │
│  │ Output: Image-space waypoints + gripper commands                 │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│          ┌──────────────────────┼──────────────────────┐                │
│          │                      │                      │                │
│          ▼                      ▼                      ▼                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ UR5 Decoder      │  │ Franka Decoder   │  │ Custom Decoder   │       │
│  │ (6-DOF + grip)   │  │ (7-DOF + grip)   │  │ (N-DOF + grip)   │       │
│  │                  │  │                  │  │                  │       │
│  │ Waypoints →      │  │ Waypoints →      │  │ Waypoints →      │       │
│  │ IK Solution      │  │ IK Solution      │  │ IK Solution      │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                          │
│  Each decoder is a small network trained per-embodiment                  │
│  Enables rapid transfer to new robots with minimal fine-tuning           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Files to Create**:
- `src/spatial_intelligence/action_decoder/base.py` - Abstract decoder
- `src/spatial_intelligence/action_decoder/ur5_decoder.py` - UR5 specific
- `src/spatial_intelligence/action_decoder/franka_decoder.py` - Franka specific
- `src/spatial_intelligence/action_decoder/registry.py` - Decoder registry

---

## 6. Implementation Plan

### 6.1 Phase 1: Foundation (Weeks 1-4)

#### Milestone 1.1: Spatial Tokenizer & VLA Integration

**Objective**: Create spatial tokens from existing Depth Anything V3 and integrate with action model

**Already Complete** (from existing codebase):
- ✅ Depth Anything V3 TensorRT inference (`src/core/depth_estimation/depth_anything_v3.py`)
- ✅ Depth-pose fusion (`src/core/depth_estimation/depth_pose_fusion.py`)
- ✅ Point cloud generation
- ✅ DINOv3 feature extraction (`src/meta_ai/dinov3.py`)

**Tasks**:

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 1.1.1 | Create spatial tokenizer (DINOv3 + Depth Anything V3 fusion) | `src/spatial_intelligence/spatial_tokenizer.py` | 4 days |
| 1.1.2 | Add positional encoding for 3D coordinates | `src/spatial_intelligence/spatial_tokenizer.py` | 2 days |
| 1.1.3 | Create VLA adapter to accept spatial tokens | `src/spatial_intelligence/pi0/spatial_adapter.py` | 3 days |
| 1.1.4 | Integrate spatial tokenizer with DIL pipeline | `src/spatial_intelligence/deep_imitative_learning.py` | 2 days |
| 1.1.5 | Benchmark latency on Jetson Thor | `tests/benchmark_spatial_tokens.py` | 1 day |
| 1.1.6 | Unit tests | `tests/test_spatial_tokenizer.py` | 1 day |

**Deliverables**:
- Spatial tokenizer producing tokens with (semantic_features, x, y, z, normal)
- VLA adapter for conditioning action prediction on spatial tokens
- End-to-end integration with existing depth pipeline

#### Milestone 1.2: Waypoint Planner Prototype

**Objective**: Create basic waypoint prediction from spatial tokens

**Tasks**:

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 1.2.1 | Design waypoint data structures | `src/spatial_intelligence/planning/waypoints.py` | 1 day |
| 1.2.2 | Implement Transformer-based waypoint predictor | `src/spatial_intelligence/planning/waypoint_planner.py` | 5 days |
| 1.2.3 | Create waypoint-to-action decoder interface | `src/spatial_intelligence/action_decoder/base.py` | 2 days |
| 1.2.4 | Implement basic IK-based action decoder | `src/spatial_intelligence/action_decoder/ik_decoder.py` | 3 days |
| 1.2.5 | Integration with DIL pipeline | `src/spatial_intelligence/deep_imitative_learning.py` | 2 days |
| 1.2.6 | Visualization tools for waypoints | `src/visualization/waypoint_viz.py` | 2 days |

**Deliverables**:
- Waypoint prediction model (can be random weights initially)
- Basic IK-based decoder for UR5/Franka
- Visualization of predicted waypoints overlaid on camera

### 6.2 Phase 2: Training Pipeline (Weeks 5-8)

#### Milestone 2.1: CoT Data Annotation Pipeline

**Objective**: Create automatic annotation tools for training data

**Tasks**:

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 2.1.1 | Perception auto-annotator (DINOv3 + SAM3 + depth) | `src/training/cot/perception_annotator.py` | 4 days |
| 2.1.2 | Waypoint extractor from demonstration trajectories | `src/training/cot/waypoint_extractor.py` | 3 days |
| 2.1.3 | Optional VLM plan description generator | `src/training/cot/plan_annotator.py` | 2 days |
| 2.1.4 | Data validation and quality checks | `src/training/cot/data_validator.py` | 2 days |
| 2.1.5 | Dataset format and storage | `src/training/cot/dataset.py` | 2 days |
| 2.1.6 | Annotation pipeline orchestration | `src/training/cot/pipeline.py` | 2 days |

**Deliverables**:
- Automated CoT annotation for existing demonstration data
- Quality metrics for annotated data
- Compatible with federated learning privacy requirements

#### Milestone 2.2: ARM Training Infrastructure

**Objective**: Train models with three-stage pipeline

**Tasks**:

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 2.2.1 | Three-stage model architecture | `src/training/arm/model.py` | 5 days |
| 2.2.2 | Stage-wise training (perception → planning → action) | `src/training/arm/trainer.py` | 4 days |
| 2.2.3 | Loss functions for each stage | `src/training/arm/losses.py` | 2 days |
| 2.2.4 | Integration with federated learning | `src/training/arm/federated_arm.py` | 3 days |
| 2.2.5 | Checkpoint management | `src/training/arm/checkpoints.py` | 1 day |
| 2.2.6 | Training scripts and configs | `scripts/train_arm.py` | 2 days |

**Deliverables**:
- Full ARM training pipeline
- Privacy-preserving training via FL
- Trained model on available data

### 6.3 Phase 3: User Interface (Weeks 9-10)

#### Milestone 3.1: Waypoint Steering Interface

**Objective**: Allow users to view and edit waypoint plans

**Tasks**:

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 3.1.1 | Waypoint overlay React component | `src/platform/ui/components/WaypointOverlay.tsx` | 3 days |
| 3.1.2 | Drag-and-drop waypoint editing | `src/platform/ui/components/WaypointEditor.tsx` | 3 days |
| 3.1.3 | Real-time plan update feedback | `src/platform/ui/hooks/useWaypointPlanning.ts` | 2 days |
| 3.1.4 | WebSocket integration for live updates | `src/platform/api/websocket_waypoints.py` | 2 days |
| 3.1.5 | Mobile-friendly touch controls | `src/platform/ui/components/TouchWaypointEditor.tsx` | 2 days |

**Deliverables**:
- Visual waypoint overlay on camera feed
- Interactive waypoint editing
- Real-time re-planning on edit

### 6.4 Phase 4: Production Integration (Weeks 11-12)

#### Milestone 4.1: Full Pipeline Integration

**Objective**: Complete integration with safety and skill systems

**Tasks**:

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 4.1.1 | Safety validation for waypoint plans | `src/safety/waypoint_validator.py` | 3 days |
| 4.1.2 | MoE skill integration with ARM | `src/core/arm_skill_orchestrator.py` | 3 days |
| 4.1.3 | Edge deployment optimization | `scripts/optimize_arm_for_jetson.py` | 3 days |
| 4.1.4 | Performance benchmarking | `tests/benchmark_arm.py` | 2 days |
| 4.1.5 | Documentation and examples | `docs/ARM_INTEGRATION.md` | 2 days |

**Deliverables**:
- Production-ready ARM pipeline
- Safety-verified waypoint execution
- Documentation and deployment guides

### 6.5 Implementation Timeline

```
Week 1-2:  ████████ Depth Perception (M1.1: 1-4)
Week 3-4:  ████████ Waypoint Planner (M1.2: 1-6)
Week 5-6:  ████████ CoT Annotation (M2.1: 1-6)
Week 7-8:  ████████ ARM Training (M2.2: 1-6)
Week 9-10: ████████ UI Interface (M3.1: 1-5)
Week 11-12:████████ Production Integration (M4.1: 1-5)
```

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Depth estimation latency on Jetson | Medium | High | Use TensorRT optimization, accept 10Hz depth |
| Waypoint prediction accuracy | High | High | Start with simple heuristics, iterate with training |
| CoT annotation quality | Medium | Medium | Manual validation on subset, human-in-the-loop |
| Training data insufficiency | Medium | High | Leverage existing FL data, augment with simulation |
| Integration complexity | High | Medium | Modular design, feature flags for gradual rollout |

### 7.2 Resource Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU compute for training | Medium | Medium | Use AWS/GCP spot instances, start with smaller models |
| Development time overrun | High | Medium | Prioritize core features, cut Phase 3 if needed |
| Hardware requirements | Low | High | Verify Jetson Thor specs early, fallback to cloud inference |

### 7.3 Mitigation Strategies

1. **Incremental Delivery**: Each phase delivers usable features
2. **Feature Flags**: Roll out ARM alongside existing Pi0.5 path
3. **A/B Testing**: Compare ARM vs Pi0.5 on same tasks
4. **Fallback Path**: Keep Pi0.5 as backup if ARM underperforms

---

## 8. Metrics & Success Criteria

### 8.1 Technical Metrics

| Metric | Baseline (Pi0.5) | Target (ARM) | Measurement |
|--------|------------------|--------------|-------------|
| **Zero-shot task success** | ~55% | >70% | SimplerEnv benchmark |
| **OOD generalization** | ~45% | >65% | Novel objects/scenes |
| **Spatial reasoning accuracy** | N/A | >85% | Distance estimation tasks |
| **User steering efficiency** | N/A | <3 edits to fix plan | User study |
| **Explainability score** | 2/5 | 4/5 | User satisfaction survey |

### 8.2 System Metrics

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| **End-to-end latency** | <150ms | Profiling |
| **Jetson Thor memory** | <24GB peak | nvidia-smi |
| **Inference throughput** | >10 Hz | Profiling |
| **Safety response time** | <1ms (unchanged) | Real-time measurement |

### 8.3 Business Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Robot programming time reduction** | 50% | Time-to-first-task |
| **Human intervention rate** | -30% | Interventions per task |
| **Multi-robot deployment speed** | 3x faster | Time to deploy to new robot |

---

## 9. Appendix

### A. MolmoAct Resources

- **Paper**: [arXiv:2508.07917](https://arxiv.org/abs/2508.07917)
- **Blog**: [allenai.org/blog/molmoact](https://allenai.org/blog/molmoact)
- **Code**: TBD (announced as open-source)
- **Dataset**: MolmoAct post-training dataset (~12k episodes)

### B. Related Work

- **TraceVLA**: Visual trace annotations for VLAs
- **Diffuser**: Diffusion for flexible behavior synthesis
- **Pi0 / Pi0.5**: Physical Intelligence VLA models
- **OpenVLA**: Open Vision-Language-Action model
- **GR00T**: NVIDIA's generalist robot model

### C. Glossary

- **ARM**: Action Reasoning Model
- **VLA**: Vision-Language-Action model
- **CoT**: Chain-of-Thought
- **VQVAE**: Vector-Quantized Variational Autoencoder
- **OOD**: Out-of-Distribution
- **FL**: Federated Learning
- **FHE**: Fully Homomorphic Encryption

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-26 | Claude | Initial analysis and implementation plan |

---

*This document was created as part of the MolmoAct integration initiative for the Dynamical Edge Platform.*

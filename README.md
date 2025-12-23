# Dynamical Edge Platform v0.7.0

![Version](https://img.shields.io/badge/version-0.7.0-blue)
![Status](https://img.shields.io/badge/status-Production-green)
![License](https://img.shields.io/badge/license-Proprietary-red)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble/Iron-22314E)
![React](https://img.shields.io/badge/React-18.3-61DAFB)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB)

> **Privacy-Preserving Imitation Learning Platform for Humanoid Robots with ROS 2 and Meta AI Foundation Models**

The Dynamical Edge Platform is an on-device runtime and training engine for humanoid robots. It runs on **NVIDIA Jetson Thor** (Blackwell architecture) with full ROS 2 integration:

- **2070 FP4 TFLOPS** - Full utilization via INT4/FP4 quantization
- **ROS 2 Native** - Real-time nodes with Isaac ROS acceleration
- **Privacy-Preserving** - N2HE/FHE encryption for all cloud communication
- **Deployment-Grade** - Robot-resident runtime, optional cloud
- **1kHz Safety** - C++ safety node with SCHED_FIFO priority

---

## System Overview: From Glove to Robot Action

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                           │
│  │ MANUS/      │   │  ONVIF      │   │  RTMPose/   │                           │
│  │ DYGlove     │   │  Cameras    │   │  MMPose     │                           │
│  │ (21-DOF)    │   │  (RGB-D)    │   │ (17 joints) │                           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                           │
│         │                 │                 │                                   │
│         ▼                 ▼                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    META AI PERCEPTION PIPELINE                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │   DINOv3    │  │    SAM3     │  │  V-JEPA 2   │                      │   │
│  │  │  (Features) │  │(Segmentation)│  │(World Model)│                      │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │   │
│  │         └────────────────┴────────────────┘                              │   │
│  │                          ▼                                               │   │
│  │                  [Feature Fusion: 512-dim]                               │   │
│  └──────────────────────────┬──────────────────────────────────────────────┘   │
│                             │                                                   │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    N2HE ENCRYPTION (Privacy Layer)                       │   │
│  │   Plain Features [512] ──→ LWE Ciphertexts ──→ Cloud-Safe Transmission  │   │
│  └──────────────────────────┬──────────────────────────────────────────────┘   │
│                             │                                                   │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    MOAI + MoE SKILL ROUTER                               │   │
│  │   Encrypted Embedding ──→ Gating Network ──→ Top-K Skills + Weights     │   │
│  └──────────────────────────┬──────────────────────────────────────────────┘   │
│                             │                                                   │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    SKILL EXECUTION (Robot Runtime Agent)                 │   │
│  │   Blended Skills ──→ Policy Inference ──→ Joint Commands ──→ Robot      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## What's New in v0.7.0

### ROS 2 Integration
- **Full ROS 2 Humble/Iron Support**: Native ROS 2 nodes for all components
- **Isaac ROS Acceleration**: TensorRT-accelerated perception via Isaac ROS
- **Real-Time Safety Node**: C++ safety node at 1kHz with SCHED_FIFO priority 99
- **Zero-Copy Communication**: Intra-process composition for Tier 1 nodes
- **Custom Messages**: Complete msg/srv/action definitions for robot control
- **API Bridge**: Seamless integration between ROS 2 and FastAPI backend

### Robot Runtime Agent
- **Deployment-Grade Architecture**: Robot-resident runtime with optional cloud
- **Three-Tier Compute Model**: Robot CPU (1kHz) → Robot GPU (30-100Hz) → Cloud (async)
- **Cascaded Perception**: Level 1 (always) → Level 2 (on-demand) → Level 3 (rare)
- **Offline Capable**: Robot functions without cloud connectivity
- **Graceful Degradation**: CPU-only mode, offline mode, degraded perception

### Privacy-Preserving Learning
- **N2HE Encryption**: LWE-based 128-bit post-quantum security
- **MOAI FHE**: Encrypted transformer operations
- **Federated Learning**: TenSEAL CKKS for gradient aggregation
- **Encrypted Skills**: Skills stored and transmitted encrypted

---

## Input Devices

### DYGlove / MANUS Glove
**File**: `src/drivers/dyglove.py`

```python
# 21 degrees of freedom per hand
@dataclass
class DYGloveState:
    joint_angles: np.ndarray     # [21] radians
    joint_velocities: np.ndarray # [21] rad/s
    servo_positions: np.ndarray  # [5] haptic feedback
```

- **21 DOF**: 5 thumb + 16 fingers
- **WiFi 6E**: <2ms latency
- **Haptic Feedback**: 5 servo motors

### RTMPose / MMPose
**File**: `src/core/pose_inference/rtmpose_real.py`

- **17 COCO keypoints** at 30Hz
- **ONNX Runtime** inference
- **Human skeleton** for teleoperation

---

## Meta AI Perception Pipeline

### DINOv3 - Scene Understanding
**File**: `src/meta_ai/dinov3.py`

```python
# Dense features for zero-shot recognition
features = dinov3.encode(image)  # [1024] or [H', W', 1024]
```

### SAM3 - Object Segmentation
**File**: `src/meta_ai/sam3.py`

```python
# Text-prompted segmentation
result = sam3.segment_text(image, "red cup")
# Returns: mask, centroid, bounding box
```

### V-JEPA 2 - World Model
**File**: `src/meta_ai/vjepa2.py`

```python
# Predict future states and collisions
prediction = vjepa2.predict(frames)
# Returns: future_embeddings, collision_probs
```

| Model | Purpose | FP4 TFLOPS |
|-------|---------|------------|
| DINOv3 | Dense features, classification | 120.0 |
| SAM3 | Segmentation, tracking | 200.0 |
| V-JEPA 2 | World model, safety | 330.0 |

---

## Privacy-Preserving Learning (N2HE + MOAI)

### N2HE Encryption
**File**: `src/moai/n2he.py`

```python
class N2HEContext:
    """LWE-based homomorphic encryption (128-bit post-quantum)"""

    def encrypt(self, value: float) -> LWECiphertext:
        # ct = (a, <a,s> + e + Δm)
        a = random_vector(n=1024)
        b = dot(a, secret_key) + gaussian_error + scale(value)
        return LWECiphertext(a, b)

    def homomorphic_add(self, ct1, ct2) -> LWECiphertext:
        # Enc(m1) + Enc(m2) = Enc(m1 + m2)
        return LWECiphertext(ct1.a + ct2.a, ct1.b + ct2.b)
```

### MOAI Architecture
**File**: `src/moai/moai_fhe.py`

```python
class MoaiTransformerFHE:
    """Transformer operating on encrypted data"""

    def forward(self, encrypted_features):
        # All operations on encrypted data
        Q = homomorphic_matmul(encrypted_features, W_q)
        K = homomorphic_matmul(encrypted_features, W_k)
        V = homomorphic_matmul(encrypted_features, W_v)
        return homomorphic_attention(Q, K, V)
```

---

## MoE Skill System

### Skill Routing
**File**: `src/platform/cloud/moe_skill_router.py`

```python
class MoESkillRouter:
    def route(self, task_embedding) -> Tuple[List[str], List[float]]:
        # Gating network selects top-k skills
        logits = self.gating_network(task_embedding)
        weights = softmax(logits)
        top_k = topk(weights, k=3)
        return skill_ids, blend_weights
```

### Skill Execution
**File**: `src/core/robot_skill_invoker.py`

```python
# Blend multiple skills based on MoE weights
blended_action = sum(w * skill.infer(obs) for w, skill in zip(weights, skills))
```

---

## ROS 2 Architecture

### Package Structure

```
src/ros2/
├── dynamical_msgs/           # Custom messages, services, actions
│   ├── msg/
│   │   ├── RobotState.msg
│   │   ├── SafetyStatus.msg
│   │   ├── PerceptionFeatures.msg
│   │   └── Detection.msg
│   ├── srv/
│   │   ├── ExecuteSkill.srv
│   │   ├── TriggerEstop.srv
│   │   └── SetControlMode.srv
│   └── action/
│       ├── ExecuteSkillAction.action
│       └── MoveToPosition.action
│
├── dynamical_runtime/        # C++ real-time nodes
│   ├── src/safety_node.cpp   # 1kHz, SCHED_FIFO
│   └── include/
│
├── dynamical_perception/     # Python perception
│   └── scripts/perception_pipeline_node.py
│
└── dynamical_bringup/        # Launch files
    ├── launch/robot_runtime.launch.py
    └── config/robot_params.yaml
```

### Node Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROS 2 NODE GRAPH                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1 CONTAINER (C++, 1kHz, intra-process)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ SafetyNode   │ │StateEstimator│ │ ActuatorNode │            │
│  │ SCHED_FIFO   │ │              │ │              │            │
│  │ Priority 99  │ │              │ │              │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │ /robot_state                         │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PerceptionPipeline (Python, 30Hz)                        │  │
│  │ • Cascaded: L1 (always) → L2 (demand) → L3 (rare)       │  │
│  │ • Isaac ROS TensorRT acceleration                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ APIBridge (Python)                                       │  │
│  │ • ROS 2 ↔ FastAPI translation                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Launch Command

```bash
# Build ROS 2 workspace
cd src/ros2
colcon build --symlink-install

# Source workspace
source install/setup.bash

# Launch full stack
ros2 launch dynamical_bringup robot_runtime.launch.py

# With simulation
ros2 launch dynamical_bringup robot_runtime.launch.py use_sim:=true
```

---

## 4-Tier Timing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: Safety (1kHz, <1ms, CPU) - NEVER MISSED                 │
│ ├─ SafetyShield: Joint limits, collision avoidance              │
│ ├─ StateEstimator: EKF state estimation                         │
│ ├─ Watchdog: 5ms hardware timeout                               │
│ └─ E-Stop: Immediate halt capability                            │
├─────────────────────────────────────────────────────────────────┤
│ TIER 2: Control (100Hz, <10ms, GPU)                             │
│ ├─ PolicyExecutor: Skill inference (TensorRT)                   │
│ ├─ SkillBlender: Combine multiple skills                        │
│ └─ ActionSmoothing: Trajectory interpolation                    │
├─────────────────────────────────────────────────────────────────┤
│ TIER 3: Perception (30Hz, <33ms, GPU)                           │
│ ├─ CascadedPerception: L1 → L2 → L3 models                      │
│ ├─ ObjectTracking: Multi-object tracking                        │
│ └─ HumanDetection: Safety-critical                              │
├─────────────────────────────────────────────────────────────────┤
│ TIER 4: Cloud (Async, Optional)                                 │
│ ├─ FederatedLearning: Encrypted gradient updates                │
│ ├─ SkillSync: Download new skills                               │
│ └─ Telemetry: Upload performance data                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform
./install.sh
python -m src.platform.api.database init
```

### 2. Build ROS 2 Packages

```bash
cd src/ros2
colcon build --symlink-install
source install/setup.bash
```

### 3. Start the Platform

```bash
# Option A: ROS 2 launch (recommended for robots)
ros2 launch dynamical_bringup robot_runtime.launch.py

# Option B: Python standalone (development)
python -m src.platform.api.main

# Frontend
cd src/platform/ui && npm install && npm run dev
```

### 4. Access Dashboard

Open: `http://localhost:5173` (Vite dev server)

---

## Project Structure

```
├── src/
│   ├── ros2/                          # v0.7.0: ROS 2 packages
│   │   ├── dynamical_msgs/                # Messages, services, actions
│   │   ├── dynamical_runtime/             # C++ real-time nodes
│   │   ├── dynamical_perception/          # Python perception
│   │   └── dynamical_bringup/             # Launch and config
│   │
│   ├── robot_runtime/                 # v0.7.0: Deployment-grade runtime
│   │   ├── agent.py                       # Robot Runtime Agent
│   │   ├── safety_shield.py               # 1kHz safety
│   │   ├── perception_pipeline.py         # Cascaded perception
│   │   ├── policy_executor.py             # Action generation
│   │   └── ros2_bridge.py                 # ROS 2 integration
│   │
│   ├── core/                          # Core algorithms
│   │   ├── unified_skill_orchestrator.py
│   │   ├── robot_skill_invoker.py
│   │   └── pose_inference/
│   │       └── rtmpose_real.py
│   │
│   ├── meta_ai/                       # Meta AI models
│   │   ├── dinov3.py
│   │   ├── sam3.py
│   │   ├── vjepa2.py
│   │   └── unified_perception.py
│   │
│   ├── moai/                          # Privacy-preserving AI
│   │   ├── n2he.py                        # LWE encryption
│   │   └── moai_fhe.py                    # FHE transformer
│   │
│   ├── drivers/                       # Hardware interfaces
│   │   ├── dyglove.py                     # 21-DOF haptic gloves
│   │   ├── manus_sdk.py                   # MANUS integration
│   │   └── onvif_ptz.py                   # Camera control
│   │
│   ├── platform/
│   │   ├── api/                           # FastAPI backend
│   │   ├── cloud/
│   │   │   └── moe_skill_router.py        # MoE routing
│   │   └── ui/                            # React frontend
│   │
│   └── shared/crypto/                 # FHE backends
│       └── fhe_backend.py
│
├── docs/
│   └── DEPLOYMENT_ARCHITECTURE.md     # v0.7.0: Deployment spec
│
└── config/config.yaml
```

---

## Security & Privacy

| Layer | Scheme | Security |
|-------|--------|----------|
| **Feature Transmission** | N2HE (LWE) | 128-bit post-quantum |
| **Gradient Aggregation** | TenSEAL (CKKS) | Approximate arithmetic |
| **Skill Storage** | N2HE encrypted | Cloud-side privacy |
| **Communication** | TLS + FHE | End-to-end encryption |

---

## Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Compute** | NVIDIA Jetson Thor | 2070 FP4 TFLOPS |
| **Memory** | 128GB LPDDR5X | All models loaded |
| **Storage** | 500GB NVMe | Training data, skills |
| **Cameras** | ONVIF (up to 22) | Multi-view perception |
| **Gloves** | DYGlove/MANUS | 21-DOF haptic |

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **0.7.0** | Dec 2024 | **ROS 2 Integration** - Full ROS 2 Humble/Iron support, Isaac ROS acceleration, C++ safety node (1kHz), Robot Runtime Agent, deployment-grade architecture |
| 0.6.0 | Dec 2024 | Frontend Modernization - React 18.3, Zustand, 3D visualization |
| 0.5.0 | Dec 2024 | Jetson Thor upgrade - 2070 TFLOPS, 128GB memory |
| 0.4.0 | Dec 2024 | Meta AI Foundation Models (DINOv3, SAM3, V-JEPA 2) |
| 0.3.0 | Dec 2024 | MoE skill architecture, N2HE encryption |
| 0.2.0 | Nov 2024 | Safety zones, federated learning |
| 0.1.0 | Oct 2024 | Initial release |

---

## Dependencies

### Backend
- Python 3.8+
- ROS 2 Humble/Iron
- FastAPI 0.100+
- PyTorch 2.0+
- TenSEAL (FHE)
- Flower (FL)

### ROS 2
- rclcpp / rclpy
- Isaac ROS (optional)
- realtime_tools
- ros2_control

### Frontend
- React 18.3
- Vite 7.2
- Zustand 5.0
- React Three Fiber 8.17

---

## Documentation

- **Deployment Architecture**: [`docs/DEPLOYMENT_ARCHITECTURE.md`](docs/DEPLOYMENT_ARCHITECTURE.md)
- **ROS 2 Packages**: [`src/ros2/`](src/ros2/)
- **API Reference**: `http://localhost:8000/docs`

---

## License

Proprietary - Dynamical.ai © 2024

---

## Support

- **Documentation**: [docs.dynamical.ai](https://docs.dynamical.ai)
- **Email**: support@dynamical.ai
- **Emergency**: emergency@dynamical.ai (safety issues)

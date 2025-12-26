# Dynamical Edge Platform v0.9.0

![Version](https://img.shields.io/badge/version-0.9.0-blue)
![Status](https://img.shields.io/badge/status-Production-green)
![License](https://img.shields.io/badge/license-Proprietary-red)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble/Iron-22314E)

> **Privacy-Preserving Imitation Learning Platform with Deterministic Safety Guarantees**

A production-grade robotics platform that enables humanoid robots to learn manipulation skills from human demonstrations while maintaining deterministic safety guarantees and data privacy.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMICAL EDGE PLATFORM v0.9.0                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                              CLOUD LAYER                                    ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        ││
│  │  │ Skill       │  │ Federated   │  │ MoE Router  │  │ MOAI FHE    │        ││
│  │  │ Library     │  │ Learning    │  │ (Training)  │  │ (Offline)   │        ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        ││
│  └────────────────────────────────────────┬────────────────────────────────────┘│
│                                           │ Skill Sync / Updates                │
│  ┌────────────────────────────────────────┼────────────────────────────────────┐│
│  │                        EDGE COMPUTE (Jetson Thor)                           ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌───▼─────────┐  ┌─────────────┐        ││
│  │  │ ONVIF       │  │ DYGlove/    │  │ Skill Cache │  │ Pi0.5 VLA   │        ││
│  │  │ Cameras     │  │ MANUS       │  │ (Local MoE) │  │ Inference   │        ││
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘  └─────────────┘        ││
│  │         │                │                                                  ││
│  │         └────────┬───────┘                                                  ││
│  │                  │ Sensor Data                                              ││
│  └──────────────────┼──────────────────────────────────────────────────────────┘│
│                     │                                                            │
│  ┌──────────────────┼──────────────────────────────────────────────────────────┐│
│  │                  │          ROBOT (Onboard Compute)                         ││
│  │                  ▼                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐   ││
│  │  │                    SAFETY STACK (Deterministic)                      │   ││
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   ││
│  │  │  │ CBF Filter  │  │ RTA Simplex │  │ Safety      │  │ Hardware   │  │   ││
│  │  │  │ (1kHz)      │  │ (100Hz)     │  │ Shield      │  │ E-Stop     │  │   ││
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   ││
│  │  └─────────────────────────────────────────────────────────────────────┘   ││
│  │                              │                                              ││
│  │                              ▼                                              ││
│  │                    ┌────────────────────┐                                   ││
│  │                    │   MOTOR CONTROL    │                                   ││
│  │                    │   (200Hz)          │                                   ││
│  │                    └────────────────────┘                                   ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Deployment Architecture

| Layer | Hardware | Responsibilities |
|-------|----------|------------------|
| **Cloud** | AWS/GCP servers | Skill Library, Federated Learning, MoE training, MOAI FHE |
| **Edge** | NVIDIA Jetson Thor | VLA inference, perception, skill caching, data collection |
| **Robot** | Onboard compute | Safety (CBF/RTA), motor control, sensor fusion |

---

## Key v0.9.0 Changes

| Change | Description |
|--------|-------------|
| **SkillBlender Removed** | VLA models learn multi-objective behavior implicitly via Deep Imitative Learning. No runtime skill blending needed. |
| **Unified Orchestrator** | Single entry point for skill routing, robot assignment, and location adaptation |
| **CBF + RTA Safety** | Deterministic safety guarantees with Control Barrier Functions and Runtime Assurance |
| **Pi0.5 VLA** | Official Physical Intelligence integration via openpi |
| **Meta AI v3** | DINOv3, SAM3, V-JEPA 2 perception stack |
| **Cleaned Codebase** | Removed mock implementations, consolidated duplicate code |

---

## Architecture

### 5-Tier Control Architecture

The platform operates on a strict timing hierarchy where higher tiers cannot override lower tier safety decisions:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TIMING ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TIER 0 │ 1000 Hz │ SAFETY KERNEL (CPU-only, <500μs)                            │
│         │         │ ├─ Control Barrier Functions (CBF)                          │
│         │         │ ├─ Runtime Assurance (RTA Simplex)                          │
│         │         │ └─ GUARANTEE: h(x) ≥ 0 always (constraints never violated)  │
│─────────┼─────────┼─────────────────────────────────────────────────────────────│
│  TIER 1 │  200 Hz │ CONTROL LOOP (Edge, <5ms)                                   │
│         │         │ ├─ RobotSkillInvoker: Action execution                       │
│         │         │ ├─ Composition Verifier: Skill chain validation             │
│         │         │ └─ Runtime Monitor: Temporal property checking              │
│─────────┼─────────┼─────────────────────────────────────────────────────────────│
│  TIER 2 │  10 Hz  │ POLICY LOOP (GPU, <100ms)                                   │
│         │         │ ├─ Pi0.5 VLA: Vision-Language-Action inference              │
│         │         │ ├─ Diffusion Planner: Trajectory refinement                 │
│         │         │ └─ DINOv3/SAM3/V-JEPA2: Perception                          │
│─────────┼─────────┼─────────────────────────────────────────────────────────────│
│  TIER 3 │   1 Hz  │ PLANNING LOOP (Cloud/Edge, <1s)                             │
│         │         │ ├─ UnifiedSkillOrchestrator: Task decomposition             │
│         │         │ ├─ MoE Router: Skill selection                              │
│         │         │ └─ Spatial Router: Robot assignment                         │
│─────────┼─────────┼─────────────────────────────────────────────────────────────│
│  TIER 4 │  Async  │ CLOUD SYNC (seconds to hours)                               │
│         │         │ ├─ Federated Learning: Privacy-preserving training          │
│         │         │ ├─ Skill Sync: MoE skill updates                            │
│         │         │ └─ MOAI FHE: Encrypted inference (offline only)             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **UnifiedSkillOrchestrator** | `src/core/unified_skill_orchestrator.py` | Task decomposition, MoE routing, robot assignment |
| **RobotSkillInvoker** | `src/core/robot_skill_invoker.py` | 200Hz skill execution on edge |
| **MoESkillRouter** | `src/platform/cloud/moe_skill_router.py` | Skill selection via Mixture-of-Experts |
| **SkillLibraryClient** | `src/platform/cloud/model_client.py` | Cloud skill library access |
| **CBFFilter** | `src/safety/cbf/filter.py` | Deterministic safety via Control Barrier Functions |
| **RTASimplex** | `src/safety/rta/simplex.py` | Learned ↔ Baseline controller switching |
| **DeepImitativeLearning** | `src/spatial_intelligence/deep_imitative_learning.py` | Pi0.5 + Diffusion + RIP pipeline |
| **IntegratedPipeline** | `src/pipeline/integrated_pipeline.py` | End-to-end data processing |

### Skill Library Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SKILL LIBRARY SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CLOUD (Skill Library)                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        ││
│  │  │ skill_grasp │  │ skill_pour  │  │ skill_place │  │ skill_...   │        ││
│  │  │ (MoE Expert)│  │ (MoE Expert)│  │ (MoE Expert)│  │ (MoE Expert)│        ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        ││
│  │                                                                              ││
│  │  EncryptedSkillStorage (N2HE, 128-bit security)                             ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                              │                                                   │
│                              │ Sync (gRPC + TLS)                                │
│                              ▼                                                   │
│  EDGE (Skill Cache)                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  EdgeSkillClient                                                            ││
│  │  ├─ Local cache: /var/lib/dynamical/skill_cache                             ││
│  │  ├─ Max size: 1GB (configurable)                                            ││
│  │  ├─ Preloaded skills for low-latency execution                              ││
│  │  └─ Automatic sync on network reconnection                                  ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Skills are:
- **MoE Experts**: Each skill is a Mixture-of-Experts expert trained on specific manipulation primitives
- **Encrypted at Rest**: N2HE (LWE-based) homomorphic encryption in cloud storage
- **Cached Locally**: Downloaded to edge device for low-latency execution
- **Privacy-Preserving**: Trained via federated learning, gradients never leave edge unencrypted

---

## Project Structure

```
src/
├── core/                           # Core System Components
│   ├── unified_skill_orchestrator.py   # Task → Skills → Robots (v0.9.0)
│   ├── robot_skill_invoker.py          # 200Hz skill execution
│   ├── gmr_retargeting.py              # Human → Robot motion transfer
│   ├── error_handling.py               # Graceful degradation
│   ├── timing_architecture.py          # 5-tier timing system
│   └── config_loader.py                # Configuration management
│
├── safety/                         # Deterministic Safety Stack
│   ├── cbf/                            # Control Barrier Functions
│   │   ├── filter.py                   # QP-based CBF filter
│   │   └── barriers.py                 # Collision, joint, velocity barriers
│   ├── rta/                            # Runtime Assurance
│   │   ├── simplex.py                  # Learned ↔ Baseline switching
│   │   └── baseline.py                 # Verified safe controllers
│   └── runtime_monitor/                # Temporal property checking
│
├── spatial_intelligence/           # Deep Imitative Learning
│   ├── pi0/                            # Physical Intelligence Pi0.5
│   ├── planning/                       # Diffusion trajectory planning
│   ├── safety/                         # RIP uncertainty estimation
│   └── deep_imitative_learning.py      # Unified DIL pipeline
│
├── meta_ai/                        # Meta AI Perception Models
│   ├── dinov3.py                       # DINOv3 features
│   ├── sam3.py                         # SAM3 segmentation
│   └── vjepa2.py                       # V-JEPA 2 temporal understanding
│
├── composition/                    # Skill Composition Framework
│   ├── contracts.py                    # Pre/post condition contracts
│   ├── verifier.py                     # Static composition verification
│   └── runtime/                        # Runtime verification
│
├── platform/                       # Platform Services
│   ├── api/                            # FastAPI backend
│   ├── edge/                           # Edge device SDKs (DYGlove, MANUS)
│   ├── cloud/                          # Cloud services (MoE, Federated Learning)
│   ├── calibration/                    # Hardware calibration
│   └── ui/                             # React/Vite web interface
│
├── robot_runtime/                  # Robot Execution Layer
│   ├── agent.py                        # ROS2 robot agent
│   ├── safety_shield.py                # Deterministic safety checks
│   └── perception_pipeline.py          # Unified perception
│
├── simulation/                     # Isaac Lab Simulation
│   ├── isaac_lab/                      # Isaac Lab environments
│   └── bridge/                         # Sim-to-real bridge
│
├── federated/                      # Federated Learning
│   └── unified_pipeline.py             # Privacy-preserving training
│
└── moai/                           # MOAI Privacy Layer
    ├── n2he.py                         # N2HE homomorphic encryption
    └── fhe_wrapper.py                  # FHE model wrapper
```

---

## ROS2 Integration

The platform integrates with ROS2 (Humble/Iron) for robot communication.

### Message Types (`src/ros2/dynamical_msgs/`)

| Message | Purpose |
|---------|---------|
| `RobotState.msg` | Joint positions, velocities, torques, temperatures, EE pose @ 1kHz |
| `Detection.msg` | Object detections with 2D/3D position, tracking, obstacle/human flags |
| `SafetyStatus.msg` | Safety violations, human/obstacle detection, status enum |
| `PerceptionFeatures.msg` | Cascade perception output (detections, depth, segmentation) |
| `SkillStatus.msg` | Skill execution progress and phase |

### Services & Actions

**Services** (synchronous):
```
ExecuteSkill.srv      - Execute skill with parameters
TriggerEstop.srv      - Emergency stop
ResetEstop.srv        - Clear E-stop
GetRobotState.srv     - Query current state
LoadSkill.srv         - Load skill from library
SetControlMode.srv    - Switch control modes
```

**Actions** (async with progress):
```
ExecuteSkillAction.action  - Skill execution with phase feedback
MoveToPosition.action      - Goal-based positioning
```

### ROS2 Bridge Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 BRIDGE (ros2_bridge.py)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Dynamical Python Runtime                      ROS2 Ecosystem                   │
│  ┌─────────────────────────┐                 ┌─────────────────────────┐        │
│  │  Robot Runtime Agent    │◄───────────────►│  /dynamical/robot_state │        │
│  │  SafetyShield           │◄───────────────►│  /dynamical/safety      │        │
│  │  SkillExecutor          │◄───────────────►│  /dynamical/skills/*    │        │
│  │  PerceptionPipeline     │◄───────────────►│  /dynamical/perception  │        │
│  └─────────────────────────┘                 └─────────────────────────┘        │
│                                                                                  │
│  Configuration:                                                                  │
│  - Node: "robot_runtime_bridge" in namespace "dynamical"                        │
│  - QoS depth: 10 (buffered)                                                     │
│  - Executor: MultiThreaded with ReentrantCallbackGroup                          │
│  - Fallback: Standalone mode if ROS2 unavailable                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detection & Safety System

### 3-Level Hierarchical Safety

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SAFETY HIERARCHY (Cannot be overridden)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LEVEL 0: HARDWARE E-STOP (Physical, cannot be software-overridden)             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  • Physical button → direct motor power cut                                 ││
│  │  • Capacitive human proximity sensor → immediate halt                       ││
│  │  • Hardware watchdog: No heartbeat for 2ms → power cut                      ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  LEVEL 1: DETERMINISTIC SOFTWARE (1kHz, <500μs, CPU-only)                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Hard-coded limits from URDF (cannot be changed at runtime):                ││
│  │  • Joint position: ±2° margin from URDF limits                              ││
│  │  • Joint velocity: Per-joint, temperature-compensated                       ││
│  │  • Joint torque: Motor spec ±10% margin                                     ││
│  │  • Obstacle proximity: 10cm minimum clearance (CBF enforced)                ││
│  │  • Self-collision: Pre-computed collision pairs                             ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  LEVEL 2: ML-ASSISTED ADVISORS (30Hz, informational only)                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  ⚠️ CANNOT OVERRIDE LEVEL 0 OR 1 - Advisory only                            ││
│  │  • V-JEPA future prediction → "caution" flag                                ││
│  │  • Human intent prediction → speed reduction suggestion                     ││
│  │  • Anomaly detection → log + alert (operator must acknowledge)              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### What Gets Detected

| Detection Type | Method | Threshold | Action |
|----------------|--------|-----------|--------|
| **Joint Limits** | URDF comparison | ±0.035 rad (2°) | Block motion |
| **Velocity Limits** | Per-joint check | 90% of max | Block/reduce |
| **Torque Limits** | Motor spec | 90% of max | Block motion |
| **Obstacles** | DINOv3 + SAM3 → sphere approx | 0.1m minimum | CBF constraint |
| **Humans** | RTMPose detection | 0.5m safety zone | Reduce speed or stop |
| **Self-Collision** | Pre-computed pairs | Sphere overlap | Block motion |
| **Heartbeat Loss** | Watchdog timer | 10ms | E-stop trigger |

### CBF Barrier Functions (`src/safety/cbf/`)

Control Barrier Functions guarantee `h(x) ≥ 0` is invariant:

```python
# Barrier principle: h(x) ≥ 0 is safe, maintained by ḣ + α·h ≥ 0

Barrier Types:
├── Joint Position Barriers (per-joint upper/lower limits)
├── Joint Velocity Barriers (max velocity enforcement)
├── Obstacle Distance Barriers (min clearance from segmented objects)
└── Self-Collision Barriers (pre-computed collision pairs)

# If proposed action would violate:
constraint(state, action) = ∇h · action + α·h ≥ 0
# Solve QP to find minimal modification that satisfies constraint
```

### Safety States & Violations

```python
SafetyStatus:
  OK (0)        → Normal operation
  WARNING (1)   → Speed reduction applied
  VIOLATION (2) → Controlled stop in progress
  ESTOP (3)     → Emergency stop active

ViolationSeverity:
  INFO      → Logged, no action
  WARNING   → Speed reduced 50%
  VIOLATION → Controlled stop
  CRITICAL  → Immediate E-stop
```

---

## RTMPose Integration

Real-time 2D pose estimation for human body tracking.

### Pipeline (`src/core/pose_inference/rtmpose_real.py`)

```
Input Image
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RTMPOSE PIPELINE                                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1. ROI Extraction                                                               │
│     • Bounding box from detector + 25% padding                                  │
│     • Resize to model input (192×256 or 288×384)                                │
│                                                                                  │
│  2. Preprocessing                                                                │
│     • BGR→RGB conversion                                                         │
│     • ImageNet normalization (mean/std)                                         │
│     • CHW format + batch dimension                                              │
│                                                                                  │
│  3. ONNX Inference (CUDA or CPU fallback)                                       │
│     • rtmpose-m: 17 COCO keypoints (body)                                       │
│     • rtmw-x-wholebody: 133 keypoints (body+hands+face)                         │
│                                                                                  │
│  4. SimCC Decoding                                                               │
│     • Two 1D heatmaps per keypoint (X and Y)                                    │
│     • Softmax → probability distributions                                       │
│     • Argmax → coordinate extraction                                            │
│     • Score = min(x_confidence, y_confidence)                                   │
│                                                                                  │
│  5. Output: Pose2DResult                                                         │
│     • keypoints [N_persons, 17/133, 3]: [x, y, confidence]                      │
│     • bboxes [N_persons, 5]: [x1, y1, x2, y2, score]                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
GMR Retargeting (human → robot motion)
```

### Model Options

| Model | Input Size | Keypoints | Use Case |
|-------|-----------|-----------|----------|
| `rtmpose-s` | 192×256 | 17 (COCO) | Fast, low resource |
| `rtmpose-m` | 192×256 | 17 (COCO) | **Default**, balanced |
| `rtmpose-l` | 192×256 | 17 (COCO) | High accuracy |
| `rtmw-x-wholebody` | 288×384 | 133 | Full body+hands+face |

---

## MANUS Glove Integration

High-precision hand tracking with haptic feedback.

### Supported Hardware

- MANUS Quantum (highest precision)
- MANUS Prime Series
- MANUS Metaglove

### 21-DOF Hand Model (`src/platform/edge/manus_sdk.py`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MANUS → DYGLOVE MAPPING (21 DOF)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  THUMB (5 DOF, indices 0-4):                                                    │
│  ├─ CMC Flexion     → tm_flex (0)                                               │
│  ├─ CMC Spread      → tm_abd  (1)                                               │
│  ├─ MCP Flexion     → mcp     (2)                                               │
│  ├─ IP Flexion      → ip      (3)                                               │
│  └─ Wrist Pro/Sup   → wrist_ps (4, from IMU quaternion)                         │
│                                                                                  │
│  INDEX (4 DOF, indices 5-8):                                                    │
│  ├─ MCP Flexion     → mcp_flex (5)                                              │
│  ├─ MCP Abduction   → mcp_abd  (6)                                              │
│  ├─ PIP Flexion     → pip      (7)                                              │
│  └─ DIP Flexion     → dip      (8)                                              │
│                                                                                  │
│  MIDDLE (4 DOF, indices 9-12):  Same pattern                                    │
│  RING (4 DOF, indices 13-16):   Same pattern                                    │
│  PINKY (4 DOF, indices 17-20):  Same pattern                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```python
# Connection
driver = ManusGloveDriver(side=Hand.RIGHT)
driver.connect()  # Blocks until glove found (5s timeout)

# Reading state (120Hz default)
state = driver.get_state()  # → GloveState with 21-DOF angles
wrist_orientation = state.wrist_orientation  # Quaternion from IMU

# Streaming callback
driver.start_streaming(callback, rate_hz=120.0)

# Haptic feedback (vibration per finger)
driver.set_haptic_feedback(fingers=[0,1,2], intensity=0.8)
```

### Discovery Methods

1. Native SDK discovery (if MANUS SDK installed)
2. ROS2 topic discovery (`/manus_glove_*` topics)
3. Network broadcast (ports 49220, 49221)

---

## Skill Compositionality

Formal contracts enabling safe skill composition with verification.

### Contract Structure (`src/composition/contracts.py`)

```python
SkillContract:
  name: str                        # Unique skill identifier
  preconditions: PredicateSet      # Must hold BEFORE execution
  postconditions: PredicateSet     # Guaranteed AFTER execution
  invariants: PredicateSet         # Must hold DURING execution
  min_duration / max_duration      # Timing bounds
  executor: Callable               # Actual implementation
```

### Standard Predicates

| Predicate | Check | Used By |
|-----------|-------|---------|
| `GRIPPER_OPEN` | `gripper_state < 0.5` | grasp, release |
| `GRIPPER_CLOSED` | `gripper_state > 0.5` | lift, place |
| `OBJECT_VISIBLE` | `object_detected == True` | grasp |
| `HOLDING_OBJECT` | `holding == True` | lift, place |
| `AT_TARGET` | `at_target == True` | grasp, place |
| `ROBOT_STATIONARY` | `‖velocity‖ < 0.01` | place |
| `PATH_CLEAR` | `path_clear == True` | reach, retract |

### Standard Skill Library (`src/composition/library.py`)

```
reach:   PATH_CLEAR → AT_TARGET                    [0.5-5.0s]
grasp:   AT_TARGET + GRIPPER_OPEN + OBJECT_VISIBLE → HOLDING_OBJECT [0.2-2.0s]
lift:    HOLDING_OBJECT + GRIPPER_CLOSED → HOLDING_OBJECT [0.3-3.0s]
place:   HOLDING_OBJECT + AT_TARGET → GRIPPER_OPEN [0.3-3.0s]
release: GRIPPER_CLOSED → GRIPPER_OPEN             [0.1-1.0s]
retract: GRIPPER_OPEN → PATH_CLEAR                 [0.3-3.0s]
```

### Composition Verification

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SKILL COMPOSITION VERIFICATION                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Skill Chain: reach → grasp → lift → place → release → retract                 │
│                                                                                  │
│  For each transition (SkillA → SkillB):                                         │
│    ✓ SkillA.postconditions ⊆ SkillB.preconditions?                              │
│                                                                                  │
│  Example: grasp → lift                                                           │
│    grasp.post   = {HOLDING_OBJECT, GRIPPER_CLOSED}                              │
│    lift.pre     = {HOLDING_OBJECT, GRIPPER_CLOSED}                              │
│    ✓ Valid transition (post satisfies pre)                                      │
│                                                                                  │
│  Runtime Verification:                                                           │
│  1. Pre-execution:  Check preconditions on current state                        │
│  2. During:         Monitor invariants (no violation allowed)                   │
│  3. Post-execution: Verify postconditions actually hold                         │
│  4. Composition:    Validate next skill can run                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Federated Learning Pipeline

Privacy-preserving distributed training across robot fleet.

### 7-Stage Pipeline (`src/federated/unified_pipeline.py`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     FEDERATED LEARNING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CLIENT (Edge)                                                                   │
│  ─────────────                                                                   │
│  Gradients                                                                       │
│      │                                                                           │
│      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 1: GRADIENT CLIPPING                                                  ││
│  │ • Clips ‖g‖ to max_norm (default 1.0)                                       ││
│  │ • Bounds magnitude BEFORE lossy operations                                  ││
│  │ • Enables differential privacy guarantees                                   ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 2: ERROR FEEDBACK                                                     ││
│  │ • Adds accumulated residuals from previous rounds                           ││
│  │ • g[t] += decay × residuals[t-1]                                            ││
│  │ • Prevents "unfairly dropped" gradients from being lost                     ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 3: TOP-K SPARSIFICATION                                               ││
│  │ • Keeps top 1% of gradients by magnitude                                    ││
│  │ • Reduces communication by 99%                                              ││
│  │ • Stores residual for error feedback                                        ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 4: ADAPTIVE MOAI COMPRESSION                                          ││
│  │ • Quantization: bits = max(2, 32/compression_ratio)                         ││
│  │ • Compression ratio: 8x-128x (adapts to quality feedback)                   ││
│  │ • Target quality: 95% retained                                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 5: QUALITY MONITOR (Observer)                                         ││
│  │ • Tracks: compression_quality, clip_ratio, sparsity, fhe_noise             ││
│  │ • Alerts: COMPRESSION_DEGRADED, CLIPPING_AGGRESSIVE, FHE_NOISE_HIGH        ││
│  │ • Provides feedback to Stage 4 for adaptation                               ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 6: NOISE-AWARE FHE (N2HE Encryption)                                  ││
│  │ • LWE-based encryption (128-bit post-quantum security)                      ││
│  │ • Noise budget tracking (bootstraps when <20% remains)                      ││
│  │ • Homomorphic addition: Enc(a) + Enc(b) = Enc(a+b)                          ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  Encrypted Update ────────────────────────────────────────────────► CLOUD       │
│                                                                                  │
│  CLOUD (Aggregation)                                                             │
│  ──────────────────                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Stage 7: HIERARCHICAL AGGREGATION                                           ││
│  │ • Tree structure: 10 clients per aggregator node                            ││
│  │ • Max depth: 4 levels                                                       ││
│  │ • Reduces FHE operations: O(log N) instead of O(N)                          ││
│  │ • All aggregation done on encrypted data (server never sees gradients)     ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                           │
│      ▼                                                                           │
│  Decrypted Aggregated Model (only server can decrypt)                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Privacy Guarantees

| Stage | Privacy Contribution |
|-------|---------------------|
| Gradient Clipping | Bounded sensitivity for differential privacy |
| Sparsification | Information loss (99% of gradients dropped) |
| Compression | Quantization noise hides exact values |
| FHE Encryption | Semantic security (cannot decrypt without key) |
| Hierarchical Agg | Aggregation on encrypted data only |

---

## N2HE / MOAI Encryption

Homomorphic encryption for privacy-preserving computation.

### N2HE Parameters (`src/moai/n2he.py`)

```python
N2HE_128 (standard):
  n (LWE dimension):          1024
  q (ciphertext modulus):     2^32
  σ (error std dev):          3.2
  t (plaintext modulus):      2^16
  Security:                   128-bit post-quantum

N2HE_192 (high security):
  n: 1536, q: 2^48, t: 2^16
  Security:                   192-bit post-quantum
```

### LWE Ciphertext Structure

```
Ciphertext ct = (a, b) where:
  a: Random vector [n]
  b: <a, s> + e + Δ·m  (s = secret key, e = small error, m = message)

Properties:
  • Additive homomorphism: Enc(m₁) + Enc(m₂) = Enc(m₁+m₂)
  • Scalar multiplication: c × Enc(m) = Enc(c×m)
  • Noise budget: Decreases with each operation (bootstrap to refresh)
```

### MOAI FHE Context (`src/moai/moai_fhe.py`)

**⚠️ WARNING: OFFLINE ONLY - NOT REAL-TIME CAPABLE**

| Operation | Latency | Use Case |
|-----------|---------|----------|
| FHE attention (512-dim) | ~30 seconds | Offline analysis |
| Full transformer forward | ~60 seconds | Batch processing |
| Batch of 100 demos | ~2 hours | Overnight training |

**Appropriate Use Cases:**
- ✅ Overnight batch processing of encrypted demonstrations
- ✅ Privacy-preserving skill distillation (hours latency OK)
- ✅ Compliance audit on encrypted logs
- ✅ Weekly analytics on encrypted fleet data

**DO NOT USE FOR:**
- ❌ Real-time inference (impossible)
- ❌ Control loop integration (violates timing contract)
- ❌ Online learning (latency incompatible)
- ❌ Anything requiring <1 second latency

### Integration with Federated Learning

```
Client Gradients
     ↓
Stage 4: Compress (8x-128x)
     ↓
Stage 6: N2HE Encrypt
     ↓  ← LWECiphertext with noise budget tracking
Stage 7: Homomorphic Aggregation
     ↓  ← Tree-based: Enc(a) + Enc(b) = Enc(a+b)
Server Decrypt (only holder of secret key)
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform

# Install dependencies
pip install -e .

# Verify installation
python -c "from src.version import __version__; print(f'Dynamical v{__version__}')"
```

### Basic Usage

```python
from src.core import get_orchestrator, get_skill_invoker
from src.safety import SafePolicyExecutor
import numpy as np

# 1. Orchestrate a task (TIER 3: Planning)
orchestrator = get_orchestrator()
orchestrator.configure({
    "workspaces": [{"id": "ws1", "name": "Assembly", ...}],
    "robots": [{"id": "thor_001", "capabilities": ["manipulation"]}]
})

result = orchestrator.orchestrate(OrchestrationRequest(
    task_description="Pick up the red cup and place it on the shelf",
    assignment_strategy=AssignmentStrategy.NEAREST
))

# 2. Execute each step (TIER 1: Control @ 200Hz)
invoker = get_skill_invoker()
for step in result.plan.steps:
    action_result = orchestrator.execute_step(step, observation)

    # 3. Safety filter (TIER 0: Safety @ 1kHz)
    safe_executor = SafePolicyExecutor.for_jetson_thor()
    safe_action = safe_executor.execute(
        learned_action=action_result["action"],
        state=robot.get_state()
    )

    robot.send_command(safe_action)  # GUARANTEED safe
```

### API Endpoints

```bash
# Task orchestration (v0.9.0)
POST /api/v1/skills/orchestrate
{
    "task_description": "Pick up the cup",
    "robot_id": "thor_001",
    "assignment_strategy": "nearest"
}

# Direct skill invocation
POST /api/v1/skills/invoke
{
    "skill_ids": ["skill_grasp"],
    "joint_positions": [0.0, 0.1, ...],
    "joint_velocities": [0.0, 0.0, ...],
    "mode": "direct"
}

# Safety check
POST /api/v1/safety/check
{
    "joint_positions": [...],
    "joint_velocities": [...]
}
```

---

## Safety Philosophy

**Safety is DETERMINISTIC, not probabilistic.**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SAFETY ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ML/AI Components                    │   Safety Components                      │
│   (CAN be uncertain)                  │   (MUST be deterministic)               │
│                                       │                                          │
│   ┌─────────────────────┐            │   ┌─────────────────────┐               │
│   │  Pi0.5 VLA          │            │   │  CBF Filter         │               │
│   │  (action proposal)  │ ──────────────>│  (hard constraints) │               │
│   └─────────────────────┘            │   └──────────┬──────────┘               │
│                                       │              │                           │
│   ┌─────────────────────┐            │   ┌──────────▼──────────┐               │
│   │  DINOv3/SAM3        │            │   │  RTA Simplex        │               │
│   │  (scene understand) │            │   │  (certified switch) │               │
│   └─────────────────────┘            │   └──────────┬──────────┘               │
│                                       │              │                           │
│   ┌─────────────────────┐            │   ┌──────────▼──────────┐               │
│   │  RIP Uncertainty    │            │   │  Hardware E-Stop    │               │
│   │  (informational)    │            │   │  (physical switch)  │               │
│   └─────────────────────┘            │   └─────────────────────┘               │
│                                       │                                          │
│   NOTE: ML predictions are           │   GUARANTEE: h(x) ≥ 0 always            │
│   INFORMATIONAL ONLY                 │   Constraints NEVER violated             │
│                                       │                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Control Barrier Functions (CBF)

```python
# CBF guarantees: h(x) ≥ 0 is invariant
# If proposed action would violate constraint, find minimal modification

class CBFFilter:
    def filter(self, proposed_action, state) -> CBFResult:
        for barrier in [collision, joint_limits, velocity, force]:
            if barrier.would_violate(proposed_action, state):
                return self._solve_qp(proposed_action, state)  # Find safe action
        return CBFResult(safe_action=proposed_action, was_modified=False)
```

---

## Hardware Requirements

### Edge Compute (Jetson Thor - Separate from Robot)
| Component | Specification | Used For |
|-----------|--------------|----------|
| **Platform** | NVIDIA Jetson Thor | Edge inference, skill caching |
| **CPU** | ARM Cortex-A78AE (12-core) | Orchestration, data processing |
| **GPU** | NVIDIA Blackwell (800 TFLOPS) | VLA inference, perception |
| **Memory** | 128GB LPDDR5X | Model loading, skill cache |

### Robot (Onboard Compute)
| Component | Specification | Used For |
|-----------|--------------|----------|
| **Safety MCU** | ARM Cortex-R (dedicated) | CBF/RTA @ 1kHz |
| **Control CPU** | Real-time processor | Motor control @ 200Hz |
| **Hardware E-Stop** | Physical switch | Emergency stop (independent) |

### Peripherals
| Component | Specification | Used For |
|-----------|--------------|----------|
| **Cameras** | ONVIF PTZ | Multi-angle capture |
| **Gloves** | DYGlove / MANUS | Hand tracking, teleoperation |
| **Network** | Ethernet/WiFi | Cloud sync (not required for operation) |

---

## Cloud vs Edge Dependencies

### What Requires Cloud
| Feature | Cloud Dependency | Notes |
|---------|------------------|-------|
| **Skill Library** | Required | MoE skills stored in cloud, synced to edge cache |
| **Federated Learning** | Required | Privacy-preserving training across fleet |
| **New Skill Training** | Required | Skills trained in cloud, deployed to edge |
| **MOAI FHE** | Required | Homomorphic encryption processing (~60s/inference) |
| **Fleet Management** | Required | Multi-robot coordination, skill distribution |

### What Works Offline (Degraded Mode)
| Feature | Offline Capability | Notes |
|---------|-------------------|-------|
| **Safety (CBF/RTA)** | Full | Runs on robot onboard compute, no network needed |
| **Cached Skills** | Full | Previously synced skills continue working |
| **VLA Inference** | Full | Pi0.5 runs on Jetson Thor edge compute |
| **Basic Perception** | Full | DINOv3-B, SAM3 cached on edge |
| **Motor Control** | Full | 200Hz control loop on robot |
| **Telemetry** | Buffered | Stored locally, uploaded when connection restored |

### Network Failure Behavior
When network connectivity is lost:
1. **Robot continues operating** with cached skills (no new skill learning)
2. **Safety stack unaffected** (runs entirely on robot hardware)
3. **Telemetry buffered** locally until connection restored
4. **No skill updates** until network returns
5. **Federated learning paused** until fleet reconnects

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **0.9.0** | 2024-12 | Removed SkillBlender, unified orchestration, CBF+RTA safety |
| 0.8.0 | 2024-11 | Deep Imitative Learning, compositional skills, Pi0.5 VLA |
| 0.7.1 | 2024-10 | Timing contracts, threat model documentation |
| 0.7.0 | 2024-09 | ROS 2 integration, C++ safety node |
| 0.6.0 | 2024-08 | Frontend modernization (React/Vite) |
| 0.5.0 | 2024-07 | Jetson Thor support |

---

## License

Proprietary - Dynamical.ai © 2024-2025

---

## References

- [Control Barrier Functions](https://arxiv.org/abs/1903.11199) - Ames et al.
- [Runtime Assurance (Simplex)](https://ntrs.nasa.gov/citations/20180002983) - NASA
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Chi et al.
- [Pi0](https://www.physicalintelligence.company/blog/pi0) - Physical Intelligence
- [DINOv2](https://arxiv.org/abs/2304.07193) - Meta AI
- [SAM2](https://arxiv.org/abs/2408.00714) - Meta AI

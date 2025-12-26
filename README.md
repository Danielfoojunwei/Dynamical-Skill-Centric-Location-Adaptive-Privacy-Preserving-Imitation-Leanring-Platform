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

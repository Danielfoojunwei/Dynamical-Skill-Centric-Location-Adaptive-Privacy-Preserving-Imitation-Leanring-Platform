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
│  HUMAN DEMONSTRATION                           ROBOT EXECUTION                   │
│  ┌────────────────────┐                       ┌────────────────────┐            │
│  │  ONVIF Cameras     │                       │  Jetson Thor       │            │
│  │  MANUS/DYGlove     │ ─────────────────────>│  (Edge Compute)    │            │
│  │  3D Pose Capture   │    Imitation          │                    │            │
│  └────────────────────┘    Learning           └─────────┬──────────┘            │
│                                                         │                        │
│                              ┌──────────────────────────┼──────────────────────┐│
│                              │         SAFETY STACK (Deterministic)            ││
│                              │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ ││
│                              │  │ CBF Filter  │  │ RTA Simplex │  │ E-Stop   │ ││
│                              │  │ (1kHz)      │  │ (100Hz)     │  │ (HW)     │ ││
│                              │  └─────────────┘  └─────────────┘  └──────────┘ ││
│                              └──────────────────────────┼──────────────────────┘│
│                                                         │                        │
│                                                         ▼                        │
│                                               ┌────────────────────┐            │
│                                               │   ROBOT CONTROL    │            │
│                                               │   (200Hz)          │            │
│                                               └────────────────────┘            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

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
| **CBFFilter** | `src/safety/cbf/filter.py` | Deterministic safety via Control Barrier Functions |
| **RTASimplex** | `src/safety/rta/simplex.py` | Learned ↔ Baseline controller switching |
| **DeepImitativeLearning** | `src/spatial_intelligence/deep_imitative_learning.py` | Pi0.5 + Diffusion + RIP pipeline |
| **IntegratedPipeline** | `src/pipeline/integrated_pipeline.py` | End-to-end data processing |

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

| Component | Specification | Used For |
|-----------|--------------|----------|
| **Platform** | NVIDIA Jetson Thor | Primary edge compute |
| **CPU** | ARM Cortex-A78AE (12-core) | Safety loop (1kHz) |
| **GPU** | NVIDIA Blackwell | VLA inference (10Hz) |
| **Memory** | 128GB LPDDR5X | Model loading |
| **Cameras** | ONVIF PTZ | Multi-angle capture |
| **Gloves** | DYGlove / MANUS | Hand tracking |

---

## Network Failure Behavior

The platform is designed for **indefinite operation without network connectivity**:

1. **Safety**: Runs entirely on edge CPU (no cloud dependency)
2. **Control**: Cached TensorRT models continue execution
3. **Perception**: Local models (DINOv3-B, SAM3) provide basic perception
4. **Skills**: Cached skills continue working; new skills sync when network returns
5. **Telemetry**: Buffered locally, uploaded when connection restored

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

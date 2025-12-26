# Dynamical Edge Platform v0.8.0

![Version](https://img.shields.io/badge/version-0.8.0-blue)
![Status](https://img.shields.io/badge/status-Production--Ready-green)
![License](https://img.shields.io/badge/license-Proprietary-red)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble/Iron-22314E)

> **Privacy-Preserving Imitation Learning Platform with Deterministic Safety Guarantees**

---

## What's New in v0.8.0

- **Deterministic Safety Stack**: Control Barrier Functions (CBF) + Runtime Assurance (RTA)
- **Compositional Skills**: Verified skill chaining with formal contracts
- **Deep Imitative Learning**: Diffusion Planner + RIP + POIR integration
- **Pi0.5 VLA**: Official Physical Intelligence integration via openpi
- **No Skill Blending**: VLA handles multi-objective implicitly (blending deprecated)
- **Compositional Training**: Automatic skill discovery from ONVIF/MANUS streams using Meta AI

---

## Key Architectural Decision: No Skill Blending

**v0.8.0 eliminates skill blending entirely.** The VLA learns multi-objective behavior implicitly.

```
OLD (v0.7.x - Skill Blending):                 NEW (v0.8.0 - Deep IL):
──────────────────────────────                 ────────────────────────
skill_grasp.infer() → action_1                 vla.infer(instruction) → action
skill_avoid.infer() → action_2                 diffusion.refine(action) → trajectory
blender.blend([a1,a2], weights) → blended      cbf.filter(trajectory) → safe_action
safety.filter(blended) → safe_action

Problems with blending:                        Why Deep IL is better:
- Jitter and oscillations                      - VLA learns multi-objective from demos
- Boundary instability                         - No runtime weight tuning needed
- Fragile weight tuning                        - Smooth trajectories from diffusion
- Safety guarantees break down                 - Hard safety from CBF
```

For long-horizon tasks, use **sequential decomposition** (not blending):
```python
from src.execution import DynamicalExecutor

executor = DynamicalExecutor.for_jetson_thor()

# Simple task - direct execution
result = executor.execute("pick up the red cup", images, state)

# Complex task - auto-decomposed into sequential sub-tasks
result = executor.execute("set the table for dinner", images, state)
# Internally: ["get plates", "place plates", "get utensils", ...]
```

---

## Design Philosophy

This platform follows principles that separate shippable robotics from research demos:

1. **Safety is deterministic, not probabilistic** — CBFs provide hard guarantees, not soft estimates
2. **Skills are composable with verified contracts** — pre/post conditions checked before execution
3. **Giant models are async assessors, not control dependencies** — DINO/SAM/V-JEPA run at 5-30Hz
4. **Privacy has a threat model, not a checkbox** — One scheme per boundary, justified by specific threats

---

## Safety Architecture (v0.8.0)

**Safety is DETERMINISTIC.** Control Barrier Functions guarantee constraint satisfaction.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TIERED CONTROL ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1000 Hz │ SAFETY LOOP: CBF Filter + RTA Simplex + Hardware Limits              │
│          │ ├─ Control Barrier Functions (collision, joint, velocity, force)     │
│          │ ├─ Runtime Assurance switching (learned ↔ baseline)                  │
│          │ └─ GUARANTEE: h(x) ≥ 0 invariant, constraints never violated         │
│──────────┼──────────────────────────────────────────────────────────────────────│
│   100 Hz │ CONTROL LOOP: Runtime Monitors + Composition Verifier                │
│          │ ├─ Temporal property checking (LTL/MTL)                              │
│          │ └─ Skill chain verification before execution                         │
│──────────┼──────────────────────────────────────────────────────────────────────│
│    10 Hz │ POLICY LOOP: Pi0.5 VLA + Diffusion Planner + RIP                     │
│          │ ├─ Vision-Language-Action inference                                  │
│          │ ├─ Diffusion trajectory refinement                                   │
│          │ └─ Epistemic uncertainty estimation (informational)                  │
│──────────┼──────────────────────────────────────────────────────────────────────│
│     1 Hz │ PLANNING LOOP: Skill Composer + Goal Planner                         │
│          │ └─ Verified composition: post(A) ⊆ pre(B)                            │
│──────────┼──────────────────────────────────────────────────────────────────────│
│    Async │ OPS LOOP: Drift Detection + MLOps + TEE Training                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Control Barrier Functions

```python
# File: src/safety/cbf/filter.py

class CBFFilter:
    """
    Deterministic safety filter using Control Barrier Functions.

    Solves: min ||a - a_proposed||²
            s.t. ∇h(x)·a + α·h(x) ≥ 0  for all barriers

    GUARANTEE: h(x) ≥ 0 is invariant (constraints never violated)
    """

    def filter(self, proposed_action: np.ndarray, state: RobotState) -> CBFResult:
        # Barriers: collision, joint limits, velocity, force
        for barrier in self.barriers:
            # Compute constraint: dh/dt + α·h ≥ 0
            margin = barrier.constraint(state, proposed_action)
            if margin < 0:
                # Solve QP to find minimally modified safe action
                return self._solve_cbf_qp(proposed_action, state)

        return CBFResult(safe_action=proposed_action, was_modified=False)
```

### Runtime Assurance (Simplex)

```python
# File: src/safety/rta/simplex.py

class RuntimeAssurance:
    """
    Simplex-based switching between learned policy and verified baseline.

    If learned policy is not certifiably safe → switch to baseline controller.
    """

    def arbitrate(self, learned_action, state) -> Tuple[np.ndarray, ControlSource]:
        certificate = self.monitor.certify(learned_action, state)

        if certificate.is_safe:
            return learned_action, ControlSource.LEARNED
        else:
            # Switch to verified baseline (impedance/safe-stop)
            return self.baseline.compute(state), ControlSource.BASELINE
```

---

## Compositional Skills (v0.8.0)

**Skills have formal contracts.** Composition is verified before execution.

```python
# File: src/composition/contracts.py

@dataclass
class SkillContract:
    name: str
    preconditions: PredicateSet   # Must hold before
    postconditions: PredicateSet  # Guaranteed after
    invariants: PredicateSet      # Must hold throughout

# File: src/composition/verifier.py

class CompositionVerifier:
    def verify_chain(self, skills: List[SkillContract]) -> VerificationResult:
        """
        Verify: post(skill_i) ⊆ pre(skill_{i+1}) for all adjacent pairs.
        Reject unsafe compositions before execution.
        """
```

### Skill Library

```python
# Pre-defined manipulation skills with contracts

SKILL_LIBRARY = {
    "reach": SkillContract(
        preconditions={"path_clear"},
        postconditions={"at_target"},
    ),
    "grasp": SkillContract(
        preconditions={"at_target", "gripper_open", "object_visible"},
        postconditions={"holding_object", "gripper_closed"},
    ),
    "place": SkillContract(
        preconditions={"holding_object", "at_target"},
        postconditions={"gripper_open"},
    ),
}
```

---

## Deep Imitative Learning Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEEP IMITATIVE LEARNING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                        │
│  │   Pi0.5     │────►│  Diffusion  │────►│    RIP      │                        │
│  │    VLA      │     │   Planner   │     │  (inform)   │                        │
│  └─────────────┘     └─────────────┘     └─────────────┘                        │
│         │                  │                   │                                 │
│         ▼                  ▼                   ▼                                 │
│   Semantic          Smooth            Uncertainty                               │
│   Understanding     Trajectories      Estimation                                │
│                                                                                  │
│                                ↓                                                 │
│                    ┌─────────────────────┐                                      │
│                    │  SafePolicyExecutor │                                      │
│                    │  (CBF + RTA wrap)   │                                      │
│                    └─────────────────────┘                                      │
│                                ↓                                                 │
│                         SAFE ACTION                                             │
│                     (guaranteed by CBF)                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from src.safety import SafePolicyExecutor
from src.spatial_intelligence import DeepImitativeLearning

# Create safety-wrapped policy executor
executor = SafePolicyExecutor.for_jetson_thor()
dil = DeepImitativeLearning.for_jetson_thor()

# In control loop:
result = dil.execute(
    instruction="pick up the red cup",
    images=camera_images,
    proprio=robot_state,
)

# Safety filtering (CBF + RTA)
safe_result = executor.execute(
    learned_action=result.actions[0],
    state=robot.get_state(),
)

robot.send_command(safe_result.action)  # GUARANTEED safe
```

---

## Compositional Training (v0.8.0)

**Solves training compositionality** using existing infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPOSITIONAL TRAINING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CONTINUOUS DATA CAPTURE (Already Have)                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                         │
│  │ ONVIF Cameras│   │ MANUS Gloves │   │Robot Proprio │                         │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                         │
│         └──────────────────┼──────────────────┘                                  │
│                            ▼                                                     │
│  META AI SEGMENTATION (Already Have)                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                         │
│  │     SAM3     │   │    DINOv3    │   │   V-JEPA2    │                         │
│  │   Objects    │   │   Features   │   │   Temporal   │                         │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                         │
│         └──────────────────┼──────────────────┘                                  │
│                            ▼                                                     │
│  SKILL DISCOVERY (New)                                                           │
│  ├─ Segment demos at object-interaction boundaries (SAM3)                        │
│  ├─ Cluster segments by DINOv3 features → primitives                             │
│  └─ Learn temporal structure with V-JEPA2 → hierarchy                            │
│                            ▼                                                     │
│  COMPOSITIONAL TRAINING (New)                                                    │
│  ├─ Train primitive policies on discovered segments                              │
│  ├─ Learn composition operators (sequence, conditional)                          │
│  └─ Enable novel task composition NOT seen in training                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

| Approach | Training Data | Novel Tasks | Transfer |
|----------|---------------|-------------|----------|
| End-to-end VLA | O(n²) combinations | Fails | None |
| **Compositional** | O(n) primitives | **Composes** | **Yes** |

### Usage

```python
from src.training.compositional import CompositionalTrainer

# Initialize with existing infrastructure
trainer = CompositionalTrainer.for_jetson_thor()
trainer.initialize()

# Continuous learning from human trainers
for demo in trainer.data_stream.stream_demos():
    trainer.add_demo(demo)

# Run full training pipeline
result = trainer.train()

# Compose novel task NOT seen in training!
task = trainer.compose_novel_task("stack blocks by color while avoiding the cat")
trainer.execute_composed_task(task, robot_state)
```

---

## Timing Contract

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TIMING CONTRACT (HARD REQUIREMENTS)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TIER 0: SAFETY KERNEL (1kHz, <500μs, CPU-ONLY)                                 │
│  ├─ CBF constraint checking and QP solving                                      │
│  ├─ RTA switching logic                                                         │
│  ├─ Hardware watchdog                                                           │
│  └─ GUARANTEE: Safe action within deadline                                      │
│                                                                                  │
│  TIER 1: CONTROL (100Hz, <5ms, CPU + cached inference)                          │
│  ├─ Runtime property monitoring (LTL)                                           │
│  ├─ Composition verification                                                    │
│  └─ Action interpolation                                                        │
│                                                                                  │
│  TIER 2: PERCEPTION (10-30Hz, <100ms, GPU)                                      │
│  ├─ Pi0.5 VLA inference                                                         │
│  ├─ Diffusion trajectory planning                                               │
│  ├─ DINOv3 / SAM3 / V-JEPA 2                                                   │
│  └─ RIP uncertainty estimation                                                  │
│                                                                                  │
│  TIER 3: CLOUD (Async, seconds-to-hours latency)                                │
│  ├─ Encrypted telemetry upload                                                  │
│  ├─ Skill sync and updates                                                      │
│  └─ MOAI FHE processing (offline)                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
src/
├── execution/                   # Main Execution Layer (NO SKILL BLENDING)
│   ├── dynamical_executor.py    # Simplified executor using Deep IL
│   └── task_decomposer.py       # Long-horizon task decomposition
│
├── training/                    # Compositional Training
│   └── compositional/           # Skill Discovery + Hierarchy Learning
│       ├── skill_discovery.py   # SAM3+DINOv3+V-JEPA2 segmentation
│       ├── hierarchical_imitation.py  # Build skill hierarchies
│       └── compositional_trainer.py   # End-to-end training pipeline
│
├── safety/                      # P0: Deterministic Safety
│   ├── cbf/                     # Control Barrier Functions
│   │   ├── barriers.py          # Collision, joint, velocity, force barriers
│   │   └── filter.py            # QP-based CBF filter
│   ├── rta/                     # Runtime Assurance
│   │   ├── baseline.py          # Verified baseline controllers
│   │   └── simplex.py           # Simplex switching
│   ├── runtime_monitor/         # Temporal property checking
│   │   └── monitor.py           # LTL/MTL monitoring
│   └── executor.py              # SafePolicyExecutor integration
│
├── composition/                 # Sequential Task Chaining (NOT blending)
│   ├── contracts.py             # Task contracts
│   ├── verifier.py              # Composition verification
│   └── library.py               # Task library
│
├── spatial_intelligence/        # Deep Imitative Learning
│   ├── pi0/                     # Pi0.5 VLA (Physical Intelligence)
│   ├── planning/                # Diffusion Planner
│   ├── safety/                  # RIP (informational)
│   ├── recovery/                # POIR recovery
│   └── deep_imitative_learning.py
│
├── meta_ai/                     # Meta AI Models (used by training)
│   ├── dinov3.py                # DINOv3 features → skill clustering
│   ├── sam3.py                  # SAM3 segmentation → object boundaries
│   └── vjepa2.py                # V-JEPA 2 → temporal structure
│
├── platform/                    # Hardware Platform
│   ├── edge/
│   │   ├── onvif_cameras.py     # Video capture for training
│   │   └── manus_sdk.py         # Glove data for training
│   └── jetson_thor.py           # Jetson Thor optimization
│
├── core/                        # Core Utilities
│   └── skill_blender.py         # DEPRECATED - use DynamicalExecutor
│
└── product/                     # Application Layer
    ├── task_api.py              # Task execution API
    └── semantic_planner.py      # Natural language planning
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform
pip install -e .

# 2. Run tests
pytest tests/ -v

# 3. Basic usage
python -c "
from src.safety import SafePolicyExecutor
from src.spatial_intelligence import DeepImitativeLearning

# Check installations
executor = SafePolicyExecutor.minimal()
print('Safety stack ready')
"
```

---

## Hardware Requirements

| Component | Specification | Used In |
|-----------|--------------|---------|
| **CPU** | ARM Cortex-A78AE (12-core) | Safety Loop (1kHz) |
| **GPU** | NVIDIA Blackwell | Policy Loop (10Hz) |
| **Memory** | 128GB LPDDR5X | Model loading |
| **Platform** | NVIDIA Jetson Thor | Primary target |

---

## Version History

| Version | Highlights |
|---------|------------|
| **0.8.0** | Deterministic safety (CBF + RTA), compositional skills, Deep Imitative Learning |
| 0.7.1 | Timing contract, threat model |
| 0.7.0 | ROS 2 integration, C++ safety node |
| 0.6.0 | Frontend modernization |
| 0.5.0 | Jetson Thor support |
| 0.4.0 | Meta AI models (DINOv2, SAM2, V-JEPA) |

---

## Architecture Comparison

| Aspect | v0.7.x (Probabilistic) | v0.8.0 (Deterministic) |
|--------|-------------------------|-------------------------|
| Safety | RIP epistemic uncertainty | CBF hard constraints |
| Recovery | POIR reactive | RTA verified baseline |
| Composition | Monolithic skills | Verified contracts |
| Guarantee | "Probably safe" | **h(x) ≥ 0 always** |

---

## License

Proprietary - Dynamical.ai © 2024-2025

---

## References

- [Control Barrier Functions](https://arxiv.org/abs/1903.11199) - Ames et al.
- [Runtime Assurance](https://ntrs.nasa.gov/citations/20180002983) - NASA Simplex
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Chi et al.
- [Pi0.5](https://www.physicalintelligence.company/blog/pi0) - Physical Intelligence

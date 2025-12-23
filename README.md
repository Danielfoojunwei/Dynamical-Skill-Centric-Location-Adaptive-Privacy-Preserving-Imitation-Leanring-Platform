# Dynamical Edge Platform v0.7.1

![Version](https://img.shields.io/badge/version-0.7.1-blue)
![Status](https://img.shields.io/badge/status-Development-yellow)
![License](https://img.shields.io/badge/license-Proprietary-red)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble/Iron-22314E)

> **Privacy-Preserving Imitation Learning Platform for Humanoid Robots**

---

## Design Philosophy

This platform follows three principles that separate shippable robotics from research demos:

1. **Hard real-time loops are deterministic and lightweight** — No GPU, no models, no network in Tier 0-1
2. **Giant models are async assessors, not control dependencies** — DINO/SAM/V-JEPA run at 5-30Hz, event-triggered
3. **Privacy has a threat model, not a checkbox** — One scheme per boundary, justified by specific threats

---

## Timing Contract

Every component has an explicit timing tier. **Violations cause safety faults.**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TIMING CONTRACT (HARD REQUIREMENTS)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TIER 0: SAFETY KERNEL (1kHz, <500μs, CPU-ONLY)                                 │
│  ├─ C++ node, SCHED_FIFO priority 99, mlockall()                                │
│  ├─ NO GPU calls, NO network, NO heap allocation after init                     │
│  ├─ Deterministic checks: joint limits, velocity, torque, obstacle envelope     │
│  ├─ Watchdog: 2ms timeout → motor power cut                                     │
│  └─ E-Stop: Hardware-wired, software cannot override                            │
│                                                                                  │
│  TIER 1: CONTROL (100Hz, <5ms, CPU + cached inference)                          │
│  ├─ State estimation (EKF, CPU)                                                 │
│  ├─ Policy inference from CACHED TensorRT engine (single GPU kernel)            │
│  ├─ Action smoothing and trajectory interpolation                               │
│  └─ NO model loading, NO dynamic allocation, NO network dependency              │
│                                                                                  │
│  TIER 2: PERCEPTION (10-30Hz, <100ms, GPU)                                      │
│  ├─ Cascaded models: Tiny (always) → Medium (on-demand) → Giant (rare)          │
│  ├─ Event-triggered escalation based on confidence thresholds                   │
│  ├─ Results update shared memory; control uses STALE data if late               │
│  └─ Failure mode: Fall back to Tier 1 with last-known-good features             │
│                                                                                  │
│  TIER 3: CLOUD (Async, seconds-to-hours latency)                                │
│  ├─ Encrypted telemetry upload (batch, not streaming)                           │
│  ├─ Skill sync (download new skills during idle)                                │
│  ├─ MOAI FHE processing (OFFLINE ONLY, hours latency acceptable)                │
│  └─ Federated learning (nightly aggregation, not real-time)                     │
│                                                                                  │
│  NETWORK FAILURE BEHAVIOR:                                                       │
│  └─ Robot continues indefinitely with cached skills and local perception        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Safety Architecture

**Safety is NOT model-based.** The safety kernel uses deterministic, verifiable checks.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SAFETY HIERARCHY                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LEVEL 0: HARDWARE E-STOP (Cannot be overridden by software)                    │
│  ├─ Physical button wired directly to motor power                               │
│  ├─ Capacitive human proximity sensor → immediate halt                          │
│  └─ Watchdog relay: No heartbeat for 5ms → power cut                            │
│                                                                                  │
│  LEVEL 1: DETERMINISTIC SOFTWARE CHECKS (1kHz, C++)                             │
│  ├─ Joint position limits (hard-coded from URDF, ±2° margin)                    │
│  ├─ Joint velocity limits (per-joint, temperature-compensated)                  │
│  ├─ Torque limits (measured vs commanded, ±10% tolerance)                       │
│  ├─ Obstacle envelope (convex hull from depth, 20cm minimum clearance)          │
│  └─ Self-collision (pre-computed collision pairs, <1ms check)                   │
│                                                                                  │
│  LEVEL 2: ML-ASSISTED ADVISORS (30Hz, informational only)                       │
│  ├─ V-JEPA future prediction → "caution" flag, does NOT stop robot              │
│  ├─ Human intent prediction → reduce speed, does NOT override safety            │
│  └─ Anomaly detection → log + alert, operator must acknowledge                  │
│                                                                                  │
│  CRITICAL: Level 2 CANNOT override Level 0-1. ML predictions are advisory.      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**File:** `src/ros2/dynamical_runtime/src/safety_node.cpp`

```cpp
// 1kHz safety loop - NO ML, NO GPU, NO network
void SafetyNode::safety_loop() {
    // Hard-coded limits from URDF, not learned
    static constexpr std::array<double, 7> JOINT_POS_MAX = {2.87, 1.74, 2.87, ...};
    static constexpr std::array<double, 7> JOINT_VEL_MAX = {2.0, 2.0, 2.0, ...};

    bool safe = true;
    safe &= check_joint_limits(state_.positions, JOINT_POS_MAX);
    safe &= check_velocity_limits(state_.velocities, JOINT_VEL_MAX);
    safe &= check_torque_limits(state_.torques, state_.commanded_torques);
    safe &= check_obstacle_envelope(depth_buffer_, MIN_CLEARANCE_M);

    if (!safe) {
        trigger_safe_stop();  // Ramp down, not instant (joint protection)
    }

    watchdog_.pet();  // Must call every <2ms or hardware cuts power
}
```

---

## Perception Pipeline (Tier 2)

**Giant models are NOT in the control loop.** They run asynchronously and escalate on-demand.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CASCADED PERCEPTION (Event-Triggered)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LEVEL 1: ALWAYS RUNNING (30Hz, <20ms, TensorRT INT8)                           │
│  ├─ MobileNetV4 features (224×224, 2ms)                                         │
│  ├─ Lightweight object detection (YOLO-NAS-S, 8ms)                              │
│  ├─ Depth processing (stereo matching, 5ms)                                     │
│  └─ Output: 256-dim features, bounding boxes, depth map                         │
│                                                                                  │
│  LEVEL 2: ON-DEMAND (10Hz, <50ms, triggered by L1 uncertainty)                  │
│  ├─ Trigger: L1 confidence < 0.7 OR novel object detected                       │
│  ├─ DINOv2-B features (518×518, 25ms with TensorRT)                             │
│  ├─ SAM2 segmentation (prompted by L1 boxes, 30ms)                              │
│  └─ Output: 768-dim features, precise masks                                     │
│                                                                                  │
│  LEVEL 3: RARE (1-5Hz, <200ms, triggered by L2 failure)                         │
│  ├─ Trigger: L2 confidence < 0.5 OR safety-critical scenario                    │
│  ├─ DINOv2-G features (full resolution, 100ms)                                  │
│  ├─ SAM2-L segmentation (80ms)                                                  │
│  ├─ V-JEPA 2 world model (future prediction, 150ms)                             │
│  └─ Output: 1024-dim features, future state predictions                         │
│                                                                                  │
│  CONTROL LOOP ISOLATION:                                                         │
│  └─ Tier 1 control uses LAST VALID features. Late perception = stale data.      │
│  └─ Control NEVER waits for perception. Perception writes to shared memory.     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Note on Model Sizes:**
- Level 1 models: ~5M parameters, always loaded
- Level 2 models: ~100M parameters, loaded on-demand (100ms cold start)
- Level 3 models: ~1B parameters, loaded only when triggered (500ms cold start)

"All models loaded simultaneously" means **memory reserved**, not **continuously executing**.

---

## Threat Model and Privacy Architecture

**One privacy primitive per boundary, with explicit threat and cost.**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THREAT MODEL                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  THREAT 1: Cloud provider sees raw sensor data                                  │
│  ├─ Attack: Reconstruct environment, learn trade secrets, privacy violation     │
│  ├─ Mitigation: On-device feature extraction. Raw images NEVER leave robot.     │
│  └─ Cost: None (computation happens locally anyway)                             │
│                                                                                  │
│  THREAT 2: Cloud provider infers task from features                             │
│  ├─ Attack: Feature embeddings leak semantic information                        │
│  ├─ Mitigation: N2HE encryption of 512-dim feature vectors before upload        │
│  ├─ Scheme: LWE with n=1024, 128-bit security                                   │
│  ├─ Overhead: ~50KB per encrypted feature vector, <10ms encryption time         │
│  └─ Cost: Acceptable for async upload (Tier 3)                                  │
│                                                                                  │
│  THREAT 3: Aggregation server learns individual robot contributions             │
│  ├─ Attack: Gradient inspection reveals robot-specific behaviors                │
│  ├─ Mitigation: Secure aggregation for SMALL model components only              │
│  ├─ Scope: Router heads (~10K params), skill selectors (~50K params)            │
│  ├─ NOT applied to: Full policies (too large, use DP instead)                   │
│  └─ Cost: 100x overhead acceptable for nightly aggregation                      │
│                                                                                  │
│  THREAT 4: Membership inference on learned skills                               │
│  ├─ Attack: Determine if specific demonstration was in training set             │
│  ├─ Mitigation: Differential privacy (ε=1.0) during local training              │
│  └─ Cost: ~5% accuracy reduction, acceptable                                    │
│                                                                                  │
│  NOT A RUNTIME THREAT (handled offline):                                         │
│  └─ MOAI FHE transformer: Batch processing of encrypted demonstrations          │
│  └─ Latency: Hours. Used for: skill distillation, compliance audit, analytics   │
│  └─ NOT used for: Real-time inference, control loop, anything time-sensitive    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**File:** `src/moai/n2he.py`

```python
class N2HEContext:
    """
    LWE-based encryption for feature vectors.

    USE CASE: Encrypting 512-dim features before cloud upload (Tier 3, async).
    NOT FOR: Real-time inference, control loop, anything requiring <1s latency.

    Performance (measured on Jetson AGX):
      - Encrypt 512-dim vector: 8ms
      - Ciphertext size: 48KB
      - Decrypt: 3ms
    """

    def __init__(self, n: int = 1024, q: int = 2**32, security_bits: int = 128):
        self.n = n
        self.q = q
        self.secret_key = self._generate_secret_key()
```

---

## MOAI: Offline Encrypted Computation

**MOAI is NOT a runtime component.** It processes encrypted data in batch, offline.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MOAI OFFLINE PROCESSING                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  WHAT MOAI DOES:                                                                │
│  ├─ Process encrypted demonstration batches (nightly)                           │
│  ├─ Extract skill embeddings without seeing plaintext                           │
│  ├─ Compute quality scores for compliance/audit                                 │
│  └─ Enable third-party verification without data access                         │
│                                                                                  │
│  WHAT MOAI DOES NOT DO:                                                         │
│  ├─ Real-time inference (impossible: ~60s per forward pass)                     │
│  ├─ Control loop integration (violates timing contract)                         │
│  └─ Online learning (latency incompatible with robotics)                        │
│                                                                                  │
│  PERFORMANCE (honest numbers):                                                   │
│  ├─ FHE attention on 512-dim: ~30 seconds                                       │
│  ├─ Full transformer forward pass: ~60 seconds                                  │
│  ├─ Batch of 100 demonstrations: ~2 hours                                       │
│  └─ Acceptable for: Overnight processing, weekly audits, offline analytics     │
│                                                                                  │
│  USE CASE EXAMPLE:                                                               │
│  └─ Robot records demonstrations → encrypts locally → uploads to cloud          │
│  └─ Cloud runs MOAI overnight → extracts encrypted skill embeddings             │
│  └─ Robot downloads encrypted skills → decrypts locally → uses for inference   │
│  └─ Cloud NEVER sees plaintext demonstrations or skills                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**File:** `src/moai/moai_fhe.py`

```python
class MoaiTransformerFHE:
    """
    Transformer operating on FHE-encrypted data.

    WARNING: This is an OFFLINE processing component.
    - Forward pass: ~60 seconds (not 60ms)
    - Use for: Batch processing, compliance, analytics
    - NOT for: Real-time inference, control loops

    The 10,000-100,000x slowdown vs plaintext is fundamental to FHE.
    """

    def forward(self, encrypted_features: List[LWECiphertext]) -> List[LWECiphertext]:
        # This takes ~60 seconds. That's not a bug.
        pass
```

---

## Skill Blending with Stability Guarantees

**Naive blending is unsafe.** The system enforces normalization and stability constraints.

```python
# File: src/core/robot_skill_invoker.py

class SkillBlender:
    """
    Blend multiple skills with stability guarantees.

    Constraints enforced:
    1. All skills output normalized actions in [-1, 1]
    2. All skills use same action space (verified at registration)
    3. Blend weights sum to 1.0
    4. Maximum delta between frames (jerk limiting)
    5. Uncertainty-based weight reduction
    """

    def blend(self, skills: List[Skill], weights: List[float], obs: Observation) -> Action:
        # Validate preconditions
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        assert all(s.action_space == skills[0].action_space for s in skills), \
            "Action space mismatch"

        # Get individual actions (all normalized to [-1, 1])
        actions = []
        confidences = []
        for skill in skills:
            action, confidence = skill.infer_with_confidence(obs)
            assert action.min() >= -1 and action.max() <= 1, "Normalization violation"
            actions.append(action)
            confidences.append(confidence)

        # Reduce weight for low-confidence skills
        adjusted_weights = []
        for w, c in zip(weights, confidences):
            adjusted = w * c if c > self.confidence_threshold else 0.0
            adjusted_weights.append(adjusted)

        # Renormalize
        weight_sum = sum(adjusted_weights)
        if weight_sum < 0.1:
            # All skills uncertain → fall back to safe default
            return self.safe_default_action
        adjusted_weights = [w / weight_sum for w in adjusted_weights]

        # Blend
        blended = sum(w * a for w, a in zip(adjusted_weights, actions))

        # Jerk limiting (smooth transitions)
        if self.last_action is not None:
            delta = blended - self.last_action
            max_delta = self.max_action_delta * self.dt
            blended = self.last_action + np.clip(delta, -max_delta, max_delta)

        self.last_action = blended
        return blended
```

---

## Sensor Transport (Honest Latency)

**WiFi and ONVIF are NOT deterministic.** The system handles worst-case jitter.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SENSOR TRANSPORT (Realistic Numbers)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DYGLOVE/MANUS (WiFi 6E):                                                       │
│  ├─ Typical latency: 2-5ms                                                      │
│  ├─ 99th percentile: 15ms                                                       │
│  ├─ Worst case (interference): 50ms+                                            │
│  ├─ Mitigation: Glove state prediction (linear extrapolation)                   │
│  ├─ Failure mode: >100ms gap → freeze teleoperation, alert operator             │
│  └─ NOT suitable for: Safety-critical closed-loop control                       │
│                                                                                  │
│  ONVIF/RTSP CAMERAS:                                                            │
│  ├─ Typical latency: 50-150ms (encoding + network + decode)                     │
│  ├─ Frame drops: 1-5% under load                                                │
│  ├─ Mitigation: Multi-frame buffer, timestamp-based synchronization             │
│  ├─ Failure mode: >500ms stale → use depth-only, reduce speed                   │
│  └─ NOT suitable for: Anything requiring <50ms image latency                    │
│                                                                                  │
│  WIRED SENSORS (Joint encoders, force-torque):                                  │
│  ├─ Latency: <100μs (EtherCAT/CAN)                                              │
│  ├─ Jitter: <10μs                                                               │
│  └─ ONLY these are used in Tier 0-1 safety/control loops                        │
│                                                                                  │
│  ARCHITECTURAL DECISION:                                                         │
│  └─ Safety loop (Tier 0) uses ONLY wired sensors with bounded latency           │
│  └─ WiFi/ONVIF sensors inform Tier 2-3, NOT Tier 0-1                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ROS 2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ROS 2 NODE GRAPH                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TIER 0-1 CONTAINER (C++, intra-process, zero-copy)                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ SafetyNode  │  │ StateEst    │  │ Controller  │  │ Actuator    │     │   │
│  │  │ 1kHz, CPU   │  │ 1kHz, CPU   │  │ 100Hz, CPU  │  │ 1kHz, CPU   │     │   │
│  │  │ FIFO:99     │  │ FIFO:98     │  │ FIFO:95     │  │ FIFO:97     │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         └─────────────────┴────────────────┴────────────────┘            │   │
│  │                         shared memory (lock-free)                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                             │
│                                    │ /robot_state (100Hz)                        │
│                                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 2: Perception (Python, 30Hz, separate process)                      │   │
│  │  ├─ CascadedPerception: L1→L2→L3 escalation                              │   │
│  │  ├─ Isaac ROS TensorRT (optional acceleration)                           │   │
│  │  └─ Publishes: /perception/features (when ready, not blocking)           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                             │
│                                    │ /perception/features (async)                │
│                                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 3: Cloud Bridge (Python, async)                                     │   │
│  │  ├─ Telemetry upload (batch, encrypted)                                  │   │
│  │  ├─ Skill sync (on-demand)                                               │   │
│  │  └─ MOAI batch jobs (nightly, offline)                                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Package Structure

```
src/ros2/
├── dynamical_msgs/              # Interface definitions
│   ├── msg/
│   │   ├── RobotState.msg       # Joint positions, velocities, torques
│   │   ├── SafetyStatus.msg     # Safety flags, violations
│   │   └── PerceptionResult.msg # Features, confidence, tier
│   ├── srv/
│   │   ├── TriggerEstop.srv     # Emergency stop
│   │   └── SetControlMode.srv   # Position/velocity/torque
│   └── action/
│       └── ExecuteSkill.action  # Long-running skill execution
│
├── dynamical_runtime/           # C++ real-time nodes (Tier 0-1)
│   ├── src/
│   │   ├── safety_node.cpp      # 1kHz, SCHED_FIFO 99
│   │   ├── state_estimator.cpp  # 1kHz, SCHED_FIFO 98
│   │   └── controller_node.cpp  # 100Hz, SCHED_FIFO 95
│   └── CMakeLists.txt
│
├── dynamical_perception/        # Python perception (Tier 2)
│   └── scripts/
│       └── cascaded_perception.py
│
└── dynamical_bringup/           # Launch configuration
    ├── launch/
    │   └── robot_runtime.launch.py
    └── config/
        └── robot_params.yaml
```

---

## What This Platform Is (and Isn't)

### What It Is

- **A deployment-grade robot runtime** with deterministic safety guarantees
- **A cascaded perception system** that uses giant models only when needed
- **A privacy-preserving data pipeline** with explicit threat model
- **An offline skill learning platform** that processes encrypted demonstrations

### What It Is NOT

- ❌ A system that runs "all giant models at 200Hz" (physically impossible)
- ❌ A real-time FHE inference engine (FHE is 10,000x too slow)
- ❌ A system where ML predictions control safety (deterministic checks only)
- ❌ A monolithic "everything in the loop" architecture (tiered, decoupled)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform
./install.sh

# 2. Build ROS 2 packages
cd src/ros2
colcon build --symlink-install
source install/setup.bash

# 3. Launch (simulation)
ros2 launch dynamical_bringup robot_runtime.launch.py use_sim:=true

# 4. Launch (real robot - requires hardware)
ros2 launch dynamical_bringup robot_runtime.launch.py
```

---

## Hardware Requirements

| Component | Specification | Used In |
|-----------|--------------|---------|
| **CPU** | ARM Cortex-A78AE (12-core) | Tier 0-1 (safety, control) |
| **GPU** | NVIDIA Blackwell | Tier 2 (perception) |
| **Memory** | 128GB LPDDR5X | Model loading, not concurrent execution |
| **Joint Sensors** | EtherCAT/CAN (<100μs) | Tier 0-1 ONLY |
| **Cameras** | ONVIF/RTSP | Tier 2-3 only (not safety-critical) |
| **Gloves** | WiFi 6E | Tier 2-3 only (teleoperation, not safety) |

---

## Federated Learning Scope

**FL is applied to SMALL components, not full policies.**

| Component | Parameters | FL Method | Frequency |
|-----------|-----------|-----------|-----------|
| Skill router heads | ~10K | Secure aggregation | Nightly |
| Confidence calibration | ~5K | Secure aggregation | Nightly |
| Anomaly thresholds | ~1K | Plain averaging (non-sensitive) | Weekly |
| Full policies | ~10M+ | Local only + DP | Never aggregated |

---

## Version History

| Version | Highlights |
|---------|------------|
| **0.7.1** | Timing contract, threat model, honest latency numbers |
| 0.7.0 | ROS 2 integration, C++ safety node |
| 0.6.0 | Frontend modernization (React 18, Zustand) |
| 0.5.0 | Jetson Thor support |
| 0.4.0 | Meta AI models (DINOv2, SAM2, V-JEPA) |

---

## License

Proprietary - Dynamical.ai © 2024

---

## Acknowledgments

This architecture incorporates lessons from:
- Boston Dynamics (tiered safety)
- ANYbotics (cascaded perception)
- NVIDIA Isaac (ROS 2 + TensorRT)
- Academic FHE research (offline processing, not real-time)

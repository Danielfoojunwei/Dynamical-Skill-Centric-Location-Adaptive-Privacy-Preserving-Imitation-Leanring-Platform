# Deployment-Grade Architecture Specification

## v0.7.0 - Robot-Resident Runtime Design

> Based on real deployment patterns from ANYmal, Spot CORE I/O, and NVIDIA Isaac ROS

---

## Executive Summary

Real robot deployments follow a consistent pattern:

| Tier | Location | Workload | Rate | Determinism |
|------|----------|----------|------|-------------|
| **Control** | Robot CPU | Safety, state estimation, actuator commands | 1kHz | Hard real-time |
| **Perception** | Robot GPU (Jetson) | Vision, learned policies | 10-200 Hz | Soft real-time |
| **Intelligence** | Edge/Cloud | Planning, training, analytics | Async | Best effort |

**Key Insight**: Dynamical must be **robot-resident first**, with edge/cloud as an **optional enhancement**, not a dependency.

---

## The Three-Tier Compute Model

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ROBOT (On-Board)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  TIER 1: ROBOT CPU (Deterministic, Hard Real-Time)                          ││
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ││
│  │                                                                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     ││
│  │  │ Safety       │  │ State        │  │ Actuator     │  │ Watchdog     │     ││
│  │  │ Shield       │  │ Estimation   │  │ Commands     │  │ + Heartbeat  │     ││
│  │  │ (1kHz)       │  │ (1kHz)       │  │ (1kHz)       │  │ (1kHz)       │     ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     ││
│  │                                                                              ││
│  │  Characteristics:                                                            ││
│  │  • PREEMPT_RT Linux or real-time OS                                         ││
│  │  • No GPU dependencies in critical path                                     ││
│  │  • Can run without network connectivity                                     ││
│  │  • Latency budget: <1ms guaranteed                                          ││
│  │                                                                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                      │                                           │
│                                      │ Shared Memory / Zero-Copy                 │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  TIER 2: ROBOT GPU PAYLOAD (Soft Real-Time, Bounded)                        ││
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ││
│  │                                                                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     ││
│  │  │ Perception   │  │ Policy       │  │ Local        │  │ Feature      │     ││
│  │  │ Pipeline     │  │ Inference    │  │ Planning     │  │ Cache        │     ││
│  │  │ (30 Hz)      │  │ (100 Hz)     │  │ (10 Hz)      │  │ (async)      │     ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     ││
│  │                                                                              ││
│  │  Characteristics:                                                            ││
│  │  • Jetson Orin/Thor or similar GPU payload                                  ││
│  │  • TensorRT-compiled models                                                 ││
│  │  • Bounded execution time (soft real-time)                                  ││
│  │  • Latency budget: <50ms typical, <100ms worst-case                         ││
│  │                                                                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ Network (Optional, Async)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: EDGE / CLOUD (Best Effort, Async)                                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Heavy        │  │ Training &   │  │ Global       │  │ Long-Horizon │         │
│  │ Segmentation │  │ FL Aggreg.   │  │ Analytics    │  │ Planning     │         │
│  │ (1-5 Hz)     │  │ (batch)      │  │ (async)      │  │ (on-demand)  │         │
│  │              │  │              │  │              │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                                  │
│  Characteristics:                                                                │
│  • Robot functions WITHOUT this tier (graceful degradation)                      │
│  • Used for capability enhancement, not core operation                           │
│  • Latency: 100ms - 10s acceptable                                              │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Robot Runtime Agent (First-Class Component)

The **Robot Runtime Agent** is the mandatory on-robot component that owns:

```python
# src/robot_runtime/agent.py

class RobotRuntimeAgent:
    """
    First-class component that runs ON THE ROBOT.

    Owns:
    - Hard scheduling (Tier 1-2)
    - Actuator interface
    - Skill execution state machine
    - Local feature cache + telemetry
    - Safety authority (can ALWAYS stop the robot)

    Everything else talks to it through stable APIs.
    """

    def __init__(self, robot_config: RobotConfig):
        # Tier 1: CPU-bound, deterministic
        self.safety_shield = SafetyShield(rate_hz=1000)
        self.state_estimator = StateEstimator(rate_hz=1000)
        self.actuator_interface = ActuatorInterface(rate_hz=1000)
        self.watchdog = Watchdog(timeout_ms=5)

        # Tier 2: GPU-accelerated, bounded
        self.perception_pipeline = PerceptionPipeline(rate_hz=30)
        self.policy_executor = PolicyExecutor(rate_hz=100)
        self.local_planner = LocalPlanner(rate_hz=10)

        # State machine
        self.execution_state = ExecutionStateMachine()

        # Local caches (survive network outages)
        self.skill_cache = SkillCache(max_skills=100)
        self.feature_cache = FeatureCache(ttl_seconds=60)

        # Telemetry buffer (async upload when connected)
        self.telemetry_buffer = TelemetryBuffer(max_size_mb=500)

    def run(self):
        """Main loop with priority scheduling."""
        # Tier 1 runs at highest priority (SCHED_FIFO)
        self.tier1_thread = Thread(
            target=self._tier1_loop,
            priority=99,  # Highest RT priority
            scheduler=SCHED_FIFO
        )

        # Tier 2 runs at lower priority
        self.tier2_thread = Thread(
            target=self._tier2_loop,
            priority=50,
            scheduler=SCHED_FIFO
        )

        self.tier1_thread.start()
        self.tier2_thread.start()

    def _tier1_loop(self):
        """1kHz deterministic loop - NEVER misses a deadline."""
        rate = Rate(1000)  # 1kHz
        while self.running:
            # 1. Read sensors
            state = self.state_estimator.update()

            # 2. Safety check (can override everything)
            safe, override_action = self.safety_shield.check(state)

            # 3. Get action (from Tier 2 or fallback)
            if safe:
                action = self.execution_state.get_current_action()
            else:
                action = override_action  # Safety takes over

            # 4. Command actuators
            self.actuator_interface.send(action)

            # 5. Heartbeat
            self.watchdog.pet()

            rate.sleep()

    def _tier2_loop(self):
        """10-100Hz perception and policy loop."""
        rate = Rate(100)  # 100Hz base rate
        while self.running:
            # Perception at 30Hz (runs every 3rd iteration)
            if self.tick % 3 == 0:
                self.perception_pipeline.update()

            # Policy inference at 100Hz
            observation = self.get_observation()
            action = self.policy_executor.infer(observation)
            self.execution_state.set_action(action)

            # Local planning at 10Hz (runs every 10th iteration)
            if self.tick % 10 == 0:
                self.local_planner.replan()

            self.tick += 1
            rate.sleep()
```

---

## Module Assignment: On-Robot vs Off-Robot

### TIER 1: Robot CPU (1kHz, Deterministic)

| Module | File | Hz | Latency | Notes |
|--------|------|----|---------| ------|
| **Safety Shield** | `robot_runtime/safety_shield.py` | 1000 | <1ms | V-JEPA backup runs here (small model) |
| **State Estimator** | `robot_runtime/state_estimator.py` | 1000 | <1ms | EKF/UKF, joint encoders |
| **Actuator Interface** | `robot_runtime/actuator_interface.py` | 1000 | <1ms | Direct motor commands |
| **Watchdog** | `robot_runtime/watchdog.py` | 1000 | <1ms | Hardware watchdog integration |
| **Execution State Machine** | `robot_runtime/execution_state.py` | 1000 | <1ms | Skill state transitions |

**Characteristics:**
- Pure C++ or Rust (no Python in critical path)
- PREEMPT_RT Linux kernel
- No heap allocations in hot path
- No network I/O
- Can run with GPU entirely offline

### TIER 2: Robot GPU Payload (10-100Hz, Bounded)

| Module | File | Hz | Latency | Notes |
|--------|------|----|---------| ------|
| **Perception Pipeline** | `robot_runtime/perception.py` | 30 | <33ms | DINOv3-small, SAM-small (TensorRT) |
| **Policy Executor** | `robot_runtime/policy_executor.py` | 100 | <10ms | Pi0-small or ACT (TensorRT) |
| **Local Planner** | `robot_runtime/local_planner.py` | 10 | <100ms | Reactive obstacle avoidance |
| **Skill Cache** | `robot_runtime/skill_cache.py` | async | - | 100 skills pre-loaded |
| **Feature Cache** | `robot_runtime/feature_cache.py` | async | - | Scene features, object detections |

**Characteristics:**
- TensorRT-compiled models
- Bounded worst-case execution time
- Graceful degradation if models timeout
- Can run without cloud connectivity

### TIER 3: Edge/Cloud (Async, Optional)

| Module | File | Hz | Latency | Notes |
|--------|------|----|---------| ------|
| **Heavy Perception** | `platform/cloud/heavy_perception.py` | 1-5 | 200ms-2s | DINOv3-Giant, SAM3-Huge |
| **World Model** | `platform/cloud/world_model.py` | 1 | 500ms-5s | V-JEPA 2 Giant |
| **Task Decomposition** | `platform/cloud/task_decomposer.py` | on-demand | 1-10s | LLM planning |
| **MoE Skill Router** | `platform/cloud/moe_skill_router.py` | on-demand | 50-500ms | Skill selection |
| **FL Aggregation** | `platform/cloud/fl_aggregator.py` | batch | minutes | Gradient aggregation |
| **Training Pipeline** | `platform/cloud/training.py` | batch | hours | IL, MOAI training |
| **Analytics** | `platform/cloud/analytics.py` | async | - | Logging, dashboards |

**Characteristics:**
- Robot functions WITHOUT this tier
- Used for capability enhancement
- Network failure = graceful degradation, not failure
- Async communication patterns

---

## Cascaded Model Strategy

Real deployments don't run giant models at high rates. Use a cascade:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CASCADED PERCEPTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Level 1: ALWAYS (30 Hz, <10ms)                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  • Tiny object detector (YOLO-Nano, 2 TFLOPS)                           │   │
│   │  • Fast depth estimation (DepthAnything-S, 1 TFLOPS)                    │   │
│   │  • Motion detection (optical flow, 0.5 TFLOPS)                          │   │
│   │  → Detects: "something needs attention"                                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                           Trigger: uncertainty > 0.3                            │
│                                      ▼                                           │
│   Level 2: ON-DEMAND (10 Hz, <50ms)                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  • DINOv3-Small (20 TFLOPS)                                             │   │
│   │  • SAM3-Base (15 TFLOPS)                                                │   │
│   │  • Pose estimation (RTMPose-M, 5 TFLOPS)                                │   │
│   │  → Refines: object identity, segmentation, poses                        │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                    Trigger: novel object OR high-stakes action                  │
│                                      ▼                                           │
│   Level 3: RARE (1-5 Hz, <500ms, can be off-robot)                              │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  • DINOv3-Giant (120 TFLOPS)                                            │   │
│   │  • SAM3-Huge (200 TFLOPS)                                               │   │
│   │  • V-JEPA 2 Giant (330 TFLOPS)                                          │   │
│   │  → Full analysis: scene understanding, prediction, anomaly detection   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Trigger Conditions

```python
class CascadeTriggers:
    """When to escalate to heavier models."""

    # Level 1 → Level 2
    @staticmethod
    def needs_refinement(level1_output: Detection) -> bool:
        return (
            level1_output.confidence < 0.7 or      # Uncertain detection
            level1_output.is_new_object or         # Never seen before
            level1_output.velocity > 0.5 or        # Fast-moving
            level1_output.in_workspace             # In manipulation zone
        )

    # Level 2 → Level 3
    @staticmethod
    def needs_full_analysis(level2_output: SceneUnderstanding) -> bool:
        return (
            level2_output.has_human_nearby or      # Safety-critical
            level2_output.occlusion > 0.5 or       # Heavy occlusion
            level2_output.is_high_value_task or    # Important manipulation
            time_since_full_analysis() > 5.0       # Periodic refresh
        )
```

---

## Compute Budget (Realistic)

### On-Robot GPU (Jetson Orin/Thor)

**Available**: ~275 FP16 TFLOPS (or ~2070 FP4 TFLOPS)

| Component | FP16 TFLOPS | Rate | Notes |
|-----------|-------------|------|-------|
| **Level 1 Perception** | 3.5 | 30 Hz | Always running |
| **Level 2 Perception** | 40 | 10 Hz | On-demand |
| **Policy Inference** | 20 | 100 Hz | Pi0-small + ACT |
| **Safety Model (backup)** | 10 | 100 Hz | Small V-JEPA |
| **Local Planning** | 5 | 10 Hz | Reactive |
| **Buffer** | ~196 | - | For Level 3 bursts |
| **TOTAL** | 78.5 base | - | +196 burst capacity |

### Off-Robot (Edge/Cloud)

**No limit** - but constrained by network latency

| Component | Compute | Rate | Latency |
|-----------|---------|------|---------|
| **Level 3 Perception** | 650 TFLOPS | 1-5 Hz | 200-500ms |
| **Task Planning (LLM)** | Variable | On-demand | 1-10s |
| **Training** | Unlimited | Batch | Hours |

---

## Graceful Degradation

The robot MUST function at reduced capability when:

### Network Offline

```python
class OfflineMode:
    """Robot behavior when cloud is unreachable."""

    capabilities = {
        "skill_execution": True,      # From local cache
        "perception": "Level 1-2",    # On-robot only
        "planning": "Local only",     # No LLM decomposition
        "training": False,            # Disabled
        "telemetry": "Buffered",      # Upload when reconnected
    }

    behavior = {
        "new_skills": "Reject",           # Can't download new skills
        "complex_tasks": "Decompose locally or reject",
        "safety": "Unchanged",            # Never degraded
    }
```

### GPU Overloaded

```python
class DegradedPerceptionMode:
    """Fallback when GPU can't keep up."""

    # Drop to Level 1 only
    perception_rate = 30  # Hz
    perception_models = ["yolo-nano", "depth-small"]

    # Reduce policy rate
    policy_rate = 50  # Hz (from 100)

    # Increase safety margins
    safety_margin_multiplier = 1.5
```

### GPU Offline

```python
class CPUOnlyMode:
    """Emergency mode if GPU fails entirely."""

    # CPU-only perception
    perception = "Depth from stereo only"
    perception_rate = 10  # Hz

    # Pre-computed policy (lookup table)
    policy = "Fallback reactive controller"

    # Immediate stop if situation ambiguous
    behavior = "Conservative, request human takeover"
```

---

## ROS 2 Integration (If Applicable)

### Use Isaac ROS for Accelerated Perception

```yaml
# isaac_ros_perception.launch.py
perception_nodes:
  - isaac_ros_dnn_inference:  # TensorRT-accelerated
      model: dinov3_small.engine
      input_topic: /camera/image_raw
      output_topic: /perception/features

  - isaac_ros_depth_estimation:
      model: depth_anything_small.engine

  - isaac_ros_pose_estimation:
      model: rtmpose_m.engine
```

### Use Type Adaptation for Zero-Copy

```cpp
// Avoid message copies between nodes
#include <rclcpp/type_adaptation.hpp>

template<>
struct rclcpp::TypeAdapter<sensor_msgs::msg::Image, cv::cuda::GpuMat>
{
  using is_specialized = std::true_type;
  using custom_type = cv::cuda::GpuMat;
  using ros_message_type = sensor_msgs::msg::Image;

  static void convert_to_ros_message(const custom_type& source, ros_message_type& destination);
  static void convert_to_custom(const ros_message_type& source, custom_type& destination);
};
```

### Intra-Process Communication

```python
# Composition for zero-copy within process
ros2 run rclcpp_components component_container_mt \
  --ros-args -r __node:=perception_container

ros2 component load /perception_container \
  dynamical_perception PerceptionPipeline \
  --node-name perception \
  --param use_intra_process_comms:=true
```

---

## Migration Path from v0.6.0

### Phase 1: Extract Robot Runtime Agent

1. Create `src/robot_runtime/` package
2. Move safety, state estimation, actuator code
3. Add proper scheduling (SCHED_FIFO)
4. Test on actual robot hardware

### Phase 2: Implement Cascaded Models

1. Create small/medium model variants
2. Implement trigger logic
3. Benchmark latencies on target hardware
4. Tune cascade thresholds

### Phase 3: Make Cloud Optional

1. Add offline mode detection
2. Implement local skill cache persistence
3. Add telemetry buffering
4. Test with network disconnected

### Phase 4: ROS 2 Optimization (If Applicable)

1. Migrate to Isaac ROS perception nodes
2. Implement type adaptation
3. Use composition for intra-process
4. Benchmark improvements

---

## Validation Criteria

Before declaring deployment-ready:

- [ ] Robot runs for 1 hour with network disconnected
- [ ] Tier 1 loop never misses 1kHz deadline (measure with oscilloscope)
- [ ] GPU perception gracefully degrades under load
- [ ] Skill execution works from local cache
- [ ] Safety override works with GPU offline
- [ ] Telemetry buffers and uploads on reconnect

---

## References

1. ANYmal elevation mapping on Jetson: GPU perception at 20 Hz, CPU control deterministic
2. Spot CORE I/O: Edge compute payload for in-field processing
3. Isaac ROS: Modular accelerated perception for ROS 2
4. ROS 2 Type Adaptation: Zero-copy message passing

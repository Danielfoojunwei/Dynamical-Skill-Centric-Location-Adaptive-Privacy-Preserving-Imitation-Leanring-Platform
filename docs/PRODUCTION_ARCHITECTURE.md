# Dynamical Production Architecture

## Deterministic Safety-First Deployment Plan

This document defines a production-ready architecture for Dynamical that prioritizes:
1. **Deterministic safety guarantees** (not probabilistic)
2. **Compositionality with verified skill chaining**
3. **Robustness against distribution shift**
4. **Fleet-scale operations**

---

## Executive Summary

### Current State (What We Have)
```
Pi0.5 VLA → Diffusion Planner → RIP (probabilistic) → POIR (reactive)
```
**Problems:** Probabilistic safety, reactive recovery, no composition verification, no hard guarantees.

### Target State (Production Ready)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIERED CONTROL ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  1000 Hz │ SAFETY LOOP: CBF Filter + RTA Simplex + Hardware Limits          │
│──────────┼──────────────────────────────────────────────────────────────────│
│   100 Hz │ CONTROL LOOP: Runtime Monitors + Composition Verifier            │
│──────────┼──────────────────────────────────────────────────────────────────│
│    10 Hz │ POLICY LOOP: Pi0.5 + Diffusion + Tactile Fusion                  │
│──────────┼──────────────────────────────────────────────────────────────────│
│     1 Hz │ PLANNING LOOP: Skill Composer + Goal Planner                     │
│──────────┼──────────────────────────────────────────────────────────────────│
│   Async  │ OPS LOOP: Drift Detection + MLOps + TEE Training                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### Layer 0: Safety Loop (1000 Hz) - DETERMINISTIC

**Purpose:** Guarantee safety envelope. Never violate hard constraints.

```
                    ┌─────────────────────────────────────┐
                    │         RUNTIME ASSURANCE           │
                    │         (Simplex/RTA Switch)        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
     Learned        │                                     │    Baseline
     Policy ───────►│      SAFETY ARBITER                │◄─── Controller
     Action         │                                     │    (verified)
                    │  if monitor.certifies(learned):     │
                    │      use learned                    │
                    │  else:                              │
                    │      switch to baseline             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       CBF SAFETY FILTER             │
                    │                                     │
                    │  action_safe = solve_QP(            │
                    │      min ||a - a_proposed||²        │
                    │      s.t. dh/dt + αh(x) ≥ 0         │
                    │  )                                  │
                    │                                     │
                    │  Constraints:                       │
                    │  • h_collision(x) ≥ 0              │
                    │  • h_joint_limits(x) ≥ 0           │
                    │  • h_velocity(x) ≥ 0               │
                    │  • h_force(x) ≥ 0                  │
                    │  • h_exclusion_zone(x) ≥ 0         │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      HARDWARE LIMITS                │
                    │  (final clamp, non-bypassable)      │
                    └─────────────────────────────────────┘
```

#### Components

| Component | Input | Output | Guarantee |
|-----------|-------|--------|-----------|
| **RTA Simplex** | learned_action, state | selected_action | Always selects safe controller when monitor fails |
| **CBF Filter** | proposed_action, state | safe_action | h(x) ≥ 0 invariant maintained |
| **Runtime Monitor** | state, temporal_properties | {SAFE, UNSAFE, UNKNOWN} | Formal property checking |
| **Hardware Limits** | action | clamped_action | Physical bounds never exceeded |

#### Implementation

```python
@dataclass
class SafetyConfig:
    """P0 Safety Layer Configuration."""
    # CBF parameters
    cbf_alpha: float = 1.0  # CBF decay rate
    min_obstacle_distance: float = 0.05  # meters
    max_joint_velocity: float = 2.0  # rad/s
    max_ee_force: float = 50.0  # Newtons

    # RTA parameters
    rta_switch_threshold: float = 0.95  # Monitor confidence
    baseline_controller: str = "impedance"  # Fallback type

    # Timing
    loop_rate_hz: int = 1000
    max_latency_us: int = 500  # Hard real-time deadline


class ControlBarrierFunction:
    """Deterministic CBF safety filter."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.constraints = [
            CollisionBarrier(config.min_obstacle_distance),
            JointLimitBarrier(),
            VelocityBarrier(config.max_joint_velocity),
            ForceBarrier(config.max_ee_force),
        ]

    def filter(self, proposed_action: np.ndarray, state: RobotState) -> np.ndarray:
        """
        Project action to satisfy ALL barrier constraints.

        Solves: min ||a - a_proposed||²
                s.t. ∇h(x)·f(x,a) + α·h(x) ≥ 0  for all h

        Returns: Minimally modified safe action
        """
        A_ub = []
        b_ub = []

        for barrier in self.constraints:
            h_val = barrier.h(state)
            grad_h = barrier.grad_h(state)

            # Constraint: ∇h·(f + g·a) + α·h ≥ 0
            # Rearranged: -∇h·g·a ≤ ∇h·f + α·h
            A_ub.append(-grad_h @ state.control_matrix)
            b_ub.append(grad_h @ state.drift + self.config.cbf_alpha * h_val)

        # Solve QP
        result = solve_qp(
            P=np.eye(len(proposed_action)),
            q=-proposed_action,
            A=np.array(A_ub),
            b=np.array(b_ub),
        )

        return result if result is not None else self.safe_default(state)


class RuntimeAssurance:
    """Simplex-based RTA switching."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.baseline = load_baseline_controller(config.baseline_controller)
        self.monitor = SafetyMonitor()

    def arbitrate(
        self,
        learned_action: np.ndarray,
        state: RobotState,
    ) -> Tuple[np.ndarray, str]:
        """
        Select between learned policy and baseline controller.

        Returns: (action, source)
        """
        # Check if learned action is certifiably safe
        safety_cert = self.monitor.certify(learned_action, state)

        if safety_cert.is_safe and safety_cert.confidence >= self.config.rta_switch_threshold:
            return learned_action, "learned"
        else:
            # Switch to formally verified baseline
            baseline_action = self.baseline.compute(state)
            return baseline_action, "baseline"
```

#### Acceptance Tests

```python
def test_cbf_never_violates_constraints():
    """CBF filter must NEVER allow constraint violation."""
    cbf = ControlBarrierFunction(SafetyConfig())

    for _ in range(10000):
        # Generate random dangerous actions
        dangerous_action = np.random.randn(7) * 10  # Large random action
        state = random_robot_state()

        safe_action = cbf.filter(dangerous_action, state)
        next_state = simulate_step(state, safe_action)

        # INVARIANT: All barrier functions remain non-negative
        for barrier in cbf.constraints:
            assert barrier.h(next_state) >= -1e-6, \
                f"CBF violation: {barrier.name} = {barrier.h(next_state)}"


def test_rta_switches_before_violation():
    """RTA must switch to baseline BEFORE unsafe state."""
    rta = RuntimeAssurance(SafetyConfig())

    # Simulate trajectory toward obstacle
    state = initial_state_far_from_obstacle()

    while not state.is_terminal:
        # Learned policy that ignores obstacles
        learned = policy_ignoring_safety(state)

        action, source = rta.arbitrate(learned, state)
        next_state = simulate_step(state, action)

        # INVARIANT: Never enter unsafe state
        assert is_safe(next_state), f"RTA failed to prevent unsafe state"

        # Verify switch happened before collision
        if source == "baseline":
            assert distance_to_obstacle(state) > SWITCH_MARGIN
```

---

### Layer 1: Control Loop (100 Hz) - VERIFIED COMPOSITION

**Purpose:** Verify skill composition, check temporal properties.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPOSITION VERIFIER                                  │
│                                                                              │
│   Skill Chain: [REACH(cup), GRASP(cup), LIFT(cup), PLACE(shelf)]           │
│                     │            │           │            │                  │
│   Verify:     ┌─────▼────┐ ┌─────▼────┐ ┌────▼─────┐ ┌────▼─────┐          │
│               │ post(A)  │ │ post(B)  │ │ post(C)  │ │ post(D)  │          │
│               │    ⊆     │ │    ⊆     │ │    ⊆     │ │    ⊆     │          │
│               │ pre(B)   │ │ pre(C)   │ │ pre(D)   │ │ GOAL     │          │
│               └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│   If any ⊆ fails → INSERT TRANSITION SKILL or REJECT                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RUNTIME MONITOR (ROS2)                                │
│                                                                              │
│   Temporal Properties (LTL/MTL):                                            │
│   □ (human_detected → ◇≤500ms robot_stopped)                                │
│   □ (gripper_force > max → ◇≤100ms gripper_open)                            │
│   □ ¬(in_zone_A ∧ arm_moving)                                               │
│                                                                              │
│   On violation → trigger fallback behavior                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Components

| Component | Input | Output | Guarantee |
|-----------|-------|--------|-----------|
| **Composition Verifier** | skill_chain | verified_chain or rejection | post(A) ⊆ pre(B) for all adjacent pairs |
| **Runtime Monitor** | state_stream, LTL_properties | {SATISFIED, VIOLATED} | Temporal properties checked online |
| **Transition Synthesizer** | state_A, pre(B) | transition_skill | Bridge incompatible skills |

#### Implementation

```python
@dataclass
class SkillContract:
    """Formal contract for a composable skill."""
    name: str

    # State predicates (symbolic or set-based)
    preconditions: Set[Predicate]   # Must hold before execution
    postconditions: Set[Predicate]  # Guaranteed after execution
    invariants: Set[Predicate]      # Must hold throughout

    # Reachability (continuous)
    initial_set: ConvexSet          # Safe starting states
    terminal_set: ConvexSet         # Guaranteed ending states

    # Timing
    min_duration: float
    max_duration: float


class CompositionVerifier:
    """Verify skill chains are safely composable."""

    def verify_chain(self, skills: List[SkillContract]) -> VerificationResult:
        """
        Verify that skill chain is safely executable.

        Checks:
        1. post(skill_i) ⊆ pre(skill_{i+1})
        2. terminal_set(skill_i) ⊆ initial_set(skill_{i+1})
        3. No invariant conflicts
        """
        issues = []

        for i in range(len(skills) - 1):
            skill_a = skills[i]
            skill_b = skills[i + 1]

            # Check predicate compatibility
            if not skill_a.postconditions.implies(skill_b.preconditions):
                issues.append(PredicateMismatch(skill_a, skill_b))

            # Check reachability compatibility
            if not skill_a.terminal_set.is_subset_of(skill_b.initial_set):
                issues.append(ReachabilityGap(skill_a, skill_b))

            # Check invariant compatibility
            if skill_a.invariants.conflicts_with(skill_b.invariants):
                issues.append(InvariantConflict(skill_a, skill_b))

        if issues:
            return VerificationResult(
                verified=False,
                issues=issues,
                suggested_repairs=self._synthesize_repairs(issues),
            )

        return VerificationResult(verified=True)

    def _synthesize_repairs(self, issues: List[Issue]) -> List[SkillContract]:
        """Synthesize transition skills to repair composition issues."""
        repairs = []
        for issue in issues:
            if isinstance(issue, ReachabilityGap):
                # Synthesize motion to bridge gap
                transition = self._synthesize_transition(
                    from_set=issue.skill_a.terminal_set,
                    to_set=issue.skill_b.initial_set,
                )
                repairs.append(transition)
        return repairs


class RuntimeMonitor:
    """Online temporal property monitoring (ROS2 integrated)."""

    def __init__(self, properties: List[TemporalProperty]):
        self.properties = properties
        self.state_history = deque(maxlen=1000)
        self.violation_callbacks = {}

    def check(self, state: RobotState, timestamp: float) -> MonitorResult:
        """Check all properties against current state."""
        self.state_history.append((timestamp, state))

        results = []
        for prop in self.properties:
            result = prop.evaluate(self.state_history)

            if result == PropertyStatus.VIOLATED:
                self._trigger_fallback(prop)

            results.append((prop.name, result))

        return MonitorResult(results)

    def _trigger_fallback(self, violated_property: TemporalProperty):
        """Execute fallback behavior for violated property."""
        if violated_property.name in self.violation_callbacks:
            self.violation_callbacks[violated_property.name]()
        else:
            self.emergency_stop()
```

---

### Layer 2: Policy Loop (10 Hz) - LEARNED + ROBUST

**Purpose:** Generate actions from learned policies, fuse multimodal sensing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POLICY ENSEMBLE                                      │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │   Pi0.5     │  │  Diffusion  │  │   Tactile   │                         │
│  │    VLA      │  │   Policy    │  │   Policy    │                         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                         │
│         │                │                │                                  │
│         └────────────────┼────────────────┘                                  │
│                          ▼                                                   │
│                 ┌─────────────────┐                                         │
│                 │  ACTION FUSION  │                                         │
│                 │  (weighted by   │                                         │
│                 │   uncertainty)  │                                         │
│                 └────────┬────────┘                                         │
│                          │                                                   │
│                          ▼                                                   │
│                 ┌─────────────────┐                                         │
│                 │  RIP GATING     │  ← Kept for uncertainty estimation      │
│                 │  (soft signal)  │     but NOT for safety decisions        │
│                 └────────┬────────┘                                         │
│                          │                                                   │
│                          ▼                                                   │
│                    proposed_action → [TO SAFETY LOOP]                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Components

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| **Pi0.5 VLA** | images, instruction, proprio | action_proposal | Semantic understanding |
| **Diffusion Policy** | observation, goal | trajectory_samples | Multimodal action distribution |
| **Tactile Policy** | tactile_images, force | contact_action | Contact-rich manipulation |
| **Action Fusion** | policy_outputs, uncertainties | fused_action | Weighted combination |
| **RIP Gating** | observation | uncertainty_estimate | Soft signal (NOT safety gate) |

**Key Change:** RIP is demoted from safety-critical to informational. Hard safety is Layer 0's job.

---

### Layer 3: Planning Loop (1 Hz) - COMPOSITIONAL

**Purpose:** Decompose instructions into verified skill chains.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SKILL COMPOSER                                       │
│                                                                              │
│   Input: "Pick up the red cup and place it on the top shelf"                │
│                                    │                                         │
│                          ┌─────────▼─────────┐                              │
│                          │   LLM PARSER      │                              │
│                          │  (Gemma / local)  │                              │
│                          └─────────┬─────────┘                              │
│                                    │                                         │
│                                    ▼                                         │
│   Skill Chain: [DETECT(red_cup), REACH(red_cup), GRASP(), LIFT(),          │
│                 DETECT(top_shelf), REACH(top_shelf), PLACE(), RETRACT()]   │
│                                    │                                         │
│                          ┌─────────▼─────────┐                              │
│                          │   COMPOSITION     │                              │
│                          │   VERIFIER        │                              │
│                          └─────────┬─────────┘                              │
│                                    │                                         │
│                                    ▼                                         │
│   Verified Chain (with transitions inserted if needed)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Skill Library with Contracts

```python
SKILL_LIBRARY = {
    "REACH": SkillContract(
        name="REACH",
        preconditions={"gripper_open", "target_visible", "path_clear"},
        postconditions={"ee_at_target", "gripper_open"},
        invariants={"no_collision"},
        initial_set=ConvexSet.from_bounds(joint_limits),
        terminal_set=ConvexSet.ball(target_position, radius=0.02),
        min_duration=0.5,
        max_duration=3.0,
    ),

    "GRASP": SkillContract(
        name="GRASP",
        preconditions={"ee_at_target", "gripper_open", "object_graspable"},
        postconditions={"holding_object", "gripper_closed"},
        invariants={"ee_at_target"},
        initial_set=ConvexSet.ball(object_position, radius=0.02),
        terminal_set=ConvexSet.ball(object_position, radius=0.02),
        min_duration=0.2,
        max_duration=1.0,
    ),

    "LIFT": SkillContract(
        name="LIFT",
        preconditions={"holding_object", "gripper_closed"},
        postconditions={"object_lifted", "holding_object"},
        invariants={"gripper_closed", "grasp_stable"},
        initial_set=ConvexSet.ball(grasp_position, radius=0.02),
        terminal_set=ConvexSet.above(grasp_position, height=0.1),
        min_duration=0.3,
        max_duration=2.0,
    ),

    # ... more skills
}
```

---

### Layer 4: Operations Loop (Async) - FLEET SCALE

**Purpose:** Monitor fleet, detect drift, manage training, secure operations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLEET OPERATIONS                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DRIFT DETECTION                                 │   │
│  │                                                                      │   │
│  │   Per-robot distribution monitoring:                                │   │
│  │   • Track observation statistics over time                          │   │
│  │   • Compare to training distribution                                │   │
│  │   • Alert on significant drift (KL divergence, MMD)                 │   │
│  │                                                                      │   │
│  │   Actions: shadow_mode → canary_deploy → full_rollout               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TEE-BASED TRAINING                              │   │
│  │                                                                      │   │
│  │   Confidential computing for:                                       │   │
│  │   • Federated aggregation (robots → server)                         │   │
│  │   • Model evaluation on sensitive data                              │   │
│  │   • Secure policy updates                                           │   │
│  │                                                                      │   │
│  │   TEE: Intel SGX / ARM TrustZone / AMD SEV                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      HARDWARE-AWARE DEPLOYMENT                       │   │
│  │                                                                      │   │
│  │   Jetson Thor optimizations:                                        │   │
│  │   • INT8/FP8 quantization for diffusion                             │   │
│  │   • TensorRT compilation                                            │   │
│  │   • Latency profiling and optimization                              │   │
│  │                                                                      │   │
│  │   Target: <100ms policy inference @ 10Hz                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Robustness: RialTo Pipeline

**Purpose:** Harden policies against distribution shift via sim stress-testing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RIALTO PIPELINE                                      │
│                                                                              │
│   1. REAL DATA COLLECTION                                                   │
│      Demos from teleop (Mobile ALOHA style)                                 │
│                           │                                                  │
│                           ▼                                                  │
│   2. DIGITAL TWIN CONSTRUCTION                                              │
│      Real → 3D Gaussian Splatting → Isaac Lab scene                        │
│                           │                                                  │
│                           ▼                                                  │
│   3. DOMAIN RANDOMIZATION                                                   │
│      • Object poses ±5cm                                                    │
│      • Lighting variations                                                  │
│      • Camera noise                                                         │
│      • Physics perturbations (friction, mass)                               │
│                           │                                                  │
│                           ▼                                                  │
│   4. STRESS TESTING                                                         │
│      • Adversarial perturbations                                            │
│      • Edge case sampling                                                   │
│      • Failure mode discovery                                               │
│                           │                                                  │
│                           ▼                                                  │
│   5. POLICY FINE-TUNING                                                     │
│      RL or IL on hard cases                                                 │
│                           │                                                  │
│                           ▼                                                  │
│   6. VALIDATION                                                             │
│      Real-world tests on discovered failure modes                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deterministic Communications (ROS2 + TSN)

**Purpose:** Guarantee message delivery timing for safety-critical coordination.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC NETWORKING STACK                            │
│                                                                              │
│   Application Layer                                                          │
│   ├── Safety commands (1000 Hz, <500μs deadline)                            │
│   ├── Control commands (100 Hz, <5ms deadline)                              │
│   └── Planning updates (10 Hz, <50ms deadline)                              │
│                           │                                                  │
│                           ▼                                                  │
│   ROS2 DDS Layer                                                            │
│   ├── Priority-based QoS                                                    │
│   ├── Deadline QoS policies                                                 │
│   └── Reliable delivery for safety                                          │
│                           │                                                  │
│                           ▼                                                  │
│   TSN (Time-Sensitive Networking)                                           │
│   ├── IEEE 802.1Qbv: Time-aware shaping                                     │
│   ├── IEEE 802.1AS: Time synchronization                                    │
│   └── Bounded latency guarantees                                            │
│                           │                                                  │
│                           ▼                                                  │
│   Hardware                                                                  │
│   └── TSN-capable switches + NICs                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority Order

### Phase 1: Safety Foundation (Weeks 1-4)
```
[P0] Control Barrier Functions
     └── Implement CBF filter with collision, joint limit, velocity barriers
     └── Acceptance: 10k random actions, zero violations

[P0] Runtime Assurance (Simplex)
     └── Implement RTA switching with impedance baseline
     └── Acceptance: Graceful degradation on policy failure

[P0] Runtime Monitors (ROS2)
     └── Implement LTL property checking
     └── Acceptance: E-stop triggers within deadline
```

### Phase 2: Composition & Robustness (Weeks 5-8)
```
[P0] Composition Verifier
     └── Implement pre/post condition checking
     └── Acceptance: Reject unsafe compositions, synthesize transitions

[P0] RialTo Pipeline
     └── Digital twin construction from real data
     └── Acceptance: 2x success rate on perturbed scenarios
```

### Phase 3: Policy & Sensing (Weeks 9-12)
```
[P1] Diffusion Policy Optimization
     └── TensorRT + INT8 for Jetson Thor
     └── Acceptance: <100ms inference

[P1] Tactile Integration
     └── Vision-tactile fusion for contact tasks
     └── Acceptance: 90% grasp success in clutter
```

### Phase 4: Fleet Operations (Weeks 13-16)
```
[P1] Deterministic ROS2 + TSN
     └── Configure DDS QoS + TSN scheduling
     └── Acceptance: <500μs jitter on safety messages

[P2] TEE Training Pipeline
     └── Confidential FL aggregation
     └── Acceptance: No plaintext model exposure

[P2] MLOps Drift Detection
     └── Per-robot distribution monitoring
     └── Acceptance: Alert within 1hr of significant drift
```

---

## Acceptance Test Summary

| Test | Component | Criteria | Deterministic? |
|------|-----------|----------|----------------|
| CBF Invariance | Safety Loop | Zero barrier violations in 10k trials | ✓ |
| RTA Switch Timing | Safety Loop | Switch before unsafe state | ✓ |
| Monitor Latency | Control Loop | Property check <1ms | ✓ |
| Composition Verification | Control Loop | Reject invalid chains | ✓ |
| E-Stop Response | Full Stack | Stop <500ms from trigger | ✓ |
| Policy Latency | Policy Loop | <100ms inference | ✓ |
| Message Jitter | Comms | <500μs on safety channel | ✓ |
| Drift Detection | Ops | Alert within 1hr | Bounded |

---

## File Structure

```
src/
├── safety/
│   ├── cbf/
│   │   ├── barriers.py           # Barrier function implementations
│   │   ├── filter.py             # QP-based CBF filter
│   │   └── constraints.py        # Constraint definitions
│   ├── rta/
│   │   ├── simplex.py            # RTA switching logic
│   │   ├── baseline.py           # Verified baseline controllers
│   │   └── monitor.py            # Safety certification
│   └── runtime_monitor/
│       ├── properties.py         # LTL/MTL property definitions
│       ├── checker.py            # Online property checking
│       └── ros2_integration.py   # ROS2 node
├── composition/
│   ├── contracts.py              # Skill contract definitions
│   ├── verifier.py               # Composition verification
│   ├── synthesizer.py            # Transition synthesis
│   └── library.py                # Skill library with contracts
├── robustness/
│   ├── rialto/
│   │   ├── digital_twin.py       # Scene reconstruction
│   │   ├── randomization.py      # Domain randomization
│   │   └── stress_test.py        # Adversarial testing
│   └── domain_rand.py            # Isaac Lab integration
├── ops/
│   ├── drift/
│   │   ├── detector.py           # Distribution drift detection
│   │   └── alerts.py             # Alert management
│   ├── tee/
│   │   ├── enclave.py            # TEE integration
│   │   └── federated.py          # Confidential FL
│   └── quantization/
│       ├── ptq.py                # Post-training quantization
│       └── tensorrt.py           # TensorRT optimization
└── comms/
    ├── tsn/
    │   ├── config.py             # TSN configuration
    │   └── scheduler.py          # Traffic scheduling
    └── ros2/
        ├── qos.py                # QoS policies
        └── deadline.py           # Deadline monitoring
```

---

## Summary: Current → Production Delta

| Aspect | Current (Probabilistic) | Production (Deterministic) |
|--------|-------------------------|----------------------------|
| Safety | RIP epistemic uncertainty | CBF hard constraints + RTA |
| Recovery | POIR reactive | Prevented by CBF, baseline fallback |
| Composition | Monolithic skills | Verified contracts + transitions |
| Robustness | Hope demos cover dist | RialTo stress testing |
| Comms | Best-effort ROS2 | TSN deterministic delivery |
| Training | Cloud (plaintext) | TEE confidential |
| Monitoring | Manual | Automated drift detection |

This architecture transforms Dynamical from a research prototype into a deployable industrial system with formal safety guarantees.

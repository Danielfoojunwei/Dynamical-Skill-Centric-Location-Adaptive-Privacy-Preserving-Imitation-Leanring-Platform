"""
Deterministic Safety Module - Hard Constraint Enforcement

This module provides deterministic safety guarantees through:

1. **Control Barrier Functions (CBF)**: Hard constraint enforcement
   - Collision avoidance barriers
   - Joint limit barriers
   - Velocity/force barriers
   - QP-based action filtering

2. **Runtime Assurance (RTA)**: Verified fallback switching
   - Simplex architecture
   - Certified safety monitors
   - Baseline controller switching

3. **Runtime Monitors**: Temporal property checking
   - LTL/MTL property verification
   - Online state monitoring
   - Fallback triggering

Usage:
    from src.safety import SafePolicyExecutor, CBFFilter, RuntimeAssurance

    executor = SafePolicyExecutor.for_jetson_thor()
    safe_action = executor.execute(proposed_action, state)
"""

from .cbf import (
    CBFFilter,
    CBFConfig,
    BarrierFunction,
    CollisionBarrier,
    JointLimitBarrier,
    VelocityBarrier,
    ForceBarrier,
)

from .rta import (
    RuntimeAssurance,
    RTAConfig,
    SafetyCertificate,
    BaselineController,
)

from .runtime_monitor import (
    RuntimeMonitor,
    MonitorConfig,
    TemporalProperty,
    PropertyStatus,
    MonitorResult,
)

from .executor import (
    SafePolicyExecutor,
    SafeExecutorConfig,
    SafeExecutionResult,
)

__all__ = [
    # CBF
    'CBFFilter',
    'CBFConfig',
    'BarrierFunction',
    'CollisionBarrier',
    'JointLimitBarrier',
    'VelocityBarrier',
    'ForceBarrier',

    # RTA
    'RuntimeAssurance',
    'RTAConfig',
    'SafetyCertificate',
    'BaselineController',

    # Runtime Monitor
    'RuntimeMonitor',
    'MonitorConfig',
    'TemporalProperty',
    'PropertyStatus',
    'MonitorResult',

    # Executor
    'SafePolicyExecutor',
    'SafeExecutorConfig',
    'SafeExecutionResult',
]

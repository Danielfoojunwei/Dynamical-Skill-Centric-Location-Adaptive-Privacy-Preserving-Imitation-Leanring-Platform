"""
Core Module - Central Engine for Dynamical Edge Platform v0.9.0

This module contains the core functionality including:
- Skill orchestration and execution (UnifiedSkillOrchestrator)
- Robot skill invocation (RobotSkillInvoker)
- Motion retargeting (human-to-robot via GMR)
- Error handling and system robustness
- Timing architecture for real-time control
- Configuration management

Primary Components:
    - UnifiedSkillOrchestrator: MoE-based skill routing and task decomposition
    - RobotSkillInvoker: Edge skill execution at 200Hz
    - TimingOrchestrator: 4-tier timing system coordination
    - SystemErrorManager: Comprehensive error handling with fallbacks
    - RobustSystemManager: System-wide robustness management

Note: SkillBlender was deprecated in v0.8.0 and removed in v0.9.0.
      VLA handles multi-objective behavior implicitly via Deep Imitative Learning.

Usage:
    from src.core import (
        TimingOrchestrator,
        SystemErrorManager,
        load_and_validate_config,
    )
"""

# Configuration
from .config_loader import load_and_validate_config, AppConfig

# Human state representation
from .human_state import HumanState, EnvObject, DexterHandState, Human3DState

# Recording
from .recorder import RobotObs, RobotAction, DemoStep, Episode

# Error handling (graceful degradation)
from .error_handling import (
    ErrorSeverity,
    ComponentState,
    ErrorEvent,
    ErrorTracker,
    FallbackCache,
    PerceptionFallback,
    ControlFallback,
    CommunicationFallback,
    SystemErrorManager,
    get_error_manager,
    init_error_manager,
    with_fallback,
    with_retry,
    with_timeout,
    safe_call,
    require_healthy,
)

# System robustness
from .system_robustness import (
    ConnectionType,
    ConnectionHealth,
    DualPathConnection,
    GPUResourceManager,
    TierPriority,
    TierMessage,
    InterTierBus,
    RedundancyManager,
    HealthPredictor,
    SafeFallbackController,
    FHESecurityConfig,
    RobustSystemManager,
)

# Timing architecture
from .timing_architecture import (
    TimingTier,
    TimingConfig,
    SafetyState,
    SafetyMonitor,
    ControlState,
    ControlCommand,
    ControlLoop,
    PerceptionResult,
    PerceptionLoop,
    LearningLoop,
    TimingOrchestrator,
)

# Memory monitoring
from .memory_monitor import MemoryMonitor, OutOfMemoryError

# Environment hazards
from .environment_hazards import (
    HazardCategory,
    HazardType,
    HazardTypeDefinition,
    EnvironmentHazard,
    HazardRegistry,
)

# Batch autotuning
from .batch_autotuner import BatchSizeAutotuner, find_optimal_batch_size

# Schema validation
from .schema_validator import ValidationResult, SchemaValidator

# Skill orchestration (v0.9.0)
from .unified_skill_orchestrator import (
    UnifiedSkillOrchestrator,
    get_orchestrator,
    configure_orchestrator,
    OrchestrationRequest,
    OrchestrationResult,
    SkillExecutionPlan,
    SkillStep,
)

from .robot_skill_invoker import (
    RobotSkillInvoker,
    get_skill_invoker,
    SkillInvocationRequest,
    SkillInvocationResult,
    ObservationState,
    RobotAction,
)

# Submodules
from . import retargeting
from . import depth_estimation
from . import pose_inference

__all__ = [
    # Configuration
    'load_and_validate_config',
    'AppConfig',

    # Human state
    'HumanState',
    'EnvObject',
    'DexterHandState',
    'Human3DState',

    # Recording
    'RobotObs',
    'RobotAction',
    'DemoStep',
    'Episode',

    # Error handling
    'ErrorSeverity',
    'ComponentState',
    'ErrorEvent',
    'ErrorTracker',
    'FallbackCache',
    'PerceptionFallback',
    'ControlFallback',
    'CommunicationFallback',
    'SystemErrorManager',
    'get_error_manager',
    'init_error_manager',
    'with_fallback',
    'with_retry',
    'with_timeout',
    'safe_call',
    'require_healthy',

    # System robustness
    'ConnectionType',
    'ConnectionHealth',
    'DualPathConnection',
    'GPUResourceManager',
    'TierPriority',
    'TierMessage',
    'InterTierBus',
    'RedundancyManager',
    'HealthPredictor',
    'SafeFallbackController',
    'FHESecurityConfig',
    'RobustSystemManager',

    # Timing
    'TimingTier',
    'TimingConfig',
    'SafetyState',
    'SafetyMonitor',
    'ControlState',
    'ControlCommand',
    'ControlLoop',
    'PerceptionResult',
    'PerceptionLoop',
    'LearningLoop',
    'TimingOrchestrator',

    # Utilities
    'MemoryMonitor',
    'OutOfMemoryError',
    'HazardCategory',
    'HazardType',
    'HazardTypeDefinition',
    'EnvironmentHazard',
    'HazardRegistry',
    'BatchSizeAutotuner',
    'find_optimal_batch_size',
    'ValidationResult',
    'SchemaValidator',

    # Skill orchestration (v0.9.0)
    'UnifiedSkillOrchestrator',
    'get_orchestrator',
    'configure_orchestrator',
    'OrchestrationRequest',
    'OrchestrationResult',
    'SkillExecutionPlan',
    'SkillStep',
    'RobotSkillInvoker',
    'get_skill_invoker',
    'SkillInvocationRequest',
    'SkillInvocationResult',
    'ObservationState',
    'RobotAction',

    # Submodules
    'retargeting',
    'depth_estimation',
    'pose_inference',
]

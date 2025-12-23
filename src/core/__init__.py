"""
Core Module - Central Engine for Dynamical Edge Platform

This module contains the core functionality including:
- Skill orchestration and execution
- Motion retargeting (human-to-robot)
- Error handling and system robustness
- Timing architecture for real-time control
- Configuration management

Primary Components:
    - UnifiedSkillOrchestrator: Central skill routing and execution
    - RobotSkillInvoker: Real-time skill invocation
    - GMRRetargeter: Human-to-robot motion mapping
    - TimingArchitecture: 4-tier timing system
    - ErrorHandler: Fault tolerance and recovery

Usage:
    from src.core import (
        UnifiedSkillOrchestrator,
        RobotSkillInvoker,
        config,
    )

    orchestrator = UnifiedSkillOrchestrator()
    orchestrator.initialize()
"""

# Configuration
from .config_loader import config, ConfigLoader, get_config

# Human state representation
from .human_state import HumanState, EnvObject, DexterHandState

# Recording
from .recorder import RobotObs, RobotAction, DemoStep, EpisodeRecorder

# Error handling
from .error_handling import (
    DynamicalError,
    SafetyError,
    CalibrationError,
    CommunicationError,
    ErrorHandler,
)

# System robustness
from .system_robustness import (
    SystemRobustness,
    RobustnessConfig,
    FallbackStrategy,
)

# Timing
from .timing_architecture import (
    TimingArchitecture,
    TimingConfig,
    TimingTier,
)

# Memory monitoring
from .memory_monitor import MemoryMonitor

# Environment hazards
from .environment_hazards import EnvironmentHazards, HazardConfig

# Batch autotuning
from .batch_autotuner import BatchAutotuner

# Schema validation
from .schema_validator import validate_config

# Submodules
from . import retargeting
from . import depth_estimation
from . import pose_inference

__all__ = [
    # Configuration
    'config',
    'ConfigLoader',
    'get_config',

    # Human state
    'HumanState',
    'EnvObject',
    'DexterHandState',

    # Recording
    'RobotObs',
    'RobotAction',
    'DemoStep',
    'EpisodeRecorder',

    # Error handling
    'DynamicalError',
    'SafetyError',
    'CalibrationError',
    'CommunicationError',
    'ErrorHandler',

    # System robustness
    'SystemRobustness',
    'RobustnessConfig',
    'FallbackStrategy',

    # Timing
    'TimingArchitecture',
    'TimingConfig',
    'TimingTier',

    # Utilities
    'MemoryMonitor',
    'EnvironmentHazards',
    'HazardConfig',
    'BatchAutotuner',
    'validate_config',

    # Submodules
    'retargeting',
    'depth_estimation',
    'pose_inference',
]

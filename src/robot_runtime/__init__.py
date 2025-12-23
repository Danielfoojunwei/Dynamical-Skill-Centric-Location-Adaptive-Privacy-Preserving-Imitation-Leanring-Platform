"""
Robot Runtime Agent - On-Robot Execution Environment

This is the MANDATORY on-robot component that owns:
- Hard scheduling (Tier 1-2)
- Actuator interface
- Skill execution state machine
- Local feature cache + telemetry
- Safety authority (can ALWAYS stop the robot)

Design based on real deployment patterns from ANYmal, Spot CORE I/O, and Isaac ROS.
"""

from .agent import RobotRuntimeAgent
from .safety_shield import SafetyShield
from .state_estimator import StateEstimator
from .actuator_interface import ActuatorInterface
from .execution_state import ExecutionStateMachine, ExecutionState
from .perception_pipeline import PerceptionPipeline, CascadeLevel
from .policy_executor import PolicyExecutor
from .skill_cache import SkillCache
from .watchdog import Watchdog
from .config import RobotRuntimeConfig, TierConfig

__all__ = [
    # Main agent
    'RobotRuntimeAgent',

    # Tier 1 components (1kHz, deterministic)
    'SafetyShield',
    'StateEstimator',
    'ActuatorInterface',
    'Watchdog',
    'ExecutionStateMachine',
    'ExecutionState',

    # Tier 2 components (10-100Hz, bounded)
    'PerceptionPipeline',
    'CascadeLevel',
    'PolicyExecutor',
    'SkillCache',

    # Configuration
    'RobotRuntimeConfig',
    'TierConfig',
]

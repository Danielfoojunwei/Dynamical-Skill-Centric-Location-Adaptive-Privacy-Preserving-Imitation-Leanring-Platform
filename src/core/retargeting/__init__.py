"""
Real Retargeting Module

Provides production-ready motion retargeting using:
- Pinocchio for robot kinematics
- Numerical IK with damped least squares
- URDF loading and parsing

Replaces all mock implementations.
"""

from .kinematics import (
    RobotKinematics,
    KinematicsConfig,
    JointLimits,
)
from .ik_solver import (
    IKSolver,
    IKSolverConfig,
    IKResult,
)
from .motion_retargeter import (
    MotionRetargeter,
    RetargetingConfig,
    RetargetingResult,
)

__all__ = [
    'RobotKinematics',
    'KinematicsConfig',
    'JointLimits',
    'IKSolver',
    'IKSolverConfig',
    'IKResult',
    'MotionRetargeter',
    'RetargetingConfig',
    'RetargetingResult',
]

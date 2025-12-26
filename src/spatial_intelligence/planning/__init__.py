"""
Planning Module - Diffusion-based Trajectory Generation

This module provides modern diffusion-based trajectory planning
that refines VLA outputs into smooth, executable action sequences.

Components:
- DiffusionPlanner: Score-based trajectory generation
- TrajectoryOptimizer: Gradient-based refinement
"""

from .diffusion_planner import (
    DiffusionPlanner,
    DiffusionConfig,
    Trajectory,
    TrajectoryBatch,
    DenoisingSchedule,
)

__all__ = [
    'DiffusionPlanner',
    'DiffusionConfig',
    'Trajectory',
    'TrajectoryBatch',
    'DenoisingSchedule',
]

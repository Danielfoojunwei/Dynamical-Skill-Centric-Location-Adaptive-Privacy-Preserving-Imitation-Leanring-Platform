"""
Execution Module - Simplified Architecture

Key insight: Deep Imitative Learning eliminates skill blending.

The VLA learns multi-objective behavior implicitly from demonstrations,
so there's no need to blend multiple skills at runtime. This simplifies
the architecture significantly:

OLD (Skill Blending):
    skill_1.infer() -> action_1
    skill_2.infer() -> action_2
    blender.blend([action_1, action_2], weights) -> blended_action
    safety.filter(blended_action) -> safe_action

    Problem: Blending causes jitter, oscillations, boundary instability

NEW (Deep Imitative Learning):
    vla.infer(instruction, images) -> action
    diffusion.refine(action) -> smooth_trajectory
    cbf.filter(smooth_trajectory) -> safe_action

    No blending needed - VLA handles multi-objective internally

For long-horizon tasks, we use task decomposition (not skill blending):
    decomposer.decompose("clean the kitchen") -> ["wash dishes", "wipe counter", ...]
    for subtask in subtasks:
        vla.infer(subtask, images) -> action
        cbf.filter(action) -> safe_action

This is fundamentally different from blending:
- Blending: simultaneous combination of multiple skill outputs
- Decomposition: sequential execution of sub-tasks
"""

from .dynamical_executor import (
    DynamicalExecutor,
    ExecutorConfig,
    ExecutionResult,
)

from .task_decomposer import (
    TaskDecomposer,
    DecomposerConfig,
    DecompositionResult,
    SubTask,
)

__all__ = [
    # Main executor
    'DynamicalExecutor',
    'ExecutorConfig',
    'ExecutionResult',

    # Task decomposition
    'TaskDecomposer',
    'DecomposerConfig',
    'DecompositionResult',
    'SubTask',
]

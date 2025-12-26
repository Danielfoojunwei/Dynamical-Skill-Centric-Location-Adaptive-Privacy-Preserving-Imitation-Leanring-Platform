"""
Compositional Task Verification - SEQUENTIAL CHAINING ONLY

IMPORTANT: This module handles SEQUENTIAL task chaining, NOT skill blending.

With Deep Imitative Learning (Pi0.5 VLA), skill blending is obsolete:
- The VLA learns multi-objective behavior implicitly from demonstrations
- There's no need to blend "grasp" + "avoid collision" at runtime
- The model learns to grasp while avoiding collisions from training data

This composition module verifies that SEQUENTIAL sub-tasks can be chained:
- post(task_A) ⊆ pre(task_B) - ensures valid transitions
- Used by TaskDecomposer for long-horizon tasks
- NOT used for parallel skill combination (that's eliminated)

Sequential Chaining (SUPPORTED):
    "set table" → ["get plates", "place plates", "get utensils", ...]
    Each sub-task executes fully before the next begins.

Skill Blending (REMOVED - handled by VLA):
    blend(grasp, avoid, weights) - VLA handles this internally

Runtime Verification (NEW):
    Uses perception (SAM3, DINOv3, V-JEPA2) to verify postconditions ACTUALLY HOLD
    before transitions. CBF ensures transition safety.

Usage:
    from src.composition import CompositionVerifier, SkillContract

    verifier = CompositionVerifier()
    result = verifier.verify_chain([subtask_a, subtask_b, subtask_c])

    if not result.verified:
        # Add transition tasks or alert user
        repaired_chain = verifier.repair_chain(skills, result.suggested_repairs)

    # Runtime verification (NEW)
    from src.composition.runtime import RuntimeTransitionChecker

    runtime_checker = RuntimeTransitionChecker.create(unified_perception)
    ok, details = runtime_checker.verify_transition(
        skill_from="grasp_cup",
        skill_to="place_cup",
        postconditions=["holding(cup)"],
        preconditions=["holding(cup)"],
        frame=camera_frame,
        robot_state=current_state,
    )
"""

from .contracts import SkillContract, Predicate, PredicateSet
from .verifier import CompositionVerifier, VerificationResult, CompositionIssue
from .library import SkillLibrary, get_default_library

# Runtime verification (uses perception for real-world verification)
from .runtime import (
    PostconditionVerifier,
    TransitionSafety,
    TransitionSafetyResult,
)

__all__ = [
    # Static verification
    'SkillContract',
    'Predicate',
    'PredicateSet',
    'CompositionVerifier',
    'VerificationResult',
    'CompositionIssue',
    'SkillLibrary',
    'get_default_library',

    # Runtime verification
    'PostconditionVerifier',
    'TransitionSafety',
    'TransitionSafetyResult',
]

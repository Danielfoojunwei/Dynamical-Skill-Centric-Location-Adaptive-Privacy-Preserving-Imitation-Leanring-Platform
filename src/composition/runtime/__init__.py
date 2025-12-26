"""
Runtime Composition Verification - Verify Postconditions Before Skill Transitions

This module provides runtime verification that:
1. Postconditions actually hold in the real world before transitions
2. Transitions between skills are safe (via CBF)
3. Recovery is possible if verification fails

Uses the existing perception system (SAM3, DINOv3, V-JEPA2) for verification.

Architecture:
============

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    RUNTIME VERIFICATION                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Skill A Complete ──┬──▶ Postcondition Verifier ────────┐           │
    │                     │    (SAM3 + DINOv3 + V-JEPA2)       │           │
    │                     │                                    │           │
    │                     └──▶ Transition Safety (CBF) ───────┼──▶ OK?    │
    │                                                          │           │
    │  ┌───────────────────────────────────────────────────────┘           │
    │  │                                                                   │
    │  ├─▶ YES ──▶ Start Skill B                                          │
    │  │                                                                   │
    │  └─▶ NO ───▶ Recovery Strategy                                       │
    │               - Retry skill A                                        │
    │               - Execute recovery skill                               │
    │               - Human handoff                                        │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from src.composition.runtime import (
        PostconditionVerifier,
        TransitionSafety,
        RuntimeVerificationResult,
    )

    # Create verifier with perception
    verifier = PostconditionVerifier.from_unified_perception()

    # Verify postcondition before transition
    result = verifier.verify(
        postcondition="holding(cup)",
        frame=current_frame,
        robot_state=robot_state,
    )

    if result.verified:
        # Safe to transition to next skill
        executor.execute(next_skill)
    else:
        # Handle failure
        recovery.execute(result.suggested_recovery)
"""

from .postcondition_verifier import (
    PostconditionVerifier,
    VerificationResult,
    Predicate,
    PredicateType,
    PredicateRegistry,
)

from .transition_safety import (
    TransitionSafety,
    TransitionBarrier,
    TransitionSafetyResult,
    TransitionConfig,
)

__all__ = [
    # Postcondition Verification
    'PostconditionVerifier',
    'VerificationResult',
    'Predicate',
    'PredicateType',
    'PredicateRegistry',

    # Transition Safety
    'TransitionSafety',
    'TransitionBarrier',
    'TransitionSafetyResult',
    'TransitionConfig',
]

"""
Compositional Skill Verification

Provides verified skill composition with formal contracts:
- Pre/post condition checking
- Reachability verification
- Transition synthesis

Usage:
    from src.composition import CompositionVerifier, SkillContract

    verifier = CompositionVerifier()
    result = verifier.verify_chain([skill_a, skill_b, skill_c])

    if not result.verified:
        repaired_chain = verifier.repair_chain(skills, result.issues)
"""

from .contracts import SkillContract, Predicate, PredicateSet
from .verifier import CompositionVerifier, VerificationResult, CompositionIssue
from .library import SkillLibrary, get_default_library

__all__ = [
    'SkillContract',
    'Predicate',
    'PredicateSet',
    'CompositionVerifier',
    'VerificationResult',
    'CompositionIssue',
    'SkillLibrary',
    'get_default_library',
]

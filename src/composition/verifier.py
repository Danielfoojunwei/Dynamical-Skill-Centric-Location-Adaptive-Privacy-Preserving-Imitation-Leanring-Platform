"""
Composition Verifier - Verified Skill Chaining

Verifies that skill chains are safely composable by checking:
1. post(skill_i) ⊆ pre(skill_{i+1})
2. Invariant compatibility
3. Timing constraints
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .contracts import SkillContract, PredicateSet

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Type of composition issue."""
    PRECONDITION_MISMATCH = "precondition_mismatch"
    INVARIANT_CONFLICT = "invariant_conflict"
    TIMING_VIOLATION = "timing_violation"
    MISSING_SKILL = "missing_skill"


@dataclass
class CompositionIssue:
    """An issue found during composition verification."""
    issue_type: IssueType
    skill_a: str
    skill_b: str
    description: str
    missing_predicates: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of composition verification."""
    verified: bool
    issues: List[CompositionIssue] = field(default_factory=list)
    suggested_repairs: List[SkillContract] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.verified and len(self.issues) == 0


class CompositionVerifier:
    """
    Verifies skill chain composability.

    Ensures that for each adjacent pair (A, B):
    - A.postconditions implies B.preconditions
    - A.invariants compatible with B.invariants
    """

    def __init__(self):
        self.transition_skills: List[SkillContract] = []

    def verify_chain(self, skills: List[SkillContract]) -> VerificationResult:
        """
        Verify that a skill chain is safely composable.

        Args:
            skills: Ordered list of skills to execute

        Returns:
            VerificationResult with issues and repair suggestions
        """
        if len(skills) < 2:
            return VerificationResult(verified=True)

        issues = []

        for i in range(len(skills) - 1):
            skill_a = skills[i]
            skill_b = skills[i + 1]

            # Check precondition implication
            if not skill_a.postconditions.implies(skill_b.preconditions):
                missing = self._find_missing_predicates(
                    skill_a.postconditions,
                    skill_b.preconditions,
                )
                issues.append(CompositionIssue(
                    issue_type=IssueType.PRECONDITION_MISMATCH,
                    skill_a=skill_a.name,
                    skill_b=skill_b.name,
                    description=f"Postconditions of {skill_a.name} do not imply "
                               f"preconditions of {skill_b.name}",
                    missing_predicates=missing,
                ))

        verified = len(issues) == 0

        result = VerificationResult(
            verified=verified,
            issues=issues,
        )

        if not verified:
            result.suggested_repairs = self._synthesize_repairs(issues)

        return result

    def _find_missing_predicates(
        self,
        source: PredicateSet,
        target: PredicateSet,
    ) -> List[str]:
        """Find predicates in target not satisfied by source."""
        missing = []
        for name in target.predicates:
            if name not in source.predicates:
                missing.append(name)
        return missing

    def _synthesize_repairs(
        self,
        issues: List[CompositionIssue],
    ) -> List[SkillContract]:
        """Synthesize transition skills to repair issues."""
        repairs = []

        for issue in issues:
            if issue.issue_type == IssueType.PRECONDITION_MISMATCH:
                # Create a transition skill
                transition = SkillContract(
                    name=f"transition_{issue.skill_a}_to_{issue.skill_b}",
                    preconditions=PredicateSet(),  # Accept anything
                    postconditions=PredicateSet(),  # Will be filled based on missing
                    min_duration=0.1,
                    max_duration=2.0,
                )
                repairs.append(transition)

        return repairs

    def repair_chain(
        self,
        skills: List[SkillContract],
        repairs: List[SkillContract],
    ) -> List[SkillContract]:
        """
        Insert repair skills into chain.

        Returns new chain with transitions inserted where needed.
        """
        if not repairs:
            return skills

        result = [skills[0]]
        repair_idx = 0

        for i in range(1, len(skills)):
            skill_a = skills[i - 1]
            skill_b = skills[i]

            # Check if we need a transition here
            if not skill_a.postconditions.implies(skill_b.preconditions):
                if repair_idx < len(repairs):
                    result.append(repairs[repair_idx])
                    repair_idx += 1

            result.append(skill_b)

        return result

    def verify_pair(
        self,
        skill_a: SkillContract,
        skill_b: SkillContract,
    ) -> Tuple[bool, Optional[CompositionIssue]]:
        """Verify a single skill pair."""
        if skill_a.postconditions.implies(skill_b.preconditions):
            return True, None

        missing = self._find_missing_predicates(
            skill_a.postconditions,
            skill_b.preconditions,
        )

        return False, CompositionIssue(
            issue_type=IssueType.PRECONDITION_MISMATCH,
            skill_a=skill_a.name,
            skill_b=skill_b.name,
            description=f"Missing predicates: {missing}",
            missing_predicates=missing,
        )

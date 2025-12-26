"""
Skill Library - Pre-defined Skill Contracts

Contains standard manipulation skill contracts with formal specifications.
"""

from typing import Dict, Optional
from .contracts import (
    SkillContract, PredicateSet, Predicate,
    GRIPPER_OPEN, GRIPPER_CLOSED, OBJECT_VISIBLE, HOLDING_OBJECT,
    AT_TARGET, PATH_CLEAR, ROBOT_STATIONARY,
)


class SkillLibrary:
    """Library of skill contracts with formal specifications."""

    def __init__(self):
        self.skills: Dict[str, SkillContract] = {}
        self._register_default_skills()

    def _register_default_skills(self):
        """Register default manipulation skills."""

        # REACH: Move to target position
        self.register(SkillContract(
            name="reach",
            preconditions=PredicateSet([PATH_CLEAR]),
            postconditions=PredicateSet([AT_TARGET]),
            invariants=PredicateSet([]),
            min_duration=0.5,
            max_duration=5.0,
        ))

        # GRASP: Close gripper on object
        self.register(SkillContract(
            name="grasp",
            preconditions=PredicateSet([AT_TARGET, GRIPPER_OPEN, OBJECT_VISIBLE]),
            postconditions=PredicateSet([HOLDING_OBJECT, GRIPPER_CLOSED]),
            invariants=PredicateSet([AT_TARGET]),
            min_duration=0.2,
            max_duration=2.0,
        ))

        # LIFT: Lift held object
        self.register(SkillContract(
            name="lift",
            preconditions=PredicateSet([HOLDING_OBJECT, GRIPPER_CLOSED]),
            postconditions=PredicateSet([HOLDING_OBJECT]),
            invariants=PredicateSet([GRIPPER_CLOSED]),
            min_duration=0.3,
            max_duration=3.0,
        ))

        # PLACE: Place object at target
        self.register(SkillContract(
            name="place",
            preconditions=PredicateSet([HOLDING_OBJECT, AT_TARGET]),
            postconditions=PredicateSet([GRIPPER_OPEN, ROBOT_STATIONARY]),
            invariants=PredicateSet([]),
            min_duration=0.3,
            max_duration=3.0,
        ))

        # RELEASE: Open gripper
        self.register(SkillContract(
            name="release",
            preconditions=PredicateSet([GRIPPER_CLOSED]),
            postconditions=PredicateSet([GRIPPER_OPEN]),
            invariants=PredicateSet([]),
            min_duration=0.1,
            max_duration=1.0,
        ))

        # RETRACT: Move away from current position
        self.register(SkillContract(
            name="retract",
            preconditions=PredicateSet([GRIPPER_OPEN]),
            postconditions=PredicateSet([PATH_CLEAR]),
            invariants=PredicateSet([GRIPPER_OPEN]),
            min_duration=0.3,
            max_duration=3.0,
        ))

        # OPEN_GRIPPER: Open the gripper
        self.register(SkillContract(
            name="open_gripper",
            preconditions=PredicateSet([]),
            postconditions=PredicateSet([GRIPPER_OPEN]),
            invariants=PredicateSet([]),
            min_duration=0.1,
            max_duration=0.5,
        ))

        # CLOSE_GRIPPER: Close the gripper
        self.register(SkillContract(
            name="close_gripper",
            preconditions=PredicateSet([]),
            postconditions=PredicateSet([GRIPPER_CLOSED]),
            invariants=PredicateSet([]),
            min_duration=0.1,
            max_duration=0.5,
        ))

    def register(self, skill: SkillContract):
        """Register a skill contract."""
        self.skills[skill.name] = skill

    def get(self, name: str) -> Optional[SkillContract]:
        """Get a skill by name."""
        return self.skills.get(name)

    def list_skills(self):
        """List all registered skill names."""
        return list(self.skills.keys())

    def get_chain(self, *names: str):
        """Get a list of skills by name."""
        return [self.skills[name] for name in names if name in self.skills]


# Global default library
_default_library: Optional[SkillLibrary] = None


def get_default_library() -> SkillLibrary:
    """Get the default skill library."""
    global _default_library
    if _default_library is None:
        _default_library = SkillLibrary()
    return _default_library

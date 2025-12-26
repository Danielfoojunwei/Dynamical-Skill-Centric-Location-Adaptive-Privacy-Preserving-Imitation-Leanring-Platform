"""
Skill Contracts for Compositional Verification

Each skill has:
- Preconditions: Must hold before execution
- Postconditions: Guaranteed after execution
- Invariants: Must hold throughout
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np


@dataclass
class Predicate:
    """A logical predicate on robot/world state."""
    name: str
    check: Callable[[Any], bool]
    description: str = ""

    def __call__(self, state: Any) -> bool:
        return self.check(state)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self.name == other.name
        return False


class PredicateSet:
    """Set of predicates with logical operations."""

    def __init__(self, predicates: Optional[List[Predicate]] = None):
        self.predicates: Dict[str, Predicate] = {}
        if predicates:
            for p in predicates:
                self.predicates[p.name] = p

    def add(self, predicate: Predicate):
        """Add a predicate."""
        self.predicates[predicate.name] = predicate

    def check_all(self, state: Any) -> bool:
        """Check if all predicates are satisfied."""
        return all(p(state) for p in self.predicates.values())

    def check_any(self, state: Any) -> bool:
        """Check if any predicate is satisfied."""
        return any(p(state) for p in self.predicates.values())

    def get_satisfied(self, state: Any) -> List[str]:
        """Get names of satisfied predicates."""
        return [name for name, p in self.predicates.items() if p(state)]

    def get_unsatisfied(self, state: Any) -> List[str]:
        """Get names of unsatisfied predicates."""
        return [name for name, p in self.predicates.items() if not p(state)]

    def implies(self, other: 'PredicateSet') -> bool:
        """
        Check if this predicate set implies another.

        Returns True if all predicates in other are also in self.
        (Syntactic check - conservative approximation)
        """
        return all(name in self.predicates for name in other.predicates)

    def __iter__(self):
        return iter(self.predicates.values())

    def __len__(self):
        return len(self.predicates)


@dataclass
class SkillContract:
    """
    Formal contract for a composable skill.

    Defines what must be true before, during, and after skill execution.
    """
    name: str

    # Logical conditions
    preconditions: PredicateSet = field(default_factory=PredicateSet)
    postconditions: PredicateSet = field(default_factory=PredicateSet)
    invariants: PredicateSet = field(default_factory=PredicateSet)

    # Parameters (bound at runtime)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Timing bounds
    min_duration: float = 0.1
    max_duration: float = 10.0

    # Skill implementation reference
    executor: Optional[Callable] = None

    def can_follow(self, other: 'SkillContract') -> bool:
        """Check if this skill can safely follow another."""
        return other.postconditions.implies(self.preconditions)

    def bind_parameters(self, **kwargs) -> 'SkillContract':
        """Create a new contract with bound parameters."""
        new_params = {**self.parameters, **kwargs}
        return SkillContract(
            name=f"{self.name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})",
            preconditions=self.preconditions,
            postconditions=self.postconditions,
            invariants=self.invariants,
            parameters=new_params,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            executor=self.executor,
        )


# Common predicates
GRIPPER_OPEN = Predicate("gripper_open", lambda s: getattr(s, 'gripper_state', 0) < 0.5)
GRIPPER_CLOSED = Predicate("gripper_closed", lambda s: getattr(s, 'gripper_state', 1) > 0.5)
OBJECT_VISIBLE = Predicate("object_visible", lambda s: getattr(s, 'object_detected', False))
HOLDING_OBJECT = Predicate("holding_object", lambda s: getattr(s, 'holding', False))
AT_TARGET = Predicate("at_target", lambda s: getattr(s, 'at_target', False))
ROBOT_STATIONARY = Predicate("robot_stationary", lambda s: np.linalg.norm(getattr(s, 'velocity', [0])) < 0.01)
PATH_CLEAR = Predicate("path_clear", lambda s: getattr(s, 'path_clear', True))

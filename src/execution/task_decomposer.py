"""
Task Decomposer - Long-Horizon Task Decomposition

This module decomposes complex, long-horizon tasks into sequential sub-tasks.

IMPORTANT: This is NOT skill blending!

Task Decomposition vs Skill Blending:
─────────────────────────────────────
Skill Blending (OLD - REMOVED):
    - Multiple skills run in parallel
    - Outputs are combined with weights
    - Causes jitter, oscillations, boundary instability
    - Example: blend(grasp, avoid_collision, weights=[0.6, 0.4])

Task Decomposition (NEW):
    - Tasks are broken into sequential steps
    - Only one sub-task executes at a time
    - VLA handles multi-objective internally
    - Example: "clean kitchen" → ["wash dishes", "wipe counter", "sweep floor"]

When to decompose:
    - Long-horizon tasks (>30 seconds expected)
    - Multi-stage tasks with distinct phases
    - Tasks requiring different object interactions

When NOT to decompose:
    - Simple pick-and-place
    - Single-stage manipulation
    - Tasks under 30 seconds

Usage:
    decomposer = TaskDecomposer()

    # Check if decomposition needed
    if decomposer.needs_decomposition("clean the entire kitchen"):
        result = decomposer.decompose("clean the entire kitchen")
        for subtask in result.subtasks:
            executor.execute(subtask.instruction, images, state)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"           # Single action, no decomposition
    MODERATE = "moderate"       # May benefit from decomposition
    COMPLEX = "complex"         # Should be decomposed
    VERY_COMPLEX = "very_complex"  # Must be decomposed


@dataclass
class SubTask:
    """A single sub-task in a decomposed task."""
    instruction: str
    index: int
    estimated_duration: float = 5.0  # seconds

    # Preconditions (from composition module)
    requires: List[str] = field(default_factory=list)  # What must be true before
    provides: List[str] = field(default_factory=list)  # What will be true after


@dataclass
class DecompositionResult:
    """Result of task decomposition."""
    original_instruction: str
    subtasks: List[SubTask]
    complexity: TaskComplexity
    total_estimated_duration: float

    @property
    def needs_decomposition(self) -> bool:
        return len(self.subtasks) > 1


@dataclass
class DecomposerConfig:
    """Configuration for task decomposer."""
    # Complexity thresholds
    simple_max_words: int = 5
    moderate_max_words: int = 10

    # Keywords that indicate complex tasks
    complex_keywords: List[str] = field(default_factory=lambda: [
        "all", "every", "each", "multiple", "several",
        "clean", "organize", "sort", "arrange", "prepare",
        "set up", "pack", "unpack", "assemble", "disassemble",
        "first", "then", "after", "before", "finally",
        "and then", "followed by", "next",
    ])

    # Duration thresholds (seconds)
    simple_duration_threshold: float = 10.0
    complex_duration_threshold: float = 30.0


class TaskDecomposer:
    """
    Decomposes long-horizon tasks into sequential sub-tasks.

    This is fundamentally different from skill blending:
    - Decomposition: Sequential execution of distinct phases
    - Blending: Parallel combination of skill outputs (REMOVED)

    The VLA handles multi-objective within each sub-task.
    Decomposition only splits temporal phases.
    """

    def __init__(self, config: Optional[DecomposerConfig] = None):
        self.config = config or DecomposerConfig()

        # Common task patterns
        self._task_patterns = self._build_task_patterns()

        # Statistics
        self.stats = {
            "decompositions": 0,
            "subtasks_generated": 0,
        }

    def _build_task_patterns(self) -> Dict[str, List[str]]:
        """Build common task decomposition patterns."""
        return {
            # Table setting
            "set the table": [
                "get plates from cabinet",
                "place plates on table",
                "get utensils from drawer",
                "place utensils next to plates",
                "get glasses from cabinet",
                "place glasses on table",
            ],

            # Kitchen cleaning
            "clean the kitchen": [
                "clear items from counter",
                "wipe counter surface",
                "wash dishes in sink",
                "dry and put away dishes",
                "wipe stovetop",
            ],

            # Desk organization
            "organize the desk": [
                "gather loose papers",
                "stack papers neatly",
                "arrange pens in holder",
                "position monitor centrally",
                "clear trash from desk",
            ],

            # Laundry
            "fold laundry": [
                "pick up shirt from pile",
                "fold shirt",
                "place in folded stack",
                "pick up next item",
            ],

            # Tool organization
            "organize tools": [
                "gather scattered tools",
                "sort tools by type",
                "place tools in designated spots",
                "close tool drawers",
            ],
        }

    def estimate_complexity(self, instruction: str) -> float:
        """
        Estimate task complexity (0.0 to 1.0).

        Higher values indicate more complex tasks that need decomposition.
        """
        instruction_lower = instruction.lower()
        score = 0.0

        # Word count factor
        word_count = len(instruction.split())
        if word_count <= self.config.simple_max_words:
            score += 0.1
        elif word_count <= self.config.moderate_max_words:
            score += 0.3
        else:
            score += 0.5

        # Keyword matching
        for keyword in self.config.complex_keywords:
            if keyword in instruction_lower:
                score += 0.15

        # Check for explicit sequencing
        if any(seq in instruction_lower for seq in ["then", "after", "next", "first"]):
            score += 0.3

        # Check for known complex patterns
        for pattern in self._task_patterns.keys():
            if pattern in instruction_lower:
                score += 0.4
                break

        return min(1.0, score)

    def needs_decomposition(self, instruction: str) -> bool:
        """Check if task needs decomposition."""
        return self.estimate_complexity(instruction) > 0.5

    def decompose(self, instruction: str) -> DecompositionResult:
        """
        Decompose a task into sub-tasks.

        Args:
            instruction: Natural language task instruction

        Returns:
            DecompositionResult with sub-tasks

        Note: This produces SEQUENTIAL sub-tasks, not parallel skills.
        Each sub-task is executed one at a time by the VLA.
        """
        self.stats["decompositions"] += 1
        instruction_lower = instruction.lower()

        # Check for known patterns first
        for pattern, subtask_list in self._task_patterns.items():
            if pattern in instruction_lower:
                subtasks = [
                    SubTask(
                        instruction=st,
                        index=i,
                        estimated_duration=5.0,
                    )
                    for i, st in enumerate(subtask_list)
                ]
                self.stats["subtasks_generated"] += len(subtasks)

                return DecompositionResult(
                    original_instruction=instruction,
                    subtasks=subtasks,
                    complexity=TaskComplexity.COMPLEX,
                    total_estimated_duration=len(subtasks) * 5.0,
                )

        # Try to parse explicit sequences
        if any(seq in instruction_lower for seq in [" then ", " and then ", " followed by "]):
            subtasks = self._parse_sequential_instruction(instruction)
            if len(subtasks) > 1:
                self.stats["subtasks_generated"] += len(subtasks)
                return DecompositionResult(
                    original_instruction=instruction,
                    subtasks=subtasks,
                    complexity=TaskComplexity.MODERATE,
                    total_estimated_duration=len(subtasks) * 5.0,
                )

        # No decomposition needed - return single task
        complexity = self._assess_complexity(instruction)
        return DecompositionResult(
            original_instruction=instruction,
            subtasks=[SubTask(instruction=instruction, index=0)],
            complexity=complexity,
            total_estimated_duration=5.0,
        )

    def _parse_sequential_instruction(self, instruction: str) -> List[SubTask]:
        """Parse instruction with explicit sequence markers."""
        # Split on sequence markers
        parts = re.split(
            r'\s+(?:then|and then|followed by|after that|next)\s+',
            instruction,
            flags=re.IGNORECASE,
        )

        if len(parts) <= 1:
            return [SubTask(instruction=instruction, index=0)]

        # Clean up parts
        subtasks = []
        for i, part in enumerate(parts):
            part = part.strip()
            # Remove leading "first" or similar
            part = re.sub(r'^(?:first|to start|initially)\s+', '', part, flags=re.IGNORECASE)
            # Remove trailing punctuation
            part = part.rstrip('.,;')

            if part:
                subtasks.append(SubTask(
                    instruction=part,
                    index=i,
                    estimated_duration=5.0,
                ))

        return subtasks

    def _assess_complexity(self, instruction: str) -> TaskComplexity:
        """Assess complexity level."""
        score = self.estimate_complexity(instruction)

        if score < 0.3:
            return TaskComplexity.SIMPLE
        elif score < 0.5:
            return TaskComplexity.MODERATE
        elif score < 0.8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX

    def add_custom_pattern(self, pattern: str, subtasks: List[str]):
        """Add a custom decomposition pattern."""
        self._task_patterns[pattern.lower()] = subtasks

    def get_statistics(self) -> Dict[str, Any]:
        """Get decomposer statistics."""
        return {
            **self.stats,
            "known_patterns": len(self._task_patterns),
        }


def verify_subtask_chain(subtasks: List[SubTask]) -> Tuple[bool, Optional[str]]:
    """
    Verify that sub-task chain is valid.

    Checks that each sub-task's postconditions satisfy the next
    sub-task's preconditions.

    This uses the composition module's contract verification.
    """
    try:
        from ..composition import CompositionVerifier, get_default_library

        library = get_default_library()
        verifier = CompositionVerifier()

        # Map subtasks to skill contracts
        contracts = []
        for st in subtasks:
            # Try to find matching skill
            skill = library.get(st.instruction)
            if skill is not None:
                contracts.append(skill)

        if len(contracts) < 2:
            return True, None

        # Verify chain
        result = verifier.verify_chain(contracts)
        if result.verified:
            return True, None
        else:
            issues = [f"{i.skill_a} -> {i.skill_b}: {i.description}" for i in result.issues]
            return False, "; ".join(issues)

    except ImportError:
        # Composition module not available
        return True, None

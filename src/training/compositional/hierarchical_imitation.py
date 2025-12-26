"""
Hierarchical Imitation Learning - Build Skill Hierarchies from Primitives

Given discovered primitives, this module:
1. Learns temporal relationships between primitives
2. Builds hierarchical skill structures
3. Trains composition operators for novel task generation

Hierarchy Levels:
    Level 0: Atomic primitives (reach, grasp, release, ...)
    Level 1: Basic skills (pick, place, push, ...)
    Level 2: Complex skills (stack, sort, pour, ...)
    Level 3: Tasks (set_table, clean_kitchen, ...)

Composition Operators:
    - Sequence: A ; B (do A then B)
    - Parallel: A || B (do A and B simultaneously) - via VLA
    - Conditional: if C then A else B
    - Loop: while C do A

This enables novel task composition at inference time.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from .skill_discovery import DiscoveredPrimitive, DemoSegment

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Levels in the skill hierarchy."""
    PRIMITIVE = 0    # Atomic learned skills
    BASIC = 1        # Single-object interactions
    COMPLEX = 2      # Multi-object interactions
    TASK = 3         # Full task sequences


class CompositionType(Enum):
    """Types of skill composition."""
    SEQUENCE = "sequence"       # A ; B
    PARALLEL = "parallel"       # A || B (via VLA multi-objective)
    CONDITIONAL = "conditional" # if C then A else B
    LOOP = "loop"              # while C do A


@dataclass
class CompositionOperator:
    """A learned composition operator."""
    op_type: CompositionType
    name: str

    # For sequence: transition model between skills
    transition_model: Optional[Any] = None

    # For conditional: condition classifier
    condition_classifier: Optional[Any] = None

    # For parallel: attention weights over primitives
    attention_weights: Optional[np.ndarray] = None


@dataclass
class SkillNode:
    """A node in the skill hierarchy."""
    skill_id: str
    name: str
    level: HierarchyLevel

    # Primitive data (for level 0)
    primitive: Optional[DiscoveredPrimitive] = None

    # Composition (for higher levels)
    composition: Optional[CompositionOperator] = None
    children: List["SkillNode"] = field(default_factory=list)

    # Learned policy
    policy: Optional[Any] = None

    # Pre/post conditions for composition verification
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class SkillHierarchy:
    """Complete skill hierarchy."""
    root_nodes: List[SkillNode] = field(default_factory=list)
    all_nodes: Dict[str, SkillNode] = field(default_factory=dict)

    # Level indices
    primitives: List[SkillNode] = field(default_factory=list)
    basic_skills: List[SkillNode] = field(default_factory=list)
    complex_skills: List[SkillNode] = field(default_factory=list)
    tasks: List[SkillNode] = field(default_factory=list)


@dataclass
class TransitionData:
    """Data about transitions between primitives."""
    from_primitive: str
    to_primitive: str
    count: int = 0
    avg_transition_time: float = 0.0
    transition_features: Optional[np.ndarray] = None


@dataclass
class HierarchyConfig:
    """Configuration for hierarchy learning."""
    # V-JEPA2 for temporal modeling
    use_vjepa_transitions: bool = True

    # Minimum co-occurrence for sequence detection
    min_sequence_count: int = 3

    # Composition learning
    learn_conditionals: bool = True
    learn_loops: bool = True

    # Hierarchy depth
    max_hierarchy_depth: int = 3

    # Device
    device: str = "cuda"


class HierarchicalImitation:
    """
    Builds and learns skill hierarchies from discovered primitives.

    Uses V-JEPA2 to:
    1. Learn temporal transitions between primitives
    2. Identify recurring sequences (basic skills)
    3. Build compositional structure (complex skills, tasks)
    """

    def __init__(
        self,
        primitives: List[DiscoveredPrimitive],
        config: Optional[HierarchyConfig] = None,
    ):
        self.primitives = primitives
        self.config = config or HierarchyConfig()

        # Build initial hierarchy with primitives at level 0
        self.hierarchy = self._init_hierarchy()

        # Transition statistics
        self.transitions: Dict[Tuple[str, str], TransitionData] = {}

        # V-JEPA2 for temporal modeling
        self._vjepa = None

    def _init_hierarchy(self) -> SkillHierarchy:
        """Initialize hierarchy with primitives."""
        hierarchy = SkillHierarchy()

        for prim in self.primitives:
            node = SkillNode(
                skill_id=prim.primitive_id,
                name=prim.name,
                level=HierarchyLevel.PRIMITIVE,
                primitive=prim,
            )
            hierarchy.primitives.append(node)
            hierarchy.all_nodes[prim.primitive_id] = node

        return hierarchy

    def learn_composition(
        self,
        demo_sequences: List[List[str]],  # Sequences of primitive IDs
    ) -> SkillHierarchy:
        """
        Learn compositional structure from primitive sequences.

        Args:
            demo_sequences: List of demos, each a list of primitive IDs

        Returns:
            Updated hierarchy with higher-level skills
        """
        logger.info("Learning compositional structure...")

        # Step 1: Learn transitions between primitives
        self._learn_transitions(demo_sequences)

        # Step 2: Discover basic skills (frequent sequences)
        self._discover_basic_skills(demo_sequences)

        # Step 3: Discover complex skills (combinations)
        self._discover_complex_skills(demo_sequences)

        # Step 4: Learn composition operators
        self._learn_composition_operators()

        return self.hierarchy

    def _learn_transitions(self, demo_sequences: List[List[str]]):
        """Learn transition statistics between primitives."""
        logger.info("Learning primitive transitions...")

        for sequence in demo_sequences:
            for i in range(len(sequence) - 1):
                from_prim = sequence[i]
                to_prim = sequence[i + 1]
                key = (from_prim, to_prim)

                if key not in self.transitions:
                    self.transitions[key] = TransitionData(
                        from_primitive=from_prim,
                        to_primitive=to_prim,
                    )
                self.transitions[key].count += 1

        logger.info(f"Found {len(self.transitions)} unique transitions")

    def _discover_basic_skills(self, demo_sequences: List[List[str]]):
        """Discover basic skills from frequent 2-3 primitive sequences."""
        logger.info("Discovering basic skills...")

        # Find frequent bigrams
        bigram_counts: Dict[Tuple[str, str], int] = {}
        for seq in demo_sequences:
            for i in range(len(seq) - 1):
                bigram = (seq[i], seq[i + 1])
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        # Find frequent trigrams
        trigram_counts: Dict[Tuple[str, str, str], int] = {}
        for seq in demo_sequences:
            for i in range(len(seq) - 2):
                trigram = (seq[i], seq[i + 1], seq[i + 2])
                trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1

        # Create basic skills from frequent patterns
        skill_idx = 0

        # Trigrams first (more specific)
        for trigram, count in sorted(trigram_counts.items(), key=lambda x: -x[1]):
            if count >= self.config.min_sequence_count:
                skill_id = f"basic_skill_{skill_idx}"
                children = [self.hierarchy.all_nodes[p] for p in trigram if p in self.hierarchy.all_nodes]

                if len(children) == 3:
                    node = SkillNode(
                        skill_id=skill_id,
                        name=f"seq_{trigram[0]}_{trigram[1]}_{trigram[2]}",
                        level=HierarchyLevel.BASIC,
                        composition=CompositionOperator(
                            op_type=CompositionType.SEQUENCE,
                            name=f"sequence_{skill_idx}",
                        ),
                        children=children,
                    )
                    self.hierarchy.basic_skills.append(node)
                    self.hierarchy.all_nodes[skill_id] = node
                    skill_idx += 1

        # Then bigrams (more general)
        for bigram, count in sorted(bigram_counts.items(), key=lambda x: -x[1]):
            if count >= self.config.min_sequence_count:
                # Check not already covered by trigram
                already_covered = False
                for basic in self.hierarchy.basic_skills:
                    child_ids = [c.skill_id for c in basic.children]
                    if bigram[0] in child_ids and bigram[1] in child_ids:
                        already_covered = True
                        break

                if not already_covered:
                    skill_id = f"basic_skill_{skill_idx}"
                    children = [self.hierarchy.all_nodes[p] for p in bigram if p in self.hierarchy.all_nodes]

                    if len(children) == 2:
                        node = SkillNode(
                            skill_id=skill_id,
                            name=f"seq_{bigram[0]}_{bigram[1]}",
                            level=HierarchyLevel.BASIC,
                            composition=CompositionOperator(
                                op_type=CompositionType.SEQUENCE,
                                name=f"sequence_{skill_idx}",
                            ),
                            children=children,
                        )
                        self.hierarchy.basic_skills.append(node)
                        self.hierarchy.all_nodes[skill_id] = node
                        skill_idx += 1

        logger.info(f"Discovered {len(self.hierarchy.basic_skills)} basic skills")

    def _discover_complex_skills(self, demo_sequences: List[List[str]]):
        """Discover complex skills from basic skill combinations."""
        logger.info("Discovering complex skills...")

        # Map sequences to basic skills
        basic_skill_sequences = []
        for seq in demo_sequences:
            basic_seq = self._map_to_basic_skills(seq)
            if basic_seq:
                basic_skill_sequences.append(basic_seq)

        # Find frequent combinations of basic skills
        combo_counts: Dict[Tuple[str, ...], int] = {}
        for seq in basic_skill_sequences:
            for i in range(len(seq) - 1):
                combo = tuple(seq[i:i + 2])
                combo_counts[combo] = combo_counts.get(combo, 0) + 1

        # Create complex skills
        skill_idx = 0
        for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
            if count >= self.config.min_sequence_count:
                skill_id = f"complex_skill_{skill_idx}"
                children = [self.hierarchy.all_nodes[s] for s in combo if s in self.hierarchy.all_nodes]

                if len(children) >= 2:
                    node = SkillNode(
                        skill_id=skill_id,
                        name=f"complex_{skill_idx}",
                        level=HierarchyLevel.COMPLEX,
                        composition=CompositionOperator(
                            op_type=CompositionType.SEQUENCE,
                            name=f"complex_sequence_{skill_idx}",
                        ),
                        children=children,
                    )
                    self.hierarchy.complex_skills.append(node)
                    self.hierarchy.all_nodes[skill_id] = node
                    skill_idx += 1

        logger.info(f"Discovered {len(self.hierarchy.complex_skills)} complex skills")

    def _map_to_basic_skills(self, primitive_seq: List[str]) -> List[str]:
        """Map a primitive sequence to basic skills where possible."""
        result = []
        i = 0

        while i < len(primitive_seq):
            matched = False

            # Try to match longest basic skill first
            for basic in sorted(self.hierarchy.basic_skills, key=lambda b: -len(b.children)):
                child_ids = [c.skill_id for c in basic.children]
                seq_len = len(child_ids)

                if i + seq_len <= len(primitive_seq):
                    if primitive_seq[i:i + seq_len] == child_ids:
                        result.append(basic.skill_id)
                        i += seq_len
                        matched = True
                        break

            if not matched:
                result.append(primitive_seq[i])
                i += 1

        return result

    def _learn_composition_operators(self):
        """Learn composition operators using V-JEPA2."""
        if not self.config.use_vjepa_transitions:
            return

        logger.info("Learning composition operators with V-JEPA2...")

        try:
            from ...meta_ai import VJEPA2Wrapper
            self._vjepa = VJEPA2Wrapper(device=self.config.device)
        except ImportError:
            logger.warning("V-JEPA2 not available, skipping operator learning")
            return

        # Learn transition models for each composition
        for node in self.hierarchy.basic_skills + self.hierarchy.complex_skills:
            if node.composition and node.composition.op_type == CompositionType.SEQUENCE:
                # Learn transition features between children
                # This would use V-JEPA2 to predict the transition dynamics
                pass

    def compose_for_task(
        self,
        task_description: str,
        available_skills: Optional[List[str]] = None,
    ) -> Optional[SkillNode]:
        """
        Compose a skill hierarchy for a novel task.

        This is where compositional training pays off:
        We can compose skills for tasks NOT seen in training!

        Args:
            task_description: Natural language task
            available_skills: Limit to these skills (optional)

        Returns:
            SkillNode representing the composed task, or None
        """
        logger.info(f"Composing skills for: {task_description}")

        # Use language model to decompose task into known skills
        # This would integrate with the TaskDecomposer
        # For now, return a placeholder

        # Find relevant skills based on task description
        relevant = self._find_relevant_skills(task_description)

        if not relevant:
            return None

        # Compose into task
        task_node = SkillNode(
            skill_id=f"composed_task_{hash(task_description) % 10000}",
            name=task_description,
            level=HierarchyLevel.TASK,
            composition=CompositionOperator(
                op_type=CompositionType.SEQUENCE,
                name="task_sequence",
            ),
            children=relevant,
        )

        return task_node

    def _find_relevant_skills(self, task_description: str) -> List[SkillNode]:
        """Find skills relevant to a task description."""
        # Simple keyword matching for now
        # Would use language model embeddings for better matching

        keywords = task_description.lower().split()
        relevant = []

        # Check all skills at all levels
        all_skills = (
            self.hierarchy.primitives +
            self.hierarchy.basic_skills +
            self.hierarchy.complex_skills
        )

        for skill in all_skills:
            skill_name = skill.name.lower()
            for keyword in keywords:
                if keyword in skill_name:
                    relevant.append(skill)
                    break

        return relevant

    def get_primitive_policy(self, primitive_id: str) -> Optional[Any]:
        """Get the learned policy for a primitive."""
        node = self.hierarchy.all_nodes.get(primitive_id)
        if node and node.level == HierarchyLevel.PRIMITIVE:
            return node.policy
        return None

    def execute_composed_skill(
        self,
        skill_node: SkillNode,
        executor_fn: Callable[[str, Any], Any],
        state: Any,
    ) -> List[Any]:
        """
        Execute a composed skill by recursively executing children.

        Args:
            skill_node: The skill to execute
            executor_fn: Function to execute primitives (primitive_id, state) -> result
            state: Current robot state

        Returns:
            List of execution results
        """
        results = []

        if skill_node.level == HierarchyLevel.PRIMITIVE:
            # Execute the primitive directly
            result = executor_fn(skill_node.skill_id, state)
            results.append(result)

        elif skill_node.composition:
            if skill_node.composition.op_type == CompositionType.SEQUENCE:
                # Execute children in sequence
                for child in skill_node.children:
                    child_results = self.execute_composed_skill(child, executor_fn, state)
                    results.extend(child_results)
                    # Update state from last result if available
                    if child_results:
                        state = child_results[-1].get('state', state)

            elif skill_node.composition.op_type == CompositionType.CONDITIONAL:
                # Evaluate condition and execute appropriate branch
                # This would use the condition_classifier
                pass

        return results

    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about the hierarchy."""
        return {
            "num_primitives": len(self.hierarchy.primitives),
            "num_basic_skills": len(self.hierarchy.basic_skills),
            "num_complex_skills": len(self.hierarchy.complex_skills),
            "num_tasks": len(self.hierarchy.tasks),
            "num_transitions": len(self.transitions),
            "total_nodes": len(self.hierarchy.all_nodes),
        }

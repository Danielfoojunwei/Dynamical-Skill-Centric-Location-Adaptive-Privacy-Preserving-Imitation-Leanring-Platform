"""
Compositional Training - Skill Discovery and Hierarchical Imitation

This module solves the TRAINING compositionality problem:
- Discovers skill primitives from continuous demonstration streams
- Builds hierarchical skill structures
- Enables composition for novel tasks NOT seen in training

Key insight: We already have the infrastructure!
- ONVIF Cameras: Continuous video of human trainers
- MANUS Gloves: Hand pose and finger movements
- Meta AI Models: SAM3 (segmentation), DINOv3 (features), V-JEPA2 (temporal)

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Continuous Demo Stream (ONVIF + MANUS + Proprio)               │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Skill Segmentation (SAM3 object boundaries + DINOv3 features)  │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Skill Discovery (cluster segments → primitives)                │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Hierarchy Learning (V-JEPA2 temporal structure)                │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Compositional Policy Training (primitives + composer)          │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from src.training.compositional import (
        SkillDiscovery,
        HierarchicalImitation,
        CompositionalTrainer,
    )

    # Discover skills from demo stream
    discovery = SkillDiscovery.from_meta_ai()
    primitives = discovery.discover_from_stream(demo_stream)

    # Build hierarchy
    hierarchy = HierarchicalImitation(primitives)
    hierarchy.learn_composition()

    # Train compositional policy
    trainer = CompositionalTrainer(hierarchy)
    trainer.train()
"""

from .skill_discovery import (
    SkillDiscovery,
    SkillSegmenter,
    SkillCluster,
    DiscoveredPrimitive,
    SegmentBoundary,
)

from .hierarchical_imitation import (
    HierarchicalImitation,
    SkillHierarchy,
    HierarchyLevel,
    CompositionOperator,
)

from .compositional_trainer import (
    CompositionalTrainer,
    CompositionalConfig,
    TrainingResult,
)

__all__ = [
    # Skill Discovery
    'SkillDiscovery',
    'SkillSegmenter',
    'SkillCluster',
    'DiscoveredPrimitive',
    'SegmentBoundary',

    # Hierarchical Imitation
    'HierarchicalImitation',
    'SkillHierarchy',
    'HierarchyLevel',
    'CompositionOperator',

    # Compositional Training
    'CompositionalTrainer',
    'CompositionalConfig',
    'TrainingResult',
]

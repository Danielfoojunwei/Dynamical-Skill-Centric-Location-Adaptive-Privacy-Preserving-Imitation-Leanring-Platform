"""
Recovery Module - Return-to-Distribution Planning

This module provides recovery mechanisms when the robot drifts
out of the training distribution.

Components:
- POIRRecovery: Plans path back to familiar states
- RecoveryPlanner: Multi-strategy recovery planning
"""

from .poir import (
    POIRRecovery,
    POIRConfig,
    RecoveryPlan,
    RecoveryStatus,
    RecoveryStrategy,
)

__all__ = [
    'POIRRecovery',
    'POIRConfig',
    'RecoveryPlan',
    'RecoveryStatus',
    'RecoveryStrategy',
]

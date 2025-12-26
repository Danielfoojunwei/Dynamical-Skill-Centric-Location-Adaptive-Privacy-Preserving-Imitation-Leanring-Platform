"""
Safety Module - Epistemic Uncertainty and Risk Assessment

This module provides safety-critical components for detecting
out-of-distribution states and gating unsafe actions.

Components:
- RIPGating: Robust Imitative Planning with epistemic uncertainty
- UncertaintyEstimator: Ensemble-based uncertainty quantification
"""

from .rip_gating import (
    RIPGating,
    RIPConfig,
    SafetyDecision,
    UncertaintyEstimate,
    RiskLevel,
)

__all__ = [
    'RIPGating',
    'RIPConfig',
    'SafetyDecision',
    'UncertaintyEstimate',
    'RiskLevel',
]

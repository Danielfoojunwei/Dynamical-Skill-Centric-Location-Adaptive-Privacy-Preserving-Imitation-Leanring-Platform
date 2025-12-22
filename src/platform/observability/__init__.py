"""
Observability Module - System Monitoring, Tracing, and Analysis

This module provides comprehensive observability for the Dynamical Edge Platform:
- TraceManager: Event capture and black box recording for incident investigation
- RootCauseAnalyzer: Heuristic analysis for determining failure causes
- FHEAuditor: Verification of encrypted data uploads
- ObservabilitySystem: Unified interface for all observability components

Usage:
    from src.platform.observability import (
        TraceManager,
        RootCauseAnalyzer,
        FHEAuditor,
        ObservabilitySystem,
    )

    # Use individual components
    trace_mgr = TraceManager()
    trace_mgr.log_event("VLA", "Decision", {"action": "grasp"})

    # Or use the unified system
    obs = ObservabilitySystem()
    obs.log_event("Safety", "Alert", {"type": "proximity"})
"""

from .observability import (
    # Data classes
    TraceEvent,
    AuditEntry,
    IncidentReport,

    # Components
    TraceManager,
    RootCauseAnalyzer,
    FHEAuditor,

    # Unified system
    ObservabilitySystem,
)

__all__ = [
    # Data classes
    'TraceEvent',
    'AuditEntry',
    'IncidentReport',

    # Components
    'TraceManager',
    'RootCauseAnalyzer',
    'FHEAuditor',

    # Unified system
    'ObservabilitySystem',
]

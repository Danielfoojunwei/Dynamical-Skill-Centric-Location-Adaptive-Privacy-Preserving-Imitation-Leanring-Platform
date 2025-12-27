"""
Spatial Intelligence Module - Deep Imitative Learning Stack

This module provides spatial reasoning and safe imitation learning through:

1. **Pi0.5 VLA**: Official VLA from Physical Intelligence
   - Pre-trained on 10k+ hours of robot data
   - Open-world generalization
   - Semantic task understanding

2. **Diffusion Planner**: Smooth trajectory generation
   - Score-based trajectory refinement
   - Multi-modal action distributions
   - Goal-conditioned planning

3. **RIP Safety Gating**: Epistemic uncertainty for safety
   - Ensemble disagreement detection
   - Out-of-distribution detection
   - Risk level classification

4. **POIR Recovery**: Return-to-distribution planning
   - Recovery trajectory generation
   - World model simulation
   - Multiple recovery strategies

Usage:
    # Full Deep Imitative Learning pipeline
    from src.spatial_intelligence import DeepImitativeLearning

    dil = DeepImitativeLearning.for_jetson_thor()
    dil.load()
    result = dil.execute(
        instruction="pick up the red cup",
        images=camera_images,
    )

    # Or use individual components
    from src.spatial_intelligence.pi0 import Pi05Model
    from src.spatial_intelligence.planning import DiffusionPlanner
    from src.spatial_intelligence.safety import RIPGating
    from src.spatial_intelligence.recovery import POIRRecovery
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Pi0.5 - Official Physical Intelligence Implementation
# =============================================================================
try:
    from .pi0 import (
        Pi05Model,
        Pi05Config,
        Pi05Variant,
        Pi05Observation,
        Pi05Result,
        load_pi05,
        list_variants,
        check_installation,
        HAS_OPENPI,
    )
except ImportError as e:
    logger.debug(f"Pi0.5 not available: {e}")
    HAS_OPENPI = False
    Pi05Model = None
    Pi05Config = None
    Pi05Variant = None
    Pi05Observation = None
    Pi05Result = None
    load_pi05 = None
    list_variants = None
    check_installation = None

# =============================================================================
# VLA Interface
# =============================================================================
try:
    from .vla_interface import (
        VLAInterface,
        VLAConfig,
        VLAObservation,
        VLAResult,
        HardwareTarget,
    )
except ImportError as e:
    logger.debug(f"VLA interface not available: {e}")
    VLAInterface = None
    VLAConfig = None
    VLAObservation = None
    VLAResult = None
    HardwareTarget = None

# =============================================================================
# Diffusion Planner
# =============================================================================
try:
    from .planning import (
        DiffusionPlanner,
        DiffusionConfig,
        Trajectory,
        TrajectoryBatch,
        DenoisingSchedule,
    )
except ImportError as e:
    logger.debug(f"Diffusion Planner not available: {e}")
    DiffusionPlanner = None
    DiffusionConfig = None
    Trajectory = None
    TrajectoryBatch = None
    DenoisingSchedule = None

# =============================================================================
# RIP Safety Gating
# =============================================================================
try:
    from .safety import (
        RIPGating,
        RIPConfig,
        SafetyDecision,
        UncertaintyEstimate,
        RiskLevel,
    )
except ImportError as e:
    logger.debug(f"RIP Safety not available: {e}")
    RIPGating = None
    RIPConfig = None
    SafetyDecision = None
    UncertaintyEstimate = None
    RiskLevel = None

# =============================================================================
# POIR Recovery
# =============================================================================
try:
    from .recovery import (
        POIRRecovery,
        POIRConfig,
        RecoveryPlan,
        RecoveryStatus,
        RecoveryStrategy,
    )
except ImportError as e:
    logger.debug(f"POIR Recovery not available: {e}")
    POIRRecovery = None
    POIRConfig = None
    RecoveryPlan = None
    RecoveryStatus = None
    RecoveryStrategy = None

# =============================================================================
# Deep Imitative Learning Integration
# =============================================================================
try:
    from .deep_imitative_learning import (
        DeepImitativeLearning,
        DILConfig,
        DILResult,
        ExecutionMode,
    )
except ImportError as e:
    logger.debug(f"Deep Imitative Learning not available: {e}")
    DeepImitativeLearning = None
    DILConfig = None
    DILResult = None
    ExecutionMode = None

# =============================================================================
# Visual Trace (MolmoAct-Inspired Steerability)
# =============================================================================
try:
    from .visual_trace import (
        VisualTrace,
        VisualTraceConfig,
        VisualTraceRenderer,
        TraceModifier,
        Waypoint,
        WaypointType,
        TraceStyle,
        create_visual_trace,
        modify_trace_with_language,
    )
except ImportError as e:
    logger.debug(f"Visual Trace not available: {e}")
    VisualTrace = None
    VisualTraceConfig = None
    VisualTraceRenderer = None
    TraceModifier = None
    Waypoint = None
    WaypointType = None
    TraceStyle = None
    create_visual_trace = None
    modify_trace_with_language = None


__all__ = [
    # Pi0.5 Official
    'Pi05Model',
    'Pi05Config',
    'Pi05Variant',
    'Pi05Observation',
    'Pi05Result',
    'load_pi05',
    'list_variants',
    'check_installation',
    'HAS_OPENPI',

    # VLA Interface
    'VLAInterface',
    'VLAConfig',
    'VLAObservation',
    'VLAResult',
    'HardwareTarget',

    # Diffusion Planner
    'DiffusionPlanner',
    'DiffusionConfig',
    'Trajectory',
    'TrajectoryBatch',
    'DenoisingSchedule',

    # RIP Safety
    'RIPGating',
    'RIPConfig',
    'SafetyDecision',
    'UncertaintyEstimate',
    'RiskLevel',

    # POIR Recovery
    'POIRRecovery',
    'POIRConfig',
    'RecoveryPlan',
    'RecoveryStatus',
    'RecoveryStrategy',

    # Deep Imitative Learning
    'DeepImitativeLearning',
    'DILConfig',
    'DILResult',
    'ExecutionMode',

    # Visual Trace (MolmoAct-Inspired)
    'VisualTrace',
    'VisualTraceConfig',
    'VisualTraceRenderer',
    'TraceModifier',
    'Waypoint',
    'WaypointType',
    'TraceStyle',
    'create_visual_trace',
    'modify_trace_with_language',
]

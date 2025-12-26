"""
Runtime Assurance (RTA) - Simplex Architecture

Provides verified fallback switching between learned policy and baseline controller.
"""

from .simplex import (
    RuntimeAssurance,
    RTAConfig,
    SafetyCertificate,
    CertificationResult,
)

from .baseline import (
    BaselineController,
    ImpedanceController,
    SafeStopController,
)

__all__ = [
    'RuntimeAssurance',
    'RTAConfig',
    'SafetyCertificate',
    'CertificationResult',
    'BaselineController',
    'ImpedanceController',
    'SafeStopController',
]

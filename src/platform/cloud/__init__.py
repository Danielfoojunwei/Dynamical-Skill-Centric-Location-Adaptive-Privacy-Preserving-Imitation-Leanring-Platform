"""
Cloud Services Module - Cloud Integration Layer (v0.9.0)

This module provides cloud connectivity for:
- Model management (base VLA + skill library)
- Federated learning aggregation
- MoE skill routing
- Vendor adapters for different VLA providers

Components:
    - UnifiedModelClient: Base models + skill library
    - MoESkillRouter: Mixture-of-Experts skill routing
    - FederatedLearning: Privacy-preserving training
    - VendorAdapter: Pi0, OpenVLA, ACT adapters

Usage:
    from src.platform.cloud import (
        UnifiedModelClient,
        MoESkillRouter,
        SecureAggregatorReal,
    )

    # Model operations
    client = UnifiedModelClient()
    path = client.download_base_model("pi0-base")

    # Skill routing
    router = MoESkillRouter()
    skills = router.route_task("pick up the cup")

Note: In v0.9.0, ffm_client.py and vendor_adapter.py stubs were removed.
      Imports now point directly to the real implementations.
"""

# Model client (unified entry point) - v0.9.0: direct import from real implementation
from .model_client import (
    UnifiedModelClient,
    BaseModelClient,
    SkillLibraryClient,
    BaseModelInfo,
    SkillInfo,
    ModelClientConfig,
    FFMClient,  # Deprecated alias
)

# Legacy FFM client (for backward compatibility)
from .ffm_client_real import (
    FFMClientReal,
    FFMClientConfig,
    ModelVersion,
    SecureAggregatorReal,
)

# MoE routing
from .moe_skill_router import (
    MoESkillRouter,
    CloudSkillService,
    SkillRequest,
    SkillResponse,
)

# Federated learning
from .federated_learning import (
    FederatedLearningServer,
    FederatedClient,
)

# Vendor adapters - v0.9.0: direct import from real implementation
from .vendor_adapter_real import (
    VendorAdapter,
    Pi0VendorAdapter,
    OpenVLAVendorAdapter,
    ACTVendorAdapter,
    ModelConfig,
)

# Secure aggregation
from .secure_aggregator import (
    SecureAggregator,
    AggregatorConfig,
)


def get_default_client() -> UnifiedModelClient:
    """Get default model client instance."""
    return UnifiedModelClient()


__all__ = [
    # Model client
    'UnifiedModelClient',
    'BaseModelClient',
    'SkillLibraryClient',
    'BaseModelInfo',
    'SkillInfo',
    'ModelClientConfig',
    'get_default_client',

    # MoE routing
    'MoESkillRouter',
    'CloudSkillService',
    'SkillRequest',
    'SkillResponse',

    # Federated learning
    'FederatedLearningServer',
    'FederatedClient',
    'SecureAggregatorReal',

    # Vendor adapters
    'VendorAdapter',
    'Pi0VendorAdapter',
    'OpenVLAVendorAdapter',
    'ACTVendorAdapter',
    'ModelConfig',

    # Secure aggregation
    'SecureAggregator',
    'AggregatorConfig',

    # Legacy (deprecated, will be removed in v1.0)
    'FFMClient',
    'FFMClientReal',
    'FFMClientConfig',
    'ModelVersion',
]

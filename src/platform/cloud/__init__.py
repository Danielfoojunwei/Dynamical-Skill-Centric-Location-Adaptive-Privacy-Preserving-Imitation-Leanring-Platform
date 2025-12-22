"""
Cloud Services Module - Cloud Integration Layer

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
"""

# Model client (unified entry point)
from .ffm_client import (
    UnifiedModelClient,
    BaseModelClient,
    SkillLibraryClient,
    BaseModelInfo,
    SkillInfo,
    ModelClientConfig,
    SecureAggregatorReal,
    # Deprecated
    FFMClient,
    FFMClientReal,
    FFMClientConfig,
    get_default_client,
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

# Vendor adapters
from .vendor_adapter import (
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

    # Deprecated
    'FFMClient',
    'FFMClientReal',
    'FFMClientConfig',
]

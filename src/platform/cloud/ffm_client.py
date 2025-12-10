"""
FFM Client - Unified Model and Skill Client

This module provides the main entry point for model operations:
- Base VLA model download (read-only, one-time)
- Skill library operations (read/write, MoE skills)

Architecture Note:
    Base VLA models (Pi0, OpenVLA) are proprietary and READ-ONLY.
    We cannot train or upload gradients to vendor models.

    Skills are MoE experts that WE train and own. They are:
    - Trained on edge devices
    - Uploaded to our cloud skill library
    - Aggregated via federated learning across our fleet

Usage:
    from src.platform.cloud.ffm_client import UnifiedModelClient

    client = UnifiedModelClient()

    # One-time base model download
    path = client.download_base_model("pi0-base")

    # Skill operations
    client.upload_skill("grasp", "Grasp objects", "manipulation", weights, config)
    skills = client.request_skills_for_task("pick up the cup")
"""

# Re-export from new model_client module
from src.platform.cloud.model_client import (
    # Main client
    UnifiedModelClient,

    # Component clients
    BaseModelClient,
    SkillLibraryClient,

    # Data classes
    BaseModelInfo,
    SkillInfo,
    ModelClientConfig,

    # Deprecated alias
    FFMClient,
)

# Re-export legacy types for backwards compatibility
from src.platform.cloud.ffm_client_real import (
    FFMClientReal,
    FFMClientConfig,
    ModelVersion,
)

# Re-export
__all__ = [
    # New unified client (recommended)
    "UnifiedModelClient",
    "BaseModelClient",
    "SkillLibraryClient",
    "BaseModelInfo",
    "SkillInfo",
    "ModelClientConfig",

    # Legacy (deprecated)
    "FFMClient",
    "FFMClientReal",
    "FFMClientConfig",
    "ModelVersion",
]


def get_default_client() -> UnifiedModelClient:
    """Get default model client instance."""
    return UnifiedModelClient()

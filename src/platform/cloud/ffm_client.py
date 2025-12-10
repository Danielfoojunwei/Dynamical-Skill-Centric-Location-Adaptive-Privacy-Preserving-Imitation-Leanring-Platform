"""
FFM Client Module

This module re-exports the real FFM client implementation.
The simulated client has been deprecated in favor of real functionality.

Features:
- Real HTTP API calls to model providers (Physical Intelligence, HuggingFace)
- Cryptographic SHA256 signature verification
- Progress tracking for downloads
- Caching and version management
- Retry logic with exponential backoff

Migration Note:
    The simulated FFMClient has been replaced with FFMClientReal:

    # Old (deprecated):
    client = FFMClient(api_key="key")

    # New:
    from src.platform.cloud.ffm_client import FFMClient
    client = FFMClient(config=FFMClientConfig(api_key="key"))
"""

# Import real implementation
from src.platform.cloud.ffm_client_real import (
    FFMClientReal,
    FFMClientConfig,
    ModelVersion,
)

# Backwards-compatible alias
FFMClient = FFMClientReal

# Re-export
__all__ = [
    "FFMClient",
    "FFMClientReal",
    "FFMClientConfig",
    "ModelVersion",
]

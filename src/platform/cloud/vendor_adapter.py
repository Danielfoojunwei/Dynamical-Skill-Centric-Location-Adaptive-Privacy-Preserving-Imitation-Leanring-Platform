"""
Vendor Adapter Module

This module re-exports the real vendor adapter implementations.
The simulated adapter has been deprecated in favor of real implementations.

Available adapters:
- Pi0VendorAdapter: Physical Intelligence Pi0 models
- OpenVLAVendorAdapter: HuggingFace OpenVLA models
- ACTVendorAdapter: Action Chunking Transformers

For backwards compatibility, VendorAdapter base class is re-exported.

Migration Note:
    The SimulatedVendorAdapter has been removed. Use Pi0VendorAdapter instead:

    # Old (deprecated):
    adapter = SimulatedVendorAdapter()

    # New:
    from src.platform.cloud.vendor_adapter import Pi0VendorAdapter
    adapter = Pi0VendorAdapter()
"""

# Import real implementations
from src.platform.cloud.vendor_adapter_real import (
    VendorAdapter,
    Pi0VendorAdapter,
    OpenVLAVendorAdapter,
    ACTVendorAdapter,
    ModelConfig,
)

# Re-export for backwards compatibility
__all__ = [
    "VendorAdapter",
    "Pi0VendorAdapter",
    "OpenVLAVendorAdapter",
    "ACTVendorAdapter",
    "ModelConfig",
]


# Deprecated alias - prints warning
def SimulatedVendorAdapter(*args, **kwargs):
    """
    DEPRECATED: Use Pi0VendorAdapter instead.

    This function exists only for backwards compatibility and will be removed.
    """
    import warnings
    warnings.warn(
        "SimulatedVendorAdapter is deprecated. Use Pi0VendorAdapter instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return Pi0VendorAdapter(*args, **kwargs)

"""
Vendor Adapter Module

This module provides vendor adapter implementations for VLA models.

Available adapters:
- Pi0VendorAdapter: Physical Intelligence Pi0 models
- OpenVLAVendorAdapter: HuggingFace OpenVLA models
- ACTVendorAdapter: Action Chunking Transformers

Usage:
    from src.platform.cloud.vendor_adapter import Pi0VendorAdapter
    adapter = Pi0VendorAdapter()
    adapter.connect()
    actions = adapter.infer(observation)
"""

# Import real implementations
from src.platform.cloud.vendor_adapter_real import (
    VendorAdapter,
    Pi0VendorAdapter,
    OpenVLAVendorAdapter,
    ACTVendorAdapter,
    ModelConfig,
)

__all__ = [
    "VendorAdapter",
    "Pi0VendorAdapter",
    "OpenVLAVendorAdapter",
    "ACTVendorAdapter",
    "ModelConfig",
]

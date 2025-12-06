"""
Dynamical.ai Edge Device SDK

Complete SDK for deploying and managing Dynamical.ai on edge devices.

Modules:
- jetson_orin_sdk: NVIDIA Jetson Orin hardware integration
- edge_service: Edge device microservice
    - dyglove_sdk: DOGlove haptic glove driver
- onvif_cameras: ONVIF camera integration

Example:
    ```python
    from platform.edge import JetsonOrinSDK, DYGloveSDKClient, ONVIFCameraManager
    
    # Initialize Jetson
    jetson = JetsonOrinSDK()
    print(f"Model: {jetson.get_hardware_info().model}")
    
    # Connect DOGlove
    glove = DYGloveSDKClient()
    glove.connect()
    
    # Discover cameras
    cameras = ONVIFCameraManager()
    cameras.discover()
    ```
"""

from .jetson_orin_sdk import (
    JetsonOrinSDK,
    JetsonHardwareDetector,
    EdgeDeviceConfig,
    HardwareInfo,
    SystemMetrics,
    NetworkInterface,
    ConnectedDevice,
    DeviceStatus,
    ServiceStatus,
    JetsonModel,
    SystemMonitor,
    NetworkManager,
    InstallationManager,
    PowerManager,
)

from .dyglove_sdk import (
    DYGloveSDKClient,
    DYGloveDiscovery,
    DYGloveSimulator,
    DYGloveQualityConfig,
    DYGloveQualityFilter,
    DYGloveAsyncReader,
    DYGloveFK,
    GloveState,
    FingerState,
    Orientation,
    GloveInfo,
    HapticCommand,
    CalibrationData,
    GloveStatus,
    Hand,
    Finger,
    Joint,
)

from .onvif_cameras import (
    ONVIFCamera,
    ONVIFCameraManager,
    ONVIFDiscovery,
    CameraInfo,
    StreamInfo,
    PTZStatus,
    DiscoveredCamera,
    CameraStatus,
)

__all__ = [
    # Jetson SDK
    "JetsonOrinSDK",
    "JetsonHardwareDetector",
    "EdgeDeviceConfig",
    "HardwareInfo",
    "SystemMetrics",
    "NetworkInterface",
    "ConnectedDevice",
    "DeviceStatus",
    "ServiceStatus",
    "JetsonModel",
    "SystemMonitor",
    "NetworkManager",
    "InstallationManager",
    "PowerManager",
    
    # DOGlove SDK
    "DYGloveSDKClient",
    "DYGloveDiscovery",
    "DYGloveSimulator",
    "DYGloveQualityConfig",
    "DYGloveQualityFilter",
    "DYGloveAsyncReader",
    "DYGloveFK",
    "GloveState",
    "FingerState",
    "Orientation",
    "GloveInfo",
    "HapticCommand",
    "CalibrationData",
    "GloveStatus",
    "Hand",
    "Finger",
    "Joint",
    
    # ONVIF Cameras
    "ONVIFCamera",
    "ONVIFCameraManager",
    "ONVIFDiscovery",
    "CameraInfo",
    "StreamInfo",
    "PTZStatus",
    "DiscoveredCamera",
    "CameraStatus",
]

__version__ = "1.0.0"

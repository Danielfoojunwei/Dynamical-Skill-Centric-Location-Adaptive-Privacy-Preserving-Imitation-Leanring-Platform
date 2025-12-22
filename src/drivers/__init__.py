"""
Drivers Module - Hardware Interface Layer

This module provides drivers for all hardware components:
- Cameras: ONVIF IP cameras with RTSP streaming
- Gloves: DYGlove haptic feedback gloves
- PTZ: Pan-Tilt-Zoom camera control
- GMR: Gaussian Mixture Regression library interface

Usage:
    from src.drivers import (
        CameraManager,
        DYGlove,
        DaimonVTLA,
    )

    # Camera setup
    cameras = CameraManager()
    cameras.add_camera("front", rtsp_url="rtsp://...")

    # Glove setup
    glove = DYGlove(port="/dev/ttyUSB0")
    glove.connect()
"""

from .cameras import (
    CameraManager,
    CameraConfig,
    Camera,
)

from .dyglove import (
    DYGlove,
    DYGloveConfig,
    GloveState,
    HapticFeedback,
)

from .glove_calibration import (
    GloveCalibrator,
    CalibrationResult,
)

from .onvif_ptz import (
    ONVIFPTZController,
    PTZConfig,
)

from .daimon_vtla import (
    DaimonVTLA,
    DaimonConfig,
)

__all__ = [
    # Cameras
    'CameraManager',
    'CameraConfig',
    'Camera',

    # Gloves
    'DYGlove',
    'DYGloveConfig',
    'GloveState',
    'HapticFeedback',
    'GloveCalibrator',
    'CalibrationResult',

    # PTZ
    'ONVIFPTZController',
    'PTZConfig',

    # Robot adapters
    'DaimonVTLA',
    'DaimonConfig',
]

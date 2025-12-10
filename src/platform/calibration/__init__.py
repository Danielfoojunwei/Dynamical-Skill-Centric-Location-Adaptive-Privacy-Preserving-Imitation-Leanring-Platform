"""
Dynamical.ai Calibration Services

Comprehensive calibration microservices for edge devices, sensors, and tracking systems.

Modules:
- mmpose_calibration: MMPose camera placement and pose estimation calibration
- doglove_calibration: DOGlove wireless conversion and sensor calibration

Camera Placement for Optimal MMPose Performance:
- Minimum 2 cameras for 3D reconstruction
- Optimal 4-6 cameras for full workspace coverage
- 60-120 degree separation between adjacent cameras
- Height: 1.5-2.5m above ground
- Tilt angle: 15-45 degrees downward

DOGlove Calibration for Wireless Operation:
- IMU calibration (accelerometer, gyroscope, magnetometer)
- Joint encoder calibration (21-DOF)
- Haptic force feedback calibration (5-DOF)
- Wireless link calibration (BLE/WiFi)
- Cross-calibration with MMPose
"""

from .mmpose_calibration import (
    # Service
    MMPoseCalibrationService,
    
    # Camera placement
    CameraPlacementOptimizer,
    CameraPlacementType,
    PlacementRecommendation,
    
    # Camera configuration
    CameraConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    WorkspaceConfig,
    
    # MMDeploy integration
    MMDeployManager,
    PoseModel,
    Backend,
    
    # Calibration
    TriangulationCalibrator,
    CalibrationResult,
    CalibrationStatus,
)

from .doglove_calibration import (
    # Service
    DOGloveCalibrationService,
    
    # Calibrators
    IMUCalibrator,
    JointEncoderCalibrator,
    HapticCalibrator,
    WirelessCalibrator,
    CrossCalibrator,
    
    # Data classes
    IMUCalibrationData,
    JointCalibrationData,
    HapticCalibrationData,
    WirelessCalibrationData,
    CrossCalibrationData,
    FullCalibrationResult,
    
    # Enums
    CalibrationPhase,
    WirelessProtocol,
    Hand,
    Finger,
    Joint,
)

__all__ = [
    # MMPose Calibration
    "MMPoseCalibrationService",
    "CameraPlacementOptimizer",
    "CameraPlacementType",
    "PlacementRecommendation",
    "CameraConfig",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "WorkspaceConfig",
    "MMDeployManager",
    "PoseModel",
    "Backend",
    "TriangulationCalibrator",
    "CalibrationResult",
    "CalibrationStatus",
    
    # DOGlove Calibration
    "DOGloveCalibrationService",
    "IMUCalibrator",
    "JointEncoderCalibrator",
    "HapticCalibrator",
    "WirelessCalibrator",
    "CrossCalibrator",
    "IMUCalibrationData",
    "JointCalibrationData",
    "HapticCalibrationData",
    "WirelessCalibrationData",
    "CrossCalibrationData",
    "FullCalibrationResult",
    "CalibrationPhase",
    "WirelessProtocol",
    "Hand",
    "Finger",
    "Joint",
]

__version__ = "1.0.0"

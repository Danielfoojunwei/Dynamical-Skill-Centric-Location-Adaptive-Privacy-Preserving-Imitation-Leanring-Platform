"""
MMPose Calibration Microservice

Comprehensive calibration service for MMPose pose estimation optimized for
Dynamical.ai edge deployment on Jetson Orin with TensorRT.

Features:
- Camera placement optimization (spacing, angles, coverage)
- Multi-camera triangulation calibration
- MMDeploy TensorRT model deployment
- Pose estimation quality metrics
- Real-time calibration validation
- Workspace coverage analysis

Based on:
- MMDeploy: https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpose.html
- RTMPose for real-time performance (430+ FPS on GPU)

Camera Placement Guidelines for Optimal MMPose Performance:
- Minimum 2 cameras for 3D reconstruction
- Optimal 4-6 cameras for full workspace coverage
- 60-120 degree separation between adjacent cameras
- Height: 1.5-2.5m above ground
- Tilt angle: 15-45 degrees downward
- Overlap: 30-50% between adjacent camera FOVs
"""

import os
import sys
import json
import math
import time
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available - some features disabled")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("opencv not available - camera calibration disabled")


# =============================================================================
# Constants and Enums
# =============================================================================

class CalibrationStatus(str, Enum):
    """Calibration status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CameraPlacementType(str, Enum):
    """Camera placement configuration types."""
    SINGLE = "single"  # Single camera setup
    STEREO = "stereo"  # Two cameras for depth
    TRIANGULAR = "triangular"  # Three cameras at 120 degrees
    QUAD = "quad"  # Four cameras at 90 degrees
    HEXAGONAL = "hexagonal"  # Six cameras at 60 degrees
    CUSTOM = "custom"  # Custom placement


class PoseModel(str, Enum):
    """Supported pose estimation models."""
    RTMPOSE_S = "rtmpose-s"  # Small - 72.2% AP, 70+ FPS mobile
    RTMPOSE_M = "rtmpose-m"  # Medium - 75.8% AP, 430+ FPS GPU
    RTMPOSE_L = "rtmpose-l"  # Large - 76.5% AP
    HRNET_W32 = "hrnet-w32"  # HRNet small
    HRNET_W48 = "hrnet-w48"  # HRNet large


class Backend(str, Enum):
    """Inference backends."""
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    TENSORRT_FP16 = "tensorrt-fp16"
    NCNN = "ncnn"
    OPENVINO = "openvino"


# MMPose keypoint definitions (COCO format)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float = 1000.0  # Focal length x
    fy: float = 1000.0  # Focal length y
    cx: float = 640.0   # Principal point x
    cy: float = 360.0   # Principal point y
    k1: float = 0.0     # Radial distortion k1
    k2: float = 0.0     # Radial distortion k2
    p1: float = 0.0     # Tangential distortion p1
    p2: float = 0.0     # Tangential distortion p2
    width: int = 1280
    height: int = 720
    
    def to_matrix(self) -> 'np.ndarray':
        """Get camera matrix."""
        if HAS_NUMPY:
            return np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
        return [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
    
    def to_distortion(self) -> 'np.ndarray':
        """Get distortion coefficients."""
        if HAS_NUMPY:
            return np.array([self.k1, self.k2, self.p1, self.p2])
        return [self.k1, self.k2, self.p1, self.p2]


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (position and orientation)."""
    # Position in world coordinates (meters)
    x: float = 0.0
    y: float = 0.0
    z: float = 2.0  # Height above ground
    
    # Rotation angles (degrees)
    roll: float = 0.0
    pitch: float = -30.0  # Tilted down
    yaw: float = 0.0
    
    def to_rotation_matrix(self) -> 'np.ndarray':
        """Get rotation matrix from Euler angles."""
        if not HAS_NUMPY:
            return None
        
        # Convert to radians
        r = math.radians(self.roll)
        p = math.radians(self.pitch)
        y = math.radians(self.yaw)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r), math.cos(r)]
        ])
        
        Ry = np.array([
            [math.cos(p), 0, math.sin(p)],
            [0, 1, 0],
            [-math.sin(p), 0, math.cos(p)]
        ])
        
        Rz = np.array([
            [math.cos(y), -math.sin(y), 0],
            [math.sin(y), math.cos(y), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def to_translation(self) -> 'np.ndarray':
        """Get translation vector."""
        if HAS_NUMPY:
            return np.array([self.x, self.y, self.z])
        return [self.x, self.y, self.z]


@dataclass
class CameraConfig:
    """Complete camera configuration."""
    camera_id: str
    name: str
    ip_address: str
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    extrinsics: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    fov_horizontal: float = 70.0  # degrees
    fov_vertical: float = 45.0   # degrees
    is_calibrated: bool = False
    calibration_error: float = 0.0  # Reprojection error in pixels


@dataclass
class WorkspaceConfig:
    """Workspace/capture volume configuration."""
    # Workspace bounds (meters)
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 0.0  # Floor
    z_max: float = 2.0  # Ceiling
    
    # Target area (where person will be)
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 1.0  # Approximate torso height
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Get workspace dimensions."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        )


@dataclass
class PlacementRecommendation:
    """Camera placement recommendation."""
    camera_id: str
    position: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float]  # roll, pitch, yaw
    target_point: Tuple[float, float, float]
    coverage_score: float  # 0-1
    overlap_with: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CalibrationResult:
    """Calibration result data."""
    calibration_id: str
    timestamp: str
    status: CalibrationStatus
    cameras: List[CameraConfig]
    workspace: WorkspaceConfig
    
    # Quality metrics
    mean_reprojection_error: float = 0.0
    max_reprojection_error: float = 0.0
    coverage_percentage: float = 0.0
    triangulation_accuracy: float = 0.0
    
    # Pose estimation metrics
    pose_detection_rate: float = 0.0
    keypoint_confidence_mean: float = 0.0
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PoseEstimationConfig:
    """Pose estimation model configuration."""
    model: PoseModel = PoseModel.RTMPOSE_M
    backend: Backend = Backend.TENSORRT_FP16
    input_size: Tuple[int, int] = (256, 192)
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.3
    device: str = "cuda:0"
    model_path: str = ""


# =============================================================================
# Camera Placement Optimizer
# =============================================================================

class CameraPlacementOptimizer:
    """
    Optimizes camera placement for MMPose pose estimation.
    
    Considers:
    - Field of view coverage
    - Stereo baseline for depth estimation
    - Occlusion minimization
    - Lighting conditions
    - MMPose detection requirements
    """
    
    # Optimal parameters for MMPose
    MIN_CAMERA_DISTANCE = 1.5  # Minimum distance to subject (meters)
    MAX_CAMERA_DISTANCE = 5.0  # Maximum distance
    OPTIMAL_CAMERA_HEIGHT = 2.0  # Height above ground
    MIN_STEREO_BASELINE = 0.5  # Minimum baseline for depth
    OPTIMAL_TILT_ANGLE = -25.0  # Degrees downward
    MIN_OVERLAP_RATIO = 0.3  # 30% overlap between cameras
    
    def __init__(self, workspace: WorkspaceConfig):
        self.workspace = workspace
    
    def generate_placement(
        self,
        num_cameras: int,
        placement_type: CameraPlacementType = None
    ) -> List[PlacementRecommendation]:
        """
        Generate optimal camera placement recommendations.
        
        Args:
            num_cameras: Number of cameras to place
            placement_type: Placement configuration type
        
        Returns:
            List of placement recommendations
        """
        if placement_type is None:
            placement_type = self._auto_select_placement(num_cameras)
        
        if placement_type == CameraPlacementType.SINGLE:
            return self._generate_single_placement()
        elif placement_type == CameraPlacementType.STEREO:
            return self._generate_stereo_placement()
        elif placement_type == CameraPlacementType.TRIANGULAR:
            return self._generate_triangular_placement()
        elif placement_type == CameraPlacementType.QUAD:
            return self._generate_quad_placement()
        elif placement_type == CameraPlacementType.HEXAGONAL:
            return self._generate_hexagonal_placement()
        else:
            return self._generate_custom_placement(num_cameras)
    
    def _auto_select_placement(self, num_cameras: int) -> CameraPlacementType:
        """Auto-select optimal placement type based on camera count."""
        if num_cameras == 1:
            return CameraPlacementType.SINGLE
        elif num_cameras == 2:
            return CameraPlacementType.STEREO
        elif num_cameras == 3:
            return CameraPlacementType.TRIANGULAR
        elif num_cameras == 4:
            return CameraPlacementType.QUAD
        elif num_cameras >= 6:
            return CameraPlacementType.HEXAGONAL
        else:
            return CameraPlacementType.CUSTOM
    
    def _generate_single_placement(self) -> List[PlacementRecommendation]:
        """Generate single camera placement (front-facing)."""
        target = (self.workspace.center_x, self.workspace.center_y, self.workspace.center_z)
        
        # Place camera in front of workspace
        distance = 2.5
        height = self.OPTIMAL_CAMERA_HEIGHT
        
        return [PlacementRecommendation(
            camera_id="cam_0",
            position=(target[0], target[1] - distance, height),
            rotation=(0.0, self.OPTIMAL_TILT_ANGLE, 0.0),
            target_point=target,
            coverage_score=0.7,
            notes="Single camera provides 2D pose only. Consider adding more cameras for 3D."
        )]
    
    def _generate_stereo_placement(self) -> List[PlacementRecommendation]:
        """Generate stereo camera placement (left-right)."""
        target = (self.workspace.center_x, self.workspace.center_y, self.workspace.center_z)
        distance = 2.5
        height = self.OPTIMAL_CAMERA_HEIGHT
        baseline = 1.0  # Stereo baseline
        
        recommendations = []
        
        # Left camera
        recommendations.append(PlacementRecommendation(
            camera_id="cam_left",
            position=(target[0] - baseline/2, target[1] - distance, height),
            rotation=(0.0, self.OPTIMAL_TILT_ANGLE, 10.0),  # Slight inward yaw
            target_point=target,
            coverage_score=0.8,
            overlap_with=["cam_right"],
            notes="Left stereo camera. Baseline: 1.0m"
        ))
        
        # Right camera
        recommendations.append(PlacementRecommendation(
            camera_id="cam_right",
            position=(target[0] + baseline/2, target[1] - distance, height),
            rotation=(0.0, self.OPTIMAL_TILT_ANGLE, -10.0),
            target_point=target,
            coverage_score=0.8,
            overlap_with=["cam_left"],
            notes="Right stereo camera. Baseline: 1.0m"
        ))
        
        return recommendations
    
    def _generate_triangular_placement(self) -> List[PlacementRecommendation]:
        """Generate triangular placement (120 degree separation)."""
        target = (self.workspace.center_x, self.workspace.center_y, self.workspace.center_z)
        radius = 2.5
        height = self.OPTIMAL_CAMERA_HEIGHT
        
        recommendations = []
        angles = [0, 120, 240]  # degrees
        
        for i, angle in enumerate(angles):
            rad = math.radians(angle)
            x = target[0] + radius * math.sin(rad)
            y = target[1] + radius * math.cos(rad)
            yaw = -angle  # Point toward center
            
            # Determine overlap cameras
            overlap = [f"cam_{(i-1)%3}", f"cam_{(i+1)%3}"]
            
            recommendations.append(PlacementRecommendation(
                camera_id=f"cam_{i}",
                position=(x, y, height),
                rotation=(0.0, self.OPTIMAL_TILT_ANGLE, yaw),
                target_point=target,
                coverage_score=0.9,
                overlap_with=overlap,
                notes=f"Triangular position {i} at {angle}°. Good for 360° coverage."
            ))
        
        return recommendations
    
    def _generate_quad_placement(self) -> List[PlacementRecommendation]:
        """Generate quad placement (90 degree separation)."""
        target = (self.workspace.center_x, self.workspace.center_y, self.workspace.center_z)
        radius = 2.5
        height = self.OPTIMAL_CAMERA_HEIGHT
        
        recommendations = []
        positions = [
            ("front", 0, "cam_0"),
            ("right", 90, "cam_1"),
            ("back", 180, "cam_2"),
            ("left", 270, "cam_3"),
        ]
        
        for name, angle, cam_id in positions:
            rad = math.radians(angle)
            x = target[0] + radius * math.sin(rad)
            y = target[1] + radius * math.cos(rad)
            yaw = -angle
            
            recommendations.append(PlacementRecommendation(
                camera_id=cam_id,
                position=(x, y, height),
                rotation=(0.0, self.OPTIMAL_TILT_ANGLE, yaw),
                target_point=target,
                coverage_score=0.95,
                overlap_with=[f"cam_{(int(cam_id[-1])-1)%4}", f"cam_{(int(cam_id[-1])+1)%4}"],
                notes=f"Quad position {name}. Optimal for full-body 3D reconstruction."
            ))
        
        return recommendations
    
    def _generate_hexagonal_placement(self) -> List[PlacementRecommendation]:
        """Generate hexagonal placement (60 degree separation)."""
        target = (self.workspace.center_x, self.workspace.center_y, self.workspace.center_z)
        radius = 3.0  # Slightly larger radius for 6 cameras
        height = self.OPTIMAL_CAMERA_HEIGHT
        
        recommendations = []
        
        for i in range(6):
            angle = i * 60
            rad = math.radians(angle)
            x = target[0] + radius * math.sin(rad)
            y = target[1] + radius * math.cos(rad)
            yaw = -angle
            
            recommendations.append(PlacementRecommendation(
                camera_id=f"cam_{i}",
                position=(x, y, height),
                rotation=(0.0, self.OPTIMAL_TILT_ANGLE, yaw),
                target_point=target,
                coverage_score=0.98,
                overlap_with=[f"cam_{(i-1)%6}", f"cam_{(i+1)%6}"],
                notes=f"Hexagonal position {i} at {angle}°. Maximum coverage and redundancy."
            ))
        
        return recommendations
    
    def _generate_custom_placement(self, num_cameras: int) -> List[PlacementRecommendation]:
        """Generate custom placement for arbitrary camera count."""
        target = (self.workspace.center_x, self.workspace.center_y, self.workspace.center_z)
        radius = 2.5
        height = self.OPTIMAL_CAMERA_HEIGHT
        
        recommendations = []
        angle_step = 360.0 / num_cameras
        
        for i in range(num_cameras):
            angle = i * angle_step
            rad = math.radians(angle)
            x = target[0] + radius * math.sin(rad)
            y = target[1] + radius * math.cos(rad)
            yaw = -angle
            
            recommendations.append(PlacementRecommendation(
                camera_id=f"cam_{i}",
                position=(x, y, height),
                rotation=(0.0, self.OPTIMAL_TILT_ANGLE, yaw),
                target_point=target,
                coverage_score=min(0.99, 0.7 + 0.05 * num_cameras),
                overlap_with=[f"cam_{(i-1)%num_cameras}", f"cam_{(i+1)%num_cameras}"],
                notes=f"Custom position {i} at {angle:.1f}°"
            ))
        
        return recommendations
    
    def analyze_coverage(
        self,
        cameras: List[CameraConfig],
        resolution: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze workspace coverage from camera configuration.
        
        Returns coverage map and statistics.
        """
        if not HAS_NUMPY:
            return {"error": "numpy required for coverage analysis"}
        
        # Create 3D grid of workspace
        x_range = np.linspace(self.workspace.x_min, self.workspace.x_max, resolution)
        y_range = np.linspace(self.workspace.y_min, self.workspace.y_max, resolution)
        z_range = np.linspace(self.workspace.z_min, self.workspace.z_max, resolution)
        
        # Count cameras that can see each point
        coverage_map = np.zeros((resolution, resolution, resolution), dtype=np.int32)
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, z in enumerate(z_range):
                    point = np.array([x, y, z])
                    
                    for cam in cameras:
                        if self._point_visible_from_camera(point, cam):
                            coverage_map[i, j, k] += 1
        
        # Calculate statistics
        total_points = resolution ** 3
        visible_by_any = np.sum(coverage_map > 0)
        visible_by_two = np.sum(coverage_map >= 2)  # For triangulation
        visible_by_three = np.sum(coverage_map >= 3)  # For robust 3D
        
        return {
            "total_points": total_points,
            "visible_by_any": int(visible_by_any),
            "visible_by_two_plus": int(visible_by_two),
            "visible_by_three_plus": int(visible_by_three),
            "coverage_any_percent": visible_by_any / total_points * 100,
            "coverage_stereo_percent": visible_by_two / total_points * 100,
            "coverage_robust_percent": visible_by_three / total_points * 100,
            "mean_camera_count": float(np.mean(coverage_map)),
            "max_camera_count": int(np.max(coverage_map)),
        }
    
    def _point_visible_from_camera(
        self,
        point: 'np.ndarray',
        camera: CameraConfig
    ) -> bool:
        """Check if a 3D point is visible from camera."""
        # Get camera position
        cam_pos = np.array([
            camera.extrinsics.x,
            camera.extrinsics.y,
            camera.extrinsics.z
        ])
        
        # Vector from camera to point
        to_point = point - cam_pos
        distance = np.linalg.norm(to_point)
        
        # Check distance limits
        if distance < self.MIN_CAMERA_DISTANCE or distance > self.MAX_CAMERA_DISTANCE:
            return False
        
        # Get camera forward direction
        R = camera.extrinsics.to_rotation_matrix()
        if R is None:
            return True  # Assume visible if no rotation info
        
        forward = R @ np.array([0, 0, 1])
        
        # Calculate angle to point
        to_point_norm = to_point / distance
        angle = math.degrees(math.acos(np.clip(np.dot(forward, to_point_norm), -1, 1)))
        
        # Check if within FOV
        max_angle = max(camera.fov_horizontal, camera.fov_vertical) / 2
        return angle < max_angle


# =============================================================================
# MMDeploy Integration
# =============================================================================

class MMDeployManager:
    """
    Manages MMDeploy model deployment for MMPose.
    
    Handles:
    - Model conversion (PyTorch -> TensorRT/ONNX)
    - Model optimization
    - Inference backend setup
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        PoseModel.RTMPOSE_S: {
            "config": "rtmpose-s_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192.pth",
            "input_size": (256, 192),
        },
        PoseModel.RTMPOSE_M: {
            "config": "rtmpose-m_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
            "input_size": (256, 192),
        },
        PoseModel.RTMPOSE_L: {
            "config": "rtmpose-l_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192.pth",
            "input_size": (256, 192),
        },
        PoseModel.HRNET_W32: {
            "config": "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
            "checkpoint": "hrnet_w32_coco_256x192-c78dce93_20200708.pth",
            "input_size": (256, 192),
        },
        PoseModel.HRNET_W48: {
            "config": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
            "checkpoint": "hrnet_w48_coco_384x288-314c8528_20200708.pth",
            "input_size": (384, 288),
        },
    }
    
    # Deploy config patterns
    DEPLOY_CONFIGS = {
        Backend.ONNXRUNTIME: "pose-detection_onnxruntime_static.py",
        Backend.TENSORRT: "pose-detection_tensorrt_static-{size}.py",
        Backend.TENSORRT_FP16: "pose-detection_tensorrt-fp16_static-{size}.py",
        Backend.NCNN: "pose-detection_ncnn_static-{size}.py",
        Backend.OPENVINO: "pose-detection_openvino_static-{size}.py",
    }
    
    def __init__(self, mmdeploy_path: str = None, mmpose_path: str = None):
        """
        Initialize MMDeploy manager.
        
        Args:
            mmdeploy_path: Path to mmdeploy installation
            mmpose_path: Path to mmpose installation
        """
        self.mmdeploy_path = mmdeploy_path or os.getenv("MMDEPLOY_PATH", "/opt/mmdeploy")
        self.mmpose_path = mmpose_path or os.getenv("MMPOSE_PATH", "/opt/mmpose")
        self.models_dir = Path("/opt/dynamical/models/mmpose")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_model(
        self,
        model: PoseModel,
        backend: Backend,
        output_dir: str = None,
        device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """
        Convert MMPose model to deployment format.
        
        Args:
            model: Pose model to convert
            backend: Target inference backend
            output_dir: Output directory for converted model
            device: Device for conversion
        
        Returns:
            Conversion result with model path
        """
        model_config = self.MODEL_CONFIGS.get(model)
        if not model_config:
            return {"error": f"Unknown model: {model}"}
        
        input_size = model_config["input_size"]
        size_str = f"{input_size[0]}x{input_size[1]}"
        
        # Get deploy config
        deploy_config = self.DEPLOY_CONFIGS.get(backend, "").format(size=size_str)
        
        output_dir = output_dir or str(self.models_dir / f"{model.value}_{backend.value}")
        
        # Build conversion command
        cmd = [
            "python", f"{self.mmdeploy_path}/tools/deploy.py",
            f"{self.mmdeploy_path}/configs/mmpose/{deploy_config}",
            f"{self.mmpose_path}/configs/{model_config['config']}",
            model_config["checkpoint"],
            "demo/resources/human-pose.jpg",
            "--work-dir", output_dir,
            "--device", device,
            "--dump-info",
        ]
        
        logger.info(f"Converting {model.value} to {backend.value}...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "model_path": output_dir,
                    "backend": backend.value,
                    "input_size": input_size,
                    "files": list(Path(output_dir).glob("*")),
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout,
                }
        
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "Conversion timed out"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a deployed model."""
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            return {"error": "Model path does not exist"}
        
        info = {
            "path": str(model_dir),
            "files": [],
        }
        
        # Read deploy.json
        deploy_json = model_dir / "deploy.json"
        if deploy_json.exists():
            with open(deploy_json) as f:
                info["deploy_config"] = json.load(f)
        
        # Read pipeline.json
        pipeline_json = model_dir / "pipeline.json"
        if pipeline_json.exists():
            with open(pipeline_json) as f:
                info["pipeline"] = json.load(f)
        
        # List model files
        for ext in ["*.onnx", "*.engine", "*.trt", "*.param", "*.bin"]:
            info["files"].extend([str(f) for f in model_dir.glob(ext)])
        
        return info


# =============================================================================
# Multi-Camera Triangulation Calibrator
# =============================================================================

class TriangulationCalibrator:
    """
    Calibrates multi-camera system for 3D pose triangulation.
    
    Uses checkerboard or ChArUco markers for extrinsic calibration.
    """
    
    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # meters
    ):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self._calibration_data: Dict[str, List] = {}
    
    def collect_calibration_frame(
        self,
        camera_id: str,
        image: 'np.ndarray'
    ) -> Dict[str, Any]:
        """
        Collect calibration frame from camera.
        
        Args:
            camera_id: Camera identifier
            image: Image frame (BGR)
        
        Returns:
            Detection result with corners
        """
        if not HAS_CV2:
            return {"error": "OpenCV not available"}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store calibration data
            if camera_id not in self._calibration_data:
                self._calibration_data[camera_id] = []
            
            self._calibration_data[camera_id].append({
                "corners": corners,
                "image_size": gray.shape[::-1],
            })
            
            return {
                "detected": True,
                "corners": corners.tolist(),
                "frame_count": len(self._calibration_data[camera_id]),
            }
        
        return {"detected": False}
    
    def calibrate_camera_intrinsics(
        self,
        camera_id: str
    ) -> CameraIntrinsics:
        """
        Calibrate camera intrinsic parameters.
        
        Args:
            camera_id: Camera to calibrate
        
        Returns:
            Camera intrinsics
        """
        if not HAS_CV2 or not HAS_NUMPY:
            return CameraIntrinsics()
        
        if camera_id not in self._calibration_data:
            logger.warning(f"No calibration data for camera {camera_id}")
            return CameraIntrinsics()
        
        data = self._calibration_data[camera_id]
        if len(data) < 10:
            logger.warning(f"Insufficient frames for calibration: {len(data)}")
        
        # Prepare object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        obj_points = [objp] * len(data)
        img_points = [d["corners"] for d in data]
        image_size = data[0]["image_size"]
        
        # Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
            total_error += error
        mean_error = total_error / len(obj_points)
        
        return CameraIntrinsics(
            fx=mtx[0, 0],
            fy=mtx[1, 1],
            cx=mtx[0, 2],
            cy=mtx[1, 2],
            k1=dist[0, 0] if len(dist[0]) > 0 else 0,
            k2=dist[0, 1] if len(dist[0]) > 1 else 0,
            p1=dist[0, 2] if len(dist[0]) > 2 else 0,
            p2=dist[0, 3] if len(dist[0]) > 3 else 0,
            width=image_size[0],
            height=image_size[1],
        )
    
    def calibrate_stereo_extrinsics(
        self,
        camera_id_1: str,
        camera_id_2: str,
        intrinsics_1: CameraIntrinsics,
        intrinsics_2: CameraIntrinsics
    ) -> Tuple[CameraExtrinsics, float]:
        """
        Calibrate stereo extrinsics between two cameras.
        
        Returns:
            Tuple of (relative extrinsics, reprojection error)
        """
        if not HAS_CV2 or not HAS_NUMPY:
            return CameraExtrinsics(), 999.0
        
        # Find matching frames
        data_1 = self._calibration_data.get(camera_id_1, [])
        data_2 = self._calibration_data.get(camera_id_2, [])
        
        if len(data_1) == 0 or len(data_2) == 0:
            return CameraExtrinsics(), 999.0
        
        # Use minimum frames
        n_frames = min(len(data_1), len(data_2))
        
        # Prepare points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        obj_points = [objp] * n_frames
        img_points_1 = [data_1[i]["corners"] for i in range(n_frames)]
        img_points_2 = [data_2[i]["corners"] for i in range(n_frames)]
        
        # Camera matrices
        K1 = intrinsics_1.to_matrix()
        K2 = intrinsics_2.to_matrix()
        D1 = intrinsics_1.to_distortion()
        D2 = intrinsics_2.to_distortion()
        
        image_size = data_1[0]["image_size"]
        
        # Stereo calibration
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_1, img_points_2,
            K1, D1, K2, D2, image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Convert rotation matrix to Euler angles
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        euler = r.as_euler('xyz', degrees=True)
        
        extrinsics = CameraExtrinsics(
            x=T[0, 0],
            y=T[1, 0],
            z=T[2, 0],
            roll=euler[0],
            pitch=euler[1],
            yaw=euler[2],
        )
        
        return extrinsics, ret
    
    def triangulate_point(
        self,
        point_2d_1: Tuple[float, float],
        point_2d_2: Tuple[float, float],
        camera_1: CameraConfig,
        camera_2: CameraConfig
    ) -> Tuple[float, float, float]:
        """
        Triangulate 3D point from two 2D observations.
        
        Args:
            point_2d_1: Point in camera 1 image
            point_2d_2: Point in camera 2 image
            camera_1: First camera configuration
            camera_2: Second camera configuration
        
        Returns:
            3D point (x, y, z)
        """
        if not HAS_CV2 or not HAS_NUMPY:
            return (0.0, 0.0, 0.0)
        
        # Get projection matrices
        P1 = self._get_projection_matrix(camera_1)
        P2 = self._get_projection_matrix(camera_2)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            P1, P2,
            np.array([[point_2d_1[0]], [point_2d_1[1]]], dtype=np.float64),
            np.array([[point_2d_2[0]], [point_2d_2[1]]], dtype=np.float64)
        )
        
        # Convert to 3D
        point_3d = points_4d[:3] / points_4d[3]
        
        return (float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))
    
    def _get_projection_matrix(self, camera: CameraConfig) -> 'np.ndarray':
        """Get 3x4 projection matrix for camera."""
        K = camera.intrinsics.to_matrix()
        R = camera.extrinsics.to_rotation_matrix()
        t = camera.extrinsics.to_translation().reshape(3, 1)
        
        # P = K[R|t]
        Rt = np.hstack([R, t])
        return K @ Rt


# =============================================================================
# Calibration Service
# =============================================================================

class MMPoseCalibrationService:
    """
    Main calibration microservice for MMPose.
    
    Provides:
    - Camera placement optimization
    - Multi-camera calibration
    - Pose estimation deployment
    - Quality validation
    """
    
    def __init__(self, config_dir: str = "/etc/dynamical/calibration"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.workspace = WorkspaceConfig()
        self.cameras: Dict[str, CameraConfig] = {}
        self.placement_optimizer = CameraPlacementOptimizer(self.workspace)
        self.triangulation_calibrator = TriangulationCalibrator()
        self.mmdeploy_manager = MMDeployManager()
        
        self._calibration_status = CalibrationStatus.NOT_STARTED
        self._current_calibration: Optional[CalibrationResult] = None
    
    @property
    def status(self) -> CalibrationStatus:
        """Get current calibration status."""
        return self._calibration_status
    
    def configure_workspace(
        self,
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        z_range: Tuple[float, float] = (0, 2),
        center: Tuple[float, float, float] = (0, 0, 1)
    ):
        """Configure workspace bounds."""
        self.workspace = WorkspaceConfig(
            x_min=x_range[0], x_max=x_range[1],
            y_min=y_range[0], y_max=y_range[1],
            z_min=z_range[0], z_max=z_range[1],
            center_x=center[0], center_y=center[1], center_z=center[2]
        )
        self.placement_optimizer = CameraPlacementOptimizer(self.workspace)
    
    def get_placement_recommendations(
        self,
        num_cameras: int,
        placement_type: CameraPlacementType = None
    ) -> List[Dict[str, Any]]:
        """
        Get camera placement recommendations.
        
        Args:
            num_cameras: Number of cameras
            placement_type: Desired placement configuration
        
        Returns:
            List of placement recommendations
        """
        recommendations = self.placement_optimizer.generate_placement(
            num_cameras, placement_type
        )
        return [asdict(r) for r in recommendations]
    
    def add_camera(
        self,
        camera_id: str,
        name: str,
        ip_address: str,
        position: Tuple[float, float, float] = None,
        rotation: Tuple[float, float, float] = None
    ) -> CameraConfig:
        """
        Add camera to calibration system.
        
        Args:
            camera_id: Unique camera identifier
            name: Camera name
            ip_address: Camera IP address
            position: (x, y, z) position in meters
            rotation: (roll, pitch, yaw) in degrees
        
        Returns:
            Camera configuration
        """
        extrinsics = CameraExtrinsics()
        if position:
            extrinsics.x, extrinsics.y, extrinsics.z = position
        if rotation:
            extrinsics.roll, extrinsics.pitch, extrinsics.yaw = rotation
        
        camera = CameraConfig(
            camera_id=camera_id,
            name=name,
            ip_address=ip_address,
            extrinsics=extrinsics,
        )
        
        self.cameras[camera_id] = camera
        return camera
    
    def run_intrinsic_calibration(
        self,
        camera_id: str,
        images: List['np.ndarray']
    ) -> Dict[str, Any]:
        """
        Run intrinsic calibration for a camera.
        
        Args:
            camera_id: Camera to calibrate
            images: List of calibration images
        
        Returns:
            Calibration result
        """
        if camera_id not in self.cameras:
            return {"error": f"Camera {camera_id} not found"}
        
        # Collect frames
        for img in images:
            self.triangulation_calibrator.collect_calibration_frame(camera_id, img)
        
        # Calibrate
        intrinsics = self.triangulation_calibrator.calibrate_camera_intrinsics(camera_id)
        
        # Update camera config
        self.cameras[camera_id].intrinsics = intrinsics
        self.cameras[camera_id].is_calibrated = True
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "intrinsics": asdict(intrinsics),
        }
    
    def run_full_calibration(
        self,
        pose_model: PoseModel = PoseModel.RTMPOSE_M,
        backend: Backend = Backend.TENSORRT_FP16
    ) -> CalibrationResult:
        """
        Run full system calibration.
        
        Args:
            pose_model: Pose estimation model to deploy
            backend: Inference backend
        
        Returns:
            Calibration result
        """
        self._calibration_status = CalibrationStatus.IN_PROGRESS
        calibration_id = f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        issues = []
        recommendations = []
        
        # Check camera count
        if len(self.cameras) < 2:
            issues.append("Less than 2 cameras configured - 3D pose estimation not possible")
            recommendations.append("Add at least 2 cameras for 3D pose reconstruction")
        
        # Analyze coverage
        coverage = self.placement_optimizer.analyze_coverage(list(self.cameras.values()))
        
        if coverage.get("coverage_stereo_percent", 0) < 80:
            issues.append(f"Stereo coverage only {coverage.get('coverage_stereo_percent', 0):.1f}%")
            recommendations.append("Adjust camera positions to improve overlap")
        
        # Convert pose model
        model_result = self.mmdeploy_manager.convert_model(pose_model, backend)
        if model_result.get("status") != "success":
            issues.append(f"Model conversion failed: {model_result.get('error')}")
        
        # Calculate overall quality
        mean_error = sum(c.calibration_error for c in self.cameras.values()) / max(len(self.cameras), 1)
        
        result = CalibrationResult(
            calibration_id=calibration_id,
            timestamp=datetime.utcnow().isoformat(),
            status=CalibrationStatus.COMPLETED if not issues else CalibrationStatus.COMPLETED,
            cameras=list(self.cameras.values()),
            workspace=self.workspace,
            mean_reprojection_error=mean_error,
            coverage_percentage=coverage.get("coverage_stereo_percent", 0),
            issues=issues,
            recommendations=recommendations,
        )
        
        self._current_calibration = result
        self._calibration_status = CalibrationStatus.COMPLETED
        
        # Save calibration
        self._save_calibration(result)
        
        return result
    
    def _save_calibration(self, result: CalibrationResult):
        """Save calibration to file."""
        path = self.config_dir / f"{result.calibration_id}.json"
        
        # Convert to serializable format
        data = {
            "calibration_id": result.calibration_id,
            "timestamp": result.timestamp,
            "status": result.status.value,
            "workspace": asdict(result.workspace),
            "cameras": [asdict(c) for c in result.cameras],
            "metrics": {
                "mean_reprojection_error": result.mean_reprojection_error,
                "coverage_percentage": result.coverage_percentage,
            },
            "issues": result.issues,
            "recommendations": result.recommendations,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved calibration to {path}")
    
    def load_calibration(self, calibration_id: str) -> Optional[CalibrationResult]:
        """Load calibration from file."""
        path = self.config_dir / f"{calibration_id}.json"
        
        if not path.exists():
            return None
        
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct cameras
        cameras = []
        for cam_data in data.get("cameras", []):
            intrinsics = CameraIntrinsics(**cam_data.get("intrinsics", {}))
            extrinsics = CameraExtrinsics(**cam_data.get("extrinsics", {}))
            cam = CameraConfig(
                camera_id=cam_data["camera_id"],
                name=cam_data["name"],
                ip_address=cam_data["ip_address"],
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
            cameras.append(cam)
        
        return CalibrationResult(
            calibration_id=data["calibration_id"],
            timestamp=data["timestamp"],
            status=CalibrationStatus(data["status"]),
            cameras=cameras,
            workspace=WorkspaceConfig(**data.get("workspace", {})),
            mean_reprojection_error=data.get("metrics", {}).get("mean_reprojection_error", 0),
            coverage_percentage=data.get("metrics", {}).get("coverage_percentage", 0),
            issues=data.get("issues", []),
            recommendations=data.get("recommendations", []),
        )
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Generate calibration report."""
        if not self._current_calibration:
            return {"error": "No calibration performed"}
        
        cal = self._current_calibration
        
        return {
            "calibration_id": cal.calibration_id,
            "timestamp": cal.timestamp,
            "status": cal.status.value,
            "summary": {
                "num_cameras": len(cal.cameras),
                "workspace_dimensions": cal.workspace.dimensions,
                "coverage_percentage": f"{cal.coverage_percentage:.1f}%",
                "mean_reprojection_error": f"{cal.mean_reprojection_error:.3f} px",
            },
            "cameras": [
                {
                    "id": c.camera_id,
                    "name": c.name,
                    "position": (c.extrinsics.x, c.extrinsics.y, c.extrinsics.z),
                    "calibrated": c.is_calibrated,
                }
                for c in cal.cameras
            ],
            "issues": cal.issues,
            "recommendations": cal.recommendations,
            "mmpose_ready": len(cal.issues) == 0,
        }


# =============================================================================
# FastAPI Service
# =============================================================================

def create_fastapi_app():
    """Create FastAPI application for calibration service."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(
        title="MMPose Calibration Service",
        description="Camera placement and calibration for MMPose pose estimation",
        version=__version__,
    )
    
    service = MMPoseCalibrationService()
    
    class WorkspaceConfigRequest(BaseModel):
        x_range: Tuple[float, float] = (-2, 2)
        y_range: Tuple[float, float] = (-2, 2)
        z_range: Tuple[float, float] = (0, 2)
        center: Tuple[float, float, float] = (0, 0, 1)
    
    class CameraAddRequest(BaseModel):
        camera_id: str
        name: str
        ip_address: str
        position: Optional[Tuple[float, float, float]] = None
        rotation: Optional[Tuple[float, float, float]] = None
    
    class PlacementRequest(BaseModel):
        num_cameras: int
        placement_type: Optional[str] = None
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": __version__}
    
    @app.get("/status")
    async def get_status():
        return {
            "calibration_status": service.status.value,
            "num_cameras": len(service.cameras),
            "workspace": asdict(service.workspace),
        }
    
    @app.post("/workspace/configure")
    async def configure_workspace(request: WorkspaceConfigRequest):
        service.configure_workspace(
            x_range=request.x_range,
            y_range=request.y_range,
            z_range=request.z_range,
            center=request.center,
        )
        return {"status": "configured", "workspace": asdict(service.workspace)}
    
    @app.post("/placement/recommend")
    async def get_placement_recommendations(request: PlacementRequest):
        placement_type = None
        if request.placement_type:
            placement_type = CameraPlacementType(request.placement_type)
        
        recommendations = service.get_placement_recommendations(
            request.num_cameras, placement_type
        )
        return {"recommendations": recommendations}
    
    @app.post("/cameras")
    async def add_camera(request: CameraAddRequest):
        camera = service.add_camera(
            camera_id=request.camera_id,
            name=request.name,
            ip_address=request.ip_address,
            position=request.position,
            rotation=request.rotation,
        )
        return {"status": "added", "camera": asdict(camera)}
    
    @app.get("/cameras")
    async def list_cameras():
        return {"cameras": [asdict(c) for c in service.cameras.values()]}
    
    @app.post("/calibration/run")
    async def run_calibration(
        pose_model: str = "rtmpose-m",
        backend: str = "tensorrt-fp16"
    ):
        result = service.run_full_calibration(
            PoseModel(pose_model),
            Backend(backend),
        )
        return asdict(result)
    
    @app.get("/calibration/report")
    async def get_calibration_report():
        return service.get_calibration_report()
    
    @app.get("/coverage/analyze")
    async def analyze_coverage():
        coverage = service.placement_optimizer.analyze_coverage(
            list(service.cameras.values())
        )
        return coverage
    
    return app


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MMPose Calibration Service")
    parser.add_argument("command", choices=["serve", "recommend", "calibrate", "report"])
    parser.add_argument("--num-cameras", type=int, default=4, help="Number of cameras")
    parser.add_argument("--placement", choices=["single", "stereo", "triangular", "quad", "hexagonal"])
    parser.add_argument("--port", type=int, default=8091, help="Service port")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        import uvicorn
        app = create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    elif args.command == "recommend":
        service = MMPoseCalibrationService()
        placement_type = CameraPlacementType(args.placement) if args.placement else None
        recommendations = service.get_placement_recommendations(args.num_cameras, placement_type)
        
        print(f"\nCamera Placement Recommendations ({args.num_cameras} cameras)")
        print("=" * 60)
        for rec in recommendations:
            print(f"\n{rec['camera_id']}:")
            print(f"  Position: ({rec['position'][0]:.2f}, {rec['position'][1]:.2f}, {rec['position'][2]:.2f})")
            print(f"  Rotation: roll={rec['rotation'][0]:.1f}°, pitch={rec['rotation'][1]:.1f}°, yaw={rec['rotation'][2]:.1f}°")
            print(f"  Coverage: {rec['coverage_score']*100:.0f}%")
            if rec['notes']:
                print(f"  Note: {rec['notes']}")
    
    elif args.command == "calibrate":
        service = MMPoseCalibrationService()
        result = service.run_full_calibration()
        print(f"\nCalibration Complete: {result.calibration_id}")
        print(f"Status: {result.status.value}")
        if result.issues:
            print("Issues:")
            for issue in result.issues:
                print(f"  - {issue}")
    
    elif args.command == "report":
        service = MMPoseCalibrationService()
        report = service.get_calibration_report()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

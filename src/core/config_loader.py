"""
Configuration Loader & Validation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

# =============================================================================
# Configuration Models
# =============================================================================

class SystemConfig(BaseModel):
    simulation_mode: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

class SafetyConfig(BaseModel):
    stop_dist: float = Field(1.5, ge=0.5, le=5.0)
    sensitivity: float = Field(0.8, ge=0.1, le=1.0)
    max_speed_factor: float = Field(1.0, ge=0.1, le=1.0)

class CameraConfig(BaseModel):
    rtsp_url: str = "rtsp://127.0.0.1:554/stream"
    width: int = 1920
    height: int = 1080
    fps: int = 30

class ModelsConfig(BaseModel):
    paligemma_path: str = "models/paligemma"
    yolo_path: Optional[str] = None

class TFLOPSAllocation(BaseModel):
    """TFLOPS allocation for system components.

    Note: trajectory_prediction removed - now handled by V-JEPA 2 world model.
    See meta_ai.tflops_allocation for Meta AI model budgets.
    """
    safety_detection: float = 15.0
    spatial_brain: float = 3.0
    navigation_detection: float = 30.0      # Reduced: SAM3 handles segmentation
    depth_estimation: float = 5.0
    il_training: float = 9.0
    moai_compression: float = 3.0
    fhe_encryption: float = 1.0
    pi0_vla: float = 10.0
    full_perception: float = 15.0           # Reduced: DINOv3 handles features
    anomaly_detection: float = 3.0

class TFLOPSBudget(BaseModel):
    total_fp16: float = 137.0
    safe_utilization: float = 0.85
    burst_utilization: float = 0.95
    allocations: TFLOPSAllocation = Field(default_factory=TFLOPSAllocation)

class PipelineQueues(BaseModel):
    safety_maxsize: int = 100
    perception_maxsize: int = 500
    learning_maxsize: int = 1000

class PipelineRouting(BaseModel):
    safety_cameras: List[int] = [0, 1, 2, 3]
    perception_cameras: List[int] = [4, 5, 6, 7]

class PipelineLearning(BaseModel):
    batch_size: int = 1000
    batch_timeout: float = 60.0

class PipelineConfig(BaseModel):
    queues: PipelineQueues = Field(default_factory=PipelineQueues)
    routing: PipelineRouting = Field(default_factory=PipelineRouting)
    learning: PipelineLearning = Field(default_factory=PipelineLearning)

class RetargetingIK(BaseModel):
    max_position_error: float = 0.05
    max_rotation_error: float = 0.2
    fallback_to_previous: bool = True

class RetargetingConfig(BaseModel):
    workspace_min: List[float] = [-1.0, -1.0, 0.0]
    workspace_max: List[float] = [1.0, 1.0, 1.5]
    max_joint_velocity: float = 1.0
    ik: RetargetingIK = Field(default_factory=RetargetingIK)

class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    cameras: CameraConfig = Field(default_factory=CameraConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tflops_budget: TFLOPSBudget = Field(default_factory=TFLOPSBudget)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    retargeting: RetargetingConfig = Field(default_factory=RetargetingConfig)

# =============================================================================
# Loader
# =============================================================================

def load_and_validate_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Load configuration from YAML, apply env overrides, and validate.
    """
    path = Path(config_path)
    config_data = {}
    
    # 1. Load YAML
    if path.exists():
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {path}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            # Continue with defaults
    else:
        logger.warning(f"Config file {path} not found. Using defaults.")

    # 2. Environment Overrides
    # System
    if os.getenv("SIMULATION_MODE"):
        if "system" not in config_data: config_data["system"] = {}
        config_data["system"]["simulation_mode"] = os.getenv("SIMULATION_MODE").lower() == "true"
    
    if os.getenv("LOG_LEVEL"):
        if "system" not in config_data: config_data["system"] = {}
        config_data["system"]["log_level"] = os.getenv("LOG_LEVEL").upper()

    if os.getenv("API_PORT"):
        if "system" not in config_data: config_data["system"] = {}
        try:
            config_data["system"]["api_port"] = int(os.getenv("API_PORT"))
        except ValueError:
            pass

    # Safety
    if os.getenv("SAFETY_SENSITIVITY"):
        if "safety" not in config_data: config_data["safety"] = {}
        try:
            config_data["safety"]["sensitivity"] = float(os.getenv("SAFETY_SENSITIVITY"))
        except ValueError:
            pass

    # 3. Validation
    try:
        config = AppConfig(**config_data)
        logger.info("Configuration validated successfully.")
        return config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        # In production, we might want to raise here. 
        # For now, we'll try to return a default config or raise if critical.
        logger.warning("Falling back to default configuration due to validation errors.")
        return AppConfig()

# Global Config Instance
# Can be imported as `from src.core.config_loader import config`
# Note: This loads on import. For dynamic reloading, call load_and_validate_config() explicitly.
try:
    config = load_and_validate_config()
except Exception as e:
    logger.critical(f"Fatal error loading config: {e}")
    config = AppConfig()

#!/usr/bin/env python3
"""
Dynamical.ai Integrated Data Pipeline - AGX Orin 32GB Edition

Complete data pipeline from camera capture to encrypted cloud upload.

PIPELINE ARCHITECTURE:
======================

┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGX Orin 32GB Edge Device                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │ Camera 1 │   │ Camera 2 │   │   ...    │   │Camera 12 │                 │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘                 │
│       │              │              │              │                        │
│       └──────────────┴──────┬───────┴──────────────┘                        │
│                             │                                               │
│                    ┌────────▼────────┐                                      │
│                    │   DeepStream    │  12 streams @ 30fps                  │
│                    │   + YOLOv8l     │  ~60 TFLOPS                          │
│                    └────────┬────────┘                                      │
│                             │                                               │
│              ┌──────────────┼──────────────┐                                │
│              │              │              │                                │
│     ┌────────▼────────┐ ┌───▼───┐ ┌───────▼───────┐                        │
│     │ Spatial Brain   │ │ MOAI  │ │ IL Training   │                        │
│     │ (Safety/Anomaly)│ │Compress│ │(Diffusion+ACT)│                        │
│     │ ~3 TFLOPS       │ │~3 TFLOP│ │ ~9 TFLOPS     │                        │
│     └────────┬────────┘ └───┬───┘ └───────┬───────┘                        │
│              │              │              │                                │
│              │         ┌────▼────┐         │                                │
│              │         │  FHE    │         │                                │
│              │         │Encryption│        │                                │
│              │         │~1 TFLOPS│         │                                │
│              │         └────┬────┘         │                                │
│              │              │              │                                │
│              └──────────────┼──────────────┘                                │
│                             │                                               │
│                    ┌────────▼────────┐                                      │
│                    │  NVMe Buffer    │                                      │
│                    │   500GB SSD     │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                    ┌────────▼────────┐                                      │
│                    │   Wi-Fi 6E      │                                      │
│                    │ Cloud Upload    │                                      │
│                    └─────────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

TFLOPS BUDGET (137 TFLOPS available):
=====================================

Perception (Fixed, continuous):
- YOLOv8l detection:        59.5 TFLOPS
- YOLOv8l-pose:             60.7 TFLOPS  
- Depth Anything v2:         8.9 TFLOPS
- SegFormer-B2:             22.5 TFLOPS
- Subtotal Perception:     ~110 TFLOPS (80% utilization)

NOTE: The above assumes ALL 12 cameras running ALL models at 30fps.
In practice, we use intelligent scheduling:

OPTIMIZED TFLOPS BUDGET:
========================

Tier 1 - Always On (Safety Critical):
- YOLOv8l human detection:   15.0 TFLOPS  (4 cams @ 30fps)
- Spatial Brain safety:       3.0 TFLOPS
- Subtotal Always-On:        18.0 TFLOPS

Tier 2 - High Priority (Navigation):
- Full detection (8 cams):   40.0 TFLOPS
- Depth estimation:           5.0 TFLOPS
- Trajectory prediction:      0.5 TFLOPS
- Subtotal Navigation:       45.5 TFLOPS

Tier 3 - Background (Learning):
- IL Training (10 Hz):        9.0 TFLOPS
- MOAI compression:           3.0 TFLOPS
- FHE encryption:             1.0 TFLOPS
- Subtotal Learning:         13.0 TFLOPS

Tier 4 - On-Demand (Full Analysis):
- Full perception (all 12):  30.0 TFLOPS
- Anomaly detection:          3.0 TFLOPS
- Subtotal On-Demand:        33.0 TFLOPS

TOTAL SUSTAINED:             ~80 TFLOPS (58% utilization)
BURST CAPACITY:              ~110 TFLOPS (80% utilization)
HEADROOM:                    ~27-57 TFLOPS for spikes
"""

import os
import sys
import time
import math
import json
import yaml
import queue
import threading
import logging
import asyncio
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque

from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

# Drivers
try:
    from src.drivers.cameras import MultiViewCameraRig, OnvifCamera, CameraCalibration
    HAS_PERCEPTION = True
except ImportError:
    HAS_PERCEPTION = False
    logger.warning("Perception module not found. Running without cameras.")

from src.drivers.dyglove import DYGloveDriver
from src.drivers.daimon_vtla import DaimonVTLAAdapter

# Cloud / Platform
from src.platform.cloud.ffm_client import FFMClient
from src.platform.cloud.secure_aggregator import SecureAggregator
from src.platform.safety_manager import safety_manager

# Core
from src.core.gmr_retargeting import Retargeter
# from src.core.quality.pose_quality import PoseQuality
# from src.core.quality.integrated_quality import IntegratedQuality
from src.core.neuracore_client import get_client as get_neuracore_client
from src.core.schema_validator import SchemaValidator

# MOAI
from src.moai.moai_pt import MoaiConfig

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not found.")

# Import Pi0
try:
    from src.spatial_intelligence.pi0.model import Pi0
    HAS_PI0 = True
except ImportError:
    HAS_PI0 = False
    logger.warning("Pi0 module not found.")




# =============================================================================
# TFLOPS Budget Manager
# =============================================================================

@dataclass
class TFLOPSBudget:
    """TFLOPS budget allocation."""
    component: str
    allocated_tflops: float
    current_tflops: float = 0.0
    priority: int = 1  # 1=highest, 4=lowest
    is_critical: bool = False
    can_throttle: bool = True


class TFLOPSManager:
    """
    Manages TFLOPS allocation across system components.
    
    Ensures:
    - Critical components always have resources
    - Total usage stays within hardware limits
    - Dynamic throttling during contention
    """
    
    # AGX Orin 32GB limits
    TOTAL_TFLOPS_FP16 = 137.0
    SAFE_UTILIZATION = 0.85  # 85% max sustained
    BURST_UTILIZATION = 0.95  # 95% for short bursts
    
    def __init__(self):
        self.config = {} # Will be injected or loaded
        self.budgets: Dict[str, TFLOPSBudget] = {}
        self.current_utilization = 0.0
        self._lock = threading.Lock()
        
        # Initialize default budgets (will be overridden by config if provided)
        self._init_default_budgets()

    def load_config(self, config: Dict):
        """Load budgets from config dictionary."""
        self.config = config
        if "tflops_budget" in config:
            tflops_cfg = config["tflops_budget"]
            self.TOTAL_TFLOPS_FP16 = tflops_cfg.get("total_fp16", 137.0)
            self.SAFE_UTILIZATION = tflops_cfg.get("safe_utilization", 0.85)
            self.BURST_UTILIZATION = tflops_cfg.get("burst_utilization", 0.95)
            
            if "allocations" in tflops_cfg:
                for name, value in tflops_cfg["allocations"].items():
                    # Update existing or create new budget
                    if name in self.budgets:
                        self.budgets[name].allocated_tflops = value
                    else:
                        # Default priority/criticality for new items
                        self.budgets[name] = TFLOPSBudget(name, value, priority=3)
                        
    def _init_default_budgets(self):
        """Initialize default TFLOPS allocations."""
        default_budgets = [
            # Tier 1 - Always On (Critical)
            TFLOPSBudget("safety_detection", 15.0, priority=1, is_critical=True, can_throttle=False),
            TFLOPSBudget("spatial_brain", 3.0, priority=1, is_critical=True, can_throttle=False),
            
            # Tier 2 - High Priority
            TFLOPSBudget("navigation_detection", 40.0, priority=2, is_critical=False, can_throttle=True),
            TFLOPSBudget("depth_estimation", 5.0, priority=2, is_critical=False, can_throttle=True),
            TFLOPSBudget("trajectory_prediction", 0.5, priority=2, is_critical=False, can_throttle=True),
            
            # Tier 3 - Background
            TFLOPSBudget("il_training", 9.0, priority=3, is_critical=False, can_throttle=True),
            TFLOPSBudget("moai_compression", 3.0, priority=3, is_critical=False, can_throttle=True),
            TFLOPSBudget("fhe_encryption", 1.0, priority=3, is_critical=False, can_throttle=True),
            
            TFLOPSBudget("fhe_encryption", 1.0, priority=3, is_critical=False, can_throttle=True),
            
            # Tier 4 - On-Demand
            TFLOPSBudget("pi0_vla", 10.0, priority=4, is_critical=False, can_throttle=True),
            TFLOPSBudget("full_perception", 30.0, priority=4, is_critical=False, can_throttle=True),
            TFLOPSBudget("anomaly_detection", 3.0, priority=4, is_critical=False, can_throttle=True),
        ]
        
        for budget in default_budgets:
            self.budgets[budget.component] = budget
    
    def request_tflops(self, component: str, requested: float) -> float:
        """
        Request TFLOPS allocation for a component.
        
        Returns:
            Actual TFLOPS granted (may be less than requested)
        """
        with self._lock:
            if component not in self.budgets:
                return 0.0
            
            budget = self.budgets[component]
            max_available = self.TOTAL_TFLOPS_FP16 * self.SAFE_UTILIZATION
            
            # Calculate current total (excluding this component)
            current_total = sum(
                b.current_tflops for name, b in self.budgets.items()
                if name != component
            )
            
            # Available for this component
            available = max_available - current_total
            
            # Grant up to allocated budget or available, whichever is less
            granted = min(requested, budget.allocated_tflops, available)
            
            if granted < requested and not budget.can_throttle:
                # Critical component - force allocation by throttling others
                granted = self._force_allocation(component, requested)
            
            budget.current_tflops = granted
            self._update_utilization()
            
            return granted
    
    def _force_allocation(self, component: str, needed: float) -> float:
        """Force allocation for critical component by throttling others."""
        budget = self.budgets[component]
        
        # Throttle lower priority components
        for priority in [4, 3, 2]:
            for name, other in self.budgets.items():
                if name == component:
                    continue
                if other.priority >= priority and other.can_throttle:
                    reduction = min(other.current_tflops, needed - budget.current_tflops)
                    other.current_tflops -= reduction
                    
                    if budget.current_tflops + reduction >= needed:
                        return needed
        
        return budget.allocated_tflops
    
    def release_tflops(self, component: str):
        """Release TFLOPS allocation."""
        with self._lock:
            if component in self.budgets:
                self.budgets[component].current_tflops = 0.0
                self._update_utilization()
    
    def _update_utilization(self):
        """Update current utilization."""
        total = sum(b.current_tflops for b in self.budgets.values())
        self.current_utilization = total / self.TOTAL_TFLOPS_FP16
    
    def get_status(self) -> Dict[str, Any]:
        """Get current TFLOPS status."""
        with self._lock:
            return {
                "total_available": self.TOTAL_TFLOPS_FP16,
                "safe_limit": self.TOTAL_TFLOPS_FP16 * self.SAFE_UTILIZATION,
                "current_utilization": self.current_utilization,
                "current_tflops": sum(b.current_tflops for b in self.budgets.values()),
                "budgets": {
                    name: {
                        "allocated": b.allocated_tflops,
                        "current": b.current_tflops,
                        "priority": b.priority,
                        "critical": b.is_critical,
                    }
                    for name, b in self.budgets.items()
                },
            }


# =============================================================================
# Data Flow Controller
# =============================================================================

class DataFlowController:
    """
    Controls data flow through the pipeline.
    
    Manages:
    - Frame routing (which cameras to which models)
    - Priority queuing
    - Backpressure handling
    - Quality of service
    """
    
    def __init__(
        self,
        num_cameras: int = 12,
        num_robots: int = 4
    ):
        self.num_cameras = num_cameras
        self.num_robots = num_robots
        
        # Queues by priority (defaults, can be resized)
        self.safety_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=100)
        self.perception_queue: queue.Queue = queue.Queue(maxsize=500)
        self.learning_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        # Frame counters
        self.frame_counters: Dict[str, int] = {}
        
        # Routing rules
        self.safety_cameras = set()  # Cameras dedicated to safety
        self.perception_cameras = set()  # Cameras for navigation
        
        # Statistics
        self.stats = {
            "frames_routed": 0,
            "safety_frames": 0,
            "perception_frames": 0,
            "learning_frames": 0,
            "dropped_frames": 0,
        }
    
    def configure_routing(
        self,
        safety_camera_ids: List[str],
        perception_camera_ids: List[str]
    ):
        """Configure camera routing."""
        self.safety_cameras = set(safety_camera_ids)
        self.perception_cameras = set(perception_camera_ids)
    
    def route_frame(
        self,
        robot_id: str,
        camera_id: str,
        frame_data: Dict,
        timestamp: float
    ) -> str:
        """
        Route frame to appropriate queue.
        
        Returns:
            Queue name where frame was routed
        """
        key = f"{robot_id}_{camera_id}"
        self.frame_counters[key] = self.frame_counters.get(key, 0) + 1
        frame_idx = self.frame_counters[key]
        
        # Priority routing
        if camera_id in self.safety_cameras:
            # Safety-critical: highest priority
            try:
                self.safety_queue.put_nowait((timestamp, frame_data))
                self.stats["safety_frames"] += 1
                return "safety"
            except queue.Full:
                self.stats["dropped_frames"] += 1
                return "dropped"
        
        elif camera_id in self.perception_cameras:
            # Navigation: medium priority
            try:
                self.perception_queue.put_nowait(frame_data)
                self.stats["perception_frames"] += 1
                return "perception"
            except queue.Full:
                self.stats["dropped_frames"] += 1
                return "dropped"
        
        else:
            # Learning/storage: lowest priority, subsample
            if frame_idx % 3 == 0:  # Every 3rd frame
                try:
                    self.learning_queue.put_nowait(frame_data)
                    self.stats["learning_frames"] += 1
                    return "learning"
                except queue.Full:
                    pass
        
        self.stats["frames_routed"] += 1
        return "skipped"
    
    def get_safety_frame(self, timeout: float = 0.01) -> Optional[Dict]:
        """Get next safety-critical frame."""
        try:
            _, frame = self.safety_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def get_perception_frame(self, timeout: float = 0.01) -> Optional[Dict]:
        """Get next perception frame."""
        try:
            return self.perception_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_learning_frame(self, timeout: float = 0.01) -> Optional[Dict]:
        """Get next learning frame."""
        try:
            return self.learning_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get flow statistics."""
        return {
            **self.stats,
            "safety_queue_size": self.safety_queue.qsize(),
            "perception_queue_size": self.perception_queue.qsize(),
            "learning_queue_size": self.learning_queue.qsize(),
        }


# =============================================================================
# Integrated Pipeline
# =============================================================================

class IntegratedDataPipeline:
    """
    Complete integrated data pipeline for AGX Orin 32GB.
    
    Integrates:
    - TFLOPS management
    - Data flow control
    - MOAI compression
    - FHE encryption
    - Cloud upload scheduling
    """
    
    def __init__(
        self,
        num_robots: int = 4,
        cameras_per_robot: int = 12,
        storage_path: str = "/data/pipeline",
        config_path: str = "config/config.yaml",
        **kwargs
    ):
        self.num_robots = num_robots
        self.cameras_per_robot = cameras_per_robot
        self.storage_path = storage_path
        
        # Load Config
        from src.core.config_loader import load_and_validate_config
        self.app_config = load_and_validate_config(config_path)
        self.cfg = self.app_config.model_dump() # Keep dict for compatibility with existing code if needed, or migrate fully
        
        self.learning_batch_size = self.app_config.pipeline.learning.batch_size
        self.learning_batch_timeout = self.app_config.pipeline.learning.batch_timeout
            
        # Allow overrides from kwargs (for testing)
        if "learning_batch_size" in kwargs:
            self.learning_batch_size = kwargs["learning_batch_size"]
        if "learning_batch_timeout" in kwargs:
            self.learning_batch_timeout = kwargs["learning_batch_timeout"]

        
        # Core components
        self.tflops_manager = TFLOPSManager()
        self.tflops_manager.load_config(self.cfg)
        
        self.flow_controller = DataFlowController(
            num_cameras=cameras_per_robot,
            num_robots=num_robots
        )
        
        # Configure queues from config
        if "pipeline" in self.cfg and "queues" in self.cfg["pipeline"]:
            q_cfg = self.cfg["pipeline"]["queues"]
            self.flow_controller.safety_queue = queue.PriorityQueue(maxsize=q_cfg.get("safety_maxsize", 100))
            self.flow_controller.perception_queue = queue.Queue(maxsize=q_cfg.get("perception_maxsize", 500))
            self.flow_controller.learning_queue = queue.Queue(maxsize=q_cfg.get("learning_maxsize", 1000))
        
        # MOAI and FHE will be integrated
        self.moai_system = None
        self.fhe_system = None
        self.pi0_model = None
        self.camera_rig = None
        
        # Processing threads
        self._running = False
        self._threads: List[threading.Thread] = []
        
        # Upload scheduler
        self.upload_queue: queue.Queue = queue.Queue(maxsize=50)
        self.pending_uploads: List[Dict] = []
        
        # Statistics
        self.stats = {
            "pipeline_uptime_s": 0.0,
            "total_frames_processed": 0,
            "total_bytes_compressed": 0,
            "total_bytes_encrypted": 0,
            "total_bytes_uploaded": 0,
            "avg_latency_ms": 0.0,
            "validation_errors": 0,
        }
        
        self._start_time = time.time()
        
        # Neuracore Integration
        self.neuracore = get_neuracore_client(storage_path=os.path.join(storage_path, "neuracore_logs"))
        self.validator = SchemaValidator(strict=False)
        self.neuracore.start()
        
        # FHE Context (N2HE)
        if self.fhe_system:
             self.fhe_context = self.fhe_system.context # Access internal context
        else:
             self.fhe_context = None
        
        os.makedirs(storage_path, exist_ok=True)
    
        logger.info("Pipeline initialized!")

    def setup_cameras(self, camera_configs: List[Dict]):
        """
        Setup camera rig for direct capture.
        
        Args:
            camera_configs: List of dicts with 'id', 'url', 'calibration_path'
        """
        if not HAS_PERCEPTION:
            logger.warning("Perception module not available")
            return
            
        logger.info(f"Setting up {len(camera_configs)} cameras...")
        self.camera_rig = MultiViewCameraRig()
        
        for conf in camera_configs:
            calib = None
            if 'calibration_path' in conf and conf['calibration_path']:
                try:
                    calib = CameraCalibration.load(Path(conf['calibration_path']))
                except Exception as e:
                    logger.error(f"Failed to load calibration for {conf['id']}: {e}")
            
            self.camera_rig.add_camera(
                camera_id=conf['id'],
                rtsp_url=conf['url'],
                calibration=calib
            )
        
        logger.info(f"Camera rig setup complete with {self.camera_rig.num_cameras} cameras")

    def initialize(self, moai_system=None, fhe_system=None):
        """Initialize pipeline with MOAI and FHE systems."""
        logger.info("=" * 60)
        logger.info("Initializing Integrated Data Pipeline")
        logger.info("=" * 60)
        
        self.moai_system = moai_system
        self.fhe_system = fhe_system
        
        if HAS_PI0:
            logger.info("Initializing Pi0 VLA Model...")
            # In a real scenario, we'd load weights here. 
            # For integration, we assume it's passed or lazy-loaded.
            self.pi0_model = Pi0(
                action_dim=7, 
                action_horizon=16,
                device="cuda" if self.tflops_manager.TOTAL_TFLOPS_FP16 > 0 else "cpu"
            )
        
        # Configure routing
        # Configure routing
        if "pipeline" in self.cfg and "routing" in self.cfg["pipeline"]:
            r_cfg = self.cfg["pipeline"]["routing"]
            safety_indices = r_cfg.get("safety_cameras", [0, 1, 2, 3])
            perception_indices = r_cfg.get("perception_cameras", [4, 5, 6, 7])
            
            safety_cams = [f"cam_{i}" for i in safety_indices]
            perception_cams = [f"cam_{i}" for i in perception_indices]
        else:
            # Fallback defaults
            safety_cams = [f"cam_{i}" for i in range(4)]
            perception_cams = [f"cam_{i}" for i in range(4, 8)]
        
        self.flow_controller.configure_routing(safety_cams, perception_cams)
        
        logger.info(f"Configuration:")
        logger.info(f"  Robots: {self.num_robots}")
        logger.info(f"  Cameras per robot: {self.cameras_per_robot}")
        logger.info(f"  Safety cameras: {len(safety_cams)} per robot")
        logger.info(f"  Perception cameras: {len(perception_cams)} per robot")
        
        # Print TFLOPS budget
        tflops_status = self.tflops_manager.get_status()
        logger.info(f"TFLOPS Budget:")
        logger.info(f"  Total available: {tflops_status['total_available']:.1f}")
        logger.info(f"  Safe limit: {tflops_status['safe_limit']:.1f}")
        
        logger.info("Pipeline initialized!")
    
    def start(self):
        """Start pipeline processing."""
        self._running = True
        self._start_time = time.time()
        
        # Start Drivers
        if self.camera_rig:
            self.camera_rig.start()
        
        # TODO: Initialize glove_driver and robot_adapter properly
        # if self.glove_driver:
        #     self.glove_driver.connect()
            
        # if self.robot_adapter:
        #     self.robot_adapter.enable_motors()
        
        # Start MOAI if available
        if self.moai_system:
            self.moai_system.start()
        
        # Start FHE if available
        if self.fhe_system:
            self.fhe_system.start()
        
        # Start processing threads
        safety_thread = threading.Thread(target=self._safety_processing_loop)
        safety_thread.daemon = True
        safety_thread.start()
        self._threads.append(safety_thread)
        
        learning_thread = threading.Thread(target=self._learning_processing_loop)
        learning_thread.daemon = True
        learning_thread.start()
        self._threads.append(learning_thread)
        
        upload_thread = threading.Thread(target=self._upload_loop)
        upload_thread.daemon = True
        upload_thread.start()
        self._threads.append(upload_thread)
        
        if self.pi0_model:
            vla_thread = threading.Thread(target=self._vla_inference_loop)
            vla_thread.daemon = True
            vla_thread.start()
            self._threads.append(vla_thread)
            
        if self.camera_rig:
            cam_thread = threading.Thread(target=self._camera_ingestion_loop)
            cam_thread.daemon = True
            cam_thread.start()
            self._threads.append(cam_thread)
        
        logger.info("[IntegratedPipeline] Started")
    
    def stop(self):
        """Stop pipeline."""
        self._running = False
        
        # Stop Drivers
        if self.camera_rig:
            self.camera_rig.stop()
        
        # if self.glove_driver:
        #     self.glove_driver.disconnect()
            
        # if self.robot_adapter:
        #     self.robot_adapter.disable_motors()
        
        for thread in self._threads:
            thread.join(timeout=2.0)
        
        if self.moai_system:
            self.moai_system.stop()
        
        if self.fhe_system:
            self.fhe_system.stop()
            
        if self.camera_rig:
            self.camera_rig.stop()
            
        self.neuracore.stop()
        
        logger.info("[IntegratedPipeline] Stopped")
    
    def submit_frame(
        self,
        robot_id: str,
        camera_id: str,
        frame: 'np.ndarray',
        timestamp: float,
        metadata: Optional[Dict] = None
    ):
        """Submit frame to pipeline."""
        frame_data = {
            "robot_id": robot_id,
            "camera_id": camera_id,
            "frame": frame,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        
        queue_name = self.flow_controller.route_frame(
            robot_id, camera_id, frame_data, timestamp
        )
        
        # Validate frame (Schema Enforcement)
        validation = self.validator.validate_frame(f"{robot_id}/{camera_id}", frame, timestamp)
        if not validation.valid:
            self.stats["validation_errors"] += 1
            # In strict mode we might drop, but for now just log
        
        self.stats["total_frames_processed"] += 1
        
        return queue_name
    
    def _safety_processing_loop(self):
        """Process safety-critical frames."""
        # Request TFLOPS
        granted = self.tflops_manager.request_tflops("safety_detection", 15.0)
        granted += self.tflops_manager.request_tflops("spatial_brain", 3.0)
        
        logger.info(f"[SafetyLoop] Allocated {granted:.1f} TFLOPS")
        
        while self._running:
            try:
                frame_data = self.flow_controller.get_safety_frame(timeout=0.01)
                
                if frame_data is None:
                    continue
                
                # Process for safety (would call Spatial Brain)
                # In production: run detection + safety zone update
                
                # --- SIMULATED HAZARD DETECTION ---
                # In a real system, this comes from YOLOv8/Depth Anything
                import random
                if random.random() < 0.05: # 5% chance of hazard per frame
                    hazard_type = random.choice(["PERSON", "FORKLIFT", "WET_FLOOR", "UNEVEN_GROUND", "OVERHANG"])
                    distance = random.uniform(0.5, 5.0)
                    
                    action = safety_manager.evaluate_hazard(hazard_type, distance)
                    
                    if action != "SAFE":
                        logger.warning(f"[SAFETY] Hazard: {hazard_type} @ {distance:.1f}m -> ACTION: {action}")
                        
                        # Update Safety Manager State
                        safety_manager.active_hazards.append({
                            "type": hazard_type,
                            "distance": distance,
                            "action": action,
                            "timestamp": time.time()
                        })
                        
                        # Keep only recent hazards (last 5s)
                        cutoff = time.time() - 5.0
                        safety_manager.active_hazards = [h for h in safety_manager.active_hazards if h["timestamp"] > cutoff]
            except Exception as e:
                logger.error(f"[SafetyLoop] Error: {e}")
                # Don't crash the thread, just continue
                time.sleep(0.1)
    
    def _learning_processing_loop(self):
        """Process learning/storage frames."""
        # Request TFLOPS
        granted = self.tflops_manager.request_tflops("moai_compression", 3.0)
        granted += self.tflops_manager.request_tflops("fhe_encryption", 1.0)
        
        logger.info(f"[LearningLoop] Allocated {granted:.1f} TFLOPS")
        
        batch_buffer = []
        batch_start = time.time()
        
        while self._running:
            try:
                frame_data = self.flow_controller.get_learning_frame(timeout=0.1)
                
                if frame_data is not None:
                    batch_buffer.append(frame_data)
                
                # Create batch
                if len(batch_buffer) >= self.learning_batch_size or (time.time() - batch_start) >= self.learning_batch_timeout:
                    if batch_buffer:
                        self._process_learning_batch(batch_buffer)
                        batch_buffer = []
                        batch_start = time.time()
            except Exception as e:
                logger.error(f"[LearningLoop] Error: {e}")
                time.sleep(1.0)
    
    def _process_learning_batch(self, frames: List[Dict]):
        """Process batch of learning frames."""
        # Compress with MOAI and Encrypt with N2HE
        if self.moai_system and self.fhe_system:
            for frame_data in frames:
                # 1. Compress/Encode (MOAI)
                # In a real system, this would return an embedding
                # Here we simulate it by just passing the frame
                
                # 2. Encrypt (N2HE)
                # We encrypt the frame data before sending it to the "Cloud" (MOAI System)
                if "frame" in frame_data and isinstance(frame_data["frame"], np.ndarray):
                    encrypted_frame = self.fhe_system.context.col_pack_matrix(frame_data["frame"])
                    frame_data["encrypted_frame"] = encrypted_frame
                    # Remove raw frame to ensure privacy
                    del frame_data["frame"]
                
                self.moai_system.submit_frame(
                    frame_data["robot_id"],
                    frame_data["camera_id"],
                    frame_data.get("encrypted_frame"), # Send encrypted
                    frame_data["timestamp"],
                )
                
        # Async Logging (Neuracore)
        # We log the encrypted data if available, or raw if not (depending on policy)
        # For this implementation, we assume we log raw for local debugging but encrypted for cloud
        for frame_data in frames:
             data_to_log = frame_data.get("encrypted_frame") if "encrypted_frame" in frame_data else frame_data.get("frame")
             self.neuracore.log(
                channel=f"{frame_data['robot_id']}/{frame_data['camera_id']}",
                data=data_to_log, 
                timestamp=frame_data["timestamp"]
            )
        
        # Statistics
        estimated_bytes = len(frames) * 1024  # Rough estimate
        self.stats["total_bytes_compressed"] += estimated_bytes
    
    def _upload_loop(self):
        """Handle encrypted batch uploads."""
        while self._running:
            try:
                # Get encrypted batch from FHE
                if self.fhe_system:
                    encrypted = self.fhe_system.get_encrypted_batch(timeout=1.0)
                    
                    if encrypted:
                        self.pending_uploads.append(encrypted)
                        self.stats["total_bytes_encrypted"] += encrypted.get("output_bytes", 0)
                
                # Simulate upload (in production: actual Wi-Fi 6E upload)
                if self.pending_uploads:
                    batch = self.pending_uploads.pop(0)
                    # Would upload via Wi-Fi 6E here
                    self.stats["total_bytes_uploaded"] += batch.get("output_bytes", 0)
                
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"[UploadLoop] Error: {e}")
                time.sleep(5.0)
            
    def _camera_ingestion_loop(self):
        """Ingest frames from camera rig."""
        print("[CameraLoop] Starting ingestion")
        
        while self._running and self.camera_rig:
            try:
                # Get synchronized frames if possible, else just iterate
                # For simplicity, we'll just iterate and submit latest
                for camera_id, camera in self.camera_rig.cameras.items():
                    frame = camera.get_frame()
                    if frame:
                        # Submit to pipeline
                        # We use robot_id="robot_0" for this single-node setup
                        self.submit_frame(
                            robot_id="robot_0",
                            camera_id=camera_id,
                            frame=frame.image,
                            timestamp=frame.timestamp
                        )
                
                time.sleep(0.033)  # ~30 FPS polling
            except Exception as e:
                logger.error(f"[CameraLoop] Error: {e}")
                time.sleep(1.0)
            
    def _vla_inference_loop(self):
        """Process VLA inference on perception frames."""
        print("[VLALoop] Starting Pi0 inference loop")
        
        while self._running:
            try:
                # Request TFLOPS
                granted = self.tflops_manager.request_tflops("pi0_vla", 10.0)
                
                if granted < 5.0:
                    # Not enough compute, sleep and retry
                    time.sleep(0.1)
                    continue
                    
                # Get frame from perception queue (peek or copy to not consume it from other consumers if any)
                # For this demo, we'll just try to get a frame from the flow controller's perception queue
                # Note: In a real system, we might need a dedicated queue or multicast
                frame_data = self.flow_controller.get_perception_frame(timeout=0.1)
                
                if frame_data:
                    # Simulate VLA inference
                    # action = self.pi0_model.predict_action(frame_data["frame"], "do something")
                    # For now, just sleep to simulate compute
                    time.sleep(0.05) 
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"[VLALoop] Error: {e}")
                time.sleep(1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        self.stats["pipeline_uptime_s"] = time.time() - self._start_time
        
        return {
            **self.stats,
            "tflops": self.tflops_manager.get_status(),
            "flow": self.flow_controller.get_statistics(),
            "pending_uploads": len(self.pending_uploads),
        }


# =============================================================================
# Complete System TFLOPS Analysis
# =============================================================================

def analyze_complete_system():
    """Analyze TFLOPS usage for complete system."""
    
    print("\n" + "=" * 70)
    print("COMPLETE SYSTEM TFLOPS ANALYSIS - AGX ORIN 32GB")
    print("=" * 70)
    
    # Hardware
    AGX_TFLOPS = 137.0
    
    # Component breakdown
    components = {
        # Perception (12 cameras, but intelligently scheduled)
        "perception": {
            "safety_detection": {"tflops": 15.0, "desc": "YOLOv8l on 4 safety cams @ 30fps"},
            "navigation_detection": {"tflops": 25.0, "desc": "YOLOv8l on 4 nav cams @ 20fps"},
            "depth_estimation": {"tflops": 5.0, "desc": "Depth Anything v2 on 4 cams @ 15fps"},
            "pose_estimation": {"tflops": 8.0, "desc": "YOLOv8l-pose on detected humans"},
            "segmentation": {"tflops": 3.0, "desc": "SegFormer on 2 cams @ 10fps"},
        },
        "spatial_brain": {
            "anomaly_detection": {"tflops": 2.0, "desc": "RTFM on 4 cams @ 2 clips/s"},
            "trajectory_prediction": {"tflops": 0.5, "desc": "Social LSTM per human"},
            "safety_zones": {"tflops": 0.3, "desc": "Zone computation @ 30Hz"},
            "occupancy_grid": {"tflops": 0.2, "desc": "Grid updates @ 10Hz"},
        },
        "il_training": {
            "diffusion_policy": {"tflops": 6.0, "desc": "Training @ 5Hz"},
            "act_policy": {"tflops": 2.0, "desc": "Training @ 10Hz"},
            "quality_scoring": {"tflops": 0.5, "desc": "Episode evaluation"},
            "replay_sampling": {"tflops": 0.5, "desc": "Priority sampling"},
        },
        "data_pipeline": {
            "moai_compression": {"tflops": 2.0, "desc": "ViT compression @ 30fps"},
            "hierarchical_encoding": {"tflops": 0.5, "desc": "Multi-res encoding"},
            "fhe_encryption": {"tflops": 0.8, "desc": "RLWE encryption"},
            "batch_preparation": {"tflops": 0.2, "desc": "Serialization"},
        },
    }
    
    # Print breakdown
    total = 0.0
    
    for category, items in components.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print("-" * 60)
        
        cat_total = 0.0
        for name, info in items.items():
            print(f"  {name:<25} {info['tflops']:>6.1f} TFLOPS  ({info['desc']})")
            cat_total += info['tflops']
        
        print(f"  {'SUBTOTAL':<25} {cat_total:>6.1f} TFLOPS")
        total += cat_total
    
    print("\n" + "=" * 60)
    print(f"TOTAL SUSTAINED:          {total:>6.1f} TFLOPS")
    print(f"AVAILABLE:                {AGX_TFLOPS:>6.1f} TFLOPS")
    print(f"UTILIZATION:              {total/AGX_TFLOPS*100:>6.1f}%")
    print(f"HEADROOM:                 {AGX_TFLOPS - total:>6.1f} TFLOPS")
    print("=" * 60)
    
    # Verify we're within budget
    total_utilization = total / AGX_TFLOPS
    SAFE_UTILIZATION = 0.85 # 85%
    
    if total_utilization <= SAFE_UTILIZATION:
        print("[OK] WITHIN SAFE OPERATING LIMITS (85%)")
    else:
        print("[WARNING] EXCEEDS SAFE OPERATING LIMITS - THROTTLING REQUIRED")
    
    return {
        "components": components,
        "total_tflops": total,
        "available_tflops": AGX_TFLOPS,
        "utilization": total / AGX_TFLOPS,
    }


def analyze_data_throughput():
    """Analyze data throughput for the complete system."""
    
    print("\n" + "=" * 70)
    print("DATA THROUGHPUT ANALYSIS")
    print("=" * 70)
    
    # Per robot
    robot = {
        "cameras": 12,
        "fps_per_camera": 30,
        "resolution": (1920, 1080),
        "bytes_per_pixel": 3,
    }
    
    num_robots = 4
    
    # Raw data
    frame_bytes = robot["resolution"][0] * robot["resolution"][1] * robot["bytes_per_pixel"]
    raw_per_robot_mbps = (frame_bytes * robot["cameras"] * robot["fps_per_camera"]) / (1024 ** 2)
    raw_total_mbps = raw_per_robot_mbps * num_robots
    
    print(f"\nRaw Data Rate:")
    print(f"  Per robot: {raw_per_robot_mbps:.1f} MB/s")
    print(f"  Total (4 robots): {raw_total_mbps:.1f} MB/s ({raw_total_mbps * 8 / 1000:.1f} Gbps)")
    
    # SMART SELECTIVE UPLOAD STRATEGY
    # Not all data needs cloud upload - only high-quality IL demonstrations
    print(f"\n" + "-" * 50)
    print("SMART SELECTIVE UPLOAD STRATEGY")
    print("-" * 50)
    
    # Only upload:
    # 1. Keyframes (1 in 10 frames)
    # 2. High-quality episodes (70% pass rate)
    # 3. 3 cameras per robot for IL (not all 12)
    
    keyframe_ratio = 0.1  # 1 in 10 frames
    quality_pass_rate = 0.7  # 70% episodes accepted
    il_cameras = 3  # Only 3 cameras for IL upload
    effective_fps = 10  # Subsample to 10fps for upload
    
    # Effective data for upload
    effective_frames = num_robots * il_cameras * effective_fps
    effective_bytes = frame_bytes * effective_frames * keyframe_ratio * quality_pass_rate
    effective_mbps = effective_bytes / (1024 ** 2)
    
    print(f"\n  Keyframe ratio: {keyframe_ratio} (1 in 10)")
    print(f"  Quality filter: {quality_pass_rate:.0%} pass rate")
    print(f"  IL cameras per robot: {il_cameras}")
    print(f"  Effective FPS: {effective_fps}")
    
    # After MOAI compression (50:1)
    compression_ratio = 50
    compressed_mbps = effective_mbps / compression_ratio
    
    # After FHE encryption (3× expansion)
    fhe_expansion = 3.0
    encrypted_mbps = compressed_mbps * fhe_expansion
    
    # Upload budget
    wifi_6e_mbps = 300  # Practical sustained
    
    print(f"\nAfter Selective Filtering:")
    print(f"  Effective rate: {effective_mbps:.2f} MB/s")
    
    print(f"\nAfter MOAI Compression ({compression_ratio}:1):")
    print(f"  Total: {compressed_mbps:.4f} MB/s")
    print(f"  Per hour: {compressed_mbps * 3600:.1f} MB")
    
    print(f"\nAfter FHE Encryption ({fhe_expansion}× expansion):")
    print(f"  Total: {encrypted_mbps:.4f} MB/s")
    print(f"  Per hour: {encrypted_mbps * 3600:.1f} MB")
    print(f"  Per day: {encrypted_mbps * 86400 / 1024:.2f} GB")
    
    print(f"\nWi-Fi 6E Upload Capacity:")
    print(f"  Available: {wifi_6e_mbps} MB/s")
    print(f"  Required: {encrypted_mbps:.4f} MB/s")
    print(f"  Utilization: {encrypted_mbps / wifi_6e_mbps * 100:.2f}%")
    
    if encrypted_mbps < wifi_6e_mbps:
        print("✅ WITHIN UPLOAD CAPACITY")
    else:
        print("❌ EXCEEDS UPLOAD CAPACITY")
    
    # Local storage analysis
    print(f"\n" + "-" * 50)
    print("LOCAL STORAGE (500GB NVMe)")
    print("-" * 50)
    
    # Store compressed keyframes + features locally for on-device training
    # Not raw video - only compressed features
    feature_dim = 512  # Features per frame
    feature_bytes = feature_dim * 4  # float32
    local_frames_per_second = num_robots * il_cameras * effective_fps
    local_bytes_per_second = local_frames_per_second * feature_bytes
    local_mb_per_hour = local_bytes_per_second * 3600 / (1024 ** 2)
    local_gb_per_day = local_mb_per_hour * 24 / 1024
    
    nvme_capacity = 500  # GB
    # Reserve space: 100GB for system, 50GB for checkpoints, 50GB for replay buffer
    usable_capacity = nvme_capacity - 200
    days_of_storage = usable_capacity / max(0.1, local_gb_per_day)
    
    print(f"  Feature storage rate: {local_bytes_per_second / 1024:.1f} KB/s")
    print(f"  Per hour: {local_mb_per_hour:.1f} MB")
    print(f"  Per day: {local_gb_per_day:.2f} GB")
    print(f"  Usable capacity: {usable_capacity} GB")
    print(f"  Days of storage: {days_of_storage:.0f} days")
    print(f"  Retention policy: Auto-cleanup after 30 days")
    
    return {
        "raw_mbps": raw_total_mbps,
        "effective_mbps": effective_mbps,
        "compressed_mbps": compressed_mbps,
        "encrypted_mbps": encrypted_mbps,
        "upload_capacity_mbps": wifi_6e_mbps,
        "local_gb_per_day": local_gb_per_day,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTEGRATED DATA PIPELINE - AGX ORIN 32GB EDITION")
    print("=" * 70)
    
    # TFLOPS analysis
    tflops_result = analyze_complete_system()
    
    # Data throughput analysis
    throughput_result = analyze_data_throughput()
    
    # Demo pipeline
    print("\n" + "-" * 70)
    print("Demo: Initializing Pipeline")
    print("-" * 70)
    
    pipeline = IntegratedDataPipeline(
        num_robots=4,
        cameras_per_robot=12,
        storage_path="/tmp/pipeline_test"
    )
    
    pipeline.initialize()
    
    # Show final status
    tflops_status = pipeline.tflops_manager.get_status()
    
    print(f"\nTFLOPS Manager Status:")
    print(f"  Total: {tflops_status['total_available']:.1f} TFLOPS")
    print(f"  Safe limit: {tflops_status['safe_limit']:.1f} TFLOPS")
    print(f"  Budgeted components: {len(tflops_status['budgets'])}")
    
    print("\n" + "=" * 70)
    print("Pipeline initialized successfully!")
    print("=" * 70)

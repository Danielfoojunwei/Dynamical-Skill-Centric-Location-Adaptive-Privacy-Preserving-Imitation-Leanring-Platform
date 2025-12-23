"""
Perception Pipeline - Tier 2 Cascaded Perception

Runs at 30Hz with cascaded model strategy.
Level 1: Always (tiny models, <10ms)
Level 2: On-demand (medium models, <50ms)
Level 3: Rare (giant models, <500ms, can be off-robot)
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from .config import PerceptionConfig
except ImportError:
    from src.robot_runtime.config import PerceptionConfig

logger = logging.getLogger(__name__)


class CascadeLevel(Enum):
    """Perception cascade level."""
    LEVEL_1 = 1  # Always running (30Hz, <10ms)
    LEVEL_2 = 2  # On-demand (10Hz, <50ms)
    LEVEL_3 = 3  # Rare (1-5Hz, <500ms)


@dataclass
class PerceptionFeatures:
    """Extracted perception features."""
    timestamp: float = 0.0
    cascade_level: CascadeLevel = CascadeLevel.LEVEL_1

    # Level 1 features (always available)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    motion_vectors: Optional[np.ndarray] = None

    # Level 2 features (on-demand)
    dense_features: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    poses: List[Dict[str, Any]] = field(default_factory=list)

    # Level 3 features (rare)
    scene_understanding: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None


class PerceptionPipeline:
    """
    Cascaded perception pipeline.

    Uses trigger-based escalation:
    - Level 1 runs always
    - Level 2 triggers on uncertainty or new objects
    - Level 3 triggers on humans or high-stakes actions
    """

    def __init__(self, rate_hz: int, config: PerceptionConfig):
        self.rate_hz = rate_hz
        self.config = config

        self._features = PerceptionFeatures()
        self._current_level = CascadeLevel.LEVEL_1

        # Level timing
        self._last_level2_time = 0.0
        self._last_level3_time = 0.0
        self._level3_cooldown = config.cascade_cooldown_seconds if hasattr(config, 'cascade_cooldown_seconds') else 5.0

        # Models (placeholders - would be TensorRT engines in production)
        self._level1_models = {}
        self._level2_models = {}
        self._level3_models = {}

        self._initialized = False

    def initialize(self) -> None:
        """Initialize perception models."""
        logger.info("Initializing perception pipeline")

        # In production, load TensorRT engines here
        # self._level1_models['yolo'] = load_tensorrt('yolo_nano.engine')
        # self._level1_models['depth'] = load_tensorrt('depth_small.engine')

        self._initialized = True
        logger.info(f"Perception pipeline initialized with {len(self.config.level1_models)} level 1 models")

    def update(self) -> None:
        """
        Run perception update.

        Automatically determines which cascade level to run.
        """
        if not self._initialized:
            return

        start_time = time.time()

        # Always run Level 1
        self._run_level1()

        # Check if Level 2 needed
        if self._needs_level2():
            self._run_level2()

        # Check if Level 3 needed (with cooldown)
        if self._needs_level3() and self._can_run_level3():
            self._run_level3()

        self._features.timestamp = start_time

    def get_features(self) -> Dict[str, Any]:
        """Get current perception features."""
        return {
            'timestamp': self._features.timestamp,
            'cascade_level': self._features.cascade_level.value,
            'detections': self._features.detections,
            'obstacles': self._get_obstacles(),
            'humans': self._get_humans(),
            'depth_map': self._features.depth_map,
            'dense_features': self._features.dense_features,
            'segmentation_mask': self._features.segmentation_mask,
        }

    def _run_level1(self) -> None:
        """
        Run Level 1 perception (always, 30Hz, <10ms).

        - Tiny object detector (YOLO-Nano)
        - Fast depth estimation
        - Motion detection
        """
        self._features.cascade_level = CascadeLevel.LEVEL_1

        # Simulate detections (would be real model inference)
        self._features.detections = self._mock_detections()
        self._features.depth_map = self._mock_depth()
        self._features.motion_vectors = self._mock_motion()

    def _run_level2(self) -> None:
        """
        Run Level 2 perception (on-demand, 10Hz, <50ms).

        - DINOv3-Small for dense features
        - SAM3-Base for segmentation
        - RTMPose-M for pose estimation
        """
        self._features.cascade_level = CascadeLevel.LEVEL_2
        self._last_level2_time = time.time()

        # Simulate Level 2 outputs
        self._features.dense_features = np.random.randn(256).astype(np.float32)
        self._features.segmentation_mask = np.zeros((480, 640), dtype=np.uint8)
        self._features.poses = []

        logger.debug("Level 2 perception executed")

    def _run_level3(self) -> None:
        """
        Run Level 3 perception (rare, 1-5Hz, <500ms).

        - DINOv3-Giant for full scene understanding
        - SAM3-Huge for detailed segmentation
        - V-JEPA 2 Giant for prediction

        Can be offloaded to edge/cloud if available.
        """
        self._features.cascade_level = CascadeLevel.LEVEL_3
        self._last_level3_time = time.time()

        # Simulate Level 3 outputs
        self._features.scene_understanding = {
            'objects': [],
            'relationships': [],
            'affordances': [],
        }
        self._features.predictions = {
            'trajectories': [],
            'collision_risk': 0.0,
        }

        logger.info("Level 3 perception executed")

    def _needs_level2(self) -> bool:
        """Check if Level 2 perception is needed."""
        # Trigger on low confidence detections
        for det in self._features.detections:
            if det.get('confidence', 1.0) < 0.7:
                return True

        # Trigger on new objects
        # (Would track object IDs in production)

        return False

    def _needs_level3(self) -> bool:
        """Check if Level 3 perception is needed."""
        # Trigger on human detection
        for det in self._features.detections:
            if det.get('class') == 'person':
                return True

        # Trigger on high-stakes action
        # (Would check skill context)

        return False

    def _can_run_level3(self) -> bool:
        """Check cooldown for Level 3."""
        elapsed = time.time() - self._last_level3_time
        return elapsed > self._level3_cooldown

    def _get_obstacles(self) -> List[Dict[str, Any]]:
        """Extract obstacles from detections."""
        obstacles = []
        for det in self._features.detections:
            if det.get('class') not in ['person', 'background']:
                obstacles.append({
                    'position': det.get('position', [0, 0, 0]),
                    'distance': det.get('distance', float('inf')),
                    'class': det.get('class'),
                })
        return obstacles

    def _get_humans(self) -> List[Dict[str, Any]]:
        """Extract humans from detections."""
        humans = []
        for det in self._features.detections:
            if det.get('class') == 'person':
                humans.append({
                    'position': det.get('position', [0, 0, 0]),
                    'distance': det.get('distance', float('inf')),
                    'pose': det.get('pose'),
                })
        return humans

    # Mock functions for simulation
    def _mock_detections(self) -> List[Dict[str, Any]]:
        return [
            {'class': 'cup', 'confidence': 0.95, 'bbox': [100, 100, 50, 50], 'distance': 0.5},
            {'class': 'table', 'confidence': 0.99, 'bbox': [0, 200, 640, 280], 'distance': 0.8},
        ]

    def _mock_depth(self) -> np.ndarray:
        return np.ones((480, 640), dtype=np.float32) * 2.0

    def _mock_motion(self) -> np.ndarray:
        return np.zeros((480 // 8, 640 // 8, 2), dtype=np.float32)

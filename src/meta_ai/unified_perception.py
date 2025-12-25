"""
Unified Perception Pipeline for Dynamical Edge Platform

Integrates Meta AI models (DINOv3, SAM3, V-JEPA 2) with:
- MoE skill routing
- N2HE/FHE privacy preservation
- 4-tier timing system
- Robot skill invocation

Architecture:
============

┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED PERCEPTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Camera Input ───┬───▶ DINOv3 (Vision) ───────────────────┐                 │
│                  │                                         │                 │
│                  ├───▶ SAM3 (Segmentation) ───────────────┤                 │
│                  │     ↑ text prompt                       │                 │
│                  │                                         ▼                 │
│                  └───▶ V-JEPA 2 (World Model) ───▶ Feature Fusion ──┐       │
│                                                                      │       │
│  ┌───────────────────────────────────────────────────────────────────┘       │
│  │                                                                           │
│  │  N2HE Encryption ◄─── Fused Features                                      │
│  │         │                                                                 │
│  │         ▼                                                                 │
│  │  MoE Skill Router ───▶ Skill Weights ───▶ Skill Executor                 │
│  │         │                                         │                       │
│  │         │                                         ▼                       │
│  │         └────────────────────────────────▶ Robot Actions                  │
│  │                                                                           │
│  │  Timing Integration:                                                      │
│  │  - Tier 1 (1kHz): V-JEPA 2 safety prediction                             │
│  │  - Tier 2 (100Hz): Full perception + skill execution                      │
│  │  - Tier 3 (10Hz): Learning loop updates                                   │
│  │  - Tier 4 (0.1Hz): Cloud sync                                             │
│  │                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import os
import time
import logging
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Import Meta AI models
from .dinov3 import DINOv3Encoder, DINOv3Config, DINOv3Features, DINOv3ModelSize
from .sam3 import SAM3Segmenter, SAM3Config, SegmentationResult, SAM3ModelSize
from .vjepa2 import (
    VJEPA2WorldModel, VJEPA2Config, WorldModelPrediction,
    VJEPA2ModelSize, WorldState, ActionPlan
)
from .privacy_wrapper import (
    MetaAIPrivacyWrapper, PrivacyConfig, EncryptedFeatures
)

# Import platform components
try:
    from src.platform.cloud.moe_skill_router import (
        CloudSkillService, MoESkillRouter, SkillRequest, SkillResponse
    )
    HAS_MOE = True
except ImportError:
    HAS_MOE = False
    CloudSkillService = None
    MoESkillRouter = None
    SkillRequest = None
    SkillResponse = None
    logger.warning("MoE skill router not available")

try:
    from src.core.robot_skill_invoker import (
        RobotSkillInvoker, SkillInvocationRequest, SkillInvocationResult,
        ObservationState, InvocationMode
    )
    HAS_INVOKER = True
except ImportError:
    HAS_INVOKER = False
    RobotSkillInvoker = None
    SkillInvocationRequest = None
    SkillInvocationResult = None
    ObservationState = None
    InvocationMode = None
    logger.warning("Robot skill invoker not available")


# =============================================================================
# Configuration
# =============================================================================

class PerceptionTier(str, Enum):
    """Perception timing tiers."""
    SAFETY = "safety"       # 1kHz - collision prediction only
    CONTROL = "control"     # 100Hz - full perception
    LEARNING = "learning"   # 10Hz - skill updates
    CLOUD = "cloud"         # 0.1Hz - cloud sync


@dataclass
class PerceptionConfig:
    """Configuration for unified perception pipeline."""
    # Model configurations
    dinov3_config: DINOv3Config = field(default_factory=lambda: DINOv3Config(
        model_size=DINOv3ModelSize.LARGE,
    ))
    sam3_config: SAM3Config = field(default_factory=lambda: SAM3Config(
        model_size=SAM3ModelSize.LARGE,
    ))
    vjepa2_config: VJEPA2Config = field(default_factory=lambda: VJEPA2Config(
        model_size=VJEPA2ModelSize.LARGE,
        enable_safety_prediction=True,
    ))
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)

    # Pipeline settings
    enable_dinov3: bool = True
    enable_sam3: bool = True
    enable_vjepa2: bool = True
    enable_privacy: bool = True
    enable_moe_routing: bool = True

    # Feature fusion
    fusion_method: str = "concat"  # "concat", "attention", "mlp"
    fused_dim: int = 2048

    # Timing (Hz)
    safety_rate_hz: float = 1000.0
    control_rate_hz: float = 100.0
    learning_rate_hz: float = 10.0
    cloud_rate_hz: float = 0.1

    # Safety thresholds
    collision_threshold: float = 0.7
    emergency_stop_threshold: float = 0.9

    # Device
    device: str = "cuda"


@dataclass
class PerceptionResult:
    """Output from unified perception pipeline."""
    # Features
    dinov3_features: Optional[DINOv3Features] = None
    segmentation: Optional[SegmentationResult] = None
    world_prediction: Optional[WorldModelPrediction] = None

    # Fused features
    fused_features: Optional[np.ndarray] = None
    encrypted_features: Optional[EncryptedFeatures] = None

    # Skill routing
    skill_ids: List[str] = field(default_factory=list)
    skill_weights: Optional[np.ndarray] = None

    # Safety
    is_safe: bool = True
    collision_probability: float = 0.0
    safety_action: str = "CONTINUE"  # CONTINUE, SLOW, STOP

    # Timing
    total_time_ms: float = 0.0
    tier: PerceptionTier = PerceptionTier.CONTROL

    # Manipulation target (if segmentation ran)
    manipulation_target: Optional[Dict[str, Any]] = None


# =============================================================================
# Unified Perception Pipeline
# =============================================================================

class UnifiedPerceptionPipeline:
    """
    Unified perception pipeline integrating Meta AI models.

    Provides:
    - Multi-model perception (DINOv3 + SAM3 + V-JEPA 2)
    - Privacy-preserving feature encryption
    - MoE skill routing
    - 4-tier timing integration
    - Robot skill invocation

    Usage:
        pipeline = UnifiedPerceptionPipeline()
        pipeline.initialize()

        # Full perception (100Hz control tier)
        result = pipeline.process_frame(frame, task="pick up cup")

        # Safety-only perception (1kHz safety tier)
        is_safe = pipeline.check_safety(frame)

        # Get skill actions
        actions = pipeline.get_skill_actions(result)
    """

    def __init__(self, config: PerceptionConfig = None):
        self.config = config or PerceptionConfig()

        # Models
        self.dinov3: Optional[DINOv3Encoder] = None
        self.sam3: Optional[SAM3Segmenter] = None
        self.vjepa2: Optional[VJEPA2WorldModel] = None

        # Privacy wrapper
        self.privacy_wrapper: Optional[MetaAIPrivacyWrapper] = None

        # MoE routing
        self.skill_service: Optional[CloudSkillService] = None

        # Skill invoker
        self.skill_invoker: Optional[RobotSkillInvoker] = None

        # State
        self._is_initialized = False
        self._last_perception_time = 0.0
        self._last_safety_check_time = 0.0

        # Async processing
        self._perception_queue: Queue = Queue(maxsize=10)
        self._result_queue: Queue = Queue(maxsize=10)
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'safety_checks': 0,
            'skills_invoked': 0,
            'emergency_stops': 0,
            'avg_perception_time_ms': 0.0,
            'avg_safety_time_ms': 0.0,
        }

        # Callbacks
        self._safety_callbacks: List[Callable[[float], None]] = []
        self._perception_callbacks: List[Callable[[PerceptionResult], None]] = []

    def initialize(self) -> bool:
        """
        Initialize all pipeline components.

        Returns:
            True if successful
        """
        logger.info("Initializing Unified Perception Pipeline...")

        try:
            # Initialize DINOv3
            if self.config.enable_dinov3:
                logger.info("Loading DINOv3...")
                self.dinov3 = DINOv3Encoder(self.config.dinov3_config)
                self.dinov3.load_model()

            # Initialize SAM3
            if self.config.enable_sam3:
                logger.info("Loading SAM3...")
                self.sam3 = SAM3Segmenter(self.config.sam3_config)
                self.sam3.load_model()

            # Initialize V-JEPA 2
            if self.config.enable_vjepa2:
                logger.info("Loading V-JEPA 2...")
                self.vjepa2 = VJEPA2WorldModel(self.config.vjepa2_config)
                self.vjepa2.load_model()

            # Initialize privacy wrapper
            if self.config.enable_privacy:
                logger.info("Initializing privacy wrapper...")
                self.privacy_wrapper = MetaAIPrivacyWrapper(self.config.privacy_config)

            # Initialize MoE skill service
            if self.config.enable_moe_routing and HAS_MOE:
                logger.info("Initializing MoE skill service...")
                self.skill_service = CloudSkillService()

            # Initialize skill invoker
            if HAS_INVOKER:
                logger.info("Initializing skill invoker...")
                self.skill_invoker = RobotSkillInvoker()
                self.skill_invoker.start()

            self._is_initialized = True
            logger.info("Unified Perception Pipeline initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False

    def shutdown(self):
        """Shutdown pipeline and release resources."""
        self._running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

        if self.skill_invoker:
            self.skill_invoker.stop()

        logger.info("Unified Perception Pipeline shutdown complete")

    def process_frame(
        self,
        frame: np.ndarray,
        task_description: Optional[str] = None,
        tier: PerceptionTier = PerceptionTier.CONTROL,
        proprioception: Optional[np.ndarray] = None,
    ) -> PerceptionResult:
        """
        Process a single frame through the perception pipeline.

        Args:
            frame: Input frame [H, W, 3]
            task_description: Optional task for routing (e.g., "pick up cup")
            tier: Perception tier (affects processing depth)
            proprioception: Optional robot proprioception [DOF]

        Returns:
            PerceptionResult with all perception outputs
        """
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()
        result = PerceptionResult(tier=tier)

        # Tier 1: Safety-only (1kHz)
        if tier == PerceptionTier.SAFETY:
            return self._process_safety_only(frame, result, start_time)

        # Tier 2+: Full perception
        # Step 1: DINOv3 encoding
        if self.dinov3:
            result.dinov3_features = self.dinov3.encode(frame, return_dense=True)

        # Step 2: SAM3 segmentation (if task provided)
        if self.sam3 and task_description:
            result.segmentation = self.sam3.segment_text(frame, task_description)
            result.manipulation_target = self.sam3.get_manipulation_target(frame, task_description)

        # Step 3: V-JEPA 2 prediction
        if self.vjepa2:
            self.vjepa2.add_frame(frame)
            result.world_prediction = self.vjepa2.predict()

            # Extract safety info
            if result.world_prediction.collision_probabilities is not None:
                result.collision_probability = result.world_prediction.max_collision_prob
                result.is_safe = result.collision_probability < self.config.collision_threshold

                if result.collision_probability >= self.config.emergency_stop_threshold:
                    result.safety_action = "STOP"
                    self.stats['emergency_stops'] += 1
                elif result.collision_probability >= self.config.collision_threshold:
                    result.safety_action = "SLOW"
                else:
                    result.safety_action = "CONTINUE"

        # Step 4: Feature fusion
        result.fused_features = self._fuse_features(result)

        # Step 5: Encrypt features (if enabled)
        if self.privacy_wrapper and result.fused_features is not None:
            result.encrypted_features = self.privacy_wrapper.encrypt_features(
                result.fused_features,
                source_model="unified"
            )

        # Step 6: MoE skill routing
        if self.skill_service and task_description:
            skill_response = self._route_skills(result, task_description)
            if skill_response:
                result.skill_ids = [s.metadata.id for s in skill_response.skills]
                result.skill_weights = np.array(skill_response.routing_weights)

        # Timing
        result.total_time_ms = (time.time() - start_time) * 1000

        # Update stats
        self.stats['frames_processed'] += 1
        alpha = 0.1
        self.stats['avg_perception_time_ms'] = (
            alpha * result.total_time_ms +
            (1 - alpha) * self.stats['avg_perception_time_ms']
        )

        # Call callbacks
        for callback in self._perception_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Perception callback error: {e}")

        return result

    def _process_safety_only(
        self,
        frame: np.ndarray,
        result: PerceptionResult,
        start_time: float,
    ) -> PerceptionResult:
        """Fast safety-only processing for 1kHz tier."""
        if self.vjepa2:
            # Only run collision prediction
            self.vjepa2.add_frame(frame)
            prediction = self.vjepa2.predict(num_steps=1)

            if prediction.collision_probabilities is not None:
                result.collision_probability = float(prediction.collision_probabilities[0])
                result.is_safe = result.collision_probability < self.config.collision_threshold

                if result.collision_probability >= self.config.emergency_stop_threshold:
                    result.safety_action = "STOP"
                    self.stats['emergency_stops'] += 1
                elif result.collision_probability >= self.config.collision_threshold:
                    result.safety_action = "SLOW"

        result.total_time_ms = (time.time() - start_time) * 1000

        self.stats['safety_checks'] += 1
        alpha = 0.1
        self.stats['avg_safety_time_ms'] = (
            alpha * result.total_time_ms +
            (1 - alpha) * self.stats['avg_safety_time_ms']
        )

        # Call safety callbacks
        for callback in self._safety_callbacks:
            try:
                callback(result.collision_probability)
            except Exception as e:
                logger.error(f"Safety callback error: {e}")

        return result

    def _fuse_features(self, result: PerceptionResult) -> Optional[np.ndarray]:
        """Fuse features from multiple models."""
        features = []

        # DINOv3 features
        if result.dinov3_features is not None:
            features.append(result.dinov3_features.global_features.flatten())

        # Segmentation features (mask centroid + area)
        if result.segmentation is not None and result.segmentation.masks:
            seg_features = []
            for mask in result.segmentation.masks[:5]:  # Top 5 masks
                if mask.bbox:
                    cx = (mask.bbox[0] + mask.bbox[2]) / 2 / 640  # Normalize
                    cy = (mask.bbox[1] + mask.bbox[3]) / 2 / 480
                    area = mask.area / (640 * 480)
                    seg_features.extend([cx, cy, area, mask.confidence])
            if seg_features:
                # Pad to fixed size
                seg_features = seg_features[:20]
                seg_features += [0] * (20 - len(seg_features))
                features.append(np.array(seg_features, dtype=np.float32))

        # V-JEPA 2 prediction features
        if result.world_prediction is not None:
            features.append(result.world_prediction.future_embeddings[0])  # First future step

        if not features:
            return None

        # Fusion based on method
        if self.config.fusion_method == "concat":
            fused = np.concatenate(features)
        elif self.config.fusion_method == "attention":
            # Simple attention-weighted fusion
            weights = np.softmax(np.ones(len(features)) / len(features))
            padded = [np.pad(f, (0, self.config.fused_dim - len(f))) if len(f) < self.config.fused_dim else f[:self.config.fused_dim] for f in features]
            fused = sum(w * f for w, f in zip(weights, padded))
        else:  # mlp
            fused = np.concatenate(features)

        return fused.astype(np.float32)

    def _route_skills(
        self,
        result: PerceptionResult,
        task_description: str,
    ) -> Optional[SkillResponse]:
        """Route to appropriate skills using MoE."""
        if not self.skill_service:
            return None

        # Create skill request
        request = SkillRequest(
            task_description=task_description,
            task_embedding=result.fused_features.tolist() if result.fused_features is not None else None,
            max_skills=3,
        )

        try:
            response = self.skill_service.request_skills(request)
            return response
        except Exception as e:
            logger.error(f"Skill routing failed: {e}")
            return None

    def check_safety(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        """
        Quick safety check for 1kHz tier.

        Args:
            frame: Input frame

        Returns:
            Tuple of (is_safe, collision_probability, action)
        """
        result = self.process_frame(frame, tier=PerceptionTier.SAFETY)
        return result.is_safe, result.collision_probability, result.safety_action

    def get_skill_actions(
        self,
        result: PerceptionResult,
        proprioception: Optional[np.ndarray] = None,
    ) -> Optional[SkillInvocationResult]:
        """
        Invoke skills based on perception result.

        Args:
            result: Perception result
            proprioception: Current robot state [DOF]

        Returns:
            SkillInvocationResult with actions
        """
        if not self.skill_invoker or not result.skill_ids:
            return None

        # Create observation state
        observation = ObservationState(
            joint_positions=proprioception if proprioception is not None else np.zeros(23),
            joint_velocities=np.zeros(23),
            vision_embedding=result.fused_features,
        )

        # Create invocation request
        request = SkillInvocationRequest(
            skill_ids=result.skill_ids,
            blend_weights=result.skill_weights.tolist() if result.skill_weights is not None else None,
            observation=observation,
            mode=InvocationMode.BLENDED,
            check_safety=result.is_safe,
        )

        try:
            invocation_result = self.skill_invoker.invoke(request)
            self.stats['skills_invoked'] += 1
            return invocation_result
        except Exception as e:
            logger.error(f"Skill invocation failed: {e}")
            return None

    def plan_to_target(
        self,
        current_frame: np.ndarray,
        target_description: str,
        proprioception: Optional[np.ndarray] = None,
    ) -> Optional[ActionPlan]:
        """
        Plan action sequence to reach target.

        Args:
            current_frame: Current camera frame
            target_description: Target description (e.g., "the red cup")
            proprioception: Current robot state

        Returns:
            ActionPlan with action sequence
        """
        if not self.vjepa2:
            return None

        # Get current perception
        current_result = self.process_frame(
            current_frame,
            task_description=target_description,
        )

        if current_result.fused_features is None:
            return None

        # Create current state
        current_state = WorldState(
            frame_embedding=current_result.fused_features,
            joint_positions=proprioception,
        )

        # For goal, we need to estimate what the world looks like at the goal
        # Use segmentation to find target and create goal embedding
        if current_result.manipulation_target:
            # Modify embedding to represent goal state
            target_centroid = current_result.manipulation_target['centroid']
            # Simple: shift object towards robot
            goal_embedding = current_result.fused_features.copy()
            # This is a simplified goal representation

            goal_state = WorldState(
                frame_embedding=goal_embedding,
            )

            return self.vjepa2.plan_to_goal(current_state, goal_state)

        return None

    def register_safety_callback(self, callback: Callable[[float], None]):
        """Register callback for safety events."""
        self._safety_callbacks.append(callback)

    def register_perception_callback(self, callback: Callable[[PerceptionResult], None]):
        """Register callback for perception results."""
        self._perception_callbacks.append(callback)

    def start_async_processing(self):
        """Start async processing thread."""
        if self._running:
            return

        self._running = True
        self._processing_thread = threading.Thread(
            target=self._async_processing_loop,
            daemon=True,
        )
        self._processing_thread.start()
        logger.info("Started async perception processing")

    def stop_async_processing(self):
        """Stop async processing."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

    def submit_frame(
        self,
        frame: np.ndarray,
        task_description: Optional[str] = None,
    ):
        """Submit frame for async processing."""
        try:
            self._perception_queue.put_nowait({
                'frame': frame,
                'task': task_description,
                'timestamp': time.time(),
            })
        except:
            pass  # Drop if queue full

    def get_result(self, timeout: float = 0.1) -> Optional[PerceptionResult]:
        """Get result from async processing."""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None

    def _async_processing_loop(self):
        """Background processing loop."""
        while self._running:
            try:
                item = self._perception_queue.get(timeout=0.1)
                result = self.process_frame(
                    item['frame'],
                    task_description=item['task'],
                )
                self._result_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Async processing error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        model_stats = {}

        if self.dinov3:
            model_stats['dinov3'] = self.dinov3.get_statistics()
        if self.sam3:
            model_stats['sam3'] = self.sam3.get_statistics()
        if self.vjepa2:
            model_stats['vjepa2'] = self.vjepa2.get_statistics()
        if self.privacy_wrapper:
            model_stats['privacy'] = self.privacy_wrapper.get_statistics()

        return {
            **self.stats,
            'is_initialized': self._is_initialized,
            'models': model_stats,
        }


# =============================================================================
# Testing
# =============================================================================

def test_unified_pipeline():
    """Test unified perception pipeline."""
    print("\n" + "=" * 60)
    print("UNIFIED PERCEPTION PIPELINE TEST")
    print("=" * 60)

    # Create pipeline
    config = PerceptionConfig(
        enable_dinov3=True,
        enable_sam3=True,
        enable_vjepa2=True,
        enable_privacy=True,
        enable_moe_routing=False,  # Skip MoE for test
    )
    pipeline = UnifiedPerceptionPipeline(config)

    print("\n1. Initialize Pipeline")
    print("-" * 40)
    success = pipeline.initialize()
    print(f"   Initialized: {success}")

    print("\n2. Full Perception (Control Tier)")
    print("-" * 40)

    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    result = pipeline.process_frame(
        test_frame,
        task_description="pick up the red cup",
        tier=PerceptionTier.CONTROL,
    )

    print(f"   Total time: {result.total_time_ms:.2f}ms")
    print(f"   Is safe: {result.is_safe}")
    print(f"   Collision prob: {result.collision_probability:.3f}")
    print(f"   Safety action: {result.safety_action}")

    if result.dinov3_features:
        print(f"   DINOv3 features: {result.dinov3_features.global_features.shape}")
    if result.segmentation:
        print(f"   Segmentation masks: {result.segmentation.num_masks}")
    if result.world_prediction:
        print(f"   World prediction horizon: {result.world_prediction.horizon}")
    if result.fused_features is not None:
        print(f"   Fused features: {result.fused_features.shape}")
    if result.encrypted_features:
        print(f"   Encrypted size: {result.encrypted_features.size_bytes / 1024:.2f} KB")
    if result.manipulation_target:
        print(f"   Manipulation target: {result.manipulation_target['centroid']}")

    print("\n3. Safety-Only Check (Safety Tier)")
    print("-" * 40)

    is_safe, collision_prob, action = pipeline.check_safety(test_frame)
    print(f"   Is safe: {is_safe}")
    print(f"   Collision prob: {collision_prob:.3f}")
    print(f"   Action: {action}")

    print("\n4. Multiple Frames")
    print("-" * 40)

    times = []
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, task_description="grasp object")
        times.append(result.total_time_ms)
        print(f"   Frame {i+1}: {result.total_time_ms:.2f}ms")

    print(f"   Average: {np.mean(times):.2f}ms")

    print("\n5. Action Planning")
    print("-" * 40)

    plan = pipeline.plan_to_target(test_frame, "the red cup")
    if plan:
        print(f"   Plan actions: {plan.actions.shape}")
        print(f"   Success probability: {plan.success_probability:.3f}")
        print(f"   Collision free: {plan.collision_free}")
    else:
        print("   Planning skipped (requires full V-JEPA 2)")

    print("\n6. Statistics")
    print("-" * 40)

    stats = pipeline.get_statistics()
    print(f"   Frames processed: {stats['frames_processed']}")
    print(f"   Safety checks: {stats['safety_checks']}")
    print(f"   Avg perception time: {stats['avg_perception_time_ms']:.2f}ms")
    print(f"   Avg safety time: {stats['avg_safety_time_ms']:.2f}ms")

    # Shutdown
    pipeline.shutdown()

    print("\n" + "=" * 60)
    print("UNIFIED PIPELINE TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_unified_pipeline()

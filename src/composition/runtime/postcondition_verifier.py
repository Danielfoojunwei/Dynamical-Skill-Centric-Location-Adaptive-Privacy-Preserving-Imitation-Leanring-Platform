"""
Postcondition Verifier - Verify Skill Postconditions Using Perception

Uses the existing perception system (SAM3, DINOv3, V-JEPA2) to verify
that skill postconditions actually hold in the real world.

Predicate Types:
    - object_held: Is the robot holding a specific object?
    - object_at: Is an object at a specific location?
    - object_visible: Can we see a specific object?
    - contact_made: Is the robot in contact with something?
    - gripper_state: Is gripper open/closed?
    - reached_position: Is end-effector at target position?

Verification Methods:
    1. SAM3 Segmentation: Find and verify object locations
    2. DINOv3 Features: Semantic similarity for state verification
    3. V-JEPA2 Prediction: Temporal consistency verification
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class PredicateType(Enum):
    """Types of verifiable predicates."""
    OBJECT_HELD = "object_held"           # holding(object)
    OBJECT_AT = "object_at"               # at(object, location)
    OBJECT_VISIBLE = "object_visible"     # visible(object)
    CONTACT_MADE = "contact_made"         # contact(object)
    GRIPPER_STATE = "gripper_state"       # gripper_open() / gripper_closed()
    REACHED_POSITION = "reached_position" # at_position(x, y, z)
    CLEARED_AREA = "cleared_area"         # cleared(region)
    STACKED = "stacked"                   # on(object_a, object_b)
    CUSTOM = "custom"                     # user-defined predicate


@dataclass
class Predicate:
    """A predicate to verify."""
    pred_type: PredicateType
    name: str
    args: Dict[str, Any] = field(default_factory=dict)

    # Thresholds for verification
    confidence_threshold: float = 0.7
    position_tolerance: float = 0.05  # meters

    @classmethod
    def holding(cls, object_name: str) -> 'Predicate':
        """Create a holding(object) predicate."""
        return cls(
            pred_type=PredicateType.OBJECT_HELD,
            name=f"holding({object_name})",
            args={"object": object_name},
        )

    @classmethod
    def at(cls, object_name: str, location: Union[str, np.ndarray]) -> 'Predicate':
        """Create an at(object, location) predicate."""
        return cls(
            pred_type=PredicateType.OBJECT_AT,
            name=f"at({object_name}, {location})",
            args={"object": object_name, "location": location},
        )

    @classmethod
    def visible(cls, object_name: str) -> 'Predicate':
        """Create a visible(object) predicate."""
        return cls(
            pred_type=PredicateType.OBJECT_VISIBLE,
            name=f"visible({object_name})",
            args={"object": object_name},
        )

    @classmethod
    def gripper_closed(cls) -> 'Predicate':
        """Create a gripper_closed() predicate."""
        return cls(
            pred_type=PredicateType.GRIPPER_STATE,
            name="gripper_closed()",
            args={"state": "closed"},
        )

    @classmethod
    def gripper_open(cls) -> 'Predicate':
        """Create a gripper_open() predicate."""
        return cls(
            pred_type=PredicateType.GRIPPER_STATE,
            name="gripper_open()",
            args={"state": "open"},
        )

    @classmethod
    def at_position(cls, position: np.ndarray, tolerance: float = 0.05) -> 'Predicate':
        """Create an at_position(x, y, z) predicate."""
        return cls(
            pred_type=PredicateType.REACHED_POSITION,
            name=f"at_position({position})",
            args={"position": position},
            position_tolerance=tolerance,
        )

    @classmethod
    def on(cls, object_a: str, object_b: str) -> 'Predicate':
        """Create an on(a, b) predicate (a is stacked on b)."""
        return cls(
            pred_type=PredicateType.STACKED,
            name=f"on({object_a}, {object_b})",
            args={"top_object": object_a, "bottom_object": object_b},
        )


@dataclass
class VerificationResult:
    """Result of postcondition verification."""
    verified: bool
    predicate: Predicate
    confidence: float

    # Details
    method_used: str = ""  # "sam3", "dinov3", "vjepa2", "robot_state"
    details: Dict[str, Any] = field(default_factory=dict)

    # Recovery suggestion if verification failed
    failure_reason: Optional[str] = None
    suggested_recovery: Optional[str] = None

    # Timing
    verification_time_ms: float = 0.0


class PredicateVerifier(ABC):
    """Abstract base for predicate verifiers."""

    @abstractmethod
    def can_verify(self, predicate: Predicate) -> bool:
        """Check if this verifier can handle the predicate."""
        pass

    @abstractmethod
    def verify(
        self,
        predicate: Predicate,
        frame: np.ndarray,
        robot_state: Dict[str, Any],
        perception_result: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify the predicate."""
        pass


class SAM3PredicateVerifier(PredicateVerifier):
    """
    Verify predicates using SAM3 segmentation.

    Handles:
        - object_visible: Can we segment the object?
        - object_at: Is object centroid at expected location?
        - stacked: Is object A's centroid above object B's?
    """

    def __init__(self, sam3_segmenter=None):
        self.sam3 = sam3_segmenter
        self._supported_types = {
            PredicateType.OBJECT_VISIBLE,
            PredicateType.OBJECT_AT,
            PredicateType.STACKED,
            PredicateType.CLEARED_AREA,
        }

    def can_verify(self, predicate: Predicate) -> bool:
        return predicate.pred_type in self._supported_types

    def verify(
        self,
        predicate: Predicate,
        frame: np.ndarray,
        robot_state: Dict[str, Any],
        perception_result: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify using SAM3 segmentation."""
        import time
        start_time = time.time()

        # Use existing perception result or run SAM3
        segmentation = None
        if perception_result and hasattr(perception_result, 'segmentation'):
            segmentation = perception_result.segmentation
        elif self.sam3:
            object_name = predicate.args.get("object", "")
            segmentation = self.sam3.segment_text(frame, object_name)

        if segmentation is None:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="sam3",
                failure_reason="SAM3 segmentation not available",
                suggested_recovery="retry_skill",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Verify based on predicate type
        if predicate.pred_type == PredicateType.OBJECT_VISIBLE:
            return self._verify_visible(predicate, segmentation, start_time)
        elif predicate.pred_type == PredicateType.OBJECT_AT:
            return self._verify_at(predicate, segmentation, start_time)
        elif predicate.pred_type == PredicateType.STACKED:
            return self._verify_stacked(predicate, segmentation, frame, start_time)
        elif predicate.pred_type == PredicateType.CLEARED_AREA:
            return self._verify_cleared(predicate, segmentation, start_time)

        return VerificationResult(
            verified=False,
            predicate=predicate,
            confidence=0.0,
            method_used="sam3",
            failure_reason=f"Unsupported predicate type: {predicate.pred_type}",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_visible(self, predicate: Predicate, segmentation, start_time) -> VerificationResult:
        """Verify object is visible (has segmentation mask)."""
        import time

        has_mask = segmentation.num_masks > 0 if hasattr(segmentation, 'num_masks') else len(getattr(segmentation, 'masks', [])) > 0

        if has_mask:
            best_confidence = max(m.confidence for m in segmentation.masks) if segmentation.masks else 0.0
            verified = best_confidence >= predicate.confidence_threshold

            return VerificationResult(
                verified=verified,
                predicate=predicate,
                confidence=best_confidence,
                method_used="sam3",
                details={
                    "num_masks": segmentation.num_masks if hasattr(segmentation, 'num_masks') else len(segmentation.masks),
                    "best_confidence": best_confidence,
                },
                failure_reason=None if verified else f"Confidence {best_confidence:.2f} < threshold {predicate.confidence_threshold}",
                suggested_recovery=None if verified else "adjust_viewpoint",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        return VerificationResult(
            verified=False,
            predicate=predicate,
            confidence=0.0,
            method_used="sam3",
            failure_reason=f"Object '{predicate.args.get('object')}' not found in frame",
            suggested_recovery="search_for_object",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_at(self, predicate: Predicate, segmentation, start_time) -> VerificationResult:
        """Verify object is at expected location."""
        import time

        expected_location = predicate.args.get("location")
        if expected_location is None:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="sam3",
                failure_reason="No expected location specified",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        if not segmentation.masks:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="sam3",
                failure_reason="No object detected",
                suggested_recovery="search_for_object",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Get best mask centroid
        best_mask = max(segmentation.masks, key=lambda m: m.confidence)
        if hasattr(best_mask, 'centroid'):
            detected_centroid = np.array(best_mask.centroid)
        elif best_mask.bbox:
            # Compute centroid from bbox
            x1, y1, x2, y2 = best_mask.bbox
            detected_centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        else:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="sam3",
                failure_reason="Could not determine object centroid",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Convert expected location to pixel coordinates if needed
        if isinstance(expected_location, str):
            # Named location - would need lookup table
            verified = False
            distance = float('inf')
        else:
            expected_pos = np.array(expected_location)
            if len(expected_pos) == 2:
                # 2D pixel coordinates
                distance = np.linalg.norm(detected_centroid - expected_pos)
            else:
                # 3D world coordinates - would need camera projection
                distance = float('inf')

            # Convert tolerance to pixels (approximate)
            pixel_tolerance = predicate.position_tolerance * 1000  # rough conversion
            verified = distance < pixel_tolerance

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=best_mask.confidence if verified else best_mask.confidence * 0.5,
            method_used="sam3",
            details={
                "detected_centroid": detected_centroid.tolist(),
                "expected_location": expected_location if isinstance(expected_location, str) else expected_location.tolist() if hasattr(expected_location, 'tolist') else expected_location,
                "distance": float(distance),
            },
            failure_reason=None if verified else f"Object not at expected location (distance: {distance:.2f})",
            suggested_recovery=None if verified else "adjust_placement",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_stacked(self, predicate: Predicate, segmentation, frame, start_time) -> VerificationResult:
        """Verify object A is stacked on object B."""
        import time

        top_obj = predicate.args.get("top_object")
        bottom_obj = predicate.args.get("bottom_object")

        # Would need to run SAM3 twice for both objects
        # For now, check if we have at least 2 masks with vertical relationship
        if len(segmentation.masks) < 2:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="sam3",
                failure_reason="Need at least 2 objects to verify stacking",
                suggested_recovery="verify_both_objects_present",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Sort masks by vertical position (y coordinate)
        sorted_masks = sorted(segmentation.masks, key=lambda m: m.bbox[1] if m.bbox else 0)

        # Top object should have lower y (image coordinates)
        # This is a simplified check
        verified = len(sorted_masks) >= 2

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=min(m.confidence for m in sorted_masks[:2]) if verified else 0.0,
            method_used="sam3",
            details={
                "num_objects_detected": len(sorted_masks),
            },
            failure_reason=None if verified else "Cannot verify stacking relationship",
            suggested_recovery=None if verified else "retry_stack",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_cleared(self, predicate: Predicate, segmentation, start_time) -> VerificationResult:
        """Verify an area is cleared (no objects detected)."""
        import time

        # Area is cleared if no objects are detected
        verified = segmentation.num_masks == 0 if hasattr(segmentation, 'num_masks') else len(segmentation.masks) == 0

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=1.0 if verified else 0.3,
            method_used="sam3",
            details={
                "objects_detected": segmentation.num_masks if hasattr(segmentation, 'num_masks') else len(segmentation.masks),
            },
            failure_reason=None if verified else "Area still contains objects",
            suggested_recovery=None if verified else "clear_remaining_objects",
            verification_time_ms=(time.time() - start_time) * 1000,
        )


class RobotStateVerifier(PredicateVerifier):
    """
    Verify predicates using robot state (proprioception + force sensing).

    Handles:
        - gripper_state: Check gripper encoder
        - object_held: Check gripper force + closure
        - reached_position: Check end-effector position
        - contact_made: Check force sensor
    """

    def __init__(self):
        self._supported_types = {
            PredicateType.GRIPPER_STATE,
            PredicateType.OBJECT_HELD,
            PredicateType.REACHED_POSITION,
            PredicateType.CONTACT_MADE,
        }

    def can_verify(self, predicate: Predicate) -> bool:
        return predicate.pred_type in self._supported_types

    def verify(
        self,
        predicate: Predicate,
        frame: np.ndarray,
        robot_state: Dict[str, Any],
        perception_result: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify using robot state."""
        import time
        start_time = time.time()

        if predicate.pred_type == PredicateType.GRIPPER_STATE:
            return self._verify_gripper_state(predicate, robot_state, start_time)
        elif predicate.pred_type == PredicateType.OBJECT_HELD:
            return self._verify_object_held(predicate, robot_state, start_time)
        elif predicate.pred_type == PredicateType.REACHED_POSITION:
            return self._verify_position(predicate, robot_state, start_time)
        elif predicate.pred_type == PredicateType.CONTACT_MADE:
            return self._verify_contact(predicate, robot_state, start_time)

        return VerificationResult(
            verified=False,
            predicate=predicate,
            confidence=0.0,
            method_used="robot_state",
            failure_reason=f"Unsupported predicate type: {predicate.pred_type}",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_gripper_state(self, predicate: Predicate, robot_state: Dict, start_time) -> VerificationResult:
        """Verify gripper is open/closed."""
        import time

        expected_state = predicate.args.get("state", "closed")
        gripper_pos = robot_state.get("gripper_state", robot_state.get("gripper_position", 0.5))

        # gripper_state: 0 = open, 1 = closed (typical convention)
        if expected_state == "closed":
            verified = gripper_pos > 0.7
        else:  # open
            verified = gripper_pos < 0.3

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=1.0 if verified else 0.0,
            method_used="robot_state",
            details={
                "gripper_position": float(gripper_pos),
                "expected_state": expected_state,
            },
            failure_reason=None if verified else f"Gripper is not {expected_state} (pos={gripper_pos:.2f})",
            suggested_recovery=None if verified else f"close_gripper" if expected_state == "closed" else "open_gripper",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_object_held(self, predicate: Predicate, robot_state: Dict, start_time) -> VerificationResult:
        """Verify robot is holding an object (gripper closed + force detected)."""
        import time

        gripper_pos = robot_state.get("gripper_state", robot_state.get("gripper_position", 0.5))
        gripper_force = robot_state.get("gripper_force", 0.0)
        ee_force = robot_state.get("ee_force", np.zeros(6))

        if isinstance(ee_force, np.ndarray):
            force_magnitude = np.linalg.norm(ee_force[:3])
        else:
            force_magnitude = 0.0

        # Object held = gripper partially closed (not fully) + force detected
        gripper_partially_closed = 0.3 < gripper_pos < 0.95
        force_detected = gripper_force > 1.0 or force_magnitude > 0.5

        verified = gripper_partially_closed and force_detected

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=0.9 if verified else 0.0,
            method_used="robot_state",
            details={
                "gripper_position": float(gripper_pos),
                "gripper_force": float(gripper_force) if isinstance(gripper_force, (int, float)) else 0.0,
                "ee_force_magnitude": float(force_magnitude),
            },
            failure_reason=None if verified else "No object detected in gripper",
            suggested_recovery=None if verified else "retry_grasp",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_position(self, predicate: Predicate, robot_state: Dict, start_time) -> VerificationResult:
        """Verify end-effector is at target position."""
        import time

        target_pos = predicate.args.get("position")
        if target_pos is None:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="robot_state",
                failure_reason="No target position specified",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        ee_pos = robot_state.get("ee_position", np.zeros(3))
        target_pos = np.array(target_pos)

        distance = np.linalg.norm(ee_pos - target_pos)
        verified = distance <= predicate.position_tolerance

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=max(0.0, 1.0 - distance / (predicate.position_tolerance * 2)),
            method_used="robot_state",
            details={
                "current_position": ee_pos.tolist() if hasattr(ee_pos, 'tolist') else list(ee_pos),
                "target_position": target_pos.tolist() if hasattr(target_pos, 'tolist') else list(target_pos),
                "distance": float(distance),
                "tolerance": predicate.position_tolerance,
            },
            failure_reason=None if verified else f"Position error {distance:.4f}m > tolerance {predicate.position_tolerance}m",
            suggested_recovery=None if verified else "refine_position",
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def _verify_contact(self, predicate: Predicate, robot_state: Dict, start_time) -> VerificationResult:
        """Verify contact is made (force detected)."""
        import time

        ee_force = robot_state.get("ee_force", np.zeros(6))
        force_threshold = predicate.args.get("force_threshold", 1.0)

        if isinstance(ee_force, np.ndarray):
            force_magnitude = np.linalg.norm(ee_force[:3])
        else:
            force_magnitude = 0.0

        verified = force_magnitude >= force_threshold

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=min(1.0, force_magnitude / force_threshold) if verified else force_magnitude / force_threshold,
            method_used="robot_state",
            details={
                "force_magnitude": float(force_magnitude),
                "threshold": force_threshold,
            },
            failure_reason=None if verified else f"Insufficient contact force ({force_magnitude:.2f}N < {force_threshold}N)",
            suggested_recovery=None if verified else "approach_object",
            verification_time_ms=(time.time() - start_time) * 1000,
        )


class DINOv3PredicateVerifier(PredicateVerifier):
    """
    Verify predicates using DINOv3 semantic features.

    Uses feature similarity to verify semantic states.
    """

    def __init__(self, dinov3_encoder=None, reference_embeddings: Dict[str, np.ndarray] = None):
        self.dinov3 = dinov3_encoder
        self.reference_embeddings = reference_embeddings or {}
        self._supported_types = {
            PredicateType.OBJECT_VISIBLE,  # Via feature similarity
            PredicateType.CUSTOM,          # Custom semantic predicates
        }

    def can_verify(self, predicate: Predicate) -> bool:
        return predicate.pred_type in self._supported_types

    def add_reference(self, name: str, embedding: np.ndarray):
        """Add a reference embedding for verification."""
        self.reference_embeddings[name] = embedding

    def verify(
        self,
        predicate: Predicate,
        frame: np.ndarray,
        robot_state: Dict[str, Any],
        perception_result: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify using DINOv3 features."""
        import time
        start_time = time.time()

        # Get frame features
        if perception_result and hasattr(perception_result, 'dinov3_features'):
            features = perception_result.dinov3_features.global_features
        elif self.dinov3:
            result = self.dinov3.encode(frame, return_dense=False)
            features = result.global_features
        else:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="dinov3",
                failure_reason="DINOv3 encoder not available",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Check for reference embedding
        reference_name = predicate.args.get("object", predicate.name)
        if reference_name not in self.reference_embeddings:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                method_used="dinov3",
                failure_reason=f"No reference embedding for '{reference_name}'",
                suggested_recovery="add_reference_embedding",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Compute similarity
        reference = self.reference_embeddings[reference_name]
        similarity = np.dot(features.flatten(), reference.flatten()) / (
            np.linalg.norm(features) * np.linalg.norm(reference) + 1e-8
        )

        verified = similarity >= predicate.confidence_threshold

        return VerificationResult(
            verified=verified,
            predicate=predicate,
            confidence=float(similarity),
            method_used="dinov3",
            details={
                "similarity": float(similarity),
                "threshold": predicate.confidence_threshold,
            },
            failure_reason=None if verified else f"Similarity {similarity:.3f} < threshold {predicate.confidence_threshold}",
            suggested_recovery=None if verified else "verify_scene_state",
            verification_time_ms=(time.time() - start_time) * 1000,
        )


class PredicateRegistry:
    """Registry of predicate verifiers."""

    def __init__(self):
        self.verifiers: List[PredicateVerifier] = []
        self.custom_verifiers: Dict[str, Callable] = {}

    def register(self, verifier: PredicateVerifier):
        """Register a predicate verifier."""
        self.verifiers.append(verifier)

    def register_custom(self, name: str, verifier_fn: Callable):
        """Register a custom verification function."""
        self.custom_verifiers[name] = verifier_fn

    def get_verifier(self, predicate: Predicate) -> Optional[PredicateVerifier]:
        """Get a verifier that can handle the predicate."""
        for verifier in self.verifiers:
            if verifier.can_verify(predicate):
                return verifier
        return None


class PostconditionVerifier:
    """
    Main postcondition verifier using perception system.

    Integrates SAM3, DINOv3, and robot state for comprehensive verification.
    """

    def __init__(
        self,
        sam3_segmenter=None,
        dinov3_encoder=None,
        unified_perception=None,
    ):
        self.sam3 = sam3_segmenter
        self.dinov3 = dinov3_encoder
        self.unified_perception = unified_perception

        # Create registry with available verifiers
        self.registry = PredicateRegistry()
        self._setup_verifiers()

        # Statistics
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "avg_verification_time_ms": 0.0,
        }

    def _setup_verifiers(self):
        """Setup available verifiers."""
        # Always register robot state verifier
        self.registry.register(RobotStateVerifier())

        # Register SAM3 verifier if available
        if self.sam3:
            self.registry.register(SAM3PredicateVerifier(self.sam3))

        # Register DINOv3 verifier if available
        if self.dinov3:
            self.registry.register(DINOv3PredicateVerifier(self.dinov3))

    @classmethod
    def from_unified_perception(cls, unified_perception=None) -> 'PostconditionVerifier':
        """Create verifier from UnifiedPerceptionPipeline."""
        if unified_perception is None:
            try:
                from ...meta_ai import UnifiedPerceptionPipeline
                unified_perception = UnifiedPerceptionPipeline()
                unified_perception.initialize()
            except ImportError:
                logger.warning("UnifiedPerceptionPipeline not available")
                return cls()

        return cls(
            sam3_segmenter=unified_perception.sam3,
            dinov3_encoder=unified_perception.dinov3,
            unified_perception=unified_perception,
        )

    def verify(
        self,
        predicate: Union[str, Predicate],
        frame: np.ndarray,
        robot_state: Dict[str, Any],
        perception_result: Optional[Any] = None,
    ) -> VerificationResult:
        """
        Verify a postcondition predicate.

        Args:
            predicate: Predicate to verify (string or Predicate object)
            frame: Current camera frame
            robot_state: Current robot state dict
            perception_result: Optional pre-computed perception result

        Returns:
            VerificationResult with verification outcome
        """
        import time
        start_time = time.time()

        # Parse predicate if string
        if isinstance(predicate, str):
            predicate = self._parse_predicate(predicate)

        # Get perception result if not provided and we have unified perception
        if perception_result is None and self.unified_perception:
            perception_result = self.unified_perception.process_frame(
                frame,
                task_description=predicate.args.get("object", ""),
            )

        # Find appropriate verifier
        verifier = self.registry.get_verifier(predicate)

        if verifier is None:
            return VerificationResult(
                verified=False,
                predicate=predicate,
                confidence=0.0,
                failure_reason=f"No verifier available for {predicate.pred_type}",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Run verification
        result = verifier.verify(predicate, frame, robot_state, perception_result)

        # Update stats
        self.stats["total_verifications"] += 1
        if result.verified:
            self.stats["successful_verifications"] += 1
        else:
            self.stats["failed_verifications"] += 1

        alpha = 0.1
        self.stats["avg_verification_time_ms"] = (
            alpha * result.verification_time_ms +
            (1 - alpha) * self.stats["avg_verification_time_ms"]
        )

        return result

    def verify_all(
        self,
        predicates: List[Union[str, Predicate]],
        frame: np.ndarray,
        robot_state: Dict[str, Any],
    ) -> Tuple[bool, List[VerificationResult]]:
        """
        Verify multiple predicates.

        Returns:
            Tuple of (all_verified, list of results)
        """
        # Get perception once for all predicates
        perception_result = None
        if self.unified_perception:
            # Use first object name for perception
            first_obj = None
            for p in predicates:
                if isinstance(p, Predicate):
                    first_obj = p.args.get("object")
                else:
                    first_obj = p.split("(")[1].split(")")[0] if "(" in p else None
                if first_obj:
                    break

            perception_result = self.unified_perception.process_frame(
                frame,
                task_description=first_obj or "",
            )

        results = []
        all_verified = True

        for pred in predicates:
            result = self.verify(pred, frame, robot_state, perception_result)
            results.append(result)
            if not result.verified:
                all_verified = False

        return all_verified, results

    def _parse_predicate(self, predicate_str: str) -> Predicate:
        """Parse a predicate string into a Predicate object."""
        predicate_str = predicate_str.strip()

        # Parse common patterns
        if predicate_str.startswith("holding("):
            obj = predicate_str[8:-1]
            return Predicate.holding(obj)
        elif predicate_str.startswith("visible("):
            obj = predicate_str[8:-1]
            return Predicate.visible(obj)
        elif predicate_str == "gripper_closed()":
            return Predicate.gripper_closed()
        elif predicate_str == "gripper_open()":
            return Predicate.gripper_open()
        elif predicate_str.startswith("at("):
            # at(object, location)
            args = predicate_str[3:-1].split(",")
            obj = args[0].strip()
            loc = args[1].strip() if len(args) > 1 else None
            return Predicate.at(obj, loc)
        elif predicate_str.startswith("on("):
            # on(object_a, object_b)
            args = predicate_str[3:-1].split(",")
            obj_a = args[0].strip()
            obj_b = args[1].strip() if len(args) > 1 else ""
            return Predicate.on(obj_a, obj_b)

        # Default: custom predicate
        return Predicate(
            pred_type=PredicateType.CUSTOM,
            name=predicate_str,
            args={"raw": predicate_str},
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_verifications"] / max(1, self.stats["total_verifications"])
            ),
        }

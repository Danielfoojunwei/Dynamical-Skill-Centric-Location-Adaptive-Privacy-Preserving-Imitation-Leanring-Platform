"""
Skill Discovery - Automatic Primitive Extraction from Demonstrations

Uses Meta AI models to segment demonstrations and discover skill primitives:
1. SAM3: Segment objects being manipulated → interaction boundaries
2. DINOv3: Extract semantic features → cluster similar segments
3. V-JEPA2: Understand temporal dynamics → predict skill boundaries

This enables compositional training by discovering reusable primitives
from continuous demonstration streams (ONVIF cameras + MANUS gloves).

Key insight: Skill boundaries correlate with:
- Object contact changes (SAM3 detects)
- Feature space transitions (DINOv3 detects)
- Temporal prediction errors (V-JEPA2 detects)

Fusing these signals gives robust skill segmentation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Type of skill boundary detected."""
    OBJECT_CONTACT = "object_contact"       # Object interaction change (SAM3)
    FEATURE_TRANSITION = "feature_transition"  # Semantic feature shift (DINOv3)
    TEMPORAL_BREAK = "temporal_break"       # Prediction error spike (V-JEPA2)
    GRIPPER_STATE = "gripper_state"         # Gripper open/close (MANUS)
    FUSED = "fused"                         # Multiple signals agree


@dataclass
class SegmentBoundary:
    """A detected skill boundary in the demonstration."""
    timestamp: float
    frame_index: int
    boundary_type: BoundaryType
    confidence: float

    # Supporting evidence
    sam3_score: float = 0.0      # Object interaction change score
    dino_score: float = 0.0      # Feature transition score
    vjepa_score: float = 0.0     # Temporal prediction error
    gripper_delta: float = 0.0   # Gripper state change


@dataclass
class DemoSegment:
    """A segment of demonstration between boundaries."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    # Data
    frames: List[Any] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    proprio: List[np.ndarray] = field(default_factory=list)
    hand_pose: List[np.ndarray] = field(default_factory=list)  # MANUS

    # Features (computed)
    dino_features: Optional[np.ndarray] = None  # [T, D] DINOv3 features
    vjepa_latents: Optional[np.ndarray] = None  # [T, D] V-JEPA2 latents
    object_masks: Optional[List[Any]] = None    # SAM3 masks per frame


@dataclass
class DiscoveredPrimitive:
    """A discovered skill primitive from clustering segments."""
    primitive_id: str
    name: str  # Auto-generated or labeled

    # Representative data
    exemplar_segments: List[DemoSegment] = field(default_factory=list)
    centroid_features: Optional[np.ndarray] = None

    # Statistics
    occurrence_count: int = 0
    avg_duration: float = 0.0
    std_duration: float = 0.0

    # Learned model (trained later)
    policy_checkpoint: Optional[str] = None


@dataclass
class SkillCluster:
    """A cluster of similar segments forming a primitive."""
    cluster_id: int
    segments: List[DemoSegment] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    intra_cluster_variance: float = 0.0


@dataclass
class DiscoveryConfig:
    """Configuration for skill discovery."""
    # Boundary detection thresholds
    sam3_threshold: float = 0.3        # Object interaction change threshold
    dino_threshold: float = 0.4        # Feature transition threshold
    vjepa_threshold: float = 0.5       # Prediction error threshold
    gripper_threshold: float = 0.3     # Gripper state change threshold

    # Fusion
    min_boundary_confidence: float = 0.5
    boundary_fusion_window: int = 5    # Frames to fuse nearby boundaries

    # Clustering
    min_segment_length: int = 10       # Minimum frames per segment
    max_segment_length: int = 300      # Maximum frames per segment
    num_clusters: Optional[int] = None # Auto if None
    min_cluster_size: int = 3          # Minimum segments per cluster

    # Feature extraction
    dino_model: str = "dinov3_large"
    vjepa_model: str = "vjepa2_base"

    # Device
    device: str = "cuda"


class SkillSegmenter:
    """
    Segments demonstrations into skills using Meta AI models.

    Fuses signals from:
    - SAM3: Object contact/interaction boundaries
    - DINOv3: Semantic feature transitions
    - V-JEPA2: Temporal prediction errors
    - MANUS: Gripper state changes
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()

        # Models (lazy loaded)
        self._sam3 = None
        self._dino = None
        self._vjepa = None

        self._loaded = False

    def load_models(self):
        """Load Meta AI models for segmentation."""
        if self._loaded:
            return

        logger.info("Loading Meta AI models for skill segmentation...")

        # Load SAM3
        try:
            from ...meta_ai.sam3 import SAM3Segmenter, SAM3Config
            sam3_config = SAM3Config(
                model_size="sam3_large",  # Use large for efficiency
                device=self.config.device,
            )
            self._sam3 = SAM3Segmenter(sam3_config)
            self._sam3.load_model()
            logger.info("Loaded SAM3 for object segmentation")
        except ImportError as e:
            logger.warning(f"SAM3 not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load SAM3: {e}")

        # Load DINOv3
        try:
            from ...meta_ai.dinov3 import DINOv3Encoder, DINOv3Config

            # Map config model name to actual size
            dino_size_map = {
                "dinov3_small": "small",
                "dinov3_base": "base",
                "dinov3_large": "large",
                "dinov3_giant": "giant",
            }
            dino_size = dino_size_map.get(self.config.dino_model, "large")

            dino_config = DINOv3Config(
                model_size=dino_size,
                device=self.config.device,
            )
            self._dino = DINOv3Encoder(dino_config)
            self._dino.load_model()
            logger.info(f"Loaded DINOv3 ({dino_size}) for feature extraction")
        except ImportError as e:
            logger.warning(f"DINOv3 not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load DINOv3: {e}")

        # Load V-JEPA2
        try:
            from ...meta_ai.vjepa2 import VJEPA2WorldModel, VJEPA2Config

            # Map config model name to actual size
            vjepa_size_map = {
                "vjepa2_base": "base",
                "vjepa2_large": "large",
                "vjepa2_huge": "huge",
                "vjepa2_giant": "giant",
            }
            vjepa_size = vjepa_size_map.get(self.config.vjepa_model, "base")

            vjepa_config = VJEPA2Config(
                model_size=vjepa_size,
                device=self.config.device,
            )
            self._vjepa = VJEPA2WorldModel(vjepa_config)
            self._vjepa.load_model()
            logger.info(f"Loaded V-JEPA2 ({vjepa_size}) for temporal modeling")
        except ImportError as e:
            logger.warning(f"V-JEPA2 not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load V-JEPA2: {e}")

        self._loaded = True
        logger.info(f"Skill segmentation models loaded: SAM3={self._sam3 is not None}, "
                   f"DINOv3={self._dino is not None}, V-JEPA2={self._vjepa is not None}")

    def segment_demonstration(
        self,
        frames: List[np.ndarray],
        actions: List[np.ndarray],
        proprio: List[np.ndarray],
        hand_pose: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[List[DemoSegment], List[SegmentBoundary]]:
        """
        Segment a demonstration into skill segments.

        Args:
            frames: Video frames [T, H, W, C]
            actions: Robot actions [T, A]
            proprio: Proprioceptive state [T, P]
            hand_pose: MANUS glove data [T, H] (optional)
            timestamps: Frame timestamps (optional)

        Returns:
            (segments, boundaries)
        """
        if not self._loaded:
            self.load_models()

        T = len(frames)
        if timestamps is None:
            timestamps = [i / 30.0 for i in range(T)]  # Assume 30fps

        # Detect boundaries from each source
        boundaries = []

        # SAM3: Object interaction boundaries
        if self._sam3 is not None:
            sam_boundaries = self._detect_sam3_boundaries(frames)
            boundaries.extend(sam_boundaries)

        # DINOv3: Feature transition boundaries
        if self._dino is not None:
            dino_boundaries = self._detect_dino_boundaries(frames)
            boundaries.extend(dino_boundaries)

        # V-JEPA2: Temporal prediction error boundaries
        if self._vjepa is not None:
            vjepa_boundaries = self._detect_vjepa_boundaries(frames)
            boundaries.extend(vjepa_boundaries)

        # MANUS: Gripper state change boundaries
        if hand_pose is not None:
            gripper_boundaries = self._detect_gripper_boundaries(hand_pose, timestamps)
            boundaries.extend(gripper_boundaries)

        # Fuse boundaries
        fused_boundaries = self._fuse_boundaries(boundaries, T)

        # Create segments between boundaries
        segments = self._create_segments(
            frames, actions, proprio, hand_pose, timestamps, fused_boundaries
        )

        return segments, fused_boundaries

    def _detect_sam3_boundaries(self, frames: List[np.ndarray]) -> List[SegmentBoundary]:
        """Detect boundaries from object interaction changes."""
        boundaries = []

        prev_masks = None
        for i, frame in enumerate(frames):
            # Segment objects in frame
            masks = self._sam3.segment_all(frame)

            if prev_masks is not None:
                # Compute mask change score (IoU-based)
                change_score = self._compute_mask_change(prev_masks, masks)

                if change_score > self.config.sam3_threshold:
                    boundaries.append(SegmentBoundary(
                        timestamp=i / 30.0,
                        frame_index=i,
                        boundary_type=BoundaryType.OBJECT_CONTACT,
                        confidence=change_score,
                        sam3_score=change_score,
                    ))

            prev_masks = masks

        return boundaries

    def _detect_dino_boundaries(self, frames: List[np.ndarray]) -> List[SegmentBoundary]:
        """Detect boundaries from DINOv3 feature transitions."""
        boundaries = []

        # Extract features for all frames
        features = []
        for frame in frames:
            feat = self._dino.extract_features(frame)
            features.append(feat)

        features = np.stack(features)  # [T, D]

        # Compute feature differences
        diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)

        # Normalize
        diffs = (diffs - diffs.mean()) / (diffs.std() + 1e-8)

        # Find peaks above threshold
        for i in range(len(diffs)):
            if diffs[i] > self.config.dino_threshold:
                boundaries.append(SegmentBoundary(
                    timestamp=(i + 1) / 30.0,
                    frame_index=i + 1,
                    boundary_type=BoundaryType.FEATURE_TRANSITION,
                    confidence=float(diffs[i]),
                    dino_score=float(diffs[i]),
                ))

        return boundaries

    def _detect_vjepa_boundaries(self, frames: List[np.ndarray]) -> List[SegmentBoundary]:
        """Detect boundaries from V-JEPA2 prediction errors."""
        boundaries = []

        # V-JEPA2 predicts future frames from context
        # High prediction error = skill boundary
        context_length = 4

        for i in range(context_length, len(frames) - 1):
            context = frames[i - context_length:i]
            target = frames[i]

            # Predict next frame
            predicted = self._vjepa.predict_next(context)

            # Compute prediction error
            error = self._vjepa.compute_prediction_error(predicted, target)

            if error > self.config.vjepa_threshold:
                boundaries.append(SegmentBoundary(
                    timestamp=i / 30.0,
                    frame_index=i,
                    boundary_type=BoundaryType.TEMPORAL_BREAK,
                    confidence=float(error),
                    vjepa_score=float(error),
                ))

        return boundaries

    def _detect_gripper_boundaries(
        self,
        hand_pose: List[np.ndarray],
        timestamps: List[float],
    ) -> List[SegmentBoundary]:
        """Detect boundaries from gripper state changes (MANUS gloves)."""
        boundaries = []

        # Extract gripper aperture from hand pose
        # Assuming hand_pose contains finger joint angles
        gripper_states = []
        for pose in hand_pose:
            # Simplified: average finger flexion
            aperture = 1.0 - np.mean(pose[:5])  # First 5 = finger flexion
            gripper_states.append(aperture)

        gripper_states = np.array(gripper_states)

        # Detect state changes
        diffs = np.abs(np.diff(gripper_states))

        for i in range(len(diffs)):
            if diffs[i] > self.config.gripper_threshold:
                boundaries.append(SegmentBoundary(
                    timestamp=timestamps[i + 1],
                    frame_index=i + 1,
                    boundary_type=BoundaryType.GRIPPER_STATE,
                    confidence=float(diffs[i]),
                    gripper_delta=float(diffs[i]),
                ))

        return boundaries

    def _compute_mask_change(self, prev_masks: List[Any], curr_masks: List[Any]) -> float:
        """Compute change score between mask sets."""
        if not prev_masks or not curr_masks:
            return 0.0

        # Simplified: compute IoU between combined masks
        prev_combined = np.zeros_like(prev_masks[0]) if prev_masks else None
        curr_combined = np.zeros_like(curr_masks[0]) if curr_masks else None

        for m in prev_masks:
            prev_combined = np.logical_or(prev_combined, m)
        for m in curr_masks:
            curr_combined = np.logical_or(curr_combined, m)

        intersection = np.logical_and(prev_combined, curr_combined).sum()
        union = np.logical_or(prev_combined, curr_combined).sum()

        iou = intersection / (union + 1e-8)
        return 1.0 - iou  # Change score = 1 - IoU

    def _fuse_boundaries(
        self,
        boundaries: List[SegmentBoundary],
        num_frames: int,
    ) -> List[SegmentBoundary]:
        """Fuse nearby boundaries from multiple sources."""
        if not boundaries:
            return []

        # Sort by frame index
        boundaries = sorted(boundaries, key=lambda b: b.frame_index)

        fused = []
        i = 0

        while i < len(boundaries):
            # Collect boundaries within fusion window
            group = [boundaries[i]]
            j = i + 1

            while (j < len(boundaries) and
                   boundaries[j].frame_index - boundaries[i].frame_index <= self.config.boundary_fusion_window):
                group.append(boundaries[j])
                j += 1

            # Fuse group into single boundary
            if len(group) >= 2:
                # Multiple sources agree - high confidence
                fused_boundary = SegmentBoundary(
                    timestamp=np.mean([b.timestamp for b in group]),
                    frame_index=int(np.mean([b.frame_index for b in group])),
                    boundary_type=BoundaryType.FUSED,
                    confidence=min(1.0, sum(b.confidence for b in group) / len(group) + 0.2),
                    sam3_score=max(b.sam3_score for b in group),
                    dino_score=max(b.dino_score for b in group),
                    vjepa_score=max(b.vjepa_score for b in group),
                    gripper_delta=max(b.gripper_delta for b in group),
                )
            else:
                fused_boundary = group[0]

            if fused_boundary.confidence >= self.config.min_boundary_confidence:
                fused.append(fused_boundary)

            i = j

        return fused

    def _create_segments(
        self,
        frames: List[np.ndarray],
        actions: List[np.ndarray],
        proprio: List[np.ndarray],
        hand_pose: Optional[List[np.ndarray]],
        timestamps: List[float],
        boundaries: List[SegmentBoundary],
    ) -> List[DemoSegment]:
        """Create segments between boundaries."""
        segments = []

        # Add implicit boundaries at start and end
        boundary_indices = [0] + [b.frame_index for b in boundaries] + [len(frames)]

        for i in range(len(boundary_indices) - 1):
            start_idx = boundary_indices[i]
            end_idx = boundary_indices[i + 1]

            # Check length constraints
            length = end_idx - start_idx
            if length < self.config.min_segment_length:
                continue
            if length > self.config.max_segment_length:
                # Split into smaller segments
                for sub_start in range(start_idx, end_idx, self.config.max_segment_length):
                    sub_end = min(sub_start + self.config.max_segment_length, end_idx)
                    if sub_end - sub_start >= self.config.min_segment_length:
                        segment = self._create_single_segment(
                            frames, actions, proprio, hand_pose, timestamps,
                            sub_start, sub_end
                        )
                        segments.append(segment)
            else:
                segment = self._create_single_segment(
                    frames, actions, proprio, hand_pose, timestamps,
                    start_idx, end_idx
                )
                segments.append(segment)

        return segments

    def _create_single_segment(
        self,
        frames, actions, proprio, hand_pose, timestamps,
        start_idx, end_idx,
    ) -> DemoSegment:
        """Create a single segment."""
        return DemoSegment(
            start_frame=start_idx,
            end_frame=end_idx,
            start_time=timestamps[start_idx],
            end_time=timestamps[end_idx - 1],
            frames=frames[start_idx:end_idx],
            actions=[actions[i] for i in range(start_idx, min(end_idx, len(actions)))],
            proprio=[proprio[i] for i in range(start_idx, min(end_idx, len(proprio)))],
            hand_pose=[hand_pose[i] for i in range(start_idx, min(end_idx, len(hand_pose)))] if hand_pose else [],
        )


class SkillDiscovery:
    """
    Discovers skill primitives by clustering demonstration segments.

    Pipeline:
    1. Segment demonstrations using SkillSegmenter
    2. Extract features for each segment (DINOv3)
    3. Cluster segments to discover primitives
    4. Return DiscoveredPrimitive objects for training
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.segmenter = SkillSegmenter(self.config)

        # Discovered primitives
        self.primitives: Dict[str, DiscoveredPrimitive] = {}
        self.clusters: List[SkillCluster] = []

    def discover_from_demos(
        self,
        demos: List[Dict[str, Any]],
    ) -> List[DiscoveredPrimitive]:
        """
        Discover primitives from a list of demonstrations.

        Args:
            demos: List of demo dicts with 'frames', 'actions', 'proprio', etc.

        Returns:
            List of discovered primitives
        """
        logger.info(f"Discovering skills from {len(demos)} demonstrations...")

        # Step 1: Segment all demonstrations
        all_segments = []
        for demo in demos:
            segments, _ = self.segmenter.segment_demonstration(
                frames=demo['frames'],
                actions=demo['actions'],
                proprio=demo['proprio'],
                hand_pose=demo.get('hand_pose'),
                timestamps=demo.get('timestamps'),
            )
            all_segments.extend(segments)

        logger.info(f"Extracted {len(all_segments)} segments")

        # Step 2: Extract features for clustering
        segment_features = self._extract_segment_features(all_segments)

        # Step 3: Cluster segments
        self.clusters = self._cluster_segments(all_segments, segment_features)

        # Step 4: Create primitives from clusters
        primitives = []
        for cluster in self.clusters:
            primitive = self._cluster_to_primitive(cluster)
            self.primitives[primitive.primitive_id] = primitive
            primitives.append(primitive)

        logger.info(f"Discovered {len(primitives)} skill primitives")
        return primitives

    def discover_from_stream(
        self,
        stream_callback: Callable[[], Optional[Dict[str, Any]]],
        max_demos: int = 1000,
    ) -> List[DiscoveredPrimitive]:
        """
        Discover primitives from a continuous demonstration stream.

        This is designed for integration with ONVIF cameras and MANUS gloves
        that continuously capture human trainer data.

        Args:
            stream_callback: Function that returns next demo or None
            max_demos: Maximum demos to process

        Returns:
            List of discovered primitives
        """
        demos = []

        for _ in range(max_demos):
            demo = stream_callback()
            if demo is None:
                break
            demos.append(demo)

            # Incremental discovery every 100 demos
            if len(demos) % 100 == 0:
                logger.info(f"Processed {len(demos)} demos, running incremental discovery...")
                self.discover_from_demos(demos[-100:])

        # Final discovery pass
        return self.discover_from_demos(demos)

    def _extract_segment_features(
        self,
        segments: List[DemoSegment],
    ) -> np.ndarray:
        """Extract DINOv3 features for each segment."""
        features = []

        for segment in segments:
            if segment.dino_features is not None:
                # Already computed
                feat = segment.dino_features.mean(axis=0)
            elif self.segmenter._dino is not None and segment.frames:
                # Compute features
                frame_feats = []
                for frame in segment.frames[::5]:  # Sample every 5th frame
                    feat = self.segmenter._dino.extract_features(frame)
                    frame_feats.append(feat)
                segment.dino_features = np.stack(frame_feats)
                feat = segment.dino_features.mean(axis=0)
            else:
                # Fallback: use action statistics
                actions = np.array(segment.actions)
                feat = np.concatenate([
                    actions.mean(axis=0),
                    actions.std(axis=0),
                ])

            features.append(feat)

        return np.stack(features)

    def _cluster_segments(
        self,
        segments: List[DemoSegment],
        features: np.ndarray,
    ) -> List[SkillCluster]:
        """Cluster segments to discover primitives."""
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.preprocessing import StandardScaler

        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        # Determine number of clusters
        if self.config.num_clusters is not None:
            n_clusters = self.config.num_clusters
        else:
            # Use elbow method or heuristic
            n_clusters = max(5, len(segments) // 20)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_norm)

        # Create cluster objects
        clusters = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_segments = [segments[i] for i in range(len(segments)) if mask[i]]

            if len(cluster_segments) >= self.config.min_cluster_size:
                clusters.append(SkillCluster(
                    cluster_id=cluster_id,
                    segments=cluster_segments,
                    centroid=kmeans.cluster_centers_[cluster_id],
                    intra_cluster_variance=float(features_norm[mask].var()),
                ))

        return clusters

    def _cluster_to_primitive(self, cluster: SkillCluster) -> DiscoveredPrimitive:
        """Convert a cluster to a discovered primitive."""
        # Generate name from cluster characteristics
        durations = [s.end_time - s.start_time for s in cluster.segments]
        avg_duration = np.mean(durations)

        # Simple naming based on duration
        if avg_duration < 1.0:
            name_prefix = "quick"
        elif avg_duration < 3.0:
            name_prefix = "standard"
        else:
            name_prefix = "extended"

        primitive_id = f"primitive_{cluster.cluster_id}"
        name = f"{name_prefix}_skill_{cluster.cluster_id}"

        return DiscoveredPrimitive(
            primitive_id=primitive_id,
            name=name,
            exemplar_segments=cluster.segments[:5],  # Keep top 5 exemplars
            centroid_features=cluster.centroid,
            occurrence_count=len(cluster.segments),
            avg_duration=float(avg_duration),
            std_duration=float(np.std(durations)),
        )

    @classmethod
    def from_meta_ai(cls, device: str = "cuda") -> "SkillDiscovery":
        """Create with Meta AI models loaded."""
        config = DiscoveryConfig(device=device)
        discovery = cls(config)
        discovery.segmenter.load_models()
        return discovery

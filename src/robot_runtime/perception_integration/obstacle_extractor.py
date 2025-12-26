"""
Obstacle Extractor - Convert Perception to CBF-Compatible Obstacles

Converts SAM3 segmentation masks and depth maps to 3D obstacle
positions and radii for use in CBF collision barriers.

Methods:
    1. SAM3 + Depth: Use segmentation mask to isolate depth region
    2. Depth Only: Cluster nearby depth points as obstacles
    3. Human Detection: Special handling for detected humans (larger radius)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractorConfig:
    """Configuration for obstacle extraction."""
    # Depth processing
    min_depth_m: float = 0.1  # Ignore closer (noise)
    max_depth_m: float = 5.0  # Ignore farther (out of range)
    depth_scale: float = 1000.0  # Convert depth units to meters

    # Obstacle clustering
    cluster_distance_m: float = 0.1  # Points closer than this = same obstacle
    min_obstacle_points: int = 100  # Minimum points to count as obstacle

    # Safety margins
    human_safety_radius_m: float = 0.5  # Extra margin for humans
    object_safety_radius_m: float = 0.1  # Extra margin for objects
    unknown_safety_radius_m: float = 0.3  # Extra margin for unknown

    # Robot workspace
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([-2, -2, 0]))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([2, 2, 2]))

    # Maximum obstacles to track
    max_obstacles: int = 20


@dataclass
class ObstacleData:
    """Extracted obstacle data for CBF."""
    # Core data (required by CBF)
    positions: np.ndarray  # [N, 3] obstacle centers in world frame
    radii: np.ndarray  # [N] obstacle radii

    # Metadata
    labels: List[str]  # ["human", "object", "unknown", ...]
    confidences: np.ndarray  # [N] detection confidence
    velocities: Optional[np.ndarray] = None  # [N, 3] estimated velocities

    # Statistics
    num_obstacles: int = 0
    extraction_time_ms: float = 0.0

    @property
    def has_humans(self) -> bool:
        return "human" in self.labels

    @property
    def min_distance(self) -> float:
        """Minimum distance to any obstacle (for monitoring)."""
        if self.num_obstacles == 0:
            return float('inf')
        # Distance from origin (robot base) - simplified
        return float(np.min(np.linalg.norm(self.positions, axis=1) - self.radii))


class ObstacleExtractor:
    """
    Extract obstacles from SAM3 segmentation and depth maps.

    Converts perception data to CBF-compatible obstacle representation.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()

        # Camera intrinsics (set via set_camera_intrinsics)
        self.K: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

        # Transform from camera to world frame
        self.T_world_camera: np.ndarray = np.eye(4)

        # Previous obstacles for velocity estimation
        self._prev_obstacles: Optional[ObstacleData] = None
        self._prev_time: float = 0.0

        # Statistics
        self.stats = {
            "extractions": 0,
            "obstacles_detected": 0,
            "humans_detected": 0,
        }

    def set_camera_intrinsics(self, K: np.ndarray):
        """Set camera intrinsics matrix [3x3]."""
        self.K = K.astype(np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def set_camera_transform(self, T_world_camera: np.ndarray):
        """Set transform from camera to world frame [4x4]."""
        self.T_world_camera = T_world_camera.astype(np.float64)

    def extract(
        self,
        depth_map: np.ndarray,
        sam3_result: Optional[Any] = None,
        human_detections: Optional[List[Dict]] = None,
    ) -> ObstacleData:
        """
        Extract obstacles from depth and segmentation.

        Args:
            depth_map: Depth image [H, W] in depth_scale units
            sam3_result: Optional SAM3 segmentation result
            human_detections: Optional list of human detections

        Returns:
            ObstacleData with positions and radii for CBF
        """
        import time
        start_time = time.time()

        self.stats["extractions"] += 1

        positions = []
        radii = []
        labels = []
        confidences = []

        # 1. Process human detections (highest priority)
        if human_detections:
            human_obs = self._extract_human_obstacles(
                depth_map, human_detections
            )
            positions.extend(human_obs["positions"])
            radii.extend(human_obs["radii"])
            labels.extend(human_obs["labels"])
            confidences.extend(human_obs["confidences"])
            self.stats["humans_detected"] += len(human_obs["positions"])

        # 2. Process SAM3 segmentation masks (for objects)
        if sam3_result is not None:
            sam3_obs = self._extract_sam3_obstacles(depth_map, sam3_result)
            positions.extend(sam3_obs["positions"])
            radii.extend(sam3_obs["radii"])
            labels.extend(sam3_obs["labels"])
            confidences.extend(sam3_obs["confidences"])

        # 3. Process raw depth for unknown obstacles
        depth_obs = self._extract_depth_obstacles(
            depth_map,
            exclude_masks=self._get_combined_mask(sam3_result, human_detections),
        )
        positions.extend(depth_obs["positions"])
        radii.extend(depth_obs["radii"])
        labels.extend(depth_obs["labels"])
        confidences.extend(depth_obs["confidences"])

        # Limit to max obstacles (prioritize humans)
        if len(positions) > self.config.max_obstacles:
            # Sort by priority: humans first, then by distance
            indices = self._prioritize_obstacles(
                positions, labels, confidences
            )[:self.config.max_obstacles]
            positions = [positions[i] for i in indices]
            radii = [radii[i] for i in indices]
            labels = [labels[i] for i in indices]
            confidences = [confidences[i] for i in indices]

        # Convert to numpy arrays
        if positions:
            positions_arr = np.array(positions)
            radii_arr = np.array(radii)
            confidences_arr = np.array(confidences)
        else:
            positions_arr = np.zeros((0, 3))
            radii_arr = np.zeros(0)
            confidences_arr = np.zeros(0)

        self.stats["obstacles_detected"] += len(positions)

        # Estimate velocities from previous frame
        velocities = self._estimate_velocities(positions_arr, labels)

        # Store for next frame
        result = ObstacleData(
            positions=positions_arr,
            radii=radii_arr,
            labels=labels,
            confidences=confidences_arr,
            velocities=velocities,
            num_obstacles=len(positions),
            extraction_time_ms=(time.time() - start_time) * 1000,
        )

        self._prev_obstacles = result
        self._prev_time = time.time()

        return result

    def _extract_human_obstacles(
        self,
        depth_map: np.ndarray,
        human_detections: List[Dict],
    ) -> Dict:
        """Extract obstacles from human detections."""
        positions = []
        radii = []
        labels = []
        confidences = []

        for det in human_detections:
            # Get bounding box
            bbox = det.get("bbox", det.get("bounding_box"))
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Get depth in bounding box
            depth_roi = depth_map[y1:y2, x1:x2]
            valid_depth = depth_roi[depth_roi > 0]

            if len(valid_depth) < self.config.min_obstacle_points:
                continue

            # Use median depth (robust to noise)
            depth_m = np.median(valid_depth) / self.config.depth_scale

            if depth_m < self.config.min_depth_m or depth_m > self.config.max_depth_m:
                continue

            # Compute 3D position
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            position = self._pixel_to_world(cx, cy, depth_m)

            if position is None or not self._in_workspace(position):
                continue

            # Human radius: based on bbox size + safety margin
            bbox_width_m = (x2 - x1) * depth_m / self.K[0, 0] if self.K is not None else 0.5
            radius = bbox_width_m / 2 + self.config.human_safety_radius_m

            positions.append(position)
            radii.append(radius)
            labels.append("human")
            confidences.append(det.get("confidence", 0.8))

        return {
            "positions": positions,
            "radii": radii,
            "labels": labels,
            "confidences": confidences,
        }

    def _extract_sam3_obstacles(
        self,
        depth_map: np.ndarray,
        sam3_result: Any,
    ) -> Dict:
        """Extract obstacles from SAM3 segmentation masks."""
        positions = []
        radii = []
        labels = []
        confidences = []

        masks = getattr(sam3_result, 'masks', [])

        for mask_data in masks:
            # Get mask
            if hasattr(mask_data, 'mask'):
                mask = mask_data.mask
            elif isinstance(mask_data, np.ndarray):
                mask = mask_data
            else:
                continue

            if mask.sum() < self.config.min_obstacle_points:
                continue

            # Get depth values in mask
            depth_values = depth_map[mask > 0]
            valid_depth = depth_values[depth_values > 0]

            if len(valid_depth) < self.config.min_obstacle_points:
                continue

            depth_m = np.median(valid_depth) / self.config.depth_scale

            if depth_m < self.config.min_depth_m or depth_m > self.config.max_depth_m:
                continue

            # Compute centroid
            ys, xs = np.where(mask > 0)
            cx, cy = np.mean(xs), np.mean(ys)

            position = self._pixel_to_world(cx, cy, depth_m)

            if position is None or not self._in_workspace(position):
                continue

            # Compute radius from mask extent
            mask_width = np.max(xs) - np.min(xs)
            mask_height = np.max(ys) - np.min(ys)
            mask_radius_pixels = max(mask_width, mask_height) / 2
            mask_radius_m = mask_radius_pixels * depth_m / self.K[0, 0] if self.K is not None else 0.2

            radius = mask_radius_m + self.config.object_safety_radius_m

            positions.append(position)
            radii.append(radius)
            labels.append("object")

            conf = getattr(mask_data, 'confidence', 0.7)
            confidences.append(conf)

        return {
            "positions": positions,
            "radii": radii,
            "labels": labels,
            "confidences": confidences,
        }

    def _extract_depth_obstacles(
        self,
        depth_map: np.ndarray,
        exclude_masks: Optional[np.ndarray] = None,
    ) -> Dict:
        """Extract obstacles from raw depth (for areas not covered by SAM3)."""
        positions = []
        radii = []
        labels = []
        confidences = []

        # Convert depth to meters
        depth_m = depth_map.astype(np.float32) / self.config.depth_scale

        # Mask out excluded regions and invalid depths
        valid_mask = (
            (depth_m > self.config.min_depth_m) &
            (depth_m < self.config.max_depth_m)
        )

        if exclude_masks is not None:
            valid_mask = valid_mask & (~exclude_masks)

        # Find close obstacles (potential collision risk)
        close_mask = valid_mask & (depth_m < 1.0)  # Within 1m

        if close_mask.sum() < self.config.min_obstacle_points:
            return {"positions": [], "radii": [], "labels": [], "confidences": []}

        # Simple clustering: find connected components
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(close_mask)
        except ImportError:
            # Fallback: treat all close points as one obstacle
            labeled = close_mask.astype(np.int32)
            num_features = 1

        for i in range(1, min(num_features + 1, 5)):  # Max 5 depth obstacles
            component_mask = labeled == i

            if component_mask.sum() < self.config.min_obstacle_points:
                continue

            # Get component depth
            component_depth = depth_m[component_mask]
            median_depth = np.median(component_depth)

            # Get centroid
            ys, xs = np.where(component_mask)
            cx, cy = np.mean(xs), np.mean(ys)

            position = self._pixel_to_world(cx, cy, median_depth)

            if position is None or not self._in_workspace(position):
                continue

            # Estimate radius
            component_width = np.max(xs) - np.min(xs)
            component_height = np.max(ys) - np.min(ys)
            radius_pixels = max(component_width, component_height) / 2
            radius_m = radius_pixels * median_depth / self.K[0, 0] if self.K is not None else 0.2

            radius = radius_m + self.config.unknown_safety_radius_m

            positions.append(position)
            radii.append(radius)
            labels.append("unknown")
            confidences.append(0.5)  # Lower confidence for depth-only

        return {
            "positions": positions,
            "radii": radii,
            "labels": labels,
            "confidences": confidences,
        }

    def _pixel_to_world(
        self,
        u: float,
        v: float,
        depth: float,
    ) -> Optional[np.ndarray]:
        """Convert pixel coordinates + depth to world frame."""
        if self.K_inv is None:
            # Default: assume simple projection
            # This is approximate if intrinsics not set
            fx, fy = 500.0, 500.0  # Approximate focal length
            cx, cy = 320.0, 240.0  # Approximate principal point

            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth

            point_camera = np.array([x, y, z])
        else:
            # Use actual intrinsics
            pixel_h = np.array([u, v, 1.0])
            ray = self.K_inv @ pixel_h
            point_camera = ray * depth

        # Transform to world frame
        point_h = np.array([*point_camera, 1.0])
        point_world = self.T_world_camera @ point_h

        return point_world[:3]

    def _in_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within robot workspace."""
        return (
            np.all(position >= self.config.workspace_min) and
            np.all(position <= self.config.workspace_max)
        )

    def _get_combined_mask(
        self,
        sam3_result: Optional[Any],
        human_detections: Optional[List[Dict]],
    ) -> Optional[np.ndarray]:
        """Get combined mask of all detected regions."""
        # Would combine SAM3 masks and human detection boxes
        # For now, return None (process all depth)
        return None

    def _prioritize_obstacles(
        self,
        positions: List[np.ndarray],
        labels: List[str],
        confidences: List[float],
    ) -> List[int]:
        """Prioritize obstacles: humans first, then by distance."""
        if not positions:
            return []

        # Compute priority scores
        scores = []
        for i, (pos, label, conf) in enumerate(zip(positions, labels, confidences)):
            distance = np.linalg.norm(pos)

            if label == "human":
                priority = 1000.0  # Highest priority
            elif label == "object":
                priority = 100.0
            else:
                priority = 10.0

            # Closer obstacles get higher priority
            score = priority - distance + conf
            scores.append((score, i))

        # Sort by score (descending)
        scores.sort(reverse=True)
        return [idx for _, idx in scores]

    def _estimate_velocities(
        self,
        positions: np.ndarray,
        labels: List[str],
    ) -> Optional[np.ndarray]:
        """Estimate obstacle velocities from previous frame."""
        if self._prev_obstacles is None or len(positions) == 0:
            return None

        dt = time.time() - self._prev_time if self._prev_time > 0 else 0.1

        if dt <= 0 or len(self._prev_obstacles.positions) == 0:
            return None

        # Simple nearest-neighbor velocity estimation
        velocities = np.zeros_like(positions)

        for i, (pos, label) in enumerate(zip(positions, labels)):
            if label != "human":
                continue  # Only track human velocities

            # Find closest previous human
            prev_humans = [
                j for j, l in enumerate(self._prev_obstacles.labels)
                if l == "human"
            ]

            if not prev_humans:
                continue

            prev_positions = self._prev_obstacles.positions[prev_humans]
            distances = np.linalg.norm(prev_positions - pos, axis=1)
            closest_idx = prev_humans[np.argmin(distances)]

            if distances.min() < 0.5:  # Reasonable tracking threshold
                prev_pos = self._prev_obstacles.positions[closest_idx]
                velocities[i] = (pos - prev_pos) / dt

        return velocities

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return dict(self.stats)


# Import time for velocity estimation
import time

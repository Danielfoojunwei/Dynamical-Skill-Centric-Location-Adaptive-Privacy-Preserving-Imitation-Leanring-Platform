"""
User Steerability Interface - Human-in-the-Loop Control

Enables USER STEERABILITY by allowing operators to:
1. Sketch waypoints on a tablet/phone
2. Define avoid regions
3. Adjust path preferences
4. Override model predictions

This addresses MolmoAct Gap #2: "Dynamical has no intervention mechanism"

Usage:
    guidance = UserGuidance(
        forced_waypoints=[(100, 150), (200, 200)],  # User-drawn points
        avoid_regions=[polygon1, polygon2],
        path_bias="left",
    )
    modified_trace = guidance.apply_to_trace(predicted_trace)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Local imports
from .trajectory_trace import TrajectoryTrace, WaypointType

# Optional imports for visualization
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class GuidanceMode(str, Enum):
    """How user guidance is applied."""
    REPLACE = "replace"      # Replace predicted waypoints with user-provided
    BLEND = "blend"          # Blend user and model waypoints
    CONSTRAINT = "constraint"  # Use user waypoints as constraints
    AVOID_ONLY = "avoid_only"  # Only apply avoid regions


class PathBias(str, Enum):
    """Path direction preferences."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    SHORTER = "shorter"
    LONGER = "longer"


@dataclass
class AvoidRegion:
    """
    Region to avoid during trajectory planning.

    Can be specified as:
    - Polygon vertices
    - Circle (center + radius)
    - Bounding box
    """
    region_id: str

    # Polygon vertices [N, 2] in image coords [0, 256)
    polygon: Optional[np.ndarray] = None

    # Circle definition
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None

    # Bounding box [x_min, y_min, x_max, y_max]
    bbox: Optional[Tuple[float, float, float, float]] = None

    # Priority (higher = more important to avoid)
    priority: float = 1.0

    # Margin around region (extra buffer)
    margin: float = 5.0

    def __post_init__(self):
        """Validate region definition."""
        num_defined = sum([
            self.polygon is not None,
            self.center is not None and self.radius is not None,
            self.bbox is not None,
        ])
        if num_defined == 0:
            raise ValueError("Must define polygon, circle, or bbox")
        if num_defined > 1:
            logger.warning("Multiple region types defined, using first available")

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside the region."""
        if self.polygon is not None:
            return self._point_in_polygon(point, self.polygon, self.margin)
        elif self.center is not None and self.radius is not None:
            dist = np.linalg.norm(point - np.array(self.center))
            return dist < (self.radius + self.margin)
        elif self.bbox is not None:
            x_min, y_min, x_max, y_max = self.bbox
            return (
                x_min - self.margin <= point[0] <= x_max + self.margin and
                y_min - self.margin <= point[1] <= y_max + self.margin
            )
        return False

    def nearest_edge_point(self, point: np.ndarray) -> np.ndarray:
        """Find nearest point on region boundary (for rerouting)."""
        if self.polygon is not None:
            return self._nearest_polygon_edge(point, self.polygon)
        elif self.center is not None and self.radius is not None:
            direction = point - np.array(self.center)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            return np.array(self.center) + direction * (self.radius + self.margin)
        elif self.bbox is not None:
            x_min, y_min, x_max, y_max = self.bbox
            x = np.clip(point[0], x_min - self.margin, x_max + self.margin)
            y = np.clip(point[1], y_min - self.margin, y_max + self.margin)
            return np.array([x, y])
        return point

    @staticmethod
    def _point_in_polygon(
        point: np.ndarray,
        polygon: np.ndarray,
        margin: float = 0,
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            if ((polygon[i, 1] > point[1]) != (polygon[j, 1] > point[1])) and \
               (point[0] < (polygon[j, 0] - polygon[i, 0]) *
                (point[1] - polygon[i, 1]) / (polygon[j, 1] - polygon[i, 1]) +
                polygon[i, 0]):
                inside = not inside
            j = i

        return inside

    @staticmethod
    def _nearest_polygon_edge(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Find nearest point on polygon boundary."""
        min_dist = float('inf')
        nearest = point

        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Project point onto edge
            edge = p2 - p1
            t = np.clip(np.dot(point - p1, edge) / (np.dot(edge, edge) + 1e-6), 0, 1)
            projection = p1 + t * edge

            dist = np.linalg.norm(point - projection)
            if dist < min_dist:
                min_dist = dist
                nearest = projection

        return nearest

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "region_id": self.region_id,
            "polygon": self.polygon.tolist() if self.polygon is not None else None,
            "center": self.center,
            "radius": self.radius,
            "bbox": self.bbox,
            "priority": self.priority,
            "margin": self.margin,
        }


@dataclass
class UserGuidance:
    """
    User-provided guidance for trajectory planning.

    Enables human operators to steer robot behavior by:
    - Specifying waypoints the robot must pass through
    - Defining regions to avoid
    - Setting path preferences
    - Adjusting speed
    """
    # User-specified waypoints [N, 2] in image coords [0, 256)
    forced_waypoints: Optional[List[Tuple[float, float]]] = None

    # Regions to avoid
    avoid_regions: List[AvoidRegion] = field(default_factory=list)

    # Path preference
    path_bias: PathBias = PathBias.NONE

    # Speed modifier (0.5 = half speed, 2.0 = double)
    speed_modifier: float = 1.0

    # Precision requirement (affects waypoint density)
    require_precision: bool = False

    # How to apply guidance
    mode: GuidanceMode = GuidanceMode.BLEND

    # Blend weight (for BLEND mode)
    user_weight: float = 0.7  # Higher = more weight on user input

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # User ID for logging
    user_id: Optional[str] = None

    def __post_init__(self):
        """Validate guidance."""
        if self.forced_waypoints is not None:
            self.forced_waypoints = [
                tuple(wp) for wp in self.forced_waypoints
            ]

    def apply_to_trace(
        self,
        trace: TrajectoryTrace,
    ) -> TrajectoryTrace:
        """
        Apply user guidance to a predicted trajectory.

        Args:
            trace: Model-predicted trajectory

        Returns:
            Modified trajectory with user guidance applied
        """
        if self.mode == GuidanceMode.AVOID_ONLY:
            return self._apply_avoid_only(trace)
        elif self.mode == GuidanceMode.REPLACE:
            return self._apply_replace(trace)
        elif self.mode == GuidanceMode.BLEND:
            return self._apply_blend(trace)
        elif self.mode == GuidanceMode.CONSTRAINT:
            return self._apply_constraint(trace)
        else:
            return trace

    def _apply_avoid_only(self, trace: TrajectoryTrace) -> TrajectoryTrace:
        """Only reroute around avoid regions."""
        if not self.avoid_regions:
            return trace

        new_waypoints = trace.waypoints.copy()
        new_types = trace.waypoint_types.copy()

        for i, wp in enumerate(new_waypoints):
            for region in sorted(self.avoid_regions, key=lambda r: -r.priority):
                if region.contains_point(wp):
                    # Move to nearest edge
                    new_waypoints[i] = region.nearest_edge_point(wp)
                    break

        return TrajectoryTrace(
            waypoints=new_waypoints,
            confidences=trace.confidences,
            waypoint_types=new_types,
            waypoint_depths=trace.waypoint_depths,
            source_image=trace.source_image,
            instruction=trace.instruction,
            perception_reasoning=trace.perception_reasoning,
            spatial_reasoning=f"[USER GUIDANCE: Avoiding {len(self.avoid_regions)} regions]",
            action_reasoning=trace.action_reasoning,
        )

    def _apply_replace(self, trace: TrajectoryTrace) -> TrajectoryTrace:
        """Replace predicted waypoints with user-provided ones."""
        if not self.forced_waypoints:
            return trace

        # Use only forced waypoints
        forced = np.array(self.forced_waypoints, dtype=np.float32)
        n_forced = len(forced)

        # Create new trace
        waypoints = forced
        confidences = np.ones(n_forced, dtype=np.float32)
        waypoint_types = [WaypointType.FORCED] * n_forced

        # Apply bias
        if self.path_bias != PathBias.NONE:
            waypoints = self._apply_path_bias(waypoints)

        # Avoid regions
        for i, wp in enumerate(waypoints):
            for region in self.avoid_regions:
                if region.contains_point(wp):
                    waypoints[i] = region.nearest_edge_point(wp)

        return TrajectoryTrace(
            waypoints=waypoints,
            confidences=confidences,
            waypoint_types=waypoint_types,
            source_image=trace.source_image,
            instruction=trace.instruction,
            spatial_reasoning=f"[USER OVERRIDE: {n_forced} forced waypoints]",
        )

    def _apply_blend(self, trace: TrajectoryTrace) -> TrajectoryTrace:
        """Blend user waypoints with model predictions."""
        if not self.forced_waypoints:
            return self._apply_avoid_only(trace)

        forced = np.array(self.forced_waypoints, dtype=np.float32)
        model_wp = trace.waypoints

        # Align waypoints
        n_output = max(len(forced), len(model_wp))

        # Interpolate to same length
        t_forced = np.linspace(0, 1, len(forced))
        t_model = np.linspace(0, 1, len(model_wp))
        t_output = np.linspace(0, 1, n_output)

        forced_interp = np.zeros((n_output, 2))
        model_interp = np.zeros((n_output, 2))

        for dim in range(2):
            forced_interp[:, dim] = np.interp(t_output, t_forced, forced[:, dim])
            model_interp[:, dim] = np.interp(t_output, t_model, model_wp[:, dim])

        # Blend
        blended = self.user_weight * forced_interp + (1 - self.user_weight) * model_interp

        # Confidence based on agreement
        agreement = 1 - np.linalg.norm(forced_interp - model_interp, axis=1) / 256.0
        confidences = np.clip(agreement, 0, 1)

        # Mark forced positions
        waypoint_types = []
        for i in range(n_output):
            # If this position is close to a forced waypoint
            is_forced = any(
                np.linalg.norm(blended[i] - np.array(fw)) < 10
                for fw in self.forced_waypoints
            )
            if is_forced:
                waypoint_types.append(WaypointType.FORCED)
            else:
                waypoint_types.append(WaypointType.PREDICTED)

        # Apply avoid regions
        for i, wp in enumerate(blended):
            for region in self.avoid_regions:
                if region.contains_point(wp):
                    blended[i] = region.nearest_edge_point(wp)

        return TrajectoryTrace(
            waypoints=blended,
            confidences=confidences,
            waypoint_types=waypoint_types,
            waypoint_depths=None,  # Depths need recomputation
            source_image=trace.source_image,
            instruction=trace.instruction,
            spatial_reasoning=f"[BLENDED: {self.user_weight:.0%} user, {1-self.user_weight:.0%} model]",
        )

    def _apply_constraint(self, trace: TrajectoryTrace) -> TrajectoryTrace:
        """Use forced waypoints as hard constraints."""
        if not self.forced_waypoints:
            return self._apply_avoid_only(trace)

        forced = np.array(self.forced_waypoints, dtype=np.float32)
        model_wp = trace.waypoints.copy()

        # Insert forced waypoints at appropriate positions
        all_waypoints = []
        all_types = []

        model_idx = 0
        for forced_wp in forced:
            # Add model waypoints until we're past this forced one
            while model_idx < len(model_wp):
                mw = model_wp[model_idx]
                # Check if forced point should come before this model point
                if np.linalg.norm(forced_wp - mw) < 20:
                    # Close to model point - replace with forced
                    all_waypoints.append(forced_wp)
                    all_types.append(WaypointType.FORCED)
                    model_idx += 1
                    break
                else:
                    all_waypoints.append(mw)
                    all_types.append(WaypointType.PREDICTED)
                    model_idx += 1

        # Add remaining model waypoints
        while model_idx < len(model_wp):
            all_waypoints.append(model_wp[model_idx])
            all_types.append(WaypointType.PREDICTED)
            model_idx += 1

        waypoints = np.array(all_waypoints, dtype=np.float32)
        confidences = np.array([
            1.0 if t == WaypointType.FORCED else trace.confidences[min(i, len(trace.confidences)-1)]
            for i, t in enumerate(all_types)
        ])

        return TrajectoryTrace(
            waypoints=waypoints,
            confidences=confidences,
            waypoint_types=all_types,
            source_image=trace.source_image,
            instruction=trace.instruction,
            spatial_reasoning=f"[CONSTRAINED: {len(forced)} forced waypoints]",
        )

    def _apply_path_bias(self, waypoints: np.ndarray) -> np.ndarray:
        """Apply path direction bias."""
        if self.path_bias == PathBias.NONE:
            return waypoints

        biased = waypoints.copy()

        if self.path_bias == PathBias.LEFT:
            # Shift x coordinates left
            biased[:, 0] -= 10
        elif self.path_bias == PathBias.RIGHT:
            biased[:, 0] += 10
        elif self.path_bias == PathBias.ABOVE:
            biased[:, 1] -= 10
        elif self.path_bias == PathBias.BELOW:
            biased[:, 1] += 10

        # Clamp to valid range
        biased = np.clip(biased, 0, 255)

        return biased

    def visualize(
        self,
        background: Optional[np.ndarray] = None,
        size: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        """
        Visualize user guidance.

        Args:
            background: Optional background image
            size: Output size

        Returns:
            Visualization image
        """
        if not HAS_CV2:
            return np.zeros((*size[::-1], 3), dtype=np.uint8)

        if background is not None:
            vis = cv2.resize(background.copy(), size)
        else:
            vis = np.ones((*size[::-1], 3), dtype=np.uint8) * 40

        scale_x = size[0] / 256.0
        scale_y = size[1] / 256.0

        # Draw avoid regions
        for region in self.avoid_regions:
            if region.polygon is not None:
                pts = (region.polygon * np.array([scale_x, scale_y])).astype(np.int32)
                cv2.fillPoly(vis, [pts], (50, 50, 150))
                cv2.polylines(vis, [pts], True, (100, 100, 255), 2)
            elif region.center is not None and region.radius is not None:
                center = (int(region.center[0] * scale_x), int(region.center[1] * scale_y))
                radius = int(region.radius * scale_x)
                cv2.circle(vis, center, radius, (50, 50, 150), -1)
                cv2.circle(vis, center, radius, (100, 100, 255), 2)
            elif region.bbox is not None:
                x1, y1, x2, y2 = region.bbox
                pt1 = (int(x1 * scale_x), int(y1 * scale_y))
                pt2 = (int(x2 * scale_x), int(y2 * scale_y))
                cv2.rectangle(vis, pt1, pt2, (50, 50, 150), -1)
                cv2.rectangle(vis, pt1, pt2, (100, 100, 255), 2)

        # Draw forced waypoints
        if self.forced_waypoints:
            for i, wp in enumerate(self.forced_waypoints):
                pt = (int(wp[0] * scale_x), int(wp[1] * scale_y))
                cv2.circle(vis, pt, 8, (0, 165, 255), -1)  # Orange
                cv2.circle(vis, pt, 8, (255, 255, 255), 2)
                cv2.putText(vis, str(i), (pt[0] - 5, pt[1] + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw path between forced waypoints
            for i in range(len(self.forced_waypoints) - 1):
                pt1 = (int(self.forced_waypoints[i][0] * scale_x),
                       int(self.forced_waypoints[i][1] * scale_y))
                pt2 = (int(self.forced_waypoints[i+1][0] * scale_x),
                       int(self.forced_waypoints[i+1][1] * scale_y))
                cv2.line(vis, pt1, pt2, (0, 165, 255), 2, cv2.LINE_AA)

        # Add legend
        cv2.putText(vis, f"Mode: {self.mode.value}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Speed: {self.speed_modifier:.1f}x", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.path_bias != PathBias.NONE:
            cv2.putText(vis, f"Bias: {self.path_bias.value}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "forced_waypoints": self.forced_waypoints,
            "avoid_regions": [r.to_dict() for r in self.avoid_regions],
            "path_bias": self.path_bias.value,
            "speed_modifier": self.speed_modifier,
            "require_precision": self.require_precision,
            "mode": self.mode.value,
            "user_weight": self.user_weight,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserGuidance":
        """Deserialize from dictionary."""
        avoid_regions = []
        for r in data.get("avoid_regions", []):
            avoid_regions.append(AvoidRegion(
                region_id=r["region_id"],
                polygon=np.array(r["polygon"]) if r.get("polygon") else None,
                center=r.get("center"),
                radius=r.get("radius"),
                bbox=r.get("bbox"),
                priority=r.get("priority", 1.0),
                margin=r.get("margin", 5.0),
            ))

        return cls(
            forced_waypoints=data.get("forced_waypoints"),
            avoid_regions=avoid_regions,
            path_bias=PathBias(data.get("path_bias", "none")),
            speed_modifier=data.get("speed_modifier", 1.0),
            require_precision=data.get("require_precision", False),
            mode=GuidanceMode(data.get("mode", "blend")),
            user_weight=data.get("user_weight", 0.7),
            timestamp=data.get("timestamp", time.time()),
            user_id=data.get("user_id"),
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_avoid_circle(
    region_id: str,
    center: Tuple[float, float],
    radius: float,
    margin: float = 5.0,
) -> AvoidRegion:
    """Create a circular avoid region."""
    return AvoidRegion(
        region_id=region_id,
        center=center,
        radius=radius,
        margin=margin,
    )


def create_avoid_box(
    region_id: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    margin: float = 5.0,
) -> AvoidRegion:
    """Create a rectangular avoid region."""
    return AvoidRegion(
        region_id=region_id,
        bbox=(x_min, y_min, x_max, y_max),
        margin=margin,
    )


def create_guidance_from_sketch(
    sketch_points: List[Tuple[int, int]],
    image_size: Tuple[int, int] = (640, 480),
) -> UserGuidance:
    """
    Create guidance from user-sketched points.

    Args:
        sketch_points: List of (x, y) points in image coordinates
        image_size: Size of the source image

    Returns:
        UserGuidance with scaled waypoints
    """
    # Scale to [0, 256)
    scale_x = 256.0 / image_size[0]
    scale_y = 256.0 / image_size[1]

    forced_waypoints = [
        (pt[0] * scale_x, pt[1] * scale_y)
        for pt in sketch_points
    ]

    return UserGuidance(
        forced_waypoints=forced_waypoints,
        mode=GuidanceMode.BLEND,
    )

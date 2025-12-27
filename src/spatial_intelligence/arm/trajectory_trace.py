"""
Trajectory Trace - Image-Space Waypoint Representation

Provides INTERPRETABILITY by visualizing robot intentions:
- Waypoints overlaid on camera images
- Confidence-weighted visualization
- Depth-aware 3D conversion
- Video export for debugging

This addresses MolmoAct Gap #1: "Dynamical's actions are opaque"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class WaypointType(str, Enum):
    """Type of waypoint in trajectory."""
    PREDICTED = "predicted"      # Model-predicted waypoint
    FORCED = "forced"            # User-specified waypoint
    INTERPOLATED = "interpolated"  # Interpolated between key waypoints
    TERMINAL = "terminal"        # Final target waypoint


@dataclass
class TrajectoryVisualizationConfig:
    """Configuration for trajectory visualization."""
    # Output size
    output_width: int = 640
    output_height: int = 480

    # Waypoint appearance
    waypoint_radius_min: int = 4
    waypoint_radius_max: int = 12
    waypoint_color_high_conf: Tuple[int, int, int] = (0, 255, 0)  # Green
    waypoint_color_low_conf: Tuple[int, int, int] = (0, 100, 0)   # Dark green
    waypoint_color_forced: Tuple[int, int, int] = (255, 165, 0)   # Orange
    waypoint_color_terminal: Tuple[int, int, int] = (255, 0, 0)   # Red

    # Path appearance
    path_thickness: int = 2
    path_color: Tuple[int, int, int] = (0, 200, 0)
    show_path_gradient: bool = True

    # Labels
    show_waypoint_numbers: bool = True
    show_confidence_values: bool = False
    show_depth_values: bool = False
    font_scale: float = 0.4
    font_color: Tuple[int, int, int] = (255, 255, 255)

    # Depth visualization
    show_depth_heatmap: bool = False
    depth_colormap: int = 2  # cv2.COLORMAP_JET if HAS_CV2 else 2

    # Animation
    animation_fps: int = 10
    trail_length: int = 5  # Number of previous waypoints to show as trail


@dataclass
class TrajectoryTrace:
    """
    Image-space trajectory representation for interpretable robot planning.

    Provides visualization of robot intentions by representing planned motion
    as a sequence of 2D waypoints in image coordinates [0, 256).

    This is the core data structure for MolmoAct-style interpretability.

    Attributes:
        waypoints: Image coordinates [N, 2] with values in [0, 256)
        confidences: Per-waypoint confidence [N] in [0, 1]
        waypoint_types: Type of each waypoint (predicted, forced, terminal)
        waypoint_depths: Depth at each waypoint in meters [N]
        source_image: Original camera image for overlay
        instruction: Natural language task description
        timestamp: When this trace was generated
    """

    # Core trajectory data
    waypoints: np.ndarray  # [N, 2] in image coords [0, 256)
    confidences: np.ndarray  # [N] confidence per waypoint

    # Waypoint metadata
    waypoint_types: Optional[List[WaypointType]] = None
    waypoint_depths: Optional[np.ndarray] = None  # [N] depth in meters

    # Context
    source_image: Optional[np.ndarray] = None  # [H, W, 3] BGR
    instruction: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    # Reasoning trace (for chain-of-thought)
    perception_reasoning: Optional[str] = None
    spatial_reasoning: Optional[str] = None
    action_reasoning: Optional[str] = None

    # Metadata
    model_name: str = "trajectory_predictor"
    inference_time_ms: float = 0.0

    def __post_init__(self):
        """Validate and initialize trajectory trace."""
        # Ensure numpy arrays
        self.waypoints = np.asarray(self.waypoints, dtype=np.float32)
        self.confidences = np.asarray(self.confidences, dtype=np.float32)

        # Validate shapes
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2:
            raise ValueError(f"waypoints must be [N, 2], got {self.waypoints.shape}")

        if len(self.confidences) != len(self.waypoints):
            raise ValueError(
                f"confidences length {len(self.confidences)} != "
                f"waypoints length {len(self.waypoints)}"
            )

        # Initialize waypoint types if not provided
        if self.waypoint_types is None:
            self.waypoint_types = [WaypointType.PREDICTED] * len(self.waypoints)
            if len(self.waypoints) > 0:
                self.waypoint_types[-1] = WaypointType.TERMINAL

        # Validate depths if provided
        if self.waypoint_depths is not None:
            self.waypoint_depths = np.asarray(self.waypoint_depths, dtype=np.float32)
            if len(self.waypoint_depths) != len(self.waypoints):
                raise ValueError("waypoint_depths length must match waypoints")

    @property
    def num_waypoints(self) -> int:
        """Number of waypoints in trajectory."""
        return len(self.waypoints)

    @property
    def mean_confidence(self) -> float:
        """Average confidence across all waypoints."""
        return float(self.confidences.mean()) if len(self.confidences) > 0 else 0.0

    @property
    def trajectory_length_pixels(self) -> float:
        """Total path length in image pixels."""
        if len(self.waypoints) < 2:
            return 0.0
        diffs = np.diff(self.waypoints, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def has_forced_waypoints(self) -> bool:
        """Check if trajectory contains user-forced waypoints."""
        return any(wt == WaypointType.FORCED for wt in self.waypoint_types)

    # =========================================================================
    # Visualization Methods (INTERPRETABILITY)
    # =========================================================================

    def visualize(
        self,
        config: Optional[TrajectoryVisualizationConfig] = None,
        highlight_waypoint: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create visualization of trajectory overlaid on source image.

        Args:
            config: Visualization configuration
            highlight_waypoint: Index of waypoint to highlight

        Returns:
            BGR image [H, W, 3] with trajectory overlay
        """
        if not HAS_CV2:
            logger.warning("OpenCV not available, returning blank visualization")
            config = config or TrajectoryVisualizationConfig()
            return np.ones(
                (config.output_height, config.output_width, 3),
                dtype=np.uint8
            ) * 128

        config = config or TrajectoryVisualizationConfig()

        # Start with source image or blank canvas
        if self.source_image is not None:
            vis = cv2.resize(
                self.source_image.copy(),
                (config.output_width, config.output_height)
            )
        else:
            vis = np.ones(
                (config.output_height, config.output_width, 3),
                dtype=np.uint8
            ) * 40  # Dark gray background

        # Scale waypoints from [0, 256) to output size
        scale_x = config.output_width / 256.0
        scale_y = config.output_height / 256.0
        scaled_wp = self.waypoints.copy()
        scaled_wp[:, 0] *= scale_x
        scaled_wp[:, 1] *= scale_y

        # Draw trajectory path
        if len(scaled_wp) > 1:
            for i in range(len(scaled_wp) - 1):
                pt1 = tuple(scaled_wp[i].astype(int))
                pt2 = tuple(scaled_wp[i + 1].astype(int))

                if config.show_path_gradient:
                    # Color gradient based on confidence
                    conf = (self.confidences[i] + self.confidences[i + 1]) / 2
                    color = self._interpolate_color(
                        config.path_color,
                        (50, 50, 50),
                        1 - conf
                    )
                else:
                    color = config.path_color

                cv2.line(vis, pt1, pt2, color, config.path_thickness)

        # Draw waypoints
        for i, (wp, conf, wp_type) in enumerate(
            zip(scaled_wp, self.confidences, self.waypoint_types)
        ):
            pt = tuple(wp.astype(int))

            # Determine color based on type
            if wp_type == WaypointType.FORCED:
                color = config.waypoint_color_forced
            elif wp_type == WaypointType.TERMINAL:
                color = config.waypoint_color_terminal
            else:
                color = self._interpolate_color(
                    config.waypoint_color_high_conf,
                    config.waypoint_color_low_conf,
                    1 - conf
                )

            # Determine radius based on confidence
            radius = int(
                config.waypoint_radius_min +
                conf * (config.waypoint_radius_max - config.waypoint_radius_min)
            )

            # Highlight if specified
            if highlight_waypoint is not None and i == highlight_waypoint:
                cv2.circle(vis, pt, radius + 4, (255, 255, 0), 2)

            # Draw waypoint
            cv2.circle(vis, pt, radius, color, -1)
            cv2.circle(vis, pt, radius, (255, 255, 255), 1)  # White border

            # Draw labels
            if config.show_waypoint_numbers:
                label = str(i)
                cv2.putText(
                    vis, label, (pt[0] - 5, pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, config.font_scale,
                    config.font_color, 1
                )

            if config.show_confidence_values:
                label = f"{conf:.2f}"
                cv2.putText(
                    vis, label, (pt[0] + radius + 2, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, config.font_scale * 0.8,
                    config.font_color, 1
                )

            if config.show_depth_values and self.waypoint_depths is not None:
                depth = self.waypoint_depths[i]
                label = f"{depth:.2f}m"
                cv2.putText(
                    vis, label, (pt[0] + radius + 2, pt[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, config.font_scale * 0.8,
                    (200, 200, 255), 1
                )

        # Add instruction text
        if self.instruction:
            cv2.putText(
                vis, f"Task: {self.instruction[:50]}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )

        # Add confidence indicator
        cv2.putText(
            vis, f"Confidence: {self.mean_confidence:.2f}",
            (10, config.output_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

        return vis

    def visualize_with_reasoning(
        self,
        config: Optional[TrajectoryVisualizationConfig] = None,
    ) -> np.ndarray:
        """
        Create visualization with chain-of-thought reasoning panel.

        Returns:
            BGR image with trajectory and reasoning text
        """
        config = config or TrajectoryVisualizationConfig()

        # Get base visualization
        vis = self.visualize(config)

        # Create reasoning panel
        panel_height = 100
        panel = np.ones((panel_height, config.output_width, 3), dtype=np.uint8) * 30

        if HAS_CV2:
            y_offset = 15
            line_height = 18

            if self.perception_reasoning:
                text = f"Perception: {self.perception_reasoning[:70]}"
                cv2.putText(panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
                y_offset += line_height

            if self.spatial_reasoning:
                text = f"Spatial: {self.spatial_reasoning[:70]}"
                cv2.putText(panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
                y_offset += line_height

            if self.action_reasoning:
                text = f"Action: {self.action_reasoning[:70]}"
                cv2.putText(panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)

        # Combine visualization and panel
        combined = np.vstack([vis, panel])
        return combined

    def create_animation_frames(
        self,
        config: Optional[TrajectoryVisualizationConfig] = None,
        num_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Create animation frames showing trajectory execution over time.

        Args:
            config: Visualization configuration
            num_frames: Number of frames (default: num_waypoints * 3)

        Returns:
            List of BGR images for animation
        """
        config = config or TrajectoryVisualizationConfig()
        num_frames = num_frames or (len(self.waypoints) * 3)

        frames = []
        for frame_idx in range(num_frames):
            # Calculate which waypoints to show
            progress = frame_idx / max(num_frames - 1, 1)
            current_wp_idx = int(progress * (len(self.waypoints) - 1))

            # Create partial trace up to current waypoint
            partial_trace = TrajectoryTrace(
                waypoints=self.waypoints[:current_wp_idx + 1],
                confidences=self.confidences[:current_wp_idx + 1],
                waypoint_types=self.waypoint_types[:current_wp_idx + 1],
                waypoint_depths=(
                    self.waypoint_depths[:current_wp_idx + 1]
                    if self.waypoint_depths is not None else None
                ),
                source_image=self.source_image,
                instruction=self.instruction,
            )

            frame = partial_trace.visualize(config, highlight_waypoint=current_wp_idx)
            frames.append(frame)

        return frames

    def save_video(
        self,
        output_path: str,
        config: Optional[TrajectoryVisualizationConfig] = None,
    ) -> bool:
        """
        Save trajectory animation as video file.

        Args:
            output_path: Path to output video file
            config: Visualization configuration

        Returns:
            True if successful
        """
        if not HAS_CV2:
            logger.error("OpenCV required for video export")
            return False

        config = config or TrajectoryVisualizationConfig()
        frames = self.create_animation_frames(config)

        if not frames:
            logger.error("No frames to save")
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, config.animation_fps,
            (config.output_width, config.output_height)
        )

        for frame in frames:
            writer.write(frame)

        writer.release()
        logger.info(f"Saved trajectory video to {output_path}")
        return True

    # =========================================================================
    # 3D Conversion Methods
    # =========================================================================

    def to_3d_points(
        self,
        camera_intrinsics: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        """
        Convert image-space waypoints to 3D camera-frame coordinates.

        Args:
            camera_intrinsics: Camera matrix [3, 3]
            depth_map: Optional depth image [H, W] in meters
            image_size: Size of the camera image (W, H)

        Returns:
            3D points [N, 3] in camera frame
        """
        # Get depths
        if self.waypoint_depths is not None:
            depths = self.waypoint_depths
        elif depth_map is not None:
            # Sample depth map at waypoint locations
            depths = []
            scale_x = image_size[0] / 256.0
            scale_y = image_size[1] / 256.0

            for wp in self.waypoints:
                u = int(wp[0] * scale_x)
                v = int(wp[1] * scale_y)
                u = np.clip(u, 0, depth_map.shape[1] - 1)
                v = np.clip(v, 0, depth_map.shape[0] - 1)
                depths.append(depth_map[v, u])
            depths = np.array(depths)
        else:
            raise ValueError("Need waypoint_depths or depth_map for 3D conversion")

        # Unproject to 3D
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        points_3d = np.zeros((len(self.waypoints), 3), dtype=np.float32)

        for i, (wp, d) in enumerate(zip(self.waypoints, depths)):
            # Scale from [0, 256) to actual image coordinates
            u = wp[0] * (image_size[0] / 256.0)
            v = wp[1] * (image_size[1] / 256.0)

            # Back-project
            points_3d[i] = [
                (u - cx) * d / fx,
                (v - cy) * d / fy,
                d
            ]

        return points_3d

    def to_world_frame(
        self,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        """
        Convert to world-frame 3D coordinates.

        Args:
            camera_intrinsics: Camera matrix [3, 3]
            camera_extrinsics: Camera pose in world frame [4, 4]
            depth_map: Optional depth image
            image_size: Camera image size

        Returns:
            3D points [N, 3] in world frame
        """
        # Get camera-frame points
        points_camera = self.to_3d_points(camera_intrinsics, depth_map, image_size)

        # Transform to world frame
        R = camera_extrinsics[:3, :3]
        t = camera_extrinsics[:3, 3]
        points_world = (R @ points_camera.T).T + t

        return points_world

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def interpolate(self, num_points: int) -> "TrajectoryTrace":
        """
        Interpolate trajectory to specified number of points.

        Args:
            num_points: Target number of waypoints

        Returns:
            New TrajectoryTrace with interpolated waypoints
        """
        if len(self.waypoints) < 2:
            return self

        # Create parameter for interpolation
        t_original = np.linspace(0, 1, len(self.waypoints))
        t_new = np.linspace(0, 1, num_points)

        # Interpolate waypoints
        new_waypoints = np.zeros((num_points, 2), dtype=np.float32)
        new_waypoints[:, 0] = np.interp(t_new, t_original, self.waypoints[:, 0])
        new_waypoints[:, 1] = np.interp(t_new, t_original, self.waypoints[:, 1])

        # Interpolate confidences
        new_confidences = np.interp(t_new, t_original, self.confidences)

        # Interpolate depths if available
        new_depths = None
        if self.waypoint_depths is not None:
            new_depths = np.interp(t_new, t_original, self.waypoint_depths)

        # Create new waypoint types
        new_types = [WaypointType.INTERPOLATED] * num_points
        new_types[-1] = WaypointType.TERMINAL

        return TrajectoryTrace(
            waypoints=new_waypoints,
            confidences=new_confidences,
            waypoint_types=new_types,
            waypoint_depths=new_depths,
            source_image=self.source_image,
            instruction=self.instruction,
            perception_reasoning=self.perception_reasoning,
            spatial_reasoning=self.spatial_reasoning,
            action_reasoning=self.action_reasoning,
        )

    def append_waypoint(
        self,
        waypoint: np.ndarray,
        confidence: float = 1.0,
        wp_type: WaypointType = WaypointType.PREDICTED,
        depth: Optional[float] = None,
    ) -> "TrajectoryTrace":
        """Append a waypoint to the trajectory."""
        new_waypoints = np.vstack([self.waypoints, waypoint])
        new_confidences = np.append(self.confidences, confidence)
        new_types = self.waypoint_types + [wp_type]
        new_depths = None
        if self.waypoint_depths is not None or depth is not None:
            old_depths = (
                self.waypoint_depths if self.waypoint_depths is not None
                else np.zeros(len(self.waypoints))
            )
            new_depths = np.append(old_depths, depth or 0.0)

        return TrajectoryTrace(
            waypoints=new_waypoints,
            confidences=new_confidences,
            waypoint_types=new_types,
            waypoint_depths=new_depths,
            source_image=self.source_image,
            instruction=self.instruction,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "waypoints": self.waypoints.tolist(),
            "confidences": self.confidences.tolist(),
            "waypoint_types": [wt.value for wt in self.waypoint_types],
            "waypoint_depths": (
                self.waypoint_depths.tolist()
                if self.waypoint_depths is not None else None
            ),
            "instruction": self.instruction,
            "timestamp": self.timestamp,
            "perception_reasoning": self.perception_reasoning,
            "spatial_reasoning": self.spatial_reasoning,
            "action_reasoning": self.action_reasoning,
            "model_name": self.model_name,
            "inference_time_ms": self.inference_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryTrace":
        """Deserialize from dictionary."""
        return cls(
            waypoints=np.array(data["waypoints"]),
            confidences=np.array(data["confidences"]),
            waypoint_types=[WaypointType(wt) for wt in data.get("waypoint_types", [])],
            waypoint_depths=(
                np.array(data["waypoint_depths"])
                if data.get("waypoint_depths") else None
            ),
            instruction=data.get("instruction"),
            timestamp=data.get("timestamp", time.time()),
            perception_reasoning=data.get("perception_reasoning"),
            spatial_reasoning=data.get("spatial_reasoning"),
            action_reasoning=data.get("action_reasoning"),
            model_name=data.get("model_name", "trajectory_predictor"),
            inference_time_ms=data.get("inference_time_ms", 0.0),
        )

    @staticmethod
    def _interpolate_color(
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        t: float,
    ) -> Tuple[int, int, int]:
        """Interpolate between two BGR colors."""
        t = np.clip(t, 0, 1)
        return tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))

    def __repr__(self) -> str:
        return (
            f"TrajectoryTrace(waypoints={self.num_waypoints}, "
            f"confidence={self.mean_confidence:.2f}, "
            f"instruction='{self.instruction[:30] if self.instruction else None}...')"
        )

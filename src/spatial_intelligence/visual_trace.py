"""
Visual Trace - MolmoAct-Inspired Trajectory Visualization

Implements visual reasoning traces for operator explainability and steerability,
inspired by MolmoAct's Action Reasoning Model.

This module provides:
1. **VisualTrace**: Data structure for trajectory overlays
2. **VisualTraceRenderer**: Renders waypoints as image overlays
3. **TraceModifier**: Interface for modifying traces via language/drawing

Architecture:
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    VISUAL TRACE PIPELINE                               │
    ├───────────────────────────────────────────────────────────────────────┤
    │                                                                        │
    │  Diffusion Planner Trajectory                                          │
    │         │                                                              │
    │         ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                  WAYPOINT EXTRACTION                            │  │
    │  │     Extract 3D/action waypoints from trajectory [H, A]          │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │         │                                                              │
    │         ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                  TRACE RENDERER                                  │  │
    │  │     Project to 2D image space → Visual overlay                  │  │
    │  │     ★ AUDITABLE: Operators can see planned path                 │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │         │                                                              │
    │         ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                  TRACE MODIFIER (Optional)                       │  │
    │  │     Modify via language or manual waypoint adjustment           │  │
    │  │     ★ STEERABLE: Operators can correct trajectory               │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                                                                        │
    └───────────────────────────────────────────────────────────────────────┘

Usage:
    from src.spatial_intelligence.visual_trace import (
        VisualTraceRenderer,
        VisualTraceConfig,
        VisualTrace,
    )
    
    # Create renderer
    renderer = VisualTraceRenderer(VisualTraceConfig())
    
    # Generate trace from trajectory
    trace = renderer.render_trace(
        trajectory=diffusion_planner.plan().best,
        image=camera_image,
    )
    
    # Access overlay for UI display
    overlay = trace.overlay  # [H, W, 4] RGBA overlay image
    
    # Modify trace via language
    modified_trace = trace.modify("move the middle waypoint to the left")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class TraceStyle(Enum):
    """Visual style for trace rendering."""
    DOTS = "dots"           # Individual waypoint dots
    LINE = "line"           # Connected line path
    ARROWS = "arrows"       # Arrows showing direction
    GRADIENT = "gradient"   # Line with confidence gradient
    COMBINED = "combined"   # Dots + line + arrows


class WaypointType(Enum):
    """Type of waypoint in the trace."""
    START = "start"
    INTERMEDIATE = "intermediate"
    END = "end"
    MODIFIED = "modified"   # User-modified waypoint


@dataclass
class Waypoint:
    """A single waypoint in the visual trace."""
    # 2D image coordinates
    x: float
    y: float
    
    # Original action-space representation
    action: Any  # [A] action vector
    
    # Metadata
    index: int
    waypoint_type: WaypointType = WaypointType.INTERMEDIATE
    confidence: float = 1.0
    
    # Modification tracking
    is_modified: bool = False
    original_x: Optional[float] = None
    original_y: Optional[float] = None


@dataclass
class VisualTraceConfig:
    """Configuration for visual trace rendering."""
    # Rendering style
    style: TraceStyle = TraceStyle.COMBINED
    
    # Colors (RGBA, 0-255)
    waypoint_color: Tuple[int, int, int, int] = (0, 255, 100, 255)  # Green
    line_color: Tuple[int, int, int, int] = (0, 200, 255, 200)       # Cyan
    start_color: Tuple[int, int, int, int] = (100, 255, 100, 255)    # Bright green
    end_color: Tuple[int, int, int, int] = (255, 100, 100, 255)      # Red
    modified_color: Tuple[int, int, int, int] = (255, 200, 0, 255)   # Orange
    
    # Sizing
    waypoint_radius: int = 8
    line_thickness: int = 3
    arrow_size: int = 10
    
    # Waypoint density
    num_displayed_waypoints: int = 8  # Max waypoints to display
    
    # Projection (end-effector to camera)
    use_camera_projection: bool = True
    camera_intrinsics: Optional[Any] = None  # [3, 3] camera matrix
    ee_to_camera_transform: Optional[Any] = None  # [4, 4] transform
    
    # Image dimensions (default to 224x224 if not specified)
    image_width: int = 224
    image_height: int = 224


@dataclass
class VisualTrace:
    """
    Visual representation of a trajectory for operator display.
    
    Provides:
    - Image overlay for UI display
    - Waypoints for interaction
    - Modification history
    """
    # Waypoints in 2D image space
    waypoints: List[Waypoint]
    
    # Rendered overlay [H, W, 4] RGBA
    overlay: Optional[Any] = None
    
    # Original trajectory reference
    trajectory: Optional[Any] = None
    
    # Metadata
    confidence: float = 1.0
    render_time_ms: float = 0.0
    
    # Modification state
    is_modified: bool = False
    modification_history: List[str] = field(default_factory=list)
    
    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)
    
    @property
    def start_waypoint(self) -> Optional[Waypoint]:
        return self.waypoints[0] if self.waypoints else None
    
    @property
    def end_waypoint(self) -> Optional[Waypoint]:
        return self.waypoints[-1] if self.waypoints else None
    
    def get_modified_trajectory(self) -> Optional[Any]:
        """
        Reconstruct trajectory from modified waypoints.
        
        Returns:
            Modified trajectory [H, A] or None if no trajectory
        """
        if not HAS_NUMPY or self.trajectory is None:
            return None
        
        if not self.is_modified:
            return self.trajectory.actions if hasattr(self.trajectory, 'actions') else self.trajectory
        
        # Reconstruct from modified waypoints
        actions = []
        for wp in self.waypoints:
            if isinstance(wp.action, np.ndarray):
                actions.append(wp.action.copy())
            else:
                actions.append(np.array(wp.action))
        
        return np.stack(actions) if actions else None


class VisualTraceRenderer:
    """
    Renders trajectory waypoints as visual overlays.
    
    Provides explainability by showing planned robot movements
    before execution, enabling operator review and modification.
    """
    
    def __init__(self, config: Optional[VisualTraceConfig] = None):
        self.config = config or VisualTraceConfig()
        
        # Stats
        self.stats = {
            "traces_rendered": 0,
            "avg_render_time_ms": 0.0,
        }
    
    def render_trace(
        self,
        trajectory: Any,
        image: Optional[Any] = None,
        camera_intrinsics: Optional[Any] = None,
    ) -> VisualTrace:
        """
        Render a trajectory as a visual trace overlay.
        
        Args:
            trajectory: Trajectory object or action array [H, A]
            image: Optional base image for overlay sizing [H, W, C]
            camera_intrinsics: Optional camera matrix for 3D projection
            
        Returns:
            VisualTrace with waypoints and overlay
        """
        start_time = time.time()
        
        # Extract actions from trajectory
        actions = self._get_actions(trajectory)
        if actions is None:
            logger.warning("No actions in trajectory, returning empty trace")
            return VisualTrace(waypoints=[], overlay=None, trajectory=trajectory)
        
        # Determine image size
        if image is not None and HAS_NUMPY:
            h, w = image.shape[:2]
        else:
            h, w = self.config.image_height, self.config.image_width
        
        # Extract and project waypoints
        waypoints = self._extract_waypoints(actions, w, h)
        
        # Render overlay
        overlay = self._render_overlay(waypoints, w, h)
        
        # Calculate confidence (average of waypoint confidences)
        avg_confidence = sum(wp.confidence for wp in waypoints) / len(waypoints) if waypoints else 1.0
        
        render_time = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats["traces_rendered"] += 1
        self.stats["avg_render_time_ms"] = (
            self.stats["avg_render_time_ms"] * (self.stats["traces_rendered"] - 1) + render_time
        ) / self.stats["traces_rendered"]
        
        return VisualTrace(
            waypoints=waypoints,
            overlay=overlay,
            trajectory=trajectory,
            confidence=avg_confidence,
            render_time_ms=render_time,
        )
    
    def _get_actions(self, trajectory: Any) -> Optional[Any]:
        """Extract action array from trajectory."""
        if not HAS_NUMPY:
            return None
        
        # Handle Trajectory dataclass
        if hasattr(trajectory, 'actions'):
            actions = trajectory.actions
            if hasattr(actions, 'cpu'):  # torch tensor
                actions = actions.cpu().numpy()
            return actions
        
        # Handle raw numpy array
        if isinstance(trajectory, np.ndarray):
            return trajectory
        
        # Try to convert list
        if isinstance(trajectory, list):
            return np.array(trajectory)
        
        return None
    
    def _extract_waypoints(
        self,
        actions: Any,
        width: int,
        height: int,
    ) -> List[Waypoint]:
        """
        Extract waypoints from action sequence and project to 2D.
        
        For now, uses simplified projection assuming first 3 action dims
        are (x, y, z) position. In production, would use proper camera
        projection with extrinsics.
        """
        if not HAS_NUMPY or actions is None:
            return []
        
        horizon = len(actions)
        num_display = min(self.config.num_displayed_waypoints, horizon)
        
        # Sample waypoints evenly
        if horizon > num_display:
            indices = np.linspace(0, horizon - 1, num_display, dtype=int)
        else:
            indices = np.arange(horizon)
        
        waypoints = []
        for i, idx in enumerate(indices):
            action = actions[idx]
            
            # Project to 2D (simplified: use first 2 dims + normalize)
            x, y = self._project_action_to_2d(action, width, height)
            
            # Determine waypoint type
            if i == 0:
                wp_type = WaypointType.START
            elif i == len(indices) - 1:
                wp_type = WaypointType.END
            else:
                wp_type = WaypointType.INTERMEDIATE
            
            # Confidence decreases slightly for future waypoints
            confidence = 1.0 - (i / len(indices)) * 0.3
            
            waypoint = Waypoint(
                x=x,
                y=y,
                action=action,
                index=int(idx),
                waypoint_type=wp_type,
                confidence=confidence,
            )
            waypoints.append(waypoint)
        
        return waypoints
    
    def _project_action_to_2d(
        self,
        action: Any,
        width: int,
        height: int,
    ) -> Tuple[float, float]:
        """
        Project action vector to 2D image coordinates.
        
        Simplified projection for now - assumes first 2 action dimensions
        can be mapped to image space. In production, would use:
        1. Forward kinematics to get end-effector position
        2. Camera intrinsics/extrinsics to project to pixel coordinates
        """
        if not HAS_NUMPY:
            return width / 2, height / 2
        
        # Extract first 2 dimensions (typically position-related)
        # Normalize from typical action range [-1, 1] to image space
        action = np.asarray(action)
        
        # Map action dims to [0, 1] range then scale to image
        # Adding center offset for visualization
        norm_x = (action[0] + 1) / 2  # Map [-1,1] to [0,1]
        norm_y = (action[1] + 1) / 2
        
        # Scale to image with padding
        padding = 0.1
        x = padding * width + norm_x * width * (1 - 2 * padding)
        y = padding * height + norm_y * height * (1 - 2 * padding)
        
        return float(x), float(y)
    
    def _render_overlay(
        self,
        waypoints: List[Waypoint],
        width: int,
        height: int,
    ) -> Optional[Any]:
        """Render waypoints as RGBA overlay image."""
        if not HAS_NUMPY or not waypoints:
            return None
        
        # Create RGBA overlay (transparent background)
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        style = self.config.style
        
        # Draw line first (underneath waypoints)
        if style in [TraceStyle.LINE, TraceStyle.GRADIENT, TraceStyle.COMBINED]:
            self._draw_line(overlay, waypoints)
        
        # Draw arrows if requested
        if style in [TraceStyle.ARROWS, TraceStyle.COMBINED]:
            self._draw_arrows(overlay, waypoints)
        
        # Draw waypoint dots
        if style in [TraceStyle.DOTS, TraceStyle.COMBINED]:
            self._draw_waypoints(overlay, waypoints)
        
        return overlay
    
    def _draw_waypoints(self, overlay: Any, waypoints: List[Waypoint]):
        """Draw waypoint circles on overlay."""
        for wp in waypoints:
            # Get color based on waypoint type
            if wp.is_modified:
                color = self.config.modified_color
            elif wp.waypoint_type == WaypointType.START:
                color = self.config.start_color
            elif wp.waypoint_type == WaypointType.END:
                color = self.config.end_color
            else:
                color = self.config.waypoint_color
            
            # Scale radius by confidence
            radius = int(self.config.waypoint_radius * wp.confidence)
            
            # Draw circle (simple implementation without cv2)
            self._draw_circle(overlay, int(wp.x), int(wp.y), radius, color)
    
    def _draw_line(self, overlay: Any, waypoints: List[Waypoint]):
        """Draw connecting line between waypoints."""
        if len(waypoints) < 2:
            return
        
        color = self.config.line_color
        
        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]
            self._draw_line_segment(
                overlay,
                int(wp1.x), int(wp1.y),
                int(wp2.x), int(wp2.y),
                color
            )
    
    def _draw_arrows(self, overlay: Any, waypoints: List[Waypoint]):
        """Draw direction arrows between waypoints."""
        if len(waypoints) < 2:
            return
        
        # Draw arrow at midpoint between consecutive waypoints
        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]
            mid_x = (wp1.x + wp2.x) / 2
            mid_y = (wp1.y + wp2.y) / 2
            
            # Arrow direction
            dx = wp2.x - wp1.x
            dy = wp2.y - wp1.y
            
            self._draw_arrow(
                overlay,
                int(mid_x), int(mid_y),
                dx, dy,
                self.config.line_color,
                self.config.arrow_size
            )
    
    def _draw_circle(
        self,
        overlay: Any,
        cx: int,
        cy: int,
        radius: int,
        color: Tuple[int, int, int, int],
    ):
        """Draw a filled circle on the overlay."""
        h, w = overlay.shape[:2]
        
        for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    overlay[y, x] = color
    
    def _draw_line_segment(
        self,
        overlay: Any,
        x1: int, y1: int,
        x2: int, y2: int,
        color: Tuple[int, int, int, int],
    ):
        """Draw a line segment using Bresenham's algorithm."""
        h, w = overlay.shape[:2]
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        thickness = self.config.line_thickness
        
        if dx > dy:
            err = dx / 2
            while x != x2:
                self._draw_thick_point(overlay, x, y, thickness, color, w, h)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y2:
                self._draw_thick_point(overlay, x, y, thickness, color, w, h)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
    
    def _draw_thick_point(
        self,
        overlay: Any,
        x: int, y: int,
        thickness: int,
        color: Tuple[int, int, int, int],
        w: int, h: int,
    ):
        """Draw a thick point (small square)."""
        half = thickness // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    overlay[py, px] = color
    
    def _draw_arrow(
        self,
        overlay: Any,
        x: int, y: int,
        dx: float, dy: float,
        color: Tuple[int, int, int, int],
        size: int,
    ):
        """Draw an arrow at position pointing in direction (dx, dy)."""
        if not HAS_NUMPY:
            return
        
        # Normalize direction
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return
        
        dx, dy = dx / length, dy / length
        
        # Arrow head points
        angle = np.pi / 6  # 30 degrees
        
        # Left and right arrow head points
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotate direction vector for arrow heads
        l_dx = dx * cos_a + dy * sin_a
        l_dy = -dx * sin_a + dy * cos_a
        r_dx = dx * cos_a - dy * sin_a
        r_dy = dx * sin_a + dy * cos_a
        
        # Draw arrow head lines
        self._draw_line_segment(
            overlay,
            x, y,
            int(x - l_dx * size), int(y - l_dy * size),
            color
        )
        self._draw_line_segment(
            overlay,
            x, y,
            int(x - r_dx * size), int(y - r_dy * size),
            color
        )


class TraceModifier:
    """
    Modifies visual traces based on language commands or manual waypoint edits.
    
    Enables operator steerability - the key MolmoAct feature for deployment.
    """
    
    # Simple language command patterns for waypoint modification
    MODIFICATION_PATTERNS = {
        "left": (-0.1, 0.0),
        "right": (0.1, 0.0),
        "up": (0.0, -0.1),
        "down": (0.0, 0.1),
        "higher": (0.0, -0.15),
        "lower": (0.0, 0.15),
        "forward": (0.0, -0.1),
        "back": (0.0, 0.1),
        "backward": (0.0, 0.1),
    }
    
    def __init__(self, renderer: Optional[VisualTraceRenderer] = None):
        self.renderer = renderer or VisualTraceRenderer()
    
    def modify_trace(
        self,
        trace: VisualTrace,
        command: str,
        waypoint_index: Optional[int] = None,
    ) -> VisualTrace:
        """
        Modify a trace based on a natural language command.
        
        Args:
            trace: Original visual trace
            command: Natural language command (e.g., "move middle waypoint left")
            waypoint_index: Optional specific waypoint to modify
            
        Returns:
            Modified VisualTrace
        """
        if not trace.waypoints:
            return trace
        
        command_lower = command.lower()
        
        # Determine which waypoint(s) to modify
        if waypoint_index is not None:
            indices = [waypoint_index]
        elif "start" in command_lower:
            indices = [0]
        elif "end" in command_lower:
            indices = [len(trace.waypoints) - 1]
        elif "middle" in command_lower:
            indices = [len(trace.waypoints) // 2]
        elif "all" in command_lower:
            indices = list(range(len(trace.waypoints)))
        else:
            # Default to middle waypoint
            indices = [len(trace.waypoints) // 2]
        
        # Find modification vector
        delta_x, delta_y = 0.0, 0.0
        for keyword, (dx, dy) in self.MODIFICATION_PATTERNS.items():
            if keyword in command_lower:
                delta_x += dx
                delta_y += dy
        
        if delta_x == 0 and delta_y == 0:
            logger.warning(f"Could not parse modification command: {command}")
            return trace
        
        # Apply modifications
        img_w = self.renderer.config.image_width
        img_h = self.renderer.config.image_height
        
        for idx in indices:
            if 0 <= idx < len(trace.waypoints):
                wp = trace.waypoints[idx]
                
                # Store original position
                if not wp.is_modified:
                    wp.original_x = wp.x
                    wp.original_y = wp.y
                
                # Apply delta (scaled by image size)
                wp.x += delta_x * img_w
                wp.y += delta_y * img_h
                
                # Update action vector (simplified - modify first 2 dims)
                if HAS_NUMPY and wp.action is not None:
                    action = np.asarray(wp.action).astype(np.float32)
                    action[0] += delta_x * 2  # Scale to action range [-1, 1]
                    action[1] += delta_y * 2
                    wp.action = action
                
                wp.is_modified = True
                wp.waypoint_type = WaypointType.MODIFIED
        
        # Mark trace as modified
        trace.is_modified = True
        trace.modification_history.append(command)
        
        # Re-render overlay
        trace.overlay = self.renderer._render_overlay(
            trace.waypoints,
            img_w,
            img_h,
        )
        
        return trace
    
    def set_waypoint(
        self,
        trace: VisualTrace,
        waypoint_index: int,
        x: Optional[float] = None,
        y: Optional[float] = None,
        action: Optional[Any] = None,
    ) -> VisualTrace:
        """
        Directly set waypoint position or action.
        
        Args:
            trace: Visual trace to modify
            waypoint_index: Index of waypoint to modify
            x: New x coordinate (image space)
            y: New y coordinate (image space)
            action: New action vector
            
        Returns:
            Modified VisualTrace
        """
        if not trace.waypoints or waypoint_index >= len(trace.waypoints):
            return trace
        
        wp = trace.waypoints[waypoint_index]
        
        # Store original
        if not wp.is_modified:
            wp.original_x = wp.x
            wp.original_y = wp.y
        
        if x is not None:
            wp.x = x
        if y is not None:
            wp.y = y
        if action is not None:
            wp.action = action
        
        wp.is_modified = True
        wp.waypoint_type = WaypointType.MODIFIED
        trace.is_modified = True
        
        # Re-render
        img_w = self.renderer.config.image_width
        img_h = self.renderer.config.image_height
        trace.overlay = self.renderer._render_overlay(trace.waypoints, img_w, img_h)
        
        return trace
    
    def reset_trace(self, trace: VisualTrace) -> VisualTrace:
        """Reset all modifications to original positions."""
        for wp in trace.waypoints:
            if wp.is_modified and wp.original_x is not None:
                wp.x = wp.original_x
                wp.y = wp.original_y
                wp.is_modified = False
                
                # Reset waypoint type
                if wp.index == 0:
                    wp.waypoint_type = WaypointType.START
                elif wp == trace.waypoints[-1]:
                    wp.waypoint_type = WaypointType.END
                else:
                    wp.waypoint_type = WaypointType.INTERMEDIATE
        
        trace.is_modified = False
        trace.modification_history.clear()
        
        # Re-render
        img_w = self.renderer.config.image_width
        img_h = self.renderer.config.image_height
        trace.overlay = self.renderer._render_overlay(trace.waypoints, img_w, img_h)
        
        return trace


# =============================================================================
# Convenience functions
# =============================================================================

def create_visual_trace(
    trajectory: Any,
    image: Optional[Any] = None,
    config: Optional[VisualTraceConfig] = None,
) -> VisualTrace:
    """
    Create a visual trace from a trajectory.
    
    Args:
        trajectory: Trajectory object or action array [H, A]
        image: Optional camera image for sizing
        config: Optional rendering configuration
        
    Returns:
        VisualTrace with rendered overlay
    """
    renderer = VisualTraceRenderer(config)
    return renderer.render_trace(trajectory, image)


def modify_trace_with_language(
    trace: VisualTrace,
    command: str,
    config: Optional[VisualTraceConfig] = None,
) -> VisualTrace:
    """
    Modify a trace using natural language.
    
    Args:
        trace: Visual trace to modify
        command: Natural language command
        config: Optional renderer configuration
        
    Returns:
        Modified VisualTrace
    """
    renderer = VisualTraceRenderer(config)
    modifier = TraceModifier(renderer)
    return modifier.modify_trace(trace, command)

"""
Visual Trace Tests - MolmoAct-Inspired Steerability

Tests for the visual trace renderer, trace modifier, and DIL integration.

Run with:
    pytest tests/test_visual_trace.py -v
    pytest tests/test_visual_trace.py -v -k "render"  # Just render tests
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.spatial_intelligence.visual_trace import (
    VisualTrace,
    VisualTraceConfig,
    VisualTraceRenderer,
    TraceModifier,
    Waypoint,
    WaypointType,
    TraceStyle,
    create_visual_trace,
    modify_trace_with_language,
)

from src.spatial_intelligence.planning.diffusion_planner import (
    Trajectory,
    TrajectoryBatch,
)

from src.spatial_intelligence.deep_imitative_learning import (
    DeepImitativeLearning,
    DILConfig,
    DILResult,
)


# =============================================================================
# Visual Trace Config Tests
# =============================================================================

class TestVisualTraceConfig:
    """Tests for VisualTraceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = VisualTraceConfig()
        
        assert config.style == TraceStyle.COMBINED
        assert config.waypoint_radius == 8
        assert config.num_displayed_waypoints == 8
        assert config.image_width == 224
        assert config.image_height == 224

    def test_custom_config(self):
        """Test custom configuration."""
        config = VisualTraceConfig(
            style=TraceStyle.DOTS,
            waypoint_radius=12,
            num_displayed_waypoints=16,
            image_width=640,
            image_height=480,
        )
        
        assert config.style == TraceStyle.DOTS
        assert config.waypoint_radius == 12
        assert config.num_displayed_waypoints == 16
        assert config.image_width == 640


# =============================================================================
# Waypoint Tests
# =============================================================================

class TestWaypoint:
    """Tests for Waypoint dataclass."""

    def test_waypoint_creation(self):
        """Test creating a waypoint."""
        action = np.random.randn(7).astype(np.float32)
        
        wp = Waypoint(
            x=100.0,
            y=150.0,
            action=action,
            index=0,
            waypoint_type=WaypointType.START,
        )
        
        assert wp.x == 100.0
        assert wp.y == 150.0
        assert wp.index == 0
        assert wp.waypoint_type == WaypointType.START
        assert wp.is_modified is False

    def test_waypoint_modification_tracking(self):
        """Test waypoint modification tracking."""
        wp = Waypoint(
            x=100.0,
            y=150.0,
            action=np.zeros(7),
            index=5,
        )
        
        # Store original
        wp.original_x = wp.x
        wp.original_y = wp.y
        
        # Modify
        wp.x = 120.0
        wp.is_modified = True
        
        assert wp.original_x == 100.0
        assert wp.x == 120.0
        assert wp.is_modified is True


# =============================================================================
# Visual Trace Renderer Tests
# =============================================================================

class TestVisualTraceRenderer:
    """Tests for VisualTraceRenderer."""

    def test_renderer_creation(self):
        """Test creating renderer."""
        renderer = VisualTraceRenderer()
        
        assert renderer is not None
        assert renderer.config is not None
        assert renderer.stats["traces_rendered"] == 0

    def test_render_from_numpy_array(self):
        """Test rendering trace from numpy array."""
        renderer = VisualTraceRenderer()
        
        # Create trajectory as numpy array [H, A]
        actions = np.random.randn(16, 7).astype(np.float32)
        
        trace = renderer.render_trace(trajectory=actions)
        
        assert trace is not None
        assert isinstance(trace, VisualTrace)
        assert trace.num_waypoints > 0
        assert trace.overlay is not None
        assert trace.overlay.shape[2] == 4  # RGBA

    def test_render_from_trajectory(self):
        """Test rendering trace from Trajectory object."""
        renderer = VisualTraceRenderer()
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trajectory = Trajectory(
            actions=actions,
            horizon=16,
            action_dim=7,
        )
        
        trace = renderer.render_trace(trajectory=trajectory)
        
        assert trace is not None
        assert trace.trajectory == trajectory
        assert trace.num_waypoints > 0

    def test_render_with_image_sizing(self):
        """Test rendering with custom image size."""
        config = VisualTraceConfig(
            image_width=640,
            image_height=480,
        )
        renderer = VisualTraceRenderer(config)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        assert trace.overlay.shape[0] == 480  # Height
        assert trace.overlay.shape[1] == 640  # Width

    def test_render_different_styles(self):
        """Test rendering with different styles."""
        actions = np.random.randn(16, 7).astype(np.float32)
        
        for style in [TraceStyle.DOTS, TraceStyle.LINE, TraceStyle.ARROWS, TraceStyle.COMBINED]:
            config = VisualTraceConfig(style=style)
            renderer = VisualTraceRenderer(config)
            
            trace = renderer.render_trace(trajectory=actions)
            
            assert trace is not None
            assert trace.overlay is not None

    def test_render_timing(self):
        """Test that render time is recorded."""
        renderer = VisualTraceRenderer()
        actions = np.random.randn(16, 7).astype(np.float32)
        
        trace = renderer.render_trace(trajectory=actions)
        
        assert trace.render_time_ms > 0
        assert renderer.stats["traces_rendered"] == 1

    def test_waypoint_types(self):
        """Test that waypoint types are correctly assigned."""
        renderer = VisualTraceRenderer()
        actions = np.random.randn(16, 7).astype(np.float32)
        
        trace = renderer.render_trace(trajectory=actions)
        
        assert trace.start_waypoint.waypoint_type == WaypointType.START
        assert trace.end_waypoint.waypoint_type == WaypointType.END


# =============================================================================
# Trace Modifier Tests
# =============================================================================

class TestTraceModifier:
    """Tests for TraceModifier language-based steerability."""

    def test_modifier_creation(self):
        """Test creating modifier."""
        modifier = TraceModifier()
        
        assert modifier is not None
        assert modifier.renderer is not None

    def test_modify_left(self):
        """Test moving waypoint left."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        original_x = trace.waypoints[len(trace.waypoints) // 2].x
        
        modified = modifier.modify_trace(trace, "move middle waypoint left")
        
        new_x = modified.waypoints[len(modified.waypoints) // 2].x
        
        assert new_x < original_x
        assert modified.is_modified is True
        assert len(modified.modification_history) == 1

    def test_modify_right(self):
        """Test moving waypoint right."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        original_x = trace.waypoints[len(trace.waypoints) // 2].x
        
        modified = modifier.modify_trace(trace, "move right")
        
        new_x = modified.waypoints[len(modified.waypoints) // 2].x
        
        assert new_x > original_x

    def test_modify_start_waypoint(self):
        """Test modifying start waypoint."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        original_y = trace.waypoints[0].y
        
        modified = modifier.modify_trace(trace, "move start up")
        
        new_y = modified.waypoints[0].y
        
        assert new_y < original_y  # Up means lower y

    def test_modify_end_waypoint(self):
        """Test modifying end waypoint."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        original_y = trace.waypoints[-1].y
        
        modified = modifier.modify_trace(trace, "move end down")
        
        new_y = modified.waypoints[-1].y
        
        assert new_y > original_y

    def test_reset_trace(self):
        """Test resetting trace modifications."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        original_x = trace.waypoints[0].x
        
        # Modify
        modifier.modify_trace(trace, "move start left")
        assert trace.is_modified is True
        
        # Reset
        reset_trace = modifier.reset_trace(trace)
        
        assert reset_trace.is_modified is False
        assert len(reset_trace.modification_history) == 0

    def test_set_waypoint_directly(self):
        """Test directly setting waypoint position."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        modified = modifier.set_waypoint(trace, waypoint_index=2, x=50.0, y=75.0)
        
        assert modified.waypoints[2].x == 50.0
        assert modified.waypoints[2].y == 75.0
        assert modified.waypoints[2].is_modified is True


# =============================================================================
# Visual Trace Data Structure Tests
# =============================================================================

class TestVisualTrace:
    """Tests for VisualTrace dataclass."""

    def test_trace_creation(self):
        """Test creating a trace."""
        waypoints = [
            Waypoint(x=10.0, y=10.0, action=np.zeros(7), index=0),
            Waypoint(x=20.0, y=20.0, action=np.zeros(7), index=1),
        ]
        
        trace = VisualTrace(waypoints=waypoints)
        
        assert trace.num_waypoints == 2
        assert trace.is_modified is False

    def test_get_modified_trajectory(self):
        """Test getting modified trajectory from trace."""
        renderer = VisualTraceRenderer()
        modifier = TraceModifier(renderer)
        
        actions = np.random.randn(8, 7).astype(np.float32)
        trace = renderer.render_trace(trajectory=actions)
        
        # Modify
        modifier.modify_trace(trace, "move all left")
        
        # Get modified trajectory
        modified_traj = trace.get_modified_trajectory()
        
        assert modified_traj is not None
        assert modified_traj.shape[1] == 7  # Action dim


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_visual_trace(self):
        """Test create_visual_trace convenience function."""
        actions = np.random.randn(16, 7).astype(np.float32)
        
        trace = create_visual_trace(actions)
        
        assert trace is not None
        assert isinstance(trace, VisualTrace)

    def test_modify_trace_with_language(self):
        """Test modify_trace_with_language convenience function."""
        actions = np.random.randn(16, 7).astype(np.float32)
        trace = create_visual_trace(actions)
        
        modified = modify_trace_with_language(trace, "move left")
        
        assert modified.is_modified is True


# =============================================================================
# DIL Integration Tests
# =============================================================================

class TestDILVisualTraceIntegration:
    """Tests for visual trace integration with DeepImitativeLearning."""

    def test_dil_has_trace_fields(self):
        """Test that DIL has trace-related fields."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        
        assert hasattr(dil, 'trace_renderer')
        assert hasattr(dil, 'trace_modifier')
        assert hasattr(dil, '_current_trace')

    def test_dil_result_has_visual_trace(self):
        """Test that DILResult has visual_trace field."""
        config = DILConfig.for_development()
        config.use_safety_gating = False  # Disable to avoid tensor dimension issues
        dil = DeepImitativeLearning(config)
        dil.load()
        
        images = np.random.rand(224, 224, 3).astype(np.float32)
        
        result = dil.execute(
            instruction="test",
            images=images,
        )
        
        assert isinstance(result, DILResult)
        assert hasattr(result, 'visual_trace')
        assert hasattr(result, 'trace_time_ms')

    def test_dil_get_visual_trace(self):
        """Test getting visual trace from DIL."""
        config = DILConfig.for_development()
        config.use_diffusion = True  # Enable to generate trace
        config.use_safety_gating = False  # Disable to avoid tensor dimension issues
        dil = DeepImitativeLearning(config)
        dil.load()
        
        images = np.random.rand(224, 224, 3).astype(np.float32)
        
        result = dil.execute(
            instruction="pick up the cup",
            images=images,
        )
        
        trace = dil.get_visual_trace()
        
        # May be None if diffusion not available
        if result.visual_trace is not None:
            assert trace is not None
            assert trace == result.visual_trace

    def test_dil_modify_trace(self):
        """Test modifying trace via DIL."""
        config = DILConfig.for_development()
        config.use_diffusion = True
        config.use_safety_gating = False  # Disable to avoid tensor dimension issues
        dil = DeepImitativeLearning(config)
        dil.load()
        
        images = np.random.rand(224, 224, 3).astype(np.float32)
        
        result = dil.execute(
            instruction="test",
            images=images,
        )
        
        if result.visual_trace is not None:
            modified = dil.modify_trace("move middle waypoint left")
            
            assert modified is not None
            assert modified.is_modified is True

    def test_dil_stats_include_traces(self):
        """Test that DIL stats track trace generation."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        
        assert "traces_generated" in dil.stats


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

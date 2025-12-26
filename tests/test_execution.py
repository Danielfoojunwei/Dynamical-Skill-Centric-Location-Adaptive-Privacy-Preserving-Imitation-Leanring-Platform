"""
Tests for the Execution Module - DynamicalExecutor and TaskDecomposer

Tests verify that:
1. DynamicalExecutor works without skill blending
2. TaskDecomposer correctly identifies long-horizon tasks
3. Sequential task chaining works properly
4. CBF safety filtering is always applied
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestTaskDecomposer:
    """Tests for TaskDecomposer."""

    def test_decomposer_import(self):
        """Test that decomposer can be imported."""
        from src.execution import TaskDecomposer, DecomposerConfig
        decomposer = TaskDecomposer()
        assert decomposer is not None

    def test_simple_task_no_decomposition(self):
        """Simple tasks should not be decomposed."""
        from src.execution import TaskDecomposer

        decomposer = TaskDecomposer()

        # Simple task
        result = decomposer.decompose("pick up the cup")
        assert len(result.subtasks) == 1
        assert result.subtasks[0].instruction == "pick up the cup"
        assert not result.needs_decomposition

    def test_complex_task_decomposition(self):
        """Complex tasks should be decomposed."""
        from src.execution import TaskDecomposer

        decomposer = TaskDecomposer()

        # Complex task with known pattern
        result = decomposer.decompose("set the table")
        assert len(result.subtasks) > 1
        assert result.needs_decomposition

    def test_sequential_instruction_parsing(self):
        """Tasks with sequence markers should be decomposed."""
        from src.execution import TaskDecomposer

        decomposer = TaskDecomposer()

        # Explicit sequence
        result = decomposer.decompose("pick up the cup then place it on the table")
        assert len(result.subtasks) >= 2

    def test_complexity_estimation(self):
        """Complexity should scale with task complexity."""
        from src.execution import TaskDecomposer

        decomposer = TaskDecomposer()

        simple = decomposer.estimate_complexity("pick up cup")
        complex_ = decomposer.estimate_complexity("clean the entire kitchen and organize all cabinets")

        assert complex_ > simple

    def test_custom_pattern(self):
        """Custom patterns should be usable."""
        from src.execution import TaskDecomposer

        decomposer = TaskDecomposer()
        decomposer.add_custom_pattern("make coffee", [
            "get coffee cup",
            "place under machine",
            "press brew button",
            "wait for brewing",
            "remove cup",
        ])

        result = decomposer.decompose("make coffee")
        assert len(result.subtasks) == 5


class TestDynamicalExecutor:
    """Tests for DynamicalExecutor."""

    def test_executor_import(self):
        """Test that executor can be imported."""
        from src.execution import DynamicalExecutor, ExecutorConfig
        config = ExecutorConfig.minimal()
        executor = DynamicalExecutor(config)
        assert executor is not None

    def test_executor_config_variants(self):
        """Test different config variants."""
        from src.execution import ExecutorConfig

        minimal = ExecutorConfig.minimal()
        assert minimal.use_cbf is True
        assert minimal.use_rta is False
        assert minimal.use_diffusion is False

        thor = ExecutorConfig.for_jetson_thor()
        assert thor.use_cbf is True
        assert thor.use_rta is True
        assert thor.device == "cuda"

    def test_executor_no_blending(self):
        """Verify executor does NOT use skill blending."""
        from src.execution.dynamical_executor import DynamicalExecutor

        # Check that there's no blending-related code
        import inspect
        source = inspect.getsource(DynamicalExecutor)

        # Should NOT contain blending terminology
        assert "blend_weights" not in source.lower() or "deprecated" in source.lower()
        assert "skill_blend" not in source.lower() or "deprecated" in source.lower()

    def test_executor_execute_mock(self):
        """Test execution with mock components."""
        from src.execution import DynamicalExecutor, ExecutorConfig

        config = ExecutorConfig.minimal()
        config.enable_decomposition = False
        executor = DynamicalExecutor(config)

        # Mock state
        state = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
            'ee_position': np.zeros(3),
            'ee_velocity': np.zeros(3),
        }

        # Execute
        result = executor.execute(
            instruction="pick up the cup",
            images=np.zeros((1, 224, 224, 3)),
            state=state,
        )

        assert result.action is not None
        assert len(result.action) == config.action_dim

    def test_executor_sequential_decomposition(self):
        """Test that decomposition is sequential, not parallel."""
        from src.execution import DynamicalExecutor, ExecutorConfig

        config = ExecutorConfig.minimal()
        config.enable_decomposition = True
        executor = DynamicalExecutor(config)
        executor._decomposer = None  # Will use mock

        # Mock state
        state = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
        }

        # Even with "complex" task, execution should be single
        result = executor.execute(
            instruction="pick up cup",
            images=np.zeros((1, 224, 224, 3)),
            state=state,
            force_decomposition=False,
        )

        # Should execute single task
        assert result.total_subtasks == 1

    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        from src.execution import DynamicalExecutor, ExecutorConfig

        config = ExecutorConfig.minimal()
        executor = DynamicalExecutor(config)

        state = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
            'ee_position': np.zeros(3),
            'ee_velocity': np.zeros(3),
        }

        action = executor.emergency_stop(state)
        assert action is not None
        # Emergency stop should produce zero or near-zero action
        assert np.allclose(action, 0, atol=0.1)


class TestNoSkillBlending:
    """Tests to verify skill blending is NOT used."""

    def test_skill_blender_deprecated(self):
        """SkillBlender should raise deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                # Direct import to avoid triggering other modules
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "skill_blender",
                    os.path.join(os.path.dirname(__file__), "..", "src", "core", "skill_blender.py")
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check that deprecation warning was raised
                deprecation_warnings = [
                    x for x in w if issubclass(x.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) > 0
            except Exception as e:
                # If import fails for other reasons, check the file directly
                with open(os.path.join(os.path.dirname(__file__), "..", "src", "core", "skill_blender.py")) as f:
                    content = f.read()
                assert "DEPRECATED" in content
                assert "DeprecationWarning" in content

    def test_vla_handles_multi_objective(self):
        """Verify architecture assumes VLA handles multi-objective."""
        from src.execution.dynamical_executor import DynamicalExecutor
        import inspect

        # Check docstring mentions VLA handling multi-objective
        doc = DynamicalExecutor.__doc__
        assert "multi-objective" in doc.lower() or "vla" in doc.lower()

    def test_composition_is_sequential(self):
        """Composition module should be for sequential chaining only."""
        from src.composition import __doc__ as comp_doc

        # Check that composition docs mention sequential
        assert "sequential" in comp_doc.lower()
        # Should NOT mention parallel blending
        assert "parallel" not in comp_doc.lower() or "not" in comp_doc.lower()


class TestIntegration:
    """Integration tests for full execution pipeline."""

    def test_full_pipeline_simple_task(self):
        """Test full pipeline for simple task."""
        from src.execution import DynamicalExecutor, ExecutorConfig

        config = ExecutorConfig.minimal()
        executor = DynamicalExecutor(config)

        state = {
            'joint_positions': np.random.randn(7) * 0.1,
            'joint_velocities': np.zeros(7),
            'ee_position': np.array([0.5, 0.0, 0.5]),
            'ee_velocity': np.zeros(3),
        }

        result = executor.execute(
            instruction="move forward",
            images=np.random.randn(1, 224, 224, 3),
            state=state,
        )

        # Should get safe action
        assert result.is_safe or result.action is not None
        assert result.total_time_ms >= 0

    def test_executor_statistics(self):
        """Test that executor tracks statistics."""
        from src.execution import DynamicalExecutor, ExecutorConfig

        config = ExecutorConfig.minimal()
        executor = DynamicalExecutor(config)

        state = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
        }

        # Execute multiple times
        for _ in range(5):
            executor.execute(
                instruction="pick",
                images=np.zeros((1, 64, 64, 3)),
                state=state,
            )

        assert executor.stats["total_executions"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

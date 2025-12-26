"""
Deep Imitative Learning Integration Tests

Tests for the Diffusion Planner, RIP Safety Gating, and POIR Recovery components.

Test Categories:
1. Diffusion Planner - trajectory generation
2. RIP Gating - uncertainty estimation and safety
3. POIR Recovery - return-to-distribution planning
4. Integration - full pipeline

Run with:
    pytest tests/test_deep_imitative_learning.py -v
    pytest tests/test_deep_imitative_learning.py -v -k "diffusion"  # Just diffusion tests
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import components directly to avoid import cascade issues
from src.spatial_intelligence.planning.diffusion_planner import (
    DiffusionPlanner,
    DiffusionConfig,
    Trajectory,
    TrajectoryBatch,
    DenoisingSchedule,
)

from src.spatial_intelligence.safety.rip_gating import (
    RIPGating,
    RIPConfig,
    SafetyDecision,
    UncertaintyEstimate,
    RiskLevel,
)

from src.spatial_intelligence.recovery.poir import (
    POIRRecovery,
    POIRConfig,
    RecoveryPlan,
    RecoveryStatus,
    RecoveryStrategy,
)

from src.spatial_intelligence.deep_imitative_learning import (
    DeepImitativeLearning,
    DILConfig,
    DILResult,
    ExecutionMode,
)


# =============================================================================
# Diffusion Planner Tests
# =============================================================================

class TestDiffusionPlannerConfig:
    """Tests for Diffusion Planner configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = DiffusionConfig()

        assert config.action_dim == 7
        assert config.horizon == 16
        assert config.num_diffusion_steps == 100
        assert config.schedule == DenoisingSchedule.COSINE

    def test_custom_config(self):
        """Test custom configuration."""
        config = DiffusionConfig(
            action_dim=10,
            horizon=32,
            num_diffusion_steps=50,
            schedule=DenoisingSchedule.LINEAR,
        )

        assert config.action_dim == 10
        assert config.horizon == 32
        assert config.num_diffusion_steps == 50


class TestDiffusionPlannerCreation:
    """Tests for Diffusion Planner creation."""

    def test_planner_creation(self):
        """Test creating diffusion planner."""
        config = DiffusionConfig(action_dim=7, horizon=16)
        planner = DiffusionPlanner(config)

        assert planner is not None
        assert planner.config.action_dim == 7

    def test_planner_load_mock(self):
        """Test loading planner in mock mode."""
        config = DiffusionConfig()
        planner = DiffusionPlanner(config)
        planner.load_model()

        assert planner.is_loaded
        assert planner.stats["backend"] in ["mock", "torch"]


class TestDiffusionPlannerInference:
    """Tests for Diffusion Planner inference."""

    def test_plan_without_initial_actions(self):
        """Test planning without initial action proposal."""
        config = DiffusionConfig(action_dim=7, horizon=16, num_samples=4)
        planner = DiffusionPlanner(config)
        planner.load_model()

        batch = planner.plan()

        assert batch is not None
        assert isinstance(batch, TrajectoryBatch)
        assert batch.num_samples == 4
        assert batch.best is not None
        assert batch.best.horizon == 16
        assert batch.best.action_dim == 7

    def test_plan_with_initial_actions(self):
        """Test planning with VLA action proposal."""
        config = DiffusionConfig(action_dim=7, horizon=16)
        planner = DiffusionPlanner(config)
        planner.load_model()

        initial_actions = np.random.randn(16, 7).astype(np.float32)

        batch = planner.plan(initial_actions=initial_actions)

        assert batch is not None
        assert batch.best.actions.shape == (16, 7)

    def test_plan_with_condition(self):
        """Test planning with scene condition."""
        config = DiffusionConfig(action_dim=7, horizon=16, condition_dim=512)
        planner = DiffusionPlanner(config)
        planner.load_model()

        condition = np.random.randn(512).astype(np.float32)

        batch = planner.plan(condition=condition)

        assert batch is not None

    def test_trajectory_smoothness(self):
        """Test that trajectory smoothness is computed."""
        config = DiffusionConfig()
        planner = DiffusionPlanner(config)
        planner.load_model()

        batch = planner.plan()

        assert batch.mean_smoothness is not None
        assert batch.best.smoothness is not None

    def test_trajectory_diversity(self):
        """Test trajectory diversity computation."""
        config = DiffusionConfig(num_samples=4)
        planner = DiffusionPlanner(config)
        planner.load_model()

        batch = planner.plan()

        assert batch.diversity is not None
        assert batch.diversity >= 0


# =============================================================================
# RIP Gating Tests
# =============================================================================

class TestRIPConfig:
    """Tests for RIP configuration."""

    def test_default_config(self):
        """Test default RIP configuration."""
        config = RIPConfig()

        assert config.ensemble_size == 5
        assert config.action_dim == 7
        assert config.safe_threshold < config.caution_threshold
        assert config.caution_threshold < config.warning_threshold

    def test_thresholds_ordering(self):
        """Test that thresholds are properly ordered."""
        config = RIPConfig()

        assert config.safe_threshold < config.caution_threshold
        assert config.caution_threshold < config.warning_threshold


class TestRIPGatingCreation:
    """Tests for RIP Gating creation."""

    def test_rip_creation(self):
        """Test creating RIP gating."""
        config = RIPConfig(ensemble_size=3)
        rip = RIPGating(config)

        assert rip is not None
        assert rip.config.ensemble_size == 3

    def test_rip_load_mock(self):
        """Test loading RIP in mock mode."""
        config = RIPConfig()
        rip = RIPGating(config)
        rip.load_model()

        assert rip.is_loaded


class TestRIPGatingInference:
    """Tests for RIP Gating inference."""

    def test_evaluate_safe_observation(self):
        """Test evaluating a safe observation."""
        config = RIPConfig()
        rip = RIPGating(config)
        rip.load_model()

        observation = np.random.randn(512).astype(np.float32)

        decision = rip.evaluate(observation)

        assert decision is not None
        assert isinstance(decision, SafetyDecision)
        assert decision.risk_level in list(RiskLevel)
        assert isinstance(decision.is_safe, bool)
        assert 0 <= decision.confidence <= 1

    def test_evaluate_with_proprio(self):
        """Test evaluation with proprioceptive state."""
        config = RIPConfig(proprio_dim=7)
        rip = RIPGating(config)
        rip.load_model()

        observation = np.random.randn(512).astype(np.float32)
        proprio = np.random.randn(7).astype(np.float32)

        decision = rip.evaluate(observation, proprio=proprio)

        assert decision is not None

    def test_uncertainty_estimate(self):
        """Test that uncertainty estimate is returned."""
        config = RIPConfig()
        rip = RIPGating(config)
        rip.load_model()

        observation = np.random.randn(512).astype(np.float32)
        decision = rip.evaluate(observation)

        assert decision.uncertainty is not None
        assert isinstance(decision.uncertainty, UncertaintyEstimate)
        assert decision.uncertainty.epistemic >= 0
        assert decision.uncertainty.aleatoric >= 0

    def test_ood_score(self):
        """Test OOD score computation."""
        config = RIPConfig()
        rip = RIPGating(config)
        rip.load_model()

        observation = np.random.randn(512).astype(np.float32)
        decision = rip.evaluate(observation)

        assert hasattr(decision, 'ood_score')
        assert decision.ood_score >= 0

    def test_risk_level_classification(self):
        """Test that risk levels are properly classified."""
        config = RIPConfig()
        rip = RIPGating(config)
        rip.load_model()

        observation = np.random.randn(512).astype(np.float32)
        decision = rip.evaluate(observation)

        # Check risk level is valid
        assert decision.risk_level in [
            RiskLevel.SAFE,
            RiskLevel.CAUTION,
            RiskLevel.WARNING,
            RiskLevel.CRITICAL,
        ]


# =============================================================================
# POIR Recovery Tests
# =============================================================================

class TestPOIRConfig:
    """Tests for POIR configuration."""

    def test_default_config(self):
        """Test default POIR configuration."""
        config = POIRConfig()

        assert config.action_dim == 7
        assert config.max_recovery_steps > 0
        assert config.recovery_success_threshold > 0

    def test_config_with_waypoints(self):
        """Test config with safe waypoints."""
        waypoint = np.zeros(7)
        config = POIRConfig(
            safe_waypoints=[waypoint],
            home_position=waypoint,
        )

        assert len(config.safe_waypoints) == 1
        assert config.home_position is not None


class TestPOIRRecoveryCreation:
    """Tests for POIR Recovery creation."""

    def test_poir_creation(self):
        """Test creating POIR recovery."""
        config = POIRConfig()
        poir = POIRRecovery(config)

        assert poir is not None
        assert not poir.is_recovering

    def test_poir_with_home(self):
        """Test POIR with home position."""
        home = np.zeros(7)
        config = POIRConfig(home_position=home)
        poir = POIRRecovery(config)

        assert poir.config.home_position is not None


class TestPOIRRecoveryPlanning:
    """Tests for POIR Recovery planning."""

    def test_plan_recovery_reverse(self):
        """Test planning recovery with reverse strategy."""
        config = POIRConfig()
        poir = POIRRecovery(config)

        # Record some action history
        for _ in range(5):
            action = np.random.randn(7).astype(np.float32)
            poir.record_step(np.zeros(512), action)

        observation = np.random.randn(512).astype(np.float32)
        plan = poir.plan_recovery(observation, ood_score=0.7)

        assert plan is not None
        assert isinstance(plan, RecoveryPlan)
        assert plan.trajectory is not None
        assert plan.confidence > 0

    def test_plan_recovery_with_waypoints(self):
        """Test planning recovery to safe waypoint."""
        waypoint = np.zeros(7)
        config = POIRConfig(safe_waypoints=[waypoint])
        poir = POIRRecovery(config)

        observation = np.random.randn(512).astype(np.float32)
        plan = poir.plan_recovery(observation, ood_score=0.7)

        assert plan is not None
        # Should consider waypoint strategy
        assert plan.strategy in list(RecoveryStrategy)

    def test_plan_recovery_home(self):
        """Test planning recovery to home."""
        home = np.zeros(7)
        config = POIRConfig(home_position=home)
        poir = POIRRecovery(config)

        observation = np.random.randn(512).astype(np.float32)
        plan = poir.plan_recovery(observation, ood_score=0.8)

        assert plan is not None

    def test_recovery_step_execution(self):
        """Test stepping through recovery."""
        config = POIRConfig()
        poir = POIRRecovery(config)

        # Record history and plan
        for _ in range(5):
            poir.record_step(np.zeros(512), np.random.randn(7))

        poir.plan_recovery(np.random.randn(512), ood_score=0.7)

        assert poir.is_recovering

        # Step through recovery
        action, is_complete = poir.step_recovery(np.random.randn(512))

        assert action is not None or is_complete

    def test_abort_recovery(self):
        """Test aborting recovery."""
        config = POIRConfig()
        poir = POIRRecovery(config)

        for _ in range(5):
            poir.record_step(np.zeros(512), np.random.randn(7))

        poir.plan_recovery(np.random.randn(512), ood_score=0.7)
        assert poir.is_recovering

        poir.abort_recovery()
        assert not poir.is_recovering


# =============================================================================
# Deep Imitative Learning Integration Tests
# =============================================================================

class TestDILConfig:
    """Tests for Deep Imitative Learning configuration."""

    def test_default_config(self):
        """Test default DIL configuration."""
        config = DILConfig()

        assert config.action_dim == 7
        assert config.action_horizon == 16
        assert config.use_diffusion is True
        assert config.use_safety_gating is True
        assert config.use_recovery is True

    def test_development_config(self):
        """Test development configuration."""
        config = DILConfig.for_development()

        assert config.device == "cpu"
        assert config.use_diffusion is False  # Disabled for speed

    def test_thor_config(self):
        """Test Jetson Thor configuration."""
        config = DILConfig.for_jetson_thor()

        assert config.device == "cuda"
        assert config.pi05_variant == "pi05_base"


class TestDILCreation:
    """Tests for Deep Imitative Learning creation."""

    def test_dil_creation(self):
        """Test creating DIL pipeline."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)

        assert dil is not None
        assert dil.mode == ExecutionMode.NORMAL

    def test_dil_factory_methods(self):
        """Test factory methods."""
        dil_dev = DeepImitativeLearning.for_development()
        assert dil_dev is not None

        dil_thor = DeepImitativeLearning.for_jetson_thor()
        assert dil_thor is not None


class TestDILExecution:
    """Tests for Deep Imitative Learning execution."""

    def test_execute_basic(self):
        """Test basic execution."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        dil.load()

        images = np.random.rand(224, 224, 3).astype(np.float32)

        result = dil.execute(
            instruction="pick up the cup",
            images=images,
        )

        assert result is not None
        assert isinstance(result, DILResult)
        assert result.actions is not None
        assert isinstance(result.is_safe, bool)

    def test_execute_with_proprio(self):
        """Test execution with proprioceptive state."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        dil.load()

        images = np.random.rand(224, 224, 3).astype(np.float32)
        proprio = np.random.randn(7).astype(np.float32)

        result = dil.execute(
            instruction="move to target",
            images=images,
            proprio=proprio,
        )

        assert result is not None

    def test_execute_timing(self):
        """Test that timing is recorded."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        dil.load()

        images = np.random.rand(224, 224, 3).astype(np.float32)

        result = dil.execute(
            instruction="test",
            images=images,
        )

        assert result.total_time_ms > 0
        assert result.vla_time_ms >= 0

    def test_execute_stats(self):
        """Test that stats are updated."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        dil.load()

        images = np.random.rand(224, 224, 3).astype(np.float32)

        dil.execute(instruction="test", images=images)

        assert dil.stats["total_executions"] == 1


class TestDILRecovery:
    """Tests for DIL recovery handling."""

    def test_set_safe_waypoints(self):
        """Test setting safe waypoints."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        dil.load()

        waypoints = [np.zeros(7), np.ones(7)]
        dil.set_safe_waypoints(waypoints)

        # No error means success
        assert True

    def test_set_home_position(self):
        """Test setting home position."""
        config = DILConfig.for_development()
        dil = DeepImitativeLearning(config)
        dil.load()

        home = np.zeros(7)
        dil.set_home_position(home)

        assert True


# =============================================================================
# Trajectory and Data Structure Tests
# =============================================================================

class TestTrajectory:
    """Tests for Trajectory data structure."""

    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        actions = np.random.randn(16, 7).astype(np.float32)

        traj = Trajectory(
            actions=actions,
            horizon=16,
            action_dim=7,
        )

        assert traj.actions is not None
        assert traj.horizon == 16
        assert traj.action_dim == 7

    def test_trajectory_to_numpy(self):
        """Test converting trajectory to numpy."""
        actions = np.random.randn(16, 7).astype(np.float32)
        traj = Trajectory(actions=actions, horizon=16, action_dim=7)

        result = traj.to_numpy()

        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 7)


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch data structure."""

    def test_batch_creation(self):
        """Test creating a trajectory batch."""
        trajectories = [
            Trajectory(np.random.randn(16, 7).astype(np.float32), 16, 7)
            for _ in range(4)
        ]

        batch = TrajectoryBatch(trajectories=trajectories, best_idx=0)

        assert batch.num_samples == 4
        assert batch.best is not None

    def test_batch_best_selection(self):
        """Test selecting best trajectory."""
        trajectories = [
            Trajectory(np.random.randn(16, 7).astype(np.float32), 16, 7)
            for _ in range(4)
        ]

        batch = TrajectoryBatch(trajectories=trajectories, best_idx=2)

        assert batch.best == trajectories[2]


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

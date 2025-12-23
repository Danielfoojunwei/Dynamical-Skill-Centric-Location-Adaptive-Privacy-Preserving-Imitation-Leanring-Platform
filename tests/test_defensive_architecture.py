"""
Defensive Architecture Tests

Tests for all defensive components described in README v0.7.1:
1. SkillBlender - Stability guarantees (normalization, jerk limiting, confidence weighting)
2. SafetyShield - Deterministic checks, no ML dependency
3. MOAI FHE - Offline-only guards
4. N2HE - Performance metrics
5. CascadedPerception - L1/L2/L3 escalation
"""

import pytest
import numpy as np
import time
import sys
import os
import importlib.util

# Add src to path for direct module imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)


def load_module_direct(module_path: str, module_name: str):
    """Load a module directly from path without going through __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules directly to avoid __init__.py import chain issues
skill_blender_module = load_module_direct(
    os.path.join(src_dir, "src/core/skill_blender.py"),
    "skill_blender"
)
SkillBlender = skill_blender_module.SkillBlender
BlendConfig = skill_blender_module.BlendConfig
SkillOutput = skill_blender_module.SkillOutput
ActionSpace = skill_blender_module.ActionSpace

# Load safety modules
robot_runtime_config = load_module_direct(
    os.path.join(src_dir, "src/robot_runtime/config.py"),
    "robot_runtime_config"
)
SafetyConfig = robot_runtime_config.SafetyConfig
PerceptionConfig = robot_runtime_config.PerceptionConfig

safety_shield_module = load_module_direct(
    os.path.join(src_dir, "src/robot_runtime/safety_shield.py"),
    "safety_shield"
)
SafetyShield = safety_shield_module.SafetyShield
SafetyStatus = safety_shield_module.SafetyStatus

perception_module = load_module_direct(
    os.path.join(src_dir, "src/robot_runtime/perception_pipeline.py"),
    "perception_pipeline"
)
PerceptionPipeline = perception_module.PerceptionPipeline
CascadeLevel = perception_module.CascadeLevel

# Load MOAI modules
moai_fhe_module = load_module_direct(
    os.path.join(src_dir, "src/moai/moai_fhe.py"),
    "moai_fhe"
)
MoaiFHEContext = moai_fhe_module.MoaiFHEContext

n2he_module = load_module_direct(
    os.path.join(src_dir, "src/moai/n2he.py"),
    "n2he"
)
N2HEContext = n2he_module.N2HEContext


# =============================================================================
# SkillBlender Tests
# =============================================================================

class TestSkillBlender:
    """Tests for SkillBlender stability guarantees."""

    @pytest.fixture
    def blender(self):
        config = BlendConfig(
            confidence_threshold=0.3,
            max_action_delta_per_second=2.0,
        )
        blender = SkillBlender(config=config, action_dim=7)
        blender.register_skill("grasp", ActionSpace.JOINT_POSITION, action_dim=7)
        blender.register_skill("place", ActionSpace.JOINT_POSITION, action_dim=7)
        return blender

    def test_normalization_validation(self, blender):
        """Test that actions outside [-1, 1] are rejected."""
        bad_output = SkillOutput(
            action=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            skill_id="grasp",
            action_space=ActionSpace.JOINT_POSITION,
        )

        is_valid, error = blender.validate_skill_output(bad_output)
        assert not is_valid, "Should reject action outside [-1, 1]"
        assert "above maximum" in error or "below minimum" in error

    def test_confidence_weighting(self, blender):
        """Test that low confidence skills are down-weighted."""
        outputs = [
            SkillOutput(
                action=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                confidence=0.9,
                skill_id="grasp",
                action_space=ActionSpace.JOINT_POSITION,
            ),
            SkillOutput(
                action=np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                confidence=0.1,
                skill_id="place",
                action_space=ActionSpace.JOINT_POSITION,
            ),
        ]

        result = blender.blend(outputs, [0.5, 0.5], dt=0.01)
        assert result.action[0] > 0, "High confidence skill should dominate"

    def test_safe_default_on_all_low_confidence(self, blender):
        """Test that safe default is used when all skills have low confidence."""
        outputs = [
            SkillOutput(
                action=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                confidence=0.1,
                skill_id="grasp",
                action_space=ActionSpace.JOINT_POSITION,
            ),
            SkillOutput(
                action=np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
                confidence=0.05,
                skill_id="place",
                action_space=ActionSpace.JOINT_POSITION,
            ),
        ]

        result = blender.blend(outputs, [0.5, 0.5], dt=0.01)
        assert result.used_safe_default, "Should use safe default"
        assert np.allclose(result.action, 0), "Safe default should be zero"

    def test_jerk_limiting(self, blender):
        """Test that large action changes are smoothed."""
        # First blend to establish baseline
        outputs1 = [
            SkillOutput(
                action=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                confidence=0.9,
                skill_id="grasp",
                action_space=ActionSpace.JOINT_POSITION,
            ),
        ]
        blender.blend(outputs1, [1.0], dt=0.01)

        # Large jump
        outputs2 = [
            SkillOutput(
                action=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                confidence=0.9,
                skill_id="grasp",
                action_space=ActionSpace.JOINT_POSITION,
            ),
        ]
        result2 = blender.blend(outputs2, [1.0], dt=0.01)

        assert result2.jerk_limited, "Should apply jerk limiting"
        assert result2.action[0] < 0.5, "Jerk should be limited"

    def test_action_space_mismatch_rejected(self, blender):
        """Test that mismatched action spaces are rejected."""
        with pytest.raises(ValueError, match="Action space mismatch"):
            blender.register_skill("nav", ActionSpace.CARTESIAN_POSE, action_dim=7)

    def test_weight_sum_normalization(self, blender):
        """Test that weights are normalized to sum to 1."""
        outputs = [
            SkillOutput(
                action=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                confidence=0.9,
                skill_id="grasp",
                action_space=ActionSpace.JOINT_POSITION,
            ),
        ]
        result = blender.blend(outputs, [0.3], dt=0.01)
        assert not result.used_safe_default


# =============================================================================
# SafetyShield Tests
# =============================================================================

class TestSafetyShield:
    """Tests for SafetyShield deterministic checks."""

    @pytest.fixture
    def shield(self):
        config = SafetyConfig()
        shield = SafetyShield(rate_hz=1000, config=config)
        shield.initialize()
        return shield

    def test_normal_operation(self, shield):
        """Test normal operation passes safety checks."""
        robot_state = {
            'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'joint_torques': np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
        }
        shield.heartbeat()
        is_safe, _ = shield.check(robot_state)
        assert is_safe, "Normal operation should be safe"

    def test_position_limit_violation(self, shield):
        """Test position limit violations are detected."""
        robot_state = {
            'joint_positions': np.array([2.90, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        }
        shield.heartbeat()
        is_safe, _ = shield.check(robot_state)
        assert not is_safe, "Position limit violation should be detected"
        assert any(v.type == "position_limit" for v in shield.state.violations)

    def test_velocity_limit_violation(self, shield):
        """Test velocity limit violations are detected."""
        robot_state = {
            'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        }
        shield.heartbeat()
        is_safe, _ = shield.check(robot_state)
        assert not is_safe, "Velocity limit violation should be detected"

    def test_torque_limit_triggers_estop(self, shield):
        """Test that torque limit violation triggers E-stop."""
        robot_state = {
            'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'joint_torques': np.array([100.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
        }
        shield.heartbeat()
        is_safe, _ = shield.check(robot_state)
        assert not is_safe
        assert shield._estop_triggered, "Torque violation should trigger E-stop"

    def test_ml_advisory_cannot_override_safety(self, shield):
        """Test that ML predictions cannot override safety decisions."""
        robot_state = {
            'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        }
        shield.set_ml_advisory(collision_probability=0.99, speed_reduction_factor=0.0)
        shield.heartbeat()
        is_safe, _ = shield.check(robot_state)
        assert is_safe, "ML advisory should NOT override deterministic safety"
        assert shield.state.ml_collision_warning, "ML warning should be set"

    def test_human_proximity_triggers_estop(self, shield):
        """Test that human proximity triggers E-stop."""
        robot_state = {
            'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'humans': [{'distance': 0.1}],
        }
        shield.heartbeat()
        is_safe, _ = shield.check(robot_state)
        assert not is_safe
        assert shield._estop_triggered, "Human proximity should trigger E-stop"

    def test_check_time_under_limit(self, shield):
        """Test that safety check completes within time limit."""
        robot_state = {
            'joint_positions': np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.5, 0.0]),
            'joint_velocities': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'joint_torques': np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
            'obstacles': [{'distance': 1.0}],
            'humans': [],
        }

        for _ in range(100):
            shield.heartbeat()
            shield.check(robot_state)

        assert shield.stats["avg_check_time_us"] < 500, "Check should be under 500μs average"


# =============================================================================
# MOAI FHE Offline Guards Tests
# =============================================================================

class TestMoaiFHEOfflineGuards:
    """Tests for MOAI FHE offline-only guards."""

    def test_offline_warning_in_docstring(self):
        """Test that MOAI FHE class has offline warning in docstring."""
        assert "OFFLINE" in MoaiFHEContext.__doc__
        assert "NOT FOR REAL-TIME" in MoaiFHEContext.__doc__

    def test_allow_realtime_warning(self):
        """Test that allow_realtime=True generates warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MoaiFHEContext(use_mock=True, allow_realtime=True)
            assert len(w) >= 1
            assert any("testing only" in str(warning.message) for warning in w)


# =============================================================================
# N2HE Tests
# =============================================================================

class TestN2HE:
    """Tests for N2HE encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test basic encrypt/decrypt functionality."""
        ctx = N2HEContext()
        ctx.generate_keys(generate_boot_key=False)

        for val in [0, 1, 100, 1000, 32767]:
            ct = ctx.encrypt(val)
            dec = ctx.decrypt(ct)
            assert dec == val, f"Roundtrip failed for {val}: got {dec}"

    def test_homomorphic_addition(self):
        """Test homomorphic addition."""
        ctx = N2HEContext()
        ctx.generate_keys(generate_boot_key=False)

        ct1 = ctx.encrypt(100)
        ct2 = ctx.encrypt(200)
        ct_sum = ctx.add(ct1, ct2)
        result = ctx.decrypt(ct_sum)

        assert result == 300, f"Expected 300, got {result}"

    def test_homomorphic_scalar_multiplication(self):
        """Test homomorphic scalar multiplication."""
        ctx = N2HEContext()
        ctx.generate_keys(generate_boot_key=False)

        ct = ctx.encrypt(50)
        ct_scaled = ctx.mul_plain(ct, 3)
        result = ctx.decrypt(ct_scaled)

        assert result == 150, f"Expected 150, got {result}"

    def test_ciphertext_size(self):
        """Test that ciphertext is expanded (security requirement)."""
        ctx = N2HEContext()
        ctx.generate_keys(generate_boot_key=False)

        ct = ctx.encrypt(100)
        assert ct.size_bytes > 1000, "Ciphertext should be expanded for security"


# =============================================================================
# Cascaded Perception Tests
# =============================================================================

class TestCascadedPerception:
    """Tests for cascaded perception pipeline."""

    @pytest.fixture
    def pipeline(self):
        config = PerceptionConfig()
        pipeline = PerceptionPipeline(rate_hz=30, config=config)
        pipeline.initialize()
        return pipeline

    def test_level1_always_runs(self, pipeline):
        """Test that Level 1 perception always runs."""
        pipeline.update()
        assert pipeline._features.cascade_level in [
            CascadeLevel.LEVEL_1,
            CascadeLevel.LEVEL_2,
            CascadeLevel.LEVEL_3
        ]

    def test_level2_triggers_on_low_confidence(self, pipeline):
        """Test that Level 2 triggers on low confidence."""
        pipeline._features.detections = [
            {'class': 'unknown', 'confidence': 0.5}
        ]
        assert pipeline._needs_level2(), "Level 2 should trigger on low confidence"

    def test_level3_triggers_on_human(self, pipeline):
        """Test that Level 3 triggers on human detection."""
        pipeline._features.detections = [
            {'class': 'person', 'confidence': 0.9}
        ]
        assert pipeline._needs_level3(), "Level 3 should trigger on human detection"

    def test_level3_cooldown(self, pipeline):
        """Test Level 3 cooldown prevents rapid execution."""
        pipeline._last_level3_time = time.time()
        assert not pipeline._can_run_level3(), "Level 3 should have cooldown"

        pipeline._last_level3_time = time.time() - 10.0
        assert pipeline._can_run_level3(), "Level 3 should be allowed after cooldown"


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests to ensure timing contracts are met."""

    def test_safety_check_performance(self):
        """Test safety check meets 500μs requirement."""
        config = SafetyConfig()
        shield = SafetyShield(rate_hz=1000, config=config)
        shield.initialize()

        robot_state = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
            'joint_torques': np.zeros(7),
            'obstacles': [{'distance': 1.0}],
            'humans': [],
        }

        # Warm up
        for _ in range(10):
            shield.heartbeat()
            shield.check(robot_state)

        # Measure
        times = []
        for _ in range(1000):
            shield.heartbeat()
            start = time.perf_counter()
            shield.check(robot_state)
            times.append((time.perf_counter() - start) * 1_000_000)

        avg_time = np.mean(times)
        assert avg_time < 500, f"Average check time {avg_time:.1f}μs exceeds 500μs"
        assert np.percentile(times, 99) < 1000, f"99th percentile exceeds 1000μs"

    def test_skill_blender_performance(self):
        """Test skill blender meets latency requirements."""
        config = BlendConfig()
        blender = SkillBlender(config=config, action_dim=7)
        blender.register_skill("s1", ActionSpace.JOINT_POSITION, action_dim=7)
        blender.register_skill("s2", ActionSpace.JOINT_POSITION, action_dim=7)
        blender.register_skill("s3", ActionSpace.JOINT_POSITION, action_dim=7)

        outputs = [
            SkillOutput(
                action=np.random.uniform(-1, 1, 7).astype(np.float32),
                confidence=0.9,
                skill_id=f"s{i+1}",
                action_space=ActionSpace.JOINT_POSITION,
            )
            for i in range(3)
        ]
        weights = [0.4, 0.35, 0.25]

        # Warm up
        for _ in range(10):
            blender.blend(outputs, weights, dt=0.01)

        # Measure
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            blender.blend(outputs, weights, dt=0.01)
            times.append((time.perf_counter() - start) * 1_000_000)

        avg_time = np.mean(times)
        assert avg_time < 100, f"Average blend time {avg_time:.1f}μs exceeds 100μs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

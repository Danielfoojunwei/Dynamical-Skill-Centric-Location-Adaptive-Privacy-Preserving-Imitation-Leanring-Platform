#!/usr/bin/env python3
"""
Comprehensive Test Suite for Configuration Loader

Tests:
- Configuration model validation
- YAML file loading
- Environment variable overrides
- Meta AI configuration
- TFLOPS allocation
"""

import sys
import os
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Configuration Model Tests
# =============================================================================

class TestConfigModels:
    """Test suite for configuration pydantic models."""

    def test_system_config_defaults(self):
        """Test SystemConfig default values."""
        from src.core.config_loader import SystemConfig

        config = SystemConfig()

        assert config.simulation_mode == False
        assert config.log_level == "INFO"
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000

    def test_system_config_custom_values(self):
        """Test SystemConfig with custom values."""
        from src.core.config_loader import SystemConfig

        config = SystemConfig(
            simulation_mode=True,
            log_level="DEBUG",
            api_host="127.0.0.1",
            api_port=3000
        )

        assert config.simulation_mode == True
        assert config.log_level == "DEBUG"
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 3000

    def test_safety_config_defaults(self):
        """Test SafetyConfig default values."""
        from src.core.config_loader import SafetyConfig

        config = SafetyConfig()

        assert config.stop_dist == 1.5
        assert config.sensitivity == 0.8
        assert config.max_speed_factor == 1.0

    def test_safety_config_validation_bounds(self):
        """Test SafetyConfig value bounds."""
        from src.core.config_loader import SafetyConfig
        from pydantic import ValidationError

        # Valid config at bounds
        config = SafetyConfig(stop_dist=0.5, sensitivity=0.1, max_speed_factor=0.1)
        assert config.stop_dist == 0.5

        config = SafetyConfig(stop_dist=5.0, sensitivity=1.0, max_speed_factor=1.0)
        assert config.stop_dist == 5.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            SafetyConfig(stop_dist=0.1)  # Min is 0.5

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            SafetyConfig(stop_dist=10.0)  # Max is 5.0

    def test_camera_config_defaults(self):
        """Test CameraConfig defaults."""
        from src.core.config_loader import CameraConfig

        config = CameraConfig()

        assert "rtsp://" in config.rtsp_url
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30

    def test_models_config(self):
        """Test ModelsConfig."""
        from src.core.config_loader import ModelsConfig

        config = ModelsConfig()
        assert config.paligemma_path == "models/paligemma"
        assert config.yolo_path is None

        config = ModelsConfig(yolo_path="models/yolo")
        assert config.yolo_path == "models/yolo"

    def test_tflops_allocation_defaults(self):
        """Test TFLOPSAllocation default values."""
        from src.core.config_loader import TFLOPSAllocation

        alloc = TFLOPSAllocation()

        assert alloc.safety_detection == 15.0
        assert alloc.navigation_detection == 30.0
        assert alloc.depth_estimation == 5.0
        assert alloc.pi0_vla == 10.0
        assert alloc.full_perception == 15.0

    def test_tflops_budget_defaults(self):
        """Test TFLOPSBudget defaults."""
        from src.core.config_loader import TFLOPSBudget

        budget = TFLOPSBudget()

        assert budget.total_fp16 == 137.0
        assert budget.safe_utilization == 0.85
        assert budget.burst_utilization == 0.95
        assert budget.allocations is not None

    def test_tflops_budget_total_allocation(self):
        """Test that TFLOPS allocations are reasonable."""
        from src.core.config_loader import TFLOPSBudget, TFLOPSAllocation

        budget = TFLOPSBudget()
        alloc = budget.allocations

        # Calculate total allocation
        total = (
            alloc.safety_detection +
            alloc.spatial_brain +
            alloc.navigation_detection +
            alloc.depth_estimation +
            alloc.il_training +
            alloc.moai_compression +
            alloc.fhe_encryption +
            alloc.pi0_vla +
            alloc.full_perception +
            alloc.anomaly_detection
        )

        # Total should be less than budget (leaving room for Meta AI)
        assert total <= budget.total_fp16 * budget.safe_utilization

    def test_pipeline_config(self):
        """Test PipelineConfig."""
        from src.core.config_loader import PipelineConfig

        config = PipelineConfig()

        assert config.queues.safety_maxsize == 100
        assert config.queues.perception_maxsize == 500
        assert config.routing.safety_cameras == [0, 1, 2, 3]
        assert config.learning.batch_size == 1000

    def test_retargeting_config(self):
        """Test RetargetingConfig."""
        from src.core.config_loader import RetargetingConfig

        config = RetargetingConfig()

        assert config.workspace_min == [-1.0, -1.0, 0.0]
        assert config.workspace_max == [1.0, 1.0, 1.5]
        assert config.max_joint_velocity == 1.0
        assert config.ik.max_position_error == 0.05
        assert config.ik.fallback_to_previous == True

    def test_app_config_defaults(self):
        """Test AppConfig with all defaults."""
        from src.core.config_loader import AppConfig

        config = AppConfig()

        assert config.system is not None
        assert config.safety is not None
        assert config.cameras is not None
        assert config.models is not None
        assert config.tflops_budget is not None
        assert config.pipeline is not None
        assert config.retargeting is not None


# =============================================================================
# Config Loader Tests
# =============================================================================

class TestConfigLoader:
    """Test suite for config loading functionality."""

    def test_load_existing_config(self):
        """Test loading the actual config file."""
        from src.core.config_loader import load_and_validate_config

        config = load_and_validate_config("config/config.yaml")

        assert config is not None
        assert config.system is not None

    def test_load_nonexistent_file_returns_defaults(self):
        """Test loading non-existent file returns defaults."""
        from src.core.config_loader import load_and_validate_config

        config = load_and_validate_config("nonexistent/path/config.yaml")

        # Should return default config, not raise
        assert config is not None
        assert config.system.simulation_mode == False

    def test_load_custom_yaml(self):
        """Test loading a custom YAML config."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  simulation_mode: true
  log_level: DEBUG
  api_port: 9000

safety:
  stop_dist: 2.0
  sensitivity: 0.9
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_and_validate_config(temp_path)

            assert config.system.simulation_mode == True
            assert config.system.log_level == "DEBUG"
            assert config.system.api_port == 9000
            assert config.safety.stop_dist == 2.0
            assert config.safety.sensitivity == 0.9
        finally:
            os.unlink(temp_path)

    def test_load_partial_yaml(self):
        """Test loading YAML with only some fields."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  simulation_mode: true
# safety section missing - should use defaults
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_and_validate_config(temp_path)

            assert config.system.simulation_mode == True
            assert config.safety.stop_dist == 1.5  # Default
        finally:
            os.unlink(temp_path)


# =============================================================================
# Environment Override Tests
# =============================================================================

class TestEnvironmentOverrides:
    """Test suite for environment variable overrides."""

    def test_simulation_mode_override(self):
        """Test SIMULATION_MODE env override."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  simulation_mode: false
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"SIMULATION_MODE": "true"}):
                config = load_and_validate_config(temp_path)
                assert config.system.simulation_mode == True
        finally:
            os.unlink(temp_path)

    def test_log_level_override(self):
        """Test LOG_LEVEL env override."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  log_level: INFO
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
                config = load_and_validate_config(temp_path)
                assert config.system.log_level == "DEBUG"
        finally:
            os.unlink(temp_path)

    def test_api_port_override(self):
        """Test API_PORT env override."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  api_port: 8000
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"API_PORT": "3000"}):
                config = load_and_validate_config(temp_path)
                assert config.system.api_port == 3000
        finally:
            os.unlink(temp_path)

    def test_safety_sensitivity_override(self):
        """Test SAFETY_SENSITIVITY env override."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
safety:
  sensitivity: 0.8
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"SAFETY_SENSITIVITY": "0.95"}):
                config = load_and_validate_config(temp_path)
                assert config.safety.sensitivity == 0.95
        finally:
            os.unlink(temp_path)

    def test_invalid_env_override_ignored(self):
        """Test that invalid env values are ignored."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  api_port: 8000
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"API_PORT": "not_a_number"}):
                config = load_and_validate_config(temp_path)
                assert config.system.api_port == 8000  # Original value kept
        finally:
            os.unlink(temp_path)


# =============================================================================
# Meta AI Config Tests (from config.yaml)
# =============================================================================

class TestMetaAIConfig:
    """Test suite for Meta AI configuration in config.yaml."""

    @pytest.fixture
    def config_yaml(self):
        """Load config.yaml as dict."""
        with open("config/config.yaml", 'r') as f:
            return yaml.safe_load(f)

    def test_meta_ai_section_exists(self, config_yaml):
        """Test meta_ai section exists in config."""
        assert "meta_ai" in config_yaml

    def test_dinov3_config(self, config_yaml):
        """Test DINOv3 configuration."""
        meta_ai = config_yaml["meta_ai"]
        assert "dinov3" in meta_ai

        dinov3 = meta_ai["dinov3"]
        assert dinov3["enabled"] == True
        assert dinov3["model_size"] == "vit_large"
        assert dinov3["input_size"] == 518
        assert dinov3["use_fp16"] == True
        assert "cache_dir" in dinov3

    def test_sam3_config(self, config_yaml):
        """Test SAM3 configuration."""
        meta_ai = config_yaml["meta_ai"]
        assert "sam3" in meta_ai

        sam3 = meta_ai["sam3"]
        assert sam3["enabled"] == True
        assert sam3["model_size"] == "sam3_large"
        assert sam3["input_size"] == 1024
        assert sam3["max_objects"] == 10
        assert sam3["confidence_threshold"] == 0.5
        assert sam3["enable_tracking"] == True

    def test_vjepa2_config(self, config_yaml):
        """Test V-JEPA 2 configuration."""
        meta_ai = config_yaml["meta_ai"]
        assert "vjepa2" in meta_ai

        vjepa2 = meta_ai["vjepa2"]
        assert vjepa2["enabled"] == True
        assert vjepa2["model_size"] == "vjepa2_large"
        assert vjepa2["num_frames"] == 16
        assert vjepa2["prediction_horizon"] == 16
        assert vjepa2["enable_safety_prediction"] == True
        assert vjepa2["collision_threshold"] == 0.7
        assert vjepa2["emergency_stop_threshold"] == 0.9

    def test_privacy_config(self, config_yaml):
        """Test privacy (N2HE) configuration."""
        meta_ai = config_yaml["meta_ai"]
        assert "privacy" in meta_ai

        privacy = meta_ai["privacy"]
        assert privacy["enabled"] == True
        assert privacy["security_bits"] == 128
        assert privacy["lwe_dimension"] == 1024
        assert privacy["enable_homomorphic_routing"] == True

    def test_pipeline_timing_config(self, config_yaml):
        """Test Meta AI pipeline timing configuration."""
        meta_ai = config_yaml["meta_ai"]
        assert "pipeline" in meta_ai

        pipeline = meta_ai["pipeline"]
        assert pipeline["fusion_method"] == "concat"
        assert pipeline["safety_rate_hz"] == 1000.0
        assert pipeline["control_rate_hz"] == 100.0
        assert pipeline["learning_rate_hz"] == 10.0
        assert pipeline["cloud_rate_hz"] == 0.1

    def test_tflops_allocation_config(self, config_yaml):
        """Test Meta AI TFLOPS allocation."""
        meta_ai = config_yaml["meta_ai"]
        assert "tflops_allocation" in meta_ai

        tflops = meta_ai["tflops_allocation"]
        assert tflops["dinov3"] == 8.0
        assert tflops["sam3"] == 15.0
        assert tflops["vjepa2"] == 10.0
        assert tflops["total"] == 33.0

    def test_tflops_allocation_sums_correctly(self, config_yaml):
        """Test that Meta AI TFLOPS allocation sums correctly."""
        tflops = config_yaml["meta_ai"]["tflops_allocation"]

        model_sum = tflops["dinov3"] + tflops["sam3"] + tflops["vjepa2"]
        assert model_sum == tflops["total"]

    def test_total_tflops_within_budget(self, config_yaml):
        """Test total TFLOPS is within Jetson Orin budget."""
        total_budget = config_yaml["tflops_budget"]["total_fp16"]
        safe_utilization = config_yaml["tflops_budget"]["safe_utilization"]
        meta_ai_total = config_yaml["meta_ai"]["tflops_allocation"]["total"]

        # Get legacy allocations
        legacy_allocs = config_yaml["tflops_budget"]["allocations"]
        legacy_total = sum(v for k, v in legacy_allocs.items() if isinstance(v, (int, float)))

        total_used = legacy_total + meta_ai_total
        max_safe = total_budget * safe_utilization

        # Total usage should be within safe utilization limit
        assert total_used <= max_safe, f"Total {total_used} exceeds safe limit {max_safe}"


# =============================================================================
# Validation Error Tests
# =============================================================================

class TestConfigValidationErrors:
    """Test suite for configuration validation errors."""

    def test_invalid_yaml_syntax(self):
        """Test handling of invalid YAML syntax."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  simulation_mode: true
  log_level: DEBUG
  invalid_yaml: [unclosed bracket
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Should not crash, returns default config
            config = load_and_validate_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_invalid_value_type(self):
        """Test handling of invalid value types."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  api_port: "not_a_number"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Should return default config on validation error
            config = load_and_validate_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)


# =============================================================================
# Global Config Tests
# =============================================================================

class TestGlobalConfig:
    """Test suite for global config instance."""

    def test_global_config_imported(self):
        """Test that global config is importable."""
        from src.core.config_loader import config

        assert config is not None
        assert hasattr(config, 'system')
        assert hasattr(config, 'safety')

    def test_global_config_is_app_config(self):
        """Test global config is AppConfig instance."""
        from src.core.config_loader import config, AppConfig

        assert isinstance(config, AppConfig)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestConfigEdgeCases:
    """Test edge cases for configuration loading."""

    def test_empty_yaml_file(self):
        """Test handling of empty YAML file."""
        from src.core.config_loader import load_and_validate_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_and_validate_config(temp_path)
            # Should return default config
            assert config is not None
            assert config.system.simulation_mode == False
        finally:
            os.unlink(temp_path)

    def test_yaml_with_only_comments(self):
        """Test handling of YAML with only comments."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
# This is a comment
# Another comment
# No actual config
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_and_validate_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_extra_fields_ignored(self):
        """Test that extra/unknown fields are ignored."""
        from src.core.config_loader import load_and_validate_config

        yaml_content = """
system:
  simulation_mode: true
  unknown_field: some_value
  another_unknown: 123
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_and_validate_config(temp_path)
            assert config.system.simulation_mode == True
            # Extra fields should not cause errors
        finally:
            os.unlink(temp_path)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

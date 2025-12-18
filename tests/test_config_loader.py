#!/usr/bin/env python3
"""
Comprehensive Test Suite for Configuration Loader

Tests:
- Configuration model validation
- YAML file loading
- Environment variable overrides
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
        """Test TFLOPSAllocation default values (v0.4.0 with Meta AI)."""
        from src.core.config_loader import TFLOPSAllocation

        alloc = TFLOPSAllocation()

        # Updated defaults for Meta AI integration
        # trajectory_prediction removed - now handled by V-JEPA 2
        assert alloc.safety_detection == 15.0
        assert alloc.spatial_brain == 3.0
        assert alloc.navigation_detection == 30.0  # Reduced: SAM3 handles segmentation
        assert alloc.depth_estimation == 5.0
        assert alloc.il_training == 9.0
        assert alloc.moai_compression == 3.0
        assert alloc.fhe_encryption == 1.0
        assert alloc.pi0_vla == 10.0
        assert alloc.full_perception == 15.0  # Reduced: DINOv3 handles features
        assert alloc.anomaly_detection == 3.0

    def test_tflops_budget_defaults(self):
        """Test TFLOPSBudget defaults."""
        from src.core.config_loader import TFLOPSBudget

        budget = TFLOPSBudget()

        assert budget.total_fp16 == 137.0
        assert budget.safe_utilization == 0.85
        assert budget.burst_utilization == 0.95
        assert budget.allocations is not None

    def test_tflops_budget_total_allocation(self):
        """Test that TFLOPS allocations sum correctly (v0.4.0 with Meta AI)."""
        from src.core.config_loader import TFLOPSBudget, TFLOPSAllocation

        budget = TFLOPSBudget()
        alloc = budget.allocations

        # Calculate total allocation (trajectory_prediction removed - handled by V-JEPA 2)
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

        # Total should be 94.0 TFLOPS (core allocations)
        # Meta AI models (DINOv3, SAM3, V-JEPA 2) add ~33 TFLOPS
        assert total == 94.0

        # Core utilization is 68.6% - leaves room for Meta AI models
        utilization = total / budget.total_fp16
        assert 0.65 < utilization < 0.75

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
# TFLOPS Config Tests (from config.yaml)
# =============================================================================

class TestTFLOPSConfig:
    """Test suite for TFLOPS configuration in config.yaml."""

    @pytest.fixture
    def config_yaml(self):
        """Load config.yaml as dict."""
        with open("config/config.yaml", 'r') as f:
            return yaml.safe_load(f)

    def test_tflops_budget_exists(self, config_yaml):
        """Test tflops_budget section exists in config."""
        assert "tflops_budget" in config_yaml

    def test_tflops_budget_values(self, config_yaml):
        """Test TFLOPS budget values."""
        budget = config_yaml["tflops_budget"]

        assert budget["total_fp16"] == 137.0
        assert budget["safe_utilization"] == 0.85
        assert budget["burst_utilization"] == 0.95

    def test_tflops_allocations_exist(self, config_yaml):
        """Test allocations exist in TFLOPS budget."""
        budget = config_yaml["tflops_budget"]
        assert "allocations" in budget

        allocs = budget["allocations"]
        # Updated for Meta AI integration (v0.4.0)
        # trajectory_prediction removed - now handled by V-JEPA 2
        expected_keys = [
            "safety_detection", "spatial_brain", "navigation_detection",
            "depth_estimation", "il_training",
            "moai_compression", "fhe_encryption", "pi0_vla",
            "full_perception", "anomaly_detection"
        ]

        for key in expected_keys:
            assert key in allocs, f"Missing allocation: {key}"

    def test_tflops_allocation_values(self, config_yaml):
        """Test TFLOPS allocation values match expected (v0.4.0 with Meta AI)."""
        allocs = config_yaml["tflops_budget"]["allocations"]

        # Updated values for Meta AI integration
        assert allocs["safety_detection"] == 15.0
        assert allocs["spatial_brain"] == 3.0
        assert allocs["navigation_detection"] == 30.0  # Reduced: SAM3 handles segmentation
        assert allocs["depth_estimation"] == 5.0
        assert allocs["il_training"] == 9.0
        assert allocs["moai_compression"] == 3.0
        assert allocs["fhe_encryption"] == 1.0
        assert allocs["pi0_vla"] == 10.0
        assert allocs["full_perception"] == 15.0  # Reduced: DINOv3 handles features
        assert allocs["anomaly_detection"] == 3.0

    def test_total_tflops_within_budget(self, config_yaml):
        """Test total TFLOPS is within Jetson Orin budget."""
        budget = config_yaml["tflops_budget"]
        allocs = budget["allocations"]

        total_used = sum(allocs.values())
        max_budget = budget["total_fp16"]

        # Total should be within 100% of budget
        assert total_used <= max_budget, f"Total {total_used} exceeds budget {max_budget}"


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

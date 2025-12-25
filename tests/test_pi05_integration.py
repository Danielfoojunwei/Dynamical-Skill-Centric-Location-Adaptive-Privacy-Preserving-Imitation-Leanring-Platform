"""
Pi0.5 Integration Tests

Tests for the Physical Intelligence Pi0.5 VLA model integration.

Test Categories:
1. Installation verification
2. Model loading
3. Inference pipeline
4. Error handling

Run with:
    pytest tests/test_pi05_integration.py -v
    pytest tests/test_pi05_integration.py -v -k "installation"  # Just installation tests
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly from pi05_model to avoid legacy model import cascade
# This allows testing without PyTorch installed
from src.spatial_intelligence.pi0.pi05_model import (
    Pi05Model,
    Pi05Config,
    Pi05Variant,
    Pi05Observation,
    Pi05Result,
    load_pi05,
    list_variants,
    check_installation,
    HAS_OPENPI,
)


class TestPi05Installation:
    """Tests for Pi0.5 installation and availability."""

    def test_pi05_module_import(self):
        """Test that Pi0.5 module can be imported."""
        # Using global imports from pi05_model
        # These should always be importable
        assert Pi05Model is not None
        assert Pi05Config is not None
        assert Pi05Variant is not None

    def test_check_installation_function(self):
        """Test the installation check function."""
        status = check_installation()

        assert isinstance(status, dict)
        assert "openpi_installed" in status
        assert "torch_installed" in status
        assert "cuda_available" in status

        # Verify consistency
        assert status["openpi_installed"] == HAS_OPENPI

    def test_list_variants_function(self):
        """Test listing available Pi0.5 variants."""
        variants = list_variants()

        assert isinstance(variants, list)
        assert len(variants) > 0
        assert "pi05_base" in variants
        assert "pi0_base" in variants

    def test_pi05_variant_enum(self):
        """Test Pi0.5 variant enumeration."""
        # Check expected variants exist
        assert Pi05Variant.PI05_BASE.value == "pi05_base"
        assert Pi05Variant.PI0_BASE.value == "pi0_base"
        assert Pi05Variant.PI05_LIBERO.value == "pi05_libero"


class TestPi05Config:
    """Tests for Pi0.5 configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = Pi05Config()

        assert config.variant == Pi05Variant.PI05_BASE
        assert config.device == "cuda"
        assert config.action_horizon == 16
        assert config.action_dim == 7

    def test_custom_config(self):
        """Test custom configuration."""
        config = Pi05Config(
            variant=Pi05Variant.PI05_LIBERO,
            device="cpu",
            action_horizon=32,
        )

        assert config.variant == Pi05Variant.PI05_LIBERO
        assert config.device == "cpu"
        assert config.action_horizon == 32


class TestPi05Observation:
    """Tests for Pi0.5 observation handling."""

    def test_observation_creation(self):
        """Test creating an observation."""
        # Create mock image data
        images = np.random.rand(3, 224, 224, 3).astype(np.float32)

        obs = Pi05Observation(
            images=images,
            instruction="pick up the red cup"
        )

        assert obs.images is not None
        assert obs.instruction == "pick up the red cup"
        assert obs.proprio is None  # Optional

    def test_observation_with_proprio(self):
        """Test observation with proprioceptive state."""
        images = np.random.rand(224, 224, 3).astype(np.float32)
        proprio = np.random.rand(7).astype(np.float32)

        obs = Pi05Observation(
            images=images,
            instruction="move to the table",
            proprio=proprio,
            gripper=0.5
        )

        assert obs.proprio is not None
        assert obs.gripper == 0.5


class TestPi05ModelCreation:
    """Tests for Pi0.5 model creation (without loading weights)."""

    def test_model_creation_without_openpi(self):
        """Test model creation behavior when openpi is not installed."""
        if HAS_OPENPI:
            pytest.skip("openpi is installed, testing without it")

        # If openpi not installed, should raise on creation
        with pytest.raises(ImportError):
            model = Pi05Model()

    @pytest.mark.skipif(
        not HAS_OPENPI,
        reason="openpi not installed"
    )
    def test_model_creation_with_openpi(self):
        """Test model creation when openpi is installed."""
        config = Pi05Config()
        model = Pi05Model(config)

        assert model is not None
        assert not model.is_loaded  # Should not be loaded yet

    @pytest.mark.skipif(
        not HAS_OPENPI,
        reason="openpi not installed"
    )
    def test_factory_methods(self):
        """Test factory methods for model creation."""
        # Test create method
        model = Pi05Model.create("pi05_base")
        assert model is not None

        # Test for_jetson_thor
        thor_model = Pi05Model.for_jetson_thor()
        assert thor_model is not None


@pytest.mark.skipif(
    not HAS_OPENPI,
    reason="openpi not installed - skipping inference tests"
)
class TestPi05Inference:
    """Tests for Pi0.5 inference (requires openpi and model weights)."""

    @pytest.fixture
    def loaded_model(self):
        """Fixture to provide a loaded model."""
        config = Pi05Config(
            variant=Pi05Variant.PI05_BASE,
            device="cuda" if self._cuda_available() else "cpu"
        )
        model = Pi05Model(config)

        try:
            model.load()
            yield model
        finally:
            model.unload()

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def test_inference_basic(self, loaded_model):
        """Test basic inference."""
        # Create mock observation
        images = np.random.rand(224, 224, 3).astype(np.float32)
        obs = Pi05Observation(
            images=images,
            instruction="pick up the cup"
        )

        result = loaded_model.infer(obs)

        assert result is not None
        assert result.actions is not None
        assert result.inference_time_ms is not None
        assert result.inference_time_ms > 0

    def test_inference_with_proprio(self, loaded_model):
        """Test inference with proprioceptive state."""
        images = np.random.rand(3, 224, 224, 3).astype(np.float32)
        proprio = np.random.rand(7).astype(np.float32)

        obs = Pi05Observation(
            images=images,
            instruction="move to target",
            proprio=proprio
        )

        result = loaded_model.infer(obs)

        assert result is not None
        assert result.actions is not None

    def test_model_unload(self, loaded_model):
        """Test model unloading."""
        assert loaded_model.is_loaded

        loaded_model.unload()

        assert not loaded_model.is_loaded


class TestPi05ErrorHandling:
    """Tests for error handling in Pi0.5."""

    @pytest.mark.skipif(
        not HAS_OPENPI,
        reason="openpi not installed"
    )
    def test_invalid_variant(self):
        """Test handling of invalid variant."""
        # Should handle gracefully and fall back to default
        model = Pi05Model.create("invalid_variant_name")
        assert model is not None

    def test_observation_missing_instruction(self):
        """Test observation without instruction."""
        images = np.random.rand(224, 224, 3).astype(np.float32)

        # This should work - instruction is required but can be empty
        obs = Pi05Observation(
            images=images,
            instruction=""
        )
        assert obs.instruction == ""


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with old API."""

    def test_old_import_names(self):
        """Test that old import names still work."""
        # Import from pi0 module to check backwards compatibility
        from src.spatial_intelligence.pi0 import (
            Pi05Backend,  # Old name
            create_pi05_for_thor,  # Old function
        )

        # These should be aliased to new names
        assert Pi05Backend == Pi05Model


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

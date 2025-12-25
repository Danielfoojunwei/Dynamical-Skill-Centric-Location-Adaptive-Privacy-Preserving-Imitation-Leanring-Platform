"""
Meta AI Models Integration Tests

Tests for DINOv3, SAM3, and V-JEPA 2 integrations.

Test Categories:
1. Installation verification
2. Model loading (mock and real)
3. Inference pipeline
4. Error handling

Run with:
    pytest tests/test_meta_ai_integration.py -v
    pytest tests/test_meta_ai_integration.py -v -k "dinov3"  # Just DINOv3 tests
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Direct imports from modules to avoid import cascade issues
from src.meta_ai.dinov3 import DINOv3Encoder, DINOv3Config, DINOv3ModelSize, DINOv3Features
from src.meta_ai.sam3 import SAM3Segmenter, SAM3Config, SAM3ModelSize, SegmentationResult
from src.meta_ai.vjepa2 import VJEPA2WorldModel, VJEPA2Config, VJEPA2ModelSize, WorldState


# =============================================================================
# DINOv3 Tests
# =============================================================================

class TestDINOv3Installation:
    """Tests for DINOv3 installation and availability."""

    def test_dinov3_module_import(self):
        """Test that DINOv3 module can be imported."""
        assert DINOv3Encoder is not None
        assert DINOv3Config is not None
        assert DINOv3ModelSize is not None

    def test_dinov3_model_sizes(self):
        """Test DINOv3 model size enumeration."""
        # Check expected sizes exist
        assert DINOv3ModelSize.SMALL is not None
        assert DINOv3ModelSize.BASE is not None
        assert DINOv3ModelSize.LARGE is not None
        assert DINOv3ModelSize.HUGE_PLUS is not None
        assert DINOv3ModelSize.GIANT_7B is not None

    def test_dinov3_encoder_creation(self):
        """Test creating DINOv3 encoder instance."""
        config = DINOv3Config(model_size=DINOv3ModelSize.BASE)
        encoder = DINOv3Encoder(config=config)

        assert encoder is not None
        assert encoder.config.model_size == DINOv3ModelSize.BASE


class TestDINOv3Inference:
    """Tests for DINOv3 inference."""

    def test_encode_mock_mode(self):
        """Test encoding in mock mode (no real model)."""
        config = DINOv3Config(model_size=DINOv3ModelSize.BASE)
        encoder = DINOv3Encoder(config=config)

        # Load model (will use mock if PyTorch/HuggingFace not available)
        encoder.load_model()

        # Create mock image
        image = np.random.rand(224, 224, 3).astype(np.float32)

        # Encode
        features = encoder.encode(image)

        assert features is not None
        assert isinstance(features, DINOv3Features)
        assert features.global_features is not None

    @pytest.mark.skipif(
        True,  # Skip by default - requires HuggingFace model download
        reason="Requires HuggingFace model download"
    )
    def test_encode_real_model(self):
        """Test encoding with real DINOv3 model."""
        config = DINOv3Config(model_size=DINOv3ModelSize.SMALL)
        encoder = DINOv3Encoder(config=config)
        encoder.load_model()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        features = encoder.encode(image)

        assert features is not None
        assert len(features.global_features.shape) >= 1


# =============================================================================
# SAM3 Tests
# =============================================================================

class TestSAM3Installation:
    """Tests for SAM3 installation and availability."""

    def test_sam3_module_import(self):
        """Test that SAM3 module can be imported."""
        assert SAM3Segmenter is not None
        assert SAM3Config is not None
        assert SAM3ModelSize is not None

    def test_sam3_model_sizes(self):
        """Test SAM3 model size enumeration."""
        assert SAM3ModelSize.TINY is not None
        assert SAM3ModelSize.SMALL is not None
        assert SAM3ModelSize.BASE is not None
        assert SAM3ModelSize.LARGE is not None

    def test_sam3_segmenter_creation(self):
        """Test creating SAM3 segmenter instance."""
        config = SAM3Config(model_size=SAM3ModelSize.BASE)
        segmenter = SAM3Segmenter(config=config)

        assert segmenter is not None


class TestSAM3Inference:
    """Tests for SAM3 inference."""

    def test_segment_mock_mode(self):
        """Test segmentation in mock mode."""
        config = SAM3Config(model_size=SAM3ModelSize.BASE)
        segmenter = SAM3Segmenter(config=config)
        segmenter.load_model()

        # Create mock image
        image = np.random.rand(480, 640, 3).astype(np.float32)

        # Segment with text prompt
        result = segmenter.segment_text(image, "red cup")

        assert result is not None
        assert isinstance(result, SegmentationResult)

    def test_segment_with_points(self):
        """Test segmentation with point prompts."""
        config = SAM3Config(model_size=SAM3ModelSize.BASE)
        segmenter = SAM3Segmenter(config=config)
        segmenter.load_model()

        image = np.random.rand(480, 640, 3).astype(np.float32)
        point = (320, 240)  # Center point

        result = segmenter.segment_point(image, point)

        assert result is not None


# =============================================================================
# V-JEPA 2 Tests
# =============================================================================

class TestVJEPA2Installation:
    """Tests for V-JEPA 2 installation and availability."""

    def test_vjepa2_module_import(self):
        """Test that V-JEPA 2 module can be imported."""
        assert VJEPA2WorldModel is not None
        assert VJEPA2Config is not None
        assert VJEPA2ModelSize is not None

    def test_vjepa2_model_sizes(self):
        """Test V-JEPA 2 model size enumeration."""
        assert VJEPA2ModelSize.LARGE is not None
        assert VJEPA2ModelSize.HUGE is not None
        assert VJEPA2ModelSize.GIANT is not None

    def test_vjepa2_world_model_creation(self):
        """Test creating V-JEPA 2 world model instance."""
        config = VJEPA2Config(model_size=VJEPA2ModelSize.LARGE)
        world_model = VJEPA2WorldModel(config=config)

        assert world_model is not None


class TestVJEPA2Inference:
    """Tests for V-JEPA 2 inference."""

    def test_encode_frame_mock_mode(self):
        """Test frame encoding in mock mode."""
        config = VJEPA2Config(model_size=VJEPA2ModelSize.LARGE)
        world_model = VJEPA2WorldModel(config=config)
        world_model.load_model()

        # Create mock frame
        frame = np.random.rand(256, 256, 3).astype(np.float32)

        # Encode
        embedding = world_model.encode_frame(frame)

        assert embedding is not None
        assert embedding.shape == (config.embed_dim,)

    def test_predict_mock_mode(self):
        """Test world prediction in mock mode."""
        config = VJEPA2Config(
            model_size=VJEPA2ModelSize.LARGE,
            prediction_horizon=8
        )
        world_model = VJEPA2WorldModel(config=config)
        world_model.load_model()

        # Add frames to buffer
        for _ in range(4):
            frame = np.random.rand(256, 256, 3).astype(np.float32)
            world_model.add_frame(frame)

        # Predict
        prediction = world_model.predict()

        assert prediction is not None
        assert prediction.future_embeddings is not None


# =============================================================================
# Integration Tests - All Models Together
# =============================================================================

class TestMetaAIIntegration:
    """Integration tests using all Meta AI models together."""

    def test_all_modules_import(self):
        """Test that all Meta AI modules can be imported together."""
        assert DINOv3Encoder is not None
        assert SAM3Segmenter is not None
        assert VJEPA2WorldModel is not None

    def test_perception_pipeline_mock(self):
        """Test a mock perception pipeline using all models."""
        # Create all models
        dinov3_config = DINOv3Config(model_size=DINOv3ModelSize.BASE)
        sam3_config = SAM3Config(model_size=SAM3ModelSize.BASE)
        vjepa2_config = VJEPA2Config(model_size=VJEPA2ModelSize.LARGE)

        dinov3_encoder = DINOv3Encoder(config=dinov3_config)
        sam3_segmenter = SAM3Segmenter(config=sam3_config)
        vjepa2_world_model = VJEPA2WorldModel(config=vjepa2_config)

        # Load models (mock mode)
        dinov3_encoder.load_model()
        sam3_segmenter.load_model()
        vjepa2_world_model.load_model()

        # Create mock image
        image = np.random.rand(480, 640, 3).astype(np.float32)

        # Run perception pipeline
        # 1. Scene features with DINOv3
        scene_features = dinov3_encoder.encode(image)
        assert scene_features is not None

        # 2. Object segmentation with SAM3
        masks = sam3_segmenter.segment_text(image, "object")
        assert masks is not None

        # 3. Temporal features with V-JEPA 2
        frame_embedding = vjepa2_world_model.encode_frame(image)
        assert frame_embedding is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestMetaAIErrorHandling:
    """Tests for error handling in Meta AI models."""

    def test_dinov3_handles_various_image_shapes(self):
        """Test DINOv3 handling of various image shapes."""
        config = DINOv3Config(model_size=DINOv3ModelSize.BASE)
        encoder = DINOv3Encoder(config=config)
        encoder.load_model()

        # Test with different valid image shapes
        shapes = [(224, 224, 3), (480, 640, 3), (512, 512, 3)]
        for shape in shapes:
            image = np.random.rand(*shape).astype(np.float32)
            features = encoder.encode(image)
            assert features is not None

    def test_sam3_empty_prompt(self):
        """Test SAM3 with empty text prompt."""
        config = SAM3Config(model_size=SAM3ModelSize.BASE)
        segmenter = SAM3Segmenter(config=config)
        segmenter.load_model()

        image = np.random.rand(480, 640, 3).astype(np.float32)

        # Empty prompt should be handled
        result = segmenter.segment_text(image, "")

        # Should return something (possibly empty masks)
        assert result is not None


# =============================================================================
# Availability Check Tests
# =============================================================================

class TestMetaAIAvailability:
    """Tests to check what's actually available."""

    def test_check_torch_availability(self):
        """Check if PyTorch is available."""
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        print(f"\nPyTorch available: {has_torch}")
        assert True  # Just informational

    def test_check_transformers_availability(self):
        """Check if HuggingFace Transformers is available."""
        try:
            from transformers import AutoModel
            has_transformers = True
        except ImportError:
            has_transformers = False

        print(f"\nHuggingFace Transformers available: {has_transformers}")
        assert True  # Just informational

    def test_print_availability_summary(self):
        """Print summary of what's available."""
        print("\n" + "=" * 60)
        print("META AI MODELS AVAILABILITY SUMMARY")
        print("=" * 60)

        # Check each model
        models = {
            "DINOv3": (DINOv3Encoder, DINOv3Config),
            "SAM3": (SAM3Segmenter, SAM3Config),
            "V-JEPA 2": (VJEPA2WorldModel, VJEPA2Config),
        }

        for name, (cls, cfg_cls) in models.items():
            try:
                config = cfg_cls()
                instance = cls(config=config)
                instance.load_model()
                backend = instance.stats.get("backend", "unknown")
                print(f"  {name}: ✓ Available (backend: {backend})")
            except Exception as e:
                print(f"  {name}: ✗ Error - {e}")

        print("=" * 60)
        assert True


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

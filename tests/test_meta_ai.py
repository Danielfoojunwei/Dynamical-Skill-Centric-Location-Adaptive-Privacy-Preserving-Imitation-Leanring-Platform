#!/usr/bin/env python3
"""
Comprehensive Test Suite for Meta AI Integration

Tests all Meta AI models:
- DINOv3 Vision Encoder
- SAM3 Segmentation
- V-JEPA 2 World Model
- Privacy Wrapper
- Unified Perception Pipeline
"""

import sys
import os
import pytest
import numpy as np
import time
from typing import Dict, Any
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# DINOv3 Tests
# =============================================================================

class TestDINOv3:
    """Test suite for DINOv3 Vision Encoder."""

    @pytest.fixture
    def dinov3_config(self):
        """Create DINOv3 config fixture."""
        from src.meta_ai.dinov3 import DINOv3Config, DINOv3ModelSize
        return DINOv3Config(
            model_size=DINOv3ModelSize.LARGE,
            input_size=518,
            use_fp16=False,  # CPU testing
            device="cpu"
        )

    @pytest.fixture
    def dinov3_encoder(self, dinov3_config):
        """Create DINOv3 encoder fixture."""
        from src.meta_ai.dinov3 import DINOv3Encoder
        encoder = DINOv3Encoder(dinov3_config)
        encoder.load_model()
        return encoder

    @pytest.fixture
    def test_image(self):
        """Create test image fixture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_dinov3_config_creation(self):
        """Test DINOv3Config creation with various model sizes."""
        from src.meta_ai.dinov3 import DINOv3Config, DINOv3ModelSize

        for model_size in DINOv3ModelSize:
            config = DINOv3Config(model_size=model_size)
            assert config.model_size == model_size
            assert config.input_size == 518  # Default
            assert config.patch_size == 14

    def test_dinov3_config_properties(self):
        """Test DINOv3Config computed properties."""
        from src.meta_ai.dinov3 import DINOv3Config

        config = DINOv3Config(input_size=518, patch_size=14)
        assert config.num_patches == (518 // 14) ** 2
        assert config.feature_map_size == 518 // 14

    def test_dinov3_encoder_initialization(self, dinov3_config):
        """Test DINOv3Encoder initialization."""
        from src.meta_ai.dinov3 import DINOv3Encoder

        encoder = DINOv3Encoder(dinov3_config)
        assert encoder.config == dinov3_config
        assert encoder._is_loaded == False
        assert encoder.model is None

    def test_dinov3_model_loading(self, dinov3_config):
        """Test DINOv3 model loading."""
        from src.meta_ai.dinov3 import DINOv3Encoder

        encoder = DINOv3Encoder(dinov3_config)
        success = encoder.load_model()

        assert success == True
        assert encoder._is_loaded == True

    def test_dinov3_encode_single_image(self, dinov3_encoder, test_image):
        """Test encoding a single image."""
        features = dinov3_encoder.encode(test_image)

        assert features is not None
        assert features.global_features is not None
        assert features.global_features.shape[0] == 1
        assert features.global_features.shape[1] == dinov3_encoder.config.output_dim
        assert features.model_size == "vit_large"
        assert features.inference_time_ms >= 0

    def test_dinov3_encode_with_dense_features(self, dinov3_encoder, test_image):
        """Test encoding with dense features."""
        features = dinov3_encoder.encode(test_image, return_dense=True)

        assert features.global_features is not None
        assert features.dense_features is not None

        expected_size = dinov3_encoder.config.feature_map_size
        assert features.dense_features.shape[1] == expected_size
        assert features.dense_features.shape[2] == expected_size

    def test_dinov3_batch_encoding(self, dinov3_encoder, test_image):
        """Test batch encoding multiple images."""
        images = [test_image] * 4
        features = dinov3_encoder.encode_batch(images)

        assert features.global_features.shape[0] == 4
        assert features.global_features.shape[1] == dinov3_encoder.config.output_dim

    def test_dinov3_get_dense_features(self, dinov3_encoder, test_image):
        """Test getting dense features directly."""
        dense = dinov3_encoder.get_dense_features(test_image)

        assert dense is not None
        assert len(dense.shape) == 3  # H, W, D

    def test_dinov3_compute_similarity(self, dinov3_encoder, test_image):
        """Test feature similarity computation."""
        features1 = dinov3_encoder.encode(test_image)
        features2 = dinov3_encoder.encode(test_image)  # Same image

        similarity = dinov3_encoder.compute_similarity(
            features1.global_features,
            features2.global_features
        )

        # Same image should have high similarity
        assert similarity > 0.9

    def test_dinov3_statistics(self, dinov3_encoder, test_image):
        """Test statistics tracking."""
        # Process some images
        for _ in range(3):
            dinov3_encoder.encode(test_image)

        stats = dinov3_encoder.get_statistics()

        assert stats["images_processed"] == 3
        assert stats["total_inference_time_ms"] > 0
        assert stats["is_loaded"] == True

    def test_dinov3_different_input_sizes(self, dinov3_config):
        """Test encoder with different image sizes."""
        from src.meta_ai.dinov3 import DINOv3Encoder

        encoder = DINOv3Encoder(dinov3_config)
        encoder.load_model()

        # Various input sizes should work
        sizes = [(224, 224, 3), (480, 640, 3), (720, 1280, 3)]

        for size in sizes:
            image = np.random.randint(0, 255, size, dtype=np.uint8)
            features = encoder.encode(image)
            assert features.global_features is not None


# =============================================================================
# SAM3 Tests
# =============================================================================

class TestSAM3:
    """Test suite for SAM3 Segmentation."""

    @pytest.fixture
    def sam3_config(self):
        """Create SAM3 config fixture."""
        from src.meta_ai.sam3 import SAM3Config, SAM3ModelSize
        return SAM3Config(
            model_size=SAM3ModelSize.LARGE,
            input_size=1024,
            max_objects=10,
            confidence_threshold=0.5,
            enable_tracking=True,
            device="cpu"
        )

    @pytest.fixture
    def sam3_segmenter(self, sam3_config):
        """Create SAM3 segmenter fixture."""
        from src.meta_ai.sam3 import SAM3Segmenter
        segmenter = SAM3Segmenter(sam3_config)
        segmenter.load_model()
        return segmenter

    @pytest.fixture
    def test_image(self):
        """Create test image fixture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_sam3_config_creation(self):
        """Test SAM3Config creation."""
        from src.meta_ai.sam3 import SAM3Config, SAM3ModelSize

        for model_size in SAM3ModelSize:
            config = SAM3Config(model_size=model_size)
            assert config.model_size == model_size

    def test_sam3_segmenter_initialization(self, sam3_config):
        """Test SAM3Segmenter initialization."""
        from src.meta_ai.sam3 import SAM3Segmenter

        segmenter = SAM3Segmenter(sam3_config)
        assert segmenter.config == sam3_config
        assert segmenter._is_loaded == False

    def test_sam3_model_loading(self, sam3_config):
        """Test SAM3 model loading."""
        from src.meta_ai.sam3 import SAM3Segmenter

        segmenter = SAM3Segmenter(sam3_config)
        success = segmenter.load_model()

        assert success == True
        assert segmenter._is_loaded == True

    def test_sam3_text_segmentation(self, sam3_segmenter, test_image):
        """Test text-prompted segmentation."""
        result = sam3_segmenter.segment_text(test_image, "the red cup")

        assert result is not None
        assert hasattr(result, 'masks')
        assert hasattr(result, 'image_size')
        assert result.prompt_text == "the red cup"

    def test_sam3_point_segmentation(self, sam3_segmenter, test_image):
        """Test point-prompted segmentation."""
        result = sam3_segmenter.segment_point(test_image, point=(320, 240))

        assert result is not None
        assert len(result.masks) > 0
        assert result.masks[0].mask.shape == test_image.shape[:2]

    def test_sam3_box_segmentation(self, sam3_segmenter, test_image):
        """Test box-prompted segmentation."""
        result = sam3_segmenter.segment_box(test_image, box=(100, 100, 300, 300))

        assert result is not None
        assert len(result.masks) > 0

        # Verify mask is within box region
        mask = result.masks[0].mask
        assert mask[150, 200] == 1  # Point inside box

    def test_sam3_auto_segmentation(self, sam3_segmenter, test_image):
        """Test automatic segmentation of all objects."""
        result = sam3_segmenter.segment_all_objects(test_image)

        assert result is not None
        assert len(result.masks) >= 1
        assert len(result.masks) <= sam3_segmenter.config.max_objects

    def test_sam3_get_manipulation_target(self, sam3_segmenter, test_image):
        """Test getting manipulation target."""
        target = sam3_segmenter.get_manipulation_target(test_image, "screwdriver")

        assert target is not None
        assert 'mask' in target
        assert 'centroid' in target
        assert 'confidence' in target

    def test_sam3_combined_mask(self, sam3_segmenter, test_image):
        """Test combined mask generation."""
        result = sam3_segmenter.segment_all_objects(test_image)
        combined = result.get_combined_mask()

        assert combined.shape == test_image.shape[:2]
        assert combined.dtype == np.int32

    def test_sam3_object_tracking(self, sam3_segmenter, test_image):
        """Test video object tracking."""
        # Create initial mask
        initial_result = sam3_segmenter.segment_text(test_image, "object")

        if initial_result.masks:
            initial_mask = initial_result.masks[0].mask

            # Track in subsequent frame
            result = sam3_segmenter.track_object(
                test_image,
                initial_mask=initial_mask,
                track_id=0
            )

            assert result is not None

    def test_sam3_statistics(self, sam3_segmenter, test_image):
        """Test statistics tracking."""
        sam3_segmenter.segment_text(test_image, "cup")
        sam3_segmenter.segment_point(test_image, (100, 100))

        stats = sam3_segmenter.get_statistics()

        assert stats["images_processed"] >= 2
        assert stats["text_queries"] >= 1
        assert stats["is_loaded"] == True


# =============================================================================
# V-JEPA 2 Tests
# =============================================================================

class TestVJEPA2:
    """Test suite for V-JEPA 2 World Model."""

    @pytest.fixture
    def vjepa2_config(self):
        """Create V-JEPA 2 config fixture."""
        from src.meta_ai.vjepa2 import VJEPA2Config, VJEPA2ModelSize
        return VJEPA2Config(
            model_size=VJEPA2ModelSize.LARGE,
            num_frames=8,
            prediction_horizon=8,
            enable_safety_prediction=True,
            collision_threshold=0.7,
            device="cpu"
        )

    @pytest.fixture
    def vjepa2_model(self, vjepa2_config):
        """Create V-JEPA 2 model fixture."""
        from src.meta_ai.vjepa2 import VJEPA2WorldModel
        model = VJEPA2WorldModel(vjepa2_config)
        model.load_model()
        return model

    @pytest.fixture
    def test_frame(self):
        """Create test frame fixture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_vjepa2_config_creation(self):
        """Test VJEPA2Config creation."""
        from src.meta_ai.vjepa2 import VJEPA2Config, VJEPA2ModelSize

        for model_size in VJEPA2ModelSize:
            config = VJEPA2Config(model_size=model_size)
            assert config.model_size == model_size

    def test_vjepa2_config_properties(self):
        """Test VJEPA2Config computed properties."""
        from src.meta_ai.vjepa2 import VJEPA2Config

        config = VJEPA2Config(robot_dof=23, gripper_dim=1)
        assert config.total_action_dim == 24

    def test_vjepa2_model_initialization(self, vjepa2_config):
        """Test V-JEPA 2 model initialization."""
        from src.meta_ai.vjepa2 import VJEPA2WorldModel

        model = VJEPA2WorldModel(vjepa2_config)
        assert model.config == vjepa2_config
        assert model._is_loaded == False

    def test_vjepa2_model_loading(self, vjepa2_config):
        """Test V-JEPA 2 model loading."""
        from src.meta_ai.vjepa2 import VJEPA2WorldModel

        model = VJEPA2WorldModel(vjepa2_config)
        success = model.load_model()

        assert success == True
        assert model._is_loaded == True

    def test_vjepa2_encode_frame(self, vjepa2_model, test_frame):
        """Test single frame encoding."""
        embedding = vjepa2_model.encode_frame(test_frame)

        assert embedding is not None
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == vjepa2_model.config.embed_dim

    def test_vjepa2_add_frame(self, vjepa2_model, test_frame):
        """Test adding frames to buffer."""
        for i in range(5):
            vjepa2_model.add_frame(test_frame)

        assert len(vjepa2_model._frame_buffer) == 5
        assert len(vjepa2_model._embedding_buffer) == 5

    def test_vjepa2_predict(self, vjepa2_model, test_frame):
        """Test future state prediction."""
        # Add context frames
        for _ in range(4):
            vjepa2_model.add_frame(test_frame)

        prediction = vjepa2_model.predict()

        assert prediction is not None
        assert prediction.future_embeddings is not None
        assert prediction.horizon == vjepa2_model.config.prediction_horizon
        assert prediction.inference_time_ms >= 0

    def test_vjepa2_safety_prediction(self, vjepa2_model, test_frame):
        """Test collision probability prediction."""
        for _ in range(4):
            vjepa2_model.add_frame(test_frame)

        prediction = vjepa2_model.predict()

        assert prediction.collision_probabilities is not None
        assert len(prediction.collision_probabilities) == prediction.horizon
        assert all(0 <= p <= 1 for p in prediction.collision_probabilities)

    def test_vjepa2_plan_to_goal(self, vjepa2_model, test_frame):
        """Test action planning to goal state."""
        from src.meta_ai.vjepa2 import WorldState

        # Create current and goal states
        current_embedding = vjepa2_model.encode_frame(test_frame)
        goal_embedding = np.random.randn(vjepa2_model.config.embed_dim).astype(np.float32)

        current_state = WorldState(frame_embedding=current_embedding)
        goal_state = WorldState(frame_embedding=goal_embedding)

        plan = vjepa2_model.plan_to_goal(current_state, goal_state)

        assert plan is not None
        assert plan.actions is not None
        assert plan.actions.shape[0] == vjepa2_model.config.prediction_horizon
        assert plan.actions.shape[1] == vjepa2_model.config.total_action_dim
        assert 0 <= plan.success_probability <= 1

    def test_vjepa2_is_action_safe(self, vjepa2_model, test_frame):
        """Test action safety check."""
        from src.meta_ai.vjepa2 import WorldState

        current_embedding = vjepa2_model.encode_frame(test_frame)
        current_state = WorldState(frame_embedding=current_embedding)

        proposed_action = np.random.randn(vjepa2_model.config.total_action_dim).astype(np.float32)

        is_safe, collision_prob = vjepa2_model.is_action_safe(current_state, proposed_action)

        assert isinstance(is_safe, bool)
        assert 0 <= collision_prob <= 1

    def test_vjepa2_clear_buffer(self, vjepa2_model, test_frame):
        """Test clearing frame buffer."""
        for _ in range(5):
            vjepa2_model.add_frame(test_frame)

        vjepa2_model.clear_buffer()

        assert len(vjepa2_model._frame_buffer) == 0
        assert len(vjepa2_model._embedding_buffer) == 0

    def test_vjepa2_statistics(self, vjepa2_model, test_frame):
        """Test statistics tracking."""
        for _ in range(3):
            vjepa2_model.add_frame(test_frame)

        vjepa2_model.predict()

        stats = vjepa2_model.get_statistics()

        assert stats["predictions_made"] >= 1
        assert stats["is_loaded"] == True
        assert stats["buffer_frames"] == 3


# =============================================================================
# Unified Perception Pipeline Tests
# =============================================================================

class TestUnifiedPerceptionPipeline:
    """Test suite for Unified Perception Pipeline."""

    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline config fixture."""
        from src.meta_ai.unified_perception import PerceptionConfig
        from src.meta_ai.dinov3 import DINOv3Config, DINOv3ModelSize
        from src.meta_ai.sam3 import SAM3Config, SAM3ModelSize
        from src.meta_ai.vjepa2 import VJEPA2Config, VJEPA2ModelSize

        return PerceptionConfig(
            dinov3_config=DINOv3Config(model_size=DINOv3ModelSize.LARGE, device="cpu"),
            sam3_config=SAM3Config(model_size=SAM3ModelSize.LARGE, device="cpu"),
            vjepa2_config=VJEPA2Config(model_size=VJEPA2ModelSize.LARGE, device="cpu"),
            enable_dinov3=True,
            enable_sam3=True,
            enable_vjepa2=True,
            enable_privacy=False,  # Skip for testing
            enable_moe_routing=False,
        )

    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create pipeline fixture."""
        from src.meta_ai.unified_perception import UnifiedPerceptionPipeline
        p = UnifiedPerceptionPipeline(pipeline_config)
        p.initialize()
        return p

    @pytest.fixture
    def test_frame(self):
        """Create test frame fixture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        from src.meta_ai.unified_perception import UnifiedPerceptionPipeline

        pipeline = UnifiedPerceptionPipeline(pipeline_config)
        success = pipeline.initialize()

        assert success == True
        assert pipeline._is_initialized == True
        pipeline.shutdown()

    def test_pipeline_process_frame(self, pipeline, test_frame):
        """Test full frame processing."""
        from src.meta_ai.unified_perception import PerceptionTier

        result = pipeline.process_frame(
            test_frame,
            task_description="pick up the cup",
            tier=PerceptionTier.CONTROL
        )

        assert result is not None
        assert result.tier == PerceptionTier.CONTROL
        assert result.total_time_ms > 0

    def test_pipeline_safety_check(self, pipeline, test_frame):
        """Test safety-only check."""
        is_safe, collision_prob, action = pipeline.check_safety(test_frame)

        assert isinstance(is_safe, bool)
        assert 0 <= collision_prob <= 1
        assert action in ["CONTINUE", "SLOW", "STOP"]

    def test_pipeline_feature_fusion(self, pipeline, test_frame):
        """Test feature fusion."""
        result = pipeline.process_frame(test_frame, task_description="grasp object")

        assert result.fused_features is not None
        assert len(result.fused_features.shape) == 1

    def test_pipeline_safety_action_levels(self, pipeline):
        """Test different safety action levels."""
        from src.meta_ai.unified_perception import PerceptionResult

        # Test CONTINUE threshold
        result = PerceptionResult()
        result.collision_probability = 0.3
        assert result.collision_probability < pipeline.config.collision_threshold

        # Test SLOW threshold
        result.collision_probability = 0.8
        assert result.collision_probability >= pipeline.config.collision_threshold
        assert result.collision_probability < pipeline.config.emergency_stop_threshold

        # Test STOP threshold
        result.collision_probability = 0.95
        assert result.collision_probability >= pipeline.config.emergency_stop_threshold

    def test_pipeline_statistics(self, pipeline, test_frame):
        """Test pipeline statistics."""
        # Process some frames
        for _ in range(3):
            pipeline.process_frame(test_frame)

        stats = pipeline.get_statistics()

        assert stats["frames_processed"] == 3
        assert stats["is_initialized"] == True
        assert "models" in stats

    def test_pipeline_shutdown(self, pipeline_config):
        """Test pipeline shutdown."""
        from src.meta_ai.unified_perception import UnifiedPerceptionPipeline

        pipeline = UnifiedPerceptionPipeline(pipeline_config)
        pipeline.initialize()
        pipeline.shutdown()

        # Should not raise any errors


# =============================================================================
# Privacy Wrapper Tests
# =============================================================================

class TestPrivacyWrapper:
    """Test suite for Meta AI Privacy Wrapper."""

    @pytest.fixture
    def privacy_wrapper(self):
        """Create privacy wrapper fixture."""
        from src.meta_ai.privacy_wrapper import MetaAIPrivacyWrapper, PrivacyConfig

        config = PrivacyConfig(
            enabled=True,
            security_bits=128,
            lwe_dimension=1024
        )
        return MetaAIPrivacyWrapper(config)

    def test_privacy_wrapper_initialization(self):
        """Test privacy wrapper initialization."""
        from src.meta_ai.privacy_wrapper import MetaAIPrivacyWrapper, PrivacyConfig

        config = PrivacyConfig(security_bits=128)
        wrapper = MetaAIPrivacyWrapper(config)

        assert wrapper.config == config
        assert wrapper.config.security_bits == 128

    def test_privacy_wrapper_encrypt_features(self, privacy_wrapper):
        """Test feature encryption."""
        features = np.random.randn(1024).astype(np.float32)

        encrypted = privacy_wrapper.encrypt_features(features, source_model="dinov3")

        assert encrypted is not None
        assert hasattr(encrypted, 'encrypted_data')
        assert encrypted.source_model == "dinov3"

    def test_privacy_wrapper_statistics(self, privacy_wrapper):
        """Test statistics tracking."""
        features = np.random.randn(1024).astype(np.float32)

        privacy_wrapper.encrypt_features(features, source_model="test")

        stats = privacy_wrapper.get_statistics()
        assert stats is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetaAIIntegration:
    """Integration tests for Meta AI models working together."""

    @pytest.fixture
    def test_frame(self):
        """Create test frame fixture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_dinov3_sam3_integration(self, test_frame):
        """Test DINOv3 + SAM3 working together."""
        from src.meta_ai.dinov3 import DINOv3Encoder, DINOv3Config
        from src.meta_ai.sam3 import SAM3Segmenter, SAM3Config

        # Initialize both models
        dinov3 = DINOv3Encoder(DINOv3Config(device="cpu"))
        sam3 = SAM3Segmenter(SAM3Config(device="cpu"))

        dinov3.load_model()
        sam3.load_model()

        # Get features from DINOv3
        features = dinov3.encode(test_frame, return_dense=True)

        # Get segmentation from SAM3
        segmentation = sam3.segment_text(test_frame, "object")

        # Both should work
        assert features.global_features is not None
        assert segmentation.masks is not None or segmentation.num_masks >= 0

    def test_full_pipeline_flow(self, test_frame):
        """Test complete pipeline flow."""
        from src.meta_ai.unified_perception import UnifiedPerceptionPipeline, PerceptionConfig

        config = PerceptionConfig(
            enable_dinov3=True,
            enable_sam3=True,
            enable_vjepa2=True,
            enable_privacy=False,
            enable_moe_routing=False,
        )

        pipeline = UnifiedPerceptionPipeline(config)
        pipeline.initialize()

        # Process multiple frames (simulating video)
        results = []
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame, task_description=f"task_{i}")
            results.append(result)

        # Verify all processed successfully
        assert len(results) == 5
        assert all(r.total_time_ms > 0 for r in results)

        pipeline.shutdown()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestMetaAIEdgeCases:
    """Test edge cases and error handling."""

    def test_dinov3_empty_image(self):
        """Test DINOv3 with edge case images."""
        from src.meta_ai.dinov3 import DINOv3Encoder, DINOv3Config

        encoder = DINOv3Encoder(DINOv3Config(device="cpu"))
        encoder.load_model()

        # Very small image
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        features = encoder.encode(tiny_image)
        assert features.global_features is not None

        # Large image
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        features = encoder.encode(large_image)
        assert features.global_features is not None

    def test_sam3_empty_prompts(self):
        """Test SAM3 with edge case prompts."""
        from src.meta_ai.sam3 import SAM3Segmenter, SAM3Config

        segmenter = SAM3Segmenter(SAM3Config(device="cpu"))
        segmenter.load_model()

        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Empty text prompt
        result = segmenter.segment_text(test_image, "")
        assert result is not None

        # Very long text prompt
        long_prompt = "a " * 100 + "object"
        result = segmenter.segment_text(test_image, long_prompt)
        assert result is not None

    def test_vjepa2_single_frame_prediction(self):
        """Test V-JEPA 2 with minimal context."""
        from src.meta_ai.vjepa2 import VJEPA2WorldModel, VJEPA2Config, WorldState

        model = VJEPA2WorldModel(VJEPA2Config(device="cpu"))
        model.load_model()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Single frame should still work
        embedding = model.encode_frame(frame)
        state = WorldState(frame_embedding=embedding)

        prediction = model.predict(state, num_steps=1)
        assert prediction is not None
        assert prediction.future_embeddings.shape[0] == 1


# =============================================================================
# Performance Tests
# =============================================================================

class TestMetaAIPerformance:
    """Performance benchmarks for Meta AI models."""

    @pytest.fixture
    def test_frame(self):
        """Create test frame fixture."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_dinov3_inference_time(self, test_frame):
        """Benchmark DINOv3 inference time."""
        from src.meta_ai.dinov3 import DINOv3Encoder, DINOv3Config

        encoder = DINOv3Encoder(DINOv3Config(device="cpu"))
        encoder.load_model()

        # Warm up
        encoder.encode(test_frame)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            encoder.encode(test_frame)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        print(f"\nDINOv3 avg inference time: {avg_time:.2f}ms")

        # Should complete in reasonable time (relaxed for mock model)
        assert avg_time < 5000  # 5 seconds max

    def test_sam3_inference_time(self, test_frame):
        """Benchmark SAM3 inference time."""
        from src.meta_ai.sam3 import SAM3Segmenter, SAM3Config

        segmenter = SAM3Segmenter(SAM3Config(device="cpu"))
        segmenter.load_model()

        # Warm up
        segmenter.segment_text(test_frame, "object")

        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            segmenter.segment_text(test_frame, "cup")
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        print(f"\nSAM3 avg inference time: {avg_time:.2f}ms")

        assert avg_time < 5000


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

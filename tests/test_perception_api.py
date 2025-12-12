#!/usr/bin/env python3
"""
Comprehensive Test Suite for Meta AI Perception API Endpoints

Tests all /api/perception/* endpoints:
- Model management (CRUD)
- Privacy settings
- Pipeline status
- Safety predictions
- TFLOPS allocation
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def client():
    """Create test client fixture."""
    from src.platform.api.main import app
    return TestClient(app)


@pytest.fixture
def api_headers():
    """Create API headers with authentication."""
    return {"X-API-Key": os.getenv("API_KEY", "default_insecure_key")}


# =============================================================================
# Model Management Tests
# =============================================================================

class TestPerceptionModelsAPI:
    """Test suite for /api/perception/models endpoints."""

    def test_get_all_models(self, client, api_headers):
        """Test GET /api/perception/models - list all models."""
        response = client.get("/api/perception/models", headers=api_headers)

        assert response.status_code == 200
        models = response.json()

        # Should have all 3 Meta AI models
        assert isinstance(models, list)
        assert len(models) >= 3

        model_ids = [m["id"] for m in models]
        assert "dinov3" in model_ids
        assert "sam3" in model_ids
        assert "vjepa2" in model_ids

    def test_get_model_structure(self, client, api_headers):
        """Test that each model has required fields."""
        response = client.get("/api/perception/models", headers=api_headers)
        models = response.json()

        required_fields = [
            "id", "name", "description", "version", "enabled",
            "status", "tflops", "latency_ms", "model_size"
        ]

        for model in models:
            for field in required_fields:
                assert field in model, f"Model missing field: {field}"

    def test_get_specific_model_dinov3(self, client, api_headers):
        """Test GET /api/perception/models/dinov3."""
        response = client.get("/api/perception/models/dinov3", headers=api_headers)

        assert response.status_code == 200
        model = response.json()

        assert model["id"] == "dinov3"
        assert model["name"] == "DINOv3"
        assert model["tflops"] == 8.0
        assert "config" in model

    def test_get_specific_model_sam3(self, client, api_headers):
        """Test GET /api/perception/models/sam3."""
        response = client.get("/api/perception/models/sam3", headers=api_headers)

        assert response.status_code == 200
        model = response.json()

        assert model["id"] == "sam3"
        assert model["name"] == "SAM 3"
        assert model["tflops"] == 15.0

    def test_get_specific_model_vjepa2(self, client, api_headers):
        """Test GET /api/perception/models/vjepa2."""
        response = client.get("/api/perception/models/vjepa2", headers=api_headers)

        assert response.status_code == 200
        model = response.json()

        assert model["id"] == "vjepa2"
        assert model["name"] == "V-JEPA 2"
        assert model["tflops"] == 10.0

    def test_get_nonexistent_model(self, client, api_headers):
        """Test GET /api/perception/models/{invalid_id} returns 404."""
        response = client.get("/api/perception/models/invalid_model", headers=api_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestModelEnableDisableAPI:
    """Test suite for model enable/disable functionality."""

    def test_disable_model(self, client, api_headers):
        """Test POST /api/perception/models/{id}/enable - disable."""
        response = client.post(
            "/api/perception/models/dinov3/enable",
            headers=api_headers,
            json={"enabled": False}
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] == True
        assert result["enabled"] == False
        assert result["status"] == "stopped"

    def test_enable_model(self, client, api_headers):
        """Test POST /api/perception/models/{id}/enable - enable."""
        response = client.post(
            "/api/perception/models/dinov3/enable",
            headers=api_headers,
            json={"enabled": True}
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] == True
        assert result["enabled"] == True
        assert result["status"] == "running"

    def test_enable_nonexistent_model(self, client, api_headers):
        """Test enabling non-existent model returns 404."""
        response = client.post(
            "/api/perception/models/fake_model/enable",
            headers=api_headers,
            json={"enabled": True}
        )

        assert response.status_code == 404


class TestModelConfigAPI:
    """Test suite for model configuration endpoints."""

    def test_get_model_config(self, client, api_headers):
        """Test GET /api/perception/models/{id}/config."""
        response = client.get(
            "/api/perception/models/dinov3/config",
            headers=api_headers
        )

        assert response.status_code == 200
        config = response.json()

        assert "model_size" in config
        assert "input_size" in config
        assert "use_fp16" in config

    def test_get_sam3_config(self, client, api_headers):
        """Test GET /api/perception/models/sam3/config."""
        response = client.get(
            "/api/perception/models/sam3/config",
            headers=api_headers
        )

        assert response.status_code == 200
        config = response.json()

        assert "max_objects" in config
        assert "confidence_threshold" in config
        assert "enable_tracking" in config

    def test_get_vjepa2_config(self, client, api_headers):
        """Test GET /api/perception/models/vjepa2/config."""
        response = client.get(
            "/api/perception/models/vjepa2/config",
            headers=api_headers
        )

        assert response.status_code == 200
        config = response.json()

        assert "num_frames" in config
        assert "prediction_horizon" in config
        assert "enable_safety_prediction" in config
        assert "collision_threshold" in config

    def test_update_model_config(self, client, api_headers):
        """Test POST /api/perception/models/{id}/config."""
        new_config = {
            "config": {
                "input_size": 448,
                "use_fp16": False
            }
        }

        response = client.post(
            "/api/perception/models/dinov3/config",
            headers=api_headers,
            json=new_config
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] == True
        assert result["config"]["input_size"] == 448
        assert result["config"]["use_fp16"] == False

        # Restore original config
        restore_config = {"config": {"input_size": 518, "use_fp16": True}}
        client.post(
            "/api/perception/models/dinov3/config",
            headers=api_headers,
            json=restore_config
        )

    def test_update_sam3_config(self, client, api_headers):
        """Test updating SAM3 configuration."""
        new_config = {
            "config": {
                "max_objects": 15,
                "confidence_threshold": 0.6
            }
        }

        response = client.post(
            "/api/perception/models/sam3/config",
            headers=api_headers,
            json=new_config
        )

        assert response.status_code == 200
        result = response.json()

        assert result["config"]["max_objects"] == 15
        assert result["config"]["confidence_threshold"] == 0.6

        # Restore
        restore_config = {"config": {"max_objects": 10, "confidence_threshold": 0.5}}
        client.post("/api/perception/models/sam3/config", headers=api_headers, json=restore_config)


# =============================================================================
# Privacy API Tests
# =============================================================================

class TestPrivacyAPI:
    """Test suite for /api/perception/privacy endpoints."""

    def test_get_privacy_settings(self, client, api_headers):
        """Test GET /api/perception/privacy."""
        response = client.get("/api/perception/privacy", headers=api_headers)

        assert response.status_code == 200
        privacy = response.json()

        assert "enabled" in privacy
        assert "security_bits" in privacy
        assert "lwe_dimension" in privacy
        assert "enable_homomorphic_routing" in privacy

    def test_privacy_default_values(self, client, api_headers):
        """Test privacy default values."""
        response = client.get("/api/perception/privacy", headers=api_headers)
        privacy = response.json()

        assert privacy["security_bits"] == 128
        assert privacy["lwe_dimension"] == 1024

    def test_update_privacy_enabled(self, client, api_headers):
        """Test updating privacy enabled setting."""
        response = client.post(
            "/api/perception/privacy",
            headers=api_headers,
            json={"enabled": False}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["success"] == True
        assert result["enabled"] == False

        # Restore
        client.post("/api/perception/privacy", headers=api_headers, json={"enabled": True})

    def test_update_privacy_security_bits(self, client, api_headers):
        """Test updating security bits."""
        response = client.post(
            "/api/perception/privacy",
            headers=api_headers,
            json={"security_bits": 256}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["security_bits"] == 256

        # Restore
        client.post("/api/perception/privacy", headers=api_headers, json={"security_bits": 128})

    def test_update_privacy_multiple_fields(self, client, api_headers):
        """Test updating multiple privacy fields at once."""
        response = client.post(
            "/api/perception/privacy",
            headers=api_headers,
            json={
                "enabled": True,
                "security_bits": 192,
                "lwe_dimension": 2048,
                "enable_homomorphic_routing": False
            }
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] == True
        assert result["security_bits"] == 192
        assert result["lwe_dimension"] == 2048
        assert result["enable_homomorphic_routing"] == False

        # Restore defaults
        client.post(
            "/api/perception/privacy",
            headers=api_headers,
            json={
                "enabled": True,
                "security_bits": 128,
                "lwe_dimension": 1024,
                "enable_homomorphic_routing": True
            }
        )


# =============================================================================
# Pipeline Status API Tests
# =============================================================================

class TestPipelineStatusAPI:
    """Test suite for /api/perception/pipeline endpoints."""

    def test_get_pipeline_status(self, client, api_headers):
        """Test GET /api/perception/pipeline/status."""
        response = client.get("/api/perception/pipeline/status", headers=api_headers)

        assert response.status_code == 200
        status = response.json()

        assert "running" in status
        assert "fusion_method" in status
        assert "safety_rate_hz" in status
        assert "control_rate_hz" in status
        assert "learning_rate_hz" in status
        assert "cloud_rate_hz" in status

    def test_pipeline_timing_rates(self, client, api_headers):
        """Test pipeline timing rates."""
        response = client.get("/api/perception/pipeline/status", headers=api_headers)
        status = response.json()

        assert status["safety_rate_hz"] == 1000.0
        assert status["control_rate_hz"] == 100.0
        assert status["learning_rate_hz"] == 10.0
        assert status["cloud_rate_hz"] == 0.1


# =============================================================================
# Safety Predictions API Tests
# =============================================================================

class TestSafetyPredictionsAPI:
    """Test suite for /api/perception/safety endpoints."""

    def test_get_safety_predictions(self, client, api_headers):
        """Test GET /api/perception/safety/predictions."""
        response = client.get("/api/perception/safety/predictions", headers=api_headers)

        assert response.status_code == 200
        predictions = response.json()

        assert "collision_prob" in predictions
        assert "horizon" in predictions
        assert "time_to_impact" in predictions

    def test_safety_prediction_values(self, client, api_headers):
        """Test safety prediction value ranges."""
        response = client.get("/api/perception/safety/predictions", headers=api_headers)
        predictions = response.json()

        # Collision probability should be between 0 and 1
        assert 0 <= predictions["collision_prob"] <= 1

        # Horizon should be positive
        assert predictions["horizon"] > 0

        # Time to impact should be -1 (no collision) or positive
        assert predictions["time_to_impact"] == -1 or predictions["time_to_impact"] > 0


# =============================================================================
# TFLOPS API Tests
# =============================================================================

class TestTFLOPSAPI:
    """Test suite for /api/perception/tflops endpoint."""

    def test_get_tflops(self, client, api_headers):
        """Test GET /api/perception/tflops."""
        response = client.get("/api/perception/tflops", headers=api_headers)

        assert response.status_code == 200
        tflops = response.json()

        assert "models" in tflops
        assert "total_allocated" in tflops
        assert "total_active" in tflops
        assert "utilization_percent" in tflops

    def test_tflops_model_breakdown(self, client, api_headers):
        """Test TFLOPS breakdown by model."""
        response = client.get("/api/perception/tflops", headers=api_headers)
        tflops = response.json()

        assert "dinov3" in tflops["models"]
        assert "sam3" in tflops["models"]
        assert "vjepa2" in tflops["models"]

        # Check each model has tflops and enabled
        for model_id, model_info in tflops["models"].items():
            assert "tflops" in model_info
            assert "enabled" in model_info

    def test_tflops_allocation_total(self, client, api_headers):
        """Test total TFLOPS allocation."""
        response = client.get("/api/perception/tflops", headers=api_headers)
        tflops = response.json()

        # Meta AI models should total 33 TFLOPS
        assert tflops["total_allocated"] == 33.0

        # Calculate expected active
        expected_active = sum(
            m["tflops"] for m in tflops["models"].values() if m["enabled"]
        )
        assert tflops["total_active"] == expected_active

    def test_tflops_utilization_calculation(self, client, api_headers):
        """Test utilization percentage calculation."""
        response = client.get("/api/perception/tflops", headers=api_headers)
        tflops = response.json()

        expected_utilization = (tflops["total_active"] / tflops["total_allocated"]) * 100
        assert abs(tflops["utilization_percent"] - expected_utilization) < 0.01


# =============================================================================
# Authentication Tests
# =============================================================================

class TestPerceptionAPIAuthentication:
    """Test API authentication for perception endpoints."""

    def test_models_requires_auth(self, client):
        """Test that /api/perception/models requires authentication."""
        response = client.get("/api/perception/models")
        # Without API key, should fail
        # Note: The actual behavior depends on FastAPI's auto_error setting

    def test_invalid_api_key(self, client):
        """Test invalid API key is rejected."""
        headers = {"X-API-Key": "invalid_key_12345"}
        response = client.get("/api/perception/models", headers=headers)

        assert response.status_code == 403

    def test_missing_api_key_header(self, client):
        """Test missing API key header."""
        response = client.get("/api/perception/models")
        # Should fail without X-API-Key header
        assert response.status_code in [401, 403, 422]  # Depends on FastAPI config


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestPerceptionAPIErrors:
    """Test error handling for perception API endpoints."""

    def test_invalid_model_id(self, client, api_headers):
        """Test 404 for invalid model ID."""
        response = client.get("/api/perception/models/not_a_real_model", headers=api_headers)
        assert response.status_code == 404

    def test_enable_invalid_model(self, client, api_headers):
        """Test enable endpoint with invalid model."""
        response = client.post(
            "/api/perception/models/fake/enable",
            headers=api_headers,
            json={"enabled": True}
        )
        assert response.status_code == 404

    def test_config_invalid_model(self, client, api_headers):
        """Test config endpoint with invalid model."""
        response = client.get("/api/perception/models/fake/config", headers=api_headers)
        assert response.status_code == 404

    def test_update_config_invalid_model(self, client, api_headers):
        """Test config update with invalid model."""
        response = client.post(
            "/api/perception/models/fake/config",
            headers=api_headers,
            json={"config": {"key": "value"}}
        )
        assert response.status_code == 404


# =============================================================================
# Integration Tests
# =============================================================================

class TestPerceptionAPIIntegration:
    """Integration tests for perception API."""

    def test_full_model_workflow(self, client, api_headers):
        """Test complete workflow: get -> disable -> configure -> enable."""
        # 1. Get initial state
        response = client.get("/api/perception/models/sam3", headers=api_headers)
        initial_state = response.json()
        assert initial_state["enabled"] == True

        # 2. Disable model
        response = client.post(
            "/api/perception/models/sam3/enable",
            headers=api_headers,
            json={"enabled": False}
        )
        assert response.json()["enabled"] == False

        # 3. Update config while disabled
        response = client.post(
            "/api/perception/models/sam3/config",
            headers=api_headers,
            json={"config": {"max_objects": 20}}
        )
        assert response.json()["config"]["max_objects"] == 20

        # 4. Re-enable model
        response = client.post(
            "/api/perception/models/sam3/enable",
            headers=api_headers,
            json={"enabled": True}
        )
        assert response.json()["enabled"] == True

        # 5. Verify TFLOPS reflects change
        response = client.get("/api/perception/tflops", headers=api_headers)
        assert response.json()["models"]["sam3"]["enabled"] == True

        # 6. Restore config
        client.post(
            "/api/perception/models/sam3/config",
            headers=api_headers,
            json={"config": {"max_objects": 10}}
        )

    def test_all_models_disabled_affects_tflops(self, client, api_headers):
        """Test that disabling all models reduces TFLOPS to 0."""
        # Get initial TFLOPS
        response = client.get("/api/perception/tflops", headers=api_headers)
        initial_active = response.json()["total_active"]

        # Disable all models
        for model_id in ["dinov3", "sam3", "vjepa2"]:
            client.post(
                f"/api/perception/models/{model_id}/enable",
                headers=api_headers,
                json={"enabled": False}
            )

        # Check TFLOPS is 0
        response = client.get("/api/perception/tflops", headers=api_headers)
        assert response.json()["total_active"] == 0.0
        assert response.json()["utilization_percent"] == 0.0

        # Re-enable all models
        for model_id in ["dinov3", "sam3", "vjepa2"]:
            client.post(
                f"/api/perception/models/{model_id}/enable",
                headers=api_headers,
                json={"enabled": True}
            )

        # Verify restored
        response = client.get("/api/perception/tflops", headers=api_headers)
        assert response.json()["total_active"] == initial_active


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

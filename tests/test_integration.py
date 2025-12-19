"""
Integration Tests for Dynamical.ai System

This module provides integration tests that verify components work together
correctly. Includes simulated hardware-in-loop tests.

Test Categories:
1. Connection failover tests
2. GPU resource contention tests
3. Inter-tier communication tests
4. End-to-end pipeline tests
5. Stress tests
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def robust_manager():
    """Create a robust system manager for testing."""
    from src.core.system_robustness import RobustSystemManager
    manager = RobustSystemManager()
    yield manager
    manager.stop()


@pytest.fixture
def dual_connection():
    """Create dual path connection for testing."""
    from src.core.system_robustness import DualPathConnection
    conn = DualPathConnection()
    yield conn


@pytest.fixture
def gpu_manager():
    """Create GPU resource manager for testing."""
    from src.core.system_robustness import GPUResourceManager
    return GPUResourceManager(max_memory_gb=8.0)


@pytest.fixture
def tier_bus():
    """Create inter-tier bus for testing."""
    from src.core.system_robustness import InterTierBus
    return InterTierBus()


@pytest.fixture
def health_predictor():
    """Create health predictor for testing."""
    from src.core.system_robustness import HealthPredictor
    return HealthPredictor(window_size=50)


@pytest.fixture
def fallback_controller():
    """Create fallback controller for testing."""
    from src.core.system_robustness import SafeFallbackController
    return SafeFallbackController(n_joints=7)


# =============================================================================
# Connection Failover Tests
# =============================================================================

class TestConnectionFailover:
    """Tests for dual-path connection failover."""

    def test_connection_health_defaults(self, dual_connection):
        """Test connection health initialization."""
        from src.core.system_robustness import ConnectionType

        assert dual_connection._active_connection == ConnectionType.WIFI_6E
        assert dual_connection._failover_count == 0
        assert dual_connection._recovery_count == 0

    def test_failover_threshold(self, dual_connection):
        """Test failover occurs after consecutive failures."""
        from src.core.system_robustness import ConnectionType

        # Simulate consecutive failures
        for _ in range(dual_connection.FAILOVER_CONSECUTIVE_FAILURES):
            dual_connection._update_health(dual_connection._wifi_health, success=False)

        assert dual_connection._should_failover()

    def test_health_score_calculation(self, dual_connection):
        """Test connection health score calculation."""
        health = dual_connection._wifi_health
        health.latency_ms = 2.0
        health.jitter_ms = 1.0
        health.packet_loss_percent = 0.5

        score = health.quality_score
        assert 0 <= score <= 1
        assert score > 0.5  # Good health should score high

    def test_connection_status(self, dual_connection):
        """Test connection status reporting."""
        status = dual_connection.get_status()

        assert "active_connection" in status
        assert "wifi_health" in status
        assert "usb_health" in status
        assert "failover_count" in status


# =============================================================================
# GPU Resource Management Tests
# =============================================================================

class TestGPUResourceManagement:
    """Tests for GPU resource management."""

    def test_memory_budgets(self, gpu_manager):
        """Test memory budgets are defined."""
        assert "dinov3" in gpu_manager.memory_budgets
        assert "sam3" in gpu_manager.memory_budgets
        assert "vjepa2" in gpu_manager.memory_budgets

    def test_acquire_gpu_context(self, gpu_manager):
        """Test GPU acquisition context manager."""
        with gpu_manager.acquire_gpu("dinov3"):
            assert "dinov3" in gpu_manager._model_last_used

    def test_memory_status(self, gpu_manager):
        """Test memory status reporting."""
        status = gpu_manager.get_memory_status()

        assert "allocated_gb" in status
        assert "max_gb" in status
        assert status["max_gb"] == 8.0


# =============================================================================
# Inter-Tier Communication Tests
# =============================================================================

class TestInterTierCommunication:
    """Tests for inter-tier message bus."""

    def test_publish_receive(self, tier_bus):
        """Test basic publish/receive."""
        from src.core.system_robustness import TierMessage, TierPriority

        msg = TierMessage(
            source_tier=TierPriority.PERCEPTION,
            target_tier=TierPriority.CONTROL,
            message_type="pose_update",
            payload={"keypoints": [1, 2, 3]},
        )

        tier_bus.publish(msg)
        received = tier_bus.receive(TierPriority.CONTROL, timeout=0.1)

        assert received is not None
        assert received.message_type == "pose_update"

    def test_priority_ordering(self, tier_bus):
        """Test messages are ordered by priority."""
        from src.core.system_robustness import TierMessage, TierPriority

        # Send low priority first
        low = TierMessage(
            source_tier=TierPriority.LEARNING,
            target_tier=TierPriority.CONTROL,
            message_type="low",
            payload=None,
            priority=0,
        )

        # Send high priority second
        high = TierMessage(
            source_tier=TierPriority.SAFETY,
            target_tier=TierPriority.CONTROL,
            message_type="high",
            payload=None,
            priority=10,
        )

        tier_bus.publish(low)
        tier_bus.publish(high)

        # High priority should come first
        first = tier_bus.receive(TierPriority.CONTROL, timeout=0.1)
        assert first.message_type == "high"

    def test_estop_mechanism(self, tier_bus):
        """Test e-stop trigger and check."""
        assert not tier_bus.is_estopped()

        tier_bus.trigger_estop()
        assert tier_bus.is_estopped()

        tier_bus.reset_estop()
        assert not tier_bus.is_estopped()

    def test_latest_state(self, tier_bus):
        """Test latest state storage."""
        from src.core.system_robustness import TierPriority

        test_state = {"position": [1, 2, 3]}
        tier_bus.set_latest_state(TierPriority.CONTROL, test_state)

        state, timestamp = tier_bus.get_latest_state(TierPriority.CONTROL)
        assert state == test_state
        assert timestamp > 0


# =============================================================================
# Health Prediction Tests
# =============================================================================

class TestHealthPrediction:
    """Tests for proactive health monitoring."""

    def test_record_metrics(self, health_predictor):
        """Test metric recording."""
        for i in range(20):
            health_predictor.record("latency_ms", 2.0 + i * 0.1)

        assert "latency_ms" in health_predictor._metrics
        assert len(health_predictor._metrics["latency_ms"]) == 20

    def test_health_status(self, health_predictor):
        """Test health status determination."""
        # Record healthy values
        for _ in range(10):
            health_predictor.record("latency_ms", 1.0)

        status = health_predictor.get_health_status("latency_ms")
        assert status == "HEALTHY"

        # Record critical values
        for _ in range(10):
            health_predictor.record("latency_ms", 10.0)

        status = health_predictor.get_health_status("latency_ms")
        assert status == "CRITICAL"

    def test_failure_prediction(self, health_predictor):
        """Test failure prediction with trend."""
        # Record increasing trend toward failure
        for i in range(20):
            value = 2.0 + i * 0.2  # Increasing toward 5.0 threshold
            health_predictor.record("latency_ms", value)

        will_fail, predicted = health_predictor.predict_failure("latency_ms", horizon_s=5.0)
        assert predicted > 2.0  # Should predict higher than current

    def test_all_predictions(self, health_predictor):
        """Test getting all predictions."""
        for i in range(15):
            health_predictor.record("latency_ms", 2.0)
            health_predictor.record("jitter_ms", 1.0)

        predictions = health_predictor.get_all_predictions()
        assert "latency_ms" in predictions
        assert "jitter_ms" in predictions


# =============================================================================
# Fallback Controller Tests
# =============================================================================

class TestFallbackController:
    """Tests for safe fallback behaviors."""

    def test_initial_state(self, fallback_controller):
        """Test initial fallback state."""
        from src.core.system_robustness import SafeFallbackController

        assert fallback_controller.current_level == SafeFallbackController.FallbackLevel.NORMAL

    def test_record_good_action(self, fallback_controller):
        """Test recording good actions."""
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        target = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        fallback_controller.record_good_action(action, target)

        assert fallback_controller._last_good_action is not None
        assert np.allclose(fallback_controller._last_good_action, action)

    def test_fallback_hierarchy(self, fallback_controller):
        """Test fallback hierarchy progression."""
        from src.core.system_robustness import SafeFallbackController

        # No actions recorded - should go to gravity comp
        action, level = fallback_controller.get_fallback_action("vla_failure")
        assert level in [SafeFallbackController.FallbackLevel.GRAVITY_COMP,
                        SafeFallbackController.FallbackLevel.HOLD_POSITION]

    def test_cached_action_fallback(self, fallback_controller):
        """Test cached action is used when available."""
        from src.core.system_robustness import SafeFallbackController

        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        target = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        fallback_controller.record_good_action(action, target)

        # Get fallback immediately (cached action should be valid)
        fallback_action, level = fallback_controller.get_fallback_action("failure")
        assert level == SafeFallbackController.FallbackLevel.CACHED_ACTION

    def test_hold_position_fallback(self, fallback_controller):
        """Test hold position when only position is known."""
        from src.core.system_robustness import SafeFallbackController

        position = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        fallback_controller.update_position(position)

        # Wait for cached actions to expire
        time.sleep(0.6)

        action, level = fallback_controller.get_fallback_action("failure")
        assert level == SafeFallbackController.FallbackLevel.HOLD_POSITION
        assert np.allclose(action, position)

    def test_estop_trigger(self, fallback_controller):
        """Test e-stop trigger sets level."""
        from src.core.system_robustness import SafeFallbackController

        fallback_controller.trigger_estop()
        assert fallback_controller.current_level == SafeFallbackController.FallbackLevel.ESTOP


# =============================================================================
# FHE Security Tests
# =============================================================================

class TestFHESecurity:
    """Tests for FHE security configuration."""

    def test_default_config_valid(self):
        """Test default FHE config is valid."""
        from src.core.system_robustness import FHESecurityConfig

        config = FHESecurityConfig()
        assert config.validate()

    def test_security_level(self):
        """Test security level estimation."""
        from src.core.system_robustness import FHESecurityConfig

        config = FHESecurityConfig()
        estimated = config._estimate_security()
        assert estimated >= 128  # Should meet 128-bit security

    def test_threat_model_complete(self):
        """Test threat model documentation is complete."""
        from src.core.system_robustness import FHESecurityConfig

        config = FHESecurityConfig()
        threat_model = config.get_threat_model()

        assert "adversary_model" in threat_model
        assert "security_assumption" in threat_model
        assert "protected_data" in threat_model
        assert "not_protected" in threat_model
        assert "key_management" in threat_model

    def test_polynomial_degree_power_of_two(self):
        """Test polynomial degree is power of 2."""
        from src.core.system_robustness import FHESecurityConfig

        config = FHESecurityConfig()
        n = config.polynomial_degree
        assert n & (n - 1) == 0  # Power of 2 check


# =============================================================================
# Redundancy Tests
# =============================================================================

class TestRedundancy:
    """Tests for system redundancy."""

    def test_component_registration(self):
        """Test component registration."""
        from src.core.system_robustness import RedundancyManager

        manager = RedundancyManager()
        manager.register_component("test", lambda: True, backup_name="test_backup")

        assert "test" in manager._health_checks
        assert manager._backup_components["test"] == "test_backup"

    def test_health_check_failover(self):
        """Test failover on health check failure."""
        from src.core.system_robustness import RedundancyManager

        manager = RedundancyManager()

        # Primary fails, backup works
        manager.register_component("primary", lambda: False)
        manager.register_component("backup", lambda: True)
        manager._backup_components["primary"] = "backup"

        result = manager.check_health("primary")
        assert result  # Should succeed via backup
        assert manager.get_active_component("primary") == "backup"

    def test_health_summary(self):
        """Test health summary reporting."""
        from src.core.system_robustness import RedundancyManager

        manager = RedundancyManager()
        manager.register_component("comp1", lambda: True)
        manager.check_health("comp1")

        summary = manager.get_health_summary()
        assert "component_health" in summary
        assert "active_components" in summary
        assert "failovers" in summary


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_system_initialization(self, robust_manager):
        """Test full system can initialize."""
        # Manager should be creatable
        assert robust_manager is not None
        assert robust_manager.connection is not None
        assert robust_manager.gpu_manager is not None
        assert robust_manager.tier_bus is not None

    def test_system_status_complete(self, robust_manager):
        """Test system status contains all components."""
        status = robust_manager.get_system_status()

        assert "connection" in status
        assert "gpu" in status
        assert "redundancy" in status
        assert "health_predictions" in status
        assert "fallback_level" in status
        assert "estopped" in status
        assert "fhe_security" in status

    def test_concurrent_tier_communication(self, tier_bus):
        """Test concurrent message passing between tiers."""
        from src.core.system_robustness import TierMessage, TierPriority

        messages_sent = 0
        messages_received = 0

        def sender():
            nonlocal messages_sent
            for i in range(100):
                msg = TierMessage(
                    source_tier=TierPriority.PERCEPTION,
                    target_tier=TierPriority.CONTROL,
                    message_type=f"msg_{i}",
                    payload=i,
                )
                tier_bus.publish(msg)
                messages_sent += 1
                time.sleep(0.001)

        def receiver():
            nonlocal messages_received
            end_time = time.time() + 1.0
            while time.time() < end_time:
                msg = tier_bus.receive(TierPriority.CONTROL, timeout=0.01)
                if msg:
                    messages_received += 1

        sender_thread = threading.Thread(target=sender)
        receiver_thread = threading.Thread(target=receiver)

        sender_thread.start()
        receiver_thread.start()

        sender_thread.join()
        receiver_thread.join()

        # Most messages should be received
        assert messages_received > 50

    def test_gpu_memory_under_load(self, gpu_manager):
        """Test GPU memory management under simulated load."""
        # Simulate loading multiple models
        with gpu_manager.acquire_gpu("dinov3"):
            with gpu_manager.acquire_gpu("sam3"):
                # Both should be tracked
                assert "dinov3" in gpu_manager._model_last_used
                assert "sam3" in gpu_manager._model_last_used


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for robustness verification."""

    def test_high_frequency_health_recording(self, health_predictor):
        """Test health recording at high frequency."""
        start = time.time()
        count = 0

        while time.time() - start < 0.1:  # 100ms
            health_predictor.record("latency_ms", 2.0 + np.random.random())
            count += 1

        # Should handle at least 100 records in 100ms
        assert count > 100

    def test_rapid_failover_recovery(self, dual_connection):
        """Test rapid failover/recovery cycles."""
        from src.core.system_robustness import ConnectionType

        for _ in range(10):
            # Simulate failure
            for _ in range(5):
                dual_connection._update_health(dual_connection._wifi_health, success=False)

            if dual_connection._should_failover():
                dual_connection._failover_to_usb()

            # Simulate recovery
            for _ in range(20):
                dual_connection._update_health(dual_connection._wifi_health, success=True)

        # Should have some failovers
        assert dual_connection._failover_count > 0

    def test_message_queue_overflow(self, tier_bus):
        """Test message queue handles overflow gracefully."""
        from src.core.system_robustness import TierMessage, TierPriority

        # Send more messages than queue can hold
        for i in range(200):
            msg = TierMessage(
                source_tier=TierPriority.PERCEPTION,
                target_tier=TierPriority.CONTROL,
                message_type=f"overflow_{i}",
                payload=i,
            )
            tier_bus.publish(msg)

        # Should still be able to receive messages
        msg = tier_bus.receive(TierPriority.CONTROL, timeout=0.1)
        assert msg is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

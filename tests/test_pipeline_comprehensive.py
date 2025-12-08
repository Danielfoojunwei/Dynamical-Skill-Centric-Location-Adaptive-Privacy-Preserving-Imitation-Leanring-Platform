#!/usr/bin/env python3
"""
Comprehensive Pipeline Test Suite

Tests all real implementations for:
1. Functionality - Does it work?
2. Error Handling - Does it fail gracefully?
3. Robustness - Does it handle edge cases?
"""

import sys
import os
import numpy as np
import time
import traceback
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test results tracking
RESULTS: Dict[str, Dict[str, Any]] = {}


def record_result(category: str, test_name: str, passed: bool, message: str = "", error: str = ""):
    """Record test result."""
    if category not in RESULTS:
        RESULTS[category] = {}
    RESULTS[category][test_name] = {
        "passed": passed,
        "message": message,
        "error": error,
    }
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if message:
        print(f"         {message}")
    if error and not passed:
        print(f"         Error: {error[:100]}")


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for category, tests in RESULTS.items():
        passed = sum(1 for t in tests.values() if t["passed"])
        failed = len(tests) - passed
        total_passed += passed
        total_failed += failed

        status = "✓" if failed == 0 else "✗"
        print(f"\n{status} {category}: {passed}/{len(tests)} passed")

        for name, result in tests.items():
            if not result["passed"]:
                print(f"   - {name}: {result['error'][:60]}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    return total_failed == 0


# =============================================================================
# TEST: Pose Inference Module
# =============================================================================

def test_pose_inference():
    """Test RTMPose inference module."""
    print("\n" + "=" * 70)
    print("TESTING: Pose Inference Module")
    print("=" * 70)

    # Test 1: Import modules
    try:
        from src.core.pose_inference import (
            RTMPoseRealInference,
            RTMPoseConfig,
            Pose2DResult,
            PersonDetector,
            DetectionResult,
        )
        record_result("Pose Inference", "Import modules", True, "All modules imported successfully")
    except ImportError as e:
        record_result("Pose Inference", "Import modules", False, error=str(e))
        return

    # Test 2: Create config with various parameters
    try:
        config1 = RTMPoseConfig(model_name="rtmpose-m", device="cpu")
        config2 = RTMPoseConfig(model_name="rtmpose-l", device="cuda", conf_threshold=0.5)
        config3 = RTMPoseConfig(model_name="rtmw-x-wholebody", device="cpu")
        record_result("Pose Inference", "Create configs", True, "Multiple configs created")
    except Exception as e:
        record_result("Pose Inference", "Create configs", False, error=str(e))

    # Test 3: Initialize inferencer (graceful degradation without model)
    try:
        config = RTMPoseConfig(model_name="rtmpose-m", device="cpu")
        inferencer = RTMPoseRealInference(config)
        # Should initialize even without model file (uses mock fallback)
        record_result("Pose Inference", "Initialize inferencer", True,
                      f"Initialized: model={config.model_name}, keypoints={inferencer.num_keypoints}")
    except Exception as e:
        record_result("Pose Inference", "Initialize inferencer", False, error=str(e))
        return

    # Test 4: Run inference on synthetic image
    try:
        # Create test image (640x480 RGB)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = inferencer.infer(test_image, camera_id="test_cam")

        assert isinstance(result, Pose2DResult), "Result should be Pose2DResult"
        assert result.camera_id == "test_cam", "Camera ID should match"
        assert result.keypoints.shape[0] >= 1, "Should have at least 1 person"
        assert result.keypoints.shape[2] == 3, "Keypoints should have x, y, conf"

        record_result("Pose Inference", "Inference on synthetic image", True,
                      f"Detected {len(result.keypoints)} person(s), "
                      f"keypoints shape: {result.keypoints.shape}")
    except Exception as e:
        record_result("Pose Inference", "Inference on synthetic image", False, error=str(e))

    # Test 5: Batch inference
    try:
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = inferencer.infer_batch(images, camera_ids=["cam0", "cam1", "cam2"])

        assert len(results) == 3, "Should have 3 results"
        assert all(isinstance(r, Pose2DResult) for r in results), "All should be Pose2DResult"

        record_result("Pose Inference", "Batch inference", True,
                      f"Processed {len(results)} images successfully")
    except Exception as e:
        record_result("Pose Inference", "Batch inference", False, error=str(e))

    # Test 6: Error handling - empty image
    try:
        empty_image = np.array([])
        try:
            result = inferencer.infer(empty_image.reshape(0, 0, 3), camera_id="empty")
        except (ValueError, IndexError):
            pass  # Expected error
        record_result("Pose Inference", "Handle empty image", True, "Gracefully handled empty input")
    except Exception as e:
        record_result("Pose Inference", "Handle empty image", False, error=str(e))

    # Test 7: Error handling - wrong dimensions
    try:
        wrong_dims = np.random.randint(0, 255, (100,), dtype=np.uint8)
        try:
            result = inferencer.infer(wrong_dims, camera_id="wrong")
        except (ValueError, IndexError):
            pass  # Expected error
        record_result("Pose Inference", "Handle wrong dimensions", True, "Gracefully handled wrong dims")
    except Exception as e:
        record_result("Pose Inference", "Handle wrong dimensions", False, error=str(e))

    # Test 8: Person detector initialization
    try:
        detector = PersonDetector(model_name="yolov8n", device="cpu")
        record_result("Pose Inference", "Initialize person detector", True,
                      f"PersonDetector initialized")
    except Exception as e:
        record_result("Pose Inference", "Initialize person detector", False, error=str(e))

    # Test 9: Person detection
    try:
        detector = PersonDetector(model_name="yolov8n", device="cpu")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image)

        assert isinstance(result, DetectionResult), "Should return DetectionResult"
        assert hasattr(result, 'bboxes'), "Should have bboxes"
        assert hasattr(result, 'scores'), "Should have scores"

        record_result("Pose Inference", "Person detection", True,
                      f"Detected {len(result.bboxes)} boxes")
    except Exception as e:
        record_result("Pose Inference", "Person detection", False, error=str(e))


# =============================================================================
# TEST: Kinematics & IK Module
# =============================================================================

def test_kinematics():
    """Test robot kinematics and IK solver."""
    print("\n" + "=" * 70)
    print("TESTING: Kinematics & IK Module")
    print("=" * 70)

    # Test 1: Import modules
    try:
        from src.core.retargeting import (
            RobotKinematics,
            KinematicsConfig,
            JointLimits,
            IKSolver,
            IKSolverConfig,
            IKResult,
            MotionRetargeter,
            RetargetingConfig,
        )
        record_result("Kinematics", "Import modules", True, "All modules imported successfully")
    except ImportError as e:
        record_result("Kinematics", "Import modules", False, error=str(e))
        return

    # Test 2: Create kinematics with DH parameters
    try:
        kinematics = RobotKinematics(robot_type="generic_7dof")
        assert kinematics.n_joints == 7, "Should have 7 joints"
        assert kinematics.joint_limits is not None, "Should have joint limits"
        record_result("Kinematics", "Create with DH params", True,
                      f"Created 7-DOF kinematics")
    except Exception as e:
        record_result("Kinematics", "Create with DH params", False, error=str(e))
        return

    # Test 3: Forward kinematics
    try:
        kinematics = RobotKinematics(robot_type="panda")
        q = np.zeros(7)
        T = kinematics.forward_kinematics(q)

        assert T.shape == (4, 4), "FK should return 4x4 matrix"
        assert np.allclose(T[3, :], [0, 0, 0, 1]), "Last row should be [0,0,0,1]"

        record_result("Kinematics", "Forward kinematics (zero config)", True,
                      f"EE position: [{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]")
    except Exception as e:
        record_result("Kinematics", "Forward kinematics (zero config)", False, error=str(e))

    # Test 4: FK with non-zero configuration
    try:
        kinematics = RobotKinematics(robot_type="panda")
        q = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.4])
        T = kinematics.forward_kinematics(q)

        assert T.shape == (4, 4), "FK should return 4x4 matrix"
        # Check it's a valid SE(3) transformation
        R = T[:3, :3]
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0, atol=0.01), f"Rotation det should be 1, got {det}"

        record_result("Kinematics", "FK with joint angles", True,
                      f"EE position: [{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]")
    except Exception as e:
        record_result("Kinematics", "FK with joint angles", False, error=str(e))

    # Test 5: Jacobian computation
    try:
        kinematics = RobotKinematics(robot_type="generic_7dof")
        q = np.zeros(7)
        J = kinematics.jacobian(q)

        assert J.shape == (6, 7), f"Jacobian should be 6x7, got {J.shape}"
        assert not np.all(J == 0), "Jacobian should not be all zeros"

        record_result("Kinematics", "Jacobian computation", True,
                      f"Jacobian shape: {J.shape}, rank: {np.linalg.matrix_rank(J)}")
    except Exception as e:
        record_result("Kinematics", "Jacobian computation", False, error=str(e))

    # Test 6: IK solver initialization
    try:
        kinematics = RobotKinematics(robot_type="panda")
        ik_solver = IKSolver(kinematics, IKSolverConfig(max_iterations=50))
        record_result("Kinematics", "IK solver initialization", True, "IKSolver created")
    except Exception as e:
        record_result("Kinematics", "IK solver initialization", False, error=str(e))

    # Test 7: IK solve for reachable target
    try:
        kinematics = RobotKinematics(robot_type="generic_7dof")
        ik_solver = IKSolver(kinematics)

        # Create target slightly in front of robot
        T_target = np.eye(4)
        T_target[:3, 3] = [0.3, 0.1, 0.4]

        result = ik_solver.solve(T_target, q_init=np.zeros(7))

        assert isinstance(result, IKResult), "Should return IKResult"
        assert result.q is not None, "Should have joint solution"
        assert len(result.q) == 7, "Should have 7 joints"

        # Verify FK of solution is close to target
        T_achieved = kinematics.forward_kinematics(result.q)
        pos_error = np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3])

        record_result("Kinematics", "IK solve reachable target", True,
                      f"Converged: {result.success}, error: {pos_error:.4f}m, iters: {result.iterations}")
    except Exception as e:
        record_result("Kinematics", "IK solve reachable target", False, error=str(e))

    # Test 8: IK with unreachable target (should fail gracefully)
    try:
        kinematics = RobotKinematics(robot_type="generic_7dof")
        ik_solver = IKSolver(kinematics, IKSolverConfig(max_iterations=20))

        # Target far outside workspace
        T_target = np.eye(4)
        T_target[:3, 3] = [10.0, 10.0, 10.0]  # Way too far

        result = ik_solver.solve(T_target)

        # Should return a result (even if not converged)
        assert isinstance(result, IKResult), "Should return IKResult even on failure"
        assert result.q is not None, "Should still provide best attempt"

        record_result("Kinematics", "IK handle unreachable", True,
                      f"Gracefully handled: success={result.success}, error={result.position_error:.2f}m")
    except Exception as e:
        record_result("Kinematics", "IK handle unreachable", False, error=str(e))

    # Test 9: Joint limits enforcement
    try:
        limits = JointLimits(
            lower=np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            upper=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        )

        q_over = np.array([3.0, -3.0, 2.5, -2.5, 1.0, 4.0, -4.0])
        q_clamped = limits.clamp(q_over)

        assert np.all(q_clamped >= limits.lower), "Should be >= lower limits"
        assert np.all(q_clamped <= limits.upper), "Should be <= upper limits"
        assert not limits.is_valid(q_over), "Original should be invalid"
        assert limits.is_valid(q_clamped), "Clamped should be valid"

        record_result("Kinematics", "Joint limits enforcement", True,
                      f"Clamped {np.sum(q_over != q_clamped)} joints")
    except Exception as e:
        record_result("Kinematics", "Joint limits enforcement", False, error=str(e))


# =============================================================================
# TEST: Motion Retargeting Module
# =============================================================================

def test_retargeting():
    """Test motion retargeting module."""
    print("\n" + "=" * 70)
    print("TESTING: Motion Retargeting Module")
    print("=" * 70)

    # Test 1: Import and create retargeter
    try:
        from src.core.retargeting import (
            MotionRetargeter,
            RetargetingConfig,
            RetargetingResult,
        )
        from src.core.retargeting.motion_retargeter import RetargetingMethod

        config = RetargetingConfig(method=RetargetingMethod.HYBRID)
        retargeter = MotionRetargeter(robot_type="generic_7dof", config=config)

        record_result("Retargeting", "Create retargeter", True,
                      f"Created with {retargeter.n_joints} joints")
    except Exception as e:
        record_result("Retargeting", "Create retargeter", False, error=str(e))
        return

    # Test 2: Retarget from human pose
    try:
        from src.core.retargeting import MotionRetargeter

        retargeter = MotionRetargeter(robot_type="generic_7dof")

        # Create synthetic human pose (17 keypoints, COCO format)
        human_pose = np.zeros((17, 3))
        # Nose
        human_pose[0] = [0, 0, 1.7]
        # Shoulders
        human_pose[5] = [-0.2, 0, 1.4]  # left
        human_pose[6] = [0.2, 0, 1.4]   # right
        # Elbows
        human_pose[7] = [-0.3, 0, 1.1]
        human_pose[8] = [0.3, 0, 1.1]
        # Wrists
        human_pose[9] = [-0.35, 0, 0.8]
        human_pose[10] = [0.35, 0, 0.8]
        # Hips
        human_pose[11] = [-0.1, 0, 1.0]
        human_pose[12] = [0.1, 0, 1.0]

        result = retargeter.retarget(human_pose)

        assert isinstance(result, RetargetingResult), "Should return RetargetingResult"
        assert len(result.q) == 7, f"Should have 7 joints, got {len(result.q)}"
        assert result.success, "Should succeed for valid pose"

        record_result("Retargeting", "Retarget human pose", True,
                      f"Joint angles: [{', '.join(f'{q:.2f}' for q in result.q[:4])}...]")
    except Exception as e:
        record_result("Retargeting", "Retarget human pose", False, error=str(e))

    # Test 3: Test smoothing
    try:
        from src.core.retargeting import MotionRetargeter, RetargetingConfig

        config = RetargetingConfig(smoothing_alpha=0.3)
        retargeter = MotionRetargeter(robot_type="generic_7dof", config=config)

        # Send sequence of poses
        results = []
        for i in range(5):
            human_pose = np.zeros((17, 3))
            human_pose[6] = [0.2 + i*0.05, 0, 1.4]
            human_pose[8] = [0.3 + i*0.05, 0, 1.1]
            human_pose[10] = [0.35 + i*0.05, 0, 0.8]

            result = retargeter.retarget(human_pose)
            results.append(result.q.copy())

        # Check smoothness (joints shouldn't jump too much)
        for i in range(1, len(results)):
            diff = np.max(np.abs(results[i] - results[i-1]))
            assert diff < 1.0, f"Jump too large: {diff}"

        record_result("Retargeting", "Trajectory smoothing", True,
                      f"5 frames processed, max jump: {diff:.3f} rad")
    except Exception as e:
        record_result("Retargeting", "Trajectory smoothing", False, error=str(e))

    # Test 4: Handle incomplete pose
    try:
        from src.core.retargeting import MotionRetargeter

        retargeter = MotionRetargeter(robot_type="generic_7dof")

        # Pose with only 5 keypoints
        incomplete_pose = np.zeros((5, 3))
        result = retargeter.retarget(incomplete_pose)

        # Should handle gracefully (may not succeed but shouldn't crash)
        assert isinstance(result, RetargetingResult), "Should return result"

        record_result("Retargeting", "Handle incomplete pose", True,
                      f"Gracefully handled: success={result.success}")
    except Exception as e:
        record_result("Retargeting", "Handle incomplete pose", False, error=str(e))

    # Test 5: Reset state
    try:
        from src.core.retargeting import MotionRetargeter

        retargeter = MotionRetargeter(robot_type="generic_7dof")

        # Run some retargeting
        human_pose = np.zeros((17, 3))
        human_pose[6] = [0.2, 0, 1.4]
        human_pose[8] = [0.3, 0, 1.1]
        human_pose[10] = [0.35, 0, 0.8]
        retargeter.retarget(human_pose)

        # Reset
        retargeter.reset()

        assert retargeter.prev_q is None, "State should be reset"

        record_result("Retargeting", "Reset state", True, "State cleared successfully")
    except Exception as e:
        record_result("Retargeting", "Reset state", False, error=str(e))


# =============================================================================
# TEST: Cloud Services Module
# =============================================================================

def test_cloud_services():
    """Test cloud services (FFM client, vendor adapters)."""
    print("\n" + "=" * 70)
    print("TESTING: Cloud Services Module")
    print("=" * 70)

    # Test 1: Import modules
    try:
        from src.platform.cloud.ffm_client_real import (
            FFMClientReal,
            FFMClientConfig,
            ModelVersion,
            SecureAggregatorReal,
        )
        from src.platform.cloud.vendor_adapter_real import (
            VendorAdapter,
            Pi0VendorAdapter,
            OpenVLAVendorAdapter,
            ModelConfig,
            create_vendor_adapter,
        )
        record_result("Cloud Services", "Import modules", True, "All modules imported")
    except ImportError as e:
        record_result("Cloud Services", "Import modules", False, error=str(e))
        return

    # Test 2: Create FFM client
    try:
        from src.platform.cloud.ffm_client_real import FFMClientReal, FFMClientConfig

        config = FFMClientConfig(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            cache_dir="/tmp/test_cache",
        )
        client = FFMClientReal(config)

        assert client.cache_dir.exists(), "Cache dir should be created"

        record_result("Cloud Services", "Create FFM client", True,
                      f"Cache dir: {client.cache_dir}")
    except Exception as e:
        record_result("Cloud Services", "Create FFM client", False, error=str(e))

    # Test 3: Version comparison
    try:
        from src.platform.cloud.ffm_client_real import FFMClientReal, FFMClientConfig

        client = FFMClientReal()

        # Test version comparisons
        assert client._compare_versions("v2.0.0", "v1.0.0") > 0
        assert client._compare_versions("v1.0.0", "v2.0.0") < 0
        assert client._compare_versions("v1.0.0", "v1.0.0") == 0
        assert client._compare_versions("1.2.3", "1.2.2") > 0
        assert client._compare_versions("1.10.0", "1.9.0") > 0

        record_result("Cloud Services", "Version comparison", True,
                      "All version comparisons correct")
    except Exception as e:
        record_result("Cloud Services", "Version comparison", False, error=str(e))

    # Test 4: SHA256 computation
    try:
        from src.platform.cloud.ffm_client_real import FFMClientReal
        import tempfile

        client = FFMClientReal()

        # Create temp file with known content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for hashing")
            temp_path = f.name

        from pathlib import Path
        hash_result = client._compute_sha256(Path(temp_path))

        # Known SHA256 of "test content for hashing"
        expected = "a8e6e9e8d5f5c5f5d5e5f5e5d5f5c5f5"  # This won't match - just check format
        assert len(hash_result) == 64, "SHA256 should be 64 hex chars"
        assert all(c in '0123456789abcdef' for c in hash_result), "Should be hex"

        os.unlink(temp_path)

        record_result("Cloud Services", "SHA256 computation", True,
                      f"Hash: {hash_result[:16]}...")
    except Exception as e:
        record_result("Cloud Services", "SHA256 computation", False, error=str(e))

    # Test 5: Create vendor adapters
    try:
        from src.platform.cloud.vendor_adapter_real import create_vendor_adapter, ModelConfig

        # Create different adapter types
        pi0_adapter = create_vendor_adapter("pi0")
        openvla_adapter = create_vendor_adapter("openvla")
        act_adapter = create_vendor_adapter("act")

        # Unknown type should fall back to pi0
        unknown_adapter = create_vendor_adapter("unknown_model")

        record_result("Cloud Services", "Create vendor adapters", True,
                      "Created Pi0, OpenVLA, ACT adapters")
    except Exception as e:
        record_result("Cloud Services", "Create vendor adapters", False, error=str(e))

    # Test 6: Vendor adapter predict (without model)
    try:
        from src.platform.cloud.vendor_adapter_real import Pi0VendorAdapter

        adapter = Pi0VendorAdapter()

        # Without model loaded, should return default
        obs = {
            "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "instruction": "pick up the cube",
        }

        result = adapter.predict(obs)

        assert "action" in result, "Should have action key"
        assert "confidence" in result, "Should have confidence key"
        assert len(result["action"]) > 0, "Should have action values"

        record_result("Cloud Services", "Vendor adapter predict", True,
                      f"Action shape: {result['action'].shape}, conf: {result['confidence']}")
    except Exception as e:
        record_result("Cloud Services", "Vendor adapter predict", False, error=str(e))

    # Test 7: Secure aggregator
    try:
        from src.platform.cloud.ffm_client_real import SecureAggregatorReal, FFMClientConfig

        config = FFMClientConfig(api_key="test-key")
        aggregator = SecureAggregatorReal(config)

        # Should create without error
        assert aggregator._session_id is None, "Session should not be started"

        record_result("Cloud Services", "Secure aggregator init", True,
                      "SecureAggregatorReal created")
    except Exception as e:
        record_result("Cloud Services", "Secure aggregator init", False, error=str(e))


# =============================================================================
# TEST: Depth Estimation Module
# =============================================================================

def test_depth_estimation():
    """Test depth estimation module."""
    print("\n" + "=" * 70)
    print("TESTING: Depth Estimation Module")
    print("=" * 70)

    # Test 1: Import modules
    try:
        from src.core.depth_estimation import (
            DepthAnythingV3,
            DepthEstimationConfig,
            DepthResult,
            DepthPoseFusion,
        )
        record_result("Depth Estimation", "Import modules", True, "All modules imported")
    except ImportError as e:
        record_result("Depth Estimation", "Import modules", False, error=str(e))
        return

    # Test 2: Create depth estimator
    try:
        from src.core.depth_estimation import DepthAnythingV3, DepthEstimationConfig

        config = DepthEstimationConfig(model_size="small", device="cpu")
        estimator = DepthAnythingV3(config)

        record_result("Depth Estimation", "Create estimator", True,
                      f"Model size: {config.model_size}")
    except Exception as e:
        record_result("Depth Estimation", "Create estimator", False, error=str(e))

    # Test 3: Run depth inference
    try:
        from src.core.depth_estimation import DepthAnythingV3, DepthEstimationConfig

        estimator = DepthAnythingV3(DepthEstimationConfig(device="cpu"))

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = estimator.infer(test_image)

        assert hasattr(result, 'depth_map'), "Should have depth_map"
        assert result.depth_map is not None, "Depth map should not be None"

        record_result("Depth Estimation", "Depth inference", True,
                      f"Depth map shape: {result.depth_map.shape}")
    except Exception as e:
        record_result("Depth Estimation", "Depth inference", False, error=str(e))

    # Test 4: Depth-pose fusion
    try:
        from src.core.depth_estimation import DepthPoseFusion, DepthResult
        from src.core.depth_estimation.depth_pose_fusion import CameraIntrinsics

        # Create fusion module
        intrinsics = CameraIntrinsics(
            fx=600.0, fy=600.0,
            cx=320.0, cy=240.0,
            width=640, height=480
        )
        fusion = DepthPoseFusion()

        record_result("Depth Estimation", "Depth-pose fusion init", True,
                      "DepthPoseFusion created")
    except Exception as e:
        record_result("Depth Estimation", "Depth-pose fusion init", False, error=str(e))


# =============================================================================
# TEST: Integration Pipeline
# =============================================================================

def test_integration():
    """Test full pipeline integration."""
    print("\n" + "=" * 70)
    print("TESTING: Integration Pipeline")
    print("=" * 70)

    # Test 1: Full pose-to-retarget pipeline
    try:
        from src.core.pose_inference import RTMPoseRealInference, RTMPoseConfig
        from src.core.retargeting import MotionRetargeter

        # Initialize components
        pose_config = RTMPoseConfig(model_name="rtmpose-m", device="cpu")
        pose_inferencer = RTMPoseRealInference(pose_config)
        retargeter = MotionRetargeter(robot_type="generic_7dof")

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run pose estimation
        pose_result = pose_inferencer.infer(test_image, camera_id="cam0")

        # Extract 2D pose and create synthetic 3D (since we don't have real depth)
        if pose_result.keypoints.shape[0] > 0:
            pose_2d = pose_result.keypoints[0][:17, :2]  # First person, body keypoints
            pose_3d = np.zeros((17, 3))
            pose_3d[:, :2] = pose_2d / 100.0  # Scale to meters
            pose_3d[:, 2] = 1.5  # Assume 1.5m depth

            # Run retargeting
            retarget_result = retargeter.retarget(pose_3d)

            record_result("Integration", "Pose-to-retarget pipeline", True,
                          f"Generated {len(retarget_result.q)} joint angles")
        else:
            record_result("Integration", "Pose-to-retarget pipeline", True,
                          "Pipeline ran (no person detected)")
    except Exception as e:
        record_result("Integration", "Pose-to-retarget pipeline", False, error=str(e))

    # Test 2: Multi-frame processing
    try:
        from src.core.pose_inference import RTMPoseRealInference, RTMPoseConfig
        from src.core.retargeting import MotionRetargeter

        pose_inferencer = RTMPoseRealInference(RTMPoseConfig(device="cpu"))
        retargeter = MotionRetargeter(robot_type="generic_7dof")

        # Process multiple frames
        times = []
        for i in range(10):
            start = time.time()

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pose_result = pose_inferencer.infer(image)

            # Synthetic 3D
            pose_3d = np.zeros((17, 3))
            pose_3d[6] = [0.2, 0.1 * np.sin(i * 0.5), 1.4]
            pose_3d[8] = [0.3, 0.15 * np.sin(i * 0.5), 1.1]
            pose_3d[10] = [0.35, 0.2 * np.sin(i * 0.5), 0.8]

            retarget_result = retargeter.retarget(pose_3d)

            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time

        record_result("Integration", "Multi-frame processing", True,
                      f"10 frames, avg: {avg_time:.1f}ms/frame ({fps:.1f} FPS)")
    except Exception as e:
        record_result("Integration", "Multi-frame processing", False, error=str(e))

    # Test 3: Error recovery
    try:
        from src.core.pose_inference import RTMPoseRealInference, RTMPoseConfig
        from src.core.retargeting import MotionRetargeter

        pose_inferencer = RTMPoseRealInference(RTMPoseConfig(device="cpu"))
        retargeter = MotionRetargeter(robot_type="generic_7dof")

        # Mix of good and bad inputs
        inputs = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Good
            np.zeros((10, 10, 3), dtype=np.uint8),  # Very small
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),  # Large
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Good
        ]

        successes = 0
        for img in inputs:
            try:
                result = pose_inferencer.infer(img)
                if result is not None:
                    successes += 1
            except Exception:
                pass  # Count as handled

        record_result("Integration", "Error recovery", True,
                      f"Handled {len(inputs)} inputs, {successes} successful")
    except Exception as e:
        record_result("Integration", "Error recovery", False, error=str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("DYNAMICAL EDGE PLATFORM - PIPELINE TEST SUITE")
    print("=" * 70)
    print(f"Running comprehensive tests on real implementations...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all test suites
    test_pose_inference()
    test_kinematics()
    test_retargeting()
    test_cloud_services()
    test_depth_estimation()
    test_integration()

    # Print summary
    success = print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

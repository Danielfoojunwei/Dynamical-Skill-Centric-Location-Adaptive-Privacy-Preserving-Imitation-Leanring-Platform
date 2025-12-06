"""
Unit tests for GMR whole-body retargeting module.

These tests verify that:
1. RTMW3D to GMR joint mapping runs without errors
2. Joint conversion produces finite values
3. MockGMRRetargeter produces reasonable outputs
4. WholeBodyRetargeterGMR handles various inputs correctly
"""

import numpy as np
from pathlib import Path
import unittest

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.whole_body_gmr import (
    WholeBodyRetargeterGMR,
    convert_rtmw3d_to_gmr_format,
    RTMW3D_BODY_JOINTS,
    build_rtmw3d_to_gmr_joint_map,
    fuse_whole_body_with_current,
    create_whole_body_retargeter,
    MockGMRRetargeter,
    GMR_JOINT_ORDER,
    GMRConfig,
)
from src.core.human_state import Human3DState


class TestJointMapping(unittest.TestCase):
    """Tests for RTMW3D to GMR joint mapping."""
    
    def test_mapping_covers_body_joints(self):
        """Mapping should cover all RTMW3D body joints."""
        mapping = build_rtmw3d_to_gmr_joint_map()
        
        # All RTMW3D joints should map to something
        for rtmw_idx, rtmw_name in RTMW3D_BODY_JOINTS.items():
            if rtmw_name in GMR_JOINT_ORDER:
                self.assertIn(rtmw_idx, mapping, 
                    f"RTMW3D joint {rtmw_name} should be in mapping")
    
    def test_mapping_indices_valid(self):
        """GMR indices in mapping should be valid."""
        mapping = build_rtmw3d_to_gmr_joint_map()
        
        for rtmw_idx, gmr_idx in mapping.items():
            self.assertGreaterEqual(gmr_idx, 0)
            self.assertLess(gmr_idx, len(GMR_JOINT_ORDER))


class TestJointConversion(unittest.TestCase):
    """Tests for convert_rtmw3d_to_gmr_format function."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        keypoints_3d = np.random.randn(17, 3)
        keypoint_confidence = np.random.rand(17)
        
        gmr_pos, gmr_conf = convert_rtmw3d_to_gmr_format(
            keypoints_3d, keypoint_confidence
        )
        
        self.assertEqual(gmr_pos.shape, (22, 3), "GMR positions should be [22, 3]")
        self.assertEqual(gmr_conf.shape, (22,), "GMR confidence should be [22]")
    
    def test_output_finite(self):
        """Output should be finite (no NaN or Inf)."""
        keypoints_3d = np.random.randn(17, 3)
        keypoint_confidence = np.random.rand(17)
        
        gmr_pos, gmr_conf = convert_rtmw3d_to_gmr_format(
            keypoints_3d, keypoint_confidence
        )
        
        self.assertTrue(np.all(np.isfinite(gmr_pos)), "GMR positions should be finite")
        self.assertTrue(np.all(np.isfinite(gmr_conf)), "GMR confidence should be finite")
    
    def test_zero_input(self):
        """Should handle zero input."""
        keypoints_3d = np.zeros((17, 3))
        keypoint_confidence = np.ones(17)
        
        gmr_pos, gmr_conf = convert_rtmw3d_to_gmr_format(
            keypoints_3d, keypoint_confidence
        )
        
        # Should still produce valid output
        self.assertEqual(gmr_pos.shape, (22, 3))
        self.assertTrue(np.all(np.isfinite(gmr_pos)))
    
    def test_preserves_known_joints(self):
        """Known joint mappings should preserve positions."""
        keypoints_3d = np.arange(17 * 3).reshape(17, 3).astype(float)
        keypoint_confidence = np.ones(17)
        
        gmr_pos, gmr_conf = convert_rtmw3d_to_gmr_format(
            keypoints_3d, keypoint_confidence
        )
        
        # Pelvis should match (RTMW3D idx 0 -> GMR idx 0)
        np.testing.assert_array_equal(
            gmr_pos[0], keypoints_3d[0],
            "Pelvis position should be preserved"
        )


class TestMockGMRRetargeter(unittest.TestCase):
    """Tests for MockGMRRetargeter."""
    
    def test_output_shape(self):
        """Output should match n_joints."""
        retargeter = MockGMRRetargeter(n_joints=7)
        
        human_motion = np.random.randn(22, 3)
        q = retargeter.retarget(human_motion)
        
        self.assertEqual(q.shape, (7,), "Output should be [n_joints]")
    
    def test_output_finite(self):
        """Output should be finite."""
        retargeter = MockGMRRetargeter(n_joints=7)
        
        human_motion = np.random.randn(22, 3)
        q = retargeter.retarget(human_motion)
        
        self.assertTrue(np.all(np.isfinite(q)), "Output should be finite")
    
    def test_different_n_joints(self):
        """Should work with different joint counts."""
        for n in [3, 7, 14, 26]:
            retargeter = MockGMRRetargeter(n_joints=n)
            q = retargeter.retarget(np.random.randn(22, 3))
            self.assertEqual(len(q), n)


class TestWholeBodyRetargeterGMR(unittest.TestCase):
    """Tests for WholeBodyRetargeterGMR class."""
    
    def setUp(self):
        """Create retargeter for tests."""
        self.retargeter = WholeBodyRetargeterGMR(n_robot_joints=7)
    
    def test_human_pose_to_robot_q(self):
        """Should convert human pose to robot joints."""
        human_3d = Human3DState(
            timestamp=0.0,
            keypoints_3d=np.random.randn(17, 3),
            keypoint_confidence=np.random.rand(17),
        )
        
        q = self.retargeter.human_pose_to_robot_q(human_3d)
        
        self.assertEqual(q.shape, (7,))
        self.assertTrue(np.all(np.isfinite(q)))
    
    def test_temporal_smoothing(self):
        """Sequential calls should produce smooth output."""
        # Create a trajectory of poses
        base_keypoints = np.random.randn(17, 3)
        
        qs = []
        for i in range(10):
            # Small perturbation from base
            keypoints = base_keypoints + np.random.randn(17, 3) * 0.01
            
            human_3d = Human3DState(
                timestamp=i * 0.05,
                keypoints_3d=keypoints,
                keypoint_confidence=np.ones(17) * 0.9,
            )
            
            q = self.retargeter.human_pose_to_robot_q(human_3d, apply_smoothing=True)
            qs.append(q)
        
        # Check that consecutive frames are similar
        qs = np.array(qs)
        diffs = np.diff(qs, axis=0)
        max_diff = np.max(np.abs(diffs))
        
        self.assertLess(max_diff, 1.0, "Consecutive outputs should be similar")
    
    def test_reset(self):
        """Reset should clear temporal state."""
        human_3d = Human3DState(
            timestamp=0.0,
            keypoints_3d=np.random.randn(17, 3),
            keypoint_confidence=np.ones(17),
        )
        
        # First call
        self.retargeter.human_pose_to_robot_q(human_3d)
        self.assertIsNotNone(self.retargeter.prev_q)
        
        # Reset
        self.retargeter.reset()
        self.assertIsNone(self.retargeter.prev_q)


class TestFuseWholeBodyWithCurrent(unittest.TestCase):
    """Tests for fuse_whole_body_with_current function."""
    
    def test_blend_factor_zero(self):
        """Blend factor 0 should return current state."""
        q_whole = np.array([1.0, 2.0, 3.0])
        q_current = np.array([4.0, 5.0, 6.0])
        
        result = fuse_whole_body_with_current(q_whole, q_current, blend_factor=0.0)
        
        np.testing.assert_array_almost_equal(result, q_current)
    
    def test_blend_factor_one(self):
        """Blend factor 1 should return whole-body state."""
        q_whole = np.array([1.0, 2.0, 3.0])
        q_current = np.array([4.0, 5.0, 6.0])
        
        result = fuse_whole_body_with_current(q_whole, q_current, blend_factor=1.0)
        
        np.testing.assert_array_almost_equal(result, q_whole)
    
    def test_blend_factor_half(self):
        """Blend factor 0.5 should average."""
        q_whole = np.array([0.0, 0.0])
        q_current = np.array([1.0, 1.0])
        
        result = fuse_whole_body_with_current(q_whole, q_current, blend_factor=0.5)
        
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])
    
    def test_per_joint_weights(self):
        """Per-joint weights should blend individually."""
        q_whole = np.array([1.0, 2.0, 3.0])
        q_current = np.array([0.0, 0.0, 0.0])
        joint_weights = np.array([1.0, 0.5, 0.0])  # Full, half, none
        
        result = fuse_whole_body_with_current(
            q_whole, q_current, blend_factor=1.0, joint_weights=joint_weights
        )
        
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 0.0])


class TestCreateWholeBodyRetargeter(unittest.TestCase):
    """Tests for factory function."""
    
    def test_generic_7dof(self):
        """Should create retargeter for generic 7-DOF arm."""
        retargeter = create_whole_body_retargeter(robot_type='generic_7dof')
        
        self.assertIsInstance(retargeter, WholeBodyRetargeterGMR)
        self.assertEqual(retargeter.n_robot_joints, 7)
    
    def test_unknown_robot_type(self):
        """Unknown robot type should fall back to generic."""
        retargeter = create_whole_body_retargeter(robot_type='unknown_robot')
        
        self.assertIsInstance(retargeter, WholeBodyRetargeterGMR)


if __name__ == '__main__':
    unittest.main()

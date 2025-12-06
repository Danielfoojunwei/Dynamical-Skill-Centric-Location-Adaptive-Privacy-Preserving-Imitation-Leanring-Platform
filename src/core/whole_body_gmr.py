"""
GMR (General Motion Retargeting) Whole-Body Wrapper

This module wraps the GMR library (https://github.com/YanjieZe/GMR) to provide
whole-body motion retargeting from RTMW3D-format human poses to robot joint
configurations.

GMR is designed for real-time whole-body retargeting and supports many humanoids
(Unitree H1, Galaxea R1 Pro, Kuavo, etc.). It separates:
    - Human motion format adapters
    - Robot model (URDF/XML)
    - Retargeter core

Key Integration Points:
    1. RTMW3D → GMR joint mapping (different naming/ordering conventions)
    2. GMR → Robot joint configuration
    3. Fallback when GMR is not available

Conceptual Reference:
    - OKAMI: Object-aware retargeting (T_hand_object constraint)
    - GMR handles whole-body posture, OKAMI-style handles manipulation

The retargeting strategy:
    - GMR provides whole-body joint targets as a "soft prior"
    - IK solver then enforces end-effector constraints for manipulation
    - This preserves natural body posture while ensuring accurate grasping
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint Mapping: RTMW3D (COCO-WholeBody) → GMR Human Format
# ---------------------------------------------------------------------------

# RTMW3D uses COCO-WholeBody format with 133 keypoints (17 body + 6 feet + 68 face + 42 hands)
# GMR expects a specific human motion format depending on the source
# This mapping converts from RTMW3D body keypoints to GMR's expected format

RTMW3D_BODY_JOINTS = {
    # RTMW3D index: joint name
    0: 'pelvis',
    1: 'r_hip',
    2: 'r_knee', 
    3: 'r_ankle',
    4: 'l_hip',
    5: 'l_knee',
    6: 'l_ankle',
    7: 'spine',
    8: 'chest',
    9: 'neck',
    10: 'head',
    11: 'l_shoulder',
    12: 'l_elbow',
    13: 'l_wrist',
    14: 'r_shoulder',
    15: 'r_elbow',
    16: 'r_wrist',
}

# GMR expects joints in a specific order (varies by human motion format)
# This is the mapping for SMPL-compatible format
GMR_JOINT_ORDER = [
    'pelvis',       # 0
    'l_hip',        # 1
    'r_hip',        # 2
    'spine',        # 3  (lower spine)
    'l_knee',       # 4
    'r_knee',       # 5
    'chest',        # 6  (mid spine)
    'l_ankle',      # 7
    'r_ankle',      # 8
    'neck',         # 9  (upper spine / neck)
    'l_foot',       # 10 (not in RTMW3D body, use ankle)
    'r_foot',       # 11 (not in RTMW3D body, use ankle)
    'head',         # 12
    'l_collar',     # 13 (not in RTMW3D, interpolate)
    'r_collar',     # 14 (not in RTMW3D, interpolate)
    'head_top',     # 15 (use head)
    'l_shoulder',   # 16
    'r_shoulder',   # 17
    'l_elbow',      # 18
    'r_elbow',      # 19
    'l_wrist',      # 20
    'r_wrist',      # 21
]


def build_rtmw3d_to_gmr_joint_map() -> Dict[int, int]:
    """
    Build mapping from RTMW3D joint indices to GMR joint indices.
    
    Returns:
        Dict mapping RTMW3D index → GMR index
    """
    rtmw3d_name_to_idx = {name: idx for idx, name in RTMW3D_BODY_JOINTS.items()}
    
    mapping = {}
    for gmr_idx, gmr_name in enumerate(GMR_JOINT_ORDER):
        if gmr_name in rtmw3d_name_to_idx:
            mapping[rtmw3d_name_to_idx[gmr_name]] = gmr_idx
    
    return mapping


def convert_rtmw3d_to_gmr_format(
    keypoints_3d: np.ndarray,
    keypoint_confidence: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RTMW3D keypoints to GMR human motion format.
    
    Args:
        keypoints_3d: [17, 3] RTMW3D body keypoints in world frame
        keypoint_confidence: [17] confidence scores (optional)
        
    Returns:
        gmr_positions: [22, 3] positions in GMR joint order
        gmr_confidence: [22] confidence scores
    """
    n_gmr_joints = len(GMR_JOINT_ORDER)
    gmr_positions = np.zeros((n_gmr_joints, 3))
    gmr_confidence = np.zeros(n_gmr_joints)
    
    joint_map = build_rtmw3d_to_gmr_joint_map()
    
    # Map direct correspondences
    for rtmw3d_idx, gmr_idx in joint_map.items():
        if rtmw3d_idx < len(keypoints_3d):
            gmr_positions[gmr_idx] = keypoints_3d[rtmw3d_idx]
            if keypoint_confidence is not None and rtmw3d_idx < len(keypoint_confidence):
                gmr_confidence[gmr_idx] = keypoint_confidence[rtmw3d_idx]
            else:
                gmr_confidence[gmr_idx] = 1.0
    
    # Interpolate missing joints
    # l_foot/r_foot: use ankle positions
    if 'l_ankle' in RTMW3D_BODY_JOINTS.values():
        ankle_idx = [k for k, v in RTMW3D_BODY_JOINTS.items() if v == 'l_ankle'][0]
        gmr_positions[10] = keypoints_3d[ankle_idx]  # l_foot
        gmr_positions[11] = keypoints_3d[3]          # r_foot (r_ankle is index 3)
        gmr_confidence[10] = gmr_confidence[7]
        gmr_confidence[11] = gmr_confidence[8]
    
    # l_collar/r_collar: interpolate between neck and shoulder
    neck_pos = keypoints_3d[9] if len(keypoints_3d) > 9 else np.zeros(3)
    l_shoulder_pos = keypoints_3d[11] if len(keypoints_3d) > 11 else np.zeros(3)
    r_shoulder_pos = keypoints_3d[14] if len(keypoints_3d) > 14 else np.zeros(3)
    
    gmr_positions[13] = 0.5 * neck_pos + 0.5 * l_shoulder_pos  # l_collar
    gmr_positions[14] = 0.5 * neck_pos + 0.5 * r_shoulder_pos  # r_collar
    gmr_confidence[13] = 0.5 * (gmr_confidence[9] + gmr_confidence[16])
    gmr_confidence[14] = 0.5 * (gmr_confidence[9] + gmr_confidence[17])
    
    # head_top: use head position
    gmr_positions[15] = keypoints_3d[10] if len(keypoints_3d) > 10 else np.zeros(3)
    gmr_confidence[15] = gmr_confidence[12]
    
    return gmr_positions, gmr_confidence


# ---------------------------------------------------------------------------
# GMR Configuration
# ---------------------------------------------------------------------------

@dataclass
class GMRConfig:
    """Configuration for GMR-based whole-body retargeting."""
    
    # Robot model paths
    robot_urdf_path: str = ""
    robot_config_path: str = ""
    
    # Retargeting parameters
    use_position_targets: bool = True       # Use position-based retargeting
    use_rotation_targets: bool = True       # Include rotation matching
    
    # Joint groups (which joints to retarget)
    retarget_upper_body: bool = True
    retarget_lower_body: bool = True
    retarget_hands: bool = False            # Usually handled by glove
    
    # IK parameters
    ik_iterations: int = 50
    ik_tolerance: float = 1e-4
    
    # Blending with current state
    blend_factor: float = 0.8               # How much to trust GMR vs current state


# ---------------------------------------------------------------------------
# Mock GMR Interface (when library not available)
# ---------------------------------------------------------------------------

class MockGMRRetargeter:
    """
    Mock GMR retargeter for testing when the actual library is not installed.
    
    Returns plausible joint configurations based on simple kinematic heuristics.
    """
    
    def __init__(self, n_joints: int = 7):
        self.n_joints = n_joints
        logger.warning("Using MockGMRRetargeter - install GMR for real retargeting")
    
    def retarget(self, human_motion: np.ndarray) -> np.ndarray:
        """
        Mock retargeting that returns reasonable joint angles.
        
        Args:
            human_motion: [22, 3] GMR-format human positions
            
        Returns:
            q: [n_joints] robot joint configuration
        """
        # Simple heuristic: map shoulder/elbow/wrist positions to joint angles
        q = np.zeros(self.n_joints)
        
        # This is a placeholder - real GMR would use proper IK
        # Just create some motion based on arm pose
        if len(human_motion) >= 22:
            # Use right arm (indices 17, 19, 21 = shoulder, elbow, wrist)
            r_shoulder = human_motion[17]
            r_elbow = human_motion[19]
            r_wrist = human_motion[21]
            
            # Shoulder to elbow direction gives first few joints
            upper_arm = r_elbow - r_shoulder
            q[0] = np.arctan2(upper_arm[1], upper_arm[0])
            q[1] = np.arctan2(upper_arm[2], np.linalg.norm(upper_arm[:2]))
            
            # Elbow to wrist direction gives elbow angle
            forearm = r_wrist - r_elbow
            q[2] = np.arctan2(forearm[2], np.linalg.norm(forearm[:2]))
            
            # Small random noise for realism
            q += np.random.randn(self.n_joints) * 0.01
        
        return q


# ---------------------------------------------------------------------------
# GMR Wrapper Class
# ---------------------------------------------------------------------------

class WholeBodyRetargeterGMR:
    """
    Wrapper around GMR for whole-body motion retargeting.
    
    This class:
        1. Converts RTMW3D poses to GMR format
        2. Calls GMR's retargeting core
        3. Returns robot joint configurations
    
    Usage:
        retargeter = WholeBodyRetargeterGMR(
            robot_model_path='path/to/robot.urdf',
            config_path='path/to/config.yaml'
        )
        
        # In demo loop:
        q_whole = retargeter.human_pose_to_robot_q(human_3d_state)
    """
    
    def __init__(
        self,
        robot_model_path: str = "",
        config_path: str = "",
        config: GMRConfig = None,
        n_robot_joints: int = 7
    ):
        """
        Initialize GMR wrapper.
        
        Args:
            robot_model_path: Path to robot URDF/XML
            config_path: Path to GMR configuration file
            config: GMRConfig instance (optional)
            n_robot_joints: Number of robot joints (fallback for mock)
        """
        self.config = config or GMRConfig(
            robot_urdf_path=robot_model_path,
            robot_config_path=config_path
        )
        self.n_robot_joints = n_robot_joints
        
        # Try to import and initialize real GMR
        self.gmr = None
        self._init_gmr()
        
        # Build joint mapping
        self.joint_map = build_rtmw3d_to_gmr_joint_map()
        
        # State for temporal filtering
        self.prev_q: Optional[np.ndarray] = None
        self.smoothing_alpha: float = 0.3
    
    def _init_gmr(self):
        """Initialize GMR library if available."""
        try:
            # Try to import GMR
            # Note: The actual import path depends on how GMR is installed
            # This is a placeholder for the real import
            from gmr import GeneralMotionRetargeting  # type: ignore
            
            self.gmr = GeneralMotionRetargeting(
                robot_model=self.config.robot_urdf_path,
                config=self.config.robot_config_path
            )
            logger.info("GMR library initialized successfully")
            
        except ImportError:
            logger.warning(
                "GMR library not found. Install from https://github.com/YanjieZe/GMR\n"
                "Using mock retargeter for testing."
            )
            self.gmr = MockGMRRetargeter(n_joints=self.n_robot_joints)
        
        except Exception as e:
            logger.error(f"Failed to initialize GMR: {e}")
            self.gmr = MockGMRRetargeter(n_joints=self.n_robot_joints)
    
    def human_pose_to_robot_q(
        self,
        human_3d_state: 'Human3DState',
        apply_smoothing: bool = True
    ) -> np.ndarray:
        """
        Convert RTMW3D 3D keypoints to robot joint configuration.
        
        Args:
            human_3d_state: Human3DState with keypoints_3d and keypoint_confidence
            apply_smoothing: Whether to apply temporal smoothing
            
        Returns:
            q_whole: [n_joints] robot joint configuration
        """
        # Convert RTMW3D format to GMR format
        gmr_positions, gmr_confidence = convert_rtmw3d_to_gmr_format(
            human_3d_state.keypoints_3d,
            human_3d_state.keypoint_confidence
        )
        
        # Call GMR retargeting
        if isinstance(self.gmr, MockGMRRetargeter):
            q_whole = self.gmr.retarget(gmr_positions)
        else:
            # Real GMR interface
            try:
                q_whole = self.gmr.retarget(gmr_positions)
            except Exception as e:
                logger.warning(f"GMR retarget failed: {e}, using fallback")
                q_whole = np.zeros(self.n_robot_joints)
        
        # Apply temporal smoothing
        if apply_smoothing and self.prev_q is not None:
            q_whole = self.smoothing_alpha * q_whole + (1 - self.smoothing_alpha) * self.prev_q
        
        self.prev_q = q_whole.copy()
        
        return q_whole
    
    def reset(self):
        """Reset temporal state (call between episodes)."""
        self.prev_q = None


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def fuse_whole_body_with_current(
    q_whole: np.ndarray,
    q_current: np.ndarray,
    blend_factor: float = 0.8,
    joint_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Fuse GMR whole-body configuration with current robot state.
    
    This allows smooth transitions and prevents sudden jumps when
    GMR produces significantly different configurations.
    
    Args:
        q_whole: Joint configuration from GMR
        q_current: Current robot joint configuration
        blend_factor: How much to trust GMR (0=current, 1=GMR)
        joint_weights: Per-joint blend factors (optional)
        
    Returns:
        q_fused: Blended joint configuration
    """
    if joint_weights is not None:
        # Per-joint blending
        weights = np.clip(joint_weights * blend_factor, 0, 1)
        q_fused = weights * q_whole + (1 - weights) * q_current
    else:
        # Uniform blending
        q_fused = blend_factor * q_whole + (1 - blend_factor) * q_current
    
    return q_fused


def create_whole_body_retargeter(
    robot_type: str = 'generic_7dof',
    urdf_path: str = "",
    config_path: str = ""
) -> WholeBodyRetargeterGMR:
    """
    Factory function to create a whole-body retargeter for common robots.
    
    Args:
        robot_type: Type of robot ('unitree_h1', 'galaxea_r1', 'generic_7dof')
        urdf_path: Override URDF path
        config_path: Override config path
        
    Returns:
        Configured WholeBodyRetargeterGMR instance
    """
    # Default configurations for supported robots
    robot_configs = {
        'unitree_h1': {
            'n_joints': 26,  # Full humanoid
            'urdf': 'models/unitree_h1.urdf',
            'config': 'configs/h1_retarget.yaml'
        },
        'galaxea_r1': {
            'n_joints': 22,
            'urdf': 'models/galaxea_r1.urdf', 
            'config': 'configs/r1_retarget.yaml'
        },
        'generic_7dof': {
            'n_joints': 7,
            'urdf': '',  # No specific URDF
            'config': ''
        }
    }
    
    cfg = robot_configs.get(robot_type, robot_configs['generic_7dof'])
    
    return WholeBodyRetargeterGMR(
        robot_model_path=urdf_path or cfg['urdf'],
        config_path=config_path or cfg['config'],
        n_robot_joints=cfg['n_joints']
    )

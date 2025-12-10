"""
Core data structures for human state representation.

The key insight here is that we separate:
1. Raw perception outputs (Human3DState, DexterHandState)
2. Fused state with semantic meaning (HumanState)
3. Object information needed for retargeting (EnvObject)

All poses are represented as 4x4 homogeneous transforms or 7D vectors (xyz + quaternion xyzw).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


@dataclass
class EnvObject:
    """
    Represents a detected object in the environment.
    
    The object frame convention matters for retargeting:
    - Origin at object centroid
    - Z-up for most objects (cups, boxes)
    - X along the "natural grasp approach" direction if applicable
    
    For a cup: Z points up along the cup axis, X points toward the handle (if any).
    For a box: Z up, X along the longest horizontal edge.
    """
    object_id: str                          # Unique ID for this object instance
    class_id: int                           # Semantic class (0=unknown, 1=cup, 2=box, etc.)
    class_name: str                         # Human-readable class name
    pose_world: np.ndarray                  # 4x4 transform: object frame → world frame
    bbox_3d: Optional[np.ndarray] = None   # 8x3 array of bounding box corners (object frame)
    confidence: float = 1.0                 # Detection confidence
    
    def __post_init__(self):
        assert self.pose_world.shape == (4, 4), "pose_world must be 4x4 homogeneous transform"


@dataclass
class FrameQuality:
    """
    Per-frame quality metrics for data filtering.
    
    Used to filter out low-quality frames during recording and chunking.
    High-quality data improves policy learning.
    """
    pose_confidence: float          # Mean keypoint confidence [0, 1]
    reprojection_error: float       # Mean pixel error across cameras
    smoothness_score: float         # Motion smoothness [0, 1], 1 = very smooth
    overall_score: float            # Aggregated quality [0, 1]
    
    @classmethod
    def default(cls) -> 'FrameQuality':
        """Create default quality (assume good if not computed)."""
        return cls(
            pose_confidence=1.0,
            reprojection_error=0.0,
            smoothness_score=1.0,
            overall_score=1.0
        )


@dataclass  
class Human3DState:
    """
    Output from the RTMW3D pose estimator.
    
    Joint layout follows COCO-WholeBody convention (133 keypoints), but we primarily
    care about a subset for retargeting:
    - Torso: pelvis, spine, chest, neck, head
    - Right arm: r_shoulder, r_elbow, r_wrist
    - Left arm: l_shoulder, l_elbow, l_wrist
    
    Hand keypoints from RTMW3D are coarse (21 per hand). For fine finger tracking,
    we rely on the DextaGlove and align it using the wrist keypoint as an anchor.
    """
    timestamp: float                        # Unix timestamp
    keypoints_3d: np.ndarray               # [N_joints, 3] in world frame (meters)
    keypoint_confidence: np.ndarray        # [N_joints] confidence scores from RTMW3D
    
    # 2D keypoints per camera for reprojection error computation
    keypoints_2d_per_cam: Optional[Dict[str, np.ndarray]] = None  # {cam_id: [N_joints, 2]}
    
    # Derived quantities (computed once, cached)
    _wrist_right_world: Optional[np.ndarray] = field(default=None, repr=False)
    _wrist_left_world: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Joint indices for COCO-WholeBody (subset we care about)
    JOINT_NAMES: Dict[str, int] = field(default_factory=lambda: {
        'pelvis': 0,
        'r_hip': 1, 'r_knee': 2, 'r_ankle': 3,
        'l_hip': 4, 'l_knee': 5, 'l_ankle': 6,
        'spine': 7, 'chest': 8, 'neck': 9, 'head': 10,
        'l_shoulder': 11, 'l_elbow': 12, 'l_wrist': 13,
        'r_shoulder': 14, 'r_elbow': 15, 'r_wrist': 16,
        # Hand keypoints start at index 91 (right) and 112 (left) in full COCO-WholeBody
    }, repr=False)
    
    def get_joint(self, name: str) -> np.ndarray:
        """Get 3D position of a named joint."""
        idx = self.JOINT_NAMES[name]
        return self.keypoints_3d[idx]
    
    @property
    def wrist_right_position(self) -> np.ndarray:
        """Right wrist position in world frame [x, y, z]."""
        return self.get_joint('r_wrist')
    
    @property
    def wrist_left_position(self) -> np.ndarray:
        """Left wrist position in world frame [x, y, z]."""
        return self.get_joint('l_wrist')
    
    def get_arm_direction(self, side: str = 'right') -> np.ndarray:
        """
        Get unit vector pointing from elbow to wrist.
        Useful for estimating hand orientation when glove data is noisy.
        """
        if side == 'right':
            elbow = self.get_joint('r_elbow')
            wrist = self.get_joint('r_wrist')
        else:
            elbow = self.get_joint('l_elbow')
            wrist = self.get_joint('l_wrist')
        
        direction = wrist - elbow
        return direction / (np.linalg.norm(direction) + 1e-8)


@dataclass
class DexterHandState:
    """
    Output from the DextaGlove driver.
    
    The glove provides:
    1. Finger joint angles (proximal, intermediate, distal for each finger)
    2. Finger abduction angles (spread between fingers)
    3. Wrist orientation (quaternion)
    4. Contact/force sensing per fingertip (if available)
    
    Frame convention: glove reports wrist orientation relative to its own IMU frame.
    We need to align this with the world frame using the camera-observed wrist position.
    """
    timestamp: float                        # Unix timestamp
    side: str                               # 'left' or 'right'
    
    # Finger joint angles in radians [5 fingers x 3 joints]
    # Order: thumb, index, middle, ring, pinky
    # Per finger: [proximal, intermediate, distal]
    finger_angles: np.ndarray              # [5, 3]
    
    # Finger abduction/adduction in radians [4 values]
    # Spread between adjacent fingers: [thumb-index, index-middle, middle-ring, ring-pinky]
    finger_abduction: np.ndarray           # [4]
    
    # Wrist orientation as quaternion [x, y, z, w] in glove IMU frame
    wrist_quat_local: np.ndarray           # [4]
    
    # Optional: fingertip contact forces in Newtons (if glove supports it)
    fingertip_forces: Optional[np.ndarray] = None  # [5] or None
    
    # Calibration: transform from glove IMU frame to world frame
    # This is set during the calibration phase
    T_world_glove_imu: Optional[np.ndarray] = None  # [4, 4]
    
    def __post_init__(self):
        assert self.finger_angles.shape == (5, 3), "finger_angles must be [5, 3]"
        assert self.finger_abduction.shape == (4,), "finger_abduction must be [4]"
        assert self.wrist_quat_local.shape == (4,), "wrist_quat_local must be [4] quaternion"
    
    @property
    def is_calibrated(self) -> bool:
        return self.T_world_glove_imu is not None
    
    def get_finger_closure(self) -> np.ndarray:
        """
        Compute a scalar closure value for each finger in [0, 1].
        0 = fully extended, 1 = fully flexed.
        
        This is a simplified mapping; real applications might want
        a more sophisticated model of finger pose.
        """
        # Sum of joint angles, normalized by typical max flexion
        # Typical max flexion: ~90° proximal, ~100° intermediate, ~70° distal
        max_flexion = np.array([1.57, 1.75, 1.22])  # radians
        closure = np.sum(self.finger_angles / max_flexion, axis=1) / 3.0
        return np.clip(closure, 0.0, 1.0)
    
    def get_grasp_aperture(self) -> float:
        """
        Estimate overall hand aperture (how open/closed the hand is).
        Returns value in [0, 1] where 0 = fully open, 1 = fully closed.
        
        For mapping to a parallel-jaw gripper, this single scalar is often sufficient.
        """
        # Weight fingers by their importance in power grasps
        # Thumb and index matter most for precision, all matter for power grasp
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        finger_closures = self.get_finger_closure()
        return float(np.dot(weights, finger_closures))
    
    def has_contact(self, threshold: float = 0.5) -> bool:
        """
        Check if any fingertip is in contact (force above threshold).
        """
        if self.fingertip_forces is None:
            # No force sensing; estimate from closure
            return self.get_grasp_aperture() > 0.7
        return bool(np.any(self.fingertip_forces > threshold))


@dataclass
class HumanState:
    """
    Fused state combining body pose, hand pose, and environmental context.
    
    This is the input to the retargeting module. It represents everything
    we know about the human demonstrator at a single timestep.
    """
    timestamp: float
    
    # Body pose from camera
    body: Human3DState
    
    # Hand pose from glove (could be None if glove disconnected)
    hand_right: Optional[DexterHandState] = None
    hand_left: Optional[DexterHandState] = None
    
    # Detected objects in scene
    objects: List[EnvObject] = field(default_factory=list)
    
    # The "primary" object being manipulated (set by task context or heuristics)
    primary_object_id: Optional[str] = None
    
    # Frame quality metrics for filtering
    quality: Optional[FrameQuality] = None
    
    @property
    def primary_object(self) -> Optional[EnvObject]:
        """Get the primary object being manipulated, if any."""
        if self.primary_object_id is None:
            return None
        for obj in self.objects:
            if obj.object_id == self.primary_object_id:
                return obj
        return None
    
    def get_active_hand(self) -> Optional[DexterHandState]:
        """
        Get the hand that is currently active (closer to primary object or in contact).
        Falls back to right hand if no clear signal.
        """
        if self.primary_object is not None:
            obj_pos = self.primary_object.pose_world[:3, 3]
            
            r_dist = np.linalg.norm(self.body.wrist_right_position - obj_pos)
            l_dist = np.linalg.norm(self.body.wrist_left_position - obj_pos)
            
            if r_dist < l_dist and self.hand_right is not None:
                return self.hand_right
            elif self.hand_left is not None:
                return self.hand_left
        
        # Default to right hand
        return self.hand_right if self.hand_right is not None else self.hand_left


def fuse_to_human_state(
    human3d: Human3DState,
    hand_right: Optional[DexterHandState],
    hand_left: Optional[DexterHandState],
    objects: List[EnvObject],
    primary_object_id: Optional[str] = None
) -> HumanState:
    """
    Fuse perception outputs into a unified HumanState.
    
    The main job here is aligning the glove's local coordinate frame with
    the camera-observed wrist position. The glove gives us orientation from
    its IMU, and the camera gives us position. We combine these.
    
    If primary_object_id is not specified, we use a simple heuristic:
    pick the object closest to either hand.
    """
    # Align glove frames to world using camera wrist positions
    # (This assumes calibration has been done; see calibration module)
    
    # Auto-select primary object if not specified
    if primary_object_id is None and objects:
        min_dist = float('inf')
        for obj in objects:
            obj_pos = obj.pose_world[:3, 3]
            
            r_dist = np.linalg.norm(human3d.wrist_right_position - obj_pos)
            l_dist = np.linalg.norm(human3d.wrist_left_position - obj_pos)
            
            dist = min(r_dist, l_dist)
            if dist < min_dist:
                min_dist = dist
                primary_object_id = obj.object_id
    
    return HumanState(
        timestamp=human3d.timestamp,
        body=human3d,
        hand_right=hand_right,
        hand_left=hand_left,
        objects=objects,
        primary_object_id=primary_object_id
    )

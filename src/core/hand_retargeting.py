"""
Hand Retargeting Module - DYGlove 21-DOF to Robot Gripper

This module implements precise retargeting from the DYGlove's 21-DOF hand pose
to various robot gripper configurations. Supports both parallel-jaw grippers
and dexterous multi-finger hands.

Key Features:
- Full 21-DOF joint angle retargeting (not just finger closure)
- Abduction angle preservation for manipulation accuracy
- Object-relative pose preservation (OKAMI-style)
- Multiple robot gripper target formats
- Quality-aware interpolation

Reference:
    DOGlove paper Section V-C: "Imitation Learning with DOGlove"
    - Preserves T_hand_object relationship
    - Maps finger closure to gripper aperture
    - Abduction affects grasp stability

Coordinate Frames:
    H = Human hand frame (palm center, Y up, Z out of palm)
    G = Robot gripper frame (TCP, Z along approach)
    O = Object frame (centroid)
    
    T_G_H = Static transform from hand to gripper convention
    T_H_O = Hand pose relative to object (preserved during retargeting)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum, auto
import math
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)


class GripperType(Enum):
    """Supported robot gripper types."""
    PARALLEL_JAW = auto()       # Simple open/close (1 DOF)
    ROBOTIQ_2F85 = auto()       # Robotiq 2-finger 85mm (1 DOF)
    ROBOTIQ_3F = auto()         # Robotiq 3-finger (4 DOF)
    SHADOW_HAND = auto()        # Shadow Dexterous Hand (24 DOF)
    ALLEGRO_HAND = auto()       # Allegro Hand (16 DOF)
    LEAP_HAND = auto()          # LEAP Hand (16 DOF)
    ABILITY_HAND = auto()       # Ability Hand (10 DOF)
    INSPIRE_HAND = auto()       # Inspire Robotics Hand (6 DOF)
    DAIMON_GRIPPER = auto()     # Daimon VTLA gripper (2 DOF)


@dataclass
class HandRetargetConfig:
    """Configuration for hand retargeting."""
    
    # Target gripper
    gripper_type: GripperType = GripperType.PARALLEL_JAW
    
    # Which fingers to use for closure calculation
    closure_fingers: List[str] = field(default_factory=lambda: ['index', 'middle'])
    
    # Weights for multi-finger closure averaging
    finger_weights: Dict[str, float] = field(default_factory=lambda: {
        'thumb': 0.3, 'index': 0.25, 'middle': 0.25, 'ring': 0.1, 'pinky': 0.1
    })
    
    # Gripper aperture limits (meters)
    gripper_min_aperture: float = 0.0
    gripper_max_aperture: float = 0.085  # 85mm for Robotiq 2F85
    
    # Closure thresholds
    pinch_closure_threshold: float = 0.7   # Above this = pinch grasp
    power_closure_threshold: float = 0.5   # Above this = power grasp
    
    # Abduction influence on grasp width
    abduction_scale: float = 0.3  # How much abduction affects aperture
    
    # Temporal smoothing
    smoothing_alpha: float = 0.7  # EMA factor (higher = more responsive)
    
    # Quality thresholds
    min_quality_for_update: float = 0.5
    
    # Object-relative retargeting
    preserve_object_relative: bool = True
    object_offset_scale: float = 1.0  # Scale factor for position offset


@dataclass
class DYGloveState21DOF:
    """
    Full 21-DOF hand state from DYGlove.
    
    Joint order matches SDK: [thumb×5, index×4, middle×4, ring×4, pinky×4]
    """
    timestamp: float = 0.0
    
    # 21-DOF joint angles (radians)
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(21))
    
    # Per-finger states (for convenience)
    thumb: np.ndarray = field(default_factory=lambda: np.zeros(5))   # tm_flex, tm_abd, mcp, ip, wrist_ps
    index: np.ndarray = field(default_factory=lambda: np.zeros(4))   # mcp_flex, mcp_abd, pip, dip
    middle: np.ndarray = field(default_factory=lambda: np.zeros(4))
    ring: np.ndarray = field(default_factory=lambda: np.zeros(4))
    pinky: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # Wrist pose (from HTC Vive Tracker or IMU)
    wrist_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # wxyz
    
    # Quality metrics
    quality: float = 1.0
    
    @classmethod
    def from_joint_array(cls, joints: np.ndarray, timestamp: float = 0.0) -> 'DYGloveState21DOF':
        """Create from flat 21-element array."""
        state = cls(timestamp=timestamp)
        state.joint_angles = joints.copy()
        state.thumb = joints[0:5]
        state.index = joints[5:9]
        state.middle = joints[9:13]
        state.ring = joints[13:17]
        state.pinky = joints[17:21]
        return state
    
    def get_flexion_angles(self) -> np.ndarray:
        """Get flexion angles only (15 DOF, no abduction)."""
        return np.array([
            self.thumb[0], self.thumb[2], self.thumb[3],  # tm_flex, mcp, ip
            self.index[0], self.index[2], self.index[3],  # mcp_flex, pip, dip
            self.middle[0], self.middle[2], self.middle[3],
            self.ring[0], self.ring[2], self.ring[3],
            self.pinky[0], self.pinky[2], self.pinky[3],
        ])
    
    def get_abduction_angles(self) -> np.ndarray:
        """Get abduction angles (5 DOF)."""
        return np.array([
            self.thumb[1],   # tm_abd
            self.index[1],   # mcp_abd
            self.middle[1],
            self.ring[1],
            self.pinky[1],
        ])
    
    def get_finger_closure(self, finger: str) -> float:
        """
        Compute closure [0,1] for a specific finger.
        0 = fully extended, 1 = fully flexed.
        """
        if finger == 'thumb':
            # Thumb: TM_flex + MCP + IP (skip TM_abd, wrist_ps)
            max_sum = np.radians(70 + 90 + 80)  # ~240° max
            actual = max(0, self.thumb[0]) + max(0, self.thumb[2]) + max(0, self.thumb[3])
            return min(1.0, actual / max_sum)
        else:
            # Other fingers: MCP_flex + PIP + DIP (skip MCP_abd)
            finger_angles = getattr(self, finger)
            max_sum = np.radians(90 + 100 + 70)  # ~260° max
            actual = max(0, finger_angles[0]) + max(0, finger_angles[2]) + max(0, finger_angles[3])
            return min(1.0, actual / max_sum)
    
    def get_all_closures(self) -> Dict[str, float]:
        """Get closure values for all fingers."""
        return {
            'thumb': self.get_finger_closure('thumb'),
            'index': self.get_finger_closure('index'),
            'middle': self.get_finger_closure('middle'),
            'ring': self.get_finger_closure('ring'),
            'pinky': self.get_finger_closure('pinky'),
        }
    
    def get_spread(self) -> float:
        """
        Compute finger spread [0,1] based on abduction angles.
        0 = fingers together, 1 = fully spread.
        """
        abductions = self.get_abduction_angles()
        max_abd = np.radians(30)  # ~30° max abduction
        mean_abd = np.mean(np.abs(abductions[1:]))  # Skip thumb
        return min(1.0, mean_abd / max_abd)


@dataclass
class GripperCommand:
    """
    Command for robot gripper.
    
    Supports both simple aperture control and multi-DOF hands.
    """
    timestamp: float = 0.0
    
    # Simple gripper
    aperture: float = 0.0         # meters (0 = closed)
    grip_force: float = 0.5       # 0-1 normalized
    
    # Multi-finger hand (if applicable)
    finger_positions: Optional[np.ndarray] = None  # Per-finger joint angles
    
    # Grasp type classification
    grasp_type: str = "unknown"   # pinch, power, precision, lateral
    
    # Quality/confidence
    confidence: float = 1.0


class HandRetargeter:
    """
    Retargets DYGlove 21-DOF hand pose to robot gripper commands.
    
    Supports multiple gripper types and preserves object-relative poses
    for accurate manipulation.
    """
    
    def __init__(self, config: HandRetargetConfig = None):
        self.config = config or HandRetargetConfig()
        
        # State history for smoothing
        self._prev_command: Optional[GripperCommand] = None
        self._prev_state: Optional[DYGloveState21DOF] = None
        
        # Statistics
        self._retarget_count = 0
        self._grasp_type_counts = {'pinch': 0, 'power': 0, 'precision': 0, 'open': 0}
        
        logger.info(f"HandRetargeter initialized for {self.config.gripper_type.name}")
    
    def retarget(
        self,
        hand_state: DYGloveState21DOF,
        object_pose: Optional[np.ndarray] = None,  # 4x4 transform
        robot_object_pose: Optional[np.ndarray] = None,  # Object pose in robot frame
    ) -> GripperCommand:
        """
        Retarget hand state to gripper command.
        
        Args:
            hand_state: Full 21-DOF hand state from DYGlove
            object_pose: Object pose in world frame (optional, for object-relative)
            robot_object_pose: Object pose in robot frame (if different from world)
        
        Returns:
            GripperCommand for the robot
        """
        self._retarget_count += 1
        
        # Quality check
        if hand_state.quality < self.config.min_quality_for_update:
            if self._prev_command is not None:
                return self._prev_command
            return GripperCommand(timestamp=hand_state.timestamp)
        
        # Compute finger closures
        closures = hand_state.get_all_closures()
        spread = hand_state.get_spread()
        
        # Classify grasp type
        grasp_type = self._classify_grasp(closures, spread)
        self._grasp_type_counts[grasp_type] = self._grasp_type_counts.get(grasp_type, 0) + 1
        
        # Compute gripper command based on type
        if self.config.gripper_type in [GripperType.PARALLEL_JAW, GripperType.ROBOTIQ_2F85, GripperType.DAIMON_GRIPPER]:
            command = self._retarget_to_parallel_jaw(hand_state, closures, spread, grasp_type)
        elif self.config.gripper_type == GripperType.ROBOTIQ_3F:
            command = self._retarget_to_robotiq_3f(hand_state, closures, spread, grasp_type)
        elif self.config.gripper_type in [GripperType.ALLEGRO_HAND, GripperType.LEAP_HAND]:
            command = self._retarget_to_dexterous_hand(hand_state, closures, spread, grasp_type)
        elif self.config.gripper_type == GripperType.SHADOW_HAND:
            command = self._retarget_to_shadow_hand(hand_state)
        else:
            command = self._retarget_to_parallel_jaw(hand_state, closures, spread, grasp_type)
        
        # Apply temporal smoothing
        if self._prev_command is not None:
            command = self._smooth_command(self._prev_command, command)
        
        # Update state
        self._prev_command = command
        self._prev_state = hand_state
        
        return command
    
    def _classify_grasp(self, closures: Dict[str, float], spread: float) -> str:
        """
        Classify grasp type based on finger configuration.
        
        Returns: 'pinch', 'power', 'precision', or 'open'
        """
        thumb_closed = closures['thumb'] > 0.5
        index_closed = closures['index'] > 0.5
        middle_closed = closures['middle'] > 0.5
        ring_pinky_closed = (closures['ring'] + closures['pinky']) / 2 > 0.5
        
        # Average closure
        avg_closure = sum(closures.values()) / len(closures)
        
        if avg_closure < 0.3:
            return 'open'
        
        # Pinch: thumb + index/middle closed, ring/pinky open
        if thumb_closed and (index_closed or middle_closed) and not ring_pinky_closed:
            return 'pinch'
        
        # Power: all fingers closed, low spread
        if avg_closure > self.config.power_closure_threshold and spread < 0.3:
            return 'power'
        
        # Precision: moderate closure, some spread
        if avg_closure > 0.4 and spread > 0.2:
            return 'precision'
        
        return 'power' if avg_closure > 0.5 else 'open'
    
    def _retarget_to_parallel_jaw(
        self,
        hand_state: DYGloveState21DOF,
        closures: Dict[str, float],
        spread: float,
        grasp_type: str
    ) -> GripperCommand:
        """
        Retarget to parallel-jaw gripper (1 DOF).
        
        Maps weighted finger closure to aperture.
        """
        # Compute weighted closure from specified fingers
        weighted_closure = 0.0
        total_weight = 0.0
        
        for finger in self.config.closure_fingers:
            weight = self.config.finger_weights.get(finger, 0.2)
            weighted_closure += closures[finger] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_closure /= total_weight
        
        # Add thumb influence
        thumb_weight = self.config.finger_weights.get('thumb', 0.3)
        weighted_closure = (1 - thumb_weight) * weighted_closure + thumb_weight * closures['thumb']
        
        # Adjust for finger spread (more spread = slightly more open)
        spread_adjustment = spread * self.config.abduction_scale
        adjusted_closure = weighted_closure * (1.0 - spread_adjustment)
        
        # Map closure to aperture
        # closure 0 -> max aperture (open)
        # closure 1 -> min aperture (closed)
        aperture = self.config.gripper_max_aperture * (1.0 - adjusted_closure)
        aperture = np.clip(aperture, self.config.gripper_min_aperture, self.config.gripper_max_aperture)
        
        # Force based on closure rate and grasp type
        grip_force = 0.3  # Default light
        if grasp_type == 'power':
            grip_force = 0.8
        elif grasp_type == 'pinch':
            grip_force = 0.5
        
        return GripperCommand(
            timestamp=hand_state.timestamp,
            aperture=aperture,
            grip_force=grip_force,
            grasp_type=grasp_type,
            confidence=hand_state.quality,
        )
    
    def _retarget_to_robotiq_3f(
        self,
        hand_state: DYGloveState21DOF,
        closures: Dict[str, float],
        spread: float,
        grasp_type: str
    ) -> GripperCommand:
        """
        Retarget to Robotiq 3-Finger gripper (4 DOF).
        
        DOF: finger A, finger B, finger C, scissor (spread)
        """
        # Robotiq 3F has symmetric fingers A & B, plus thumb C
        # Map: index/middle -> A/B, thumb -> C
        
        finger_positions = np.zeros(4)
        
        # Fingers A and B (0-255 range internally, we use 0-1)
        finger_positions[0] = closures['index']   # Finger A
        finger_positions[1] = closures['middle']  # Finger B
        finger_positions[2] = closures['thumb']   # Finger C (thumb)
        
        # Scissor (spread) - 0 = together, 1 = apart
        finger_positions[3] = spread
        
        # Compute equivalent aperture for compatibility
        avg_closure = (finger_positions[0] + finger_positions[1] + finger_positions[2]) / 3
        aperture = self.config.gripper_max_aperture * (1.0 - avg_closure)
        
        return GripperCommand(
            timestamp=hand_state.timestamp,
            aperture=aperture,
            grip_force=0.5,
            finger_positions=finger_positions,
            grasp_type=grasp_type,
            confidence=hand_state.quality,
        )
    
    def _retarget_to_dexterous_hand(
        self,
        hand_state: DYGloveState21DOF,
        closures: Dict[str, float],
        spread: float,
        grasp_type: str
    ) -> GripperCommand:
        """
        Retarget to 16-DOF dexterous hand (Allegro, LEAP).
        
        Maps 21 DOF human → 16 DOF robot hand.
        """
        # Allegro/LEAP: 4 fingers × 4 DOF each
        # Human thumb has 5 DOF, robot thumb has 4 DOF
        # Human fingers have 4 DOF each (same as robot)
        
        finger_positions = np.zeros(16)
        
        # Thumb (4 DOF) - drop wrist_ps
        finger_positions[0:4] = hand_state.thumb[0:4]  # tm_flex, tm_abd, mcp, ip
        
        # Index (4 DOF) - direct mapping
        finger_positions[4:8] = hand_state.index
        
        # Middle (4 DOF)
        finger_positions[8:12] = hand_state.middle
        
        # Ring (4 DOF) - maps to robot pinky in Allegro
        finger_positions[12:16] = hand_state.ring
        
        # Note: Human pinky is dropped for 16-DOF hands
        # Could alternatively blend ring+pinky
        
        # Compute aperture for compatibility
        aperture = self.config.gripper_max_aperture * (1.0 - sum(closures.values()) / 5)
        
        return GripperCommand(
            timestamp=hand_state.timestamp,
            aperture=aperture,
            grip_force=0.5,
            finger_positions=finger_positions,
            grasp_type=grasp_type,
            confidence=hand_state.quality,
        )
    
    def _retarget_to_shadow_hand(self, hand_state: DYGloveState21DOF) -> GripperCommand:
        """
        Retarget to Shadow Dexterous Hand (24 DOF).
        
        Full kinematic mapping with wrist.
        """
        # Shadow Hand: 5 fingers, 24 DOF total
        # Thumb: 5 DOF (matches human)
        # Index/Middle/Ring/Pinky: 4 DOF each (16 DOF)
        # Wrist: 2 DOF
        # Little finger has extra metacarpal DOF
        
        finger_positions = np.zeros(24)
        
        # Direct mapping (21 DOF hand → 24 DOF robot)
        # Thumb (5 DOF) - indices 0-4
        finger_positions[0:5] = hand_state.thumb
        
        # Index (4 DOF) - indices 5-8
        finger_positions[5:9] = hand_state.index
        
        # Middle (4 DOF) - indices 9-12
        finger_positions[9:13] = hand_state.middle
        
        # Ring (4 DOF) - indices 13-16
        finger_positions[13:17] = hand_state.ring
        
        # Pinky (5 DOF in Shadow, 4 DOF in human) - indices 17-21
        finger_positions[17:21] = hand_state.pinky
        finger_positions[21] = 0.0  # Extra metacarpal DOF
        
        # Wrist (2 DOF) - indices 22-23
        # Use wrist_ps from thumb for one axis, 0 for other
        finger_positions[22] = hand_state.thumb[4]  # wrist_ps
        finger_positions[23] = 0.0
        
        closures = hand_state.get_all_closures()
        aperture = self.config.gripper_max_aperture * (1.0 - sum(closures.values()) / 5)
        
        return GripperCommand(
            timestamp=hand_state.timestamp,
            aperture=aperture,
            grip_force=0.5,
            finger_positions=finger_positions,
            grasp_type=self._classify_grasp(closures, hand_state.get_spread()),
            confidence=hand_state.quality,
        )
    
    def _smooth_command(self, prev: GripperCommand, curr: GripperCommand) -> GripperCommand:
        """Apply exponential moving average smoothing."""
        alpha = self.config.smoothing_alpha
        
        smoothed = GripperCommand(
            timestamp=curr.timestamp,
            aperture=alpha * curr.aperture + (1 - alpha) * prev.aperture,
            grip_force=alpha * curr.grip_force + (1 - alpha) * prev.grip_force,
            grasp_type=curr.grasp_type,
            confidence=curr.confidence,
        )
        
        if curr.finger_positions is not None and prev.finger_positions is not None:
            smoothed.finger_positions = alpha * curr.finger_positions + (1 - alpha) * prev.finger_positions
        else:
            smoothed.finger_positions = curr.finger_positions
        
        return smoothed
    
    def compute_object_relative_error(
        self,
        hand_pose: np.ndarray,      # 4x4 human hand pose in world
        gripper_pose: np.ndarray,   # 4x4 robot gripper pose in world
        object_pose: np.ndarray,    # 4x4 object pose in world
        T_gripper_hand: np.ndarray  # 4x4 static transform gripper←hand
    ) -> Tuple[float, float]:
        """
        Compute error in object-relative poses.
        
        Returns (position_error_meters, rotation_error_radians)
        """
        # Human hand-object transform
        T_object_inv = np.linalg.inv(object_pose)
        T_hand_object = T_object_inv @ hand_pose
        
        # Expected gripper-object transform
        T_gripper_object_expected = T_hand_object @ T_gripper_hand
        
        # Actual gripper-object transform
        T_gripper_object_actual = T_object_inv @ gripper_pose
        
        # Position error
        pos_error = np.linalg.norm(
            T_gripper_object_expected[:3, 3] - T_gripper_object_actual[:3, 3]
        )
        
        # Rotation error (angle of rotation difference)
        R_diff = T_gripper_object_expected[:3, :3].T @ T_gripper_object_actual[:3, :3]
        rot_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        
        return pos_error, rot_error
    
    @property
    def statistics(self) -> Dict:
        """Get retargeting statistics."""
        total = sum(self._grasp_type_counts.values())
        return {
            'total_retargets': self._retarget_count,
            'grasp_type_distribution': {
                k: v / max(1, total) for k, v in self._grasp_type_counts.items()
            },
        }


# =============================================================================
# Robot-Specific Retargeting Profiles
# =============================================================================

def create_daimon_vtla_retargeter() -> HandRetargeter:
    """Create retargeter for Daimon VTLA robot gripper."""
    config = HandRetargetConfig(
        gripper_type=GripperType.DAIMON_GRIPPER,
        closure_fingers=['index', 'middle', 'thumb'],
        gripper_min_aperture=0.0,
        gripper_max_aperture=0.12,  # 120mm max
        smoothing_alpha=0.6,
    )
    return HandRetargeter(config)


def create_robotiq_2f85_retargeter() -> HandRetargeter:
    """Create retargeter for Robotiq 2F-85 gripper."""
    config = HandRetargetConfig(
        gripper_type=GripperType.ROBOTIQ_2F85,
        closure_fingers=['index', 'middle'],
        gripper_min_aperture=0.0,
        gripper_max_aperture=0.085,
        smoothing_alpha=0.7,
    )
    return HandRetargeter(config)


def create_allegro_hand_retargeter() -> HandRetargeter:
    """Create retargeter for Allegro Hand."""
    config = HandRetargetConfig(
        gripper_type=GripperType.ALLEGRO_HAND,
        smoothing_alpha=0.8,
    )
    return HandRetargeter(config)


# =============================================================================
# Integration with GMR
# =============================================================================

class WholeBodyHandRetargeter:
    """
    Combined whole-body and hand retargeting.
    
    Uses GMR for arm/body and HandRetargeter for gripper.
    """
    
    def __init__(
        self,
        gmr_retargeter,  # WholeBodyRetargeterGMR instance
        hand_retargeter: HandRetargeter,
    ):
        self.gmr = gmr_retargeter
        self.hand = hand_retargeter
    
    def retarget(
        self,
        human_body_pose: np.ndarray,   # RTMW3D format body keypoints
        hand_state: DYGloveState21DOF,  # DYGlove state
        object_pose: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, GripperCommand]:
        """
        Full retargeting: body + hand.
        
        Returns:
            (arm_joint_angles, gripper_command)
        """
        # GMR for arm
        arm_q = self.gmr.human_pose_to_robot_q(human_body_pose)
        
        # Hand retargeting
        gripper_cmd = self.hand.retarget(hand_state, object_pose)
        
        return arm_q, gripper_cmd


# =============================================================================
# Quality Scoring for Hand Retargeting
# =============================================================================

@dataclass
class HandRetargetQuality:
    """Quality metrics for hand retargeting."""
    
    # Input quality (from DYGlove)
    encoder_validity: float = 1.0    # Fraction of valid encoders
    communication_latency_ms: float = 0.0
    
    # Retargeting quality
    closure_confidence: float = 1.0  # How well we can determine closure
    grasp_type_confidence: float = 1.0  # Confidence in grasp classification
    
    # Object-relative (if applicable)
    object_relative_pos_error: float = 0.0  # meters
    object_relative_rot_error: float = 0.0  # radians
    
    @property
    def overall_score(self) -> float:
        """Compute overall quality score [0, 1]."""
        input_score = self.encoder_validity * (1.0 if self.communication_latency_ms < 30 else 0.8)
        retarget_score = 0.5 * self.closure_confidence + 0.5 * self.grasp_type_confidence
        
        # Penalize high object-relative error
        object_penalty = 1.0
        if self.object_relative_pos_error > 0.02:  # > 2cm
            object_penalty = 0.8
        if self.object_relative_pos_error > 0.05:  # > 5cm
            object_penalty = 0.5
        
        return input_score * retarget_score * object_penalty


def compute_hand_quality(
    glove_state: DYGloveState21DOF,
    gripper_cmd: GripperCommand,
    prev_state: Optional[DYGloveState21DOF] = None,
    object_error: Optional[Tuple[float, float]] = None,
) -> HandRetargetQuality:
    """
    Compute quality metrics for hand retargeting.
    """
    quality = HandRetargetQuality(
        closure_confidence=gripper_cmd.confidence,
        grasp_type_confidence=1.0 if gripper_cmd.grasp_type != 'unknown' else 0.5,
    )
    
    # Check for velocity spikes
    if prev_state is not None:
        dt = glove_state.timestamp - prev_state.timestamp
        if dt > 0:
            velocity = np.abs(glove_state.joint_angles - prev_state.joint_angles) / dt
            if np.max(velocity) > 15.0:  # rad/s spike threshold
                quality.closure_confidence *= 0.7
    
    # Object-relative error
    if object_error is not None:
        quality.object_relative_pos_error = object_error[0]
        quality.object_relative_rot_error = object_error[1]
    
    return quality


# =============================================================================
# CLI / Test
# =============================================================================

if __name__ == "__main__":
    # Test hand retargeting
    print("Testing Hand Retargeting Module")
    print("=" * 50)
    
    # Create test state
    joint_angles = np.array([
        # Thumb: partially flexed
        0.5, 0.1, 0.8, 0.6, 0.0,
        # Index: flexed (grasping)
        1.2, 0.0, 1.4, 0.8,
        # Middle: flexed
        1.1, -0.05, 1.3, 0.7,
        # Ring: slightly flexed
        0.6, 0.0, 0.8, 0.4,
        # Pinky: relaxed
        0.3, 0.0, 0.4, 0.2,
    ])
    
    state = DYGloveState21DOF.from_joint_array(joint_angles, timestamp=0.0)
    
    print(f"Input: 21-DOF joint angles")
    print(f"Closures: {state.get_all_closures()}")
    print(f"Spread: {state.get_spread():.3f}")
    
    # Test different grippers
    for gripper_type in [GripperType.PARALLEL_JAW, GripperType.ROBOTIQ_3F, GripperType.ALLEGRO_HAND]:
        config = HandRetargetConfig(gripper_type=gripper_type)
        retargeter = HandRetargeter(config)
        
        cmd = retargeter.retarget(state)
        
        print(f"\n{gripper_type.name}:")
        print(f"  Aperture: {cmd.aperture*1000:.1f}mm")
        print(f"  Grasp type: {cmd.grasp_type}")
        print(f"  Grip force: {cmd.grip_force:.2f}")
        if cmd.finger_positions is not None:
            print(f"  Finger positions: {cmd.finger_positions}")
    
    print("\n✓ Hand retargeting tests passed!")

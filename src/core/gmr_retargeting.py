"""
Human-to-Robot Retargeting Module

This module implements object-centric retargeting: instead of trying to match
the human's absolute pose, we match the relationship between the human's hand
and the object being manipulated.

The key insight: for manipulation tasks, what matters is T_hand_object — the
hand's pose relative to the object. If we preserve this relationship when
commanding the robot, the robot will perform the same manipulation regardless
of where the object is in its workspace.

Coordinate Frames:
    W = World frame (robot base or calibrated origin)
    H = Human hand frame (palm center, Z out of palm)
    O = Object frame (centroid, task-dependent orientation)
    E = Robot end-effector frame (TCP, Z along approach)

The retargeting flow:
    1. Get human hand pose in world: T_H_W (from camera + glove fusion)
    2. Get object pose in world: T_O_W (from perception)
    3. Compute hand-object transform: T_H_O = inv(T_O_W) @ T_H_W
    4. Get robot's view of object: T_O_W_robot (could differ from human's view)
    5. Apply frame correction: T_E_H accounts for hand vs gripper conventions
    6. Desired EE pose: T_E_W = T_O_W_robot @ T_H_O @ T_E_H
    7. Solve IK for T_E_W
    8. Map finger closure → gripper command

The T_E_H correction is crucial: a human's palm might face down during a
top-down grasp, but the gripper's TCP Z-axis might point forward. This
static transform bridges that convention difference.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Protocol
import numpy as np
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("Scipy not found. Retargeting will be limited.")

# Try to import Pinocchio for real FK/IK
try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    logger.debug("Pinocchio not available - using fallback FK")

# Try pink (Pinocchio-based IK library) as alternative
try:
    import pink
    HAS_PINK = True
except ImportError:
    HAS_PINK = False

from .human_state import HumanState, EnvObject, DexterHandState
from .recorder import RobotObs, RobotAction, DemoStep
import cv2


# ---------------------------------------------------------------------------
# IK Solver Protocol (abstract interface)
# ---------------------------------------------------------------------------

class IKSolverProtocol(Protocol):
    """
    Protocol (interface) for IK solvers.
    
    Any IK solver (Pinocchio, IKFast, KDL, etc.) should implement this interface.
    This allows the retargeter to be agnostic to the underlying solver.
    """
    
    def solve(
        self, 
        q_init: np.ndarray, 
        T_ee_target: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK for desired end-effector pose using Damped Least Squares.
        """
        q = q_init.copy()
        success = False
        error = float('inf')
        
        # Damping factor
        lambda_val = 0.01
        
        for i in range(max_iterations):
            # 1. Compute FK and error
            T_current = self.forward_kinematics(q)
            
            # Position error
            pos_err = T_ee_target[:3, 3] - T_current[:3, 3]
            
            # Orientation error (using rotation vector approximation for small angles)
            R_target = T_ee_target[:3, :3]
            R_current = T_current[:3, :3]
            R_err = R_target @ R_current.T
            
            # Convert rotation matrix error to axis-angle vector
            # This is a simplified approximation suitable for iterative convergence
            rot_vec, _ = cv2.Rodrigues(R_err)
            rot_err = rot_vec.flatten()
            
            # Combined error vector (6x1)
            e = np.concatenate([pos_err, rot_err])
            error = np.linalg.norm(e)
            
            if error < tolerance:
                success = True
                break
            
            # 2. Compute Jacobian (Numerical or Analytical)
            # Here we use a numerical approximation for simplicity/generality
            J = self._compute_jacobian(q)
            
            # 3. Compute update using Damped Least Squares: dq = J^T * (J * J^T + lambda^2 * I)^-1 * e
            # Or equivalently: dq = (J^T * J + lambda^2 * I)^-1 * J^T * e
            
            # Using pseudo-inverse with damping is often more stable
            # dq = pinv(J) * e
            # J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_val**2 * np.eye(6))
            
            # Using numpy's pinv for robustness
            dq = np.linalg.pinv(J) @ e
            
            # 4. Update joints
            q = q + dq
            
            # 5. Clamp to limits
            q = np.clip(q, self.joint_limits_lower, self.joint_limits_upper)
            
        return q, success, error

    def _compute_jacobian(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Compute numerical Jacobian."""
        n = len(q)
        J = np.zeros((6, n))
        
        T_0 = self.forward_kinematics(q)
        pos_0 = T_0[:3, 3]
        
        # For orientation, we need a consistent representation. 
        # We'll use rotation vectors relative to T_0
        
        for i in range(n):
            q_perturbed = q.copy()
            q_perturbed[i] += delta
            
            T_p = self.forward_kinematics(q_perturbed)
            pos_p = T_p[:3, 3]
            
            # Position derivative
            J[:3, i] = (pos_p - pos_0) / delta
            
            # Orientation derivative
            R_p = T_p[:3, :3]
            R_0 = T_0[:3, :3]
            R_diff = R_p @ R_0.T
            rot_vec, _ = cv2.Rodrigues(R_diff)
            J[3:, i] = rot_vec.flatten() / delta
            
        return J

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute EE pose for given joint configuration.

        Uses Pinocchio if available, otherwise falls back to a simple
        approximation based on joint angles.
        """
        # Fallback: simple approximation based on joint angles
        T = np.eye(4)
        if len(q) >= 3:
            T[0, 3] = q[0] * 0.1
            T[1, 3] = q[1] * 0.1
            T[2, 3] = 0.5 + q[2] * 0.1

        return T


class PinocchioIKSolver:
    """
    Production IK solver using Pinocchio library.

    Pinocchio provides:
    - Efficient forward/inverse kinematics
    - URDF parsing
    - Jacobian computation
    - Collision checking (optional)

    Install: pip install pin (or build from source for GPU support)

    Usage:
        solver = PinocchioIKSolver("robot.urdf", "ee_link")
        q, success, error = solver.solve(q_init, T_target)
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str = "ee_link",
        joint_limits_lower: np.ndarray = None,
        joint_limits_upper: np.ndarray = None,
    ):
        """
        Initialize Pinocchio-based IK solver.

        Args:
            urdf_path: Path to robot URDF file
            ee_frame_name: Name of end-effector frame in URDF
            joint_limits_lower: Lower joint limits (uses URDF if None)
            joint_limits_upper: Upper joint limits (uses URDF if None)
        """
        self._initialized = False
        self._urdf_path = urdf_path
        self._ee_frame_name = ee_frame_name

        if not HAS_PINOCCHIO:
            logger.warning("Pinocchio not available - IK solver will use fallback")
            # Set default limits for fallback
            self._joint_limits_lower = joint_limits_lower or np.full(7, -np.pi)
            self._joint_limits_upper = joint_limits_upper or np.full(7, np.pi)
            self._n_joints = 7
            return

        try:
            # Load URDF
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

            # Get end-effector frame ID
            self.ee_frame_id = self.model.getFrameId(ee_frame_name)
            if self.ee_frame_id >= self.model.nframes:
                # Try to find a suitable frame
                for i in range(self.model.nframes):
                    if "ee" in self.model.frames[i].name.lower() or \
                       "end" in self.model.frames[i].name.lower() or \
                       "gripper" in self.model.frames[i].name.lower():
                        self.ee_frame_id = i
                        logger.info(f"Using frame '{self.model.frames[i].name}' as EE")
                        break

            # Get joint limits from URDF
            self._joint_limits_lower = joint_limits_lower if joint_limits_lower is not None else \
                self.model.lowerPositionLimit[:self.model.nq]
            self._joint_limits_upper = joint_limits_upper if joint_limits_upper is not None else \
                self.model.upperPositionLimit[:self.model.nq]
            self._n_joints = self.model.nq

            self._initialized = True
            logger.info(f"Pinocchio IK solver initialized: {self.model.nq} DOF")

        except Exception as e:
            logger.error(f"Failed to initialize Pinocchio: {e}")
            self._joint_limits_lower = joint_limits_lower or np.full(7, -np.pi)
            self._joint_limits_upper = joint_limits_upper or np.full(7, np.pi)
            self._n_joints = 7

    @property
    def joint_limits_lower(self) -> np.ndarray:
        return self._joint_limits_lower

    @property
    def joint_limits_upper(self) -> np.ndarray:
        return self._joint_limits_upper

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose for given joint configuration.

        Args:
            q: Joint configuration [n_joints]

        Returns:
            T: 4x4 homogeneous transform of end-effector in base frame
        """
        if not self._initialized or not HAS_PINOCCHIO:
            # Fallback: simple approximation
            T = np.eye(4)
            if len(q) >= 3:
                T[0, 3] = q[0] * 0.1
                T[1, 3] = q[1] * 0.1
                T[2, 3] = 0.5 + q[2] * 0.1
            return T

        # Compute FK using Pinocchio
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get EE pose
        oMee = self.data.oMf[self.ee_frame_id]

        # Convert to 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = oMee.rotation
        T[:3, 3] = oMee.translation

        return T

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector Jacobian analytically.

        Args:
            q: Joint configuration

        Returns:
            J: 6xN Jacobian matrix (linear + angular velocity)
        """
        if not self._initialized or not HAS_PINOCCHIO:
            # Numerical fallback
            return self._compute_jacobian_numerical(q)

        pin.computeJointJacobians(self.model, self.data, q)
        J = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        return J

    def _compute_jacobian_numerical(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Compute Jacobian numerically as fallback."""
        n = len(q)
        J = np.zeros((6, n))

        T_0 = self.forward_kinematics(q)
        pos_0 = T_0[:3, 3]

        for i in range(n):
            q_perturbed = q.copy()
            q_perturbed[i] += delta

            T_p = self.forward_kinematics(q_perturbed)
            pos_p = T_p[:3, 3]

            # Position derivative
            J[:3, i] = (pos_p - pos_0) / delta

            # Orientation derivative
            R_p = T_p[:3, :3]
            R_0 = T_0[:3, :3]
            R_diff = R_p @ R_0.T
            rot_vec, _ = cv2.Rodrigues(R_diff)
            J[3:, i] = rot_vec.flatten() / delta

        return J

    def solve(
        self,
        q_init: np.ndarray,
        T_ee_target: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        whole_body_prior: np.ndarray = None,
        regularization: float = 0.01,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK using Damped Least Squares with optional whole-body regularization.

        Args:
            q_init: Initial joint configuration
            T_ee_target: Target end-effector pose (4x4)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            whole_body_prior: Optional whole-body pose prior for regularization
            regularization: Regularization weight toward prior

        Returns:
            q: Joint solution
            success: Whether IK converged
            error: Final pose error
        """
        q = q_init.copy()
        lambda_dls = 0.01  # Damping factor

        for iteration in range(max_iterations):
            # 1. Compute FK and error
            T_current = self.forward_kinematics(q)

            # Position error
            pos_err = T_ee_target[:3, 3] - T_current[:3, 3]

            # Orientation error (axis-angle)
            R_target = T_ee_target[:3, :3]
            R_current = T_current[:3, :3]
            R_err = R_target @ R_current.T
            rot_vec, _ = cv2.Rodrigues(R_err)
            rot_err = rot_vec.flatten()

            # Combined error
            e = np.concatenate([pos_err, rot_err])
            error = np.linalg.norm(e)

            if error < tolerance:
                return q, True, error

            # 2. Compute Jacobian
            J = self.compute_jacobian(q)

            # 3. Damped Least Squares update
            # With whole-body regularization: minimize ||Jdq - e||^2 + lambda||dq - dq_prior||^2
            JtJ = J.T @ J

            if whole_body_prior is not None:
                # Add regularization toward prior
                dq_prior = whole_body_prior - q
                dq = np.linalg.solve(
                    JtJ + (lambda_dls**2 + regularization) * np.eye(len(q)),
                    J.T @ e + regularization * dq_prior
                )
            else:
                # Standard DLS
                dq = np.linalg.solve(
                    JtJ + lambda_dls**2 * np.eye(len(q)),
                    J.T @ e
                )

            # 4. Update and clamp
            q = q + dq
            q = np.clip(q, self._joint_limits_lower[:len(q)], self._joint_limits_upper[:len(q)])

        # Return best effort solution
        T_final = self.forward_kinematics(q)
        pos_err = np.linalg.norm(T_ee_target[:3, 3] - T_final[:3, 3])
        return q, False, pos_err


def create_ik_solver(
    urdf_path: str = None,
    robot_type: str = "generic_7dof",
    ee_frame_name: str = "ee_link",
) -> 'IKSolverProtocol':
    """
    Factory function to create an appropriate IK solver.

    Args:
        urdf_path: Path to URDF file (if available)
        robot_type: Robot type for default parameters
        ee_frame_name: Name of end-effector frame

    Returns:
        IK solver implementing IKSolverProtocol
    """
    if urdf_path and HAS_PINOCCHIO:
        import os
        if os.path.exists(urdf_path):
            return PinocchioIKSolver(urdf_path, ee_frame_name)
        else:
            logger.warning(f"URDF not found at {urdf_path}, using fallback IK")

    # Return a protocol-compliant fallback
    logger.info(f"Using fallback IK solver for {robot_type}")

    class FallbackIKSolver:
        """Simple IK solver when Pinocchio is not available."""

        def __init__(self, robot_type: str):
            self.robot_type = robot_type
            # Default 7-DOF arm limits
            self._joint_limits_lower = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
            self._joint_limits_upper = np.array([2.89, 1.76, 2.89, -0.06, 2.89, 3.75, 2.89])

        @property
        def joint_limits_lower(self) -> np.ndarray:
            return self._joint_limits_lower

        @property
        def joint_limits_upper(self) -> np.ndarray:
            return self._joint_limits_upper

        def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
            """Simple FK approximation."""
            T = np.eye(4)
            if len(q) >= 3:
                T[0, 3] = q[0] * 0.1
                T[1, 3] = q[1] * 0.1
                T[2, 3] = 0.5 + q[2] * 0.1
            return T

        def solve(
            self,
            q_init: np.ndarray,
            T_ee_target: np.ndarray,
            max_iterations: int = 100,
            tolerance: float = 1e-4,
            **kwargs
        ) -> Tuple[np.ndarray, bool, float]:
            """Solve IK using numerical Jacobian."""
            q = q_init.copy()
            lambda_val = 0.01

            for _ in range(max_iterations):
                T_current = self.forward_kinematics(q)
                pos_err = T_ee_target[:3, 3] - T_current[:3, 3]

                R_target = T_ee_target[:3, :3]
                R_current = T_current[:3, :3]
                R_err = R_target @ R_current.T
                rot_vec, _ = cv2.Rodrigues(R_err)
                rot_err = rot_vec.flatten()

                e = np.concatenate([pos_err, rot_err])
                error = np.linalg.norm(e)

                if error < tolerance:
                    return q, True, error

                J = self._compute_jacobian(q)
                dq = np.linalg.pinv(J) @ e
                q = q + dq
                q = np.clip(q, self._joint_limits_lower[:len(q)], self._joint_limits_upper[:len(q)])

            return q, False, np.linalg.norm(e)

        def _compute_jacobian(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
            """Numerical Jacobian."""
            n = len(q)
            J = np.zeros((6, n))
            T_0 = self.forward_kinematics(q)

            for i in range(n):
                q_p = q.copy()
                q_p[i] += delta
                T_p = self.forward_kinematics(q_p)
                J[:3, i] = (T_p[:3, 3] - T_0[:3, 3]) / delta
                R_diff = T_p[:3, :3] @ T_0[:3, :3].T
                rot_vec, _ = cv2.Rodrigues(R_diff)
                J[3:, i] = rot_vec.flatten() / delta

            return J

    return FallbackIKSolver(robot_type)


# ---------------------------------------------------------------------------
# Utility Functions for SE(3) Operations
# ---------------------------------------------------------------------------

def pose_to_matrix(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Convert position + quaternion to 4x4 homogeneous transform.
    
    Args:
        position: [x, y, z]
        quat: [x, y, z, w] quaternion (Hamilton convention)
        
    Returns:
        4x4 homogeneous transform matrix
    """
    T = np.eye(4)
    if HAS_SCIPY:
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    else:
        # Fallback if scipy missing (identity rotation)
        pass
    T[:3, 3] = position
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract position and quaternion from 4x4 transform.
    
    Returns:
        position: [x, y, z]
        quat: [x, y, z, w]
    """
    position = T[:3, 3]
    if HAS_SCIPY:
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    else:
        quat = np.array([0, 0, 0, 1])
    return position, quat


def interpolate_pose(T1: np.ndarray, T2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linearly interpolate between two poses (SLERP for rotation).
    
    Args:
        T1, T2: 4x4 transforms
        alpha: Interpolation factor in [0, 1]
        
    Returns:
        Interpolated 4x4 transform
    """
    pos1, quat1 = matrix_to_pose(T1)
    pos2, quat2 = matrix_to_pose(T2)
    
    # Linear interpolation for position
    pos_interp = (1 - alpha) * pos1 + alpha * pos2
    
    # SLERP for rotation
    if HAS_SCIPY:
        r1 = Rotation.from_quat(quat1)
        r2 = Rotation.from_quat(quat2)
        
        # Scipy's Slerp requires Rotation objects
        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
        quat_interp = slerp(alpha).as_quat()
    else:
        # Linear interpolation (normalization needed)
        quat_interp = (1 - alpha) * quat1 + alpha * quat2
        quat_interp /= np.linalg.norm(quat_interp)
    
    return pose_to_matrix(pos_interp, quat_interp)


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> Tuple[float, float]:
    """
    Compute position and orientation error between two poses.
    
    Returns:
        pos_error: Euclidean distance in meters
        rot_error: Angular distance in radians
    """
    pos_error = np.linalg.norm(T_target[:3, 3] - T_current[:3, 3])
    
    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]
    R_error = R_target @ R_current.T
    
    # Angular error from rotation matrix (axis-angle magnitude)
    rot_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
    
    return pos_error, rot_error


# ---------------------------------------------------------------------------
# Hand Pose Estimation from Glove + Camera
# ---------------------------------------------------------------------------

def estimate_hand_pose_world(
    wrist_position_world: np.ndarray,
    glove_state: DexterHandState,
    arm_direction: np.ndarray
) -> np.ndarray:
    """
    Estimate the full 6-DOF hand pose in world frame.
    
    The glove gives us wrist orientation in its IMU frame, and the camera
    gives us wrist position in world frame. We need to combine these,
    accounting for the (calibrated) transform between IMU and world.
    
    If calibration isn't available, we use the arm direction from the camera
    as a proxy for hand orientation (less accurate but usable).
    
    Hand frame convention:
        - Origin: palm center (approximately at wrist)
        - Z: points outward from palm (toward grasped object)
        - X: points along fingers (toward fingertips)
        - Y: completes right-handed frame (across palm)
    
    Args:
        wrist_position_world: [3] wrist position from camera
        glove_state: Hand state including orientation
        arm_direction: [3] unit vector from elbow to wrist (fallback)
        
    Returns:
        T_hand_world: 4x4 transform from hand frame to world frame
    """
    T = np.eye(4)
    T[:3, 3] = wrist_position_world
    
    if glove_state.is_calibrated and HAS_SCIPY:
        # Use glove orientation with calibration
        R_glove = Rotation.from_quat(glove_state.wrist_quat_local).as_matrix()
        R_world_glove = glove_state.T_world_glove_imu[:3, :3]
        T[:3, :3] = R_world_glove @ R_glove
    else:
        # Fallback: construct frame from arm direction
        # This is approximate but often good enough for coarse retargeting
        
        # Z axis: assume palm faces "forward" relative to arm direction
        # This heuristic works for reaching motions; fails for rotated wrists
        z_axis = arm_direction  # rough: palm faces where arm is going
        
        # Y axis: assume gravity-aligned "up" for the palm
        world_up = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, world_up)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            # Arm is pointing straight up/down; use a different reference
            y_axis = np.array([0, 1, 0])
        else:
            y_axis = y_axis / y_norm
        
        # X axis: complete the frame
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Recompute y to ensure orthonormality
        y_axis = np.cross(z_axis, x_axis)
        
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
    
    return T


# ---------------------------------------------------------------------------
# Gripper Mapping
# ---------------------------------------------------------------------------

@dataclass
class GripperMapping:
    """
    Configuration for mapping human hand closure to robot gripper command.
    
    A parallel-jaw gripper has 1 DOF (aperture), while a human hand has ~20 DOF.
    We need a many-to-one mapping. Common approaches:
    
    1. Mean finger closure: average all finger joint angles
    2. Weighted closure: weight thumb/index more for precision grasps
    3. Contact-based: trigger close when contact is detected
    4. Aperture-matched: measure thumb-finger distance, map to gripper width
    
    This class supports configurable mapping strategies.
    """
    strategy: str = 'weighted'              # 'mean', 'weighted', 'contact', 'aperture'
    
    # For 'weighted' strategy: weights for [thumb, index, middle, ring, pinky]
    finger_weights: np.ndarray = None
    
    # For 'contact' strategy: force threshold to trigger close
    contact_threshold: float = 0.5          # Newtons
    
    # Gripper command bounds
    open_position: float = 0.0              # Command for fully open
    closed_position: float = 1.0            # Command for fully closed
    
    # Smoothing: prevents jerky gripper motion
    smoothing_alpha: float = 0.3            # EMA smoothing factor
    
    def __post_init__(self):
        if self.finger_weights is None:
            # Default: weight thumb and index more heavily
            self.finger_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    def map(
        self, 
        hand_state: DexterHandState, 
        prev_gripper: float = 0.0
    ) -> float:
        """
        Map hand state to gripper command.
        
        Args:
            hand_state: Current hand state from glove
            prev_gripper: Previous gripper command (for smoothing)
            
        Returns:
            Gripper command in [open_position, closed_position]
        """
        if self.strategy == 'mean':
            closure = np.mean(hand_state.get_finger_closure())
            
        elif self.strategy == 'weighted':
            closures = hand_state.get_finger_closure()
            closure = np.dot(self.finger_weights, closures)
            
        elif self.strategy == 'contact':
            if hand_state.has_contact(self.contact_threshold):
                closure = 1.0
            else:
                closure = hand_state.get_grasp_aperture()
                
        elif self.strategy == 'aperture':
            # Use the overall grasp aperture directly
            closure = hand_state.get_grasp_aperture()
            
        else:
            raise ValueError(f"Unknown gripper mapping strategy: {self.strategy}")
        
        # Map closure [0, 1] to gripper range
        raw_command = self.open_position + closure * (self.closed_position - self.open_position)
        
        # Apply smoothing
        smoothed = self.smoothing_alpha * raw_command + (1 - self.smoothing_alpha) * prev_gripper
        
        return float(np.clip(smoothed, self.open_position, self.closed_position))


# ---------------------------------------------------------------------------
# Main Retargeter Class
# ---------------------------------------------------------------------------

@dataclass
class RetargetConfig:
    """Configuration for the retargeter."""
    
    # Frame correction: transform from human hand frame to robot EE frame
    # This accounts for convention differences (e.g., palm down vs gripper forward)
    T_ee_hand: np.ndarray = None            # 4x4 transform
    
    # Position limits: clip desired EE position to reachable workspace
    workspace_min: np.ndarray = None        # [x_min, y_min, z_min]
    workspace_max: np.ndarray = None        # [x_max, y_max, z_max]
    
    # Velocity limits: max joint velocity for smoothing
    max_joint_velocity: float = 1.0         # rad/s
    
    # IK failure handling
    ik_fallback_to_previous: bool = True    # On IK failure, use previous solution
    ik_max_position_error: float = 0.05     # Accept IK solution if within this error (m)
    ik_max_rotation_error: float = 0.2      # Accept IK solution if within this error (rad)
    
    # Gripper mapping
    gripper_mapping: GripperMapping = None
    
    def __post_init__(self):
        if self.T_ee_hand is None:
            # Default: identity (no correction)
            # You'll almost certainly need to calibrate this for your setup
            self.T_ee_hand = np.eye(4)
        
        if self.workspace_min is None:
            # Default workspace for a typical 7-DOF arm
            self.workspace_min = np.array([-1.0, -1.0, 0.0])
        
        if self.workspace_max is None:
            self.workspace_max = np.array([1.0, 1.0, 1.5])
        
        if self.gripper_mapping is None:
            self.gripper_mapping = GripperMapping()

    @classmethod
    def from_dict(cls, config: dict) -> 'RetargetConfig':
        """Create config from dictionary."""
        # Extract known keys
        known_keys = [
            "workspace_min", "workspace_max", "max_joint_velocity",
            "ik_fallback_to_previous", "ik_max_position_error", "ik_max_rotation_error"
        ]
        
        kwargs = {}
        for key in known_keys:
            if key in config:
                kwargs[key] = config[key]
        
        # Handle nested IK config if flattened or nested
        if "ik" in config:
            ik_cfg = config["ik"]
            if "max_position_error" in ik_cfg:
                kwargs["ik_max_position_error"] = ik_cfg["max_position_error"]
            if "max_rotation_error" in ik_cfg:
                kwargs["ik_max_rotation_error"] = ik_cfg["max_rotation_error"]
            if "fallback_to_previous" in ik_cfg:
                kwargs["ik_fallback_to_previous"] = ik_cfg["fallback_to_previous"]
                
        # Handle numpy arrays
        if "workspace_min" in kwargs and isinstance(kwargs["workspace_min"], list):
            kwargs["workspace_min"] = np.array(kwargs["workspace_min"])
        if "workspace_max" in kwargs and isinstance(kwargs["workspace_max"], list):
            kwargs["workspace_max"] = np.array(kwargs["workspace_max"])
            
        return cls(**kwargs)


class Retargeter:
    """
    Object-centric human-to-robot retargeting with optional whole-body support.
    
    This class transforms human demonstrations into robot actions by:
    1. Computing the human hand's pose relative to the manipulated object
    2. Optionally using GMR for whole-body posture as a soft prior
    3. Applying that relative pose to the robot's end-effector
    4. Solving IK to find joint configurations
    5. Mapping finger state to gripper commands
    
    The integration follows OKAMI-style object-centric retargeting:
        - T_hand_object is the key invariant we preserve
        - Whole-body posture from GMR provides a natural pose prior
        - IK enforces the end-effector constraint while staying near the prior
    
    Usage:
        retargeter = Retargeter(ik_solver, config, whole_body_retargeter=gmr)
        
        # In demo loop:
        step = retargeter.human_to_robot_action(
            human_state, robot_obs, prev_q,
            episode_id, step_idx, task_id, env_id
        )
    """
    
    def __init__(
        self, 
        ik_solver: IKSolverProtocol, 
        config: RetargetConfig = None,
        whole_body_retargeter: 'WholeBodyRetargeterGMR' = None,
        gripper_mapping: GripperMapping = None
    ):
        """
        Initialize the retargeter.
        
        Args:
            ik_solver: IK solver implementing IKSolverProtocol
            config: Retargeting configuration
            whole_body_retargeter: Optional GMR-based whole-body retargeter.
                                   If provided, uses whole-body pose as IK prior.
            gripper_mapping: Custom gripper mapping (overrides config if provided)
        """
        self.ik = ik_solver
        self.config = config or RetargetConfig()
        self.whole_body_retargeter = whole_body_retargeter
        
        # Override gripper mapping if provided separately
        if gripper_mapping is not None:
            self.config.gripper_mapping = gripper_mapping
        
        # State for smoothing and fallback
        self._prev_q: Optional[np.ndarray] = None
        self._prev_T_ee: Optional[np.ndarray] = None
        self._prev_gripper: float = 0.0
        
        # Statistics for debugging
        self._ik_success_count: int = 0
        self._ik_failure_count: int = 0
    
    def reset(self):
        """Reset internal state (call at start of each episode)."""
        self._prev_q = None
        self._prev_T_ee = None
        self._prev_gripper = 0.0
        
        # Also reset whole-body retargeter if present
        if self.whole_body_retargeter is not None:
            self.whole_body_retargeter.reset()
    
    def _compute_hand_object_transform(
        self,
        human: HumanState,
        hand_side: str = 'right'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute T_hand_object: the hand's pose relative to the primary object.
        
        Args:
            human: Current human state
            hand_side: Which hand to use ('right' or 'left')
            
        Returns:
            T_hand_world: Hand pose in world frame (4x4)
            T_hand_object: Hand pose in object frame (4x4), or None if no object
        """
        # Get hand state
        if hand_side == 'right':
            glove = human.hand_right
            wrist_pos = human.body.wrist_right_position
            arm_dir = human.body.get_arm_direction('right')
        else:
            glove = human.hand_left
            wrist_pos = human.body.wrist_left_position
            arm_dir = human.body.get_arm_direction('left')
        
        # Estimate hand pose in world frame
        T_hand_world = estimate_hand_pose_world(wrist_pos, glove, arm_dir)
        
        # If there's a primary object, compute relative transform
        obj = human.primary_object
        if obj is not None:
            T_object_world = obj.pose_world
            T_object_world_inv = np.linalg.inv(T_object_world)
            T_hand_object = T_object_world_inv @ T_hand_world
            return T_hand_world, T_hand_object
        
        return T_hand_world, None
    
    def _compute_desired_ee_pose(
        self,
        T_hand_world: np.ndarray,
        T_hand_object: Optional[np.ndarray],
        object_pose_robot: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute desired end-effector pose in robot's world frame.
        
        If we have object-relative information (T_hand_object) and the robot
        can see the same object (object_pose_robot), we apply the object-centric
        transformation. Otherwise, we fall back to direct hand-to-EE mapping.
        
        Args:
            T_hand_world: Human hand pose in world frame
            T_hand_object: Hand pose relative to object (None if no object)
            object_pose_robot: Object pose in robot's view (None if unknown)
            
        Returns:
            T_ee_world_desired: Desired EE pose in robot's world frame
        """
        if T_hand_object is not None and object_pose_robot is not None:
            # Object-centric retargeting: 
            # T_ee = T_object_robot @ T_hand_object @ T_ee_hand
            T_ee_world = object_pose_robot @ T_hand_object @ self.config.T_ee_hand
        else:
            # Direct retargeting (no object information):
            # Just apply the hand-to-EE correction
            T_ee_world = T_hand_world @ self.config.T_ee_hand
        
        # Clip position to workspace bounds
        pos = T_ee_world[:3, 3]
        pos_clipped = np.clip(pos, self.config.workspace_min, self.config.workspace_max)
        T_ee_world[:3, 3] = pos_clipped
        
        return T_ee_world
    
    def _smooth_trajectory(
        self,
        q_target: np.ndarray,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Apply velocity limiting to smooth joint trajectory.
        
        Large jumps in joint space create jerky motion and can be unsafe.
        We limit the per-step change based on max velocity and timestep.
        
        Args:
            q_target: Desired joint configuration
            dt: Time since last step (seconds)
            
        Returns:
            Smoothed joint configuration
        """
        if self._prev_q is None:
            return q_target
        
        max_delta = self.config.max_joint_velocity * dt
        delta = q_target - self._prev_q
        
        # Clip each joint's delta to max velocity
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        
        return self._prev_q + delta_clipped
    
    def human_to_robot_action(
        self,
        human: HumanState,
        robot_obs: RobotObs,
        prev_q: Optional[np.ndarray],
        episode_id: str,
        t_index: int,
        task_id: str,
        env_id: str,
        object_pose_robot: Optional[np.ndarray] = None,
        dt: float = 0.1
    ) -> DemoStep:
        """
        Convert human state to robot action (the main retargeting function).
        
        This is the core method called at each timestep during demonstration
        recording. It:
        1. Extracts hand pose from human state
        2. Computes object-relative transform
        3. Generates desired EE pose for robot
        4. Solves IK
        5. Maps gripper
        6. Packages everything into a DemoStep
        
        Args:
            human: Current human demonstrator state
            robot_obs: Current robot observation
            prev_q: Previous joint configuration (for IK seeding)
            episode_id: Unique episode identifier
            t_index: Timestep index within episode
            task_id: Task being demonstrated
            env_id: Environment/site identifier
            object_pose_robot: Object pose from robot's perception (if available)
            dt: Time since last step
            
        Returns:
            DemoStep containing observation, action, and metadata
        """
        # Initialize previous state if needed
        if prev_q is not None:
            self._prev_q = prev_q
        elif self._prev_q is None:
            self._prev_q = robot_obs.joint_positions.copy()
        
        # Determine which hand to use
        active_hand = human.get_active_hand()
        hand_side = active_hand.side if active_hand else 'right'
        
        # Compute hand pose and object-relative transform
        T_hand_world, T_hand_object = self._compute_hand_object_transform(human, hand_side)
        
        # If we don't have object_pose_robot but human has a primary object,
        # assume same pose (valid if robot and human share calibrated world frame)
        if object_pose_robot is None and human.primary_object is not None:
            object_pose_robot = human.primary_object.pose_world
        
        # Compute desired end-effector pose
        T_ee_desired = self._compute_desired_ee_pose(
            T_hand_world, T_hand_object, object_pose_robot
        )
        
        # Determine IK initial guess
        # If we have whole-body retargeter, use it to get a posture prior
        q_init = self._prev_q
        whole_body_prior = None
        
        if self.whole_body_retargeter is not None:
            try:
                # Get whole-body configuration from GMR
                q_whole = self.whole_body_retargeter.human_pose_to_robot_q(human.body)
                
                # Fuse with current state (blend to avoid sudden jumps)
                from .whole_body_gmr import fuse_whole_body_with_current
                q_init = fuse_whole_body_with_current(
                    q_whole, 
                    self._prev_q,
                    blend_factor=0.8  # Trust GMR 80%, current 20%
                )
                
                # Store as soft prior for IK regularization
                whole_body_prior = q_whole
                
            except Exception as e:
                # GMR failed, fall back to previous configuration
                logger.warning(f"GMR retarget failed: {e}")
        
        # Solve IK with optional whole-body prior
        # If IK solver supports whole_body_prior, it will regularize toward it
        try:
            # Try extended IK interface with prior
            q_solution, ik_success, ik_error = self.ik.solve(
                q_init, 
                T_ee_desired,
                whole_body_prior=whole_body_prior
            )
        except TypeError:
            # IK solver doesn't support whole_body_prior, use standard interface
            q_solution, ik_success, ik_error = self.ik.solve(q_init, T_ee_desired)
        
        # Handle IK failure
        is_valid = True
        quality_score = 1.0
        
        if not ik_success:
            self._ik_failure_count += 1
            
            # Check if solution is "close enough"
            T_achieved = self.ik.forward_kinematics(q_solution)
            pos_err, rot_err = pose_error(T_achieved, T_ee_desired)
            
            if pos_err < self.config.ik_max_position_error and rot_err < self.config.ik_max_rotation_error:
                # Accept approximate solution
                quality_score = 1.0 - (pos_err / self.config.ik_max_position_error)
            elif self.config.ik_fallback_to_previous:
                # Use previous configuration
                q_solution = self._prev_q.copy()
                quality_score = 0.5
            else:
                # Mark as invalid
                is_valid = False
                quality_score = 0.0
        else:
            self._ik_success_count += 1
        
        # Apply velocity smoothing
        q_smoothed = self._smooth_trajectory(q_solution, dt)
        
        # Map gripper
        if active_hand is not None:
            gripper_cmd = self.config.gripper_mapping.map(active_hand, self._prev_gripper)
        else:
            gripper_cmd = self._prev_gripper
        
        # Create action
        action = RobotAction(
            timestamp=human.timestamp,
            joint_position_target=q_smoothed,
            gripper_target=gripper_cmd,
            ee_pose_target=T_ee_desired,
        )
        
        # Create demo step
        step = DemoStep(
            step_index=t_index,
            obs=robot_obs,
            action=action,
            episode_id=episode_id,
            task_id=task_id,
            env_id=env_id,
            human_state=human,  # Kept locally for debugging
            is_valid=is_valid,
            quality_score=quality_score,
        )
        
        # Update state for next iteration
        self._prev_q = q_smoothed
        self._prev_T_ee = T_ee_desired
        self._prev_gripper = gripper_cmd
        
        return step
    
    @property
    def ik_success_rate(self) -> float:
        """Get IK success rate over all calls."""
        total = self._ik_success_count + self._ik_failure_count
        if total == 0:
            return 1.0
        return self._ik_success_count / total


# ---------------------------------------------------------------------------
# Calibration Utilities
# ---------------------------------------------------------------------------

def calibrate_hand_ee_transform(
    hand_poses: np.ndarray,
    ee_poses: np.ndarray,
    method: str = 'least_squares'
) -> np.ndarray:
    """
    Calibrate the T_ee_hand transform from paired hand and EE poses.
    
    During a calibration routine, the operator holds the robot's gripper
    with their gloved hand in various poses. We record both hand pose
    (from glove+camera) and EE pose (from robot FK), then solve for the
    transform that best relates them.
    
    The relationship is: T_ee_world = T_hand_world @ T_ee_hand
    So: T_ee_hand = inv(T_hand_world) @ T_ee_world
    
    We collect multiple samples and find the best average or least-squares fit.
    
    Args:
        hand_poses: [N, 4, 4] array of hand poses in world frame
        ee_poses: [N, 4, 4] array of corresponding EE poses in world frame
        method: 'average' or 'least_squares'
        
    Returns:
        T_ee_hand: Best-fit 4x4 transform from hand frame to EE frame
    """
    n_samples = hand_poses.shape[0]
    assert ee_poses.shape[0] == n_samples
    
    # Compute T_ee_hand for each sample
    T_samples = []
    for i in range(n_samples):
        T_hand_inv = np.linalg.inv(hand_poses[i])
        T_ee_hand_i = T_hand_inv @ ee_poses[i]
        T_samples.append(T_ee_hand_i)
    
    T_samples = np.array(T_samples)
    
    if method == 'average':
        # Simple average (works okay for small rotational variance)
        T_avg = np.mean(T_samples, axis=0)
        
        # Re-orthogonalize the rotation matrix
        U, _, Vt = np.linalg.svd(T_avg[:3, :3])
        T_avg[:3, :3] = U @ Vt
        
        return T_avg
    
    elif method == 'least_squares':
        # More sophisticated: average position, SLERP-average rotation
        positions = T_samples[:, :3, 3]
        pos_mean = np.mean(positions, axis=0)
        
        # For rotation averaging, convert to quaternions and average
        quats = []
        for T in T_samples:
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            quats.append(q)
        quats = np.array(quats)
        
        # Simple quaternion averaging (sign-corrected)
        # Make sure all quaternions are in same hemisphere
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        
        quat_mean = np.mean(quats, axis=0)
        quat_mean = quat_mean / np.linalg.norm(quat_mean)
        
        T_result = np.eye(4)
        T_result[:3, :3] = Rotation.from_quat(quat_mean).as_matrix()
        T_result[:3, 3] = pos_mean
        
        return T_result
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def verify_retargeting_quality(
    retargeter: Retargeter,
    test_poses: np.ndarray,
    robot_obs: RobotObs,
    threshold_pos: float = 0.02,
    threshold_rot: float = 0.1
) -> dict:
    """
    Verify retargeting quality by testing reachability of sample poses.
    
    Useful for checking whether a calibration is good before starting
    a long recording session.
    
    Args:
        retargeter: Configured retargeter instance
        test_poses: [N, 4, 4] array of sample EE poses to test
        robot_obs: Current robot observation (for IK seeding)
        threshold_pos: Max acceptable position error (meters)
        threshold_rot: Max acceptable rotation error (radians)
        
    Returns:
        Dictionary with statistics: success_rate, mean_pos_error, mean_rot_error
    """
    n_tests = test_poses.shape[0]
    pos_errors = []
    rot_errors = []
    successes = 0
    
    q_current = robot_obs.joint_positions.copy()
    
    for i in range(n_tests):
        T_target = test_poses[i]
        q_solution, success, _ = retargeter.ik.solve(q_current, T_target)
        
        T_achieved = retargeter.ik.forward_kinematics(q_solution)
        pos_err, rot_err = pose_error(T_achieved, T_target)
        
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)
        
        if pos_err < threshold_pos and rot_err < threshold_rot:
            successes += 1
        
        # Use solution as seed for next test
        q_current = q_solution
    
    return {
        'success_rate': successes / n_tests,
        'mean_pos_error': np.mean(pos_errors),
        'max_pos_error': np.max(pos_errors),
        'mean_rot_error': np.mean(rot_errors),
        'max_rot_error': np.max(rot_errors),
    }

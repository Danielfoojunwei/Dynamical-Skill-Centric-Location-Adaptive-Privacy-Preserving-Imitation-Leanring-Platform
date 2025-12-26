"""
Transition Safety - CBF-Based Safety Verification for Skill Transitions

Ensures that transitions between skills are safe using Control Barrier Functions.

Transition Safety Issues:
    1. State discontinuity: Skill A ends in state S1, Skill B expects state S2
    2. Velocity jumps: Sudden changes in velocity can cause instability
    3. Force spikes: Transitioning during contact can cause force spikes
    4. Safety invariant violation: Transition may violate safety constraints

Solution:
    - TransitionBarrier: Custom CBF for transition safety
    - Pre-transition check: Verify starting skill B from current state is safe
    - Smooth transition: Use CBF to compute safe transition trajectory
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransitionConfig:
    """Configuration for transition safety checking."""
    # Velocity limits during transition
    max_velocity_jump: float = 0.5  # rad/s or m/s
    max_acceleration: float = 2.0  # rad/s^2 or m/s^2

    # Position tolerance for precondition matching
    position_tolerance: float = 0.05  # meters

    # Force limits
    max_force_during_transition: float = 30.0  # Newtons

    # Transition duration
    min_transition_duration: float = 0.1  # seconds
    max_transition_duration: float = 1.0  # seconds

    # CBF parameters
    cbf_alpha: float = 1.0  # CBF decay rate
    safety_margin: float = 0.02  # meters

    # Action space
    action_dim: int = 7


@dataclass
class TransitionSafetyResult:
    """Result of transition safety check."""
    is_safe: bool
    skill_from: str
    skill_to: str

    # Safety metrics
    velocity_safe: bool = True
    position_safe: bool = True
    force_safe: bool = True
    cbf_safe: bool = True

    # Barrier values
    barrier_values: Dict[str, float] = field(default_factory=dict)
    min_barrier_value: float = 0.0

    # If not safe, suggested mitigation
    mitigation_action: Optional[np.ndarray] = None
    mitigation_strategy: Optional[str] = None

    # Details
    details: Dict[str, Any] = field(default_factory=dict)


class TransitionBarrier:
    """
    Control Barrier Function for safe transitions between skills.

    Ensures:
        1. Velocity continuity: ||v_current - v_target|| < threshold
        2. Acceleration limits: ||a|| < max_acceleration
        3. Position validity: position in safe set for next skill
        4. Force limits: ||f|| < max_force during transition
    """

    def __init__(self, config: TransitionConfig = None):
        self.config = config or TransitionConfig()
        self.name = "transition_barrier"
        self.alpha = self.config.cbf_alpha

    def h_velocity(
        self,
        current_vel: np.ndarray,
        target_vel: np.ndarray,
    ) -> float:
        """
        Velocity continuity barrier.

        h(x) = v_max^2 - ||v_current - v_target||^2
        """
        vel_diff = np.linalg.norm(current_vel - target_vel)
        return self.config.max_velocity_jump ** 2 - vel_diff ** 2

    def h_acceleration(
        self,
        current_vel: np.ndarray,
        next_vel: np.ndarray,
        dt: float,
    ) -> float:
        """
        Acceleration limit barrier.

        h(x) = a_max^2 - ||a||^2
        """
        if dt <= 0:
            return 0.0

        acceleration = np.linalg.norm((next_vel - current_vel) / dt)
        return self.config.max_acceleration ** 2 - acceleration ** 2

    def h_position(
        self,
        current_pos: np.ndarray,
        safe_region_center: np.ndarray,
        safe_region_radius: float,
    ) -> float:
        """
        Position validity barrier.

        h(x) = r_safe^2 - ||p - p_center||^2
        """
        dist = np.linalg.norm(current_pos - safe_region_center)
        return safe_region_radius ** 2 - dist ** 2

    def h_force(self, current_force: np.ndarray) -> float:
        """
        Force limit barrier.

        h(x) = f_max^2 - ||f||^2
        """
        force_mag = np.linalg.norm(current_force)
        return self.config.max_force_during_transition ** 2 - force_mag ** 2

    def compute_combined_barrier(
        self,
        current_state: Dict[str, Any],
        target_state: Dict[str, Any],
        dt: float = 0.01,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined barrier value for transition.

        Returns:
            Tuple of (min_barrier_value, dict of individual barriers)
        """
        barriers = {}

        # Velocity barrier
        current_vel = np.array(current_state.get('joint_velocities', np.zeros(7)))
        target_vel = np.array(target_state.get('joint_velocities', np.zeros(7)))
        barriers['velocity'] = self.h_velocity(current_vel, target_vel)

        # Acceleration barrier
        barriers['acceleration'] = self.h_acceleration(current_vel, target_vel, dt)

        # Position barrier (if target position specified)
        if 'ee_position' in current_state and 'expected_position' in target_state:
            current_pos = np.array(current_state['ee_position'])
            target_pos = np.array(target_state['expected_position'])
            barriers['position'] = self.h_position(
                current_pos,
                target_pos,
                self.config.position_tolerance * 2,  # Allow 2x tolerance
            )

        # Force barrier
        if 'ee_force' in current_state:
            current_force = np.array(current_state['ee_force'][:3])
            barriers['force'] = self.h_force(current_force)

        min_value = min(barriers.values()) if barriers else 0.0
        return min_value, barriers

    def compute_safe_transition_action(
        self,
        current_state: Dict[str, Any],
        proposed_action: np.ndarray,
        target_state: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute safe action that satisfies transition barriers.

        Uses gradient projection to find minimally modified action.
        """
        dt = 0.01  # Assume 100Hz control

        min_barrier, barriers = self.compute_combined_barrier(
            current_state, target_state, dt
        )

        # If already safe, return proposed action
        if min_barrier >= 0:
            return proposed_action

        # Find the most violated constraint
        most_violated = min(barriers.items(), key=lambda x: x[1])
        constraint_name, constraint_value = most_violated

        # Compute gradient for the violated constraint
        if constraint_name == 'velocity':
            # Gradient points toward reducing velocity difference
            current_vel = np.array(current_state.get('joint_velocities', np.zeros(7)))
            target_vel = np.array(target_state.get('joint_velocities', np.zeros(7)))
            vel_diff = current_vel - target_vel
            grad = -2 * vel_diff / (np.linalg.norm(vel_diff) + 1e-8)
        elif constraint_name == 'acceleration':
            # Gradient points toward reducing acceleration
            grad = -proposed_action / (np.linalg.norm(proposed_action) + 1e-8)
        elif constraint_name == 'force':
            # Gradient points toward reducing force
            current_force = np.array(current_state.get('ee_force', np.zeros(6))[:3])
            grad = np.zeros(self.config.action_dim)
            grad[:3] = -current_force / (np.linalg.norm(current_force) + 1e-8)
        else:
            grad = np.zeros(self.config.action_dim)

        # Project action
        step_size = 0.5
        safe_action = proposed_action + step_size * grad * abs(constraint_value)

        # Clamp to action limits
        safe_action = np.clip(safe_action, -1.0, 1.0)

        return safe_action


class TransitionSafety:
    """
    Main transition safety checker.

    Uses CBF to verify and ensure safe transitions between skills.
    """

    def __init__(self, config: TransitionConfig = None):
        self.config = config or TransitionConfig()
        self.barrier = TransitionBarrier(self.config)

        # Import CBF filter if available
        self._cbf_filter = None
        try:
            from ...safety.cbf.filter import CBFFilter
            self._cbf_filter = CBFFilter()
        except ImportError:
            logger.warning("CBFFilter not available for transition safety")

        # Statistics
        self.stats = {
            "transitions_checked": 0,
            "safe_transitions": 0,
            "unsafe_transitions": 0,
            "mitigations_applied": 0,
        }

    def check_transition_safety(
        self,
        skill_from: str,
        skill_to: str,
        current_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]] = None,
        proposed_action: Optional[np.ndarray] = None,
    ) -> TransitionSafetyResult:
        """
        Check if transitioning from skill_from to skill_to is safe.

        Args:
            skill_from: ID of current skill
            skill_to: ID of next skill
            current_state: Current robot state
            target_state: Expected state for next skill (optional)
            proposed_action: First action of next skill (optional)

        Returns:
            TransitionSafetyResult with safety assessment
        """
        self.stats["transitions_checked"] += 1

        # Default target state if not provided
        if target_state is None:
            target_state = {
                'joint_velocities': np.zeros(7),  # Expect zero velocity at start
            }

        # Compute barrier values
        min_barrier, barriers = self.barrier.compute_combined_barrier(
            current_state, target_state
        )

        # Check individual safety criteria
        velocity_safe = barriers.get('velocity', 0) >= 0
        acceleration_safe = barriers.get('acceleration', 0) >= 0
        position_safe = barriers.get('position', 0) >= 0 if 'position' in barriers else True
        force_safe = barriers.get('force', 0) >= 0 if 'force' in barriers else True

        is_safe = min_barrier >= 0

        # Use CBF filter for additional check if available
        cbf_safe = True
        if self._cbf_filter and proposed_action is not None:
            from ...safety.cbf.barriers import RobotState
            robot_state = RobotState.from_observation(current_state)
            cbf_result = self._cbf_filter.filter(proposed_action, robot_state)
            cbf_safe = not cbf_result.was_modified or cbf_result.modification_magnitude < 0.1
            is_safe = is_safe and cbf_safe

        # Compute mitigation if not safe
        mitigation_action = None
        mitigation_strategy = None

        if not is_safe:
            self.stats["unsafe_transitions"] += 1

            # Determine mitigation strategy
            if not velocity_safe:
                mitigation_strategy = "slow_down_before_transition"
            elif not force_safe:
                mitigation_strategy = "reduce_contact_force"
            elif not position_safe:
                mitigation_strategy = "adjust_position"
            else:
                mitigation_strategy = "use_safe_action"

            # Compute safe action if proposed action given
            if proposed_action is not None:
                mitigation_action = self.barrier.compute_safe_transition_action(
                    current_state, proposed_action, target_state
                )
                self.stats["mitigations_applied"] += 1
        else:
            self.stats["safe_transitions"] += 1

        return TransitionSafetyResult(
            is_safe=is_safe,
            skill_from=skill_from,
            skill_to=skill_to,
            velocity_safe=velocity_safe,
            position_safe=position_safe,
            force_safe=force_safe,
            cbf_safe=cbf_safe,
            barrier_values=barriers,
            min_barrier_value=min_barrier,
            mitigation_action=mitigation_action,
            mitigation_strategy=mitigation_strategy,
            details={
                "current_velocity_norm": float(np.linalg.norm(
                    current_state.get('joint_velocities', np.zeros(7))
                )),
                "current_force_norm": float(np.linalg.norm(
                    current_state.get('ee_force', np.zeros(6))[:3]
                )) if 'ee_force' in current_state else 0.0,
            },
        )

    def verify_transition_chain(
        self,
        skill_sequence: List[str],
        current_state: Dict[str, Any],
        skill_preconditions: Dict[str, Dict[str, Any]] = None,
    ) -> Tuple[bool, List[TransitionSafetyResult]]:
        """
        Verify safety of entire skill sequence.

        Args:
            skill_sequence: List of skill IDs in execution order
            current_state: Initial robot state
            skill_preconditions: Dict mapping skill ID to expected state

        Returns:
            Tuple of (all_transitions_safe, list of results)
        """
        if len(skill_sequence) < 2:
            return True, []

        skill_preconditions = skill_preconditions or {}
        results = []
        all_safe = True

        for i in range(len(skill_sequence) - 1):
            skill_from = skill_sequence[i]
            skill_to = skill_sequence[i + 1]

            # Get target state from preconditions
            target_state = skill_preconditions.get(skill_to, {})

            result = self.check_transition_safety(
                skill_from=skill_from,
                skill_to=skill_to,
                current_state=current_state,
                target_state=target_state,
            )

            results.append(result)
            if not result.is_safe:
                all_safe = False

            # Update current state estimate for next check
            # In practice, this would be predicted by the world model
            current_state = target_state.copy() if target_state else current_state.copy()

        return all_safe, results

    def compute_safe_transition_trajectory(
        self,
        current_state: Dict[str, Any],
        target_state: Dict[str, Any],
        num_steps: int = 10,
    ) -> List[np.ndarray]:
        """
        Compute a safe trajectory from current to target state.

        Uses CBF constraints to ensure safety at each step.
        """
        trajectory = []

        # Linear interpolation as initial trajectory
        current_pos = np.array(current_state.get('ee_position', np.zeros(3)))
        target_pos = np.array(target_state.get('ee_position', current_pos))

        current_vel = np.array(current_state.get('joint_velocities', np.zeros(7)))
        target_vel = np.array(target_state.get('joint_velocities', np.zeros(7)))

        for i in range(num_steps):
            t = (i + 1) / num_steps

            # Interpolate position and velocity
            interp_pos = current_pos + t * (target_pos - current_pos)
            interp_vel = current_vel + t * (target_vel - current_vel)

            # Create waypoint action
            # This is simplified - real implementation would use IK
            waypoint = np.zeros(self.config.action_dim)
            waypoint[:3] = (interp_pos - current_pos) / (num_steps * 0.01)  # Position delta

            # Check and modify for safety
            interp_state = {
                'ee_position': interp_pos,
                'joint_velocities': interp_vel,
            }

            safe_waypoint = self.barrier.compute_safe_transition_action(
                interp_state, waypoint, target_state
            )

            trajectory.append(safe_waypoint)

        return trajectory

    def get_statistics(self) -> Dict[str, Any]:
        """Get transition safety statistics."""
        total = self.stats["transitions_checked"]
        return {
            **self.stats,
            "safety_rate": self.stats["safe_transitions"] / max(1, total),
        }


class RuntimeTransitionChecker:
    """
    Runtime checker that combines postcondition verification and transition safety.

    This is the main interface for runtime composition verification.
    """

    def __init__(
        self,
        postcondition_verifier=None,
        transition_safety=None,
    ):
        self.postcondition_verifier = postcondition_verifier
        self.transition_safety = transition_safety or TransitionSafety()

    @classmethod
    def create(cls, unified_perception=None) -> 'RuntimeTransitionChecker':
        """Create a runtime checker with full perception support."""
        from .postcondition_verifier import PostconditionVerifier

        return cls(
            postcondition_verifier=PostconditionVerifier.from_unified_perception(unified_perception),
            transition_safety=TransitionSafety(),
        )

    def verify_skill_completion(
        self,
        skill_id: str,
        postconditions: List[str],
        frame: np.ndarray,
        robot_state: Dict[str, Any],
    ) -> Tuple[bool, List[Any]]:
        """
        Verify that a skill completed successfully by checking postconditions.

        Args:
            skill_id: ID of completed skill
            postconditions: List of postcondition predicates
            frame: Current camera frame
            robot_state: Current robot state

        Returns:
            Tuple of (all_verified, verification results)
        """
        if not self.postcondition_verifier or not postconditions:
            return True, []

        return self.postcondition_verifier.verify_all(
            postconditions, frame, robot_state
        )

    def verify_transition(
        self,
        skill_from: str,
        skill_to: str,
        postconditions: List[str],
        preconditions: List[str],
        frame: np.ndarray,
        robot_state: Dict[str, Any],
        proposed_action: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Complete verification for transitioning from one skill to another.

        1. Verifies postconditions of skill_from
        2. Verifies transition is safe (CBF)
        3. Verifies preconditions of skill_to are met

        Args:
            skill_from: Completing skill ID
            skill_to: Next skill ID
            postconditions: Postconditions of skill_from
            preconditions: Preconditions of skill_to
            frame: Current camera frame
            robot_state: Current robot state
            proposed_action: First action of skill_to

        Returns:
            Tuple of (transition_allowed, details dict)
        """
        details = {
            "skill_from": skill_from,
            "skill_to": skill_to,
            "postconditions_verified": False,
            "transition_safe": False,
            "preconditions_met": False,
            "transition_allowed": False,
        }

        # Step 1: Verify postconditions
        if postconditions and self.postcondition_verifier:
            post_verified, post_results = self.postcondition_verifier.verify_all(
                postconditions, frame, robot_state
            )
            details["postconditions_verified"] = post_verified
            details["postcondition_results"] = [
                {"predicate": r.predicate.name, "verified": r.verified, "confidence": r.confidence}
                for r in post_results
            ]

            if not post_verified:
                details["failure_stage"] = "postconditions"
                details["suggested_recovery"] = next(
                    (r.suggested_recovery for r in post_results if not r.verified),
                    "retry_skill"
                )
                return False, details
        else:
            details["postconditions_verified"] = True

        # Step 2: Verify transition safety
        safety_result = self.transition_safety.check_transition_safety(
            skill_from=skill_from,
            skill_to=skill_to,
            current_state=robot_state,
            proposed_action=proposed_action,
        )
        details["transition_safe"] = safety_result.is_safe
        details["safety_result"] = {
            "velocity_safe": safety_result.velocity_safe,
            "force_safe": safety_result.force_safe,
            "cbf_safe": safety_result.cbf_safe,
            "min_barrier": safety_result.min_barrier_value,
        }

        if not safety_result.is_safe:
            details["failure_stage"] = "transition_safety"
            details["suggested_recovery"] = safety_result.mitigation_strategy
            details["mitigation_action"] = (
                safety_result.mitigation_action.tolist()
                if safety_result.mitigation_action is not None else None
            )
            return False, details

        # Step 3: Verify preconditions
        if preconditions and self.postcondition_verifier:
            pre_verified, pre_results = self.postcondition_verifier.verify_all(
                preconditions, frame, robot_state
            )
            details["preconditions_met"] = pre_verified
            details["precondition_results"] = [
                {"predicate": r.predicate.name, "verified": r.verified, "confidence": r.confidence}
                for r in pre_results
            ]

            if not pre_verified:
                details["failure_stage"] = "preconditions"
                details["suggested_recovery"] = next(
                    (r.suggested_recovery for r in pre_results if not r.verified),
                    "setup_preconditions"
                )
                return False, details
        else:
            details["preconditions_met"] = True

        # All checks passed
        details["transition_allowed"] = True
        return True, details

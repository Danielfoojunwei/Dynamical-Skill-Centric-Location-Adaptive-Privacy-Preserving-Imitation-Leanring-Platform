"""
Tests for Runtime Composition Verification

Tests verify that:
1. Postcondition verification works with perception
2. Transition safety checking uses CBF correctly
3. Integration between verification and safety checking
4. Recovery suggestions are appropriate
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPostconditionVerifier:
    """Tests for PostconditionVerifier."""

    def test_verifier_import(self):
        """Test that verifier can be imported."""
        from src.composition.runtime import PostconditionVerifier
        verifier = PostconditionVerifier()
        assert verifier is not None

    def test_predicate_types(self):
        """Test predicate type creation."""
        from src.composition.runtime.postcondition_verifier import (
            Predicate, PredicateType
        )

        # Test holding predicate
        holding = Predicate.holding("cup")
        assert holding.pred_type == PredicateType.OBJECT_HELD
        assert "cup" in holding.name
        assert holding.args["object"] == "cup"

        # Test visible predicate
        visible = Predicate.visible("cup")
        assert visible.pred_type == PredicateType.OBJECT_VISIBLE

        # Test gripper state predicates
        closed = Predicate.gripper_closed()
        assert closed.pred_type == PredicateType.GRIPPER_STATE
        assert closed.args["state"] == "closed"

        opened = Predicate.gripper_open()
        assert opened.args["state"] == "open"

        # Test at_position predicate
        pos = np.array([0.5, 0.0, 0.3])
        at_pos = Predicate.at_position(pos)
        assert at_pos.pred_type == PredicateType.REACHED_POSITION
        assert np.allclose(at_pos.args["position"], pos)

        # Test stacked predicate
        stacked = Predicate.on("block_a", "block_b")
        assert stacked.pred_type == PredicateType.STACKED
        assert stacked.args["top_object"] == "block_a"
        assert stacked.args["bottom_object"] == "block_b"

    def test_robot_state_verifier_gripper(self):
        """Test gripper state verification."""
        from src.composition.runtime.postcondition_verifier import (
            RobotStateVerifier, Predicate
        )

        verifier = RobotStateVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Test gripper closed detection
        closed_state = {"gripper_state": 0.9}
        closed_pred = Predicate.gripper_closed()
        result = verifier.verify(closed_pred, frame, closed_state)
        assert result.verified is True
        assert result.method_used == "robot_state"

        # Test gripper open detection
        open_state = {"gripper_state": 0.1}
        open_pred = Predicate.gripper_open()
        result = verifier.verify(open_pred, frame, open_state)
        assert result.verified is True

        # Test failure case
        result = verifier.verify(closed_pred, frame, open_state)
        assert result.verified is False
        assert result.suggested_recovery == "close_gripper"

    def test_robot_state_verifier_object_held(self):
        """Test object held verification."""
        from src.composition.runtime.postcondition_verifier import (
            RobotStateVerifier, Predicate
        )

        verifier = RobotStateVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Object held = gripper partially closed + force detected
        holding_state = {
            "gripper_state": 0.6,  # Partially closed
            "gripper_force": 5.0,  # Force detected
        }

        holding_pred = Predicate.holding("object")
        result = verifier.verify(holding_pred, frame, holding_state)
        assert result.verified is True

        # Not holding - gripper too open
        not_holding_state = {
            "gripper_state": 0.1,
            "gripper_force": 0.0,
        }
        result = verifier.verify(holding_pred, frame, not_holding_state)
        assert result.verified is False
        assert result.suggested_recovery == "retry_grasp"

    def test_robot_state_verifier_position(self):
        """Test position verification."""
        from src.composition.runtime.postcondition_verifier import (
            RobotStateVerifier, Predicate
        )

        verifier = RobotStateVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        target_pos = np.array([0.5, 0.0, 0.3])
        position_pred = Predicate.at_position(target_pos, tolerance=0.05)

        # At target
        at_target_state = {"ee_position": np.array([0.51, 0.01, 0.29])}
        result = verifier.verify(position_pred, frame, at_target_state)
        assert result.verified == True

        # Not at target
        far_state = {"ee_position": np.array([0.0, 0.0, 0.0])}
        result = verifier.verify(position_pred, frame, far_state)
        assert result.verified == False
        assert "distance" in result.details

    def test_robot_state_verifier_contact(self):
        """Test contact detection verification."""
        from src.composition.runtime.postcondition_verifier import (
            RobotStateVerifier, Predicate, PredicateType
        )

        verifier = RobotStateVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        contact_pred = Predicate(
            pred_type=PredicateType.CONTACT_MADE,
            name="contact(surface)",
            args={"force_threshold": 1.0},
        )

        # Contact made
        contact_state = {"ee_force": np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])}
        result = verifier.verify(contact_pred, frame, contact_state)
        assert result.verified == True

        # No contact
        no_contact_state = {"ee_force": np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])}
        result = verifier.verify(contact_pred, frame, no_contact_state)
        assert result.verified == False

    def test_predicate_parsing(self):
        """Test string predicate parsing."""
        from src.composition.runtime import PostconditionVerifier
        from src.composition.runtime.postcondition_verifier import PredicateType

        verifier = PostconditionVerifier()

        # Parse holding predicate
        pred = verifier._parse_predicate("holding(cup)")
        assert pred.pred_type == PredicateType.OBJECT_HELD
        assert pred.args["object"] == "cup"

        # Parse visible predicate
        pred = verifier._parse_predicate("visible(bottle)")
        assert pred.pred_type == PredicateType.OBJECT_VISIBLE

        # Parse gripper predicates
        pred = verifier._parse_predicate("gripper_closed()")
        assert pred.pred_type == PredicateType.GRIPPER_STATE

        pred = verifier._parse_predicate("gripper_open()")
        assert pred.pred_type == PredicateType.GRIPPER_STATE

        # Parse at predicate
        pred = verifier._parse_predicate("at(cup, table)")
        assert pred.pred_type == PredicateType.OBJECT_AT
        assert pred.args["object"] == "cup"

        # Parse on predicate
        pred = verifier._parse_predicate("on(block_a, block_b)")
        assert pred.pred_type == PredicateType.STACKED

    def test_verify_with_string_predicate(self):
        """Test verification with string predicate."""
        from src.composition.runtime import PostconditionVerifier

        verifier = PostconditionVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot_state = {"gripper_state": 0.9}

        result = verifier.verify("gripper_closed()", frame, robot_state)
        assert result.verified is True

    def test_verify_all(self):
        """Test verifying multiple predicates."""
        from src.composition.runtime import PostconditionVerifier

        verifier = PostconditionVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot_state = {
            "gripper_state": 0.6,
            "gripper_force": 5.0,
            "ee_position": np.array([0.5, 0.0, 0.3]),
        }

        predicates = [
            "holding(object)",
        ]

        all_verified, results = verifier.verify_all(predicates, frame, robot_state)
        assert len(results) == 1
        assert results[0].verified is True

    def test_statistics(self):
        """Test verification statistics tracking."""
        from src.composition.runtime import PostconditionVerifier

        verifier = PostconditionVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Run some verifications
        verifier.verify("gripper_closed()", frame, {"gripper_state": 0.9})
        verifier.verify("gripper_open()", frame, {"gripper_state": 0.1})
        verifier.verify("gripper_closed()", frame, {"gripper_state": 0.1})

        stats = verifier.get_statistics()
        assert stats["total_verifications"] == 3
        assert stats["successful_verifications"] == 2
        assert stats["failed_verifications"] == 1


class TestTransitionSafety:
    """Tests for TransitionSafety."""

    def test_transition_safety_import(self):
        """Test that transition safety can be imported."""
        from src.composition.runtime import TransitionSafety, TransitionConfig
        config = TransitionConfig()
        checker = TransitionSafety(config)
        assert checker is not None

    def test_transition_barrier(self):
        """Test transition barrier functions."""
        from src.composition.runtime.transition_safety import (
            TransitionBarrier, TransitionConfig
        )

        config = TransitionConfig(max_velocity_jump=0.5)
        barrier = TransitionBarrier(config)

        # Test velocity barrier
        current_vel = np.zeros(7)
        target_vel = np.zeros(7)
        h = barrier.h_velocity(current_vel, target_vel)
        assert h > 0  # Same velocity = safe

        # Velocity jump
        current_vel = np.zeros(7)
        target_vel = np.array([1.0, 0, 0, 0, 0, 0, 0])  # Big jump
        h = barrier.h_velocity(current_vel, target_vel)
        assert h < 0  # Large jump = unsafe

        # Test force barrier
        low_force = np.array([5.0, 0, 0])
        h = barrier.h_force(low_force)
        assert h > 0  # Low force = safe

        high_force = np.array([50.0, 0, 0])
        h = barrier.h_force(high_force)
        assert h < 0  # High force = unsafe

    def test_check_transition_safety_safe(self):
        """Test checking a safe transition."""
        from src.composition.runtime import TransitionSafety

        checker = TransitionSafety()

        current_state = {
            "joint_velocities": np.zeros(7),
            "ee_force": np.zeros(6),
        }

        target_state = {
            "joint_velocities": np.zeros(7),  # Same velocity = safe
        }

        result = checker.check_transition_safety(
            skill_from="grasp",
            skill_to="place",
            current_state=current_state,
            target_state=target_state,
        )

        assert result.is_safe == True
        assert result.velocity_safe == True
        assert result.force_safe == True

    def test_check_transition_safety_unsafe_velocity(self):
        """Test detecting unsafe velocity transition."""
        from src.composition.runtime import TransitionSafety, TransitionConfig

        config = TransitionConfig(max_velocity_jump=0.3)
        checker = TransitionSafety(config)

        current_state = {
            "joint_velocities": np.array([1.0, 0, 0, 0, 0, 0, 0]),
            "ee_force": np.zeros(6),
        }

        target_state = {
            "joint_velocities": np.zeros(7),  # Sudden stop
        }

        result = checker.check_transition_safety(
            skill_from="move",
            skill_to="stop",
            current_state=current_state,
            target_state=target_state,
        )

        assert result.is_safe == False
        assert result.velocity_safe == False
        assert result.mitigation_strategy is not None

    def test_check_transition_safety_unsafe_force(self):
        """Test detecting unsafe force during transition."""
        from src.composition.runtime import TransitionSafety, TransitionConfig

        config = TransitionConfig(max_force_during_transition=10.0)
        checker = TransitionSafety(config)

        current_state = {
            "joint_velocities": np.zeros(7),
            "ee_force": np.array([20.0, 0, 0, 0, 0, 0]),  # High force
        }

        result = checker.check_transition_safety(
            skill_from="push",
            skill_to="release",
            current_state=current_state,
        )

        assert result.is_safe == False
        assert result.force_safe == False
        assert "reduce_contact_force" in result.mitigation_strategy

    def test_verify_transition_chain(self):
        """Test verifying a chain of transitions."""
        from src.composition.runtime import TransitionSafety

        checker = TransitionSafety()

        skill_sequence = ["approach", "grasp", "lift", "move", "place"]
        current_state = {
            "joint_velocities": np.zeros(7),
            "ee_force": np.zeros(6),
        }

        all_safe, results = checker.verify_transition_chain(
            skill_sequence, current_state
        )

        # With zero velocities and forces, should be safe
        assert all_safe is True
        assert len(results) == 4  # n-1 transitions

    def test_safe_transition_trajectory(self):
        """Test computing safe transition trajectory."""
        from src.composition.runtime import TransitionSafety

        checker = TransitionSafety()

        current_state = {
            "ee_position": np.array([0.0, 0.0, 0.0]),
            "joint_velocities": np.zeros(7),
        }

        target_state = {
            "ee_position": np.array([0.5, 0.0, 0.3]),
            "joint_velocities": np.zeros(7),
        }

        trajectory = checker.compute_safe_transition_trajectory(
            current_state, target_state, num_steps=5
        )

        assert len(trajectory) == 5
        assert all(isinstance(wp, np.ndarray) for wp in trajectory)

    def test_statistics(self):
        """Test transition safety statistics."""
        from src.composition.runtime import TransitionSafety

        checker = TransitionSafety()

        state = {"joint_velocities": np.zeros(7), "ee_force": np.zeros(6)}

        # Run some checks
        checker.check_transition_safety("a", "b", state)
        checker.check_transition_safety("b", "c", state)

        stats = checker.get_statistics()
        assert stats["transitions_checked"] == 2
        assert "safety_rate" in stats


class TestRuntimeTransitionChecker:
    """Tests for RuntimeTransitionChecker integration."""

    def test_runtime_checker_import(self):
        """Test that RuntimeTransitionChecker can be imported."""
        from src.composition.runtime.transition_safety import RuntimeTransitionChecker
        checker = RuntimeTransitionChecker()
        assert checker is not None

    def test_verify_skill_completion(self):
        """Test verifying skill completion."""
        from src.composition.runtime.transition_safety import RuntimeTransitionChecker
        from src.composition.runtime import PostconditionVerifier

        checker = RuntimeTransitionChecker(
            postcondition_verifier=PostconditionVerifier(),
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot_state = {"gripper_state": 0.9}

        verified, results = checker.verify_skill_completion(
            skill_id="close_gripper",
            postconditions=["gripper_closed()"],
            frame=frame,
            robot_state=robot_state,
        )

        assert verified is True
        assert len(results) == 1

    def test_full_transition_verification(self):
        """Test complete transition verification."""
        from src.composition.runtime.transition_safety import RuntimeTransitionChecker
        from src.composition.runtime import PostconditionVerifier, TransitionSafety

        checker = RuntimeTransitionChecker(
            postcondition_verifier=PostconditionVerifier(),
            transition_safety=TransitionSafety(),
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot_state = {
            "gripper_state": 0.6,
            "gripper_force": 5.0,
            "joint_velocities": np.zeros(7),
            "ee_force": np.zeros(6),
        }

        allowed, details = checker.verify_transition(
            skill_from="grasp",
            skill_to="lift",
            postconditions=["holding(object)"],
            preconditions=["holding(object)"],
            frame=frame,
            robot_state=robot_state,
        )

        assert allowed == True
        assert details["postconditions_verified"] == True
        assert details["transition_safe"] == True
        assert details["preconditions_met"] == True

    def test_transition_failure_postconditions(self):
        """Test transition failure due to postconditions."""
        from src.composition.runtime.transition_safety import RuntimeTransitionChecker
        from src.composition.runtime import PostconditionVerifier, TransitionSafety

        checker = RuntimeTransitionChecker(
            postcondition_verifier=PostconditionVerifier(),
            transition_safety=TransitionSafety(),
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        robot_state = {
            "gripper_state": 0.1,  # Open - not holding
            "gripper_force": 0.0,
            "joint_velocities": np.zeros(7),
            "ee_force": np.zeros(6),
        }

        allowed, details = checker.verify_transition(
            skill_from="grasp",
            skill_to="lift",
            postconditions=["holding(object)"],
            preconditions=[],
            frame=frame,
            robot_state=robot_state,
        )

        assert allowed is False
        assert details["failure_stage"] == "postconditions"
        assert "suggested_recovery" in details


class TestIntegrationWithComposition:
    """Integration tests with composition module."""

    def test_import_from_composition(self):
        """Test that runtime verification can be imported from composition."""
        from src.composition import (
            PostconditionVerifier,
            TransitionSafety,
            TransitionSafetyResult,
        )

        assert PostconditionVerifier is not None
        assert TransitionSafety is not None
        assert TransitionSafetyResult is not None

    def test_combined_static_runtime_verification(self):
        """Test combining static contract verification with runtime verification."""
        from src.composition import CompositionVerifier
        from src.composition.runtime.transition_safety import RuntimeTransitionChecker
        from src.composition.runtime import PostconditionVerifier, TransitionSafety

        # Static verification (before execution)
        static_verifier = CompositionVerifier()

        # Runtime verification (during execution)
        runtime_checker = RuntimeTransitionChecker(
            postcondition_verifier=PostconditionVerifier(),
            transition_safety=TransitionSafety(),
        )

        # Both should be usable together
        assert static_verifier is not None
        assert runtime_checker is not None


class TestRecoverySuggestions:
    """Tests for recovery suggestions."""

    def test_gripper_recovery(self):
        """Test recovery suggestions for gripper failures."""
        from src.composition.runtime import PostconditionVerifier

        verifier = PostconditionVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Gripper open when should be closed
        result = verifier.verify(
            "gripper_closed()",
            frame,
            {"gripper_state": 0.1}
        )
        assert result.suggested_recovery == "close_gripper"

        # Gripper closed when should be open
        result = verifier.verify(
            "gripper_open()",
            frame,
            {"gripper_state": 0.9}
        )
        assert result.suggested_recovery == "open_gripper"

    def test_grasp_recovery(self):
        """Test recovery suggestions for grasp failures."""
        from src.composition.runtime import PostconditionVerifier

        verifier = PostconditionVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = verifier.verify(
            "holding(object)",
            frame,
            {"gripper_state": 0.1, "gripper_force": 0.0}
        )
        assert result.suggested_recovery == "retry_grasp"

    def test_position_recovery(self):
        """Test recovery suggestions for position failures."""
        from src.composition.runtime.postcondition_verifier import (
            RobotStateVerifier, Predicate
        )

        verifier = RobotStateVerifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        target = np.array([0.5, 0.0, 0.3])
        current = np.array([0.0, 0.0, 0.0])

        result = verifier.verify(
            Predicate.at_position(target),
            frame,
            {"ee_position": current}
        )
        assert result.suggested_recovery == "refine_position"

    def test_transition_safety_mitigation(self):
        """Test transition safety mitigation actions."""
        from src.composition.runtime import TransitionSafety, TransitionConfig

        config = TransitionConfig(max_velocity_jump=0.1)
        checker = TransitionSafety(config)

        current_state = {
            "joint_velocities": np.array([0.5, 0, 0, 0, 0, 0, 0]),
            "ee_force": np.zeros(6),
        }

        proposed_action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = checker.check_transition_safety(
            skill_from="move",
            skill_to="stop",
            current_state=current_state,
            proposed_action=proposed_action,
        )

        # Should suggest slowing down
        assert result.mitigation_strategy == "slow_down_before_transition"
        # Should provide mitigation action
        assert result.mitigation_action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

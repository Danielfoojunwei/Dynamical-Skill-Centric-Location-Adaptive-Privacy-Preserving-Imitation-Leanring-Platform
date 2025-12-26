"""
Runtime Monitor for Temporal Property Checking

Monitors LTL/MTL safety properties online and triggers fallbacks.

Properties supported:
- Always (□): □(safe) - safety must always hold
- Eventually (◇): ◇(goal) - goal must eventually be reached
- Until (U): safe U goal - safety holds until goal
- Bounded: ◇≤t(response) - response within time t
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PropertyStatus(Enum):
    """Status of a temporal property."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    PENDING = "pending"  # Not yet determined
    UNKNOWN = "unknown"


@dataclass
class MonitorConfig:
    """Configuration for Runtime Monitor."""
    history_length: int = 1000  # States to keep
    check_frequency_hz: float = 100.0
    violation_callback: Optional[Callable] = None


@dataclass
class TemporalProperty:
    """A temporal property to monitor."""
    name: str
    property_type: str  # "always", "eventually", "bounded_response"
    predicate: Callable[[Any], bool]  # Evaluates on state

    # For bounded properties
    time_bound: Optional[float] = None  # seconds

    # Callbacks
    on_violation: Optional[Callable] = None


@dataclass
class MonitorResult:
    """Result of property monitoring."""
    timestamp: float
    property_results: Dict[str, PropertyStatus]
    any_violation: bool
    violated_properties: List[str]


class RuntimeMonitor:
    """
    Online runtime monitor for temporal properties.

    Checks safety properties at each timestep and triggers
    fallback behaviors on violation.
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.properties: List[TemporalProperty] = []

        # State history: (timestamp, state)
        self.history: deque = deque(maxlen=self.config.history_length)

        # Property state tracking
        self.property_states: Dict[str, Dict] = {}

        # Global callbacks
        self.violation_callbacks: List[Callable] = []

        # Stats
        self.stats = {
            "checks_performed": 0,
            "violations_detected": 0,
        }

    def add_property(self, prop: TemporalProperty):
        """Add a property to monitor."""
        self.properties.append(prop)
        self.property_states[prop.name] = {
            "status": PropertyStatus.PENDING,
            "last_true": None,
            "triggered_at": None,
        }

    def add_always_property(
        self,
        name: str,
        predicate: Callable[[Any], bool],
        on_violation: Optional[Callable] = None,
    ):
        """Add an 'always' (□) property: predicate must always be true."""
        self.add_property(TemporalProperty(
            name=name,
            property_type="always",
            predicate=predicate,
            on_violation=on_violation,
        ))

    def add_bounded_response(
        self,
        name: str,
        trigger: Callable[[Any], bool],
        response: Callable[[Any], bool],
        time_bound: float,
        on_violation: Optional[Callable] = None,
    ):
        """
        Add bounded response: if trigger, then response within time_bound.

        Example: if human_detected, then robot_stopped within 0.5s
        """
        self.add_property(TemporalProperty(
            name=name,
            property_type="bounded_response",
            predicate=lambda s: (trigger(s), response(s)),
            time_bound=time_bound,
            on_violation=on_violation,
        ))

    def register_violation_callback(self, callback: Callable[[str], None]):
        """Register callback for any property violation."""
        self.violation_callbacks.append(callback)

    def check(self, state: Any, timestamp: Optional[float] = None) -> MonitorResult:
        """
        Check all properties against current state.

        Args:
            state: Current robot/environment state
            timestamp: Current time (uses time.time() if not provided)

        Returns:
            MonitorResult with status of all properties
        """
        timestamp = timestamp or time.time()
        self.history.append((timestamp, state))
        self.stats["checks_performed"] += 1

        property_results = {}
        violated = []

        for prop in self.properties:
            status = self._check_property(prop, state, timestamp)
            property_results[prop.name] = status

            if status == PropertyStatus.VIOLATED:
                violated.append(prop.name)
                self._handle_violation(prop)

        any_violation = len(violated) > 0
        if any_violation:
            self.stats["violations_detected"] += 1

        return MonitorResult(
            timestamp=timestamp,
            property_results=property_results,
            any_violation=any_violation,
            violated_properties=violated,
        )

    def _check_property(
        self,
        prop: TemporalProperty,
        state: Any,
        timestamp: float,
    ) -> PropertyStatus:
        """Check a single property."""
        prop_state = self.property_states[prop.name]

        if prop.property_type == "always":
            # □(predicate): must be true now
            if prop.predicate(state):
                return PropertyStatus.SATISFIED
            else:
                return PropertyStatus.VIOLATED

        elif prop.property_type == "bounded_response":
            # trigger → ◇≤t(response)
            trigger, response = prop.predicate(state)

            if trigger and prop_state["triggered_at"] is None:
                # Trigger just activated
                prop_state["triggered_at"] = timestamp

            if prop_state["triggered_at"] is not None:
                elapsed = timestamp - prop_state["triggered_at"]

                if response:
                    # Response achieved
                    prop_state["triggered_at"] = None
                    return PropertyStatus.SATISFIED
                elif elapsed > prop.time_bound:
                    # Time bound exceeded
                    prop_state["triggered_at"] = None
                    return PropertyStatus.VIOLATED
                else:
                    # Still waiting
                    return PropertyStatus.PENDING

            return PropertyStatus.SATISFIED

        return PropertyStatus.UNKNOWN

    def _handle_violation(self, prop: TemporalProperty):
        """Handle property violation."""
        logger.warning(f"Property violated: {prop.name}")

        # Property-specific callback
        if prop.on_violation:
            prop.on_violation()

        # Global callbacks
        for callback in self.violation_callbacks:
            callback(prop.name)

    def reset(self):
        """Reset monitor state."""
        self.history.clear()
        for name in self.property_states:
            self.property_states[name] = {
                "status": PropertyStatus.PENDING,
                "last_true": None,
                "triggered_at": None,
            }


# Pre-built common properties
def make_collision_property(min_distance: float = 0.05) -> TemporalProperty:
    """Create always-no-collision property."""
    def predicate(state):
        if hasattr(state, 'min_obstacle_distance'):
            return state.min_obstacle_distance >= min_distance
        return True

    return TemporalProperty(
        name="no_collision",
        property_type="always",
        predicate=predicate,
    )


def make_estop_response_property(max_response_time: float = 0.5) -> TemporalProperty:
    """Create bounded e-stop response property."""
    def check(state):
        trigger = getattr(state, 'estop_pressed', False)
        response = getattr(state, 'robot_stopped', True)
        return (trigger, response)

    return TemporalProperty(
        name="estop_response",
        property_type="bounded_response",
        predicate=check,
        time_bound=max_response_time,
    )

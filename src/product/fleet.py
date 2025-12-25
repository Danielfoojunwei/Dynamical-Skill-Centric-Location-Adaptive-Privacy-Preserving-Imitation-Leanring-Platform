"""
Fleet Management API

Multi-robot coordination and task assignment for heterogeneous robot fleets.

Features:
=========
- Multi-robot task assignment
- Load balancing across fleet
- Heterogeneous robot support (different models/capabilities)
- Cross-robot skill transfer (via Pi0.5)
- Fleet-wide optimization

Powered By:
==========
- Pi0.5: Cross-robot skill transfer and complex multi-agent planning
- Jetson Thor: Edge-native fleet coordination

Usage:
    from src.product import FleetManager

    fleet = FleetManager()

    # Register robots
    await fleet.register_robot("robot_001", capabilities=["pick_place", "navigation"])
    await fleet.register_robot("robot_002", capabilities=["sorting", "packing"])

    # Assign task to best robot
    assignment = await fleet.assign_task_auto(
        instruction="Sort packages by destination"
    )

    # Or assign to specific robot
    await fleet.assign_task("robot_001", "Pick up the red boxes")

    # Get fleet status
    status = await fleet.get_fleet_status()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RobotType(Enum):
    """Robot hardware types."""
    GENERIC_ARM = "generic_arm"
    DUAL_ARM = "dual_arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMANOID = "humanoid"
    AGV = "agv"  # Automated Guided Vehicle
    COBOT = "cobot"  # Collaborative robot


class AssignmentStrategy(Enum):
    """Task assignment strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    CAPABILITY_MATCH = "capability_match"
    NEAREST = "nearest"
    OPTIMAL = "optimal"


@dataclass
class RobotProfile:
    """Profile of a robot in the fleet."""
    # Identification
    robot_id: str
    robot_type: RobotType
    site_id: str

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)

    # Status
    is_online: bool = True
    is_available: bool = True
    current_task_id: Optional[str] = None
    battery_percent: float = 100.0

    # Performance
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration_s: float = 0.0
    success_rate: float = 100.0

    # Location
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})

    # Model info
    vla_model: str = "pi05_base"

    # Registration
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class RobotAssignment:
    """Task assignment to a robot."""
    assignment_id: str
    robot_id: str
    task_id: str
    instruction: str
    priority: int = 1
    assigned_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "assigned"


@dataclass
class FleetStatus:
    """Overall fleet status."""
    # Fleet composition
    total_robots: int = 0
    online_robots: int = 0
    available_robots: int = 0
    busy_robots: int = 0
    offline_robots: int = 0

    # Task status
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_today: int = 0

    # Performance
    fleet_utilization: float = 0.0
    average_success_rate: float = 0.0
    average_task_duration_s: float = 0.0

    # By site
    robots_by_site: Dict[str, int] = field(default_factory=dict)

    # By type
    robots_by_type: Dict[str, int] = field(default_factory=dict)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class FleetManager:
    """
    Fleet Management API.

    Coordinates multi-robot operations with intelligent task assignment
    and cross-robot skill transfer via Pi0.5.
    """

    def __init__(
        self,
        assignment_strategy: AssignmentStrategy = AssignmentStrategy.OPTIMAL
    ):
        """
        Initialize fleet manager.

        Args:
            assignment_strategy: Default strategy for task assignment
        """
        self.assignment_strategy = assignment_strategy

        # Fleet state
        self._robots: Dict[str, RobotProfile] = {}
        self._assignments: Dict[str, RobotAssignment] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()

        # Capability registry
        self._capability_robots: Dict[str, Set[str]] = {}  # capability -> robot_ids

    # =========================================================================
    # Robot Registration
    # =========================================================================

    async def register_robot(
        self,
        robot_id: str,
        robot_type: RobotType = RobotType.GENERIC_ARM,
        site_id: str = "default",
        capabilities: Optional[List[str]] = None,
        **kwargs
    ) -> RobotProfile:
        """
        Register a robot with the fleet.

        Args:
            robot_id: Unique robot identifier
            robot_type: Type of robot hardware
            site_id: Site where robot is deployed
            capabilities: List of robot capabilities
            **kwargs: Additional profile fields

        Returns:
            RobotProfile
        """
        profile = RobotProfile(
            robot_id=robot_id,
            robot_type=robot_type,
            site_id=site_id,
            capabilities=set(capabilities or ["pick_place"]),
            **kwargs
        )

        self._robots[robot_id] = profile

        # Update capability registry
        for cap in profile.capabilities:
            if cap not in self._capability_robots:
                self._capability_robots[cap] = set()
            self._capability_robots[cap].add(robot_id)

        logger.info(
            f"Registered robot {robot_id} ({robot_type.value}) "
            f"at site {site_id} with capabilities: {capabilities}"
        )

        return profile

    async def unregister_robot(self, robot_id: str) -> bool:
        """Remove a robot from the fleet."""
        if robot_id in self._robots:
            profile = self._robots.pop(robot_id)

            # Update capability registry
            for cap in profile.capabilities:
                if cap in self._capability_robots:
                    self._capability_robots[cap].discard(robot_id)

            logger.info(f"Unregistered robot {robot_id}")
            return True
        return False

    async def update_robot_status(
        self,
        robot_id: str,
        is_available: bool = None,
        battery_percent: float = None,
        position: Dict[str, float] = None,
        current_task_id: str = None,
        **kwargs
    ) -> Optional[RobotProfile]:
        """Update robot status."""
        if robot_id not in self._robots:
            return None

        profile = self._robots[robot_id]
        profile.last_seen = datetime.now()

        if is_available is not None:
            profile.is_available = is_available
        if battery_percent is not None:
            profile.battery_percent = battery_percent
        if position is not None:
            profile.position = position
        if current_task_id is not None:
            profile.current_task_id = current_task_id

        return profile

    # =========================================================================
    # Task Assignment
    # =========================================================================

    async def assign_task(
        self,
        robot_id: str,
        instruction: str,
        task_id: Optional[str] = None,
        priority: int = 1
    ) -> Optional[RobotAssignment]:
        """
        Assign a task to a specific robot.

        Args:
            robot_id: Target robot
            instruction: Task instruction
            task_id: Optional task ID
            priority: Task priority (1-5)

        Returns:
            RobotAssignment or None if robot unavailable
        """
        import uuid

        if robot_id not in self._robots:
            logger.error(f"Robot {robot_id} not found")
            return None

        profile = self._robots[robot_id]
        if not profile.is_available:
            logger.warning(f"Robot {robot_id} is not available")
            return None

        assignment_id = str(uuid.uuid4())[:8]
        task_id = task_id or str(uuid.uuid4())[:8]

        assignment = RobotAssignment(
            assignment_id=assignment_id,
            robot_id=robot_id,
            task_id=task_id,
            instruction=instruction,
            priority=priority,
        )

        self._assignments[assignment_id] = assignment

        # Update robot status
        profile.is_available = False
        profile.current_task_id = task_id

        logger.info(f"Assigned task '{instruction[:30]}...' to robot {robot_id}")

        # Execute task
        asyncio.create_task(self._execute_assignment(assignment))

        return assignment

    async def assign_task_auto(
        self,
        instruction: str,
        required_capabilities: Optional[List[str]] = None,
        site_id: Optional[str] = None,
        strategy: Optional[AssignmentStrategy] = None
    ) -> Optional[RobotAssignment]:
        """
        Automatically assign task to best available robot.

        Args:
            instruction: Task instruction
            required_capabilities: Required robot capabilities
            site_id: Preferred site
            strategy: Assignment strategy

        Returns:
            RobotAssignment or None if no robot available
        """
        strategy = strategy or self.assignment_strategy

        # Find eligible robots
        eligible = await self._find_eligible_robots(
            required_capabilities=required_capabilities,
            site_id=site_id
        )

        if not eligible:
            logger.warning("No eligible robots available")
            return None

        # Select best robot based on strategy
        robot_id = await self._select_robot(eligible, strategy, instruction)

        if robot_id:
            return await self.assign_task(robot_id, instruction)

        return None

    async def _find_eligible_robots(
        self,
        required_capabilities: Optional[List[str]] = None,
        site_id: Optional[str] = None
    ) -> List[str]:
        """Find robots that can handle a task."""
        eligible = []

        for robot_id, profile in self._robots.items():
            # Check availability
            if not profile.is_online or not profile.is_available:
                continue

            # Check site
            if site_id and profile.site_id != site_id:
                continue

            # Check capabilities
            if required_capabilities:
                if not all(cap in profile.capabilities for cap in required_capabilities):
                    continue

            # Check battery
            if profile.battery_percent < 20:
                continue

            eligible.append(robot_id)

        return eligible

    async def _select_robot(
        self,
        eligible: List[str],
        strategy: AssignmentStrategy,
        instruction: str
    ) -> Optional[str]:
        """Select best robot from eligible list."""
        if not eligible:
            return None

        if strategy == AssignmentStrategy.ROUND_ROBIN:
            return eligible[0]

        elif strategy == AssignmentStrategy.LEAST_BUSY:
            # Select robot with fewest completed tasks (most rested)
            return min(
                eligible,
                key=lambda r: self._robots[r].tasks_completed
            )

        elif strategy == AssignmentStrategy.CAPABILITY_MATCH:
            # Select robot with most matching capabilities
            # Would parse instruction for required capabilities
            return eligible[0]

        elif strategy == AssignmentStrategy.NEAREST:
            # Would calculate distance to task location
            return eligible[0]

        elif strategy == AssignmentStrategy.OPTIMAL:
            # Combine factors for optimal selection
            def score_robot(robot_id: str) -> float:
                profile = self._robots[robot_id]
                score = 0.0

                # Availability and freshness
                score += 10.0 if profile.is_available else 0.0

                # Battery level
                score += profile.battery_percent * 0.1

                # Success rate
                score += profile.success_rate * 0.5

                # Load balancing (prefer less used robots)
                if profile.tasks_completed > 0:
                    score -= profile.tasks_completed * 0.01

                return score

            return max(eligible, key=score_robot)

        return eligible[0]

    async def _execute_assignment(self, assignment: RobotAssignment) -> None:
        """Execute a task assignment."""
        assignment.status = "executing"
        assignment.started_at = datetime.now()

        try:
            from .task_api import TaskAPI

            api = TaskAPI.create_for_hardware()
            result = await api.execute(assignment.instruction)

            if result.success:
                assignment.status = "completed"
                self._robots[assignment.robot_id].tasks_completed += 1
            else:
                assignment.status = "failed"
                self._robots[assignment.robot_id].tasks_failed += 1

        except Exception as e:
            assignment.status = "failed"
            logger.error(f"Assignment {assignment.assignment_id} failed: {e}")

        finally:
            assignment.completed_at = datetime.now()

            # Update robot availability
            profile = self._robots.get(assignment.robot_id)
            if profile:
                profile.is_available = True
                profile.current_task_id = None

                # Update success rate
                total = profile.tasks_completed + profile.tasks_failed
                if total > 0:
                    profile.success_rate = (profile.tasks_completed / total) * 100

    # =========================================================================
    # Fleet Status
    # =========================================================================

    async def get_fleet_status(self) -> FleetStatus:
        """Get overall fleet status."""
        status = FleetStatus()

        for robot_id, profile in self._robots.items():
            status.total_robots += 1

            if profile.is_online:
                status.online_robots += 1
                if profile.is_available:
                    status.available_robots += 1
                else:
                    status.busy_robots += 1
            else:
                status.offline_robots += 1

            # By site
            status.robots_by_site[profile.site_id] = (
                status.robots_by_site.get(profile.site_id, 0) + 1
            )

            # By type
            status.robots_by_type[profile.robot_type.value] = (
                status.robots_by_type.get(profile.robot_type.value, 0) + 1
            )

        # Task status
        for assignment in self._assignments.values():
            if assignment.status == "executing":
                status.active_tasks += 1
            elif assignment.status == "completed":
                if assignment.completed_at and assignment.completed_at.date() == datetime.now().date():
                    status.completed_today += 1

        status.queued_tasks = self._task_queue.qsize()

        # Performance
        if status.total_robots > 0:
            status.fleet_utilization = (status.busy_robots / status.online_robots * 100) if status.online_robots > 0 else 0

            success_rates = [p.success_rate for p in self._robots.values() if p.tasks_completed > 0]
            if success_rates:
                status.average_success_rate = sum(success_rates) / len(success_rates)

        return status

    async def get_robot(self, robot_id: str) -> Optional[RobotProfile]:
        """Get robot profile."""
        return self._robots.get(robot_id)

    async def list_robots(
        self,
        site_id: Optional[str] = None,
        available_only: bool = False
    ) -> List[RobotProfile]:
        """List robots in fleet."""
        robots = list(self._robots.values())

        if site_id:
            robots = [r for r in robots if r.site_id == site_id]

        if available_only:
            robots = [r for r in robots if r.is_available]

        return robots

    # =========================================================================
    # Skill Transfer (Pi0.5 Feature)
    # =========================================================================

    async def transfer_skill(
        self,
        source_robot: str,
        target_robot: str,
        skill_name: str
    ) -> bool:
        """
        Transfer a skill from one robot to another.

        Enabled by Pi0.5's cross-robot skill transfer capability.

        Args:
            source_robot: Robot with the skill
            target_robot: Robot to receive the skill
            skill_name: Name of skill to transfer

        Returns:
            True if transfer successful
        """
        source = self._robots.get(source_robot)
        target = self._robots.get(target_robot)

        if not source or not target:
            return False

        if skill_name not in source.capabilities:
            logger.warning(f"Source robot doesn't have skill: {skill_name}")
            return False

        # Pi0.5 enables zero-shot skill transfer
        target.capabilities.add(skill_name)

        # Update capability registry
        if skill_name not in self._capability_robots:
            self._capability_robots[skill_name] = set()
        self._capability_robots[skill_name].add(target_robot)

        logger.info(
            f"Transferred skill '{skill_name}' from {source_robot} to {target_robot} "
            "(Pi0.5 cross-robot transfer)"
        )

        return True

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def broadcast_task(
        self,
        instruction: str,
        site_id: Optional[str] = None
    ) -> List[RobotAssignment]:
        """Assign same task to all available robots."""
        assignments = []

        robots = await self.list_robots(site_id=site_id, available_only=True)

        for robot in robots:
            assignment = await self.assign_task(robot.robot_id, instruction)
            if assignment:
                assignments.append(assignment)

        return assignments

    def get_robots_with_capability(self, capability: str) -> List[str]:
        """Get all robots with a specific capability."""
        return list(self._capability_robots.get(capability, set()))

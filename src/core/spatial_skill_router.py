"""
Dynamical.ai Spatial Skill Router

DEPRECATION NOTICE:
===================
This module has been SUPERSEDED by unified_skill_orchestrator.py
The UnifiedSkillOrchestrator combines:
- MoE routing (what skill) - from moe_skill_router.py
- Spatial routing (which robot) - from this file
- Location adaptation (how to adapt) - from this file

Use unified_skill_orchestrator.py for new code:
    from src.core.unified_skill_orchestrator import get_orchestrator
    orchestrator = get_orchestrator()
    result = orchestrator.orchestrate(request)

This file is kept for backwards compatibility only.

Original Description:
=====================
This module handles location-aware skill routing and deployment:
1. Maps cameras to workspaces (physical zones)
2. Tracks which robots are visible in which cameras
3. Routes tasks to robots based on location and capability
4. Adapts skills to location-specific parameters
5. Deploys skills from cloud to edge devices

Architecture:
============

    ┌─────────────────────────────────────────────────────────────────┐
    │                         CLOUD                                    │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
    │  │  Task Planner   │  │  Skill Registry │  │  Location DB    │  │
    │  │  (LLM-based)    │  │  (all skills)   │  │  (site configs) │  │
    │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
    │           │                    │                    │           │
    │           ▼                    ▼                    ▼           │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │              SPATIAL SKILL ROUTER (this module)          │   │
    │  │                                                          │   │
    │  │  1. Camera → Workspace mapping                           │   │
    │  │  2. Robot → Workspace tracking (via perception)          │   │
    │  │  3. Task → Robot assignment (based on location)          │   │
    │  │  4. Skill → Location adaptation                          │   │
    │  │  5. Skill deployment to edge                             │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              │                                   │
    └──────────────────────────────┼───────────────────────────────────┘
                                   │ gRPC / REST
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      EDGE (Jetson Thor)                          │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
    │  │  ONVIF Cameras  │  │  Skill Cache    │  │  Robot Control  │  │
    │  │  (perception)   │  │  (loaded skills)│  │  (VLA + skills) │  │
    │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Key Concepts:
=============
- Workspace: A physical zone covered by one or more cameras
- Location Context: Parameters that vary by location (heights, obstacles, etc.)
- Skill Deployment: Sending skill weights from cloud to edge cache
- Location Adaptation: Modifying skill parameters for specific locations
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Workspace:
    """A physical workspace zone covered by cameras."""
    workspace_id: str
    name: str

    # Physical bounds (meters, relative to site origin)
    bounds_min: np.ndarray  # [x, y, z]
    bounds_max: np.ndarray  # [x, y, z]

    # Cameras covering this workspace
    camera_ids: List[str] = field(default_factory=list)

    # Location-specific parameters
    floor_height: float = 0.0
    ceiling_height: float = 3.0
    obstacles: List[Dict[str, Any]] = field(default_factory=list)

    # Workspace capabilities
    has_table: bool = False
    table_height: float = 0.75
    has_shelf: bool = False
    shelf_heights: List[float] = field(default_factory=list)

    # Safety zones within workspace
    safety_zones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RobotLocation:
    """Tracked location of a robot."""
    robot_id: str
    workspace_id: Optional[str]

    # Position in workspace (meters)
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [x, y, z, w]

    # Which cameras currently see this robot
    visible_in_cameras: List[str] = field(default_factory=list)

    # Confidence and freshness
    confidence: float = 1.0
    last_updated: float = 0.0

    # Robot capabilities at this location
    reachable_objects: List[str] = field(default_factory=list)


@dataclass
class LocationContext:
    """Context for adapting skills to a specific location."""
    workspace_id: str

    # Physical context
    floor_height: float
    table_height: Optional[float]
    shelf_heights: List[float]

    # Obstacle avoidance
    obstacles: List[Dict[str, Any]]
    safety_zones: List[Dict[str, Any]]

    # Lighting and visibility
    lighting_condition: str = "normal"  # normal, bright, dim

    # Other robots in workspace
    other_robots: List[str] = field(default_factory=list)

    # Humans in workspace (for HRI safety)
    humans_present: bool = False
    human_positions: List[np.ndarray] = field(default_factory=list)


@dataclass
class SkillDeployment:
    """Skill deployment request to edge device."""
    skill_id: str
    robot_id: str
    edge_device_ip: str

    # Skill weights (or reference to weights in cloud storage)
    weights_url: Optional[str] = None
    weights_bytes: Optional[bytes] = None

    # Location-adapted parameters
    location_params: Dict[str, Any] = field(default_factory=dict)

    # Deployment metadata
    priority: int = 0
    expires_at: Optional[float] = None
    prefetch: bool = False  # True if pre-loading, not immediate execution


class TaskAssignmentStrategy(Enum):
    """Strategy for assigning tasks to robots."""
    NEAREST = "nearest"           # Assign to nearest robot
    LEAST_BUSY = "least_busy"     # Assign to robot with fewest tasks
    CAPABILITY = "capability"      # Assign based on robot capabilities
    ROUND_ROBIN = "round_robin"   # Rotate through robots
    MANUAL = "manual"             # Explicit robot assignment


# =============================================================================
# Camera-Workspace Mapping
# =============================================================================

class CameraWorkspaceMapper:
    """Maps ONVIF cameras to physical workspaces."""

    def __init__(self):
        # Camera ID -> Workspace IDs
        self._camera_to_workspaces: Dict[str, List[str]] = {}

        # Workspace ID -> Workspace
        self._workspaces: Dict[str, Workspace] = {}

        # Camera calibration (camera ID -> extrinsics)
        self._camera_extrinsics: Dict[str, np.ndarray] = {}

        self._lock = threading.RLock()

    def register_workspace(self, workspace: Workspace):
        """Register a workspace."""
        with self._lock:
            self._workspaces[workspace.workspace_id] = workspace

            # Update camera mappings
            for camera_id in workspace.camera_ids:
                if camera_id not in self._camera_to_workspaces:
                    self._camera_to_workspaces[camera_id] = []
                if workspace.workspace_id not in self._camera_to_workspaces[camera_id]:
                    self._camera_to_workspaces[camera_id].append(workspace.workspace_id)

        logger.info(f"Registered workspace: {workspace.workspace_id} with cameras: {workspace.camera_ids}")

    def set_camera_extrinsics(self, camera_id: str, extrinsics: np.ndarray):
        """Set camera extrinsics (4x4 transformation matrix)."""
        with self._lock:
            self._camera_extrinsics[camera_id] = extrinsics

    def get_workspaces_for_camera(self, camera_id: str) -> List[Workspace]:
        """Get workspaces visible from a camera."""
        with self._lock:
            workspace_ids = self._camera_to_workspaces.get(camera_id, [])
            return [self._workspaces[wid] for wid in workspace_ids if wid in self._workspaces]

    def get_cameras_for_workspace(self, workspace_id: str) -> List[str]:
        """Get cameras covering a workspace."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            return workspace.camera_ids if workspace else []

    def point_to_workspace(self, point_3d: np.ndarray) -> Optional[str]:
        """Find which workspace contains a 3D point."""
        with self._lock:
            for workspace_id, workspace in self._workspaces.items():
                if (np.all(point_3d >= workspace.bounds_min) and
                    np.all(point_3d <= workspace.bounds_max)):
                    return workspace_id
            return None

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        with self._lock:
            return self._workspaces.get(workspace_id)


# =============================================================================
# Robot Location Tracker
# =============================================================================

class RobotLocationTracker:
    """Tracks robot locations based on camera perception."""

    def __init__(self, workspace_mapper: CameraWorkspaceMapper):
        self._workspace_mapper = workspace_mapper

        # Robot ID -> RobotLocation
        self._robot_locations: Dict[str, RobotLocation] = {}

        # Edge device IP -> Robot IDs
        self._edge_to_robots: Dict[str, List[str]] = {}

        self._lock = threading.RLock()

    def update_robot_location(
        self,
        robot_id: str,
        position: np.ndarray,
        orientation: np.ndarray,
        visible_in_cameras: List[str],
        confidence: float = 1.0
    ):
        """Update robot location from perception."""
        with self._lock:
            # Determine workspace from position
            workspace_id = self._workspace_mapper.point_to_workspace(position)

            # Find objects robot can reach
            reachable = self._find_reachable_objects(position, workspace_id)

            location = RobotLocation(
                robot_id=robot_id,
                workspace_id=workspace_id,
                position=position,
                orientation=orientation,
                visible_in_cameras=visible_in_cameras,
                confidence=confidence,
                last_updated=time.time(),
                reachable_objects=reachable
            )

            self._robot_locations[robot_id] = location

    def _find_reachable_objects(
        self,
        robot_position: np.ndarray,
        workspace_id: Optional[str]
    ) -> List[str]:
        """Find objects within robot's reach."""
        # This would integrate with object detection
        # For now, return empty list
        return []

    def get_robot_location(self, robot_id: str) -> Optional[RobotLocation]:
        """Get current robot location."""
        with self._lock:
            return self._robot_locations.get(robot_id)

    def get_robots_in_workspace(self, workspace_id: str) -> List[RobotLocation]:
        """Get all robots in a workspace."""
        with self._lock:
            return [
                loc for loc in self._robot_locations.values()
                if loc.workspace_id == workspace_id
            ]

    def get_robots_visible_in_camera(self, camera_id: str) -> List[RobotLocation]:
        """Get robots visible in a camera."""
        with self._lock:
            return [
                loc for loc in self._robot_locations.values()
                if camera_id in loc.visible_in_cameras
            ]

    def register_edge_device(self, edge_ip: str, robot_ids: List[str]):
        """Register which robots are controlled by which edge device."""
        with self._lock:
            self._edge_to_robots[edge_ip] = robot_ids

    def get_edge_device_for_robot(self, robot_id: str) -> Optional[str]:
        """Get edge device IP for a robot."""
        with self._lock:
            for edge_ip, robots in self._edge_to_robots.items():
                if robot_id in robots:
                    return edge_ip
            return None


# =============================================================================
# Task Router
# =============================================================================

class TaskRouter:
    """Routes tasks to robots based on location and capability."""

    def __init__(
        self,
        location_tracker: RobotLocationTracker,
        workspace_mapper: CameraWorkspaceMapper,
        strategy: TaskAssignmentStrategy = TaskAssignmentStrategy.NEAREST
    ):
        self._location_tracker = location_tracker
        self._workspace_mapper = workspace_mapper
        self._strategy = strategy

        # Robot ID -> current task count
        self._robot_task_counts: Dict[str, int] = {}

        # Robot ID -> capabilities
        self._robot_capabilities: Dict[str, Set[str]] = {}

        # Round-robin index
        self._rr_index = 0

        self._lock = threading.RLock()

    def register_robot_capabilities(self, robot_id: str, capabilities: Set[str]):
        """Register robot capabilities (e.g., 'manipulation', 'locomotion')."""
        with self._lock:
            self._robot_capabilities[robot_id] = capabilities

    def assign_task(
        self,
        task_description: str,
        target_workspace_id: Optional[str] = None,
        target_object_position: Optional[np.ndarray] = None,
        required_capabilities: Optional[Set[str]] = None,
        preferred_robot_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Assign a task to a robot.

        Args:
            task_description: Natural language task description
            target_workspace_id: Workspace where task should be performed
            target_object_position: Position of target object (if known)
            required_capabilities: Skills needed for this task
            preferred_robot_id: Explicitly requested robot

        Returns:
            robot_id of assigned robot, or None if no suitable robot
        """
        with self._lock:
            # If explicit robot requested, use it
            if preferred_robot_id:
                return preferred_robot_id

            # Find candidate robots
            candidates = self._find_candidate_robots(
                target_workspace_id,
                target_object_position,
                required_capabilities
            )

            if not candidates:
                logger.warning(f"No suitable robots for task: {task_description}")
                return None

            # Apply assignment strategy
            if self._strategy == TaskAssignmentStrategy.NEAREST:
                return self._assign_nearest(candidates, target_object_position)
            elif self._strategy == TaskAssignmentStrategy.LEAST_BUSY:
                return self._assign_least_busy(candidates)
            elif self._strategy == TaskAssignmentStrategy.CAPABILITY:
                return self._assign_by_capability(candidates, required_capabilities)
            elif self._strategy == TaskAssignmentStrategy.ROUND_ROBIN:
                return self._assign_round_robin(candidates)
            else:
                return candidates[0]

    def _find_candidate_robots(
        self,
        workspace_id: Optional[str],
        target_position: Optional[np.ndarray],
        required_capabilities: Optional[Set[str]]
    ) -> List[str]:
        """Find robots that can potentially do the task."""
        candidates = []

        for robot_id, location in self._location_tracker._robot_locations.items():
            # Check workspace if specified
            if workspace_id and location.workspace_id != workspace_id:
                # Robot not in target workspace - could still be candidate if nearby
                pass

            # Check capabilities
            if required_capabilities:
                robot_caps = self._robot_capabilities.get(robot_id, set())
                if not required_capabilities.issubset(robot_caps):
                    continue

            candidates.append(robot_id)

        return candidates

    def _assign_nearest(
        self,
        candidates: List[str],
        target_position: Optional[np.ndarray]
    ) -> str:
        """Assign to nearest robot."""
        if target_position is None or len(candidates) == 1:
            return candidates[0]

        min_dist = float('inf')
        nearest = candidates[0]

        for robot_id in candidates:
            location = self._location_tracker.get_robot_location(robot_id)
            if location:
                dist = np.linalg.norm(location.position - target_position)
                if dist < min_dist:
                    min_dist = dist
                    nearest = robot_id

        return nearest

    def _assign_least_busy(self, candidates: List[str]) -> str:
        """Assign to robot with fewest tasks."""
        min_tasks = float('inf')
        least_busy = candidates[0]

        for robot_id in candidates:
            task_count = self._robot_task_counts.get(robot_id, 0)
            if task_count < min_tasks:
                min_tasks = task_count
                least_busy = robot_id

        return least_busy

    def _assign_by_capability(
        self,
        candidates: List[str],
        required_capabilities: Optional[Set[str]]
    ) -> str:
        """Assign to robot with best matching capabilities."""
        if not required_capabilities:
            return candidates[0]

        best_match = candidates[0]
        best_score = 0

        for robot_id in candidates:
            caps = self._robot_capabilities.get(robot_id, set())
            score = len(caps.intersection(required_capabilities))
            if score > best_score:
                best_score = score
                best_match = robot_id

        return best_match

    def _assign_round_robin(self, candidates: List[str]) -> str:
        """Assign in round-robin fashion."""
        self._rr_index = (self._rr_index + 1) % len(candidates)
        return candidates[self._rr_index]

    def increment_task_count(self, robot_id: str):
        """Called when task is assigned to robot."""
        with self._lock:
            self._robot_task_counts[robot_id] = self._robot_task_counts.get(robot_id, 0) + 1

    def decrement_task_count(self, robot_id: str):
        """Called when robot completes a task."""
        with self._lock:
            count = self._robot_task_counts.get(robot_id, 0)
            self._robot_task_counts[robot_id] = max(0, count - 1)


# =============================================================================
# Location-Adaptive Skill Manager
# =============================================================================

class LocationAdaptiveSkillManager:
    """Adapts skills to location-specific parameters."""

    def __init__(self, workspace_mapper: CameraWorkspaceMapper):
        self._workspace_mapper = workspace_mapper

        # Skill ID -> base parameters
        self._skill_base_params: Dict[str, Dict[str, Any]] = {}

        # Workspace ID -> parameter overrides
        self._location_overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_skill(self, skill_id: str, base_params: Dict[str, Any]):
        """Register skill with base parameters."""
        self._skill_base_params[skill_id] = base_params

    def set_location_override(
        self,
        workspace_id: str,
        skill_id: str,
        param_overrides: Dict[str, Any]
    ):
        """Set location-specific parameter overrides."""
        if workspace_id not in self._location_overrides:
            self._location_overrides[workspace_id] = {}
        self._location_overrides[workspace_id][skill_id] = param_overrides

    def get_adapted_params(
        self,
        skill_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Get skill parameters adapted for a specific location."""
        # Start with base params
        params = self._skill_base_params.get(skill_id, {}).copy()

        # Get workspace
        workspace = self._workspace_mapper.get_workspace(workspace_id)

        if workspace:
            # Apply automatic adaptations based on workspace
            params.update(self._auto_adapt(skill_id, workspace))

            # Apply explicit overrides
            overrides = self._location_overrides.get(workspace_id, {}).get(skill_id, {})
            params.update(overrides)

        return params

    def _auto_adapt(self, skill_id: str, workspace: Workspace) -> Dict[str, Any]:
        """Automatically adapt parameters based on workspace properties."""
        adapted = {}

        # Adapt picking height based on table
        if workspace.has_table:
            adapted["default_pick_height"] = workspace.table_height

        # Adapt for shelf locations
        if workspace.has_shelf and workspace.shelf_heights:
            adapted["shelf_heights"] = workspace.shelf_heights

        # Adapt for ceiling (e.g., for tall manipulations)
        adapted["max_reach_height"] = min(
            adapted.get("max_reach_height", 2.0),
            workspace.ceiling_height - 0.2
        )

        # Adapt for floor
        adapted["floor_height"] = workspace.floor_height

        return adapted

    def create_location_context(
        self,
        workspace_id: str,
        robot_tracker: RobotLocationTracker
    ) -> LocationContext:
        """Create full location context for skill execution."""
        workspace = self._workspace_mapper.get_workspace(workspace_id)

        if not workspace:
            # Return default context
            return LocationContext(
                workspace_id=workspace_id,
                floor_height=0.0,
                table_height=None,
                shelf_heights=[],
                obstacles=[],
                safety_zones=[]
            )

        # Find other robots in workspace
        other_robots = [
            loc.robot_id
            for loc in robot_tracker.get_robots_in_workspace(workspace_id)
        ]

        return LocationContext(
            workspace_id=workspace_id,
            floor_height=workspace.floor_height,
            table_height=workspace.table_height if workspace.has_table else None,
            shelf_heights=workspace.shelf_heights.copy(),
            obstacles=workspace.obstacles.copy(),
            safety_zones=workspace.safety_zones.copy(),
            other_robots=other_robots
        )


# =============================================================================
# Skill Deployment Manager
# =============================================================================

class SkillDeploymentManager:
    """Deploys skills from cloud to edge devices."""

    def __init__(
        self,
        location_tracker: RobotLocationTracker,
        skill_manager: LocationAdaptiveSkillManager
    ):
        self._location_tracker = location_tracker
        self._skill_manager = skill_manager

        # Edge IP -> cached skill IDs
        self._edge_skill_cache: Dict[str, Set[str]] = {}

        # Pending deployments
        self._pending_deployments: List[SkillDeployment] = []

        self._lock = threading.RLock()

    def deploy_skill(
        self,
        skill_id: str,
        robot_id: str,
        weights_url: Optional[str] = None,
        prefetch: bool = False,
        priority: int = 0
    ) -> Optional[SkillDeployment]:
        """
        Deploy a skill to a robot's edge device.

        Args:
            skill_id: ID of skill to deploy
            robot_id: Target robot
            weights_url: URL to download skill weights (if not cached)
            prefetch: If True, just cache the skill, don't execute
            priority: Higher priority = faster deployment

        Returns:
            SkillDeployment object, or None if deployment failed
        """
        # Get edge device for robot
        edge_ip = self._location_tracker.get_edge_device_for_robot(robot_id)
        if not edge_ip:
            logger.error(f"No edge device found for robot: {robot_id}")
            return None

        # Get robot location for skill adaptation
        location = self._location_tracker.get_robot_location(robot_id)
        workspace_id = location.workspace_id if location else None

        # Get location-adapted parameters
        location_params = {}
        if workspace_id:
            location_params = self._skill_manager.get_adapted_params(skill_id, workspace_id)

        # Check if skill already cached on edge
        with self._lock:
            cached = self._edge_skill_cache.get(edge_ip, set())
            already_cached = skill_id in cached

        # Create deployment
        deployment = SkillDeployment(
            skill_id=skill_id,
            robot_id=robot_id,
            edge_device_ip=edge_ip,
            weights_url=weights_url if not already_cached else None,
            location_params=location_params,
            priority=priority,
            prefetch=prefetch
        )

        # Send to edge device
        success = self._send_to_edge(deployment)

        if success:
            with self._lock:
                if edge_ip not in self._edge_skill_cache:
                    self._edge_skill_cache[edge_ip] = set()
                self._edge_skill_cache[edge_ip].add(skill_id)

            logger.info(f"Deployed skill {skill_id} to {robot_id} at {edge_ip}")
            return deployment
        else:
            logger.error(f"Failed to deploy skill {skill_id} to {edge_ip}")
            return None

    def _send_to_edge(self, deployment: SkillDeployment) -> bool:
        """Send deployment request to edge device."""
        # This would use gRPC or REST to send to edge
        # For now, return True (placeholder)

        # In real implementation:
        # 1. If weights_url provided, edge downloads weights
        # 2. Edge loads skill into its cache
        # 3. Edge applies location_params to skill
        # 4. Edge confirms deployment

        logger.debug(f"Sending deployment to {deployment.edge_device_ip}: {deployment.skill_id}")
        return True

    def prefetch_skills(
        self,
        robot_id: str,
        predicted_skills: List[str]
    ):
        """Pre-load skills likely to be needed soon."""
        for skill_id in predicted_skills:
            self.deploy_skill(
                skill_id=skill_id,
                robot_id=robot_id,
                prefetch=True,
                priority=-1  # Low priority
            )

    def get_cached_skills(self, edge_ip: str) -> Set[str]:
        """Get skills cached on an edge device."""
        with self._lock:
            return self._edge_skill_cache.get(edge_ip, set()).copy()

    def invalidate_cache(self, edge_ip: str, skill_id: Optional[str] = None):
        """Invalidate skill cache on edge device."""
        with self._lock:
            if skill_id:
                self._edge_skill_cache.get(edge_ip, set()).discard(skill_id)
            else:
                self._edge_skill_cache[edge_ip] = set()


# =============================================================================
# Unified Spatial Skill Router
# =============================================================================

class SpatialSkillRouter:
    """
    Main interface for location-aware skill routing.

    Combines all components:
    - Camera-workspace mapping
    - Robot location tracking
    - Task routing
    - Skill adaptation
    - Skill deployment
    """

    def __init__(self):
        self.workspace_mapper = CameraWorkspaceMapper()
        self.location_tracker = RobotLocationTracker(self.workspace_mapper)
        self.task_router = TaskRouter(
            self.location_tracker,
            self.workspace_mapper
        )
        self.skill_manager = LocationAdaptiveSkillManager(self.workspace_mapper)
        self.deployment_manager = SkillDeploymentManager(
            self.location_tracker,
            self.skill_manager
        )

    def configure_site(self, site_config: Dict[str, Any]):
        """
        Configure site with workspaces, cameras, and robots.

        Args:
            site_config: Site configuration including:
                - workspaces: List of workspace definitions
                - cameras: Camera configurations
                - robots: Robot configurations
        """
        # Register workspaces
        for ws_config in site_config.get("workspaces", []):
            workspace = Workspace(
                workspace_id=ws_config["id"],
                name=ws_config["name"],
                bounds_min=np.array(ws_config["bounds_min"]),
                bounds_max=np.array(ws_config["bounds_max"]),
                camera_ids=ws_config.get("cameras", []),
                floor_height=ws_config.get("floor_height", 0.0),
                ceiling_height=ws_config.get("ceiling_height", 3.0),
                has_table=ws_config.get("has_table", False),
                table_height=ws_config.get("table_height", 0.75),
                has_shelf=ws_config.get("has_shelf", False),
                shelf_heights=ws_config.get("shelf_heights", [])
            )
            self.workspace_mapper.register_workspace(workspace)

        # Register robot capabilities and edge devices
        for robot_config in site_config.get("robots", []):
            robot_id = robot_config["id"]
            capabilities = set(robot_config.get("capabilities", []))
            edge_ip = robot_config.get("edge_ip")

            self.task_router.register_robot_capabilities(robot_id, capabilities)

            if edge_ip:
                self.location_tracker.register_edge_device(edge_ip, [robot_id])

        logger.info(f"Configured site with {len(site_config.get('workspaces', []))} workspaces")

    def route_task(
        self,
        task_description: str,
        skill_ids: List[str],
        target_location: Optional[Dict[str, Any]] = None,
        preferred_robot: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Route a task to a robot and deploy necessary skills.

        Args:
            task_description: Natural language task
            skill_ids: Skills needed for the task
            target_location: Optional target location info
            preferred_robot: Optional preferred robot ID

        Returns:
            Routing result including robot_id, workspace_id, deployments
        """
        # Determine target workspace and position
        target_workspace_id = None
        target_position = None

        if target_location:
            if "workspace_id" in target_location:
                target_workspace_id = target_location["workspace_id"]
            if "position" in target_location:
                target_position = np.array(target_location["position"])
                if not target_workspace_id:
                    target_workspace_id = self.workspace_mapper.point_to_workspace(target_position)

        # Assign robot
        robot_id = self.task_router.assign_task(
            task_description=task_description,
            target_workspace_id=target_workspace_id,
            target_object_position=target_position,
            preferred_robot_id=preferred_robot
        )

        if not robot_id:
            return None

        # Deploy skills
        deployments = []
        for skill_id in skill_ids:
            deployment = self.deployment_manager.deploy_skill(
                skill_id=skill_id,
                robot_id=robot_id
            )
            if deployment:
                deployments.append(deployment)

        # Get location context
        robot_location = self.location_tracker.get_robot_location(robot_id)
        workspace_id = robot_location.workspace_id if robot_location else target_workspace_id

        location_context = None
        if workspace_id:
            location_context = self.skill_manager.create_location_context(
                workspace_id,
                self.location_tracker
            )

        # Increment task count
        self.task_router.increment_task_count(robot_id)

        return {
            "robot_id": robot_id,
            "workspace_id": workspace_id,
            "deployments": deployments,
            "location_context": location_context,
            "task_description": task_description,
            "skill_ids": skill_ids
        }

    def update_robot_perception(
        self,
        camera_id: str,
        robot_detections: List[Dict[str, Any]]
    ):
        """
        Update robot locations from camera perception.

        Called by perception pipeline when robots are detected.
        """
        for detection in robot_detections:
            robot_id = detection.get("robot_id")
            if not robot_id:
                continue

            position = np.array(detection.get("position", [0, 0, 0]))
            orientation = np.array(detection.get("orientation", [0, 0, 0, 1]))
            confidence = detection.get("confidence", 1.0)

            # Get all cameras that can see this position
            visible_cameras = [camera_id]
            workspace_id = self.workspace_mapper.point_to_workspace(position)
            if workspace_id:
                all_cameras = self.workspace_mapper.get_cameras_for_workspace(workspace_id)
                # In real implementation, check if robot is actually visible from each camera
                visible_cameras = all_cameras

            self.location_tracker.update_robot_location(
                robot_id=robot_id,
                position=position,
                orientation=orientation,
                visible_in_cameras=visible_cameras,
                confidence=confidence
            )

    def complete_task(self, robot_id: str):
        """Mark a task as complete for a robot."""
        self.task_router.decrement_task_count(robot_id)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create router
    router = SpatialSkillRouter()

    # Configure site
    site_config = {
        "workspaces": [
            {
                "id": "assembly_station_1",
                "name": "Assembly Station 1",
                "bounds_min": [0, 0, 0],
                "bounds_max": [2, 2, 2],
                "cameras": ["cam_01", "cam_02"],
                "has_table": True,
                "table_height": 0.8
            },
            {
                "id": "storage_area",
                "name": "Storage Area",
                "bounds_min": [3, 0, 0],
                "bounds_max": [5, 2, 2],
                "cameras": ["cam_03", "cam_04"],
                "has_shelf": True,
                "shelf_heights": [0.5, 1.0, 1.5]
            }
        ],
        "robots": [
            {
                "id": "thor_001",
                "capabilities": ["manipulation", "locomotion"],
                "edge_ip": "192.168.1.10"
            },
            {
                "id": "thor_002",
                "capabilities": ["manipulation"],
                "edge_ip": "192.168.1.11"
            }
        ]
    }

    router.configure_site(site_config)

    # Simulate robot detection
    router.update_robot_perception(
        camera_id="cam_01",
        robot_detections=[
            {
                "robot_id": "thor_001",
                "position": [1.0, 1.0, 0.0],
                "orientation": [0, 0, 0, 1],
                "confidence": 0.95
            }
        ]
    )

    # Route a task
    result = router.route_task(
        task_description="Pick up the red cube from the table",
        skill_ids=["grasp_v2", "pick_place_v1"],
        target_location={
            "workspace_id": "assembly_station_1",
            "position": [1.2, 0.8, 0.8]
        }
    )

    if result:
        print(f"Task routed to: {result['robot_id']}")
        print(f"Workspace: {result['workspace_id']}")
        print(f"Skills deployed: {result['skill_ids']}")
    else:
        print("Failed to route task")

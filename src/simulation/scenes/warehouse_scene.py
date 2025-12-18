"""
Warehouse Scene for Isaac Lab Simulation

Defines a warehouse environment with shelves, objects, and robots.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ObjectConfig:
    """Configuration for scene objects."""
    object_id: str
    object_type: str  # cube, cylinder, sphere, box
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.1


@dataclass
class ShelfConfig:
    """Configuration for warehouse shelf."""
    shelf_id: str
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float] = (2.0, 0.5, 2.0)  # width, depth, height
    num_levels: int = 4
    level_height: float = 0.5


@dataclass
class WarehouseConfig:
    """Configuration for warehouse scene."""
    # Scene dimensions
    floor_size: Tuple[float, float] = (20.0, 20.0)
    wall_height: float = 5.0

    # Lighting
    ambient_intensity: float = 0.3
    directional_intensity: float = 0.7

    # Shelves
    shelf_rows: int = 3
    shelf_cols: int = 4
    aisle_width: float = 3.0

    # Objects
    objects_per_shelf: int = 5
    object_types: List[str] = field(default_factory=lambda: ["cube", "cylinder", "box"])

    # Robot spawn
    robot_spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class WarehouseScene:
    """
    Warehouse scene for Isaac Lab simulation.

    Features:
    - Configurable shelf layout
    - Random object placement
    - Multiple robot spawn points
    - Collision geometry
    """

    def __init__(self, config: Optional[WarehouseConfig] = None):
        """Initialize warehouse scene."""
        self.config = config or WarehouseConfig()

        # Scene objects
        self._shelves: Dict[str, ShelfConfig] = {}
        self._objects: Dict[str, ObjectConfig] = {}
        self._spawn_points: List[Tuple[float, float, float]] = []

        # Generate scene
        self._generate_scene()

        logger.info(f"WarehouseScene created with {len(self._shelves)} shelves, {len(self._objects)} objects")

    def _generate_scene(self) -> None:
        """Generate warehouse scene layout."""
        # Generate shelf grid
        start_x = -self.config.floor_size[0] / 2 + 2.0
        start_y = -self.config.floor_size[1] / 2 + 2.0

        shelf_spacing_x = (self.config.floor_size[0] - 4.0) / max(1, self.config.shelf_cols - 1)
        shelf_spacing_y = self.config.aisle_width + 0.5

        for row in range(self.config.shelf_rows):
            for col in range(self.config.shelf_cols):
                shelf_id = f"shelf_{row}_{col}"
                x = start_x + col * shelf_spacing_x
                y = start_y + row * shelf_spacing_y * 2

                shelf = ShelfConfig(
                    shelf_id=shelf_id,
                    position=(x, y, 0.0),
                    dimensions=(1.5, 0.4, 1.8),
                    num_levels=4,
                    level_height=0.45,
                )
                self._shelves[shelf_id] = shelf

                # Add objects on shelf
                self._add_shelf_objects(shelf)

        # Generate spawn points (in aisles)
        for row in range(self.config.shelf_rows + 1):
            for i in range(3):
                x = start_x + i * shelf_spacing_x
                y = start_y + row * shelf_spacing_y * 2 - self.config.aisle_width / 2
                self._spawn_points.append((x, y, 0.0))

    def _add_shelf_objects(self, shelf: ShelfConfig) -> None:
        """Add objects to shelf levels."""
        np.random.seed(hash(shelf.shelf_id) % 2**32)

        for level in range(shelf.num_levels):
            num_objects = np.random.randint(1, self.config.objects_per_shelf + 1)

            for i in range(num_objects):
                obj_type = np.random.choice(self.config.object_types)
                obj_id = f"{shelf.shelf_id}_obj_{level}_{i}"

                # Random position on shelf level
                shelf_x, shelf_y, shelf_z = shelf.position
                level_z = shelf_z + level * shelf.level_height + 0.05

                obj_x = shelf_x + np.random.uniform(-shelf.dimensions[0]/3, shelf.dimensions[0]/3)
                obj_y = shelf_y

                # Random size and color
                scale = np.random.uniform(0.03, 0.08)
                color = tuple(np.random.uniform(0.2, 1.0, 3))

                obj_config = ObjectConfig(
                    object_id=obj_id,
                    object_type=obj_type,
                    position=(obj_x, obj_y, level_z),
                    scale=(scale, scale, scale),
                    color=color,
                    mass=np.random.uniform(0.1, 2.0),
                )
                self._objects[obj_id] = obj_config

    def get_shelves(self) -> Dict[str, ShelfConfig]:
        """Get all shelves."""
        return self._shelves

    def get_objects(self) -> Dict[str, ObjectConfig]:
        """Get all objects."""
        return self._objects

    def get_spawn_points(self) -> List[Tuple[float, float, float]]:
        """Get robot spawn points."""
        return self._spawn_points

    def get_random_spawn_point(self) -> Tuple[float, float, float]:
        """Get random spawn point."""
        return self._spawn_points[np.random.randint(len(self._spawn_points))]

    def get_random_object(self) -> Optional[ObjectConfig]:
        """Get random object from scene."""
        if not self._objects:
            return None
        obj_id = np.random.choice(list(self._objects.keys()))
        return self._objects[obj_id]

    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scene bounding box."""
        return (
            np.array([-self.config.floor_size[0]/2, -self.config.floor_size[1]/2, 0.0]),
            np.array([self.config.floor_size[0]/2, self.config.floor_size[1]/2, self.config.wall_height]),
        )

    def to_usd_commands(self) -> List[Dict[str, Any]]:
        """
        Generate USD commands for Isaac Lab.

        Returns list of commands to create scene in Isaac Sim.
        """
        commands = []

        # Ground plane
        commands.append({
            "type": "create_ground_plane",
            "size": self.config.floor_size,
            "color": (0.3, 0.3, 0.35),
        })

        # Walls
        for i, (pos, size) in enumerate([
            ((-self.config.floor_size[0]/2, 0, self.config.wall_height/2), (0.1, self.config.floor_size[1], self.config.wall_height)),
            ((self.config.floor_size[0]/2, 0, self.config.wall_height/2), (0.1, self.config.floor_size[1], self.config.wall_height)),
            ((0, -self.config.floor_size[1]/2, self.config.wall_height/2), (self.config.floor_size[0], 0.1, self.config.wall_height)),
            ((0, self.config.floor_size[1]/2, self.config.wall_height/2), (self.config.floor_size[0], 0.1, self.config.wall_height)),
        ]):
            commands.append({
                "type": "create_cube",
                "prim_path": f"/World/Walls/wall_{i}",
                "position": pos,
                "scale": size,
                "color": (0.7, 0.7, 0.7),
                "is_static": True,
            })

        # Shelves
        for shelf_id, shelf in self._shelves.items():
            commands.append({
                "type": "create_shelf",
                "prim_path": f"/World/Shelves/{shelf_id}",
                "position": shelf.position,
                "dimensions": shelf.dimensions,
                "num_levels": shelf.num_levels,
                "color": (0.6, 0.4, 0.2),
            })

        # Objects
        for obj_id, obj in self._objects.items():
            commands.append({
                "type": f"create_{obj.object_type}",
                "prim_path": f"/World/Objects/{obj_id}",
                "position": obj.position,
                "orientation": obj.orientation,
                "scale": obj.scale,
                "color": obj.color,
                "mass": obj.mass,
                "friction": obj.friction,
            })

        # Lighting
        commands.append({
            "type": "create_dome_light",
            "prim_path": "/World/Lights/dome",
            "intensity": self.config.ambient_intensity * 1000,
        })
        commands.append({
            "type": "create_distant_light",
            "prim_path": "/World/Lights/sun",
            "intensity": self.config.directional_intensity * 1000,
            "direction": (-0.5, -0.5, -1.0),
        })

        return commands

    def get_scene_info(self) -> Dict[str, Any]:
        """Get scene information for dashboard."""
        return {
            "type": "warehouse",
            "floor_size": self.config.floor_size,
            "num_shelves": len(self._shelves),
            "num_objects": len(self._objects),
            "num_spawn_points": len(self._spawn_points),
            "shelves": [
                {
                    "id": s.shelf_id,
                    "position": s.position,
                    "dimensions": s.dimensions,
                }
                for s in self._shelves.values()
            ],
        }

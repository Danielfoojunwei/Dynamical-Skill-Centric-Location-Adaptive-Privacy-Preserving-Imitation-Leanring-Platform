"""
Tabletop Scene for Isaac Lab Simulation

Defines a tabletop manipulation environment with objects.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TabletopConfig:
    """Configuration for tabletop scene."""
    # Table dimensions
    table_position: Tuple[float, float, float] = (0.5, 0.0, 0.0)
    table_size: Tuple[float, float, float] = (0.8, 1.2, 0.05)
    table_height: float = 0.75

    # Objects
    num_objects: int = 5
    object_types: List[str] = field(default_factory=lambda: ["cube", "cylinder", "sphere"])
    object_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
    ])

    # Placement zone (relative to table center)
    placement_zone_min: Tuple[float, float] = (-0.3, -0.4)
    placement_zone_max: Tuple[float, float] = (0.3, 0.4)

    # Robot configuration
    robot_base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Bins/targets
    enable_bins: bool = True
    num_bins: int = 3


class TabletopScene:
    """
    Tabletop manipulation scene for Isaac Lab.

    Features:
    - Configurable table setup
    - Random object placement
    - Target bins for sorting tasks
    - Collision-free spawning
    """

    def __init__(self, config: Optional[TabletopConfig] = None):
        """Initialize tabletop scene."""
        self.config = config or TabletopConfig()

        # Scene elements
        self._table_surface_z = self.config.table_height + self.config.table_size[2] / 2
        self._objects: Dict[str, Dict[str, Any]] = {}
        self._bins: Dict[str, Dict[str, Any]] = {}
        self._target_positions: List[Tuple[float, float, float]] = []

        # Generate scene
        self._generate_scene()

        logger.info(f"TabletopScene created with {len(self._objects)} objects")

    def _generate_scene(self) -> None:
        """Generate tabletop scene."""
        # Generate objects with collision-free placement
        self._generate_objects()

        # Generate bins
        if self.config.enable_bins:
            self._generate_bins()

    def _generate_objects(self) -> None:
        """Generate manipulation objects on table."""
        placed_positions = []
        min_distance = 0.08  # Minimum distance between objects

        table_x, table_y, _ = self.config.table_position
        zone_min = self.config.placement_zone_min
        zone_max = self.config.placement_zone_max

        for i in range(self.config.num_objects):
            # Try to find collision-free position
            max_attempts = 50
            for attempt in range(max_attempts):
                x = table_x + np.random.uniform(zone_min[0], zone_max[0])
                y = table_y + np.random.uniform(zone_min[1], zone_max[1])

                # Check distance to other objects
                valid = True
                for px, py in placed_positions:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                        valid = False
                        break

                if valid:
                    placed_positions.append((x, y))
                    break

            if len(placed_positions) <= i:
                # Couldn't place object, skip
                continue

            # Create object
            obj_type = self.config.object_types[i % len(self.config.object_types)]
            obj_color = self.config.object_colors[i % len(self.config.object_colors)]
            obj_id = f"obj_{i}"

            # Random size
            size = np.random.uniform(0.03, 0.06)

            self._objects[obj_id] = {
                "id": obj_id,
                "type": obj_type,
                "position": np.array([x, y, self._table_surface_z + size / 2]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                "size": size,
                "color": obj_color,
                "mass": np.random.uniform(0.05, 0.5),
                "grasped": False,
            }

    def _generate_bins(self) -> None:
        """Generate target bins for sorting."""
        table_x, table_y, _ = self.config.table_position

        # Place bins at the back of the table
        bin_y = table_y + self.config.table_size[1] / 2 - 0.1
        bin_spacing = self.config.table_size[0] / (self.config.num_bins + 1)

        for i in range(self.config.num_bins):
            bin_x = table_x - self.config.table_size[0] / 2 + (i + 1) * bin_spacing
            bin_id = f"bin_{i}"

            self._bins[bin_id] = {
                "id": bin_id,
                "position": np.array([bin_x, bin_y, self._table_surface_z + 0.025]),
                "size": (0.1, 0.1, 0.05),
                "color": self.config.object_colors[i % len(self.config.object_colors)],
            }

            # Add as target position
            self._target_positions.append((bin_x, bin_y, self._table_surface_z + 0.05))

    def get_objects(self) -> Dict[str, Dict[str, Any]]:
        """Get all objects."""
        return self._objects

    def get_bins(self) -> Dict[str, Dict[str, Any]]:
        """Get all bins."""
        return self._bins

    def get_target_positions(self) -> List[Tuple[float, float, float]]:
        """Get target positions for placing objects."""
        return self._target_positions

    def get_random_object(self) -> Optional[Dict[str, Any]]:
        """Get random ungrasped object."""
        ungrasped = [obj for obj in self._objects.values() if not obj["grasped"]]
        if not ungrasped:
            return None
        return np.random.choice(ungrasped)

    def get_table_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get table surface bounds."""
        tx, ty, _ = self.config.table_position
        tw, th = self.config.table_size[0], self.config.table_size[1]

        return (
            np.array([tx - tw/2, ty - th/2, self._table_surface_z]),
            np.array([tx + tw/2, ty + th/2, self._table_surface_z + 0.5]),
        )

    def reset_objects(self) -> None:
        """Reset all objects to initial positions."""
        self._objects.clear()
        self._generate_objects()

    def randomize_objects(self, seed: Optional[int] = None) -> None:
        """Randomize object positions."""
        if seed is not None:
            np.random.seed(seed)

        self._objects.clear()
        self._generate_objects()

    def update_object(
        self,
        obj_id: str,
        position: Optional[np.ndarray] = None,
        grasped: Optional[bool] = None,
    ) -> None:
        """Update object state."""
        if obj_id not in self._objects:
            return

        if position is not None:
            self._objects[obj_id]["position"] = position
        if grasped is not None:
            self._objects[obj_id]["grasped"] = grasped

    def check_object_in_bin(self, obj_id: str) -> Optional[str]:
        """Check if object is in any bin."""
        if obj_id not in self._objects:
            return None

        obj = self._objects[obj_id]
        obj_pos = obj["position"]

        for bin_id, bin_data in self._bins.items():
            bin_pos = bin_data["position"]
            bin_size = bin_data["size"]

            # Check if object is within bin bounds
            dx = abs(obj_pos[0] - bin_pos[0])
            dy = abs(obj_pos[1] - bin_pos[1])

            if dx < bin_size[0] / 2 and dy < bin_size[1] / 2:
                return bin_id

        return None

    def get_scene_state(self) -> Dict[str, Any]:
        """Get complete scene state for telemetry."""
        return {
            "table": {
                "position": self.config.table_position,
                "size": self.config.table_size,
                "height": self.config.table_height,
            },
            "objects": {
                obj_id: {
                    "position": obj["position"].tolist(),
                    "type": obj["type"],
                    "color": obj["color"],
                    "size": obj["size"],
                    "grasped": obj["grasped"],
                }
                for obj_id, obj in self._objects.items()
            },
            "bins": {
                bin_id: {
                    "position": bin_data["position"].tolist(),
                    "size": bin_data["size"],
                    "color": bin_data["color"],
                }
                for bin_id, bin_data in self._bins.items()
            },
            "surface_z": self._table_surface_z,
        }

    def to_usd_commands(self) -> List[Dict[str, Any]]:
        """Generate USD commands for Isaac Lab."""
        commands = []

        # Ground plane
        commands.append({
            "type": "create_ground_plane",
            "size": (5.0, 5.0),
            "color": (0.3, 0.3, 0.35),
        })

        # Table
        tx, ty, tz = self.config.table_position
        commands.append({
            "type": "create_cube",
            "prim_path": "/World/Table/surface",
            "position": (tx, ty, self.config.table_height),
            "scale": self.config.table_size,
            "color": (0.6, 0.4, 0.2),
            "is_static": True,
        })

        # Table legs
        leg_positions = [
            (tx - self.config.table_size[0]/2 + 0.05, ty - self.config.table_size[1]/2 + 0.05),
            (tx + self.config.table_size[0]/2 - 0.05, ty - self.config.table_size[1]/2 + 0.05),
            (tx - self.config.table_size[0]/2 + 0.05, ty + self.config.table_size[1]/2 - 0.05),
            (tx + self.config.table_size[0]/2 - 0.05, ty + self.config.table_size[1]/2 - 0.05),
        ]
        for i, (lx, ly) in enumerate(leg_positions):
            commands.append({
                "type": "create_cylinder",
                "prim_path": f"/World/Table/leg_{i}",
                "position": (lx, ly, self.config.table_height / 2),
                "radius": 0.03,
                "height": self.config.table_height,
                "color": (0.5, 0.35, 0.15),
                "is_static": True,
            })

        # Objects
        for obj_id, obj in self._objects.items():
            commands.append({
                "type": f"create_{obj['type']}",
                "prim_path": f"/World/Objects/{obj_id}",
                "position": tuple(obj["position"]),
                "scale": (obj["size"], obj["size"], obj["size"]),
                "color": obj["color"],
                "mass": obj["mass"],
            })

        # Bins
        for bin_id, bin_data in self._bins.items():
            commands.append({
                "type": "create_bin",
                "prim_path": f"/World/Bins/{bin_id}",
                "position": tuple(bin_data["position"]),
                "size": bin_data["size"],
                "color": bin_data["color"],
                "is_static": True,
            })

        # Lighting
        commands.append({
            "type": "create_dome_light",
            "prim_path": "/World/Lights/dome",
            "intensity": 500,
        })

        return commands

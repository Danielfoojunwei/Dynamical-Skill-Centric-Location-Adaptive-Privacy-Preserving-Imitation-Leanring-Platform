"""
Environment Hazard Model & Registry

Defines the central representation for environmental hazards (both built-in and custom)
and a registry for managing them.
"""

import json
import os
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

# =============================================================================
# Data Structures
# =============================================================================

class HazardCategory(str, Enum):
    BUILTIN = "builtin"
    CUSTOM = "custom"

class HazardType(str, Enum):
    """Built-in hazard types."""
    OVERHANG = "OVERHANG"
    SLIPPERY_FLOOR = "SLIPPERY_FLOOR"
    STAIRS = "STAIRS"
    UNEVEN_GROUND = "UNEVEN_GROUND"
    PERSON = "PERSON"
    FORKLIFT = "FORKLIFT"
    UNKNOWN = "UNKNOWN"

@dataclass
class HazardTypeDefinition:
    """Definition of a hazard type (metadata)."""
    type_key: str
    category: HazardCategory
    display_name: str
    description: str
    default_severity: float  # 0.0 to 1.0
    default_behaviour: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"action": "STOP", "speed_factor": 0.0, "clearance_m": 2.0}
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HazardTypeDefinition':
        # Handle Enum conversion
        if "category" in data:
            data["category"] = HazardCategory(data["category"])
        return cls(**data)

@dataclass
class EnvironmentHazard:
    """An instance of a detected hazard in the environment."""
    id: str
    type_key: str
    category: HazardCategory
    severity: float
    confidence: float
    position: Optional[Dict[str, Any]] = None  # e.g. {"x": 1.0, "y": 2.0, "z": 0.0} or polygon
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None

# =============================================================================
# Registry
# =============================================================================

class HazardRegistry:
    """
    Central registry for hazard type definitions.
    Manages built-in types and persists custom user-defined types.
    """
    
    CONFIG_DIR = Path("config")
    CUSTOM_HAZARDS_FILE = CONFIG_DIR / "custom_hazards.json"
    
    def __init__(self):
        self._definitions: Dict[str, HazardTypeDefinition] = {}
        self._load_builtin_definitions()
        self._load_custom_definitions()
        
    def _load_builtin_definitions(self):
        """Register built-in hardcoded hazards."""
        builtins = [
            HazardTypeDefinition(
                type_key=HazardType.OVERHANG.value,
                category=HazardCategory.BUILTIN,
                display_name="Overhang",
                description="Low clearance obstacle at head height.",
                default_severity=0.8,
                default_behaviour={"action": "DUCK", "clearance_m": 0.5}
            ),
            HazardTypeDefinition(
                type_key=HazardType.SLIPPERY_FLOOR.value,
                category=HazardCategory.BUILTIN,
                display_name="Slippery Floor",
                description="Wet or slippery surface detected.",
                default_severity=0.6,
                default_behaviour={"action": "CRAWL", "speed_factor": 0.3}
            ),
            HazardTypeDefinition(
                type_key=HazardType.UNEVEN_GROUND.value,
                category=HazardCategory.BUILTIN,
                display_name="Uneven Ground",
                description="Rough terrain or debris.",
                default_severity=0.5,
                default_behaviour={"action": "SLOW", "speed_factor": 0.5}
            ),
            HazardTypeDefinition(
                type_key=HazardType.PERSON.value,
                category=HazardCategory.BUILTIN,
                display_name="Person",
                description="Human detected in workspace.",
                default_severity=1.0,
                default_behaviour={"action": "STOP", "clearance_m": 1.5}
            ),
            HazardTypeDefinition(
                type_key=HazardType.FORKLIFT.value,
                category=HazardCategory.BUILTIN,
                display_name="Forklift",
                description="Moving machinery detected.",
                default_severity=1.0,
                default_behaviour={"action": "STOP", "clearance_m": 2.0}
            ),
        ]
        
        for h in builtins:
            self._definitions[h.type_key] = h
            
    def _load_custom_definitions(self):
        """Load custom hazards from disk."""
        if not self.CUSTOM_HAZARDS_FILE.exists():
            return
            
        try:
            with open(self.CUSTOM_HAZARDS_FILE, 'r') as f:
                data = json.load(f)
                
            for item in data:
                try:
                    definition = HazardTypeDefinition.from_dict(item)
                    # Ensure it's marked as custom
                    definition.category = HazardCategory.CUSTOM
                    self._definitions[definition.type_key] = definition
                except Exception as e:
                    logger.error(f"Failed to load custom hazard: {e}")
                    
            logger.info(f"Loaded {len(data)} custom hazards from {self.CUSTOM_HAZARDS_FILE}")
            
        except Exception as e:
            logger.error(f"Error loading custom hazards file: {e}")

    def _save_custom_definitions(self):
        """Persist custom hazards to disk."""
        customs = [
            d.to_dict() 
            for d in self._definitions.values() 
            if d.category == HazardCategory.CUSTOM
        ]
        
        # Ensure directory exists
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.CUSTOM_HAZARDS_FILE, 'w') as f:
                json.dump(customs, f, indent=4)
            logger.info(f"Saved {len(customs)} custom hazards to {self.CUSTOM_HAZARDS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save custom hazards: {e}")

    def register_custom(self, definition: HazardTypeDefinition) -> None:
        """Register a new custom hazard type and persist it."""
        if definition.category != HazardCategory.CUSTOM:
            logger.warning(f"Forcing category to CUSTOM for {definition.type_key}")
            definition.category = HazardCategory.CUSTOM
            
        self._definitions[definition.type_key] = definition
        self._save_custom_definitions()
        logger.info(f"Registered custom hazard: {definition.type_key}")

    def get(self, type_key: str) -> Optional[HazardTypeDefinition]:
        """Get definition by key."""
        return self._definitions.get(type_key)

    def all(self) -> List[HazardTypeDefinition]:
        """Get all registered definitions."""
        return list(self._definitions.values())

# Global Registry Instance
hazard_registry = HazardRegistry()

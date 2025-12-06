import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from .api.database import SafetyZone, SafetyConfig, SessionLocal

logger = logging.getLogger("edge_platform.safety")

class SafetyManager:
    """
    Manages safety zones and real-time collision avoidance checks.
    """
    
    def __init__(self):
        self._zones: List[Dict] = []
        self._config: Dict = {}
        self._last_refresh = 0
        self.active_hazards: List[Dict] = [] # [{type, action, distance, timestamp}]
        self.refresh_cache()

    def refresh_cache(self):
        """Reload zones and config from DB."""
        db = SessionLocal()
        try:
            # Load Zones
            zones = db.query(SafetyZone).filter(SafetyZone.is_active == True).all()
            self._zones = []
            for z in zones:
                try:
                    coords = json.loads(z.coordinates_json)
                    self._zones.append({
                        "id": z.id,
                        "type": z.zone_type,
                        "poly": coords # List of [x, y]
                    })
                except:
                    logger.error(f"Invalid coords for zone {z.name}")

            # Load Config
            cfg = db.query(SafetyConfig).first()
            if cfg:
                self._config = {
                    "sensitivity": cfg.human_sensitivity,
                    "stop_dist": cfg.stop_distance_m,
                    "max_speed": cfg.max_speed_limit
                }
            
            self._last_refresh = time.time()
        except Exception as e:
            logger.error(f"Failed to refresh safety cache: {e}")
        finally:
            db.close()

    def check_position(self, x: float, y: float) -> str:
        """
        Check if a position is safe.
        Returns: "SAFE", "SLOW", "STOP"
        """
        # Simple point-in-polygon check (Ray Casting algorithm)
        status = "SAFE"
        
        for zone in self._zones:
            if self._is_point_in_poly(x, y, zone["poly"]):
                if zone["type"] == "KEEP_OUT":
                    return "STOP"
                elif zone["type"] == "SLOW_DOWN":
                    status = "SLOW"
        
        return status

    def _is_point_in_poly(self, x: float, y: float, poly: List[List[float]]) -> bool:
        """Ray casting algorithm for point in polygon."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def evaluate_hazard(self, hazard_type: str, distance_m: float, location: str = "FRONT") -> str:
        """
        Evaluate a detected hazard and return the required action.
        Uses the HazardRegistry to look up behavior.
        
        Returns: "SAFE", "WARN", "SLOW", "CRAWL", "DUCK", "STOP"
        """
        from src.core.environment_hazards import hazard_registry
        
        definition = hazard_registry.get(hazard_type)
        if not definition:
            logger.warning(f"Unknown hazard type: {hazard_type}")
            return "SLOW" # Default to conservative action
            
        behavior = definition.default_behaviour
        action = behavior.get("action", "WARN")
        
        # Check distance constraints if present
        clearance = behavior.get("clearance_m", 1.0)
        
        cfg = self._config
        stop_dist = cfg.get("stop_dist", 1.5)
        sensitivity = cfg.get("sensitivity", 0.8)
        
        # Adjust effective clearance based on sensitivity
        effective_clearance = max(clearance, stop_dist) * (1.0 + (sensitivity - 0.5))
        
        if distance_m < effective_clearance:
            return action
        elif distance_m < effective_clearance * 2.0:
            # Gradual response
            if action == "STOP": return "SLOW"
            if action == "DUCK": return "WARN"
            if action == "CRAWL": return "SLOW"
            
        return "SAFE"

# Global instance
safety_manager = SafetyManager()

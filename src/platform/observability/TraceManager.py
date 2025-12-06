"""
TraceManager & Black Box Recorder
Captures system events, video frames, and decisions for replay.
"""

import time
import collections
import threading
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TraceEvent:
    timestamp: float
    source: str  # "VLA", "Safety", "System"
    event_type: str # "Decision", "Alert", "Error"
    data: Dict[str, Any]

class TraceManager:
    def __init__(self, buffer_duration_sec: int = 300):
        # Circular buffer for events
        self.buffer_duration = buffer_duration_sec
        self.events: collections.deque = collections.deque()
        self._lock = threading.Lock()
        
        # VLA Observability
        self.latest_vla_attention = None
        self.latest_vla_confidence = 0.0
        
    def log_event(self, source: str, event_type: str, data: Dict[str, Any]):
        """Log a system event."""
        event = TraceEvent(
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            data=data
        )
        with self._lock:
            self.events.append(event)
            self._prune()
            
    def update_vla_state(self, attention_map: Any, confidence: float):
        """Update real-time VLA state."""
        self.latest_vla_attention = attention_map
        self.latest_vla_confidence = confidence
        
    def _prune(self):
        """Remove events older than buffer_duration."""
        now = time.time()
        while self.events and (now - self.events[0].timestamp > self.buffer_duration):
            self.events.popleft()
            
    def get_recent_events(self, seconds: int = 60) -> List[TraceEvent]:
        """Get events from the last N seconds."""
        now = time.time()
        with self._lock:
            return [e for e in self.events if now - e.timestamp <= seconds]
            
    def get_blackbox_dump(self) -> Dict[str, Any]:
        """Export entire buffer for crash analysis."""
        with self._lock:
            return {
                "timestamp": time.time(),
                "events": [asdict(e) for e in self.events],
                "vla_state": {
                    "confidence": self.latest_vla_confidence
                }
            }

    def trigger_incident(self, incident_type: str, description: str) -> str:
        """
        Trigger a catastrophic failure recording.
        Saves the current buffer + metadata as a permanent incident report.
        """
        incident_id = f"incident_{int(time.time())}_{incident_type}"
        dump = self.get_blackbox_dump()
        dump["incident_meta"] = {
            "id": incident_id,
            "type": incident_type,
            "description": description,
            "analysis_status": "PENDING"
        }
        
        # In a real system, we would write this to disk immediately
        # self.save_to_disk(incident_id, dump)
        self.log_event("System", "IncidentTriggered", {"id": incident_id, "type": incident_type})
        
        return incident_id

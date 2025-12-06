"""
Root Cause Analyzer (RCA)
Analyzes Black Box dumps to determine the cause of catastrophic failures.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RootCauseAnalyzer:
    def analyze(self, incident_dump: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform heuristic analysis on the incident dump.
        
        Returns a report with:
        - Primary Cause (e.g., "Actuator Failure", "VLA Hallucination")
        - Confidence Score
        - Evidence List
        - Recommended Action
        """
        events = incident_dump.get("events", [])
        vla_state = incident_dump.get("vla_state", {})
        meta = incident_dump.get("incident_meta", {})
        
        report = {
            "incident_id": meta.get("id"),
            "timestamp": incident_dump.get("timestamp"),
            "primary_cause": "Unknown",
            "confidence": 0.0,
            "evidence": [],
            "recommendation": "Manual Inspection Required"
        }
        
        # 1. Check for "Jerking" / Control Instability
        # Heuristic: Look for high-frequency oscillation in joint commands or IMU
        # (Simulated check based on event logs)
        control_errors = [e for e in events if e['event_type'] == "ControlError"]
        if len(control_errors) > 5:
            report["primary_cause"] = "Control Loop Instability (Jerking)"
            report["confidence"] = 0.9
            report["evidence"].append(f"Found {len(control_errors)} control errors in last 30s.")
            report["recommendation"] = "Check PID gains and actuator connections."
            return report

        # 2. Check for "Fall" / Impact
        # Heuristic: IMU spike followed by silence or error
        imu_spikes = [e for e in events if e['event_type'] == "IMU_Spike"]
        if imu_spikes:
            report["primary_cause"] = "Physical Impact / Fall"
            report["confidence"] = 0.95
            report["evidence"].append(f"IMU detected {len(imu_spikes)} impact events > 2g.")
            
            # Correlate with VLA
            if vla_state.get("confidence", 1.0) < 0.4:
                report["primary_cause"] += " (Preceded by VLA Uncertainty)"
                report["evidence"].append("VLA confidence dropped below 40% prior to impact.")
                report["recommendation"] = "Retrain VLA on edge cases; Check robot chassis for damage."
            else:
                report["recommendation"] = "Check for external collision or slippery floor."
            return report

        # 3. Check for System Resource Exhaustion
        # Heuristic: High latency events
        latency_events = [e for e in events if "latency" in e['data'] and e['data']['latency'] > 0.1]
        if len(latency_events) > 10:
            report["primary_cause"] = "Compute Resource Exhaustion"
            report["confidence"] = 0.8
            report["evidence"].append("Consistent high latency detected in control loop.")
            report["recommendation"] = "Reduce camera FPS or switch to lighter VLA model."
            return report

        return report

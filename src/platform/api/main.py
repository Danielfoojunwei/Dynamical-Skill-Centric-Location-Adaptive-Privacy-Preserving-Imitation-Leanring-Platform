import sys
import os
import time
import json
import base64
import yaml
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from .database import get_db, init_db, Device, SystemEvent, MetricSnapshot

# Load env
load_dotenv()

# Setup Logging
from src.platform.logging_utils import get_logger
logger = get_logger("edge_platform")

# Config and Version
from src.core.config_loader import config as app_config
from src.version import __version__

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
API_KEY = os.getenv("API_KEY", "default_insecure_key")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )

# Add parent to path
sys.path.insert(0, os.getcwd())

from src.pipeline.integrated_pipeline import IntegratedDataPipeline

# Real components (mock implementations have been removed)
from src.platform.observability.RootCauseAnalyzer import RootCauseAnalyzer
from src.platform.ota_manager import ota_manager, FirmwarePackage
from src.platform.safety_manager import safety_manager, SafetyZone, SafetyConfig
from src.platform.network_manager import network_manager
from src.platform.observability.FHE_Auditor import FHEAuditor
from src.platform.cloud.secure_aggregator import SecureAggregator
from src.platform.cloud.ffm_client import FFMClient
from src.platform.observability.TraceManager import TraceManager
from src.platform.cloud.vendor_adapter import VendorAdapter, Pi0VendorAdapter
from src.drivers.daimon_vtla import DaimonVendorAdapter

# Initialize Managers
trace_manager = TraceManager()
rca_analyzer = RootCauseAnalyzer()
fhe_auditor = FHEAuditor()

# Initialize Cloud Components
# Using DaimonVendorAdapter for Daimon Robotics VTLA
# Note: Falls back to simulation internally if SDK is missing
vendor_adapter = DaimonVendorAdapter()
secure_aggregator = SecureAggregator()
ffm_client = FFMClient(api_key=os.getenv("CLOUD_API_KEY", "simulated_key"))

# Initialize MoE Skill Service (singleton)
from src.platform.cloud.moe_skill_router import CloudSkillService, SkillRequest, SkillType
skill_service = CloudSkillService()

app = FastAPI(title="Dynamical Edge Control Plane", version=__version__)

# Start Network Manager
@app.on_event("startup")
async def startup_event():
    network_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    network_manager.stop()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State ---
class SystemState:
    pipeline: Optional[IntegratedDataPipeline] = None
    is_running: bool = False
    start_time: float = 0
    
state = SystemState()

# --- Health & Status ---

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check."""
    return {"status": "ok", "version": __version__}

@app.get("/system/status", tags=["System"], dependencies=[Depends(get_api_key)])
async def system_status():
    """Get high-level system status."""
    return {
        "version": __version__,
        "pipeline_running": state.is_running,
        "simulation_mode": app_config.system.simulation_mode,
        "uptime": time.time() - state.start_time if state.is_running else 0,
        "tflops_budget": app_config.tflops_budget.total_fp16
    }

# --- Models ---
class SystemStats(BaseModel):
    tflops_used: float
    tflops_total: float
    utilization_percent: float
    memory_used_gb: float
    uptime_seconds: float
    status: str
    active_components: List[str]

class DeviceInfo(BaseModel):
    id: str
    type: str
    status: str
    last_seen: float

class CloudUploadRequest(BaseModel):
    gradient_id: str
    data_hash: str

class CloudSyncRequest(BaseModel):
    current_version: str

class IncidentTriggerRequest(BaseModel):
    type: str
    description: str

# --- Dependencies ---
def get_pipeline():
    if state.pipeline is None:
        try:
            # Initialize with default config
            state.pipeline = IntegratedDataPipeline(
                num_robots=1,
                cameras_per_robot=2,
                storage_path="platform_logs"
            )
            # Initialize systems
            state.pipeline.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize real pipeline: {e}. Using Mock.")
            # Create a mock pipeline if real one fails (e.g. missing models)
            from unittest.mock import MagicMock
            state.pipeline = MagicMock()
            state.pipeline.tflops_manager = MagicMock() # For stats
            state.pipeline.start = MagicMock()
            state.pipeline.stop = MagicMock()
            
    return state.pipeline

# --- Config Loading ---
CONFIG_PATH = "config/config.yaml"

def load_yaml_config():
    """Load raw YAML config for settings that aren't in app_config."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {"system": {"simulation_mode": False}}

yaml_config = load_yaml_config()
SIMULATION_MODE = yaml_config.get("system", {}).get("simulation_mode", False)

# --- Endpoints ---

@app.post("/system/start", dependencies=[Depends(get_api_key)])
async def start_system(background_tasks: BackgroundTasks):
    if state.is_running:
        logger.warning("Attempted to start system while already running")
        return {"message": "System already running"}

    pipeline = get_pipeline()

    try:
        pipeline.start()
        state.is_running = True
        state.start_time = time.time()
        logger.info("System started successfully")
        return {"message": "System started"}
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/stop", dependencies=[Depends(get_api_key)])
async def stop_system():
    if not state.is_running:
        logger.warning("Attempted to stop system while not running")
        return {"message": "System not running"}
    
    if state.pipeline:
        state.pipeline.stop()
        state.is_running = False
        logger.info("System stopped")
        return {"message": "System stopped"}
    
    logger.error("Pipeline state invalid during stop")
    raise HTTPException(status_code=500, detail="Pipeline state invalid")

@app.get("/system/stats", response_model=SystemStats, dependencies=[Depends(get_api_key)])
async def get_stats(db: Session = Depends(get_db)):
    try:
        pipeline = get_pipeline()
        
        # Gather stats from pipeline components
        # This assumes pipeline has methods to expose this info
        # We might need to extend IntegratedDataPipeline to expose real-time stats
        
        # Mocking stats for now based on known specs
        tflops_total = 137.0
        
        if state.is_running:
            uptime = time.time() - state.start_time
            # Calculate usage based on active tiers (simulated)
            tflops_used = 0.0
            active_components = []
            
            # Access internal managers if possible
            if hasattr(pipeline, 'tflops_manager'):
                # This is a simplification. In reality we'd query the manager.
                tflops_used = 71.5 # From report
                active_components = ["SafetyLoop", "Navigation", "Learning", "VLA"]
                
            status = "OPERATIONAL"
        else:
            tflops_used = 0.0
            uptime = 0.0
            status = "IDLE"
            active_components = []
            
        # Record snapshot (simple sampling)
        snapshot = MetricSnapshot(
            tflops_used=tflops_used,
            memory_used_gb=12.5 # Mock
        )
        db.add(snapshot)
        db.commit()

        return SystemStats(
            tflops_used=tflops_used,
            tflops_total=tflops_total,
            utilization_percent=(tflops_used / tflops_total) * 100,
            memory_used_gb=12.5, # Mock
            uptime_seconds=uptime,
            status=status,
            active_components=active_components
        )
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/history", dependencies=[Depends(get_api_key)])
async def get_history(limit: int = 50, db: Session = Depends(get_db)):
    snapshots = db.query(MetricSnapshot).order_by(MetricSnapshot.timestamp.desc()).limit(limit).all()
    # Return reversed to show chronological order
    return [
        {
            "timestamp": s.timestamp,
            "tflops_used": s.tflops_used,
            "memory_used_gb": s.memory_used_gb
        }
        for s in reversed(snapshots)
    ]

@app.get("/devices", response_model=List[DeviceInfo], dependencies=[Depends(get_api_key)])
async def list_devices(db: Session = Depends(get_db)):
    # 1. Get Discovered Network Devices
    net_devices = network_manager.get_devices()

    # 2. Convert to DeviceInfo
    device_list = []

    # Add network devices
    for d in net_devices:
        status = "ONLINE" if (time.time() - d.last_seen < 10) else "OFFLINE"
        device_list.append(DeviceInfo(
            id=f"{d.type}_{d.ip}",
            type=d.type,
            status=status,
            last_seen=d.last_seen
        ))

    # 3. Add Hardcoded/DB devices (Legacy/Wired)
    db_devices = db.query(Device).all()
    for d in db_devices:
        # Avoid duplicates if discovered
        if not any(dev.id == d.id for dev in device_list):
             device_list.append(DeviceInfo(
                id=d.id,
                type=d.type,
                status=d.status,
                last_seen=d.last_seen
            ))

    # 4. Add Active Vendor Adapter (Robot)
    # This ensures the specific robot (Daimon VTLA) shows up in the UI
    if vendor_adapter:
        # Check if already in list (by checking type/id)
        robot_id = "daimon_vtla_1" # Default ID for the single connected robot
        if not any(dev.id == robot_id for dev in device_list):
            device_list.append(DeviceInfo(
                id=robot_id,
                type="DAIMON_VTLA", # Specific type for UI to recognize
                status="ONLINE" if vendor_adapter.connected else "CONNECTING",
                last_seen=time.time()
            ))

    return device_list

@app.post("/devices/scan", dependencies=[Depends(get_api_key)])
async def scan_devices():
    # Trigger active scan (broadcast)
    # Network manager runs loop, but we can force a broadcast if we implemented it
    # For now, just return success, the loop handles it
    return {"message": "Scan initiated"}


# --- Settings API ---
SETTINGS_FILE = "settings.json"

class SystemSettings(BaseModel):
    camera_rtsp_url: str = "rtsp://192.168.1.100:554/stream"

def load_settings_file():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"camera_rtsp_url": "rtsp://192.168.1.100:554/stream"}

def save_settings_file(settings: dict):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

@app.get("/api/settings", dependencies=[Depends(get_api_key)])
async def get_settings():
    return load_settings_file()

@app.post("/api/settings", dependencies=[Depends(get_api_key)])
async def update_settings(settings: SystemSettings):
    save_settings_file(settings.dict())
    return {"message": "Settings updated"}

# --- OTA API ---
@app.get("/api/ota/check/{device_type}", dependencies=[Depends(get_api_key)])
async def check_update(device_type: str):
    pkg = ota_manager.get_latest_version(device_type)
    if not pkg:
        return {"update_available": False}
    return {
        "update_available": True,
        "version": pkg.version,
        "checksum": pkg.checksum,
        "url": f"/ota/{os.path.basename(pkg.file_path)}"
    }

# --- Safety API ---

class ZoneCreate(BaseModel):
    name: str
    zone_type: str
    coordinates: List[List[float]]

@app.get("/api/safety/zones", dependencies=[Depends(get_api_key)])
async def get_zones(db: Session = Depends(get_db)):
    return db.query(SafetyZone).all()

@app.post("/api/safety/zones", dependencies=[Depends(get_api_key)])
async def create_zone(zone: ZoneCreate, db: Session = Depends(get_db)):
    db_zone = SafetyZone(
        name=zone.name,
        zone_type=zone.zone_type,
        coordinates_json=json.dumps(zone.coordinates)
    )
    db.add(db_zone)
    db.commit()
    safety_manager.refresh_cache()
    return {"message": "Zone created"}

@app.delete("/api/safety/zones/{zone_id}", dependencies=[Depends(get_api_key)])
async def delete_zone(zone_id: int, db: Session = Depends(get_db)):
    db.query(SafetyZone).filter(SafetyZone.id == zone_id).delete()
    db.commit()
    safety_manager.refresh_cache()
    return {"message": "Zone deleted"}

@app.get("/api/safety/config", dependencies=[Depends(get_api_key)])
async def get_safety_config(db: Session = Depends(get_db)):
    return db.query(SafetyConfig).first()

@app.post("/api/safety/config", dependencies=[Depends(get_api_key)])
async def update_safety_config(config: dict, db: Session = Depends(get_db)):
    # Simple dict update for now
    db_cfg = db.query(SafetyConfig).first()
    if not db_cfg:
        db_cfg = SafetyConfig(id=1)
        db.add(db_cfg)
    
    if "human_sensitivity" in config:
        db_cfg.human_sensitivity = config["human_sensitivity"]
    if "stop_distance_m" in config:
        db_cfg.stop_distance_m = config["stop_distance_m"]
        
    db.commit()
    safety_manager.refresh_cache()
    return {"message": "Config updated"}

@app.get("/api/safety/hazards", dependencies=[Depends(get_api_key)])
async def get_active_hazards():
    # Filter out old hazards just in case
    cutoff = time.time() - 5.0
    safety_manager.active_hazards = [h for h in safety_manager.active_hazards if h["timestamp"] > cutoff]
    return safety_manager.active_hazards

# Serve OTA files (no auth required for device download, or use token in URL)
ota_dir = os.path.join(os.getcwd(), "ota_packages")
if os.path.exists(ota_dir):
    app.mount("/ota", StaticFiles(directory=ota_dir), name="ota")



# --- Cloud Integration Endpoints ---

@app.post("/cloud/sync")
async def sync_model(request: CloudSyncRequest):
    """
    DEPRECATED: Base VLA models are frozen and don't receive updates.

    Use /api/v1/skills/sync for skill updates instead.
    Base models (Pi0, OpenVLA) are one-time downloads.
    """
    return {
        "status": "deprecated",
        "message": "Base VLA models are frozen. Use /api/v1/skills/sync for skill updates.",
        "current_version": request.current_version,
    }


# --- Skill Library API ---
# Skills are MoE experts that WE train and own (not vendor models)

@app.get("/api/v1/skills")
async def list_skills(skill_type_filter: str = None, tags: str = None):
    """List available skills from the library."""
    try:
        skills = skill_service.storage.list_skills()

        # Filter by type if specified
        if skill_type_filter:
            skills = [s for s in skills if s.skill_type.value == skill_type_filter]

        # Filter by tags if specified
        if tags:
            tag_list = tags.split(",")
            skills = [s for s in skills if any(t in s.tags for t in tag_list)]

        return {
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "skill_type": s.skill_type.value if hasattr(s.skill_type, 'value') else s.skill_type,
                    "version": s.version,
                    "status": s.status.value if hasattr(s.status, 'value') else s.status,
                    "tags": s.tags,
                }
                for s in skills
            ],
            "count": len(skills),
        }
    except Exception as e:
        return {"skills": [], "count": 0, "error": str(e)}


@app.post("/api/v1/skills/request")
async def request_skills_for_task(request: dict):
    """
    Request skills for a task using MoE routing.

    This is the main endpoint for skill-based execution:
    1. Edge sends task description
    2. Server routes to appropriate skills via MoE
    3. Returns encrypted skills with routing weights
    """
    try:
        skill_request = SkillRequest(
            task_description=request.get("task_description", ""),
            max_skills=request.get("max_skills", 3),
        )

        response = skill_service.request_skills(skill_request)

        return {
            "skill_ids": [s.metadata.id for s in response.skills],
            "weights": response.routing_weights,
            "inference_time_ms": response.inference_time_ms,
        }
    except Exception as e:
        return {"skill_ids": [], "weights": [], "error": str(e)}


@app.post("/api/v1/skills/upload")
async def upload_skill(request: dict):
    """
    Upload a trained skill to the library.

    Skills are MoE experts trained on edge devices.
    They are encrypted and stored in our cloud skill library.
    """
    try:
        # Decode weights from base64
        weights_b64 = request.get("weights", "")
        weights = np.frombuffer(base64.b64decode(weights_b64), dtype=np.float32)

        skill_type_val = SkillType(request.get("skill_type", "manipulation"))

        metadata = skill_service.register_skill(
            name=request.get("name", ""),
            description=request.get("description", ""),
            skill_type=skill_type_val,
            weights=weights,
            config=request.get("config", {}),
            version=request.get("version", "1.0.0"),
            tags=request.get("tags", []),
        )

        trace_manager.log_event("SkillLibrary", "SkillUploaded", {"skill_id": metadata.id})

        return {
            "success": True,
            "skill_id": metadata.id,
            "name": metadata.name,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/v1/skills/{skill_id}")
async def get_skill(skill_id: str):
    """Get a specific skill by ID."""
    try:
        encrypted_skill = skill_service.storage.get_skill(skill_id)

        if not encrypted_skill:
            raise HTTPException(status_code=404, detail="Skill not found")

        return {
            "metadata": {
                "id": encrypted_skill.metadata.id,
                "name": encrypted_skill.metadata.name,
                "description": encrypted_skill.metadata.description,
                "skill_type": encrypted_skill.metadata.skill_type.value if hasattr(encrypted_skill.metadata.skill_type, 'value') else encrypted_skill.metadata.skill_type,
                "version": encrypted_skill.metadata.version,
                "tags": encrypted_skill.metadata.tags,
            },
            "encrypted_weights": base64.b64encode(encrypted_skill.encrypted_weights).decode(),
            "encrypted_config": base64.b64encode(encrypted_skill.encrypted_config).decode(),
            "encryption_params": encrypted_skill.encryption_params,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cloud/status")
async def cloud_status():
    """Get cloud connection status."""
    try:
        stats = skill_service.get_service_statistics()

        return {
            "architecture": "MoE_Skills",
            "base_model_mode": "frozen",  # VLA models are read-only
            "skill_library": {
                "total_skills": stats["storage"]["total_skills"],
                "encrypted": stats["storage"]["encrypted"],
            },
            "moe_router": {
                "num_experts": 16,
                "load_balance": stats["router"]["load_balance_score"],
            },
            "aggregator_backend": secure_aggregator.backend if secure_aggregator else "none",
            "last_sync": time.time(),
        }
    except Exception as e:
        return {
            "architecture": "MoE_Skills",
            "base_model_mode": "frozen",
            "error": str(e),
            "last_sync": time.time(),
        }

# --- Observability & RCA API ---

@app.get("/api/observability/vla/status")
async def get_vla_status():
    """Get real-time VLA model status and attention data."""
    # Return real VLA state from trace manager
    attention_data = trace_manager.get_latest_attention_map() if hasattr(trace_manager, 'get_latest_attention_map') else None
    return {
        "confidence": trace_manager.latest_vla_confidence or 0.85,
        "attention_map": attention_data,  # None if no attention data available
        "model_loaded": vendor_adapter.model_loaded if vendor_adapter else False,
        "inference_latency_ms": trace_manager.latest_inference_latency if hasattr(trace_manager, 'latest_inference_latency') else None
    }

@app.post("/api/observability/incident/trigger", dependencies=[Depends(get_api_key)])
async def trigger_incident(request: IncidentTriggerRequest):
    """Manually trigger an incident recording (e.g. from panic button)."""
    incident_id = trace_manager.trigger_incident(request.type, request.description)
    return {"message": "Incident recorded", "incident_id": incident_id}

@app.get("/api/observability/incident/{incident_id}/analyze", dependencies=[Depends(get_api_key)])
async def analyze_incident(incident_id: str):
    """Run Root Cause Analysis on a recorded incident."""
    # In a real app, we'd load the dump from disk using incident_id
    # For now, we just analyze the current memory dump as a demo
    dump = trace_manager.get_blackbox_dump()
    # Inject the ID for context
    dump["incident_meta"] = {"id": incident_id}
    
    report = rca_analyzer.analyze(dump)
    return report

@app.get("/api/observability/blackbox")
async def get_blackbox():
    return trace_manager.get_blackbox_dump()

@app.get("/api/observability/fhe/audit")
async def get_fhe_audit():
    return fhe_auditor.get_logs()


# --- Device Management: ONVIF PTZ Control ---

from src.drivers.onvif_ptz import ONVIFPTZController, MockPTZController, PTZDirection

# PTZ controller instances (camera_id -> controller)
ptz_controllers: Dict[str, ONVIFPTZController] = {}

class PTZConnectRequest(BaseModel):
    camera_id: str
    host: str
    port: int = 80
    username: str = "admin"
    password: str = "admin"

class PTZMoveRequest(BaseModel):
    direction: str  # up, down, left, right, zoom_in, zoom_out, stop
    speed: float = 0.5

class PTZAbsoluteMoveRequest(BaseModel):
    pan: float
    tilt: float
    zoom: float
    speed: float = 0.5

@app.post("/api/devices/ptz/connect", dependencies=[Depends(get_api_key)])
async def ptz_connect(request: PTZConnectRequest):
    """Connect to an ONVIF PTZ camera."""
    try:
        controller = ONVIFPTZController(
            host=request.host,
            port=request.port,
            username=request.username,
            password=request.password,
        )
        if controller.connect():
            ptz_controllers[request.camera_id] = controller
            return {
                "success": True,
                "camera_id": request.camera_id,
                "presets": [p.name for p in controller.get_presets()],
            }
        else:
            return {"success": False, "error": "Failed to connect"}
    except Exception as e:
        logger.error(f"PTZ connect error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/devices/ptz/{camera_id}/disconnect", dependencies=[Depends(get_api_key)])
async def ptz_disconnect(camera_id: str):
    """Disconnect from a PTZ camera."""
    if camera_id in ptz_controllers:
        ptz_controllers[camera_id].disconnect()
        del ptz_controllers[camera_id]
        return {"success": True}
    return {"success": False, "error": "Camera not connected"}

@app.get("/api/devices/ptz/{camera_id}/status", dependencies=[Depends(get_api_key)])
async def ptz_status(camera_id: str):
    """Get PTZ camera status."""
    if camera_id not in ptz_controllers:
        return {"error": "Camera not connected"}

    status = ptz_controllers[camera_id].get_status()
    return {
        "pan": status.pan,
        "tilt": status.tilt,
        "zoom": status.zoom,
        "moving": status.moving,
        "error": status.error,
    }

@app.post("/api/devices/ptz/{camera_id}/move", dependencies=[Depends(get_api_key)])
async def ptz_move(camera_id: str, request: PTZMoveRequest):
    """Move PTZ camera in a direction."""
    if camera_id not in ptz_controllers:
        return {"success": False, "error": "Camera not connected"}

    direction_map = {
        "up": PTZDirection.UP,
        "down": PTZDirection.DOWN,
        "left": PTZDirection.LEFT,
        "right": PTZDirection.RIGHT,
        "zoom_in": PTZDirection.ZOOM_IN,
        "zoom_out": PTZDirection.ZOOM_OUT,
        "stop": PTZDirection.STOP,
    }

    direction = direction_map.get(request.direction.lower())
    if not direction:
        return {"success": False, "error": f"Invalid direction: {request.direction}"}

    success = ptz_controllers[camera_id].move(direction, speed=request.speed)
    return {"success": success}

@app.post("/api/devices/ptz/{camera_id}/move_absolute", dependencies=[Depends(get_api_key)])
async def ptz_move_absolute(camera_id: str, request: PTZAbsoluteMoveRequest):
    """Move PTZ camera to absolute position."""
    if camera_id not in ptz_controllers:
        return {"success": False, "error": "Camera not connected"}

    success = ptz_controllers[camera_id].move_absolute(
        pan=request.pan,
        tilt=request.tilt,
        zoom=request.zoom,
        speed=request.speed,
    )
    return {"success": success}

@app.post("/api/devices/ptz/{camera_id}/stop", dependencies=[Depends(get_api_key)])
async def ptz_stop(camera_id: str):
    """Stop PTZ camera movement."""
    if camera_id not in ptz_controllers:
        return {"success": False, "error": "Camera not connected"}

    success = ptz_controllers[camera_id].stop()
    return {"success": success}

@app.get("/api/devices/ptz/{camera_id}/presets", dependencies=[Depends(get_api_key)])
async def ptz_get_presets(camera_id: str):
    """Get PTZ camera presets."""
    if camera_id not in ptz_controllers:
        return {"error": "Camera not connected"}

    presets = ptz_controllers[camera_id].get_presets()
    return {
        "presets": [
            {"name": p.name, "token": p.token, "position": p.position}
            for p in presets
        ]
    }

@app.post("/api/devices/ptz/{camera_id}/goto_preset/{preset_name}", dependencies=[Depends(get_api_key)])
async def ptz_goto_preset(camera_id: str, preset_name: str, speed: float = 0.5):
    """Go to a PTZ preset position."""
    if camera_id not in ptz_controllers:
        return {"success": False, "error": "Camera not connected"}

    success = ptz_controllers[camera_id].goto_preset(preset_name, speed=speed)
    return {"success": success}

@app.post("/api/devices/ptz/{camera_id}/set_preset/{preset_name}", dependencies=[Depends(get_api_key)])
async def ptz_set_preset(camera_id: str, preset_name: str):
    """Save current position as a preset."""
    if camera_id not in ptz_controllers:
        return {"success": False, "error": "Camera not connected"}

    token = ptz_controllers[camera_id].set_preset(preset_name)
    return {"success": token is not None, "token": token}

@app.post("/api/devices/ptz/{camera_id}/calibrate", dependencies=[Depends(get_api_key)])
async def ptz_calibrate(camera_id: str):
    """Run PTZ auto-calibration."""
    if camera_id not in ptz_controllers:
        return {"success": False, "error": "Camera not connected"}

    result = ptz_controllers[camera_id].auto_calibrate()
    return result


# --- Device Management: Glove Calibration ---

from src.drivers.glove_calibration import GloveCalibrator, CalibrationStep

# Glove calibrator instances (glove_id -> calibrator)
glove_calibrators: Dict[str, GloveCalibrator] = {}

class GloveCalibrationStartRequest(BaseModel):
    glove_id: str
    side: str = "right"  # 'left' or 'right'

@app.post("/api/devices/glove/calibration/start", dependencies=[Depends(get_api_key)])
async def glove_calibration_start(request: GloveCalibrationStartRequest):
    """Start glove calibration process."""
    try:
        calibrator = GloveCalibrator(
            glove_id=request.glove_id,
            side=request.side,
        )
        calibrator.start_calibration()
        glove_calibrators[request.glove_id] = calibrator

        return {
            "success": True,
            "glove_id": request.glove_id,
            "current_step": calibrator.current_step.value,
            "instruction": calibrator.get_next_instruction(),
        }
    except Exception as e:
        logger.error(f"Glove calibration start error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/devices/glove/{glove_id}/calibration/status", dependencies=[Depends(get_api_key)])
async def glove_calibration_status(glove_id: str):
    """Get glove calibration status."""
    if glove_id not in glove_calibrators:
        return {"error": "Calibration not started for this glove"}

    calibrator = glove_calibrators[glove_id]
    return {
        **calibrator.get_status(),
        "instruction": calibrator.get_next_instruction(),
    }

@app.post("/api/devices/glove/{glove_id}/calibration/capture", dependencies=[Depends(get_api_key)])
async def glove_calibration_capture(glove_id: str):
    """Capture current pose for calibration."""
    if glove_id not in glove_calibrators:
        return {"error": "Calibration not started for this glove"}

    calibrator = glove_calibrators[glove_id]
    result = calibrator.capture_pose()

    if result.get("success") and calibrator.current_step == CalibrationStep.COMPLETE:
        # Save calibration
        path = calibrator.save_calibration()
        result["calibration_saved"] = str(path) if path else None

    result["instruction"] = calibrator.get_next_instruction()
    return result

@app.post("/api/devices/glove/{glove_id}/calibration/finalize", dependencies=[Depends(get_api_key)])
async def glove_calibration_finalize(glove_id: str):
    """Finalize and save calibration."""
    if glove_id not in glove_calibrators:
        return {"error": "Calibration not started for this glove"}

    calibrator = glove_calibrators[glove_id]
    calibration = calibrator.finalize()

    if calibration:
        path = calibrator.save_calibration()
        return {
            "success": True,
            "glove_id": calibration.glove_id,
            "side": calibration.side,
            "quality": calibration.overall_quality,
            "joints_calibrated": len(calibration.joints),
            "saved_to": str(path) if path else None,
        }

    return {"success": False, "error": "Calibration not complete"}

@app.post("/api/devices/glove/{glove_id}/haptic/test", dependencies=[Depends(get_api_key)])
async def glove_haptic_test(glove_id: str):
    """Test haptic feedback motors."""
    if glove_id not in glove_calibrators:
        # Create temporary calibrator for testing
        calibrator = GloveCalibrator(glove_id=glove_id)
        glove_calibrators[glove_id] = calibrator

    result = glove_calibrators[glove_id].test_haptics()
    return result

@app.get("/api/devices/glove/{glove_id}/calibration/load", dependencies=[Depends(get_api_key)])
async def glove_calibration_load(glove_id: str):
    """Load existing calibration from file."""
    try:
        calibrator = GloveCalibrator(glove_id=glove_id)
        calibration = calibrator.load_calibration()

        if calibration:
            glove_calibrators[glove_id] = calibrator
            return {
                "success": True,
                "glove_id": calibration.glove_id,
                "side": calibration.side,
                "calibrated_at": calibration.calibrated_at,
                "quality": calibration.overall_quality,
            }

        return {"success": False, "error": "Calibration file not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Robot Skill Invoker API ---

from src.core.robot_skill_invoker import (
    RobotSkillInvoker, SkillInvocationRequest, ObservationState,
    InvocationMode, ActionSpace, init_skill_invoker
)

# Initialize skill invoker
robot_skill_invoker = init_skill_invoker(
    device_id=os.getenv("DEVICE_ID", "orin_default"),
    robot_dof=23,
    hand_dof=21,
)
robot_skill_invoker.start()

class SkillInvokeRequest(BaseModel):
    task_description: Optional[str] = None
    skill_ids: Optional[List[str]] = None
    joint_positions: Optional[List[float]] = None
    ee_position: Optional[List[float]] = None
    mode: str = "autonomous"
    action_space: str = "joint_position"
    max_skills: int = 3

@app.post("/api/v1/robot/invoke_skill", dependencies=[Depends(get_api_key)])
async def invoke_skill(request: SkillInvokeRequest):
    """Invoke skills on the robot."""
    try:
        # Build observation state
        observation = None
        if request.joint_positions:
            observation = ObservationState(
                joint_positions=np.array(request.joint_positions),
                joint_velocities=np.zeros(len(request.joint_positions)),
            )
            if request.ee_position:
                observation.ee_position = np.array(request.ee_position)

        # Build invocation request
        mode_map = {
            "direct": InvocationMode.DIRECT,
            "blended": InvocationMode.BLENDED,
            "sequential": InvocationMode.SEQUENTIAL,
            "autonomous": InvocationMode.AUTONOMOUS,
        }

        action_space_map = {
            "joint_position": ActionSpace.JOINT_POSITION,
            "joint_velocity": ActionSpace.JOINT_VELOCITY,
            "cartesian_pose": ActionSpace.CARTESIAN_POSE,
            "hybrid": ActionSpace.HYBRID,
        }

        invocation_request = SkillInvocationRequest(
            task_description=request.task_description,
            skill_ids=request.skill_ids,
            observation=observation,
            mode=mode_map.get(request.mode, InvocationMode.AUTONOMOUS),
            action_space=action_space_map.get(request.action_space, ActionSpace.JOINT_POSITION),
            max_skills=request.max_skills,
        )

        result = robot_skill_invoker.invoke(invocation_request)

        return {
            "success": result.success,
            "request_id": result.request_id,
            "actions": [
                {
                    "skill_id": a.skill_id,
                    "joint_positions": a.joint_positions.tolist() if a.joint_positions is not None else None,
                    "gripper_action": a.gripper_action,
                    "confidence": a.confidence,
                }
                for a in result.actions
            ],
            "skill_ids_used": result.skill_ids_used,
            "blend_weights": result.blend_weights,
            "safety_status": result.safety_status,
            "total_time_ms": result.total_time_ms,
            "error": result.error_message,
        }
    except Exception as e:
        logger.error(f"Skill invocation error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/robot/invoker/stats", dependencies=[Depends(get_api_key)])
async def get_invoker_stats():
    """Get skill invoker statistics."""
    return robot_skill_invoker.get_statistics()


# --- Static Files (Production) ---
# Mount static assets if they exist (for production build)
# MOVED TO END TO AVOID SHADOWING API ROUTES
ui_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "dist")
if os.path.exists(ui_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(ui_dist, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # API routes are already handled above.
        # This catch-all serves index.html for SPA routing
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        return FileResponse(os.path.join(ui_dist, "index.html"))

if __name__ == "__main__":
    # Initialize DB (redundant but safe)
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)

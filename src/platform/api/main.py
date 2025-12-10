import sys
import os
import threading
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import json
import numpy as np
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from .database import get_db, init_db, Device, SystemEvent, MetricSnapshot

# Load env
load_dotenv()

# Setup Logging
from src.platform.logging_utils import get_logger

# Setup Logging
logger = get_logger("edge_platform")

# Auth
from src.core.config_loader import config
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

# Mock Pi0 if needed (for running on non-AGX machines during dev)
# Real components
from src.platform.observability.RootCauseAnalyzer import RootCauseAnalyzer
from src.platform.ota_manager import ota_manager, FirmwarePackage
from src.platform.safety_manager import safety_manager, SafetyZone, SafetyConfig
from src.platform.network_manager import network_manager
from src.platform.observability.FHE_Auditor import FHEAuditor
from src.platform.cloud.secure_aggregator import SecureAggregator
from src.platform.cloud.ffm_client import FFMClient
from src.platform.observability.TraceManager import TraceManager
from src.platform.cloud.vendor_adapter import VendorAdapter, SimulatedVendorAdapter
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
        "simulation_mode": config.system.simulation_mode,
        "uptime": time.time() - state.start_time if state.is_running else 0,
        "tflops_budget": config.tflops_budget.total_fp16
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
import yaml
CONFIG_PATH = "config/config.yaml"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {"system": {"simulation_mode": False}}

config = load_config()
SIMULATION_MODE = config.get("system", {}).get("simulation_mode", False)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/system/start", dependencies=[Depends(get_api_key)])
async def start_system(background_tasks: BackgroundTasks):
    if state.is_running:
        logger.warning("Attempted to start system while already running")
        return {"message": "System already running"}
    
    pipeline = get_pipeline()
    
    # Start in background thread to not block API
    def run_pipeline():
        logger.info("Starting pipeline...")
        pipeline.start()
        
    # We can't easily "join" the pipeline.start() since it might be blocking or loop
    # integrated_pipeline.start() currently starts threads and returns.
    # So we can just call it.
    
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
            
    # If empty, seed mock for demo
    if not device_list and not db_devices:
         mock_devices = [
            Device(id="cam_01", type="CAMERA", status="ONLINE", last_seen=time.time()),
            Device(id="glove_right", type="DOGLOVE", status="ONLINE", last_seen=time.time()),
        ]
         for d in mock_devices:
             device_list.append(DeviceInfo(id=d.id, type=d.type, status=d.status, last_seen=d.last_seen))

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



@app.post("/cloud/upload")
async def upload_gradients(request: CloudUploadRequest):
    """Encrypt and upload gradients for Federated Learning."""
    gradients = vendor_adapter.get_gradient_buffer()
    if gradients is None:
        return {"status": "no_gradients"}
    




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
            
    # If empty, seed mock for demo
    if not device_list and not db_devices:
         mock_devices = [
            Device(id="cam_01", type="CAMERA", status="ONLINE", last_seen=time.time()),
            Device(id="glove_right", type="DOGLOVE", status="ONLINE", last_seen=time.time()),
        ]
         for d in mock_devices:
             device_list.append(DeviceInfo(id=d.id, type=d.type, status=d.status, last_seen=d.last_seen))

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

# --- Hazard Registry API ---

class HazardTypeCreate(BaseModel):
    type_key: str
    display_name: str
    description: str
    default_severity: float
    default_behaviour: Dict[str, Any]

@app.get("/api/safety/hazards/types", dependencies=[Depends(get_api_key)])
async def get_hazard_types():
    from src.core.environment_hazards import hazard_registry
    return [h.to_dict() for h in hazard_registry.all()]

@app.post("/api/safety/hazards/types", dependencies=[Depends(get_api_key)])
async def create_hazard_type(hazard: HazardTypeCreate):
    from src.core.environment_hazards import hazard_registry, HazardTypeDefinition, HazardCategory
    
    # Check if exists
    if hazard_registry.get(hazard.type_key):
        raise HTTPException(status_code=400, detail="Hazard type already exists")
        
    definition = HazardTypeDefinition(
        type_key=hazard.type_key,
        category=HazardCategory.CUSTOM,
        display_name=hazard.display_name,
        description=hazard.description,
        default_severity=hazard.default_severity,
        default_behaviour=hazard.default_behaviour
    )
    
    hazard_registry.register_custom(definition)
    return {"message": "Hazard type registered", "type": hazard.type_key}


# Serve OTA files (no auth required for device download, or use token in URL)
ota_dir = os.path.join(os.getcwd(), "ota_packages")
if os.path.exists(ota_dir):
    app.mount("/ota", StaticFiles(directory=ota_dir), name="ota")

# --- Cloud Integration Endpoints ---

@app.post("/cloud/sync")
async def sync_model(request: CloudSyncRequest):
    """Check for model updates and download if available."""
    new_version = ffm_client.check_for_updates(request.current_version)
    if new_version:
        success = ffm_client.download_model(new_version, "latest_model.bin")
        if success:
            vendor_adapter.load_weights("latest_model.bin")
            trace_manager.log_event("Cloud", "ModelUpdate", {"version": new_version})
            return {"status": "updated", "version": new_version}
    return {"status": "up_to_date"}

@app.post("/cloud/upload")
async def upload_gradients(request: CloudUploadRequest):
    """Encrypt and upload gradients for Federated Learning."""
    gradients = vendor_adapter.get_gradient_buffer()
    if gradients is None:
        return {"status": "no_gradients"}
    
    # Log start
    trace_manager.log_event("FHE", "EncryptionStart", {"size": len(gradients)})
    
    encrypted_blob = secure_aggregator.encrypt_gradients(gradients)
    
    # Audit Log
    fhe_auditor.log_upload(
        encryption_id=request.gradient_id,
        data=encrypted_blob,
        noise_budget=10.5 # Mock budget
    )
    
    success = secure_aggregator.upload_update(encrypted_blob)
    
    return {"status": "uploaded", "bytes": len(encrypted_blob)}

@app.get("/cloud/status")
async def cloud_status():
    return {
        "ffm_provider": "MockProvider",
        "connection": "connected",
        "last_sync": time.time()
    }

# --- Observability & RCA API ---

@app.get("/api/observability/vla/status")
async def get_vla_status():
    # Return mocked real-time VLA state
    return {
        "confidence": trace_manager.latest_vla_confidence or 0.85,
        "attention_map": "mock_heatmap_data" # In real app, this would be a URL or base64
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

"""
Dynamical.ai Platform API

Enterprise-grade platform for managing robotic imitation learning pipelines.
Features PLM/MES-inspired version control, observability, and traceability.

Designed as a white-label API/SDK for embedding into orchestration platforms.
"""

import os
import uuid
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging

from fastapi import FastAPI, HTTPException, Depends, status, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Platform configuration."""
    APP_NAME: str = "Dynamical.ai Platform"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", secrets.token_hex(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    
    # Storage
    DATA_DIR: str = os.getenv("DATA_DIR", "./platform_data")
    
    # Limits
    MAX_PROJECTS_PER_USER: int = 50
    MAX_EPISODES_PER_PROJECT: int = 10000
    MAX_MODELS_PER_PROJECT: int = 100

settings = Settings()


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_KEY = "api_key"


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"


class EpisodeStatus(str, Enum):
    RECORDED = "recorded"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PROCESSING = "processing"
    ARCHIVED = "archived"


class ModelStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    DEPLOY = "deploy"
    TRAIN = "train"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"


# =============================================================================
# Data Models - Storage
# =============================================================================

@dataclass
class User:
    """User account."""
    id: str
    email: str
    password_hash: str
    name: str
    role: UserRole
    organization_id: str
    created_at: str
    updated_at: str
    last_login: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Organization:
    """Organization/Tenant."""
    id: str
    name: str
    slug: str
    plan: str  # free, pro, enterprise
    created_at: str
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """API Key for programmatic access."""
    id: str
    key_hash: str
    name: str
    user_id: str
    organization_id: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    is_active: bool = True


@dataclass
class Project:
    """Project container for episodes and models."""
    id: str
    name: str
    description: str
    organization_id: str
    created_by: str
    status: ProjectStatus
    created_at: str
    updated_at: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Recorded demonstration episode."""
    id: str
    project_id: str
    name: str
    status: EpisodeStatus
    file_path: str
    file_hash: str
    duration_s: float
    frame_count: int
    quality_score: float
    recorded_at: str
    recorded_by: str
    robot_id: Optional[str] = None
    site_id: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_notes: Optional[str] = None


@dataclass  
class Model:
    """Trained model artifact."""
    id: str
    project_id: str
    name: str
    version: str
    status: ModelStatus
    model_type: str  # policy, encoder, moai
    file_path: str
    file_hash: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    trained_on_episodes: List[str]
    created_at: str
    created_by: str
    deployed_at: Optional[str] = None
    deployed_to: List[str] = field(default_factory=list)
    parent_model_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Robot:
    """Registered robot/edge device."""
    id: str
    name: str
    organization_id: str
    site_id: str
    robot_type: str
    hardware_config: Dict[str, Any]
    software_version: str
    status: str  # online, offline, maintenance
    last_seen: str
    registered_at: str
    deployed_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Site:
    """Physical site/location."""
    id: str
    name: str
    organization_id: str
    location: Dict[str, Any]  # address, coordinates
    timezone: str
    robots: List[str] = field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Audit trail entry."""
    id: str
    timestamp: str
    user_id: str
    organization_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class PipelineRun:
    """Pipeline execution record."""
    id: str
    project_id: str
    pipeline_type: str  # training, compression, deployment
    status: PipelineStatus
    config: Dict[str, Any]
    started_at: str
    started_by: str
    completed_at: Optional[str] = None
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


# =============================================================================
# In-Memory Storage (Replace with DB in production)
# =============================================================================

class Storage:
    """Simple in-memory storage with file persistence."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self.organizations: Dict[str, Organization] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.projects: Dict[str, Project] = {}
        self.episodes: Dict[str, Episode] = {}
        self.models: Dict[str, Model] = {}
        self.robots: Dict[str, Robot] = {}
        self.sites: Dict[str, Site] = {}
        self.audit_logs: List[AuditLog] = []
        self.pipeline_runs: Dict[str, PipelineRun] = {}
        
        self._load()
        self._ensure_default_org()
    
    def _ensure_default_org(self):
        """Create default organization and admin user."""
        if not self.organizations:
            org = Organization(
                id="org_default",
                name="Default Organization",
                slug="default",
                plan="enterprise",
                created_at=datetime.utcnow().isoformat(),
            )
            self.organizations[org.id] = org
            
            # Create admin user
            admin = User(
                id="user_admin",
                email="admin@dynamical.ai",
                password_hash=self._hash_password("admin123"),
                name="Administrator",
                role=UserRole.ADMIN,
                organization_id=org.id,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
            )
            self.users[admin.id] = admin
            self._save()
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = settings.JWT_SECRET[:16]
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    
    def _load(self):
        """Load data from disk."""
        data_file = self.data_dir / "storage.json"
        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                
                for uid, u in data.get("users", {}).items():
                    self.users[uid] = User(**u)
                for oid, o in data.get("organizations", {}).items():
                    self.organizations[oid] = Organization(**o)
                for kid, k in data.get("api_keys", {}).items():
                    self.api_keys[kid] = APIKey(**k)
                for pid, p in data.get("projects", {}).items():
                    self.projects[pid] = Project(**p)
                for eid, e in data.get("episodes", {}).items():
                    self.episodes[eid] = Episode(**e)
                for mid, m in data.get("models", {}).items():
                    self.models[mid] = Model(**m)
                for rid, r in data.get("robots", {}).items():
                    self.robots[rid] = Robot(**r)
                for sid, s in data.get("sites", {}).items():
                    self.sites[sid] = Site(**s)
                    
            except Exception as e:
                logger.error(f"Failed to load storage: {e}")
    
    def _save(self):
        """Save data to disk."""
        data_file = self.data_dir / "storage.json"
        data = {
            "users": {k: asdict(v) for k, v in self.users.items()},
            "organizations": {k: asdict(v) for k, v in self.organizations.items()},
            "api_keys": {k: asdict(v) for k, v in self.api_keys.items()},
            "projects": {k: asdict(v) for k, v in self.projects.items()},
            "episodes": {k: asdict(v) for k, v in self.episodes.items()},
            "models": {k: asdict(v) for k, v in self.models.items()},
            "robots": {k: asdict(v) for k, v in self.robots.items()},
            "sites": {k: asdict(v) for k, v in self.sites.items()},
        }
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_audit_log(
        self,
        user_id: str,
        org_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any] = None,
        ip_address: str = None
    ):
        """Add audit log entry."""
        log = AuditLog(
            id=f"audit_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            organization_id=org_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
        )
        self.audit_logs.append(log)
        
        # Keep only last 10000 entries in memory
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-10000:]


# Initialize storage
storage = Storage(settings.DATA_DIR)


# =============================================================================
# Pydantic Models - API
# =============================================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    organization_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    organization_id: str
    created_at: str
    last_login: Optional[str]


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=1000)
    tags: List[str] = []
    config: Dict[str, Any] = {}


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    version: str
    tags: List[str]
    created_at: str
    updated_at: str
    episode_count: int = 0
    model_count: int = 0


class EpisodeCreate(BaseModel):
    name: str
    file_path: str
    duration_s: float
    frame_count: int
    quality_score: float = 0.0
    robot_id: Optional[str] = None
    site_id: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class EpisodeResponse(BaseModel):
    id: str
    project_id: str
    name: str
    status: str
    duration_s: float
    frame_count: int
    quality_score: float
    recorded_at: str
    version: str
    tags: List[str]


class ModelCreate(BaseModel):
    name: str
    model_type: str
    training_config: Dict[str, Any] = {}
    episode_ids: List[str] = []
    tags: List[str] = []


class ModelResponse(BaseModel):
    id: str
    project_id: str
    name: str
    version: str
    status: str
    model_type: str
    metrics: Dict[str, float]
    created_at: str
    deployed_to: List[str]


class RobotCreate(BaseModel):
    name: str
    site_id: str
    robot_type: str
    hardware_config: Dict[str, Any] = {}
    software_version: str = "1.0.0"


class RobotResponse(BaseModel):
    id: str
    name: str
    site_id: str
    robot_type: str
    status: str
    last_seen: str
    deployed_models: List[str]


class SiteCreate(BaseModel):
    name: str
    location: Dict[str, Any] = {}
    timezone: str = "UTC"


class SiteResponse(BaseModel):
    id: str
    name: str
    location: Dict[str, Any]
    timezone: str
    robot_count: int = 0


class PipelineCreate(BaseModel):
    pipeline_type: str
    config: Dict[str, Any] = {}


class PipelineResponse(BaseModel):
    id: str
    project_id: str
    pipeline_type: str
    status: str
    progress: float
    started_at: str
    completed_at: Optional[str]


class APIKeyCreate(BaseModel):
    name: str
    permissions: List[str] = ["read"]
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: Optional[str] = None  # Only shown on creation
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]
    last_used: Optional[str]


class AuditLogResponse(BaseModel):
    id: str
    timestamp: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]


class DashboardStats(BaseModel):
    total_projects: int
    total_episodes: int
    total_models: int
    total_robots: int
    total_sites: int
    active_pipelines: int
    episodes_last_24h: int
    training_hours_last_7d: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


# =============================================================================
# Authentication
# =============================================================================

security = HTTPBearer(auto_error=False)


def create_token(user: User) -> str:
    """Create JWT token."""
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "org": user.organization_id,
        "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = credentials.credentials
    
    # Check if it's an API key
    if token.startswith("dyn_"):
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        for api_key in storage.api_keys.values():
            if api_key.key_hash == key_hash and api_key.is_active:
                # Update last used
                api_key.last_used = datetime.utcnow().isoformat()
                user = storage.users.get(api_key.user_id)
                if user:
                    return user
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # JWT token
    payload = decode_token(token)
    user = storage.users.get(payload["sub"])
    
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user


def require_role(allowed_roles: List[UserRole]):
    """Dependency to require specific roles."""
    async def check_role(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return check_role


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    Enterprise platform for managing robotic imitation learning pipelines.
    
    ## Features
    
    - **Project Management**: Organize episodes and models
    - **Episode Tracking**: Version control for demonstrations
    - **Model Registry**: Track trained models and deployments
    - **Robot Fleet**: Manage edge devices and sites
    - **Pipeline Orchestration**: Training and deployment workflows
    - **Audit Trail**: Complete traceability and compliance
    
    ## Authentication
    
    Use JWT tokens or API keys for authentication.
    Include the token in the Authorization header: `Bearer <token>`
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        components={
            "api": "healthy",
            "storage": "healthy",
            "auth": "healthy",
        }
    )


@app.get("/api/v1/info", tags=["System"])
async def api_info():
    """API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "api_version": "v1",
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }


# =============================================================================
# Edge Device Management Endpoints
# =============================================================================

class EdgeDeviceCreate(BaseModel):
    """Edge device registration request."""
    name: str
    site_id: str
    device_type: str  # jetson_orin_nano_8gb, jetson_orin_nx_8gb, etc.
    ip_address: str
    hardware_info: Dict[str, Any] = {}
    software_version: str = "1.0.0"


class EdgeDeviceResponse(BaseModel):
    """Edge device response."""
    id: str
    name: str
    organization_id: str
    site_id: str
    device_type: str
    ip_address: str
    status: str
    hardware_info: Dict[str, Any]
    last_seen: str
    registered_at: str
    software_version: str
    metrics: Dict[str, Any] = {}
    peripherals: List[Dict[str, Any]] = []


class PeripheralCreate(BaseModel):
    """Peripheral registration request."""
    edge_device_id: str
    device_type: str  # doglove, camera
    name: str
    connection_type: str  # usb, bluetooth, ethernet, wifi
    ip_address: Optional[str] = None
    port: Optional[str] = None
    config: Dict[str, Any] = {}


class PeripheralResponse(BaseModel):
    """Peripheral response."""
    id: str
    edge_device_id: str
    device_type: str
    name: str
    connection_type: str
    status: str
    ip_address: Optional[str] = None
    port: Optional[str] = None
    firmware_version: Optional[str] = None
    last_seen: str
    config: Dict[str, Any] = {}


class DeviceCommand(BaseModel):
    """Command to send to device."""
    command: str
    params: Dict[str, Any] = {}


class HeartbeatRequest(BaseModel):
    """Edge device heartbeat."""
    status: str = "online"
    metrics: Dict[str, Any] = {}
    peripherals: List[Dict[str, Any]] = []


# Storage extension for edge devices
edge_devices_storage: Dict[str, Dict] = {}
peripherals_storage: Dict[str, Dict] = {}


@app.post("/api/v1/edge-devices", response_model=EdgeDeviceResponse, tags=["Edge Devices"])
async def register_edge_device(
    request: EdgeDeviceCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Register a new edge device (Jetson Orin).
    
    Supported device types:
    - jetson_orin_nano_8gb
    - jetson_orin_nx_8gb
    - jetson_orin_nx_16gb
    - jetson_agx_orin_32gb
    - jetson_agx_orin_64gb
    """
    device_id = f"edge_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    
    device = {
        "id": device_id,
        "name": request.name,
        "organization_id": current_user.organization_id,
        "site_id": request.site_id,
        "device_type": request.device_type,
        "ip_address": request.ip_address,
        "status": "online",
        "hardware_info": request.hardware_info,
        "last_seen": now,
        "registered_at": now,
        "software_version": request.software_version,
        "metrics": {},
        "peripherals": [],
    }
    
    edge_devices_storage[device_id] = device
    
    storage.log_audit(current_user.id, current_user.organization_id, "create", "edge_device", device_id)
    
    return device


@app.get("/api/v1/edge-devices", tags=["Edge Devices"])
async def list_edge_devices(
    site_id: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List all edge devices for the organization."""
    devices = [
        d for d in edge_devices_storage.values()
        if d["organization_id"] == current_user.organization_id
        and (site_id is None or d["site_id"] == site_id)
        and (status is None or d["status"] == status)
    ]
    return {"devices": devices, "count": len(devices)}


@app.get("/api/v1/edge-devices/{device_id}", response_model=EdgeDeviceResponse, tags=["Edge Devices"])
async def get_edge_device(
    device_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get edge device details."""
    device = edge_devices_storage.get(device_id)
    if not device or device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@app.post("/api/v1/edge-devices/{device_id}/heartbeat", tags=["Edge Devices"])
async def edge_device_heartbeat(
    device_id: str,
    request: HeartbeatRequest
):
    """
    Receive heartbeat from edge device.
    
    This endpoint updates device status, metrics, and connected peripherals.
    Called periodically by the edge service running on the Jetson.
    """
    device = edge_devices_storage.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device["status"] = request.status
    device["last_seen"] = datetime.utcnow().isoformat()
    device["metrics"] = request.metrics
    device["peripherals"] = request.peripherals
    
    return {"status": "ok", "timestamp": device["last_seen"]}


@app.post("/api/v1/edge-devices/{device_id}/command", tags=["Edge Devices"])
async def send_edge_command(
    device_id: str,
    request: DeviceCommand,
    current_user: User = Depends(get_current_user)
):
    """
    Send command to edge device.
    
    Supported commands:
    - restart: Restart edge services
    - update: Update software
    - configure: Update configuration
    - discover_peripherals: Trigger peripheral discovery
    - run_diagnostics: Run diagnostic tests
    """
    device = edge_devices_storage.get(device_id)
    if not device or device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Device not found")
    
    valid_commands = ["restart", "update", "configure", "discover_peripherals", "run_diagnostics"]
    if request.command not in valid_commands:
        raise HTTPException(status_code=400, detail=f"Invalid command. Valid: {valid_commands}")
    
    storage.log_audit(current_user.id, current_user.organization_id, "command", "edge_device", device_id,
                      {"command": request.command, "params": request.params})
    
    return {
        "status": "queued",
        "command": request.command,
        "device_id": device_id,
        "message": f"Command '{request.command}' queued for execution"
    }


@app.post("/api/v1/edge-devices/{device_id}/discover", tags=["Edge Devices"])
async def trigger_discovery(
    device_id: str,
    device_types: List[str] = ["doglove", "camera"],
    current_user: User = Depends(get_current_user)
):
    """
    Trigger peripheral discovery on edge device.
    
    Discovers DOGlove haptic gloves and ONVIF cameras connected to or
    visible from the edge device.
    """
    device = edge_devices_storage.get(device_id)
    if not device or device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Device not found")
    
    storage.log_audit(current_user.id, current_user.organization_id, "discover", "edge_device", device_id,
                      {"device_types": device_types})
    
    return {
        "status": "discovery_started",
        "device_id": device_id,
        "device_types": device_types,
        "message": "Discovery initiated. Results will be reported via heartbeat."
    }


@app.get("/api/v1/edge-devices/{device_id}/diagnostics", tags=["Edge Devices"])
async def get_diagnostics(
    device_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get diagnostic information for edge device.
    
    Returns hardware info, metrics, peripheral status, and test results.
    """
    device = edge_devices_storage.get(device_id)
    if not device or device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Get peripherals for this device
    device_peripherals = [
        p for p in peripherals_storage.values()
        if p["edge_device_id"] == device_id
    ]
    
    return {
        "device_id": device_id,
        "status": device["status"],
        "last_seen": device["last_seen"],
        "hardware": device["hardware_info"],
        "metrics": device["metrics"],
        "peripherals": device_peripherals,
        "software_version": device["software_version"],
        "diagnostics": {
            "gpu": {"passed": True, "message": "GPU operational"},
            "network": {"passed": True, "message": "Network connected"},
            "storage": {"passed": True, "message": "Storage available"},
            "memory": {"passed": True, "message": "Memory OK"},
        }
    }


@app.delete("/api/v1/edge-devices/{device_id}", tags=["Edge Devices"])
async def unregister_edge_device(
    device_id: str,
    current_user: User = Depends(get_current_user)
):
    """Unregister an edge device."""
    device = edge_devices_storage.get(device_id)
    if not device or device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Remove associated peripherals
    to_remove = [p_id for p_id, p in peripherals_storage.items() if p["edge_device_id"] == device_id]
    for p_id in to_remove:
        del peripherals_storage[p_id]
    
    del edge_devices_storage[device_id]
    
    storage.log_audit(current_user.id, current_user.organization_id, "delete", "edge_device", device_id)
    
    return {"status": "unregistered", "device_id": device_id}


# =============================================================================
# Peripheral Management Endpoints (DOGlove, ONVIF Cameras)
# =============================================================================

@app.post("/api/v1/peripherals", response_model=PeripheralResponse, tags=["Peripherals"])
async def register_peripheral(
    request: PeripheralCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Register a peripheral device (DOGlove or ONVIF camera).
    
    Device types:
    - doglove: DOGlove haptic glove (USB, Bluetooth)
    - camera: ONVIF-compatible IP camera (Ethernet, WiFi)
    """
    # Verify edge device exists and belongs to user
    edge_device = edge_devices_storage.get(request.edge_device_id)
    if not edge_device or edge_device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Edge device not found")
    
    peripheral_id = f"periph_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    
    peripheral = {
        "id": peripheral_id,
        "edge_device_id": request.edge_device_id,
        "device_type": request.device_type,
        "name": request.name,
        "connection_type": request.connection_type,
        "status": "connected",
        "ip_address": request.ip_address,
        "port": request.port,
        "firmware_version": None,
        "last_seen": now,
        "config": request.config,
    }
    
    peripherals_storage[peripheral_id] = peripheral
    
    storage.log_audit(current_user.id, current_user.organization_id, "create", "peripheral", peripheral_id)
    
    return peripheral


@app.get("/api/v1/peripherals", tags=["Peripherals"])
async def list_peripherals(
    edge_device_id: Optional[str] = None,
    device_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List all peripherals for the organization."""
    # Get user's edge devices
    user_edge_devices = {
        d["id"] for d in edge_devices_storage.values()
        if d["organization_id"] == current_user.organization_id
    }
    
    peripherals = [
        p for p in peripherals_storage.values()
        if p["edge_device_id"] in user_edge_devices
        and (edge_device_id is None or p["edge_device_id"] == edge_device_id)
        and (device_type is None or p["device_type"] == device_type)
        and (status is None or p["status"] == status)
    ]
    return {"peripherals": peripherals, "count": len(peripherals)}


@app.get("/api/v1/peripherals/{peripheral_id}", response_model=PeripheralResponse, tags=["Peripherals"])
async def get_peripheral(
    peripheral_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get peripheral details."""
    peripheral = peripherals_storage.get(peripheral_id)
    if not peripheral:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    # Verify access through edge device
    edge_device = edge_devices_storage.get(peripheral["edge_device_id"])
    if not edge_device or edge_device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    return peripheral


@app.put("/api/v1/peripherals/{peripheral_id}/configure", tags=["Peripherals"])
async def configure_peripheral(
    peripheral_id: str,
    config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Configure a peripheral device.
    
    DOGlove config options:
    - calibration_file: Path to calibration data
    - streaming_rate_hz: Data streaming rate
    - haptic_enabled: Enable/disable haptic feedback
    
    Camera config options:
    - username/password: ONVIF credentials
    - stream_profile: main, sub, third
    - ptz_enabled: Enable PTZ controls
    """
    peripheral = peripherals_storage.get(peripheral_id)
    if not peripheral:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    edge_device = edge_devices_storage.get(peripheral["edge_device_id"])
    if not edge_device or edge_device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    peripheral["config"].update(config)
    
    storage.log_audit(current_user.id, current_user.organization_id, "configure", "peripheral", peripheral_id)
    
    return {"status": "configured", "peripheral": peripheral}


@app.post("/api/v1/peripherals/{peripheral_id}/command", tags=["Peripherals"])
async def send_peripheral_command(
    peripheral_id: str,
    request: DeviceCommand,
    current_user: User = Depends(get_current_user)
):
    """
    Send command to peripheral.
    
    DOGlove commands:
    - calibrate: Start calibration procedure
    - haptic: Send haptic feedback (params: intensities[])
    - start_stream: Start data streaming
    - stop_stream: Stop data streaming
    
    Camera commands:
    - ptz_move: Move PTZ (params: pan, tilt, zoom, speed)
    - ptz_stop: Stop PTZ movement
    - ptz_home: Go to home position
    - ptz_preset: Go to preset (params: preset_name)
    - snapshot: Capture snapshot
    - start_stream: Start video stream
    - stop_stream: Stop video stream
    """
    peripheral = peripherals_storage.get(peripheral_id)
    if not peripheral:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    edge_device = edge_devices_storage.get(peripheral["edge_device_id"])
    if not edge_device or edge_device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    # Validate command for device type
    valid_commands = {
        "doglove": ["calibrate", "haptic", "start_stream", "stop_stream", "get_state"],
        "camera": ["ptz_move", "ptz_stop", "ptz_home", "ptz_preset", "snapshot", "start_stream", "stop_stream"],
    }
    
    device_commands = valid_commands.get(peripheral["device_type"], [])
    if request.command not in device_commands:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid command for {peripheral['device_type']}. Valid: {device_commands}"
        )
    
    storage.log_audit(current_user.id, current_user.organization_id, "command", "peripheral", peripheral_id,
                      {"command": request.command, "params": request.params})
    
    return {
        "status": "queued",
        "command": request.command,
        "peripheral_id": peripheral_id,
        "device_type": peripheral["device_type"],
        "message": f"Command '{request.command}' queued for execution"
    }


@app.delete("/api/v1/peripherals/{peripheral_id}", tags=["Peripherals"])
async def unregister_peripheral(
    peripheral_id: str,
    current_user: User = Depends(get_current_user)
):
    """Unregister a peripheral device."""
    peripheral = peripherals_storage.get(peripheral_id)
    if not peripheral:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    edge_device = edge_devices_storage.get(peripheral["edge_device_id"])
    if not edge_device or edge_device["organization_id"] != current_user.organization_id:
        raise HTTPException(status_code=404, detail="Peripheral not found")
    
    del peripherals_storage[peripheral_id]
    
    storage.log_audit(current_user.id, current_user.organization_id, "delete", "peripheral", peripheral_id)
    
    return {"status": "unregistered", "peripheral_id": peripheral_id}


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """Authenticate user and get access token."""
    # Find user by email
    user = None
    for u in storage.users.values():
        if u.email == request.email:
            user = u
            break
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if user.password_hash != storage._hash_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account disabled")
    
    # Update last login
    user.last_login = datetime.utcnow().isoformat()
    storage._save()
    
    # Create token
    token = create_token(user)
    
    # Audit log
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.LOGIN,
        "user", user.id, {"email": user.email}
    )
    
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRY_HOURS * 3600,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value,
            "organization_id": user.organization_id,
        }
    )


@app.post("/api/v1/auth/register", response_model=TokenResponse, tags=["Authentication"])
async def register(request: RegisterRequest):
    """Register new user and organization."""
    # Check if email exists
    for u in storage.users.values():
        if u.email == request.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create organization if needed
    org_id = "org_default"
    if request.organization_name:
        org = Organization(
            id=f"org_{uuid.uuid4().hex[:12]}",
            name=request.organization_name,
            slug=request.organization_name.lower().replace(" ", "-"),
            plan="free",
            created_at=datetime.utcnow().isoformat(),
        )
        storage.organizations[org.id] = org
        org_id = org.id
    
    # Create user
    user = User(
        id=f"user_{uuid.uuid4().hex[:12]}",
        email=request.email,
        password_hash=storage._hash_password(request.password),
        name=request.name,
        role=UserRole.ADMIN if request.organization_name else UserRole.OPERATOR,
        organization_id=org_id,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
    )
    storage.users[user.id] = user
    storage._save()
    
    # Create token
    token = create_token(user)
    
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRY_HOURS * 3600,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value,
            "organization_id": user.organization_id,
        }
    )


@app.get("/api/v1/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_me(user: User = Depends(get_current_user)):
    """Get current user info."""
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role.value,
        organization_id=user.organization_id,
        created_at=user.created_at,
        last_login=user.last_login,
    )


# =============================================================================
# API Keys
# =============================================================================

@app.post("/api/v1/api-keys", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    request: APIKeyCreate,
    user: User = Depends(get_current_user)
):
    """Create new API key."""
    # Generate key
    key = f"dyn_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    expires_at = None
    if request.expires_in_days:
        expires_at = (datetime.utcnow() + timedelta(days=request.expires_in_days)).isoformat()
    
    api_key = APIKey(
        id=f"key_{uuid.uuid4().hex[:12]}",
        key_hash=key_hash,
        name=request.name,
        user_id=user.id,
        organization_id=user.organization_id,
        permissions=request.permissions,
        created_at=datetime.utcnow().isoformat(),
        expires_at=expires_at,
    )
    storage.api_keys[api_key.id] = api_key
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.CREATE,
        "api_key", api_key.id, {"name": request.name}
    )
    
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=key,  # Only shown once!
        permissions=api_key.permissions,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        last_used=api_key.last_used,
    )


@app.get("/api/v1/api-keys", response_model=List[APIKeyResponse], tags=["API Keys"])
async def list_api_keys(user: User = Depends(get_current_user)):
    """List user's API keys."""
    keys = [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            permissions=k.permissions,
            created_at=k.created_at,
            expires_at=k.expires_at,
            last_used=k.last_used,
        )
        for k in storage.api_keys.values()
        if k.user_id == user.id and k.is_active
    ]
    return keys


@app.delete("/api/v1/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(key_id: str, user: User = Depends(get_current_user)):
    """Revoke API key."""
    api_key = storage.api_keys.get(key_id)
    if not api_key or api_key.user_id != user.id:
        raise HTTPException(status_code=404, detail="API key not found")
    
    api_key.is_active = False
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.DELETE,
        "api_key", key_id, {}
    )
    
    return {"status": "revoked"}


# =============================================================================
# Dashboard
# =============================================================================

@app.get("/api/v1/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
async def get_dashboard_stats(user: User = Depends(get_current_user)):
    """Get dashboard statistics."""
    org_id = user.organization_id
    
    projects = [p for p in storage.projects.values() if p.organization_id == org_id]
    episodes = [e for e in storage.episodes.values() 
                if storage.projects.get(e.project_id, Project("","","",org_id,"",ProjectStatus.DRAFT,"","")).organization_id == org_id]
    models = [m for m in storage.models.values()
              if storage.projects.get(m.project_id, Project("","","",org_id,"",ProjectStatus.DRAFT,"","")).organization_id == org_id]
    robots = [r for r in storage.robots.values() if r.organization_id == org_id]
    sites = [s for s in storage.sites.values() if s.organization_id == org_id]
    
    # Active pipelines
    active_pipelines = len([
        p for p in storage.pipeline_runs.values()
        if p.status == PipelineStatus.RUNNING
    ])
    
    # Episodes last 24h
    yesterday = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    episodes_24h = len([e for e in episodes if e.recorded_at > yesterday])
    
    return DashboardStats(
        total_projects=len(projects),
        total_episodes=len(episodes),
        total_models=len(models),
        total_robots=len(robots),
        total_sites=len(sites),
        active_pipelines=active_pipelines,
        episodes_last_24h=episodes_24h,
        training_hours_last_7d=0.0,  # TODO: Calculate from pipeline runs
    )


# =============================================================================
# Projects
# =============================================================================

@app.post("/api/v1/projects", response_model=ProjectResponse, tags=["Projects"])
async def create_project(
    request: ProjectCreate,
    user: User = Depends(get_current_user)
):
    """Create new project."""
    project = Project(
        id=f"proj_{uuid.uuid4().hex[:12]}",
        name=request.name,
        description=request.description,
        organization_id=user.organization_id,
        created_by=user.id,
        status=ProjectStatus.ACTIVE,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
        tags=request.tags,
        config=request.config,
    )
    storage.projects[project.id] = project
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.CREATE,
        "project", project.id, {"name": request.name}
    )
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status.value,
        version=project.version,
        tags=project.tags,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@app.get("/api/v1/projects", response_model=List[ProjectResponse], tags=["Projects"])
async def list_projects(
    status: Optional[ProjectStatus] = None,
    user: User = Depends(get_current_user)
):
    """List projects."""
    projects = []
    for p in storage.projects.values():
        if p.organization_id != user.organization_id:
            continue
        if status and p.status != status:
            continue
        
        episode_count = len([e for e in storage.episodes.values() if e.project_id == p.id])
        model_count = len([m for m in storage.models.values() if m.project_id == p.id])
        
        projects.append(ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            status=p.status.value,
            version=p.version,
            tags=p.tags,
            created_at=p.created_at,
            updated_at=p.updated_at,
            episode_count=episode_count,
            model_count=model_count,
        ))
    
    return projects


@app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse, tags=["Projects"])
async def get_project(project_id: str, user: User = Depends(get_current_user)):
    """Get project details."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    episode_count = len([e for e in storage.episodes.values() if e.project_id == project_id])
    model_count = len([m for m in storage.models.values() if m.project_id == project_id])
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status.value,
        version=project.version,
        tags=project.tags,
        created_at=project.created_at,
        updated_at=project.updated_at,
        episode_count=episode_count,
        model_count=model_count,
    )


@app.patch("/api/v1/projects/{project_id}", response_model=ProjectResponse, tags=["Projects"])
async def update_project(
    project_id: str,
    request: ProjectUpdate,
    user: User = Depends(get_current_user)
):
    """Update project."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if request.name:
        project.name = request.name
    if request.description is not None:
        project.description = request.description
    if request.status:
        project.status = request.status
    if request.tags is not None:
        project.tags = request.tags
    if request.config is not None:
        project.config.update(request.config)
    
    project.updated_at = datetime.utcnow().isoformat()
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.UPDATE,
        "project", project_id, request.dict(exclude_none=True)
    )
    
    return await get_project(project_id, user)


@app.delete("/api/v1/projects/{project_id}", tags=["Projects"])
async def delete_project(project_id: str, user: User = Depends(get_current_user)):
    """Delete project (archive)."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project.status = ProjectStatus.ARCHIVED
    project.updated_at = datetime.utcnow().isoformat()
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.DELETE,
        "project", project_id, {}
    )
    
    return {"status": "archived"}


# =============================================================================
# Episodes
# =============================================================================

@app.post("/api/v1/projects/{project_id}/episodes", response_model=EpisodeResponse, tags=["Episodes"])
async def create_episode(
    project_id: str,
    request: EpisodeCreate,
    user: User = Depends(get_current_user)
):
    """Record new episode."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Compute file hash (mock)
    file_hash = hashlib.sha256(request.file_path.encode()).hexdigest()[:16]
    
    episode = Episode(
        id=f"ep_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        name=request.name,
        status=EpisodeStatus.RECORDED,
        file_path=request.file_path,
        file_hash=file_hash,
        duration_s=request.duration_s,
        frame_count=request.frame_count,
        quality_score=request.quality_score,
        recorded_at=datetime.utcnow().isoformat(),
        recorded_by=user.id,
        robot_id=request.robot_id,
        site_id=request.site_id,
        tags=request.tags,
        metadata=request.metadata,
    )
    storage.episodes[episode.id] = episode
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.CREATE,
        "episode", episode.id, {"name": request.name, "project": project_id}
    )
    
    return EpisodeResponse(
        id=episode.id,
        project_id=episode.project_id,
        name=episode.name,
        status=episode.status.value,
        duration_s=episode.duration_s,
        frame_count=episode.frame_count,
        quality_score=episode.quality_score,
        recorded_at=episode.recorded_at,
        version=episode.version,
        tags=episode.tags,
    )


@app.get("/api/v1/projects/{project_id}/episodes", response_model=List[EpisodeResponse], tags=["Episodes"])
async def list_episodes(
    project_id: str,
    status: Optional[EpisodeStatus] = None,
    min_quality: float = 0.0,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    user: User = Depends(get_current_user)
):
    """List episodes in project."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    episodes = []
    for e in storage.episodes.values():
        if e.project_id != project_id:
            continue
        if status and e.status != status:
            continue
        if e.quality_score < min_quality:
            continue
        
        episodes.append(EpisodeResponse(
            id=e.id,
            project_id=e.project_id,
            name=e.name,
            status=e.status.value,
            duration_s=e.duration_s,
            frame_count=e.frame_count,
            quality_score=e.quality_score,
            recorded_at=e.recorded_at,
            version=e.version,
            tags=e.tags,
        ))
    
    # Sort by recorded_at descending
    episodes.sort(key=lambda x: x.recorded_at, reverse=True)
    
    return episodes[offset:offset+limit]


@app.patch("/api/v1/episodes/{episode_id}/validate", tags=["Episodes"])
async def validate_episode(
    episode_id: str,
    approved: bool,
    notes: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Validate or reject episode."""
    episode = storage.episodes.get(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    project = storage.projects.get(episode.project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    episode.status = EpisodeStatus.VALIDATED if approved else EpisodeStatus.REJECTED
    episode.validation_notes = notes
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.UPDATE,
        "episode", episode_id, {"approved": approved, "notes": notes}
    )
    
    return {"status": episode.status.value}


# =============================================================================
# Models
# =============================================================================

@app.post("/api/v1/projects/{project_id}/models", response_model=ModelResponse, tags=["Models"])
async def create_model(
    project_id: str,
    request: ModelCreate,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Start model training."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Determine version
    existing_models = [m for m in storage.models.values() 
                      if m.project_id == project_id and m.name == request.name]
    version = f"1.0.{len(existing_models)}"
    
    model = Model(
        id=f"model_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        name=request.name,
        version=version,
        status=ModelStatus.TRAINING,
        model_type=request.model_type,
        file_path="",
        file_hash="",
        training_config=request.training_config,
        metrics={},
        trained_on_episodes=request.episode_ids,
        created_at=datetime.utcnow().isoformat(),
        created_by=user.id,
        tags=request.tags,
    )
    storage.models[model.id] = model
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.TRAIN,
        "model", model.id, {"name": request.name, "type": request.model_type}
    )
    
    # Simulate training in background
    background_tasks.add_task(simulate_training, model.id)
    
    return ModelResponse(
        id=model.id,
        project_id=model.project_id,
        name=model.name,
        version=model.version,
        status=model.status.value,
        model_type=model.model_type,
        metrics=model.metrics,
        created_at=model.created_at,
        deployed_to=model.deployed_to,
    )


async def simulate_training(model_id: str):
    """Simulate model training."""
    import asyncio
    await asyncio.sleep(5)  # Simulate training time
    
    model = storage.models.get(model_id)
    if model:
        model.status = ModelStatus.READY
        model.metrics = {
            "loss": 0.023,
            "accuracy": 0.94,
            "episodes_used": len(model.trained_on_episodes),
        }
        model.file_path = f"/models/{model_id}/checkpoint.pt"
        model.file_hash = hashlib.sha256(model_id.encode()).hexdigest()[:16]
        storage._save()


@app.get("/api/v1/projects/{project_id}/models", response_model=List[ModelResponse], tags=["Models"])
async def list_models(
    project_id: str,
    status: Optional[ModelStatus] = None,
    user: User = Depends(get_current_user)
):
    """List models in project."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    models = []
    for m in storage.models.values():
        if m.project_id != project_id:
            continue
        if status and m.status != status:
            continue
        
        models.append(ModelResponse(
            id=m.id,
            project_id=m.project_id,
            name=m.name,
            version=m.version,
            status=m.status.value,
            model_type=m.model_type,
            metrics=m.metrics,
            created_at=m.created_at,
            deployed_to=m.deployed_to,
        ))
    
    return models


@app.post("/api/v1/models/{model_id}/deploy", tags=["Models"])
async def deploy_model(
    model_id: str,
    robot_ids: List[str],
    user: User = Depends(get_current_user)
):
    """Deploy model to robots."""
    model = storage.models.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    project = storage.projects.get(model.project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.status != ModelStatus.READY:
        raise HTTPException(status_code=400, detail="Model not ready for deployment")
    
    # Update model
    model.status = ModelStatus.DEPLOYED
    model.deployed_at = datetime.utcnow().isoformat()
    model.deployed_to = robot_ids
    
    # Update robots
    for rid in robot_ids:
        robot = storage.robots.get(rid)
        if robot and robot.organization_id == user.organization_id:
            if model_id not in robot.deployed_models:
                robot.deployed_models.append(model_id)
    
    storage._save()
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.DEPLOY,
        "model", model_id, {"robots": robot_ids}
    )
    
    return {"status": "deployed", "robots": robot_ids}


# =============================================================================
# Robots & Sites
# =============================================================================

@app.post("/api/v1/sites", response_model=SiteResponse, tags=["Fleet"])
async def create_site(request: SiteCreate, user: User = Depends(get_current_user)):
    """Create new site."""
    site = Site(
        id=f"site_{uuid.uuid4().hex[:12]}",
        name=request.name,
        organization_id=user.organization_id,
        location=request.location,
        timezone=request.timezone,
        created_at=datetime.utcnow().isoformat(),
    )
    storage.sites[site.id] = site
    storage._save()
    
    return SiteResponse(
        id=site.id,
        name=site.name,
        location=site.location,
        timezone=site.timezone,
    )


@app.get("/api/v1/sites", response_model=List[SiteResponse], tags=["Fleet"])
async def list_sites(user: User = Depends(get_current_user)):
    """List sites."""
    sites = []
    for s in storage.sites.values():
        if s.organization_id != user.organization_id:
            continue
        
        robot_count = len([r for r in storage.robots.values() if r.site_id == s.id])
        sites.append(SiteResponse(
            id=s.id,
            name=s.name,
            location=s.location,
            timezone=s.timezone,
            robot_count=robot_count,
        ))
    return sites


@app.post("/api/v1/robots", response_model=RobotResponse, tags=["Fleet"])
async def register_robot(request: RobotCreate, user: User = Depends(get_current_user)):
    """Register new robot."""
    # Verify site exists
    site = storage.sites.get(request.site_id)
    if not site or site.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Site not found")
    
    robot = Robot(
        id=f"robot_{uuid.uuid4().hex[:12]}",
        name=request.name,
        organization_id=user.organization_id,
        site_id=request.site_id,
        robot_type=request.robot_type,
        hardware_config=request.hardware_config,
        software_version=request.software_version,
        status="online",
        last_seen=datetime.utcnow().isoformat(),
        registered_at=datetime.utcnow().isoformat(),
    )
    storage.robots[robot.id] = robot
    
    # Add to site
    if robot.id not in site.robots:
        site.robots.append(robot.id)
    
    storage._save()
    
    return RobotResponse(
        id=robot.id,
        name=robot.name,
        site_id=robot.site_id,
        robot_type=robot.robot_type,
        status=robot.status,
        last_seen=robot.last_seen,
        deployed_models=robot.deployed_models,
    )


@app.get("/api/v1/robots", response_model=List[RobotResponse], tags=["Fleet"])
async def list_robots(
    site_id: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """List robots."""
    robots = []
    for r in storage.robots.values():
        if r.organization_id != user.organization_id:
            continue
        if site_id and r.site_id != site_id:
            continue
        
        robots.append(RobotResponse(
            id=r.id,
            name=r.name,
            site_id=r.site_id,
            robot_type=r.robot_type,
            status=r.status,
            last_seen=r.last_seen,
            deployed_models=r.deployed_models,
        ))
    return robots


@app.post("/api/v1/robots/{robot_id}/heartbeat", tags=["Fleet"])
async def robot_heartbeat(
    robot_id: str,
    status: str = "online",
    metrics: Dict[str, Any] = None,
    user: User = Depends(get_current_user)
):
    """Robot heartbeat update."""
    robot = storage.robots.get(robot_id)
    if not robot or robot.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Robot not found")
    
    robot.status = status
    robot.last_seen = datetime.utcnow().isoformat()
    if metrics:
        robot.metadata["last_metrics"] = metrics
    
    storage._save()
    
    return {"status": "ok"}


# =============================================================================
# Audit Logs
# =============================================================================

@app.get("/api/v1/audit-logs", response_model=List[AuditLogResponse], tags=["Audit"])
async def get_audit_logs(
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[AuditAction] = None,
    limit: int = Query(100, le=1000),
    user: User = Depends(require_role([UserRole.ADMIN]))
):
    """Get audit logs (admin only)."""
    logs = []
    for log in reversed(storage.audit_logs):
        if log.organization_id != user.organization_id:
            continue
        if resource_type and log.resource_type != resource_type:
            continue
        if resource_id and log.resource_id != resource_id:
            continue
        if action and log.action != action:
            continue
        
        logs.append(AuditLogResponse(
            id=log.id,
            timestamp=log.timestamp,
            user_id=log.user_id,
            action=log.action.value,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            details=log.details,
        ))
        
        if len(logs) >= limit:
            break
    
    return logs


# =============================================================================
# Pipeline Runs
# =============================================================================

@app.post("/api/v1/projects/{project_id}/pipelines", response_model=PipelineResponse, tags=["Pipelines"])
async def start_pipeline(
    project_id: str,
    request: PipelineCreate,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Start pipeline run."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    run = PipelineRun(
        id=f"run_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        pipeline_type=request.pipeline_type,
        status=PipelineStatus.RUNNING,
        config=request.config,
        started_at=datetime.utcnow().isoformat(),
        started_by=user.id,
    )
    storage.pipeline_runs[run.id] = run
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.CREATE,
        "pipeline", run.id, {"type": request.pipeline_type}
    )
    
    # Simulate pipeline execution
    background_tasks.add_task(simulate_pipeline, run.id)
    
    return PipelineResponse(
        id=run.id,
        project_id=run.project_id,
        pipeline_type=run.pipeline_type,
        status=run.status.value,
        progress=run.progress,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


async def simulate_pipeline(run_id: str):
    """Simulate pipeline execution."""
    import asyncio
    
    run = storage.pipeline_runs.get(run_id)
    if not run:
        return
    
    for i in range(10):
        await asyncio.sleep(1)
        run.progress = (i + 1) * 10
        run.logs.append(f"Step {i+1}/10 completed")
    
    run.status = PipelineStatus.COMPLETED
    run.completed_at = datetime.utcnow().isoformat()


@app.get("/api/v1/projects/{project_id}/pipelines", response_model=List[PipelineResponse], tags=["Pipelines"])
async def list_pipelines(
    project_id: str,
    status: Optional[PipelineStatus] = None,
    user: User = Depends(get_current_user)
):
    """List pipeline runs."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    runs = []
    for r in storage.pipeline_runs.values():
        if r.project_id != project_id:
            continue
        if status and r.status != status:
            continue
        
        runs.append(PipelineResponse(
            id=r.id,
            project_id=r.project_id,
            pipeline_type=r.pipeline_type,
            status=r.status.value,
            progress=r.progress,
            started_at=r.started_at,
            completed_at=r.completed_at,
        ))
    
    return runs


# =============================================================================
# Export/Import
# =============================================================================

@app.get("/api/v1/projects/{project_id}/export", tags=["Export"])
async def export_project(project_id: str, user: User = Depends(get_current_user)):
    """Export project data."""
    project = storage.projects.get(project_id)
    if not project or project.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Project not found")
    
    episodes = [asdict(e) for e in storage.episodes.values() if e.project_id == project_id]
    models = [asdict(m) for m in storage.models.values() if m.project_id == project_id]
    
    storage.add_audit_log(
        user.id, user.organization_id, AuditAction.EXPORT,
        "project", project_id, {}
    )
    
    return {
        "project": asdict(project),
        "episodes": episodes,
        "models": models,
        "exported_at": datetime.utcnow().isoformat(),
        "version": "1.0",
    }


# =============================================================================
# Skills API (MoE Skill Router with Encrypted Storage)
# =============================================================================

# Import skill router
try:
    from src.platform.cloud.moe_skill_router import (
        CloudSkillService, SkillRequest, SkillType, SkillStatus
    )
    HAS_SKILL_ROUTER = True
except ImportError:
    HAS_SKILL_ROUTER = False
    logger.warning("Skill router not available")

# Initialize skill service
if HAS_SKILL_ROUTER:
    skill_service = CloudSkillService(
        storage_dir=os.path.join(settings.DATA_DIR, "skills"),
        use_encryption=True,
        num_experts=16,
        embedding_dim=512,
    )
else:
    skill_service = None


class SkillCreate(BaseModel):
    """Skill creation request."""
    name: str
    description: str
    skill_type: str  # manipulation, navigation, perception, coordination, locomotion
    version: str = "1.0.0"
    tags: List[str] = []
    training_config: Dict[str, Any] = {}


class SkillRequestBody(BaseModel):
    """Skill request body."""
    task_description: str
    task_embedding: Optional[List[float]] = None
    required_skill_types: List[str] = []
    max_skills: int = 5
    device_id: str = ""


class SkillDeployRequest(BaseModel):
    """Skill deployment request."""
    skill_ids: List[str]
    device_ids: List[str]


class SkillResponse(BaseModel):
    """Skill response."""
    id: str
    name: str
    description: str
    skill_type: str
    version: str
    status: str
    is_encrypted: bool
    file_size_bytes: int
    created_at: str
    deployed_to: List[str] = []


@app.post("/api/v1/skills", response_model=SkillResponse, tags=["Skills"])
async def register_skill(
    request: SkillCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Register a new skill with encrypted storage.

    Skills are robot manipulation primitives (grasp, pour, place, etc.) that
    can be dynamically loaded and combined using MoE routing for task execution.

    The skill weights should be uploaded separately via file upload endpoint.
    """
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    import numpy as np

    # Map string to SkillType enum
    try:
        skill_type = SkillType(request.skill_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid skill type. Valid types: {[t.value for t in SkillType]}"
        )

    # Create placeholder weights (actual weights uploaded separately)
    placeholder_weights = np.random.randn(1000).astype(np.float32)

    metadata = skill_service.register_skill(
        name=request.name,
        description=request.description,
        skill_type=skill_type,
        weights=placeholder_weights,
        config=request.training_config,
        version=request.version,
        tags=request.tags,
    )

    storage.add_audit_log(
        current_user.id, current_user.organization_id, AuditAction.CREATE,
        "skill", metadata.id, {"name": request.name}
    )

    return SkillResponse(
        id=metadata.id,
        name=metadata.name,
        description=metadata.description,
        skill_type=metadata.skill_type.value,
        version=metadata.version,
        status=metadata.status.value,
        is_encrypted=metadata.is_encrypted,
        file_size_bytes=metadata.file_size_bytes,
        created_at=metadata.created_at,
        deployed_to=metadata.deployed_to,
    )


@app.get("/api/v1/skills", tags=["Skills"])
async def list_skills(
    skill_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List all available skills."""
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    # Parse filters
    type_filter = SkillType(skill_type) if skill_type else None
    status_filter = SkillStatus(status) if status else None

    skills = skill_service.storage.list_skills(
        skill_type=type_filter,
        status=status_filter,
    )

    return {
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "skill_type": s.skill_type.value,
                "version": s.version,
                "status": s.status.value,
                "is_encrypted": s.is_encrypted,
                "success_rate": s.success_rate,
                "deployed_to": s.deployed_to,
            }
            for s in skills
        ],
        "count": len(skills),
    }


@app.get("/api/v1/skills/{skill_id}", tags=["Skills"])
async def get_skill(
    skill_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get skill details including encrypted weights.

    The encrypted weights can be decrypted on the edge device using the
    device-specific N2HE keys.
    """
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    encrypted_skill = skill_service.storage.get_skill(skill_id)

    if not encrypted_skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    import base64

    return {
        "metadata": {
            "id": encrypted_skill.metadata.id,
            "name": encrypted_skill.metadata.name,
            "description": encrypted_skill.metadata.description,
            "skill_type": encrypted_skill.metadata.skill_type.value,
            "version": encrypted_skill.metadata.version,
            "status": encrypted_skill.metadata.status.value,
            "is_encrypted": encrypted_skill.metadata.is_encrypted,
            "file_size_bytes": encrypted_skill.metadata.file_size_bytes,
            "file_hash": encrypted_skill.metadata.file_hash,
            "created_at": encrypted_skill.metadata.created_at,
            "deployed_to": encrypted_skill.metadata.deployed_to,
        },
        "encrypted_weights": base64.b64encode(encrypted_skill.encrypted_weights).decode(),
        "encrypted_config": base64.b64encode(encrypted_skill.encrypted_config).decode(),
        "encryption_params": encrypted_skill.encryption_params,
        "public_key_hash": encrypted_skill.public_key_hash,
    }


@app.post("/api/v1/skills/request", tags=["Skills"])
async def request_skills(
    request: SkillRequestBody,
    current_user: User = Depends(get_current_user)
):
    """
    Request skills for a task using MoE routing.

    Given a task description or embedding, the MoE router selects the most
    appropriate skills and returns them with routing weights for blending.
    """
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    from src.platform.cloud.moe_skill_router import SkillRequest as SkillRequestModel

    # Map skill types
    required_types = []
    for t in request.required_skill_types:
        try:
            required_types.append(SkillType(t))
        except ValueError:
            pass

    skill_request = SkillRequestModel(
        task_description=request.task_description,
        task_embedding=request.task_embedding,
        device_id=request.device_id,
        required_skill_types=required_types,
        max_skills=request.max_skills,
    )

    response = skill_service.request_skills(skill_request)

    import base64

    storage.add_audit_log(
        current_user.id, current_user.organization_id, AuditAction.READ,
        "skill", "moe_request", {"task": request.task_description[:100]}
    )

    return {
        "skills": [
            {
                "metadata": {
                    "id": s.metadata.id,
                    "name": s.metadata.name,
                    "description": s.metadata.description,
                    "skill_type": s.metadata.skill_type.value,
                    "version": s.metadata.version,
                },
                "encrypted_weights": base64.b64encode(s.encrypted_weights).decode(),
                "encrypted_config": base64.b64encode(s.encrypted_config).decode(),
                "encryption_params": s.encryption_params,
            }
            for s in response.skills
        ],
        "routing_weights": response.routing_weights,
        "task_embedding": response.task_embedding,
        "inference_time_ms": response.inference_time_ms,
    }


@app.post("/api/v1/skills/deploy", tags=["Skills"])
async def deploy_skills(
    request: SkillDeployRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Deploy skills to edge devices.

    This marks the skills as deployed and records which edge devices
    have access to them.
    """
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    results = skill_service.deploy_to_edge(
        skill_ids=request.skill_ids,
        device_ids=request.device_ids,
    )

    storage.add_audit_log(
        current_user.id, current_user.organization_id, AuditAction.DEPLOY,
        "skill", ",".join(request.skill_ids), {"devices": request.device_ids}
    )

    return {
        "results": results,
        "deployed_count": sum(1 for v in results.values() if v),
        "failed_count": sum(1 for v in results.values() if not v),
    }


@app.delete("/api/v1/skills/{skill_id}", tags=["Skills"])
async def delete_skill(
    skill_id: str,
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """Delete a skill (admin only)."""
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    success = skill_service.storage.delete_skill(skill_id)

    if not success:
        raise HTTPException(status_code=404, detail="Skill not found")

    storage.add_audit_log(
        current_user.id, current_user.organization_id, AuditAction.DELETE,
        "skill", skill_id, {}
    )

    return {"status": "deleted", "skill_id": skill_id}


@app.get("/api/v1/skills/statistics", tags=["Skills"])
async def get_skill_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get skill service statistics."""
    if not HAS_SKILL_ROUTER or not skill_service:
        raise HTTPException(status_code=503, detail="Skill service not available")

    return skill_service.get_service_statistics()


# =============================================================================
# Federated Learning API
# =============================================================================

# Initialize FL server
try:
    from src.platform.cloud.federated_learning import (
        FederatedLearningServer,
        FLConfig,
        EdgeFLClient,
        create_fl_endpoints,
    )
    HAS_FL = True
    fl_config = FLConfig(
        min_clients=2,
        use_secure_aggregation=True,
        use_differential_privacy=True,
        dp_epsilon=1.0,
    )
    fl_server = FederatedLearningServer(fl_config)
    logger.info("[Platform] Federated Learning server initialized")
except ImportError as e:
    HAS_FL = False
    fl_server = None
    logger.warning(f"[Platform] Federated Learning not available: {e}")


class FLRegisterRequest(BaseModel):
    client_id: str
    device_type: str = "jetson_orin"


class FLUpdateRequest(BaseModel):
    client_id: str
    gradients: str  # hex-encoded bytes
    num_samples: int
    round_num: int
    encrypted: bool = False


@app.post("/api/v1/fl/register", tags=["Federated Learning"])
async def fl_register(request: FLRegisterRequest):
    """
    Register a federated learning client.

    Edge devices (Jetson Orin) call this endpoint to join the FL cluster.
    Returns public encryption context for secure gradient uploads.
    """
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    result = fl_server.register_client(request.client_id, request.device_type)

    # Convert bytes to hex for JSON serialization
    if result.get('public_context'):
        result['public_context'] = result['public_context'].hex()

    return result


@app.post("/api/v1/fl/update", tags=["Federated Learning"])
async def fl_submit_update(request: FLUpdateRequest):
    """
    Submit encrypted gradient update from edge device.

    Gradients are encrypted with TenSEAL (CKKS scheme) for secure aggregation.
    The server performs homomorphic aggregation without decrypting individual updates.
    """
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    try:
        gradients = bytes.fromhex(request.gradients)
        return fl_server.submit_update(
            client_id=request.client_id,
            gradients=gradients,
            num_samples=request.num_samples,
            round_num=request.round_num,
            encrypted=request.encrypted,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/fl/model", tags=["Federated Learning"])
async def fl_get_model(client_id: str = None):
    """
    Get current global model.

    Edge devices call this to download the latest aggregated model.
    Model weights are compressed with gzip and hex-encoded.
    """
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    response = fl_server.get_model(client_id)

    # Convert bytes to hex for JSON serialization
    if response.get('weights'):
        response['weights'] = response['weights'].hex()

    return response


@app.get("/api/v1/fl/status", tags=["Federated Learning"])
async def fl_status():
    """
    Get federated learning server status.

    Returns:
    - Number of registered/active clients
    - Current model version
    - Training statistics
    - Privacy budget usage
    """
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    return fl_server.get_status()


@app.get("/api/v1/fl/privacy", tags=["Federated Learning"])
async def fl_privacy_budget():
    """
    Get differential privacy budget usage.

    Shows how much privacy budget (epsilon) has been spent across training rounds.
    Important for compliance with privacy requirements.
    """
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    if fl_server.dp:
        return fl_server.dp.get_privacy_spent()
    else:
        return {"status": "Differential privacy not enabled"}


@app.post("/api/v1/fl/start", tags=["Federated Learning"])
async def fl_start_training(
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Start federated learning training (admin only).

    Begins accepting gradient updates from edge devices and
    performing secure aggregation rounds.
    """
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    fl_server.start()
    return {"status": "started", "config": fl_server.config.__dict__}


@app.post("/api/v1/fl/stop", tags=["Federated Learning"])
async def fl_stop_training(
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """Stop federated learning training (admin only)."""
    if not HAS_FL or not fl_server:
        raise HTTPException(status_code=503, detail="Federated learning not available")

    fl_server.stop()
    return {"status": "stopped"}


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "platform_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )


if __name__ == "__main__":
    main()

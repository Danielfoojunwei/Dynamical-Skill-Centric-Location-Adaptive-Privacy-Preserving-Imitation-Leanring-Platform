from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import time
import uuid

SQLALCHEMY_DATABASE_URL = "sqlite:///./system.db"

# Check same thread = False is needed for SQLite with FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Device(Base):
    __tablename__ = "devices"

    id = Column(String, primary_key=True, index=True)
    type = Column(String)
    status = Column(String)
    last_seen = Column(Float)
    config = Column(JSON, nullable=True)

class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time)
    event_type = Column(String)
    details = Column(String)

class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time)
    tflops_used = Column(Float)
    memory_used_gb = Column(Float)

class SafetyZone(Base):
    __tablename__ = "safety_zones"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    zone_type = Column(String)  # "KEEP_OUT", "SLOW_DOWN"
    coordinates_json = Column(String)  # JSON list of [x, y] points
    is_active = Column(Boolean, default=True)

class SafetyConfig(Base):
    __tablename__ = "safety_config"
    id = Column(Integer, primary_key=True, index=True)
    human_sensitivity = Column(Float, default=0.8)
    stop_distance_m = Column(Float, default=1.5)
    max_speed_limit = Column(Float, default=2.0)


# =============================================================================
# NEW TABLES FOR SYSTEM INTEGRATOR UX
# =============================================================================

class Deployment(Base):
    """Represents a deployment/site managed by the system integrator."""
    __tablename__ = "deployments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    site_name = Column(String, nullable=False)  # Client site name
    location = Column(String, nullable=True)  # Physical location
    description = Column(Text, nullable=True)
    status = Column(String, default="active")  # active, inactive, maintenance, error
    environment = Column(String, default="production")  # dev, staging, production
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)
    config_version = Column(String, nullable=True)  # Current config version
    platform_version = Column(String, nullable=True)  # Platform version running
    robot_count = Column(Integer, default=1)
    tags = Column(JSON, default=list)  # Tags for filtering
    health_score = Column(Float, default=100.0)  # 0-100 health score
    last_heartbeat = Column(Float, nullable=True)
    contact_email = Column(String, nullable=True)
    notes = Column(Text, nullable=True)


class AuditLog(Base):
    """Audit log for tracking all system changes."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time, index=True)
    deployment_id = Column(String, ForeignKey("deployments.id"), nullable=True)
    user_id = Column(String, default="system")  # Who made the change
    action = Column(String, nullable=False)  # create, update, delete, invoke, etc.
    resource_type = Column(String, nullable=False)  # skill, config, deployment, safety, etc.
    resource_id = Column(String, nullable=True)
    resource_name = Column(String, nullable=True)
    old_value = Column(JSON, nullable=True)  # Previous state
    new_value = Column(JSON, nullable=True)  # New state
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional context


class ConfigVersion(Base):
    """Version history for system configurations."""
    __tablename__ = "config_versions"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(String, ForeignKey("deployments.id"), nullable=True)
    version = Column(String, nullable=False)  # Semantic version
    config_type = Column(String, nullable=False)  # safety, perception, skills, system, etc.
    config_data = Column(JSON, nullable=False)  # The actual configuration
    created_at = Column(Float, default=time.time)
    created_by = Column(String, default="system")
    comment = Column(Text, nullable=True)  # Change description
    is_active = Column(Boolean, default=False)  # Is this the active version?
    parent_version_id = Column(Integer, ForeignKey("config_versions.id"), nullable=True)
    checksum = Column(String, nullable=True)  # For integrity verification


class SkillExecution(Base):
    """Tracks skill invocations and their outcomes."""
    __tablename__ = "skill_executions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time, index=True)
    deployment_id = Column(String, ForeignKey("deployments.id"), nullable=True)
    skill_id = Column(String, nullable=False, index=True)
    skill_name = Column(String, nullable=True)
    skill_version = Column(String, nullable=True)
    task_description = Column(Text, nullable=True)
    execution_mode = Column(String, nullable=True)  # direct, blended, sequential, autonomous
    robot_id = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    confidence = Column(Float, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    safety_status = Column(String, nullable=True)
    blend_weights = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)


class Alert(Base):
    """System alerts and notifications."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, default=time.time, index=True)
    deployment_id = Column(String, ForeignKey("deployments.id"), nullable=True)
    alert_type = Column(String, nullable=False)  # error, warning, info, critical
    category = Column(String, nullable=False)  # safety, performance, skill, system, etc.
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String, nullable=True)
    acknowledged_at = Column(Float, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(Float, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)


class SystemPreset(Base):
    """Saved system configuration presets for quick switching."""
    __tablename__ = "system_presets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    preset_type = Column(String, default="full")  # full, partial, template
    config_data = Column(JSON, nullable=False)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)
    created_by = Column(String, default="system")
    is_default = Column(Boolean, default=False)
    tags = Column(JSON, default=list)


def init_db():
    Base.metadata.create_all(bind=engine)

    # Seed default config if empty
    db = SessionLocal()
    if not db.query(SafetyConfig).first():
        db.add(SafetyConfig(id=1))
        db.commit()
    db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

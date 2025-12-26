"""
Integrator API - Endpoints for Robot System Integrators

This module provides APIs for:
- Deployment/site management
- Audit logging
- Configuration versioning
- Skill execution tracking
- Alerts and notifications
- System presets

These features are designed for system integrators managing robots at client sites.

@version 0.7.0
"""

import time
import hashlib
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .database import (
    get_db, Deployment, AuditLog, ConfigVersion,
    SkillExecution, Alert, SystemPreset
)

router = APIRouter(prefix="/api/v1/integrator", tags=["Integrator"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DeploymentCreate(BaseModel):
    name: str
    site_name: str
    location: Optional[str] = None
    description: Optional[str] = None
    environment: str = "production"
    robot_count: int = 1
    tags: List[str] = []
    contact_email: Optional[str] = None
    notes: Optional[str] = None


class DeploymentUpdate(BaseModel):
    name: Optional[str] = None
    site_name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    environment: Optional[str] = None
    robot_count: Optional[int] = None
    tags: Optional[List[str]] = None
    contact_email: Optional[str] = None
    notes: Optional[str] = None


class ConfigVersionCreate(BaseModel):
    deployment_id: Optional[str] = None
    config_type: str
    config_data: Dict[str, Any]
    comment: Optional[str] = None


class AlertCreate(BaseModel):
    deployment_id: Optional[str] = None
    alert_type: str  # error, warning, info, critical
    category: str
    title: str
    message: str
    metadata: Optional[Dict[str, Any]] = None


class PresetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    preset_type: str = "full"
    config_data: Dict[str, Any]
    tags: List[str] = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_audit(
    db: Session,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    old_value: Optional[Dict] = None,
    new_value: Optional[Dict] = None,
    deployment_id: Optional[str] = None,
    user_id: str = "system",
    request: Optional[Request] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    metadata: Optional[Dict] = None,
):
    """Create an audit log entry."""
    log = AuditLog(
        deployment_id=deployment_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
        old_value=old_value,
        new_value=new_value,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None,
        success=success,
        error_message=error_message,
        metadata=metadata,
    )
    db.add(log)
    db.commit()
    return log


def generate_config_checksum(config_data: Dict) -> str:
    """Generate a checksum for configuration data."""
    config_str = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def generate_version_number(db: Session, config_type: str, deployment_id: Optional[str] = None) -> str:
    """Generate the next version number for a config type."""
    query = db.query(ConfigVersion).filter(ConfigVersion.config_type == config_type)
    if deployment_id:
        query = query.filter(ConfigVersion.deployment_id == deployment_id)

    count = query.count()
    return f"1.{count}.0"


# =============================================================================
# DEPLOYMENT MANAGEMENT
# =============================================================================

@router.get("/deployments")
async def list_deployments(
    status: Optional[str] = None,
    environment: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all deployments with optional filtering."""
    query = db.query(Deployment).order_by(desc(Deployment.updated_at))

    if status:
        query = query.filter(Deployment.status == status)
    if environment:
        query = query.filter(Deployment.environment == environment)

    deployments = query.limit(limit).all()
    return {
        "deployments": [
            {
                "id": d.id,
                "name": d.name,
                "site_name": d.site_name,
                "location": d.location,
                "description": d.description,
                "status": d.status,
                "environment": d.environment,
                "robot_count": d.robot_count,
                "tags": d.tags or [],
                "health_score": d.health_score,
                "last_heartbeat": d.last_heartbeat,
                "platform_version": d.platform_version,
                "config_version": d.config_version,
                "created_at": d.created_at,
                "updated_at": d.updated_at,
            }
            for d in deployments
        ],
        "total": len(deployments)
    }


@router.post("/deployments")
async def create_deployment(
    deployment: DeploymentCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a new deployment."""
    from src.version import __version__

    db_deployment = Deployment(
        name=deployment.name,
        site_name=deployment.site_name,
        location=deployment.location,
        description=deployment.description,
        environment=deployment.environment,
        robot_count=deployment.robot_count,
        tags=deployment.tags,
        contact_email=deployment.contact_email,
        notes=deployment.notes,
        platform_version=__version__,
    )
    db.add(db_deployment)
    db.commit()
    db.refresh(db_deployment)

    # Log the action
    log_audit(
        db, "create", "deployment",
        resource_id=db_deployment.id,
        resource_name=deployment.name,
        new_value=deployment.dict(),
        request=request
    )

    return {
        "success": True,
        "deployment_id": db_deployment.id,
        "name": db_deployment.name
    }


@router.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str, db: Session = Depends(get_db)):
    """Get a specific deployment."""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return {
        "id": deployment.id,
        "name": deployment.name,
        "site_name": deployment.site_name,
        "location": deployment.location,
        "description": deployment.description,
        "status": deployment.status,
        "environment": deployment.environment,
        "robot_count": deployment.robot_count,
        "tags": deployment.tags or [],
        "health_score": deployment.health_score,
        "last_heartbeat": deployment.last_heartbeat,
        "platform_version": deployment.platform_version,
        "config_version": deployment.config_version,
        "contact_email": deployment.contact_email,
        "notes": deployment.notes,
        "created_at": deployment.created_at,
        "updated_at": deployment.updated_at,
    }


@router.patch("/deployments/{deployment_id}")
async def update_deployment(
    deployment_id: str,
    update: DeploymentUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Update a deployment."""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    old_value = {
        "name": deployment.name,
        "site_name": deployment.site_name,
        "status": deployment.status,
        "environment": deployment.environment,
    }

    update_data = update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(deployment, key, value)

    deployment.updated_at = time.time()
    db.commit()

    log_audit(
        db, "update", "deployment",
        resource_id=deployment_id,
        resource_name=deployment.name,
        old_value=old_value,
        new_value=update_data,
        deployment_id=deployment_id,
        request=request
    )

    return {"success": True, "deployment_id": deployment_id}


@router.delete("/deployments/{deployment_id}")
async def delete_deployment(
    deployment_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Delete a deployment."""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    name = deployment.name
    db.delete(deployment)
    db.commit()

    log_audit(
        db, "delete", "deployment",
        resource_id=deployment_id,
        resource_name=name,
        request=request
    )

    return {"success": True}


@router.post("/deployments/{deployment_id}/heartbeat")
async def deployment_heartbeat(
    deployment_id: str,
    health_score: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Update deployment heartbeat and optional health score."""
    deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment.last_heartbeat = time.time()
    if health_score is not None:
        deployment.health_score = max(0, min(100, health_score))

    db.commit()
    return {"success": True, "last_heartbeat": deployment.last_heartbeat}


# =============================================================================
# AUDIT LOGS
# =============================================================================

@router.get("/audit-logs")
async def list_audit_logs(
    deployment_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    action: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List audit logs with filtering."""
    query = db.query(AuditLog).order_by(desc(AuditLog.timestamp))

    if deployment_id:
        query = query.filter(AuditLog.deployment_id == deployment_id)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    if action:
        query = query.filter(AuditLog.action == action)
    if start_time:
        query = query.filter(AuditLog.timestamp >= start_time)
    if end_time:
        query = query.filter(AuditLog.timestamp <= end_time)

    total = query.count()
    logs = query.offset(offset).limit(limit).all()

    return {
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp,
                "deployment_id": log.deployment_id,
                "user_id": log.user_id,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "resource_name": log.resource_name,
                "old_value": log.old_value,
                "new_value": log.new_value,
                "success": log.success,
                "error_message": log.error_message,
            }
            for log in logs
        ],
        "total": total,
        "offset": offset,
        "limit": limit
    }


@router.get("/audit-logs/summary")
async def get_audit_summary(
    deployment_id: Optional[str] = None,
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get summary of audit activity."""
    cutoff = time.time() - (hours * 3600)
    query = db.query(AuditLog).filter(AuditLog.timestamp >= cutoff)

    if deployment_id:
        query = query.filter(AuditLog.deployment_id == deployment_id)

    logs = query.all()

    # Summarize by action and resource type
    by_action = {}
    by_resource = {}
    by_user = {}

    for log in logs:
        by_action[log.action] = by_action.get(log.action, 0) + 1
        by_resource[log.resource_type] = by_resource.get(log.resource_type, 0) + 1
        by_user[log.user_id] = by_user.get(log.user_id, 0) + 1

    return {
        "period_hours": hours,
        "total_actions": len(logs),
        "by_action": by_action,
        "by_resource": by_resource,
        "by_user": by_user,
        "success_rate": sum(1 for l in logs if l.success) / len(logs) if logs else 1.0
    }


# =============================================================================
# CONFIGURATION VERSIONING
# =============================================================================

@router.get("/config-versions")
async def list_config_versions(
    config_type: Optional[str] = None,
    deployment_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List configuration versions."""
    query = db.query(ConfigVersion).order_by(desc(ConfigVersion.created_at))

    if config_type:
        query = query.filter(ConfigVersion.config_type == config_type)
    if deployment_id:
        query = query.filter(ConfigVersion.deployment_id == deployment_id)

    versions = query.limit(limit).all()

    return {
        "versions": [
            {
                "id": v.id,
                "version": v.version,
                "config_type": v.config_type,
                "deployment_id": v.deployment_id,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "comment": v.comment,
                "is_active": v.is_active,
                "checksum": v.checksum,
            }
            for v in versions
        ],
        "total": len(versions)
    }


@router.post("/config-versions")
async def create_config_version(
    config: ConfigVersionCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a new configuration version."""
    version_num = generate_version_number(db, config.config_type, config.deployment_id)
    checksum = generate_config_checksum(config.config_data)

    # Deactivate previous active versions of this type
    db.query(ConfigVersion).filter(
        ConfigVersion.config_type == config.config_type,
        ConfigVersion.deployment_id == config.deployment_id,
        ConfigVersion.is_active == True
    ).update({"is_active": False})

    db_version = ConfigVersion(
        deployment_id=config.deployment_id,
        version=version_num,
        config_type=config.config_type,
        config_data=config.config_data,
        comment=config.comment,
        is_active=True,
        checksum=checksum,
    )
    db.add(db_version)
    db.commit()
    db.refresh(db_version)

    log_audit(
        db, "create", "config_version",
        resource_id=str(db_version.id),
        resource_name=f"{config.config_type}@{version_num}",
        new_value={"version": version_num, "config_type": config.config_type},
        deployment_id=config.deployment_id,
        request=request
    )

    return {
        "success": True,
        "version_id": db_version.id,
        "version": version_num,
        "checksum": checksum
    }


@router.get("/config-versions/{version_id}")
async def get_config_version(version_id: int, db: Session = Depends(get_db)):
    """Get a specific configuration version with full data."""
    version = db.query(ConfigVersion).filter(ConfigVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Config version not found")

    return {
        "id": version.id,
        "version": version.version,
        "config_type": version.config_type,
        "config_data": version.config_data,
        "deployment_id": version.deployment_id,
        "created_at": version.created_at,
        "created_by": version.created_by,
        "comment": version.comment,
        "is_active": version.is_active,
        "checksum": version.checksum,
        "parent_version_id": version.parent_version_id,
    }


@router.post("/config-versions/{version_id}/activate")
async def activate_config_version(
    version_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Activate a specific configuration version (rollback)."""
    version = db.query(ConfigVersion).filter(ConfigVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Config version not found")

    # Deactivate other versions of the same type
    db.query(ConfigVersion).filter(
        ConfigVersion.config_type == version.config_type,
        ConfigVersion.deployment_id == version.deployment_id,
        ConfigVersion.is_active == True
    ).update({"is_active": False})

    version.is_active = True
    db.commit()

    log_audit(
        db, "rollback", "config_version",
        resource_id=str(version_id),
        resource_name=f"{version.config_type}@{version.version}",
        deployment_id=version.deployment_id,
        request=request,
        metadata={"action": "activated_version"}
    )

    return {"success": True, "activated_version": version.version}


@router.get("/config-versions/compare/{version_id_a}/{version_id_b}")
async def compare_config_versions(
    version_id_a: int,
    version_id_b: int,
    db: Session = Depends(get_db)
):
    """Compare two configuration versions."""
    version_a = db.query(ConfigVersion).filter(ConfigVersion.id == version_id_a).first()
    version_b = db.query(ConfigVersion).filter(ConfigVersion.id == version_id_b).first()

    if not version_a or not version_b:
        raise HTTPException(status_code=404, detail="One or both versions not found")

    # Simple diff - find keys that differ
    config_a = version_a.config_data or {}
    config_b = version_b.config_data or {}

    all_keys = set(config_a.keys()) | set(config_b.keys())
    differences = {}

    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            differences[key] = {"old": val_a, "new": val_b}

    return {
        "version_a": {
            "id": version_a.id,
            "version": version_a.version,
            "created_at": version_a.created_at
        },
        "version_b": {
            "id": version_b.id,
            "version": version_b.version,
            "created_at": version_b.created_at
        },
        "differences": differences,
        "total_changes": len(differences)
    }


# =============================================================================
# SKILL EXECUTION TRACKING
# =============================================================================

@router.get("/skill-executions")
async def list_skill_executions(
    deployment_id: Optional[str] = None,
    skill_id: Optional[str] = None,
    success: Optional[bool] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List skill execution history."""
    query = db.query(SkillExecution).order_by(desc(SkillExecution.timestamp))

    if deployment_id:
        query = query.filter(SkillExecution.deployment_id == deployment_id)
    if skill_id:
        query = query.filter(SkillExecution.skill_id == skill_id)
    if success is not None:
        query = query.filter(SkillExecution.success == success)
    if start_time:
        query = query.filter(SkillExecution.timestamp >= start_time)
    if end_time:
        query = query.filter(SkillExecution.timestamp <= end_time)

    total = query.count()
    executions = query.offset(offset).limit(limit).all()

    return {
        "executions": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "deployment_id": e.deployment_id,
                "skill_id": e.skill_id,
                "skill_name": e.skill_name,
                "skill_version": e.skill_version,
                "task_description": e.task_description,
                "execution_mode": e.execution_mode,
                "robot_id": e.robot_id,
                "success": e.success,
                "confidence": e.confidence,
                "execution_time_ms": e.execution_time_ms,
                "safety_status": e.safety_status,
                "error_message": e.error_message,
            }
            for e in executions
        ],
        "total": total,
        "offset": offset,
        "limit": limit
    }


@router.get("/skill-executions/stats")
async def get_skill_execution_stats(
    deployment_id: Optional[str] = None,
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get skill execution statistics."""
    cutoff = time.time() - (hours * 3600)
    query = db.query(SkillExecution).filter(SkillExecution.timestamp >= cutoff)

    if deployment_id:
        query = query.filter(SkillExecution.deployment_id == deployment_id)

    executions = query.all()

    if not executions:
        return {
            "period_hours": hours,
            "total_executions": 0,
            "success_rate": 1.0,
            "avg_execution_time_ms": 0,
            "by_skill": {},
            "by_mode": {}
        }

    successful = sum(1 for e in executions if e.success)
    by_skill = {}
    by_mode = {}
    total_time = 0
    count_with_time = 0

    for e in executions:
        skill_name = e.skill_name or e.skill_id
        by_skill[skill_name] = by_skill.get(skill_name, {"count": 0, "success": 0})
        by_skill[skill_name]["count"] += 1
        if e.success:
            by_skill[skill_name]["success"] += 1

        mode = e.execution_mode or "unknown"
        by_mode[mode] = by_mode.get(mode, 0) + 1

        if e.execution_time_ms:
            total_time += e.execution_time_ms
            count_with_time += 1

    return {
        "period_hours": hours,
        "total_executions": len(executions),
        "success_rate": successful / len(executions),
        "avg_execution_time_ms": total_time / count_with_time if count_with_time else 0,
        "by_skill": by_skill,
        "by_mode": by_mode
    }


@router.post("/skill-executions")
async def record_skill_execution(
    skill_id: str,
    success: bool,
    skill_name: Optional[str] = None,
    skill_version: Optional[str] = None,
    task_description: Optional[str] = None,
    execution_mode: Optional[str] = None,
    robot_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    confidence: Optional[float] = None,
    execution_time_ms: Optional[float] = None,
    safety_status: Optional[str] = None,
    blend_weights: Optional[List[float]] = None,
    error_message: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Record a skill execution."""
    execution = SkillExecution(
        deployment_id=deployment_id,
        skill_id=skill_id,
        skill_name=skill_name,
        skill_version=skill_version,
        task_description=task_description,
        execution_mode=execution_mode,
        robot_id=robot_id,
        success=success,
        confidence=confidence,
        execution_time_ms=execution_time_ms,
        safety_status=safety_status,
        blend_weights=blend_weights,
        error_message=error_message,
    )
    db.add(execution)
    db.commit()

    return {"success": True, "execution_id": execution.id}


# =============================================================================
# ALERTS
# =============================================================================

@router.get("/alerts")
async def list_alerts(
    deployment_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    category: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    resolved: Optional[bool] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List system alerts."""
    query = db.query(Alert).order_by(desc(Alert.timestamp))

    if deployment_id:
        query = query.filter(Alert.deployment_id == deployment_id)
    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)
    if category:
        query = query.filter(Alert.category == category)
    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)
    if resolved is not None:
        query = query.filter(Alert.resolved == resolved)

    alerts = query.limit(limit).all()

    return {
        "alerts": [
            {
                "id": a.id,
                "timestamp": a.timestamp,
                "deployment_id": a.deployment_id,
                "alert_type": a.alert_type,
                "category": a.category,
                "title": a.title,
                "message": a.message,
                "acknowledged": a.acknowledged,
                "acknowledged_by": a.acknowledged_by,
                "acknowledged_at": a.acknowledged_at,
                "resolved": a.resolved,
                "resolved_at": a.resolved_at,
                "resolution_notes": a.resolution_notes,
            }
            for a in alerts
        ],
        "total": len(alerts)
    }


@router.post("/alerts")
async def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    """Create a new alert."""
    db_alert = Alert(
        deployment_id=alert.deployment_id,
        alert_type=alert.alert_type,
        category=alert.category,
        title=alert.title,
        message=alert.message,
        metadata=alert.metadata,
    )
    db.add(db_alert)
    db.commit()

    return {"success": True, "alert_id": db_alert.id}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    user_id: str = "system",
    db: Session = Depends(get_db)
):
    """Acknowledge an alert."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.acknowledged = True
    alert.acknowledged_by = user_id
    alert.acknowledged_at = time.time()
    db.commit()

    return {"success": True}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    resolution_notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Resolve an alert."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.resolved = True
    alert.resolved_at = time.time()
    alert.resolution_notes = resolution_notes
    db.commit()

    return {"success": True}


# =============================================================================
# SYSTEM PRESETS
# =============================================================================

@router.get("/presets")
async def list_presets(
    preset_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List system presets."""
    query = db.query(SystemPreset).order_by(desc(SystemPreset.updated_at))

    if preset_type:
        query = query.filter(SystemPreset.preset_type == preset_type)

    presets = query.all()

    return {
        "presets": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "preset_type": p.preset_type,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "created_by": p.created_by,
                "is_default": p.is_default,
                "tags": p.tags or [],
            }
            for p in presets
        ],
        "total": len(presets)
    }


@router.post("/presets")
async def create_preset(
    preset: PresetCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a new system preset."""
    db_preset = SystemPreset(
        name=preset.name,
        description=preset.description,
        preset_type=preset.preset_type,
        config_data=preset.config_data,
        tags=preset.tags,
    )
    db.add(db_preset)
    db.commit()
    db.refresh(db_preset)

    log_audit(
        db, "create", "preset",
        resource_id=str(db_preset.id),
        resource_name=preset.name,
        request=request
    )

    return {"success": True, "preset_id": db_preset.id}


@router.get("/presets/{preset_id}")
async def get_preset(preset_id: int, db: Session = Depends(get_db)):
    """Get a system preset with full configuration."""
    preset = db.query(SystemPreset).filter(SystemPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    return {
        "id": preset.id,
        "name": preset.name,
        "description": preset.description,
        "preset_type": preset.preset_type,
        "config_data": preset.config_data,
        "created_at": preset.created_at,
        "updated_at": preset.updated_at,
        "created_by": preset.created_by,
        "is_default": preset.is_default,
        "tags": preset.tags or [],
    }


@router.post("/presets/{preset_id}/apply")
async def apply_preset(
    preset_id: int,
    deployment_id: Optional[str] = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Apply a preset configuration."""
    preset = db.query(SystemPreset).filter(SystemPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    # Create a new config version from the preset
    version_num = generate_version_number(db, "preset_applied", deployment_id)

    db_version = ConfigVersion(
        deployment_id=deployment_id,
        version=version_num,
        config_type="preset_applied",
        config_data=preset.config_data,
        comment=f"Applied preset: {preset.name}",
        is_active=True,
        checksum=generate_config_checksum(preset.config_data),
    )
    db.add(db_version)
    db.commit()

    log_audit(
        db, "apply", "preset",
        resource_id=str(preset_id),
        resource_name=preset.name,
        deployment_id=deployment_id,
        request=request,
        metadata={"config_version_id": db_version.id}
    )

    return {
        "success": True,
        "preset_name": preset.name,
        "config_version_id": db_version.id
    }


@router.delete("/presets/{preset_id}")
async def delete_preset(
    preset_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Delete a system preset."""
    preset = db.query(SystemPreset).filter(SystemPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    name = preset.name
    db.delete(preset)
    db.commit()

    log_audit(
        db, "delete", "preset",
        resource_id=str(preset_id),
        resource_name=name,
        request=request
    )

    return {"success": True}


# =============================================================================
# DASHBOARD SUMMARY
# =============================================================================

@router.get("/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get a summary for the integrator dashboard."""
    now = time.time()
    day_ago = now - 86400

    # Deployment stats
    total_deployments = db.query(Deployment).count()
    active_deployments = db.query(Deployment).filter(Deployment.status == "active").count()
    unhealthy = db.query(Deployment).filter(Deployment.health_score < 80).count()

    # Recent alerts
    unresolved_alerts = db.query(Alert).filter(Alert.resolved == False).count()
    critical_alerts = db.query(Alert).filter(
        Alert.resolved == False,
        Alert.alert_type == "critical"
    ).count()

    # Skill execution stats (last 24h)
    recent_executions = db.query(SkillExecution).filter(
        SkillExecution.timestamp >= day_ago
    ).all()

    execution_count = len(recent_executions)
    success_rate = (
        sum(1 for e in recent_executions if e.success) / execution_count
        if execution_count else 1.0
    )

    # Recent activity
    recent_audit = db.query(AuditLog).order_by(
        desc(AuditLog.timestamp)
    ).limit(10).all()

    return {
        "deployments": {
            "total": total_deployments,
            "active": active_deployments,
            "unhealthy": unhealthy,
        },
        "alerts": {
            "unresolved": unresolved_alerts,
            "critical": critical_alerts,
        },
        "executions_24h": {
            "total": execution_count,
            "success_rate": success_rate,
        },
        "recent_activity": [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_name": log.resource_name,
            }
            for log in recent_audit
        ],
        "timestamp": now,
    }

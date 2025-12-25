"""
Real-Time Monitoring Service

Live monitoring of robot status, task progress, and performance metrics
with support for dashboards and alerting.

Features:
=========
- Real-time task progress tracking
- Robot health and status monitoring
- Performance metrics (latency, throughput, success rates)
- Alert system for anomalies
- Historical data for analytics

Powered By:
==========
- 10Hz control loop visibility
- Full perception pipeline metrics
- VLA inference timing

Usage:
    from src.product import MonitoringService

    monitor = MonitoringService()

    # Get live robot status
    status = await monitor.get_robot_status("robot_001")

    # Stream task progress
    async for progress in monitor.stream_task_progress(task_id):
        print(f"Progress: {progress.percent}%")

    # Get performance dashboard data
    metrics = await monitor.get_dashboard_metrics()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Robot operational state."""
    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RobotStatus:
    """Real-time robot status."""
    # Identification
    robot_id: str
    site_id: str

    # State
    state: RobotState
    current_task_id: Optional[str] = None
    current_subtask: Optional[str] = None

    # Position
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    orientation: Dict[str, float] = field(default_factory=lambda: {"roll": 0, "pitch": 0, "yaw": 0})

    # Joint state
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    joint_torques: List[float] = field(default_factory=list)

    # Health
    battery_percent: float = 100.0
    temperature_c: float = 45.0
    cpu_usage_percent: float = 30.0
    gpu_usage_percent: float = 50.0
    memory_usage_percent: float = 40.0

    # AI Status
    vla_loaded: bool = True
    vla_model: str = "pi05_base"
    inference_time_ms: float = 10.0
    control_frequency_hz: float = 10.0

    # Timing
    last_update: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0

    # Errors
    active_errors: List[str] = field(default_factory=list)
    warning_count: int = 0


@dataclass
class TaskProgress:
    """Real-time task progress."""
    task_id: str
    instruction: str

    # Progress
    percent: float = 0.0
    current_step: int = 0
    total_steps: int = 1
    current_step_description: str = ""

    # Status
    status: str = "executing"
    started_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    remaining_seconds: float = 0.0

    # Actions
    actions_executed: int = 0
    actions_per_second: float = 0.0

    # Quality
    confidence: float = 0.95
    success_probability: float = 0.95

    # Explanation
    current_activity: str = ""
    next_activity: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for dashboard."""
    # Time range
    time_range: str = "1h"
    timestamp: datetime = field(default_factory=datetime.now)

    # Task metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    success_rate: float = 0.0
    average_task_duration_s: float = 0.0

    # Inference metrics
    average_inference_time_ms: float = 10.0
    p95_inference_time_ms: float = 15.0
    p99_inference_time_ms: float = 20.0
    inferences_per_second: float = 10.0

    # Control metrics
    control_frequency_hz: float = 10.0
    control_jitter_ms: float = 0.5

    # Perception metrics
    perception_latency_ms: float = 8.0
    camera_fps: float = 30.0
    depth_fps: float = 30.0

    # System metrics
    gpu_utilization: float = 60.0
    memory_utilization: float = 45.0
    power_consumption_w: float = 80.0

    # Jetson Thor specific
    tensor_core_utilization: float = 70.0
    fp8_operations_percent: float = 40.0


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    source: str
    robot_id: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


class MonitoringService:
    """
    Real-Time Monitoring Service.

    Provides live visibility into robot operations, task progress,
    and system performance metrics.
    """

    def __init__(self):
        """Initialize monitoring service."""
        self._robot_statuses: Dict[str, RobotStatus] = {}
        self._task_progress: Dict[str, TaskProgress] = {}
        self._alerts: List[Alert] = []
        self._metrics_history: List[PerformanceMetrics] = []

        # Subscribers
        self._status_subscribers: List[asyncio.Queue] = []
        self._progress_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._alert_subscribers: List[asyncio.Queue] = []

    # =========================================================================
    # Robot Status
    # =========================================================================

    async def get_robot_status(self, robot_id: str) -> Optional[RobotStatus]:
        """Get current status of a robot."""
        return self._robot_statuses.get(robot_id)

    async def get_all_robot_statuses(self) -> List[RobotStatus]:
        """Get status of all robots."""
        return list(self._robot_statuses.values())

    async def update_robot_status(self, status: RobotStatus) -> None:
        """Update robot status (called by robot controller)."""
        status.last_update = datetime.now()
        self._robot_statuses[status.robot_id] = status

        # Notify subscribers
        for queue in self._status_subscribers:
            await queue.put(status)

        # Check for alerts
        await self._check_robot_health(status)

    async def stream_robot_status(
        self,
        robot_id: Optional[str] = None
    ) -> AsyncIterator[RobotStatus]:
        """Stream robot status updates."""
        queue = asyncio.Queue()
        self._status_subscribers.append(queue)

        try:
            while True:
                status = await queue.get()
                if robot_id is None or status.robot_id == robot_id:
                    yield status
        finally:
            self._status_subscribers.remove(queue)

    # =========================================================================
    # Task Progress
    # =========================================================================

    async def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get current progress of a task."""
        return self._task_progress.get(task_id)

    async def update_task_progress(self, progress: TaskProgress) -> None:
        """Update task progress (called by task executor)."""
        progress.elapsed_seconds = (
            datetime.now() - progress.started_at
        ).total_seconds()

        # Estimate remaining time
        if progress.percent > 0:
            total_estimated = progress.elapsed_seconds / (progress.percent / 100)
            progress.remaining_seconds = total_estimated - progress.elapsed_seconds
            progress.estimated_completion = datetime.now() + timedelta(
                seconds=progress.remaining_seconds
            )

        self._task_progress[progress.task_id] = progress

        # Notify subscribers
        if progress.task_id in self._progress_subscribers:
            for queue in self._progress_subscribers[progress.task_id]:
                await queue.put(progress)

    async def stream_task_progress(self, task_id: str) -> AsyncIterator[TaskProgress]:
        """Stream progress updates for a specific task."""
        queue = asyncio.Queue()

        if task_id not in self._progress_subscribers:
            self._progress_subscribers[task_id] = []
        self._progress_subscribers[task_id].append(queue)

        try:
            while True:
                progress = await queue.get()
                yield progress

                if progress.status in ["completed", "failed", "cancelled"]:
                    break
        finally:
            self._progress_subscribers[task_id].remove(queue)

    # =========================================================================
    # Performance Metrics
    # =========================================================================

    async def get_dashboard_metrics(
        self,
        time_range: str = "1h"
    ) -> PerformanceMetrics:
        """Get aggregated metrics for dashboard."""
        # In production, would aggregate from time-series database
        metrics = PerformanceMetrics(time_range=time_range)

        # Calculate from active robots
        statuses = list(self._robot_statuses.values())
        if statuses:
            metrics.average_inference_time_ms = sum(
                s.inference_time_ms for s in statuses
            ) / len(statuses)

            metrics.control_frequency_hz = sum(
                s.control_frequency_hz for s in statuses
            ) / len(statuses)

            metrics.gpu_utilization = sum(
                s.gpu_usage_percent for s in statuses
            ) / len(statuses)

            metrics.memory_utilization = sum(
                s.memory_usage_percent for s in statuses
            ) / len(statuses)

        # Task metrics
        completed = [p for p in self._task_progress.values()
                    if p.status == "completed"]
        failed = [p for p in self._task_progress.values()
                 if p.status == "failed"]

        metrics.tasks_completed = len(completed)
        metrics.tasks_failed = len(failed)

        if completed or failed:
            metrics.success_rate = len(completed) / (len(completed) + len(failed)) * 100

        if completed:
            metrics.average_task_duration_s = sum(
                p.elapsed_seconds for p in completed
            ) / len(completed)

        return metrics

    async def get_metrics_history(
        self,
        time_range: str = "24h",
        interval: str = "5m"
    ) -> List[PerformanceMetrics]:
        """Get historical metrics for charts."""
        return self._metrics_history[-288:]  # Last 24h at 5m intervals

    # =========================================================================
    # Alerts
    # =========================================================================

    async def create_alert(
        self,
        severity: AlertSeverity,
        message: str,
        source: str,
        robot_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Alert:
        """Create a new alert."""
        import uuid

        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            severity=severity,
            message=message,
            source=source,
            robot_id=robot_id,
            task_id=task_id,
        )

        self._alerts.append(alert)

        # Notify subscribers
        for queue in self._alert_subscribers:
            await queue.put(alert)

        logger.warning(f"Alert [{severity.value}]: {message}")

        return alert

    async def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active (unresolved) alerts."""
        alerts = [a for a in self._alerts if not a.resolved]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False

    async def stream_alerts(self) -> AsyncIterator[Alert]:
        """Stream new alerts."""
        queue = asyncio.Queue()
        self._alert_subscribers.append(queue)

        try:
            while True:
                alert = await queue.get()
                yield alert
        finally:
            self._alert_subscribers.remove(queue)

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def _check_robot_health(self, status: RobotStatus) -> None:
        """Check robot health and create alerts if needed."""
        # Battery low
        if status.battery_percent < 20:
            await self.create_alert(
                AlertSeverity.WARNING,
                f"Low battery: {status.battery_percent:.0f}%",
                "health_check",
                robot_id=status.robot_id
            )

        # High temperature
        if status.temperature_c > 80:
            await self.create_alert(
                AlertSeverity.ERROR,
                f"High temperature: {status.temperature_c:.1f}°C",
                "health_check",
                robot_id=status.robot_id
            )

        # High inference latency
        if status.inference_time_ms > 50:
            await self.create_alert(
                AlertSeverity.WARNING,
                f"High inference latency: {status.inference_time_ms:.1f}ms",
                "health_check",
                robot_id=status.robot_id
            )

        # Active errors
        for error in status.active_errors:
            await self.create_alert(
                AlertSeverity.ERROR,
                error,
                "robot_error",
                robot_id=status.robot_id
            )

    # =========================================================================
    # Dashboard Data
    # =========================================================================

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data package."""
        metrics = await self.get_dashboard_metrics()
        robots = await self.get_all_robot_statuses()
        alerts = await self.get_active_alerts()
        tasks = list(self._task_progress.values())

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_robots": len(robots),
                "active_robots": len([r for r in robots if r.state == RobotState.EXECUTING]),
                "idle_robots": len([r for r in robots if r.state == RobotState.IDLE]),
                "error_robots": len([r for r in robots if r.state == RobotState.ERROR]),
                "active_tasks": len([t for t in tasks if t.status == "executing"]),
                "pending_alerts": len(alerts),
            },
            "metrics": {
                "success_rate": metrics.success_rate,
                "avg_task_duration": metrics.average_task_duration_s,
                "avg_inference_time": metrics.average_inference_time_ms,
                "control_frequency": metrics.control_frequency_hz,
                "gpu_utilization": metrics.gpu_utilization,
            },
            "robots": [
                {
                    "id": r.robot_id,
                    "state": r.state.value,
                    "battery": r.battery_percent,
                    "task": r.current_task_id,
                }
                for r in robots
            ],
            "alerts": [
                {
                    "id": a.alert_id,
                    "severity": a.severity.value,
                    "message": a.message,
                    "time": a.timestamp.isoformat(),
                }
                for a in alerts[:10]
            ],
        }

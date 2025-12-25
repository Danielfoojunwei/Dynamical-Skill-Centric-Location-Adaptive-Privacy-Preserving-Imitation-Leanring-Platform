"""
Dynamical Product Features

This module provides high-level product APIs that expose the platform's
AI capabilities to end users and integrations.

Product Features:
================
1. **Natural Language Task API** - Execute tasks via natural language
2. **Zero-Shot Deployment** - Deploy to new environments instantly
3. **Semantic Task Planner** - Automatic task decomposition
4. **Real-Time Monitoring** - Live task progress and robot status
5. **Multi-Language Support** - 140+ language commands
6. **Fleet Management** - Multi-robot coordination

Powered By:
==========
- Pi0.5: Open-world generalization VLA from Physical Intelligence
- Meta AI: DINOv3, SAM3, V-JEPA 2 perception models
- Jetson Thor: 128GB memory, 2070 TFLOPS

Usage:
    from src.product import TaskAPI, DeploymentService, FleetManager

    # Execute natural language task
    task_api = TaskAPI.create_for_hardware()
    result = await task_api.execute("Clean the kitchen table")

    # Deploy to new environment
    deployer = DeploymentService()
    await deployer.deploy_to_site("warehouse_123")

    # Manage robot fleet
    fleet = FleetManager()
    await fleet.assign_task("robot_001", "Sort packages in aisle 3")
"""

from .task_api import (
    TaskAPI,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

from .deployment import (
    DeploymentService,
    DeploymentConfig,
    SiteConfig,
)

from .semantic_planner import (
    SemanticPlanner,
    TaskPlan,
    Subtask,
)

from .monitoring import (
    MonitoringService,
    RobotStatus,
    TaskProgress,
    PerformanceMetrics,
)

from .multilingual import (
    MultilingualService,
    LanguageCode,
    TranslatedInstruction,
)

from .fleet import (
    FleetManager,
    RobotAssignment,
    FleetStatus,
)

__all__ = [
    # Task API
    'TaskAPI',
    'TaskRequest',
    'TaskResult',
    'TaskStatus',
    # Deployment
    'DeploymentService',
    'DeploymentConfig',
    'SiteConfig',
    # Semantic Planning
    'SemanticPlanner',
    'TaskPlan',
    'Subtask',
    # Monitoring
    'MonitoringService',
    'RobotStatus',
    'TaskProgress',
    'PerformanceMetrics',
    # Multilingual
    'MultilingualService',
    'LanguageCode',
    'TranslatedInstruction',
    # Fleet
    'FleetManager',
    'RobotAssignment',
    'FleetStatus',
]

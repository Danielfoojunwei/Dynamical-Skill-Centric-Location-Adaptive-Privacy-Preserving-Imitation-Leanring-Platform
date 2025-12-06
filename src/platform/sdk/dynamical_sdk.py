"""
Dynamical.ai Platform SDK

White-label Python SDK for integrating with the Dynamical.ai platform.
Designed for embedding into robot orchestration systems.

Example usage:
    ```python
    from dynamical_sdk import DynamicalClient
    
    # Initialize client
    client = DynamicalClient(
        base_url="https://api.dynamical.ai",
        api_key="dyn_your_api_key_here"
    )
    
    # Create project
    project = client.projects.create(
        name="Pick and Place",
        description="Warehouse picking demonstrations"
    )
    
    # Upload episode
    episode = client.episodes.create(
        project_id=project.id,
        name="demo_001",
        file_path="/data/demos/demo_001.npz",
        duration_s=45.2,
        frame_count=904,
        quality_score=0.87
    )
    
    # Train model
    model = client.models.train(
        project_id=project.id,
        name="policy_v1",
        model_type="policy",
        episode_ids=[episode.id]
    )
    
    # Deploy to robot
    client.models.deploy(
        model_id=model.id,
        robot_ids=["robot_001", "robot_002"]
    )
    ```
"""

import os
import time
import json
import hashlib
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import aiohttp
    import asyncio
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = [
    "DynamicalClient",
    "AsyncDynamicalClient",
    "Project",
    "Episode",
    "Model",
    "Robot",
    "Site",
    "PipelineRun",
    "DynamicalError",
    "AuthenticationError",
    "NotFoundError",
    "ValidationError",
]


# =============================================================================
# Exceptions
# =============================================================================

class DynamicalError(Exception):
    """Base exception for Dynamical SDK."""
    def __init__(self, message: str, status_code: int = None, response: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(DynamicalError):
    """Authentication failed."""
    pass


class NotFoundError(DynamicalError):
    """Resource not found."""
    pass


class ValidationError(DynamicalError):
    """Validation failed."""
    pass


class RateLimitError(DynamicalError):
    """Rate limit exceeded."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Project:
    """Project resource."""
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
    config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Project":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Episode:
    """Episode resource."""
    id: str
    project_id: str
    name: str
    status: str
    duration_s: float
    frame_count: int
    quality_score: float
    recorded_at: str
    version: str
    tags: List[str] = field(default_factory=list)
    robot_id: Optional[str] = None
    site_id: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Model:
    """Model resource."""
    id: str
    project_id: str
    name: str
    version: str
    status: str
    model_type: str
    metrics: Dict[str, float]
    created_at: str
    deployed_to: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Model":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Robot:
    """Robot resource."""
    id: str
    name: str
    site_id: str
    robot_type: str
    status: str
    last_seen: str
    deployed_models: List[str] = field(default_factory=list)
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Robot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Site:
    """Site resource."""
    id: str
    name: str
    location: Dict[str, Any]
    timezone: str
    robot_count: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Site":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineRun:
    """Pipeline run resource."""
    id: str
    project_id: str
    pipeline_type: str
    status: str
    progress: float
    started_at: str
    completed_at: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineRun":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class User:
    """User resource."""
    id: str
    email: str
    name: str
    role: str
    organization_id: str
    created_at: str
    last_login: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "User":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class APIKey:
    """API key resource."""
    id: str
    name: str
    permissions: List[str]
    created_at: str
    key: Optional[str] = None  # Only on creation
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "APIKey":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DashboardStats:
    """Dashboard statistics."""
    total_projects: int
    total_episodes: int
    total_models: int
    total_robots: int
    total_sites: int
    active_pipelines: int
    episodes_last_24h: int
    training_hours_last_7d: float
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DashboardStats":
        return cls(**data)


# =============================================================================
# HTTP Client Base
# =============================================================================

class HTTPClient:
    """HTTP client for API requests."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str = None,
        token: str = None,
        timeout: int = 30,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = token
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        self.session = requests.Session()
        self._setup_headers()
    
    def _setup_headers(self):
        """Setup default headers."""
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"Dynamical-SDK/{__version__}",
        })
        
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
    
    def set_token(self, token: str):
        """Set authentication token."""
        self.token = token
        self.session.headers["Authorization"] = f"Bearer {token}"
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response."""
        try:
            data = response.json()
        except:
            data = {"detail": response.text}
        
        if response.status_code == 401:
            raise AuthenticationError(
                data.get("detail", "Authentication failed"),
                response.status_code,
                data
            )
        elif response.status_code == 404:
            raise NotFoundError(
                data.get("detail", "Resource not found"),
                response.status_code,
                data
            )
        elif response.status_code == 422:
            raise ValidationError(
                data.get("detail", "Validation failed"),
                response.status_code,
                data
            )
        elif response.status_code == 429:
            raise RateLimitError(
                data.get("detail", "Rate limit exceeded"),
                response.status_code,
                data
            )
        elif response.status_code >= 400:
            raise DynamicalError(
                data.get("detail", f"Request failed: {response.status_code}"),
                response.status_code,
                data
            )
        
        return data
    
    def request(
        self,
        method: str,
        path: str,
        params: Dict = None,
        json: Dict = None,
        **kwargs
    ) -> Dict:
        """Make HTTP request with retries."""
        url = f"{self.base_url}{path}"
        
        for attempt in range(self.retry_count):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    timeout=self.timeout,
                    **kwargs
                )
                return self._handle_response(response)
            
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == self.retry_count - 1:
                    raise DynamicalError(f"Connection failed: {e}")
                time.sleep(self.retry_delay * (attempt + 1))
            
            except RateLimitError:
                if attempt == self.retry_count - 1:
                    raise
                time.sleep(self.retry_delay * (attempt + 1) * 2)
    
    def get(self, path: str, params: Dict = None) -> Dict:
        return self.request("GET", path, params=params)
    
    def post(self, path: str, json: Dict = None) -> Dict:
        return self.request("POST", path, json=json)
    
    def patch(self, path: str, json: Dict = None) -> Dict:
        return self.request("PATCH", path, json=json)
    
    def delete(self, path: str) -> Dict:
        return self.request("DELETE", path)


# =============================================================================
# Resource APIs
# =============================================================================

class AuthAPI:
    """Authentication API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Login with email and password.
        
        Args:
            email: User email
            password: User password
        
        Returns:
            Dict with access_token and user info
        """
        response = self._client.post("/api/v1/auth/login", json={
            "email": email,
            "password": password,
        })
        
        # Set token for subsequent requests
        self._client.set_token(response["access_token"])
        
        return response
    
    def register(
        self,
        email: str,
        password: str,
        name: str,
        organization_name: str = None
    ) -> Dict[str, Any]:
        """
        Register new user.
        
        Args:
            email: User email
            password: User password
            name: User name
            organization_name: Optional organization name
        
        Returns:
            Dict with access_token and user info
        """
        response = self._client.post("/api/v1/auth/register", json={
            "email": email,
            "password": password,
            "name": name,
            "organization_name": organization_name,
        })
        
        self._client.set_token(response["access_token"])
        
        return response
    
    def me(self) -> User:
        """Get current user info."""
        response = self._client.get("/api/v1/auth/me")
        return User.from_dict(response)


class ProjectsAPI:
    """Projects API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def create(
        self,
        name: str,
        description: str = "",
        tags: List[str] = None,
        config: Dict[str, Any] = None
    ) -> Project:
        """
        Create new project.
        
        Args:
            name: Project name
            description: Project description
            tags: List of tags
            config: Project configuration
        
        Returns:
            Created Project
        """
        response = self._client.post("/api/v1/projects", json={
            "name": name,
            "description": description,
            "tags": tags or [],
            "config": config or {},
        })
        return Project.from_dict(response)
    
    def list(self, status: str = None) -> List[Project]:
        """
        List projects.
        
        Args:
            status: Filter by status (active, archived, draft)
        
        Returns:
            List of Projects
        """
        params = {}
        if status:
            params["status"] = status
        
        response = self._client.get("/api/v1/projects", params=params)
        return [Project.from_dict(p) for p in response]
    
    def get(self, project_id: str) -> Project:
        """
        Get project by ID.
        
        Args:
            project_id: Project ID
        
        Returns:
            Project
        """
        response = self._client.get(f"/api/v1/projects/{project_id}")
        return Project.from_dict(response)
    
    def update(
        self,
        project_id: str,
        name: str = None,
        description: str = None,
        status: str = None,
        tags: List[str] = None,
        config: Dict[str, Any] = None
    ) -> Project:
        """
        Update project.
        
        Args:
            project_id: Project ID
            name: New name
            description: New description
            status: New status
            tags: New tags
            config: Config updates
        
        Returns:
            Updated Project
        """
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if status is not None:
            data["status"] = status
        if tags is not None:
            data["tags"] = tags
        if config is not None:
            data["config"] = config
        
        response = self._client.patch(f"/api/v1/projects/{project_id}", json=data)
        return Project.from_dict(response)
    
    def delete(self, project_id: str) -> Dict:
        """
        Delete (archive) project.
        
        Args:
            project_id: Project ID
        
        Returns:
            Status dict
        """
        return self._client.delete(f"/api/v1/projects/{project_id}")
    
    def export(self, project_id: str) -> Dict:
        """
        Export project data.
        
        Args:
            project_id: Project ID
        
        Returns:
            Export data dict
        """
        return self._client.get(f"/api/v1/projects/{project_id}/export")


class EpisodesAPI:
    """Episodes API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def create(
        self,
        project_id: str,
        name: str,
        file_path: str,
        duration_s: float,
        frame_count: int,
        quality_score: float = 0.0,
        robot_id: str = None,
        site_id: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Episode:
        """
        Create new episode.
        
        Args:
            project_id: Project ID
            name: Episode name
            file_path: Path to episode file
            duration_s: Duration in seconds
            frame_count: Number of frames
            quality_score: Quality score 0-1
            robot_id: Recording robot ID
            site_id: Recording site ID
            tags: Tags
            metadata: Additional metadata
        
        Returns:
            Created Episode
        """
        response = self._client.post(
            f"/api/v1/projects/{project_id}/episodes",
            json={
                "name": name,
                "file_path": file_path,
                "duration_s": duration_s,
                "frame_count": frame_count,
                "quality_score": quality_score,
                "robot_id": robot_id,
                "site_id": site_id,
                "tags": tags or [],
                "metadata": metadata or {},
            }
        )
        return Episode.from_dict(response)
    
    def list(
        self,
        project_id: str,
        status: str = None,
        min_quality: float = 0.0,
        limit: int = 100,
        offset: int = 0
    ) -> List[Episode]:
        """
        List episodes in project.
        
        Args:
            project_id: Project ID
            status: Filter by status
            min_quality: Minimum quality score
            limit: Max results
            offset: Pagination offset
        
        Returns:
            List of Episodes
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if min_quality > 0:
            params["min_quality"] = min_quality
        
        response = self._client.get(
            f"/api/v1/projects/{project_id}/episodes",
            params=params
        )
        return [Episode.from_dict(e) for e in response]
    
    def validate(
        self,
        episode_id: str,
        approved: bool,
        notes: str = None
    ) -> Dict:
        """
        Validate or reject episode.
        
        Args:
            episode_id: Episode ID
            approved: Whether approved
            notes: Validation notes
        
        Returns:
            Status dict
        """
        params = {"approved": approved}
        if notes:
            params["notes"] = notes
        
        return self._client.patch(
            f"/api/v1/episodes/{episode_id}/validate",
            json=params
        )


class ModelsAPI:
    """Models API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def train(
        self,
        project_id: str,
        name: str,
        model_type: str,
        episode_ids: List[str] = None,
        training_config: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> Model:
        """
        Start model training.
        
        Args:
            project_id: Project ID
            name: Model name
            model_type: Model type (policy, encoder, moai)
            episode_ids: Episodes to train on
            training_config: Training configuration
            tags: Tags
        
        Returns:
            Created Model
        """
        response = self._client.post(
            f"/api/v1/projects/{project_id}/models",
            json={
                "name": name,
                "model_type": model_type,
                "episode_ids": episode_ids or [],
                "training_config": training_config or {},
                "tags": tags or [],
            }
        )
        return Model.from_dict(response)
    
    def list(
        self,
        project_id: str,
        status: str = None
    ) -> List[Model]:
        """
        List models in project.
        
        Args:
            project_id: Project ID
            status: Filter by status
        
        Returns:
            List of Models
        """
        params = {}
        if status:
            params["status"] = status
        
        response = self._client.get(
            f"/api/v1/projects/{project_id}/models",
            params=params
        )
        return [Model.from_dict(m) for m in response]
    
    def deploy(
        self,
        model_id: str,
        robot_ids: List[str]
    ) -> Dict:
        """
        Deploy model to robots.
        
        Args:
            model_id: Model ID
            robot_ids: List of robot IDs
        
        Returns:
            Deployment status
        """
        return self._client.post(
            f"/api/v1/models/{model_id}/deploy",
            json={"robot_ids": robot_ids}
        )
    
    def wait_for_training(
        self,
        model: Model,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
        callback: Callable[[Model], None] = None
    ) -> Model:
        """
        Wait for model training to complete.
        
        Args:
            model: Model to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum wait time
            callback: Optional callback for progress updates
        
        Returns:
            Final Model state
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get updated model status
            models = self.list(model.project_id)
            current = next((m for m in models if m.id == model.id), None)
            
            if not current:
                raise NotFoundError(f"Model {model.id} not found")
            
            if callback:
                callback(current)
            
            if current.status in ["ready", "failed", "deprecated"]:
                return current
            
            time.sleep(poll_interval)
        
        raise DynamicalError(f"Training timeout after {timeout}s")


class RobotsAPI:
    """Robots API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def register(
        self,
        name: str,
        site_id: str,
        robot_type: str,
        hardware_config: Dict[str, Any] = None,
        software_version: str = "1.0.0"
    ) -> Robot:
        """
        Register new robot.
        
        Args:
            name: Robot name
            site_id: Site ID
            robot_type: Robot type
            hardware_config: Hardware configuration
            software_version: Software version
        
        Returns:
            Registered Robot
        """
        response = self._client.post("/api/v1/robots", json={
            "name": name,
            "site_id": site_id,
            "robot_type": robot_type,
            "hardware_config": hardware_config or {},
            "software_version": software_version,
        })
        return Robot.from_dict(response)
    
    def list(self, site_id: str = None) -> List[Robot]:
        """
        List robots.
        
        Args:
            site_id: Filter by site
        
        Returns:
            List of Robots
        """
        params = {}
        if site_id:
            params["site_id"] = site_id
        
        response = self._client.get("/api/v1/robots", params=params)
        return [Robot.from_dict(r) for r in response]
    
    def heartbeat(
        self,
        robot_id: str,
        status: str = "online",
        metrics: Dict[str, Any] = None
    ) -> Dict:
        """
        Send robot heartbeat.
        
        Args:
            robot_id: Robot ID
            status: Current status
            metrics: Optional metrics
        
        Returns:
            Status dict
        """
        return self._client.post(
            f"/api/v1/robots/{robot_id}/heartbeat",
            json={"status": status, "metrics": metrics or {}}
        )


class SitesAPI:
    """Sites API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def create(
        self,
        name: str,
        location: Dict[str, Any] = None,
        timezone: str = "UTC"
    ) -> Site:
        """
        Create new site.
        
        Args:
            name: Site name
            location: Location data
            timezone: Timezone
        
        Returns:
            Created Site
        """
        response = self._client.post("/api/v1/sites", json={
            "name": name,
            "location": location or {},
            "timezone": timezone,
        })
        return Site.from_dict(response)
    
    def list(self) -> List[Site]:
        """List all sites."""
        response = self._client.get("/api/v1/sites")
        return [Site.from_dict(s) for s in response]


class PipelinesAPI:
    """Pipelines API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def start(
        self,
        project_id: str,
        pipeline_type: str,
        config: Dict[str, Any] = None
    ) -> PipelineRun:
        """
        Start pipeline run.
        
        Args:
            project_id: Project ID
            pipeline_type: Pipeline type (training, compression, deployment)
            config: Pipeline configuration
        
        Returns:
            PipelineRun
        """
        response = self._client.post(
            f"/api/v1/projects/{project_id}/pipelines",
            json={
                "pipeline_type": pipeline_type,
                "config": config or {},
            }
        )
        return PipelineRun.from_dict(response)
    
    def list(
        self,
        project_id: str,
        status: str = None
    ) -> List[PipelineRun]:
        """
        List pipeline runs.
        
        Args:
            project_id: Project ID
            status: Filter by status
        
        Returns:
            List of PipelineRuns
        """
        params = {}
        if status:
            params["status"] = status
        
        response = self._client.get(
            f"/api/v1/projects/{project_id}/pipelines",
            params=params
        )
        return [PipelineRun.from_dict(p) for p in response]


class APIKeysAPI:
    """API Keys management."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def create(
        self,
        name: str,
        permissions: List[str] = None,
        expires_in_days: int = None
    ) -> APIKey:
        """
        Create new API key.
        
        Args:
            name: Key name
            permissions: Permission list
            expires_in_days: Expiration in days
        
        Returns:
            Created APIKey (with key visible)
        """
        response = self._client.post("/api/v1/api-keys", json={
            "name": name,
            "permissions": permissions or ["read"],
            "expires_in_days": expires_in_days,
        })
        return APIKey.from_dict(response)
    
    def list(self) -> List[APIKey]:
        """List API keys."""
        response = self._client.get("/api/v1/api-keys")
        return [APIKey.from_dict(k) for k in response]
    
    def revoke(self, key_id: str) -> Dict:
        """
        Revoke API key.
        
        Args:
            key_id: Key ID
        
        Returns:
            Status dict
        """
        return self._client.delete(f"/api/v1/api-keys/{key_id}")


class DashboardAPI:
    """Dashboard API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def stats(self) -> DashboardStats:
        """Get dashboard statistics."""
        response = self._client.get("/api/v1/dashboard/stats")
        return DashboardStats.from_dict(response)


class AuditAPI:
    """Audit logs API."""
    
    def __init__(self, client: HTTPClient):
        self._client = client
    
    def list(
        self,
        resource_type: str = None,
        resource_id: str = None,
        action: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get audit logs.
        
        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            action: Filter by action
            limit: Max results
        
        Returns:
            List of audit log entries
        """
        params = {"limit": limit}
        if resource_type:
            params["resource_type"] = resource_type
        if resource_id:
            params["resource_id"] = resource_id
        if action:
            params["action"] = action
        
        return self._client.get("/api/v1/audit-logs", params=params)


# =============================================================================
# Main Client
# =============================================================================

class DynamicalClient:
    """
    Dynamical.ai Platform SDK Client.
    
    Example:
        ```python
        # Using API key
        client = DynamicalClient(
            base_url="https://api.dynamical.ai",
            api_key="dyn_your_key_here"
        )
        
        # Or login
        client = DynamicalClient(base_url="https://api.dynamical.ai")
        client.auth.login("user@example.com", "password")
        
        # Use APIs
        projects = client.projects.list()
        ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = None,
        token: str = None,
        timeout: int = 30,
        retry_count: int = 3
    ):
        """
        Initialize Dynamical client.
        
        Args:
            base_url: API base URL
            api_key: API key for authentication
            token: JWT token for authentication
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
        """
        self._http = HTTPClient(
            base_url=base_url,
            api_key=api_key,
            token=token,
            timeout=timeout,
            retry_count=retry_count,
        )
        
        # Initialize API endpoints
        self.auth = AuthAPI(self._http)
        self.projects = ProjectsAPI(self._http)
        self.episodes = EpisodesAPI(self._http)
        self.models = ModelsAPI(self._http)
        self.robots = RobotsAPI(self._http)
        self.sites = SitesAPI(self._http)
        self.pipelines = PipelinesAPI(self._http)
        self.api_keys = APIKeysAPI(self._http)
        self.dashboard = DashboardAPI(self._http)
        self.audit = AuditAPI(self._http)
    
    def health(self) -> Dict:
        """Check API health."""
        return self._http.get("/health")
    
    def info(self) -> Dict:
        """Get API info."""
        return self._http.get("/api/v1/info")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    base_url: str = None,
    api_key: str = None,
    email: str = None,
    password: str = None
) -> DynamicalClient:
    """
    Create and configure a Dynamical client.
    
    Args:
        base_url: API URL (or DYNAMICAL_API_URL env var)
        api_key: API key (or DYNAMICAL_API_KEY env var)
        email: Login email
        password: Login password
    
    Returns:
        Configured DynamicalClient
    """
    base_url = base_url or os.getenv("DYNAMICAL_API_URL", "http://localhost:8080")
    api_key = api_key or os.getenv("DYNAMICAL_API_KEY")
    
    client = DynamicalClient(base_url=base_url, api_key=api_key)
    
    if email and password and not api_key:
        client.auth.login(email, password)
    
    return client


# =============================================================================
# CLI
# =============================================================================

def main():
    """Simple CLI for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamical SDK CLI")
    parser.add_argument("--url", default="http://localhost:8080", help="API URL")
    parser.add_argument("--key", help="API key")
    parser.add_argument("command", choices=["health", "info", "projects"], help="Command")
    
    args = parser.parse_args()
    
    client = DynamicalClient(base_url=args.url, api_key=args.key)
    
    if args.command == "health":
        print(json.dumps(client.health(), indent=2))
    elif args.command == "info":
        print(json.dumps(client.info(), indent=2))
    elif args.command == "projects":
        for p in client.projects.list():
            print(f"  {p.id}: {p.name} ({p.status})")


if __name__ == "__main__":
    main()

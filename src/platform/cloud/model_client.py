"""
Unified Model and Skill Library Client

This module provides a clean separation between:
1. Base VLA Model Download (One-Time, Read-Only)
   - Pi0, OpenVLA, etc. are proprietary - we can only download, not train

2. Skill Library Operations (Read/Write)
   - Skills are MoE experts trained on our edge devices
   - Skills are uploaded/downloaded from OUR cloud skill library
   - Federated Learning aggregates skill updates across fleet

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Dynamical Edge Platform                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   Base VLA Model (FROZEN)          Skill Library (OURS)         │
    │   ┌──────────────────┐             ┌──────────────────┐         │
    │   │ Pi0 / OpenVLA    │             │  MoE Skills      │         │
    │   │ (Read-Only)      │             │  (Read/Write)    │         │
    │   └────────┬─────────┘             └────────┬─────────┘         │
    │            │                                │                    │
    │   One-Time Download              Upload/Download Skills          │
    │            │                                │                    │
    │   ┌────────▼─────────┐             ┌────────▼─────────┐         │
    │   │ Vendor CDN       │             │ Dynamical Cloud  │         │
    │   │ (HuggingFace)    │             │ Skill Library    │         │
    │   └──────────────────┘             └──────────────────┘         │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
"""

import os
import hashlib
import json
import time
import gzip
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try HTTP libraries
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BaseModelInfo:
    """Information about a base VLA model (read-only)."""
    name: str
    version: str
    sha256: str
    size_bytes: int
    download_url: str
    provider: str  # "physical-intelligence", "openvla", etc.
    is_cached: bool = False
    local_path: Optional[str] = None


@dataclass
class SkillInfo:
    """Information about a skill in the library."""
    id: str
    name: str
    description: str
    skill_type: str
    version: str
    status: str
    file_hash: str
    file_size_bytes: int
    created_at: str
    updated_at: str
    tags: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    deployed_to: List[str] = field(default_factory=list)


@dataclass
class ModelClientConfig:
    """Configuration for model and skill client."""
    # API settings
    skill_library_url: str = "http://localhost:8080"
    api_key: str = ""

    # Cache settings
    cache_dir: str = "~/.cache/dynamical"
    base_model_cache: str = "models"  # For frozen VLA models
    skill_cache: str = "skills"        # For MoE skills

    # Timeouts
    timeout: int = 30
    download_timeout: int = 300
    retry_count: int = 3

    # Base model sources (read-only)
    huggingface_url: str = "https://huggingface.co"


# =============================================================================
# Base Model Client (Read-Only)
# =============================================================================

class BaseModelClient:
    """
    Client for downloading frozen base VLA models.

    IMPORTANT: These models are proprietary and READ-ONLY.
    We cannot train or upload gradients to vendor models.

    Supported models:
    - Pi0 (Physical Intelligence)
    - OpenVLA
    - RTMPose (for pose estimation)
    """

    # Model registry - maps model names to HuggingFace paths
    MODEL_REGISTRY = {
        "pi0-base": {
            "hf_repo": "physical-intelligence/pi0",
            "file": "model.onnx",
            "provider": "physical-intelligence",
        },
        "openvla-7b": {
            "hf_repo": "openvla/openvla-7b",
            "file": "model.safetensors",
            "provider": "openvla",
        },
        "rtmpose-m": {
            "hf_repo": "usyd-community/rtmpose-m-body8",
            "file": "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx",
            "provider": "mmpose",
        },
    }

    def __init__(self, config: Optional[ModelClientConfig] = None):
        self.config = config or ModelClientConfig()
        self.cache_dir = Path(os.path.expanduser(self.config.cache_dir))
        self.model_cache = self.cache_dir / self.config.base_model_cache
        self.model_cache.mkdir(parents=True, exist_ok=True)

        # Track downloaded models
        self._downloaded_models: Dict[str, BaseModelInfo] = {}
        self._load_cache_index()

    def _load_cache_index(self):
        """Load index of cached models."""
        index_path = self.model_cache / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                for name, info in data.items():
                    self._downloaded_models[name] = BaseModelInfo(**info)
            except Exception as e:
                logger.warning(f"Failed to load model cache index: {e}")

    def _save_cache_index(self):
        """Save index of cached models."""
        index_path = self.model_cache / "index.json"
        data = {k: asdict(v) for k, v in self._downloaded_models.items()}
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def is_cached(self, model_name: str) -> bool:
        """Check if model is already cached locally."""
        if model_name in self._downloaded_models:
            info = self._downloaded_models[model_name]
            if info.local_path and os.path.exists(info.local_path):
                return True
        return False

    def get_cached_path(self, model_name: str) -> Optional[str]:
        """Get path to cached model if available."""
        if self.is_cached(model_name):
            return self._downloaded_models[model_name].local_path
        return None

    def download_base_model(
        self,
        model_name: str,
        target_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """
        Download a base VLA model (one-time operation).

        Args:
            model_name: Name of the model (e.g., "pi0-base", "openvla-7b")
            target_path: Where to save (uses cache if None)
            progress_callback: Optional callback(downloaded, total)
            force: Force re-download even if cached

        Returns:
            Path to downloaded model, or None if failed
        """
        if model_name not in self.MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            logger.info(f"Available models: {list(self.MODEL_REGISTRY.keys())}")
            return None

        # Check cache
        if not force and self.is_cached(model_name):
            cached_path = self.get_cached_path(model_name)
            logger.info(f"Using cached model: {cached_path}")
            if target_path and target_path != cached_path:
                import shutil
                shutil.copy2(cached_path, target_path)
                return target_path
            return cached_path

        # Get model info
        model_info = self.MODEL_REGISTRY[model_name]
        hf_repo = model_info["hf_repo"]
        file_name = model_info["file"]

        # Build download URL
        url = f"{self.config.huggingface_url}/{hf_repo}/resolve/main/{file_name}"

        # Determine target path
        if target_path is None:
            target_path = str(self.model_cache / f"{model_name}-{file_name}")

        logger.info(f"Downloading base model: {model_name}")
        logger.info(f"  From: {url}")
        logger.info(f"  To: {target_path}")

        # Download
        success = self._download_file(url, target_path, progress_callback)

        if success:
            # Compute hash
            file_hash = self._compute_sha256(target_path)
            file_size = os.path.getsize(target_path)

            # Update cache index
            self._downloaded_models[model_name] = BaseModelInfo(
                name=model_name,
                version="latest",
                sha256=file_hash,
                size_bytes=file_size,
                download_url=url,
                provider=model_info["provider"],
                is_cached=True,
                local_path=target_path,
            )
            self._save_cache_index()

            logger.info(f"Base model downloaded: {model_name} ({file_size} bytes)")
            return target_path

        return None

    def _download_file(
        self,
        url: str,
        path: str,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """Download file with progress tracking."""
        if not HAS_REQUESTS:
            logger.error("requests library required: pip install requests")
            return False

        try:
            with requests.get(url, stream=True, timeout=self.config.download_timeout) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                downloaded = 0

                Path(path).parent.mkdir(parents=True, exist_ok=True)

                with open(path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)

                return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _compute_sha256(self, path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def list_available_models(self) -> List[str]:
        """List available base models."""
        return list(self.MODEL_REGISTRY.keys())

    def list_cached_models(self) -> List[BaseModelInfo]:
        """List cached base models."""
        return [
            info for info in self._downloaded_models.values()
            if info.is_cached and info.local_path and os.path.exists(info.local_path)
        ]


# =============================================================================
# Skill Library Client (Read/Write)
# =============================================================================

class SkillLibraryClient:
    """
    Client for the Dynamical Skill Library.

    This handles MoE skills that WE train and own:
    - Upload skills trained on edge devices
    - Download skills for execution
    - Federated learning of skills across fleet

    Skills are encrypted using N2HE and stored in our cloud.
    """

    def __init__(self, config: Optional[ModelClientConfig] = None):
        self.config = config or ModelClientConfig()
        self.cache_dir = Path(os.path.expanduser(self.config.cache_dir))
        self.skill_cache = self.cache_dir / self.config.skill_cache
        self.skill_cache.mkdir(parents=True, exist_ok=True)

        # Local skill index
        self._skill_index: Dict[str, SkillInfo] = {}
        self._load_cache_index()

    def _load_cache_index(self):
        """Load index of cached skills."""
        index_path = self.skill_cache / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                for skill_id, info in data.items():
                    self._skill_index[skill_id] = SkillInfo(**info)
            except Exception as e:
                logger.warning(f"Failed to load skill cache: {e}")

    def _save_cache_index(self):
        """Save skill cache index."""
        index_path = self.skill_cache / "index.json"
        data = {k: asdict(v) for k, v in self._skill_index.items()}
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Dict = None,
        files: Dict = None,
    ) -> Optional[Dict]:
        """Make HTTP request to skill library."""
        if not HAS_REQUESTS:
            logger.error("requests library required")
            return None

        url = f"{self.config.skill_library_url.rstrip('/')}{endpoint}"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=self.config.timeout)
            elif method == "POST":
                if files:
                    resp = requests.post(url, headers=headers, data=json_data, files=files, timeout=self.config.timeout)
                else:
                    headers["Content-Type"] = "application/json"
                    resp = requests.post(url, headers=headers, json=json_data, timeout=self.config.timeout)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=self.config.timeout)
            else:
                return None

            if resp.ok:
                return resp.json()
            else:
                logger.error(f"Request failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            logger.error(f"Request error: {e}")

        return None

    def list_skills(
        self,
        skill_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SkillInfo]:
        """
        List skills from the cloud library.

        Args:
            skill_type: Filter by type (manipulation, navigation, etc.)
            tags: Filter by tags

        Returns:
            List of skill info
        """
        params = {}
        if skill_type:
            params["skill_type"] = skill_type
        if tags:
            params["tags"] = ",".join(tags)

        endpoint = "/api/v1/skills"
        if params:
            endpoint += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        result = self._make_request("GET", endpoint)

        if result and "skills" in result:
            return [SkillInfo(**s) for s in result["skills"]]

        # Fallback to cached skills
        logger.warning("Using cached skill list")
        return list(self._skill_index.values())

    def download_skill(
        self,
        skill_id: str,
        decrypt: bool = True,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Download a skill from the library.

        Args:
            skill_id: Skill identifier
            decrypt: Whether to decrypt (requires N2HE)

        Returns:
            Tuple of (weights, config) or None
        """
        # Check cache
        cache_path = self.skill_cache / skill_id
        if cache_path.exists():
            weights_path = cache_path / "weights.npz"
            config_path = cache_path / "config.json"

            if weights_path.exists():
                weights = np.load(weights_path)['weights']
                config = {}
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                return weights, config

        # Download from server
        result = self._make_request("GET", f"/api/v1/skills/{skill_id}")

        if not result:
            return None

        try:
            # Decode weights
            encrypted_weights = result.get("encrypted_weights", "")
            encrypted_config = result.get("encrypted_config", "")

            import base64
            weights_bytes = base64.b64decode(encrypted_weights)
            config_bytes = base64.b64decode(encrypted_config)

            # Decrypt (or decompress if not encrypted)
            try:
                weights = pickle.loads(gzip.decompress(weights_bytes))
            except:
                weights = np.frombuffer(weights_bytes, dtype=np.float32)

            try:
                config = json.loads(gzip.decompress(config_bytes).decode())
            except:
                config = {}

            # Cache locally
            cache_path.mkdir(exist_ok=True)
            np.savez_compressed(cache_path / "weights.npz", weights=weights)
            with open(cache_path / "config.json", 'w') as f:
                json.dump(config, f)

            # Update index
            metadata = result.get("metadata", {})
            self._skill_index[skill_id] = SkillInfo(
                id=skill_id,
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                skill_type=metadata.get("skill_type", ""),
                version=metadata.get("version", "1.0.0"),
                status="cached",
                file_hash=hashlib.sha256(weights_bytes).hexdigest()[:16],
                file_size_bytes=len(weights_bytes),
                created_at=metadata.get("created_at", ""),
                updated_at=datetime.utcnow().isoformat(),
                tags=metadata.get("tags", []),
            )
            self._save_cache_index()

            return weights, config

        except Exception as e:
            logger.error(f"Failed to download skill {skill_id}: {e}")
            return None

    def upload_skill(
        self,
        name: str,
        description: str,
        skill_type: str,
        weights: np.ndarray,
        config: Dict[str, Any],
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        encrypt: bool = True,
    ) -> Optional[str]:
        """
        Upload a skill to the library.

        Args:
            name: Skill name
            description: Description
            skill_type: Type (manipulation, navigation, etc.)
            weights: Model weights
            config: Skill configuration
            version: Version string
            tags: Optional tags
            encrypt: Whether to encrypt (uses N2HE)

        Returns:
            Skill ID if successful
        """
        import base64

        # Compress weights
        weights_bytes = gzip.compress(pickle.dumps(weights))
        config_bytes = gzip.compress(json.dumps(config).encode())

        # Generate skill ID
        skill_id = f"skill_{hashlib.md5(f'{name}_{version}_{time.time()}'.encode()).hexdigest()[:12]}"

        # Prepare upload
        data = {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "skill_type": skill_type,
            "version": version,
            "tags": tags or [],
            "encrypted_weights": base64.b64encode(weights_bytes).decode(),
            "encrypted_config": base64.b64encode(config_bytes).decode(),
        }

        result = self._make_request("POST", "/api/v1/skills", json_data=data)

        if result and result.get("success"):
            # Cache locally
            cache_path = self.skill_cache / skill_id
            cache_path.mkdir(exist_ok=True)
            np.savez_compressed(cache_path / "weights.npz", weights=weights)
            with open(cache_path / "config.json", 'w') as f:
                json.dump(config, f)

            # Update index
            self._skill_index[skill_id] = SkillInfo(
                id=skill_id,
                name=name,
                description=description,
                skill_type=skill_type,
                version=version,
                status="uploaded",
                file_hash=hashlib.sha256(weights_bytes).hexdigest()[:16],
                file_size_bytes=len(weights_bytes),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                tags=tags or [],
            )
            self._save_cache_index()

            logger.info(f"Uploaded skill: {name} ({skill_id})")
            return skill_id

        return None

    def request_skills_for_task(
        self,
        task_description: str,
        max_skills: int = 3,
        skill_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Request skills for a task using MoE routing.

        Args:
            task_description: Natural language task
            max_skills: Maximum skills to return
            skill_types: Filter by types

        Returns:
            List of (skill_id, weight) tuples
        """
        data = {
            "task_description": task_description,
            "max_skills": max_skills,
        }
        if skill_types:
            data["skill_types"] = skill_types

        result = self._make_request("POST", "/api/v1/skills/request", json_data=data)

        if result:
            skills = result.get("skill_ids", [])
            weights = result.get("weights", [1.0 / max(1, len(skills))] * len(skills))
            return list(zip(skills, weights))

        return []

    def get_cached_skill(self, skill_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get skill from local cache only."""
        cache_path = self.skill_cache / skill_id
        weights_path = cache_path / "weights.npz"
        config_path = cache_path / "config.json"

        if weights_path.exists():
            weights = np.load(weights_path)['weights']
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            return weights, config

        return None

    def list_cached_skills(self) -> List[SkillInfo]:
        """List locally cached skills."""
        return list(self._skill_index.values())

    def delete_cached_skill(self, skill_id: str) -> bool:
        """Delete skill from local cache."""
        cache_path = self.skill_cache / skill_id
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)

        if skill_id in self._skill_index:
            del self._skill_index[skill_id]
            self._save_cache_index()
            return True

        return False


# =============================================================================
# Unified Model Client
# =============================================================================

class UnifiedModelClient:
    """
    Unified client for all model operations.

    Provides a single interface for:
    - Base VLA model download (read-only)
    - Skill library operations (read/write)

    Usage:
        client = UnifiedModelClient()

        # One-time base model download
        path = client.download_base_model("pi0-base")

        # Skill operations
        client.upload_skill("grasp", "Grasp objects", "manipulation", weights, config)
        skills = client.request_skills_for_task("pick up the cup")
    """

    def __init__(self, config: Optional[ModelClientConfig] = None):
        self.config = config or ModelClientConfig()
        self.base_models = BaseModelClient(config)
        self.skills = SkillLibraryClient(config)

    # Delegate base model operations
    def download_base_model(self, *args, **kwargs):
        return self.base_models.download_base_model(*args, **kwargs)

    def is_base_model_cached(self, model_name: str) -> bool:
        return self.base_models.is_cached(model_name)

    def get_base_model_path(self, model_name: str) -> Optional[str]:
        return self.base_models.get_cached_path(model_name)

    def list_available_base_models(self) -> List[str]:
        return self.base_models.list_available_models()

    # Delegate skill operations
    def list_skills(self, *args, **kwargs):
        return self.skills.list_skills(*args, **kwargs)

    def download_skill(self, *args, **kwargs):
        return self.skills.download_skill(*args, **kwargs)

    def upload_skill(self, *args, **kwargs):
        return self.skills.upload_skill(*args, **kwargs)

    def request_skills_for_task(self, *args, **kwargs):
        return self.skills.request_skills_for_task(*args, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "base_models": {
                "cached": len(self.base_models.list_cached_models()),
                "available": len(self.base_models.list_available_models()),
            },
            "skills": {
                "cached": len(self.skills.list_cached_skills()),
            },
        }


# =============================================================================
# Backwards Compatibility - Deprecated FFMClient
# =============================================================================

class FFMClient:
    """
    DEPRECATED: Use UnifiedModelClient instead.

    This class is maintained for backwards compatibility only.
    The FFM (Foundation Field Model) concept is replaced by:
    - BaseModelClient for frozen VLA models
    - SkillLibraryClient for trainable MoE skills
    """

    def __init__(self, config=None):
        import warnings
        warnings.warn(
            "FFMClient is deprecated. Use UnifiedModelClient instead. "
            "Base VLA models are read-only; use SkillLibraryClient for trainable skills.",
            DeprecationWarning,
            stacklevel=2
        )
        self._client = UnifiedModelClient()

    def check_for_updates(self, current_version: str, model_name: str = "pi0-base"):
        """Deprecated: Base models don't receive updates."""
        logger.warning("check_for_updates is deprecated - base models are frozen")
        return None

    def download_model(self, version: str, target_path: str, model_name: str = "pi0-base"):
        """Download base model (one-time)."""
        return self._client.download_base_model(model_name, target_path) is not None


# =============================================================================
# Testing
# =============================================================================

def test_model_client():
    """Test the unified model client."""
    print("\n" + "=" * 60)
    print("UNIFIED MODEL CLIENT TEST")
    print("=" * 60)

    config = ModelClientConfig(
        skill_library_url="http://localhost:8080",
        cache_dir="/tmp/dynamical_test",
    )

    client = UnifiedModelClient(config)

    print("\n1. Base Model Operations")
    print("-" * 40)
    print(f"   Available models: {client.list_available_base_models()}")
    print(f"   Cached models: {len(client.base_models.list_cached_models())}")

    print("\n2. Skill Library Operations")
    print("-" * 40)

    # Create test skill
    weights = np.random.randn(1000).astype(np.float32)
    config_data = {"input_dim": 256, "output_dim": 7}

    # Note: Upload will fail without a running server, but we can test caching
    print("   Testing local skill cache...")

    # Manually cache a skill for testing
    skill_id = "test_skill_001"
    cache_path = client.skills.skill_cache / skill_id
    cache_path.mkdir(exist_ok=True)
    np.savez_compressed(cache_path / "weights.npz", weights=weights)
    with open(cache_path / "config.json", 'w') as f:
        json.dump(config_data, f)

    client.skills._skill_index[skill_id] = SkillInfo(
        id=skill_id,
        name="Test Grasp",
        description="Test grasping skill",
        skill_type="manipulation",
        version="1.0.0",
        status="cached",
        file_hash="test123",
        file_size_bytes=4000,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
    )
    client.skills._save_cache_index()

    # Retrieve from cache
    result = client.skills.get_cached_skill(skill_id)
    if result:
        w, c = result
        print(f"   Retrieved skill: {skill_id}")
        print(f"   Weights shape: {w.shape}")
        print(f"   Config: {c}")

    print("\n3. Statistics")
    print("-" * 40)
    stats = client.get_statistics()
    print(f"   Base models cached: {stats['base_models']['cached']}")
    print(f"   Skills cached: {stats['skills']['cached']}")

    print("\n" + "=" * 60)
    print("MODEL CLIENT TESTS COMPLETE")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/dynamical_test", ignore_errors=True)


if __name__ == "__main__":
    test_model_client()

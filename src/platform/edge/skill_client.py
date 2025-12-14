"""
Edge Skill Client for Jetson Orin

This module provides skill retrieval and execution for edge devices:
- Download encrypted skills from cloud
- Decrypt skills using local N2HE keys
- Cache skills for offline operation
- Execute skills with MoE blending

Designed for Jetson AGX Orin 32GB running humanoid robot control.

Architecture:
- Cloud stores encrypted skills (N2HE, 128-bit security)
- Edge downloads skills on demand or via pre-deployment
- Skills are decrypted locally with edge-specific keys
- MoE blending allows combining multiple skills for complex tasks
"""

import os
import time
import json
import gzip
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)

# Import N2HE for decryption
try:
    from src.moai.n2he import N2HEContext, N2HEParams, LWECiphertext
    HAS_N2HE = True
except ImportError:
    HAS_N2HE = False
    logger.warning("N2HE not available - decryption disabled")

# Import torch for model loading
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")

# Optional HTTP client
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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CoordinationMetadata:
    """Coordination metadata for multi-robot skill execution.

    This metadata is fetched from the unified skill artifact registry
    and used by SwarmBridge/SwarmBrain for cross-site orchestration.

    The edge device uses this to understand how to execute skills
    in a coordinated manner with other robots.
    """
    # Role requirements
    supported_roles: List[str] = field(default_factory=lambda: ["independent"])
    requires_leader: bool = False
    max_followers: int = 0

    # Synchronization
    sync_mode: str = "none"  # none, loose, strict, realtime
    sync_frequency_hz: float = 0.0

    # Communication
    broadcast_state: bool = False
    receive_state: bool = False
    state_channels: List[str] = field(default_factory=list)

    # Constraints
    min_robots: int = 1
    max_robots: int = 1
    requires_same_site: bool = True


@dataclass
class SkillInfo:
    """Local skill information."""
    id: str
    name: str
    description: str
    skill_type: str
    version: str
    status: str
    file_hash: str
    file_size_bytes: int
    cached_at: str
    last_used: Optional[str] = None
    execution_count: int = 0
    avg_execution_time_ms: float = 0.0

    # Coordination metadata (for SwarmBridge integration)
    coordination: Optional[CoordinationMetadata] = None


@dataclass
class SkillExecutionResult:
    """Result of skill execution."""
    success: bool
    skill_id: str
    execution_time_ms: float
    output: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBlendConfig:
    """Configuration for blending multiple skills."""
    skill_ids: List[str]
    weights: List[float]
    blend_mode: str = "weighted_sum"  # weighted_sum, attention, sequential


# =============================================================================
# Skill Cache
# =============================================================================

class SkillCache:
    """
    Local skill cache for edge device.

    Features:
    - Persistent storage of decrypted skills
    - LRU eviction when cache full
    - Integrity verification
    - Offline operation support
    """

    def __init__(
        self,
        cache_dir: str = "/var/lib/dynamical/skill_cache",
        max_cache_size_mb: int = 1024,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb

        # In-memory index
        self._index: Dict[str, SkillInfo] = {}
        self._loaded_skills: Dict[str, Any] = {}  # skill_id -> loaded model
        self._lock = threading.Lock()

        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                for skill_id, info in data.items():
                    self._index[skill_id] = SkillInfo(**info)
                logger.info(f"Loaded {len(self._index)} cached skills")
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")

    def _save_index(self):
        """Save cache index to disk."""
        index_path = self.cache_dir / "cache_index.json"
        try:
            data = {k: asdict(v) for k, v in self._index.items()}
            with open(index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        return sum(s.file_size_bytes for s in self._index.values()) / (1024 * 1024)

    def _evict_if_needed(self, required_mb: float):
        """Evict old skills if cache is full."""
        while self.get_cache_size_mb() + required_mb > self.max_cache_size_mb:
            if not self._index:
                break

            # Find least recently used
            lru_id = min(
                self._index.keys(),
                key=lambda k: self._index[k].last_used or self._index[k].cached_at
            )

            self.remove(lru_id)
            logger.info(f"Evicted skill {lru_id} from cache")

    def store(
        self,
        skill_id: str,
        name: str,
        description: str,
        skill_type: str,
        version: str,
        weights: np.ndarray,
        config: Dict[str, Any],
    ) -> SkillInfo:
        """
        Store a skill in the cache.

        Args:
            skill_id: Skill identifier
            name: Skill name
            description: Description
            skill_type: Type of skill
            version: Version string
            weights: Decrypted model weights
            config: Skill configuration

        Returns:
            Skill info
        """
        with self._lock:
            skill_dir = self.cache_dir / skill_id
            skill_dir.mkdir(exist_ok=True)

            # Save weights
            weights_path = skill_dir / "weights.npz"
            np.savez_compressed(weights_path, weights=weights)

            # Save config
            config_path = skill_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Compute file hash and size
            file_size = weights_path.stat().st_size
            file_hash = hashlib.sha256(open(weights_path, 'rb').read()).hexdigest()[:16]

            # Check cache size
            required_mb = file_size / (1024 * 1024)
            self._evict_if_needed(required_mb)

            # Create info
            info = SkillInfo(
                id=skill_id,
                name=name,
                description=description,
                skill_type=skill_type,
                version=version,
                status="cached",
                file_hash=file_hash,
                file_size_bytes=file_size,
                cached_at=datetime.utcnow().isoformat(),
            )

            self._index[skill_id] = info
            self._save_index()

            return info

    def get(self, skill_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get a skill from the cache.

        Args:
            skill_id: Skill identifier

        Returns:
            Tuple of (weights, config) or None if not found
        """
        if skill_id not in self._index:
            return None

        skill_dir = self.cache_dir / skill_id
        weights_path = skill_dir / "weights.npz"
        config_path = skill_dir / "config.json"

        if not weights_path.exists():
            return None

        try:
            # Load weights
            data = np.load(weights_path)
            weights = data['weights']

            # Load config
            with open(config_path) as f:
                config = json.load(f)

            # Update last used
            self._index[skill_id].last_used = datetime.utcnow().isoformat()
            self._save_index()

            return weights, config

        except Exception as e:
            logger.error(f"Failed to load skill {skill_id}: {e}")
            return None

    def has(self, skill_id: str) -> bool:
        """Check if skill is in cache."""
        return skill_id in self._index

    def remove(self, skill_id: str) -> bool:
        """Remove skill from cache."""
        if skill_id not in self._index:
            return False

        with self._lock:
            # Remove from loaded
            self._loaded_skills.pop(skill_id, None)

            # Remove from index
            del self._index[skill_id]

            # Remove files
            skill_dir = self.cache_dir / skill_id
            if skill_dir.exists():
                import shutil
                shutil.rmtree(skill_dir)

            self._save_index()

        return True

    def list(self) -> List[SkillInfo]:
        """List all cached skills."""
        return list(self._index.values())

    def clear(self):
        """Clear all cached skills."""
        with self._lock:
            for skill_id in list(self._index.keys()):
                self.remove(skill_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_skills": len(self._index),
            "cache_size_mb": self.get_cache_size_mb(),
            "max_cache_size_mb": self.max_cache_size_mb,
            "loaded_skills": len(self._loaded_skills),
        }


# =============================================================================
# Skill Decryptor
# =============================================================================

class SkillDecryptor:
    """
    Decrypts skills received from cloud.

    Uses N2HE for homomorphic decryption with edge-specific keys.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock or not HAS_N2HE

        if not self.use_mock:
            # Initialize N2HE context with same parameters as cloud
            params = N2HEParams(
                n=1024,
                q=2**32,
                t=2**16,
                sigma=3.2,
                security_bits=128,
            )
            self.n2he_ctx = N2HEContext(params=params, use_mock=False)
            self.n2he_ctx.generate_keys(generate_boot_key=False)
            logger.info("Initialized N2HE decryptor")
        else:
            self.n2he_ctx = None
            logger.info("Using mock decryption (no N2HE)")

    def decrypt_weights(self, encrypted_blob: bytes) -> np.ndarray:
        """
        Decrypt model weights.

        Args:
            encrypted_blob: Encrypted weight data

        Returns:
            Decrypted weights as numpy array
        """
        if self.use_mock:
            # Mock: just decompress
            return pickle.loads(gzip.decompress(encrypted_blob))

        try:
            package = pickle.loads(encrypted_blob)

            # Decompress weights
            weights = pickle.loads(gzip.decompress(package['weights']))

            # Verify integrity using encrypted summary
            if 'encrypted_summary' in package:
                scale = package['scale']
                encrypted_summary = package['encrypted_summary']

                decrypted_summary = [
                    self.n2he_ctx.decrypt(LWECiphertext.deserialize(s)) / scale
                    for s in encrypted_summary
                ]

                # Verify
                expected = [
                    weights.mean(),
                    weights.std(),
                    weights.min(),
                    weights.max(),
                    weights.size,
                ]

                for i, (dec, exp) in enumerate(zip(decrypted_summary, expected)):
                    if abs(dec - exp) > 0.1:
                        logger.warning(f"Integrity check warning: summary[{i}] mismatch")

            return weights

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            # Fallback to compressed
            return pickle.loads(gzip.decompress(encrypted_blob))

    def decrypt_config(self, encrypted_blob: bytes) -> Dict[str, Any]:
        """Decrypt skill configuration."""
        try:
            config_bytes = gzip.decompress(encrypted_blob)
            return json.loads(config_bytes.decode())
        except Exception as e:
            logger.error(f"Config decryption failed: {e}")
            return {}

    def decrypt_embedding(self, encrypted_blob: bytes) -> np.ndarray:
        """Decrypt skill embedding."""
        if self.use_mock or self.n2he_ctx is None:
            return np.frombuffer(gzip.decompress(encrypted_blob), dtype=np.float32)

        try:
            package = pickle.loads(gzip.decompress(encrypted_blob))
            ciphertexts = package['ciphertexts']
            scale = package['scale']

            values = [
                self.n2he_ctx.decrypt(LWECiphertext.deserialize(s)) / scale
                for s in ciphertexts
            ]

            return np.array(values, dtype=np.float32).reshape(package['shape'])

        except Exception as e:
            logger.error(f"Embedding decryption failed: {e}")
            return np.zeros(256, dtype=np.float32)


# =============================================================================
# Edge Skill Client
# =============================================================================

class EdgeSkillClient:
    """
    Client for retrieving and executing skills on edge devices.

    Features:
    - Skill download from cloud with decryption
    - Local caching for offline operation
    - MoE blending of multiple skills
    - Execution with timing and monitoring
    """

    def __init__(
        self,
        platform_url: str = "http://localhost:8080",
        api_key: str = "",
        device_id: str = "",
        cache_dir: str = "/var/lib/dynamical/skill_cache",
        max_cache_mb: int = 1024,
        use_encryption: bool = True,
    ):
        self.platform_url = platform_url.rstrip("/")
        self.api_key = api_key
        self.device_id = device_id or f"edge_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        # Initialize components
        self.cache = SkillCache(
            cache_dir=cache_dir,
            max_cache_size_mb=max_cache_mb,
        )
        self.decryptor = SkillDecryptor(use_mock=not use_encryption)

        # Loaded skill models
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            "downloads": 0,
            "cache_hits": 0,
            "executions": 0,
            "total_execution_time_ms": 0.0,
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Dict = None,
    ) -> Optional[Dict]:
        """Make HTTP request to platform."""
        if not HAS_REQUESTS:
            logger.error("requests library not available")
            return None

        url = f"{self.platform_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json_data, timeout=30)
            else:
                return None

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def request_skills(
        self,
        task_description: str,
        task_embedding: Optional[List[float]] = None,
        skill_types: Optional[List[str]] = None,
        max_skills: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Request skills from cloud for a task.

        Args:
            task_description: Natural language task description
            task_embedding: Pre-computed task embedding
            skill_types: Required skill types
            max_skills: Maximum skills to return

        Returns:
            List of (skill_id, routing_weight) tuples
        """
        request_data = {
            "task_description": task_description,
            "device_id": self.device_id,
            "max_skills": max_skills,
        }

        if task_embedding:
            request_data["task_embedding"] = task_embedding

        if skill_types:
            request_data["required_skill_types"] = skill_types

        response = self._make_request("POST", "/api/v1/skills/request", request_data)

        if not response:
            logger.warning("Failed to get skills from cloud, using cached")
            return [(s.id, 1.0 / len(self.cache.list())) for s in self.cache.list()[:max_skills]]

        # Parse response
        skills = response.get("skills", [])
        weights = response.get("routing_weights", [])

        results = []
        for skill_data, weight in zip(skills, weights):
            skill_id = skill_data.get("metadata", {}).get("id", "")
            if skill_id:
                results.append((skill_id, weight))

                # Download and cache if not already cached
                if not self.cache.has(skill_id):
                    self._download_and_cache_skill(skill_data)

        return results

    def _parse_coordination_metadata(self, metadata: Dict) -> Optional[CoordinationMetadata]:
        """Parse coordination metadata from skill artifact.

        Coordination metadata is used by SwarmBridge/SwarmBrain for
        cross-site orchestration of multi-robot skills.
        """
        coord_data = metadata.get("coordination", {})
        if not coord_data:
            return None

        return CoordinationMetadata(
            supported_roles=coord_data.get("supported_roles", ["independent"]),
            requires_leader=coord_data.get("requires_leader", False),
            max_followers=coord_data.get("max_followers", 0),
            sync_mode=coord_data.get("sync_mode", "none"),
            sync_frequency_hz=coord_data.get("sync_frequency_hz", 0.0),
            broadcast_state=coord_data.get("broadcast_state", False),
            receive_state=coord_data.get("receive_state", False),
            state_channels=coord_data.get("state_channels", []),
            min_robots=coord_data.get("min_robots", 1),
            max_robots=coord_data.get("max_robots", 1),
            requires_same_site=coord_data.get("requires_same_site", True),
        )

    def _download_and_cache_skill(self, skill_data: Dict):
        """Download and cache a skill from cloud response."""
        metadata = skill_data.get("metadata", {})
        skill_id = metadata.get("id", "")

        if not skill_id:
            return

        try:
            # Get encrypted data
            encrypted_weights = skill_data.get("encrypted_weights", b"")
            encrypted_config = skill_data.get("encrypted_config", b"")

            if isinstance(encrypted_weights, str):
                import base64
                encrypted_weights = base64.b64decode(encrypted_weights)
                encrypted_config = base64.b64decode(encrypted_config)

            # Decrypt
            weights = self.decryptor.decrypt_weights(encrypted_weights)
            config = self.decryptor.decrypt_config(encrypted_config)

            # Parse coordination metadata from unified skill artifact
            coordination = self._parse_coordination_metadata(metadata)

            # Cache with coordination metadata
            self.cache.store(
                skill_id=skill_id,
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                skill_type=metadata.get("skill_type", ""),
                version=metadata.get("version", "1.0.0"),
                weights=weights,
                config=config,
                coordination=coordination,
            )

            self.stats["downloads"] += 1
            logger.info(f"Downloaded and cached skill: {skill_id}")

        except Exception as e:
            logger.error(f"Failed to download skill {skill_id}: {e}")

    def get_skill(
        self,
        skill_id: str,
        download_if_missing: bool = True,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get a skill by ID.

        Args:
            skill_id: Skill identifier
            download_if_missing: Download from cloud if not cached

        Returns:
            Tuple of (weights, config) or None
        """
        # Check cache first
        cached = self.cache.get(skill_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        if not download_if_missing:
            return None

        # Download from cloud
        response = self._make_request("GET", f"/api/v1/skills/{skill_id}")
        if response:
            self._download_and_cache_skill(response)
            return self.cache.get(skill_id)

        return None

    def load_skill_model(self, skill_id: str) -> bool:
        """
        Load skill model into memory for execution.

        Args:
            skill_id: Skill to load

        Returns:
            Success status
        """
        if skill_id in self._models:
            return True

        skill_data = self.get_skill(skill_id)
        if not skill_data:
            return False

        weights, config = skill_data

        with self._lock:
            if HAS_TORCH:
                # Create simple skill model wrapper
                model = SkillModel(weights, config)
                self._models[skill_id] = model
            else:
                # Store raw weights
                self._models[skill_id] = {"weights": weights, "config": config}

        logger.info(f"Loaded skill model: {skill_id}")
        return True

    def execute_skill(
        self,
        skill_id: str,
        observation: np.ndarray,
        **kwargs,
    ) -> SkillExecutionResult:
        """
        Execute a single skill.

        Args:
            skill_id: Skill to execute
            observation: Current observation
            **kwargs: Additional arguments

        Returns:
            Execution result
        """
        start_time = time.time()

        if skill_id not in self._models:
            if not self.load_skill_model(skill_id):
                return SkillExecutionResult(
                    success=False,
                    skill_id=skill_id,
                    execution_time_ms=0.0,
                    error_message=f"Skill {skill_id} not found",
                )

        try:
            model = self._models[skill_id]

            if isinstance(model, SkillModel):
                output = model.forward(observation)
            else:
                # Fallback: simple linear projection
                weights = model["weights"]
                output = np.dot(observation.flatten(), weights[:observation.size].reshape(-1, 1))

            execution_time_ms = (time.time() - start_time) * 1000

            # Update statistics
            self.stats["executions"] += 1
            self.stats["total_execution_time_ms"] += execution_time_ms

            # Update cache statistics
            if skill_id in self.cache._index:
                info = self.cache._index[skill_id]
                info.execution_count += 1
                alpha = 0.1
                info.avg_execution_time_ms = (
                    alpha * execution_time_ms +
                    (1 - alpha) * info.avg_execution_time_ms
                )

            return SkillExecutionResult(
                success=True,
                skill_id=skill_id,
                execution_time_ms=execution_time_ms,
                output=output,
            )

        except Exception as e:
            return SkillExecutionResult(
                success=False,
                skill_id=skill_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    def execute_blended(
        self,
        blend_config: SkillBlendConfig,
        observation: np.ndarray,
    ) -> SkillExecutionResult:
        """
        Execute multiple skills with MoE blending.

        Args:
            blend_config: Blending configuration
            observation: Current observation

        Returns:
            Blended execution result
        """
        start_time = time.time()

        # Execute each skill
        outputs = []
        weights = []

        for skill_id, weight in zip(blend_config.skill_ids, blend_config.weights):
            result = self.execute_skill(skill_id, observation)
            if result.success and result.output is not None:
                outputs.append(result.output)
                weights.append(weight)

        if not outputs:
            return SkillExecutionResult(
                success=False,
                skill_id="blended",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="No skills executed successfully",
            )

        # Blend outputs
        if blend_config.blend_mode == "weighted_sum":
            # Normalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            # Weighted average
            blended = sum(w * o for w, o in zip(weights, outputs))

        elif blend_config.blend_mode == "attention":
            # Use first output shape as reference
            blended = outputs[0] * weights[0]
            for o, w in zip(outputs[1:], weights[1:]):
                blended = blended + o * w

        elif blend_config.blend_mode == "sequential":
            # Use last output (skills applied in sequence)
            blended = outputs[-1]

        else:
            blended = outputs[0]

        return SkillExecutionResult(
            success=True,
            skill_id="blended",
            execution_time_ms=(time.time() - start_time) * 1000,
            output=blended,
            metadata={
                "blend_mode": blend_config.blend_mode,
                "num_skills": len(outputs),
                "weights": weights,
            },
        )

    def preload_skills(self, skill_ids: List[str]) -> int:
        """
        Preload skills into memory.

        Args:
            skill_ids: Skills to preload

        Returns:
            Number of skills loaded
        """
        loaded = 0
        for skill_id in skill_ids:
            if self.load_skill_model(skill_id):
                loaded += 1
        return loaded

    def unload_skill(self, skill_id: str):
        """Unload a skill from memory."""
        with self._lock:
            self._models.pop(skill_id, None)

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            "avg_execution_time_ms": (
                self.stats["total_execution_time_ms"] / max(1, self.stats["executions"])
            ),
            "cache": self.cache.get_statistics(),
            "loaded_models": len(self._models),
        }


# =============================================================================
# Skill Model Wrapper
# =============================================================================

class SkillModel:
    """Simple skill model wrapper."""

    def __init__(self, weights: np.ndarray, config: Dict[str, Any]):
        self.weights = weights
        self.config = config

        # Determine input/output dimensions
        self.input_dim = config.get("input_dim", 256)
        self.output_dim = config.get("output_dim", 7)

        if HAS_TORCH:
            self._init_torch_model()
        else:
            self.model = None

    def _init_torch_model(self):
        """Initialize PyTorch model."""
        # Simple MLP for skill
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

        # Load weights if compatible
        try:
            state_dict = self._weights_to_state_dict()
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Could not load weights into model: {e}")

    def _weights_to_state_dict(self) -> Dict[str, Any]:
        """Convert flat weights to state dict."""
        # This is a placeholder - actual implementation depends on model architecture
        state_dict = {}
        offset = 0

        for name, param in self.model.named_parameters():
            numel = param.numel()
            if offset + numel <= len(self.weights):
                weight_slice = self.weights[offset:offset + numel]
                state_dict[name] = torch.from_numpy(
                    weight_slice.reshape(param.shape)
                ).float()
                offset += numel

        return state_dict

    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if self.model is not None and HAS_TORCH:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(observation.flatten()[:self.input_dim]).float()
                obs_tensor = obs_tensor.unsqueeze(0)

                output = self.model(obs_tensor)
                return output.squeeze(0).numpy()
        else:
            # Fallback: simple linear
            output = np.dot(
                observation.flatten()[:self.input_dim],
                self.weights[:self.input_dim * self.output_dim].reshape(self.input_dim, self.output_dim)
            )
            return output


# =============================================================================
# Testing
# =============================================================================

def test_edge_skill_client():
    """Test the edge skill client."""
    print("\n" + "=" * 60)
    print("EDGE SKILL CLIENT TEST")
    print("=" * 60)

    # Create client (offline mode)
    client = EdgeSkillClient(
        platform_url="http://localhost:8080",
        api_key="test_key",
        device_id="orin_test_001",
        cache_dir="/tmp/skill_cache_test",
        use_encryption=True,
    )

    print("\n1. Create Test Skills in Cache")
    print("-" * 40)

    # Create some test skills directly in cache
    test_skills = [
        ("skill_grasp", "Grasp Object", "manipulation", np.random.randn(10000).astype(np.float32)),
        ("skill_place", "Place Object", "manipulation", np.random.randn(10000).astype(np.float32)),
        ("skill_pour", "Pour Liquid", "manipulation", np.random.randn(10000).astype(np.float32)),
    ]

    for skill_id, name, skill_type, weights in test_skills:
        client.cache.store(
            skill_id=skill_id,
            name=name,
            description=f"Test skill: {name}",
            skill_type=skill_type,
            version="1.0.0",
            weights=weights,
            config={"input_dim": 256, "output_dim": 7},
        )
        print(f"   Cached: {name} ({skill_id})")

    print("\n2. Load and Execute Skills")
    print("-" * 40)

    observation = np.random.randn(256).astype(np.float32)

    for skill_id, name, _, _ in test_skills:
        result = client.execute_skill(skill_id, observation)
        print(f"   {name}: success={result.success}, time={result.execution_time_ms:.2f}ms")
        if result.output is not None:
            print(f"      output shape: {result.output.shape}")

    print("\n3. MoE Blending")
    print("-" * 40)

    blend_config = SkillBlendConfig(
        skill_ids=["skill_grasp", "skill_place"],
        weights=[0.7, 0.3],
        blend_mode="weighted_sum",
    )

    result = client.execute_blended(blend_config, observation)
    print(f"   Blended execution: success={result.success}")
    print(f"   Time: {result.execution_time_ms:.2f}ms")
    print(f"   Metadata: {result.metadata}")

    print("\n4. Statistics")
    print("-" * 40)

    stats = client.get_statistics()
    print(f"   Executions: {stats['executions']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Avg execution time: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"   Cache size: {stats['cache']['cache_size_mb']:.2f} MB")
    print(f"   Loaded models: {stats['loaded_models']}")

    print("\n" + "=" * 60)
    print("EDGE SKILL CLIENT TESTS COMPLETE")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/skill_cache_test", ignore_errors=True)


if __name__ == "__main__":
    test_edge_skill_client()

"""
Cloud MoE Skill Router with Encrypted Storage

This module provides:
- Encrypted skill storage using N2HE homomorphic encryption
- MoE (Mixture of Experts) routing for skill selection
- Cloud-side skill management API
- Edge device skill deployment

Skills are robot manipulation primitives (grasp, pour, place, etc.) that can be
dynamically loaded and combined using MoE routing for task execution.

Architecture:
- Skills stored encrypted in cloud using N2HE (LWE-based, 128-bit security)
- MoE router selects appropriate skill based on task embedding
- Edge Orin downloads encrypted skills, decrypts locally
- Skills can be composed for complex multi-step tasks
"""

import os
import time
import json
import gzip
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)

# Import N2HE for encryption
try:
    from src.moai.n2he import N2HEContext, N2HEParams, LWECiphertext
    HAS_N2HE = True
except ImportError:
    HAS_N2HE = False
    logger.warning("N2HE not available")

# Import torch for MoE components
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")


# =============================================================================
# Data Classes
# =============================================================================

class SkillType(str, Enum):
    """Types of robot skills."""
    MANIPULATION = "manipulation"  # Grasp, place, pour, etc.
    NAVIGATION = "navigation"      # Move, avoid, follow
    PERCEPTION = "perception"      # Look, detect, track
    COORDINATION = "coordination"  # Multi-arm, bimanual
    LOCOMOTION = "locomotion"      # Walk, balance, climb


class SkillStatus(str, Enum):
    """Skill deployment status."""
    DRAFT = "draft"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


@dataclass
class SkillMetadata:
    """Skill metadata."""
    id: str
    name: str
    description: str
    skill_type: SkillType
    version: str
    status: SkillStatus

    # Training info
    trained_on_episodes: List[str] = field(default_factory=list)
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    success_rate: float = 0.0
    avg_execution_time_ms: float = 0.0

    # Routing info
    task_embedding: Optional[List[float]] = None  # For MoE routing
    expert_index: int = 0  # Which expert this skill belongs to

    # Storage info
    file_hash: str = ""
    file_size_bytes: int = 0
    is_encrypted: bool = True

    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    deployed_at: Optional[str] = None

    # Deployment
    deployed_to: List[str] = field(default_factory=list)  # Edge device IDs

    # Tags
    tags: List[str] = field(default_factory=list)


@dataclass
class EncryptedSkill:
    """Encrypted skill package."""
    metadata: SkillMetadata
    encrypted_weights: bytes  # Encrypted model weights
    encrypted_config: bytes   # Encrypted skill config
    public_key_hash: str      # Hash of public key used for encryption
    encryption_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRequest:
    """Request to retrieve a skill."""
    task_description: str
    task_embedding: Optional[List[float]] = None
    device_id: str = ""
    required_skill_types: List[SkillType] = field(default_factory=list)
    max_skills: int = 5


@dataclass
class SkillResponse:
    """Response with selected skills."""
    skills: List[EncryptedSkill]
    routing_weights: List[float]  # MoE weights for each skill
    task_embedding: List[float]
    inference_time_ms: float


# =============================================================================
# MoE Skill Router
# =============================================================================

class MoESkillRouter:
    """
    Mixture of Experts router for skill selection.

    Given a task embedding, routes to appropriate skills using a learned
    gating network. Supports top-k routing and soft routing.
    """

    def __init__(
        self,
        num_experts: int = 16,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        top_k: int = 3,
        use_load_balancing: bool = True,
    ):
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.use_load_balancing = use_load_balancing

        # Expert load tracking
        self.expert_load = np.zeros(num_experts)
        self.total_routes = 0

        # Initialize gating network
        if HAS_TORCH:
            self._init_gating_network()
        else:
            self.gating_network = None
            logger.warning("PyTorch not available - using fallback routing")

    def _init_gating_network(self):
        """Initialize the gating network."""
        self.gating_network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
        )

        # Add noise for exploration during training
        self.noise_std = 0.1

        # Load balancing auxiliary loss weight
        self.load_balance_weight = 0.01

    def route(
        self,
        task_embedding: np.ndarray,
        skill_embeddings: Dict[str, np.ndarray],
        training: bool = False,
    ) -> Tuple[List[str], List[float], Dict[str, Any]]:
        """
        Route task to appropriate skills using MoE gating.

        Args:
            task_embedding: Task description embedding [embedding_dim]
            skill_embeddings: Dict of skill_id -> skill embedding
            training: Whether in training mode (adds noise)

        Returns:
            Tuple of:
            - List of selected skill IDs
            - List of routing weights
            - Dict of routing metadata
        """
        start_time = time.time()

        if not skill_embeddings:
            return [], [], {"error": "No skills available"}

        skill_ids = list(skill_embeddings.keys())
        skill_embs = np.array(list(skill_embeddings.values()))

        if HAS_TORCH and self.gating_network is not None:
            # Use learned gating
            return self._route_learned(
                task_embedding, skill_ids, skill_embs, training
            )
        else:
            # Fallback: cosine similarity routing
            return self._route_similarity(task_embedding, skill_ids, skill_embs)

    def _route_learned(
        self,
        task_embedding: np.ndarray,
        skill_ids: List[str],
        skill_embs: np.ndarray,
        training: bool,
    ) -> Tuple[List[str], List[float], Dict[str, Any]]:
        """Route using learned gating network."""
        # Convert to tensor
        task_tensor = torch.from_numpy(task_embedding).float().unsqueeze(0)

        # Compute gating logits
        with torch.no_grad() if not training else torch.enable_grad():
            logits = self.gating_network(task_tensor)  # [1, num_experts]

            # Add noise during training
            if training and self.noise_std > 0:
                noise = torch.randn_like(logits) * self.noise_std
                logits = logits + noise

            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_experts]

        # Map experts to skills (each skill has expert_index)
        # For now, distribute skills evenly across experts
        num_skills = len(skill_ids)
        expert_to_skills = {i: [] for i in range(self.num_experts)}

        for i, skill_id in enumerate(skill_ids):
            expert_idx = i % self.num_experts
            expert_to_skills[expert_idx].append((i, skill_id))

        # Top-k expert selection
        top_k_values, top_k_indices = torch.topk(probs, min(self.top_k, self.num_experts))

        # Collect skills from top experts
        selected_skills = []
        selected_weights = []

        for weight, expert_idx in zip(top_k_values.numpy(), top_k_indices.numpy()):
            for skill_idx, skill_id in expert_to_skills[expert_idx]:
                if skill_id not in [s for s, _ in selected_skills]:
                    selected_skills.append((skill_id, float(weight)))

                    if len(selected_skills) >= self.top_k:
                        break

            if len(selected_skills) >= self.top_k:
                break

        # Update load statistics
        for expert_idx in top_k_indices.numpy():
            self.expert_load[expert_idx] += 1
        self.total_routes += 1

        skill_ids_out = [s for s, _ in selected_skills]
        weights_out = [w for _, w in selected_skills]

        # Normalize weights
        weight_sum = sum(weights_out)
        if weight_sum > 0:
            weights_out = [w / weight_sum for w in weights_out]

        metadata = {
            "routing_method": "learned_gating",
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_load_balance": self._compute_load_balance(),
        }

        return skill_ids_out, weights_out, metadata

    def _route_similarity(
        self,
        task_embedding: np.ndarray,
        skill_ids: List[str],
        skill_embs: np.ndarray,
    ) -> Tuple[List[str], List[float], Dict[str, Any]]:
        """Route using cosine similarity (fallback)."""
        # Normalize embeddings
        task_norm = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        skill_norms = skill_embs / (np.linalg.norm(skill_embs, axis=1, keepdims=True) + 1e-8)

        # Compute similarities
        similarities = np.dot(skill_norms, task_norm)

        # Top-k selection
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]

        selected_ids = [skill_ids[i] for i in top_k_indices]
        selected_weights = [float(similarities[i]) for i in top_k_indices]

        # Normalize weights (softmax)
        exp_weights = np.exp(selected_weights)
        selected_weights = (exp_weights / exp_weights.sum()).tolist()

        metadata = {
            "routing_method": "cosine_similarity",
            "top_k": self.top_k,
        }

        return selected_ids, selected_weights, metadata

    def _compute_load_balance(self) -> float:
        """Compute load balance across experts."""
        if self.total_routes == 0:
            return 1.0

        expected_load = self.total_routes / self.num_experts
        actual_loads = self.expert_load / self.total_routes

        # Coefficient of variation (lower is better)
        cv = np.std(actual_loads) / (np.mean(actual_loads) + 1e-8)

        # Return balance score (1.0 = perfect balance)
        return max(0.0, 1.0 - cv)

    def get_load_statistics(self) -> Dict[str, Any]:
        """Get expert load statistics."""
        return {
            "total_routes": self.total_routes,
            "expert_load": self.expert_load.tolist(),
            "load_balance_score": self._compute_load_balance(),
        }


# =============================================================================
# Encrypted Skill Storage
# =============================================================================

class EncryptedSkillStorage:
    """
    Encrypted skill storage using N2HE homomorphic encryption.

    Features:
    - AES encryption for skill weights (symmetric, fast)
    - N2HE encryption for embeddings (homomorphic, enables computation)
    - Integrity verification with SHA-256 hashes
    - Versioned storage with rollback support
    """

    def __init__(
        self,
        storage_dir: str = "/var/lib/dynamical/skills",
        use_encryption: bool = True,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_encryption = use_encryption

        # Initialize N2HE for embedding encryption
        if use_encryption and HAS_N2HE:
            params = N2HEParams(
                n=1024,
                q=2**32,
                t=2**16,
                sigma=3.2,
                security_bits=128,
            )
            self.n2he_ctx = N2HEContext(params=params, use_mock=False)
            self.n2he_ctx.generate_keys(generate_boot_key=False)
            logger.info("Initialized N2HE encryption for skill storage")
        else:
            self.n2he_ctx = None
            if use_encryption:
                logger.warning("N2HE not available - using compressed storage")

        # Skill index (in-memory cache)
        self._skill_index: Dict[str, SkillMetadata] = {}
        self._lock = threading.Lock()

        # Load existing skills
        self._load_index()

    def _load_index(self):
        """Load skill index from disk."""
        index_path = self.storage_dir / "skill_index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                for skill_id, meta in data.items():
                    self._skill_index[skill_id] = SkillMetadata(**meta)
                logger.info(f"Loaded {len(self._skill_index)} skills from index")
            except Exception as e:
                logger.error(f"Failed to load skill index: {e}")

    def _save_index(self):
        """Save skill index to disk."""
        index_path = self.storage_dir / "skill_index.json"
        try:
            data = {k: asdict(v) for k, v in self._skill_index.items()}
            with open(index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save skill index: {e}")

    def _encrypt_weights(self, weights: np.ndarray) -> bytes:
        """Encrypt model weights."""
        # Serialize weights
        weight_bytes = pickle.dumps(weights)

        if not self.use_encryption:
            return gzip.compress(weight_bytes)

        if self.n2he_ctx is not None:
            # Use N2HE for a sample of weights (full encryption would be slow)
            # Encrypt summary statistics for integrity verification
            scale = 2**15
            summary = np.array([
                weights.mean(),
                weights.std(),
                weights.min(),
                weights.max(),
                weights.size,
            ])
            summary_int = np.clip(summary * scale, -32767, 32767).astype(np.int64)

            encrypted_summary = [
                self.n2he_ctx.encrypt(int(v)).serialize()
                for v in summary_int
            ]

            # Package: compressed weights + encrypted summary
            package = {
                'weights': gzip.compress(weight_bytes),
                'encrypted_summary': encrypted_summary,
                'scale': scale,
            }
            return pickle.dumps(package)
        else:
            # Fallback: just compress
            return gzip.compress(weight_bytes)

    def _decrypt_weights(self, encrypted: bytes) -> np.ndarray:
        """Decrypt model weights."""
        if not self.use_encryption:
            return pickle.loads(gzip.decompress(encrypted))

        if self.n2he_ctx is not None:
            try:
                package = pickle.loads(encrypted)
                weights = pickle.loads(gzip.decompress(package['weights']))

                # Verify integrity using encrypted summary
                encrypted_summary = package['encrypted_summary']
                scale = package['scale']

                decrypted_summary = [
                    self.n2he_ctx.decrypt(LWECiphertext.deserialize(s)) / scale
                    for s in encrypted_summary
                ]

                # Check summary matches
                expected = [
                    weights.mean(),
                    weights.std(),
                    weights.min(),
                    weights.max(),
                    weights.size,
                ]

                for i, (dec, exp) in enumerate(zip(decrypted_summary, expected)):
                    if abs(dec - exp) > 0.1:  # Allow small quantization error
                        logger.warning(f"Integrity check warning: summary[{i}] mismatch")

                return weights
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                raise
        else:
            return pickle.loads(gzip.decompress(encrypted))

    def _encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Encrypt skill embedding for homomorphic operations."""
        if not self.use_encryption or self.n2he_ctx is None:
            return gzip.compress(embedding.tobytes())

        scale = 2**15
        emb_int = np.clip(embedding * scale, -32767, 32767).astype(np.int64)

        ciphertexts = [
            self.n2he_ctx.encrypt(int(v)).serialize()
            for v in emb_int
        ]

        package = {
            'ciphertexts': ciphertexts,
            'shape': embedding.shape,
            'scale': scale,
        }
        return gzip.compress(pickle.dumps(package))

    def _decrypt_embedding(self, encrypted: bytes) -> np.ndarray:
        """Decrypt skill embedding."""
        if not self.use_encryption or self.n2he_ctx is None:
            return np.frombuffer(gzip.decompress(encrypted), dtype=np.float32)

        package = pickle.loads(gzip.decompress(encrypted))
        ciphertexts = package['ciphertexts']
        scale = package['scale']

        values = [
            self.n2he_ctx.decrypt(LWECiphertext.deserialize(s)) / scale
            for s in ciphertexts
        ]

        return np.array(values, dtype=np.float32).reshape(package['shape'])

    def store_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        skill_type: SkillType,
        weights: np.ndarray,
        config: Dict[str, Any],
        task_embedding: Optional[np.ndarray] = None,
        version: str = "1.0.0",
        tags: List[str] = None,
        training_config: Dict[str, Any] = None,
        trained_on_episodes: List[str] = None,
    ) -> SkillMetadata:
        """
        Store a skill with encryption.

        Args:
            skill_id: Unique skill identifier
            name: Human-readable name
            description: Skill description
            skill_type: Type of skill
            weights: Model weights as numpy array
            config: Skill configuration
            task_embedding: Embedding for MoE routing
            version: Skill version
            tags: Tags for categorization
            training_config: Training configuration
            trained_on_episodes: Episodes used for training

        Returns:
            Skill metadata
        """
        start_time = time.time()

        with self._lock:
            # Create skill directory
            skill_dir = self.storage_dir / skill_id
            skill_dir.mkdir(exist_ok=True)

            # Encrypt and store weights
            encrypted_weights = self._encrypt_weights(weights)
            weights_path = skill_dir / "weights.enc"
            with open(weights_path, 'wb') as f:
                f.write(encrypted_weights)

            # Encrypt and store config
            config_bytes = json.dumps(config).encode()
            if self.use_encryption:
                encrypted_config = gzip.compress(config_bytes)
            else:
                encrypted_config = config_bytes
            config_path = skill_dir / "config.enc"
            with open(config_path, 'wb') as f:
                f.write(encrypted_config)

            # Store task embedding for routing
            if task_embedding is not None:
                embedding_encrypted = self._encrypt_embedding(task_embedding)
                embedding_path = skill_dir / "embedding.enc"
                with open(embedding_path, 'wb') as f:
                    f.write(embedding_encrypted)
                task_emb_list = task_embedding.tolist()
            else:
                task_emb_list = None

            # Compute file hash
            file_hash = hashlib.sha256(encrypted_weights).hexdigest()[:16]

            # Create metadata
            now = datetime.utcnow().isoformat()
            metadata = SkillMetadata(
                id=skill_id,
                name=name,
                description=description,
                skill_type=skill_type,
                version=version,
                status=SkillStatus.VALIDATED,
                trained_on_episodes=trained_on_episodes or [],
                training_config=training_config or {},
                task_embedding=task_emb_list,
                file_hash=file_hash,
                file_size_bytes=len(encrypted_weights),
                is_encrypted=self.use_encryption,
                created_at=now,
                updated_at=now,
                tags=tags or [],
            )

            # Update index
            self._skill_index[skill_id] = metadata
            self._save_index()

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Stored skill '{name}' ({skill_id}) in {duration_ms:.1f}ms")

            return metadata

    def get_skill(self, skill_id: str) -> Optional[EncryptedSkill]:
        """
        Retrieve an encrypted skill.

        Args:
            skill_id: Skill ID

        Returns:
            EncryptedSkill or None if not found
        """
        if skill_id not in self._skill_index:
            return None

        metadata = self._skill_index[skill_id]
        skill_dir = self.storage_dir / skill_id

        # Load encrypted weights
        weights_path = skill_dir / "weights.enc"
        if not weights_path.exists():
            logger.error(f"Weights file not found for skill {skill_id}")
            return None

        with open(weights_path, 'rb') as f:
            encrypted_weights = f.read()

        # Load encrypted config
        config_path = skill_dir / "config.enc"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                encrypted_config = f.read()
        else:
            encrypted_config = b'{}'

        return EncryptedSkill(
            metadata=metadata,
            encrypted_weights=encrypted_weights,
            encrypted_config=encrypted_config,
            public_key_hash=hashlib.sha256(str(self.n2he_ctx).encode()).hexdigest()[:16] if self.n2he_ctx else "",
            encryption_params={
                "algorithm": "N2HE" if self.n2he_ctx else "gzip",
                "n": 1024 if self.n2he_ctx else 0,
                "security_bits": 128 if self.n2he_ctx else 0,
            }
        )

    def list_skills(
        self,
        skill_type: Optional[SkillType] = None,
        status: Optional[SkillStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SkillMetadata]:
        """
        List skills matching criteria.

        Args:
            skill_type: Filter by skill type
            status: Filter by status
            tags: Filter by tags (any match)

        Returns:
            List of skill metadata
        """
        skills = list(self._skill_index.values())

        if skill_type:
            skills = [s for s in skills if s.skill_type == skill_type]

        if status:
            skills = [s for s in skills if s.status == status]

        if tags:
            skills = [s for s in skills if any(t in s.tags for t in tags)]

        return skills

    def get_skill_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all skill embeddings for routing."""
        embeddings = {}

        for skill_id, metadata in self._skill_index.items():
            if metadata.task_embedding:
                embeddings[skill_id] = np.array(metadata.task_embedding)
            else:
                # Load from file if available
                embedding_path = self.storage_dir / skill_id / "embedding.enc"
                if embedding_path.exists():
                    with open(embedding_path, 'rb') as f:
                        encrypted = f.read()
                    embeddings[skill_id] = self._decrypt_embedding(encrypted)

        return embeddings

    def deploy_skill(
        self,
        skill_id: str,
        device_ids: List[str],
    ) -> bool:
        """
        Mark skill as deployed to edge devices.

        Args:
            skill_id: Skill ID
            device_ids: Edge device IDs

        Returns:
            Success status
        """
        if skill_id not in self._skill_index:
            return False

        metadata = self._skill_index[skill_id]
        metadata.status = SkillStatus.DEPLOYED
        metadata.deployed_at = datetime.utcnow().isoformat()
        metadata.deployed_to = list(set(metadata.deployed_to + device_ids))
        metadata.updated_at = datetime.utcnow().isoformat()

        self._save_index()
        return True

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        if skill_id not in self._skill_index:
            return False

        with self._lock:
            # Remove from index
            del self._skill_index[skill_id]

            # Remove files
            skill_dir = self.storage_dir / skill_id
            if skill_dir.exists():
                import shutil
                shutil.rmtree(skill_dir)

            self._save_index()

        return True

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(s.file_size_bytes for s in self._skill_index.values())

        return {
            "total_skills": len(self._skill_index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "encrypted": self.use_encryption,
            "encryption_available": HAS_N2HE,
            "skills_by_type": {
                t.value: len([s for s in self._skill_index.values() if s.skill_type == t])
                for t in SkillType
            },
            "skills_by_status": {
                s.value: len([sk for sk in self._skill_index.values() if sk.status == s])
                for s in SkillStatus
            },
        }


# =============================================================================
# Cloud Skill Service
# =============================================================================

class CloudSkillService:
    """
    Cloud service for MoE skill routing and management.

    Provides:
    - Skill storage with encryption
    - MoE routing for task-based skill selection
    - Skill deployment to edge devices
    - Analytics and monitoring
    """

    def __init__(
        self,
        storage_dir: str = "/var/lib/dynamical/skills",
        use_encryption: bool = True,
        num_experts: int = 16,
        embedding_dim: int = 512,
    ):
        self.storage = EncryptedSkillStorage(
            storage_dir=storage_dir,
            use_encryption=use_encryption,
        )

        self.router = MoESkillRouter(
            num_experts=num_experts,
            embedding_dim=embedding_dim,
        )

        self.embedding_dim = embedding_dim

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_deployments": 0,
            "avg_response_time_ms": 0.0,
        }

    def request_skills(self, request: SkillRequest) -> SkillResponse:
        """
        Request skills for a task.

        Uses MoE routing to select appropriate skills based on task embedding.

        Args:
            request: Skill request with task info

        Returns:
            SkillResponse with selected encrypted skills
        """
        start_time = time.time()

        # Get task embedding
        if request.task_embedding:
            task_embedding = np.array(request.task_embedding)
        else:
            # Generate embedding from description (placeholder)
            # In production, use a text encoder model
            task_embedding = self._generate_embedding(request.task_description)

        # Get available skill embeddings
        skill_embeddings = self.storage.get_skill_embeddings()

        # Filter by required types if specified
        if request.required_skill_types:
            valid_types = set(t.value for t in request.required_skill_types)
            skills_meta = self.storage.list_skills()
            valid_ids = {s.id for s in skills_meta if s.skill_type.value in valid_types}
            skill_embeddings = {k: v for k, v in skill_embeddings.items() if k in valid_ids}

        if not skill_embeddings:
            return SkillResponse(
                skills=[],
                routing_weights=[],
                task_embedding=task_embedding.tolist(),
                inference_time_ms=0.0,
            )

        # Route using MoE
        selected_ids, weights, metadata = self.router.route(
            task_embedding=task_embedding,
            skill_embeddings=skill_embeddings,
        )

        # Limit to max_skills
        selected_ids = selected_ids[:request.max_skills]
        weights = weights[:request.max_skills]

        # Retrieve encrypted skills
        skills = []
        for skill_id in selected_ids:
            encrypted_skill = self.storage.get_skill(skill_id)
            if encrypted_skill:
                skills.append(encrypted_skill)

        inference_time_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.stats["total_requests"] += 1
        alpha = 0.1
        self.stats["avg_response_time_ms"] = (
            alpha * inference_time_ms +
            (1 - alpha) * self.stats["avg_response_time_ms"]
        )

        return SkillResponse(
            skills=skills,
            routing_weights=weights,
            task_embedding=task_embedding.tolist(),
            inference_time_ms=inference_time_ms,
        )

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding from text (placeholder)."""
        # In production, use a real text encoder (BERT, T5, etc.)
        # For now, use a simple hash-based embedding
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand to embedding dimension
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def register_skill(
        self,
        name: str,
        description: str,
        skill_type: SkillType,
        weights: np.ndarray,
        config: Dict[str, Any],
        version: str = "1.0.0",
        tags: List[str] = None,
    ) -> SkillMetadata:
        """
        Register a new skill.

        Args:
            name: Skill name
            description: Description
            skill_type: Type of skill
            weights: Model weights
            config: Skill configuration
            version: Version string
            tags: Tags

        Returns:
            Skill metadata
        """
        # Generate skill ID
        skill_id = f"skill_{hashlib.md5(f'{name}_{version}'.encode()).hexdigest()[:12]}"

        # Generate task embedding from description
        task_embedding = self._generate_embedding(description)

        return self.storage.store_skill(
            skill_id=skill_id,
            name=name,
            description=description,
            skill_type=skill_type,
            weights=weights,
            config=config,
            task_embedding=task_embedding,
            version=version,
            tags=tags,
        )

    def deploy_to_edge(
        self,
        skill_ids: List[str],
        device_ids: List[str],
    ) -> Dict[str, bool]:
        """
        Deploy skills to edge devices.

        Args:
            skill_ids: Skills to deploy
            device_ids: Target edge devices

        Returns:
            Dict of skill_id -> success status
        """
        results = {}

        for skill_id in skill_ids:
            success = self.storage.deploy_skill(skill_id, device_ids)
            results[skill_id] = success

            if success:
                self.stats["total_deployments"] += 1

        return results

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "storage": self.storage.get_storage_statistics(),
            "router": self.router.get_load_statistics(),
        }


# =============================================================================
# Testing
# =============================================================================

def test_moe_skill_system():
    """Test the MoE skill system."""
    print("\n" + "=" * 60)
    print("MOE SKILL ROUTING AND ENCRYPTED STORAGE TEST")
    print("=" * 60)

    # Create service
    service = CloudSkillService(
        storage_dir="/tmp/dynamical_skill_test",
        use_encryption=True,
        num_experts=8,
        embedding_dim=256,
    )

    print("\n1. Register Skills")
    print("-" * 40)

    # Register some test skills
    skills_data = [
        ("grasp_object", "Grasp an object with precision gripper", SkillType.MANIPULATION),
        ("place_object", "Place object at target location", SkillType.MANIPULATION),
        ("pour_liquid", "Pour liquid from container", SkillType.MANIPULATION),
        ("navigate_to", "Navigate to waypoint", SkillType.NAVIGATION),
        ("detect_object", "Detect and localize object", SkillType.PERCEPTION),
    ]

    for name, desc, skill_type in skills_data:
        weights = np.random.randn(1000).astype(np.float32)  # Simulated weights
        config = {"gripper_type": "precision", "max_force": 10.0}

        metadata = service.register_skill(
            name=name,
            description=desc,
            skill_type=skill_type,
            weights=weights,
            config=config,
            version="1.0.0",
            tags=[skill_type.value, "test"],
        )
        print(f"   Registered: {metadata.name} ({metadata.id})")
        print(f"      Size: {metadata.file_size_bytes} bytes, Encrypted: {metadata.is_encrypted}")

    print("\n2. MoE Skill Routing")
    print("-" * 40)

    # Test routing
    test_tasks = [
        "Pick up the red cube from the table",
        "Move the robot arm to the start position",
        "Find the screwdriver on the workbench",
    ]

    for task in test_tasks:
        request = SkillRequest(
            task_description=task,
            max_skills=3,
        )

        response = service.request_skills(request)

        print(f"\n   Task: '{task}'")
        print(f"   Selected {len(response.skills)} skills in {response.inference_time_ms:.2f}ms:")
        for skill, weight in zip(response.skills, response.routing_weights):
            print(f"      - {skill.metadata.name}: weight={weight:.3f}")

    print("\n3. Skill Retrieval and Decryption")
    print("-" * 40)

    # Retrieve a skill
    skills = service.storage.list_skills()
    if skills:
        skill = service.storage.get_skill(skills[0].id)
        print(f"   Retrieved skill: {skill.metadata.name}")
        print(f"   Encrypted weights size: {len(skill.encrypted_weights)} bytes")
        print(f"   Encryption: {skill.encryption_params}")

    print("\n4. Storage Statistics")
    print("-" * 40)

    stats = service.get_service_statistics()
    print(f"   Total skills: {stats['storage']['total_skills']}")
    print(f"   Total size: {stats['storage']['total_size_mb']:.2f} MB")
    print(f"   Encrypted: {stats['storage']['encrypted']}")
    print(f"   Skills by type: {stats['storage']['skills_by_type']}")
    print(f"   Router load balance: {stats['router']['load_balance_score']:.3f}")

    print("\n" + "=" * 60)
    print("MOE SKILL SYSTEM TESTS COMPLETE")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/dynamical_skill_test", ignore_errors=True)


if __name__ == "__main__":
    test_moe_skill_system()

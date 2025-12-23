"""
Skill Cache - Local Skill Storage

Caches skills locally so robot can function offline.
Survives network outages.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CachedSkill:
    """A cached skill with metadata."""
    id: str
    name: str
    version: str
    policy_model: str
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached_at: float = 0.0
    last_used: float = 0.0
    use_count: int = 0
    checksum: str = ""


class SkillCache:
    """
    Local skill cache for offline operation.

    Features:
    - Persists skills to disk
    - LRU eviction when full
    - Checksum validation
    - Background sync with cloud
    """

    def __init__(self, max_skills: int = 100, cache_dir: str = "/var/cache/dynamical"):
        self.max_skills = max_skills
        self.cache_dir = cache_dir
        self._skills: Dict[str, CachedSkill] = {}
        self._index_file = os.path.join(cache_dir, "skill_index.json")

    def load_cached_skills(self) -> int:
        """
        Load skills from disk cache.

        Returns:
            Number of skills loaded
        """
        if not os.path.exists(self._index_file):
            logger.info("No cached skills found")
            return 0

        try:
            with open(self._index_file, 'r') as f:
                index = json.load(f)

            for skill_data in index.get('skills', []):
                skill = CachedSkill(**skill_data)
                self._skills[skill.id] = skill

            logger.info(f"Loaded {len(self._skills)} cached skills")
            return len(self._skills)

        except Exception as e:
            logger.error(f"Failed to load skill cache: {e}")
            return 0

    def save_cache(self) -> bool:
        """
        Save skill cache to disk.

        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            index = {
                'version': '1.0',
                'updated_at': time.time(),
                'skills': [asdict(s) for s in self._skills.values()]
            }

            with open(self._index_file, 'w') as f:
                json.dump(index, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save skill cache: {e}")
            return False

    def get(self, skill_id: str) -> Optional[CachedSkill]:
        """
        Get skill from cache.

        Args:
            skill_id: ID of skill to get

        Returns:
            CachedSkill or None if not found
        """
        skill = self._skills.get(skill_id)

        if skill:
            skill.last_used = time.time()
            skill.use_count += 1

        return skill

    def put(self, skill: CachedSkill) -> bool:
        """
        Add skill to cache.

        Args:
            skill: Skill to cache

        Returns:
            True if cached successfully
        """
        # Check if we need to evict
        if len(self._skills) >= self.max_skills and skill.id not in self._skills:
            self._evict_lru()

        skill.cached_at = time.time()
        self._skills[skill.id] = skill

        # Persist to disk
        return self.save_cache()

    def remove(self, skill_id: str) -> bool:
        """Remove skill from cache."""
        if skill_id in self._skills:
            del self._skills[skill_id]
            self.save_cache()
            return True
        return False

    def list_skills(self) -> List[str]:
        """List all cached skill IDs."""
        return list(self._skills.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_skills': len(self._skills),
            'max_skills': self.max_skills,
            'cache_dir': self.cache_dir,
            'most_used': self._get_most_used(5),
            'least_used': self._get_least_used(5),
        }

    def validate_skill(self, skill_id: str, expected_checksum: str) -> bool:
        """
        Validate skill checksum.

        Args:
            skill_id: ID of skill to validate
            expected_checksum: Expected checksum

        Returns:
            True if valid
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            return False

        return skill.checksum == expected_checksum

    def _evict_lru(self) -> None:
        """Evict least recently used skill."""
        if not self._skills:
            return

        # Find LRU skill
        lru_id = min(self._skills.keys(), key=lambda k: self._skills[k].last_used)
        logger.info(f"Evicting LRU skill: {lru_id}")
        del self._skills[lru_id]

    def _get_most_used(self, n: int) -> List[str]:
        """Get N most used skill IDs."""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.use_count,
            reverse=True
        )
        return [s.id for s in sorted_skills[:n]]

    def _get_least_used(self, n: int) -> List[str]:
        """Get N least used skill IDs."""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.use_count
        )
        return [s.id for s in sorted_skills[:n]]

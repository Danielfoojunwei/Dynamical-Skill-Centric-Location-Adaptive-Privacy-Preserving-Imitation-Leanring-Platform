"""
Robot Registry & Lifecycle Manager
Manages robot profiles, enforcing kinematic and compute limits.
"""

import yaml
import os
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class KinematicLimits:
    max_velocity: float
    max_acceleration: float
    joint_limits: Dict[str, List[float]]  # joint_name -> [min, max]
    reach_meters: float

@dataclass
class ComputeSpecs:
    max_tflops: float
    max_ram_gb: float
    supported_models: List[str]

@dataclass
class RetargetingConfig:
    """Maps human hand joints to robot gripper joints."""
    method: str = "direct_mapping" # or "optimization"
    scaling_factor: float = 1.0
    joint_mapping: Dict[str, str] = None # e.g. {"thumb_flex": "gripper_left"}

@dataclass
class RobotProfile:
    id: str
    name: str
    model_type: str
    kinematics: KinematicLimits
    compute: ComputeSpecs
    retargeting: RetargetingConfig
    description: str = ""

class RobotRegistry:
    def __init__(self, config_dir: str = "config/robot"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, RobotProfile] = {}
        self.active_profile_id: Optional[str] = None
        
        # Load existing profiles
        self.refresh()

    def refresh(self):
        """Reload all profiles from disk."""
        self.profiles = {}
        for f in self.config_dir.glob("*.yaml"):
            try:
                with open(f, 'r') as file:
                    data = yaml.safe_load(file)
                    profile = self._parse_profile(data)
                    self.profiles[profile.id] = profile
            except Exception as e:
                logger.error(f"Failed to load profile {f}: {e}")
                
        # Ensure default exists if empty
        if not self.profiles:
            self._create_default_profile()

    def _parse_profile(self, data: dict) -> RobotProfile:
        return RobotProfile(
            id=data['id'],
            name=data['name'],
            model_type=data['model_type'],
            kinematics=KinematicLimits(**data['kinematics']),
            compute=ComputeSpecs(**data['compute']),
            retargeting=RetargetingConfig(**data.get('retargeting', {})),
            description=data.get('description', "")
        )

    def _create_default_profile(self):
        """Create a default AGX Orin profile."""
        default = RobotProfile(
            id="agx_orin_default",
            name="AGX Orin Dev Unit",
            model_type="agx_orin_32gb",
            kinematics=KinematicLimits(
                max_velocity=1.0,
                max_acceleration=0.5,
                joint_limits={
                    "shoulder": [-3.14, 3.14],
                    "elbow": [-2.0, 2.0]
                },
                reach_meters=0.8
            ),
            compute=ComputeSpecs(
                max_tflops=100.0, # Orin 32GB approx
                max_ram_gb=32.0,
                supported_models=["pi0_small", "diffusion_policy"]
            ),
            retargeting=RetargetingConfig(
                method="direct_mapping",
                scaling_factor=1.0,
                joint_mapping={"thumb": "gripper_main"}
            ),
            description="Default development configuration"
        )
        self.save_profile(default)

    def save_profile(self, profile: RobotProfile):
        """Save a profile to disk."""
        data = asdict(profile)
        path = self.config_dir / f"{profile.id}.yaml"
        with open(path, 'w') as f:
            yaml.dump(data, f)
        self.profiles[profile.id] = profile
        logger.info(f"Saved robot profile: {profile.id}")

    def get_profile(self, profile_id: str) -> Optional[RobotProfile]:
        return self.profiles.get(profile_id)

    def list_profiles(self) -> List[RobotProfile]:
        return list(self.profiles.values())

    def activate_profile(self, profile_id: str) -> bool:
        """Set the active profile. Returns True if successful."""
        if profile_id in self.profiles:
            self.active_profile_id = profile_id
            logger.info(f"Activated robot profile: {profile_id}")
            return True
        return False
    
    def get_active_profile(self) -> Optional[RobotProfile]:
        if self.active_profile_id:
            return self.profiles[self.active_profile_id]
        # Fallback to first available
        if self.profiles:
            return list(self.profiles.values())[0]
        return None

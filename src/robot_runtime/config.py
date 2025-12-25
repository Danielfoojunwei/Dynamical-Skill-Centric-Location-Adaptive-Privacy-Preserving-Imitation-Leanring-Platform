"""
Configuration for Robot Runtime Agent.

Defines compute budgets, timing contracts, and hardware settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class RobotType(Enum):
    """Supported robot types."""
    DAIMON_VTLA = "daimon_vtla"
    GENERIC_ARM = "generic_arm"
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"


@dataclass
class Tier1Config:
    """
    Tier 1: CPU-bound, deterministic (1kHz)

    These run on CPU with hard real-time guarantees.
    No GPU dependencies in critical path.
    """
    control_rate_hz: int = 1000          # Main control loop
    safety_rate_hz: int = 1000           # Safety checks
    state_estimation_rate_hz: int = 1000 # State estimation
    actuator_rate_hz: int = 1000         # Actuator commands
    watchdog_timeout_ms: int = 5         # Hardware watchdog

    # Latency budgets (microseconds)
    max_loop_time_us: int = 900          # Must complete in <1ms
    max_safety_check_us: int = 100       # Safety check budget
    max_state_estimation_us: int = 200   # State estimation budget
    max_actuator_command_us: int = 100   # Actuator command budget


@dataclass
class Tier2Config:
    """
    Tier 2: GPU-accelerated, bounded (10-100Hz)

    These run on GPU with soft real-time guarantees.
    Bounded worst-case execution time.
    """
    policy_rate_hz: int = 100            # Policy inference
    perception_rate_hz: int = 30         # Perception pipeline
    planning_rate_hz: int = 10           # Local planning

    max_cached_skills: int = 100         # Skills in local cache

    # Latency budgets (milliseconds)
    max_perception_time_ms: int = 30     # Perception pipeline
    max_policy_time_ms: int = 10         # Policy inference
    max_planning_time_ms: int = 100      # Local planning

    # Cascade configuration
    cascade_level1_threshold: float = 0.7  # Confidence to skip level 2
    cascade_level2_threshold: float = 0.9  # Confidence to skip level 3
    cascade_cooldown_seconds: float = 5.0  # Min time between level 3 calls


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    # Safety margins
    joint_limit_margin_deg: float = 5.0
    velocity_limit_margin_percent: float = 10.0
    force_limit_margin_percent: float = 20.0

    # Collision avoidance
    min_obstacle_distance_m: float = 0.1
    human_safety_distance_m: float = 0.5

    # Emergency stop
    estop_deceleration_mps2: float = 10.0

    # Watchdog
    heartbeat_timeout_ms: int = 10


@dataclass
class PerceptionConfig:
    """Perception pipeline configuration."""
    # Level 1: Always running (30Hz, <10ms)
    level1_models: List[str] = field(default_factory=lambda: [
        "yolo_nano",           # 2 TFLOPS
        "depth_anything_small", # 1 TFLOPS
    ])
    level1_tflops: float = 3.5

    # Level 2: On-demand (10Hz, <50ms)
    level2_models: List[str] = field(default_factory=lambda: [
        "dinov3_small",        # 20 TFLOPS
        "sam3_base",           # 15 TFLOPS
        "rtmpose_m",           # 5 TFLOPS
    ])
    level2_tflops: float = 40.0

    # Level 3: Rare (1-5Hz, <500ms, can be off-robot)
    level3_models: List[str] = field(default_factory=lambda: [
        "dinov3_giant",        # 120 TFLOPS
        "sam3_huge",           # 200 TFLOPS
        "vjepa2_giant",        # 330 TFLOPS
    ])
    level3_tflops: float = 650.0

    # Camera configuration
    camera_ids: List[str] = field(default_factory=list)
    image_width: int = 640
    image_height: int = 480
    camera_fps: int = 30


@dataclass
class PolicyConfig:
    """
    Policy executor configuration.

    Supports multiple VLA backends:
    - Pi0 Custom: Gemma 3 backbone with custom MoE
    - Pi0.5 OpenPI: Official Physical Intelligence implementation

    Jetson Thor enables running the largest models:
    - Gemma 3-27B VLM backbone
    - Full Pi0.5 with open-world generalization
    """
    # VLA backend selection
    vla_backend: str = "pi05_openpi"  # Options: pi0_custom, pi05_openpi

    # Pi0 Custom configuration (when vla_backend == "pi0_custom")
    pi0_vlm_backbone: str = "google/gemma-3-12b-it"  # Default for Thor
    pi0_moe_depth: int = 18

    # Pi0.5 OpenPI configuration (when vla_backend == "pi05_openpi")
    pi05_variant: str = "pi05_base"  # Options: pi0_base, pi05_base, pi05_libero, pi05_droid

    # Legacy default model (for backwards compatibility)
    default_model: str = "pi05_gemma3"
    default_model_tflops: float = 50.0  # Thor has more compute

    # Action space
    action_dim: int = 7  # 6-DOF + gripper
    action_horizon: int = 16  # Action prediction horizon
    max_action_norm: float = 1.0

    # Inference
    use_tensorrt: bool = True
    use_fp8: bool = False  # Set True for Jetson Thor
    use_flash_attention: bool = True
    batch_size: int = 1
    precision: str = "fp16"

    # Inference timing (milliseconds)
    max_inference_time_ms: float = 100.0  # 10Hz control

    def for_jetson_thor(self) -> 'PolicyConfig':
        """Configure for Jetson Thor."""
        self.vla_backend = "pi05_openpi"
        self.pi0_vlm_backbone = "google/gemma-3-27b-it"
        self.pi0_moe_depth = 24
        self.use_fp8 = True
        self.default_model_tflops = 100.0
        return self

    def for_jetson_orin(self) -> 'PolicyConfig':
        """Configure for Jetson AGX Orin."""
        self.vla_backend = "pi0_custom"
        self.pi0_vlm_backbone = "google/gemma-3-4b-it"
        self.pi0_moe_depth = 18
        self.use_fp8 = False
        self.default_model_tflops = 50.0
        return self


@dataclass
class RobotConfig:
    """Robot hardware configuration."""
    robot_id: str = "robot_001"
    robot_type: RobotType = RobotType.GENERIC_ARM

    # Degrees of freedom
    num_joints: int = 7
    joint_names: List[str] = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7"
    ])

    # Limits
    joint_position_limits_deg: List[tuple] = field(default_factory=lambda: [
        (-180, 180), (-120, 120), (-180, 180), (-120, 120),
        (-180, 180), (-120, 120), (-180, 180)
    ])
    joint_velocity_limits_dps: List[float] = field(default_factory=lambda: [
        180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0
    ])
    joint_torque_limits_nm: List[float] = field(default_factory=lambda: [
        100.0, 100.0, 50.0, 50.0, 25.0, 25.0, 10.0
    ])

    # Communication
    control_interface: str = "can"  # can, ethernet, serial
    control_address: str = "/dev/can0"


@dataclass
class RobotRuntimeConfig:
    """Complete configuration for Robot Runtime Agent."""
    # Tier configurations
    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)

    # Component configurations
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)

    # Paths
    cache_dir: str = "/var/cache/dynamical"
    log_dir: str = "/var/log/dynamical"

    # Cloud integration (optional)
    cloud_enabled: bool = True
    cloud_url: Optional[str] = None
    cloud_api_key: Optional[str] = None

    # Offline mode
    offline_capable: bool = True  # Must be True for deployment-grade

    @classmethod
    def for_jetson_orin(cls) -> 'RobotRuntimeConfig':
        """Configuration optimized for Jetson Orin."""
        config = cls()
        # Orin has 275 FP16 TFLOPS
        config.tier2.perception_rate_hz = 30
        config.tier2.policy_rate_hz = 100
        config.policy.use_tensorrt = True
        config.policy.precision = "fp16"
        return config

    @classmethod
    def for_jetson_thor(cls) -> 'RobotRuntimeConfig':
        """
        Configuration optimized for Jetson Thor (Blackwell).

        Jetson Thor Specifications:
        - 128GB LPDDR5X memory
        - 2070 FP4 TFLOPS / 517 FP8 TFLOPS
        - Native FP8 support

        This enables:
        - Gemma 3-27B VLM backbone
        - Pi0.5 with open-world generalization
        - 10Hz control with full perception pipeline
        """
        config = cls()

        # Thor has 2070 FP4 TFLOPS / 517 FP8 TFLOPS
        config.tier2.perception_rate_hz = 60  # Can run faster
        config.tier2.policy_rate_hz = 200     # Can run faster

        # VLA configuration for Thor
        config.policy.vla_backend = "pi05_openpi"
        config.policy.pi0_vlm_backbone = "google/gemma-3-27b-it"
        config.policy.pi0_moe_depth = 24
        config.policy.use_fp8 = True
        config.policy.use_flash_attention = True
        config.policy.precision = "fp8"

        # Perception can use larger models
        config.perception.level2_models = [
            "dinov3_giant",       # Giant vision encoder
            "sam3_large",         # Large segmentation
            "rtmpose_x",          # X-Large pose
        ]
        config.perception.level3_models = [
            "dinov3_giant",       # Already included in level 2
            "sam3_huge",          # Huge segmentation
            "vjepa2_giant",       # Giant world model
        ]

        return config

    @classmethod
    def for_simulation(cls) -> 'RobotRuntimeConfig':
        """Configuration for simulation/development."""
        config = cls()
        config.tier1.control_rate_hz = 100   # Slower for debugging
        config.tier2.policy_rate_hz = 30     # Slower for debugging
        config.safety.estop_deceleration_mps2 = 1.0  # Gentle in sim
        config.robot.control_interface = "mock"
        return config

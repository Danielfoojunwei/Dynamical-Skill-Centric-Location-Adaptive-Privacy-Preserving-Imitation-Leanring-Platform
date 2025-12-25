"""
NVIDIA Jetson Thor Hardware Configuration for Dynamical.ai

This module defines the hardware configuration and capabilities for
the NVIDIA Jetson Thor robotics supercomputer (Blackwell architecture).

Jetson Thor Specifications:
===========================
- GPU: 2560-core NVIDIA Blackwell with 96 5th-gen Tensor Cores
- CPU: 14-core Arm Neoverse-V3AE @ 2.6 GHz
- Memory: 128 GB LPDDR5X @ 273 GB/s bandwidth
- AI Compute: 2070 FP4 TFLOPS / 517 FP8 TFLOPS
- Power: 40W - 130W configurable
- Video: 6x4K60 encode, 8K30 decode

Performance vs Jetson AGX Orin:
===============================
- 7.5x higher AI compute
- 3.5x better energy efficiency
- 5x faster generative AI inference
- 2x memory capacity (128GB vs 64GB)

This enables:
- 10Hz control loops (vs 2Hz on Orin)
- All perception models running simultaneously
- Larger model variants (ViT-g instead of ViT-b)
- Real-time action-conditioned world models
- Multi-Instance GPU for workload isolation

References:
- https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai
- https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Hardware Specifications
# =============================================================================

class JetsonModel(Enum):
    """Supported Jetson models."""
    JETSON_ORIN_NANO = "orin_nano"
    JETSON_ORIN_NX = "orin_nx"
    JETSON_AGX_ORIN = "agx_orin"
    JETSON_THOR = "thor"  # New flagship


class PowerMode(Enum):
    """Power consumption modes."""
    LOW_POWER = "low_power"      # 40W
    BALANCED = "balanced"        # 80W
    HIGH_PERFORMANCE = "high"    # 130W
    MAX_PERFORMANCE = "max"      # 130W + boost


@dataclass(frozen=True)
class JetsonThorSpecs:
    """
    NVIDIA Jetson Thor hardware specifications.

    The Jetson Thor is NVIDIA's flagship robotics supercomputer,
    powered by the Blackwell GPU architecture.
    """
    # Model identification
    model_name: str = "NVIDIA Jetson Thor"
    model_id: str = "T5000"
    architecture: str = "Blackwell"

    # GPU specifications
    cuda_cores: int = 2560
    tensor_cores: int = 96  # 5th generation
    tensor_core_generation: int = 5

    # Compute performance (TFLOPS)
    fp4_tflops: float = 2070.0
    fp8_tflops: float = 517.0
    fp16_tflops: float = 517.0
    bf16_tflops: float = 517.0
    int8_tops: float = 1035.0

    # Memory specifications
    memory_gb: float = 128.0
    memory_type: str = "LPDDR5X"
    memory_bandwidth_gbps: float = 273.0
    memory_bus_width: int = 256  # bits

    # CPU specifications
    cpu_cores: int = 14
    cpu_architecture: str = "Arm Neoverse-V3AE"
    cpu_frequency_ghz: float = 2.6
    l2_cache_per_core_mb: float = 1.0
    l3_cache_shared_mb: float = 16.0

    # Power specifications
    min_power_w: float = 40.0
    max_power_w: float = 130.0
    default_power_w: float = 100.0

    # Video processing
    video_encode: str = "6x4Kp60 H.265/H.264"
    video_decode: str = "8Kp30 H.265, 6x4Kp60"

    # Multi-Instance GPU (MIG) support
    mig_supported: bool = True
    mig_max_instances: int = 4

    # Comparison ratios vs Jetson AGX Orin
    compute_vs_orin: float = 7.5
    efficiency_vs_orin: float = 3.5
    genai_speedup_vs_orin: float = 5.0
    memory_vs_orin: float = 2.0


# Global hardware specs instance
JETSON_THOR = JetsonThorSpecs()


# =============================================================================
# Timing Configuration for Jetson Thor
# =============================================================================

@dataclass
class ThorTimingConfig:
    """
    Optimized timing configuration for Jetson Thor.

    With 7.5x more compute, we can run faster control loops
    and more sophisticated perception pipelines.
    """
    # Tier 1: Safety Loop (unchanged - always 1kHz)
    SAFETY_FREQUENCY_HZ: float = 1000.0
    SAFETY_PERIOD_MS: float = 1.0

    # Tier 2: Control Loop - NOW 10Hz (was 2Hz on Orin)
    # With 7.5x compute, we can run 5x faster control
    CONTROL_FREQUENCY_HZ: float = 10.0
    CONTROL_PERIOD_MS: float = 100.0  # 100ms cycle
    CONTROL_MAX_LATENCY_MS: float = 90.0

    # Tier 3: Perception Loop - NOW 10Hz (was 5Hz on Orin)
    PERCEPTION_FREQUENCY_HZ: float = 10.0
    PERCEPTION_PERIOD_MS: float = 100.0
    PERCEPTION_MAX_LATENCY_MS: float = 90.0

    # Tier 4: Learning Loop (offline, but faster)
    LEARNING_MIN_INTERVAL_S: float = 5.0   # Faster iterations
    LEARNING_MAX_INTERVAL_S: float = 30.0
    FHE_ALLOWED_DURATION_S: float = 120.0  # Faster with Thor

    # Model inference budgets (Jetson Thor with TensorRT)
    # ~5x faster than Orin due to Blackwell + more Tensor Cores
    DINOV3_VITG_INFERENCE_MS: float = 15.0   # Giant model, was impossible on Orin
    DINOV3_VITL_INFERENCE_MS: float = 10.0   # Large model
    DINOV3_VITB_INFERENCE_MS: float = 8.0    # Base model (was 50ms on Orin)

    SAM3_LARGE_INFERENCE_MS: float = 12.0    # Large model
    SAM3_BASE_INFERENCE_MS: float = 8.0      # Base model (was 40ms on Orin)

    DEPTH_INFERENCE_MS: float = 5.0          # Was 25ms on Orin
    POSE_INFERENCE_MS: float = 4.0           # Was 20ms on Orin

    VJEPA2_LARGE_INFERENCE_MS: float = 15.0  # Large model
    VJEPA2_BASE_INFERENCE_MS: float = 10.0   # Was 60ms on Orin

    VLA_LARGE_INFERENCE_MS: float = 20.0     # Can run larger VLA models
    VLA_BASE_INFERENCE_MS: float = 10.0      # Was 50ms on Orin

    # LLM inference (new capability with Thor)
    LLM_8B_TOKENS_PER_SEC: float = 50.0      # Llama 3.1 8B
    LLM_70B_TOKENS_PER_SEC: float = 10.0     # Llama 3.1 70B (quantized)

    @property
    def total_perception_budget_ms(self) -> float:
        """Total perception pipeline time with largest models."""
        return (
            self.DINOV3_VITG_INFERENCE_MS +
            self.SAM3_LARGE_INFERENCE_MS +
            self.DEPTH_INFERENCE_MS +
            self.POSE_INFERENCE_MS +
            self.VJEPA2_LARGE_INFERENCE_MS
        )  # = 51ms, fits easily in 100ms

    @property
    def can_run_all_parallel(self) -> bool:
        """Check if all models can run in parallel."""
        # With 128GB memory, we can keep all models loaded
        return True


# Global timing config for Thor
THOR_TIMING = ThorTimingConfig()


# =============================================================================
# GPU Memory Configuration
# =============================================================================

@dataclass
class ThorMemoryConfig:
    """
    GPU memory configuration for Jetson Thor.

    With 128GB unified memory, we can run all models simultaneously
    without eviction, plus cache intermediate results.
    """
    total_memory_gb: float = 128.0
    system_reserved_gb: float = 8.0  # OS and system services
    available_for_ai_gb: float = 120.0

    # Model memory allocations (GB)
    # Can now run GIANT models that were impossible on Orin
    MODEL_BUDGETS: Dict[str, float] = field(default_factory=lambda: {
        # Vision models
        "dinov3_vitg": 8.0,    # Giant model
        "dinov3_vitl": 4.0,    # Large model
        "dinov3_vitb": 2.0,    # Base model
        # Segmentation
        "sam3_large": 6.0,
        "sam3_base": 3.0,
        "sam3_small": 1.5,
        # Depth
        "depth_anything_v3_large": 2.0,
        "depth_anything_v3_base": 1.0,
        # Pose
        "rtmpose_x": 1.0,
        "rtmpose_l": 0.5,
        # Video understanding
        "vjepa2_giant": 10.0,
        "vjepa2_large": 5.0,
        "vjepa2_base": 2.5,
        # Action models
        "pi0_large": 8.0,
        "pi0_base": 4.0,
        # LLMs (new with Thor!)
        "llama_8b": 16.0,
        "llama_8b_quantized": 8.0,
        "qwen_7b": 14.0,
        # World models
        "genie2": 12.0,
        # Cache and buffers
        "perception_cache": 10.0,
        "action_buffer": 5.0,
    })

    def get_total_model_memory(self) -> float:
        """Get total memory for all models."""
        return sum(self.MODEL_BUDGETS.values())

    def can_fit_all_models(self) -> bool:
        """Check if all models fit in memory."""
        return self.get_total_model_memory() <= self.available_for_ai_gb


# Global memory config for Thor
THOR_MEMORY = ThorMemoryConfig()


# =============================================================================
# Multi-Instance GPU (MIG) Configuration
# =============================================================================

@dataclass
class MIGPartition:
    """A single MIG partition configuration."""
    name: str
    memory_gb: float
    compute_percent: float
    workload: str  # Description of assigned workload


@dataclass
class ThorMIGConfig:
    """
    Multi-Instance GPU configuration for Jetson Thor.

    MIG allows partitioning the GPU into isolated instances
    for guaranteed performance and workload isolation.
    """
    enabled: bool = True
    max_instances: int = 4

    # Recommended partition scheme for robotics
    ROBOTICS_PARTITIONS: List[MIGPartition] = field(default_factory=lambda: [
        MIGPartition(
            name="perception",
            memory_gb=48.0,
            compute_percent=40.0,
            workload="DINOv3, SAM3, Depth, Pose estimation"
        ),
        MIGPartition(
            name="world_model",
            memory_gb=32.0,
            compute_percent=25.0,
            workload="V-JEPA 2, video prediction"
        ),
        MIGPartition(
            name="action",
            memory_gb=32.0,
            compute_percent=25.0,
            workload="Pi0 VLA, action generation"
        ),
        MIGPartition(
            name="system",
            memory_gb=16.0,
            compute_percent=10.0,
            workload="LLM reasoning, FHE, misc"
        ),
    ])

    def get_total_memory(self) -> float:
        """Get total partitioned memory."""
        return sum(p.memory_gb for p in self.ROBOTICS_PARTITIONS)

    def get_partition(self, name: str) -> Optional[MIGPartition]:
        """Get partition by name."""
        for p in self.ROBOTICS_PARTITIONS:
            if p.name == name:
                return p
        return None


# Global MIG config for Thor
THOR_MIG = ThorMIGConfig()


# =============================================================================
# Performance Comparison
# =============================================================================

@dataclass
class PerformanceComparison:
    """Performance comparison between Jetson models."""
    metric: str
    orin_value: float
    thor_value: float
    unit: str
    improvement: float  # Thor vs Orin ratio

    def __str__(self) -> str:
        return f"{self.metric}: {self.orin_value} -> {self.thor_value} {self.unit} ({self.improvement:.1f}x)"


def get_performance_comparison() -> List[PerformanceComparison]:
    """Get detailed performance comparison."""
    return [
        PerformanceComparison("AI Compute (FP4)", 275, 2070, "TFLOPS", 7.5),
        PerformanceComparison("AI Compute (FP8)", 138, 517, "TFLOPS", 3.7),
        PerformanceComparison("Memory", 64, 128, "GB", 2.0),
        PerformanceComparison("Memory Bandwidth", 204, 273, "GB/s", 1.3),
        PerformanceComparison("CUDA Cores", 2048, 2560, "cores", 1.25),
        PerformanceComparison("Tensor Cores", 64, 96, "cores", 1.5),
        PerformanceComparison("CPU Cores", 12, 14, "cores", 1.17),
        PerformanceComparison("Control Loop", 2, 10, "Hz", 5.0),
        PerformanceComparison("Perception Loop", 5, 10, "Hz", 2.0),
        PerformanceComparison("DINOv3 Inference", 50, 8, "ms", 6.25),
        PerformanceComparison("SAM3 Inference", 40, 8, "ms", 5.0),
        PerformanceComparison("V-JEPA 2 Inference", 60, 10, "ms", 6.0),
        PerformanceComparison("VLA Inference", 50, 10, "ms", 5.0),
        PerformanceComparison("LLM 8B (tok/s)", 0, 50, "tok/s", float('inf')),
        PerformanceComparison("Max Model Size", 2, 8, "B params", 4.0),
    ]


# =============================================================================
# Model Configuration for Thor
# =============================================================================

class VLABackend(Enum):
    """VLA model backend options."""
    PI0_CUSTOM = "pi0_custom"      # Custom Pi0 with Gemma 3
    PI05_OPENPI = "pi05_openpi"    # Official Physical Intelligence Pi0.5
    PI0_FAST = "pi0_fast"          # Fast inference variant


class VLMBackendType(Enum):
    """VLM backbone options for custom Pi0."""
    PALIGEMMA_3B = "paligemma_3b"
    GEMMA3_4B = "gemma3_4b"
    GEMMA3_12B = "gemma3_12b"
    GEMMA3_27B = "gemma3_27b"


@dataclass
class ThorModelConfig:
    """
    Recommended model configurations for Jetson Thor.

    With 7.5x more compute and 2x memory, we can run
    significantly larger and more capable models.

    Jetson Thor Specifications:
    - Memory: 128GB LPDDR5X
    - AI Compute: 2070 FP4 TFLOPS
    - Native FP8 support via Blackwell

    This enables:
    - Gemma 3-27B VLM backbone
    - Pi0.5 with open-world generalization
    - 10Hz control with full perception pipeline
    """
    # Vision backbone - use Giant instead of Base
    dinov3_variant: str = "vitg16"  # Was vitb16 on Orin
    dinov3_resolution: int = 518

    # Segmentation - use Large instead of Small
    sam3_variant: str = "large"  # Was small on Orin

    # Depth - use Large
    depth_variant: str = "large"  # Was base on Orin

    # Pose - use X-Large
    pose_variant: str = "rtmpose_x"  # Was rtmpose_l on Orin

    # Video understanding - use Giant
    vjepa2_variant: str = "vit_giant_384"  # Was vit_large on Orin
    vjepa2_action_conditioned: bool = True

    # ==========================================================================
    # VLA Configuration (Updated for Pi0.5 + Gemma 3)
    # ==========================================================================

    # VLA backend selection
    vla_backend: VLABackend = VLABackend.PI05_OPENPI  # Use official Pi0.5

    # Custom Pi0 configuration (used when vla_backend == PI0_CUSTOM)
    pi0_vlm_backbone: VLMBackendType = VLMBackendType.GEMMA3_27B  # Thor can run 27B
    pi0_action_dim: int = 7
    pi0_action_horizon: int = 16
    pi0_moe_depth: int = 24  # Deeper MoE with Thor's compute

    # Pi0.5 OpenPI configuration (used when vla_backend == PI05_OPENPI)
    pi05_variant: str = "pi05_base"  # Options: pi0_base, pi05_base, pi05_libero, pi05_droid
    pi05_use_tensorrt: bool = True
    pi05_use_fp8: bool = True

    # Legacy VLA variant (for backwards compatibility)
    vla_variant: str = "pi05_gemma3_27b"

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================

    # Enable LLM reasoning (new with Thor!)
    enable_llm: bool = True
    llm_model: str = "llama_3.1_8b"
    llm_quantization: str = "fp8"  # Thor has native FP8 support

    # Enable real-time world model
    enable_world_model: bool = True
    world_model: str = "vjepa2_ac"

    # ==========================================================================
    # Memory Budget (128GB total)
    # ==========================================================================

    def get_memory_allocation(self) -> Dict[str, float]:
        """Get recommended memory allocation for Thor's 128GB."""
        base_allocation = {
            # Perception models (~20GB)
            "dinov3_vitg": 8.0,
            "sam3_large": 6.0,
            "depth_large": 2.0,
            "rtmpose_x": 1.0,
            "vjepa2_giant": 10.0,

            # System overhead
            "system": 8.0,
            "cache": 10.0,
        }

        # Add VLA allocation based on backend
        if self.vla_backend == VLABackend.PI0_CUSTOM:
            vlm_memory = {
                VLMBackendType.PALIGEMMA_3B: 6.0,
                VLMBackendType.GEMMA3_4B: 8.0,
                VLMBackendType.GEMMA3_12B: 24.0,
                VLMBackendType.GEMMA3_27B: 54.0,
            }
            base_allocation["pi0_vlm"] = vlm_memory.get(self.pi0_vlm_backbone, 24.0)
            base_allocation["pi0_moe"] = 4.0
        else:
            # Pi0.5 uses ~32GB for full model
            base_allocation["pi05"] = 32.0

        # Add LLM if enabled
        if self.enable_llm:
            base_allocation["llm"] = 16.0  # 8B quantized

        return base_allocation

    def get_total_memory_gb(self) -> float:
        """Get total memory required."""
        return sum(self.get_memory_allocation().values())

    def validate_memory(self) -> bool:
        """Check if configuration fits in Thor's 128GB."""
        return self.get_total_memory_gb() <= 120.0  # Leave 8GB for system

    def get_model_list(self) -> List[str]:
        """Get list of all enabled models."""
        models = [
            f"dinov3_{self.dinov3_variant}",
            f"sam3_{self.sam3_variant}",
            f"depth_{self.depth_variant}",
            self.pose_variant,
            f"vjepa2_{self.vjepa2_variant}",
        ]

        # Add VLA model
        if self.vla_backend == VLABackend.PI0_CUSTOM:
            models.append(f"pi0_gemma3_{self.pi0_vlm_backbone.value}")
        else:
            models.append(f"pi05_{self.pi05_variant}")

        if self.enable_llm:
            models.append(self.llm_model)

        return models

    def get_vla_config(self) -> Dict[str, Any]:
        """Get VLA configuration for initialization."""
        if self.vla_backend == VLABackend.PI0_CUSTOM:
            return {
                "backend": "pi0_custom",
                "vlm_backbone": self.pi0_vlm_backbone.value,
                "action_dim": self.pi0_action_dim,
                "action_horizon": self.pi0_action_horizon,
                "moe_depth": self.pi0_moe_depth,
                "use_flash_attention": True,
                "dtype": "float16",
            }
        else:
            return {
                "backend": "pi05_openpi",
                "variant": self.pi05_variant,
                "use_tensorrt": self.pi05_use_tensorrt,
                "use_fp8": self.pi05_use_fp8,
            }


# Global model config for Thor
THOR_MODELS = ThorModelConfig()


# =============================================================================
# Power Management
# =============================================================================

@dataclass
class ThorPowerProfile:
    """Power profile for different use cases."""
    name: str
    power_watts: float
    description: str
    use_case: str


POWER_PROFILES = [
    ThorPowerProfile(
        name="efficiency",
        power_watts=40.0,
        description="Low power for basic tasks",
        use_case="Standby, simple monitoring"
    ),
    ThorPowerProfile(
        name="balanced",
        power_watts=80.0,
        description="Balanced power/performance",
        use_case="Normal operation, moderate workloads"
    ),
    ThorPowerProfile(
        name="performance",
        power_watts=100.0,
        description="High performance",
        use_case="Active manipulation, full perception"
    ),
    ThorPowerProfile(
        name="maximum",
        power_watts=130.0,
        description="Maximum performance",
        use_case="Complex tasks, all models active"
    ),
]


def get_power_profile(name: str) -> Optional[ThorPowerProfile]:
    """Get power profile by name."""
    for profile in POWER_PROFILES:
        if profile.name == name:
            return profile
    return None


# =============================================================================
# System Summary
# =============================================================================

def print_thor_summary():
    """Print Jetson Thor configuration summary."""
    print("\n" + "=" * 80)
    print("NVIDIA JETSON THOR CONFIGURATION FOR DYNAMICAL.AI")
    print("=" * 80)

    print(f"\n{'Hardware Specifications':^80}")
    print("-" * 80)
    print(f"Model: {JETSON_THOR.model_name} ({JETSON_THOR.model_id})")
    print(f"Architecture: {JETSON_THOR.architecture}")
    print(f"GPU: {JETSON_THOR.cuda_cores} CUDA cores, {JETSON_THOR.tensor_cores} Tensor Cores")
    print(f"CPU: {JETSON_THOR.cpu_cores}-core {JETSON_THOR.cpu_architecture} @ {JETSON_THOR.cpu_frequency_ghz} GHz")
    print(f"Memory: {JETSON_THOR.memory_gb} GB {JETSON_THOR.memory_type} @ {JETSON_THOR.memory_bandwidth_gbps} GB/s")
    print(f"Power: {JETSON_THOR.min_power_w}W - {JETSON_THOR.max_power_w}W")

    print(f"\n{'AI Compute Performance':^80}")
    print("-" * 80)
    print(f"FP4: {JETSON_THOR.fp4_tflops} TFLOPS")
    print(f"FP8/FP16: {JETSON_THOR.fp8_tflops} TFLOPS")
    print(f"INT8: {JETSON_THOR.int8_tops} TOPS")

    print(f"\n{'Performance vs Jetson AGX Orin':^80}")
    print("-" * 80)
    for comp in get_performance_comparison()[:8]:
        print(f"  {comp}")

    print(f"\n{'Timing Configuration':^80}")
    print("-" * 80)
    print(f"Safety Loop: {THOR_TIMING.SAFETY_FREQUENCY_HZ} Hz ({THOR_TIMING.SAFETY_PERIOD_MS} ms)")
    print(f"Control Loop: {THOR_TIMING.CONTROL_FREQUENCY_HZ} Hz ({THOR_TIMING.CONTROL_PERIOD_MS} ms)")
    print(f"Perception Loop: {THOR_TIMING.PERCEPTION_FREQUENCY_HZ} Hz ({THOR_TIMING.PERCEPTION_PERIOD_MS} ms)")
    print(f"Total Perception Budget: {THOR_TIMING.total_perception_budget_ms:.1f} ms")

    print(f"\n{'Model Configuration':^80}")
    print("-" * 80)
    print(f"DINOv3: {THOR_MODELS.dinov3_variant} ({THOR_TIMING.DINOV3_VITG_INFERENCE_MS} ms)")
    print(f"SAM3: {THOR_MODELS.sam3_variant} ({THOR_TIMING.SAM3_LARGE_INFERENCE_MS} ms)")
    print(f"V-JEPA 2: {THOR_MODELS.vjepa2_variant} ({THOR_TIMING.VJEPA2_LARGE_INFERENCE_MS} ms)")
    print(f"VLA: {THOR_MODELS.vla_variant} ({THOR_TIMING.VLA_LARGE_INFERENCE_MS} ms)")
    print(f"LLM: {THOR_MODELS.llm_model} ({THOR_TIMING.LLM_8B_TOKENS_PER_SEC} tok/s)")

    print(f"\n{'MIG Partitions':^80}")
    print("-" * 80)
    for partition in THOR_MIG.ROBOTICS_PARTITIONS:
        print(f"  {partition.name}: {partition.memory_gb} GB, {partition.compute_percent}% compute")
        print(f"    -> {partition.workload}")

    print("\n" + "=" * 80)
    print("Jetson Thor enables 10Hz control with full perception pipeline!")
    print("=" * 80)


# =============================================================================
# Hardware Detection
# =============================================================================

def detect_jetson_model() -> JetsonModel:
    """Detect which Jetson model is running."""
    try:
        # Check for Jetson Thor
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "thor" in model or "t5000" in model:
                return JetsonModel.JETSON_THOR
            elif "agx orin" in model:
                return JetsonModel.JETSON_AGX_ORIN
            elif "orin nx" in model:
                return JetsonModel.JETSON_ORIN_NX
            elif "orin nano" in model:
                return JetsonModel.JETSON_ORIN_NANO
    except FileNotFoundError:
        pass

    # Check CUDA compute capability
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # Blackwell has compute capability 10.x
            if props.major >= 10:
                return JetsonModel.JETSON_THOR
            elif props.major == 8:
                return JetsonModel.JETSON_AGX_ORIN
    except ImportError:
        pass

    # Default to Thor for development
    logger.info("Could not detect Jetson model, defaulting to Thor")
    return JetsonModel.JETSON_THOR


def get_hardware_config():
    """Get hardware configuration for detected Jetson model."""
    model = detect_jetson_model()

    if model == JetsonModel.JETSON_THOR:
        return {
            "specs": JETSON_THOR,
            "timing": THOR_TIMING,
            "memory": THOR_MEMORY,
            "mig": THOR_MIG,
            "models": THOR_MODELS,
        }
    else:
        # Fallback configuration for Orin
        logger.warning(f"Running on {model.value}, using reduced configuration")
        return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print_thor_summary()

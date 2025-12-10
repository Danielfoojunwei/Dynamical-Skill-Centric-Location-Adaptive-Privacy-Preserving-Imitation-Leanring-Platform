"""
Depth Anything V3 TensorRT Inference

Real-time monocular metric depth estimation using Depth Anything V3 with
TensorRT acceleration. Designed for integration with the whole-body pose
pipeline for improved 3D reconstruction.

Features:
- TensorRT FP16/FP32 inference
- Metric depth output (meters)
- Sky detection and handling
- Point cloud generation
- Focal length normalization

Based on: https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt

Model Input:
- Shape: [1, 3, 280, 504] (batch, channels, height, width)
- Type: float32
- Normalization: ImageNet mean/std to [0, 1] range

Model Output:
- depth: [1, 1, 280, 504] metric depth in meters
- sky: [1, 1, 280, 504] sky classification logits
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class DepthModelSize(Enum):
    """Available Depth Anything V3 model sizes."""
    SMALL = "small"    # Faster, less accurate
    BASE = "base"      # Balanced
    LARGE = "large"    # Most accurate, slower


class DepthPrecision(Enum):
    """TensorRT precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"  # Requires calibration


@dataclass
class DepthEstimationConfig:
    """Configuration for Depth Anything V3 inference."""

    # Model paths
    model_size: DepthModelSize = DepthModelSize.LARGE
    onnx_path: str = "models/depth_anything_v3/DA3METRIC-LARGE.onnx"
    engine_path: str = "models/depth_anything_v3/DA3METRIC-LARGE.engine"

    # Inference settings
    precision: DepthPrecision = DepthPrecision.FP16
    device: str = "cuda:0"

    # Model input size (from DA V3 spec)
    input_height: int = 280
    input_width: int = 504

    # Sky handling
    enable_sky_detection: bool = True
    sky_threshold: float = 0.3  # Lower = more sky detection
    sky_depth_cap: float = 200.0  # Max depth for sky regions (meters)

    # Focal length normalization
    # DA V3 was trained with focal_length / 300 normalization
    focal_length_normalization: float = 300.0

    # Output options
    output_original_size: bool = True  # Resize output to input image size
    interpolation: str = "cubic"  # cubic, linear, nearest

    # Point cloud options
    generate_point_cloud: bool = False
    point_cloud_downsample: int = 2  # Downsample factor for point cloud
    colorize_point_cloud: bool = True

    # Debug options
    enable_debug: bool = False
    debug_colormap: str = "JET"  # JET, VIRIDIS, PLASMA, etc.

    def get_model_path(self) -> str:
        """Get model path based on size."""
        size_map = {
            DepthModelSize.SMALL: "DA3METRIC-SMALL",
            DepthModelSize.BASE: "DA3METRIC-BASE",
            DepthModelSize.LARGE: "DA3METRIC-LARGE",
        }
        model_name = size_map[self.model_size]
        return f"models/depth_anything_v3/{model_name}.onnx"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DepthResult:
    """Result from depth estimation."""

    # Core output
    depth_map: np.ndarray  # [H, W] metric depth in meters
    timestamp: float

    # Original image size
    original_size: Tuple[int, int]  # (width, height)

    # Optional outputs
    sky_mask: Optional[np.ndarray] = None  # [H, W] bool, True = sky
    point_cloud: Optional[np.ndarray] = None  # [N, 3] XYZ points
    point_colors: Optional[np.ndarray] = None  # [N, 3] RGB colors

    # Metadata
    inference_time_ms: float = 0.0
    model_input_size: Tuple[int, int] = (504, 280)

    @property
    def height(self) -> int:
        return self.depth_map.shape[0]

    @property
    def width(self) -> int:
        return self.depth_map.shape[1]

    def get_depth_at(self, x: int, y: int) -> float:
        """Get depth at pixel location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.depth_map[y, x])
        return 0.0

    def sample_depth_bilinear(self, x: float, y: float) -> float:
        """Sample depth with bilinear interpolation for sub-pixel accuracy."""
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)

        if x0 < 0 or y0 < 0 or x0 >= self.width or y0 >= self.height:
            return 0.0

        # Bilinear weights
        wx = x - x0
        wy = y - y0

        # Sample 4 neighbors
        d00 = self.depth_map[y0, x0]
        d01 = self.depth_map[y0, x1]
        d10 = self.depth_map[y1, x0]
        d11 = self.depth_map[y1, x1]

        # Interpolate
        d0 = d00 * (1 - wx) + d01 * wx
        d1 = d10 * (1 - wx) + d11 * wx

        return float(d0 * (1 - wy) + d1 * wy)

    def to_colormap(self, colormap: str = "JET", min_depth: float = 0.0,
                    max_depth: float = 50.0) -> np.ndarray:
        """Convert depth map to colorized visualization."""
        try:
            import cv2

            # Normalize to 0-255
            depth_normalized = np.clip(
                (self.depth_map - min_depth) / (max_depth - min_depth),
                0, 1
            )
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

            # Apply colormap
            colormap_cv = getattr(cv2, f"COLORMAP_{colormap}", cv2.COLORMAP_JET)
            colored = cv2.applyColorMap(depth_uint8, colormap_cv)

            return colored
        except ImportError:
            # Fallback: grayscale
            depth_normalized = np.clip(self.depth_map / max_depth, 0, 1)
            return (depth_normalized * 255).astype(np.uint8)


# =============================================================================
# ImageNet Normalization Constants
# =============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# Depth Anything V3 Inference
# =============================================================================

class DepthAnythingV3:
    """
    TensorRT-accelerated Depth Anything V3 inference.

    Provides metric depth estimation from monocular RGB images.
    Optimized for real-time performance on NVIDIA GPUs and Jetson.

    Usage:
        config = DepthEstimationConfig(
            model_size=DepthModelSize.LARGE,
            precision=DepthPrecision.FP16,
        )
        depth_estimator = DepthAnythingV3(config)

        # Single image inference
        result = depth_estimator.infer(image)
        depth_map = result.depth_map  # [H, W] in meters

        # Sample depth at specific location
        z = result.sample_depth_bilinear(x, y)
    """

    def __init__(self, config: DepthEstimationConfig = None):
        self.config = config or DepthEstimationConfig()

        # Inference backends (lazy loaded)
        self._trt_engine = None
        self._trt_context = None
        self._onnx_session = None

        # CUDA resources
        self._cuda_stream = None
        self._device_input = None
        self._device_output_depth = None
        self._device_output_sky = None
        self._host_output_depth = None
        self._host_output_sky = None

        # State
        self._initialized = False
        self._use_tensorrt = False
        self._use_onnx = False

        # Camera intrinsics (set per image or globally)
        self._default_focal_length: Optional[float] = None

        self._initialize()

    def _initialize(self):
        """Initialize inference backend (TensorRT or ONNX)."""
        # Try TensorRT first
        if self._try_init_tensorrt():
            self._use_tensorrt = True
            self._initialized = True
            logger.info("Depth Anything V3: TensorRT backend initialized")
            return

        # Fall back to ONNX Runtime
        if self._try_init_onnx():
            self._use_onnx = True
            self._initialized = True
            logger.info("Depth Anything V3: ONNX Runtime backend initialized")
            return

        # Neither available - use mock
        logger.warning(
            "Depth Anything V3: No inference backend available. "
            "Install tensorrt or onnxruntime-gpu. Using mock inference."
        )
        self._initialized = False

    def _try_init_tensorrt(self) -> bool:
        """Try to initialize TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            engine_path = Path(self.config.engine_path)

            if not engine_path.exists():
                # Try to build engine from ONNX
                onnx_path = Path(self.config.onnx_path)
                if onnx_path.exists():
                    logger.info(f"Building TensorRT engine from {onnx_path}...")
                    self._build_trt_engine(str(onnx_path), str(engine_path))
                else:
                    logger.warning(f"ONNX model not found: {onnx_path}")
                    return False

            # Load engine
            logger.info(f"Loading TensorRT engine: {engine_path}")
            trt_logger = trt.Logger(trt.Logger.WARNING)

            with open(engine_path, "rb") as f:
                engine_data = f.read()

            runtime = trt.Runtime(trt_logger)
            self._trt_engine = runtime.deserialize_cuda_engine(engine_data)

            if self._trt_engine is None:
                logger.error("Failed to deserialize TensorRT engine")
                return False

            self._trt_context = self._trt_engine.create_execution_context()

            # Allocate CUDA buffers
            self._allocate_trt_buffers()

            return True

        except ImportError as e:
            logger.debug(f"TensorRT not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"TensorRT initialization failed: {e}")
            return False

    def _build_trt_engine(self, onnx_path: str, engine_path: str):
        """Build TensorRT engine from ONNX model."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if self.config.precision == DepthPrecision.FP16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        logger.info("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved: {engine_path}")

    def _allocate_trt_buffers(self):
        """Allocate CUDA buffers for TensorRT inference."""
        import pycuda.driver as cuda

        h, w = self.config.input_height, self.config.input_width

        # Input buffer: [1, 3, H, W] float32
        input_size = 1 * 3 * h * w * 4  # 4 bytes per float32
        self._device_input = cuda.mem_alloc(input_size)

        # Output buffers: [1, 1, H, W] float32
        output_size = 1 * 1 * h * w * 4
        self._device_output_depth = cuda.mem_alloc(output_size)
        self._device_output_sky = cuda.mem_alloc(output_size)

        # Host buffers
        self._host_output_depth = np.empty((1, 1, h, w), dtype=np.float32)
        self._host_output_sky = np.empty((1, 1, h, w), dtype=np.float32)

        # CUDA stream
        self._cuda_stream = cuda.Stream()

    def _try_init_onnx(self) -> bool:
        """Try to initialize ONNX Runtime."""
        try:
            import onnxruntime as ort

            onnx_path = Path(self.config.onnx_path)

            if not onnx_path.exists():
                logger.warning(f"ONNX model not found: {onnx_path}")
                return False

            # Configure session
            if "cuda" in self.config.device:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': int(self.config.device.split(':')[-1]),
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._onnx_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=providers
            )

            logger.info(f"ONNX model loaded: {onnx_path}")
            logger.info(f"ONNX providers: {self._onnx_session.get_providers()}")

            return True

        except ImportError as e:
            logger.debug(f"ONNX Runtime not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"ONNX initialization failed: {e}")
            return False

    def set_camera_intrinsics(self, fx: float, fy: float = None):
        """
        Set camera focal length for depth scaling.

        Depth Anything V3 was trained with focal_length / 300 normalization.
        If your camera has different focal length, set it here for accurate
        metric depth.

        Args:
            fx: Horizontal focal length in pixels
            fy: Vertical focal length (defaults to fx)
        """
        if fy is None:
            fy = fx
        self._default_focal_length = (fx + fy) / 2.0

    def infer(
        self,
        image: np.ndarray,
        focal_length: float = None,
    ) -> DepthResult:
        """
        Estimate metric depth from RGB image.

        Args:
            image: [H, W, 3] BGR or RGB image (uint8 or float32)
            focal_length: Camera focal length in pixels (overrides default)

        Returns:
            DepthResult with metric depth map
        """
        start_time = time.time()

        original_h, original_w = image.shape[:2]

        # Use provided focal length or default
        if focal_length is None:
            focal_length = self._default_focal_length
        if focal_length is None:
            # Estimate from image size (assume ~60° FOV)
            focal_length = original_w / (2 * np.tan(np.radians(30)))

        if not self._initialized:
            # Mock inference
            return self._mock_inference(image, focal_length, start_time)

        # Preprocess
        input_tensor = self._preprocess(image)

        # Run inference
        if self._use_tensorrt:
            depth_raw, sky_logits = self._infer_tensorrt(input_tensor)
        else:
            depth_raw, sky_logits = self._infer_onnx(input_tensor)

        # Postprocess
        depth_map, sky_mask = self._postprocess(
            depth_raw, sky_logits, focal_length, (original_w, original_h)
        )

        inference_time = (time.time() - start_time) * 1000

        result = DepthResult(
            depth_map=depth_map,
            timestamp=time.time(),
            original_size=(original_w, original_h),
            sky_mask=sky_mask if self.config.enable_sky_detection else None,
            inference_time_ms=inference_time,
            model_input_size=(self.config.input_width, self.config.input_height),
        )

        # Generate point cloud if requested
        if self.config.generate_point_cloud:
            result.point_cloud, result.point_colors = self._generate_point_cloud(
                depth_map, image, focal_length
            )

        return result

    def infer_batch(
        self,
        images: List[np.ndarray],
        focal_lengths: List[float] = None,
    ) -> List[DepthResult]:
        """
        Batch inference on multiple images.

        Note: Current implementation processes sequentially.
        TODO: Implement true batched inference.
        """
        if focal_lengths is None:
            focal_lengths = [None] * len(images)

        return [
            self.infer(img, fl)
            for img, fl in zip(images, focal_lengths)
        ]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        1. Convert BGR to RGB
        2. Resize to model input size
        3. Normalize with ImageNet mean/std
        4. Convert to NCHW format
        """
        try:
            import cv2

            # Ensure RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR if uint8
                if image.dtype == np.uint8:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb = image
            else:
                rgb = image

            # Resize to model input size
            target_size = (self.config.input_width, self.config.input_height)
            resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)

            # Normalize to [0, 1] then apply ImageNet normalization
            if resized.dtype == np.uint8:
                normalized = resized.astype(np.float32) / 255.0
            else:
                normalized = resized.astype(np.float32)

            # ImageNet normalization
            normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

            # Convert to NCHW: [H, W, C] -> [1, C, H, W]
            tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

            return tensor.astype(np.float32)

        except ImportError:
            # Fallback without OpenCV
            if image.dtype == np.uint8:
                normalized = image.astype(np.float32) / 255.0
            else:
                normalized = image.astype(np.float32)

            normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
            tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

            return tensor.astype(np.float32)

    def _infer_tensorrt(
        self,
        input_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run TensorRT inference."""
        import pycuda.driver as cuda

        # Copy input to device
        cuda.memcpy_htod_async(
            self._device_input,
            np.ascontiguousarray(input_tensor),
            self._cuda_stream
        )

        # Execute
        bindings = [
            int(self._device_input),
            int(self._device_output_depth),
            int(self._device_output_sky),
        ]
        self._trt_context.execute_async_v2(
            bindings=bindings,
            stream_handle=self._cuda_stream.handle
        )

        # Copy outputs to host
        cuda.memcpy_dtoh_async(
            self._host_output_depth,
            self._device_output_depth,
            self._cuda_stream
        )
        cuda.memcpy_dtoh_async(
            self._host_output_sky,
            self._device_output_sky,
            self._cuda_stream
        )

        self._cuda_stream.synchronize()

        return self._host_output_depth.copy(), self._host_output_sky.copy()

    def _infer_onnx(
        self,
        input_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ONNX Runtime inference."""
        input_name = self._onnx_session.get_inputs()[0].name

        outputs = self._onnx_session.run(None, {input_name: input_tensor})

        # Handle different output configurations
        if len(outputs) >= 2:
            depth_raw = outputs[0]
            sky_logits = outputs[1]
        else:
            depth_raw = outputs[0]
            sky_logits = np.zeros_like(depth_raw)  # No sky output

        return depth_raw, sky_logits

    def _postprocess(
        self,
        depth_raw: np.ndarray,
        sky_logits: np.ndarray,
        focal_length: float,
        original_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Postprocess model outputs.

        Steps:
        1. Extract depth from tensor
        2. Apply focal length scaling
        3. Handle sky regions
        4. Resize to original resolution
        """
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        # Extract from batch/channel dims: [1, 1, H, W] -> [H, W]
        depth = depth_raw.squeeze()

        # Clamp negative values
        depth = np.maximum(depth, 0)

        # Apply focal length scaling
        # DA V3 was trained with focal_length / 300 normalization
        scale = focal_length / self.config.focal_length_normalization
        depth = depth * scale

        # Handle sky regions
        sky_mask = None
        if self.config.enable_sky_detection and sky_logits is not None:
            sky_probs = 1.0 / (1.0 + np.exp(-sky_logits.squeeze()))  # Sigmoid
            sky_mask = sky_probs < self.config.sky_threshold

            # Get 99th percentile of valid depths for sky fill
            valid_depths = depth[~sky_mask]
            if len(valid_depths) > 0:
                sky_fill_depth = min(
                    np.percentile(valid_depths, 99),
                    self.config.sky_depth_cap
                )
            else:
                sky_fill_depth = self.config.sky_depth_cap

            depth[sky_mask] = sky_fill_depth

        # Resize to original resolution
        if self.config.output_original_size:
            if has_cv2:
                interp_map = {
                    "cubic": cv2.INTER_CUBIC,
                    "linear": cv2.INTER_LINEAR,
                    "nearest": cv2.INTER_NEAREST,
                }
                interp = interp_map.get(self.config.interpolation, cv2.INTER_CUBIC)
                depth = cv2.resize(depth, original_size, interpolation=interp)

                if sky_mask is not None:
                    sky_mask = cv2.resize(
                        sky_mask.astype(np.uint8), original_size,
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

        return depth, sky_mask

    def _generate_point_cloud(
        self,
        depth_map: np.ndarray,
        image: np.ndarray,
        focal_length: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate 3D point cloud from depth map.

        Uses pinhole camera model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth
        """
        h, w = depth_map.shape
        cx, cy = w / 2, h / 2
        fx = fy = focal_length

        # Downsample for efficiency
        ds = self.config.point_cloud_downsample

        # Create pixel coordinate grids
        u = np.arange(0, w, ds)
        v = np.arange(0, h, ds)
        u_grid, v_grid = np.meshgrid(u, v)

        # Sample depth
        z = depth_map[v_grid, u_grid]

        # Valid depth mask
        valid = (z > 0.1) & (z < self.config.sky_depth_cap)

        # Back-project to 3D
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy

        # Stack and filter
        points = np.stack([x, y, z], axis=-1)
        points = points[valid]

        # Colors
        colors = None
        if self.config.colorize_point_cloud and image is not None:
            # Resize image if needed
            if image.shape[:2] != (h, w):
                try:
                    import cv2
                    image_resized = cv2.resize(image, (w, h))
                except ImportError:
                    image_resized = image
            else:
                image_resized = image

            # Sample colors
            colors = image_resized[v_grid, u_grid]
            colors = colors[valid]

            # Convert BGR to RGB if needed
            if colors.shape[-1] == 3:
                colors = colors[..., ::-1]

        return points, colors

    def _mock_inference(
        self,
        image: np.ndarray,
        focal_length: float,
        start_time: float,
    ) -> DepthResult:
        """
        Generate mock depth result for testing without model.

        Creates a simple depth gradient with noise to simulate real output.
        """
        h, w = image.shape[:2]

        # Create depth gradient (closer at bottom, farther at top)
        # Simulates floor/ground plane
        v = np.arange(h).reshape(-1, 1)
        depth = 1.0 + (h - v) / h * 4.0  # 1m to 5m
        depth = np.broadcast_to(depth, (h, w)).copy()

        # Add some noise
        noise = np.random.randn(h, w) * 0.1
        depth += noise
        depth = np.clip(depth, 0.5, 10.0)

        inference_time = (time.time() - start_time) * 1000

        return DepthResult(
            depth_map=depth.astype(np.float32),
            timestamp=time.time(),
            original_size=(w, h),
            sky_mask=None,
            inference_time_ms=inference_time,
            model_input_size=(self.config.input_width, self.config.input_height),
        )

    def warmup(self, num_iterations: int = 5):
        """
        Warmup inference to stabilize performance.

        TensorRT benefits from warmup to optimize CUDA kernels.
        """
        dummy_input = np.zeros(
            (self.config.input_height, self.config.input_width, 3),
            dtype=np.uint8
        )

        for _ in range(num_iterations):
            self.infer(dummy_input)

        logger.info(f"Depth Anything V3: Warmup complete ({num_iterations} iterations)")

    def __del__(self):
        """Cleanup CUDA resources."""
        if hasattr(self, '_cuda_stream') and self._cuda_stream is not None:
            self._cuda_stream.synchronize()


# =============================================================================
# Factory Function
# =============================================================================

def create_depth_estimator(
    model_size: str = "large",
    precision: str = "fp16",
    device: str = "cuda:0",
) -> DepthAnythingV3:
    """
    Factory function to create Depth Anything V3 estimator.

    Args:
        model_size: "small", "base", or "large"
        precision: "fp16" or "fp32"
        device: "cuda:0", "cuda:1", or "cpu"

    Returns:
        Configured DepthAnythingV3 instance
    """
    size_map = {
        "small": DepthModelSize.SMALL,
        "base": DepthModelSize.BASE,
        "large": DepthModelSize.LARGE,
    }
    precision_map = {
        "fp16": DepthPrecision.FP16,
        "fp32": DepthPrecision.FP32,
    }

    config = DepthEstimationConfig(
        model_size=size_map.get(model_size, DepthModelSize.LARGE),
        precision=precision_map.get(precision, DepthPrecision.FP16),
        device=device,
    )

    return DepthAnythingV3(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Depth Anything V3")
    print("=" * 50)

    # Create estimator
    estimator = create_depth_estimator(model_size="large", precision="fp16")

    # Create test image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Run inference
    result = estimator.infer(test_image)

    print(f"Depth map shape: {result.depth_map.shape}")
    print(f"Depth range: {result.depth_map.min():.2f}m - {result.depth_map.max():.2f}m")
    print(f"Inference time: {result.inference_time_ms:.1f}ms")
    print(f"Sample depth at center: {result.sample_depth_bilinear(640, 360):.2f}m")

    print("\nTest passed!")

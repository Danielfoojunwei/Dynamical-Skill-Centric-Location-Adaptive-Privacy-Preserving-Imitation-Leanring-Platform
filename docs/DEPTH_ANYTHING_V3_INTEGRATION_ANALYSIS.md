# Depth Anything V3 Integration Analysis

## Comparison: Current System vs. ROS2 Depth Anything V3 TRT

**Source:** https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt

---

## Executive Summary

**Recommendation: Integrate Depth Anything V3 for significant improvements in single-camera depth estimation and motion capture accuracy.**

| Aspect | Current System | Depth Anything V3 | Improvement |
|--------|---------------|-------------------|-------------|
| Single-camera depth | Hardcoded 2m assumption | Metric depth per-pixel | **Massive** |
| Multi-camera fallback | Triangulation only | Depth + triangulation fusion | **High** |
| Point cloud generation | Not implemented | Native support | **New capability** |
| Camera calibration assist | Manual only | Depth-based extrinsic hints | **Moderate** |
| TensorRT optimization | Planned but not done | Production-ready | **Ready to use** |

---

## Current System Analysis

### What We Have

**Multi-Camera 3D Triangulation** (`src/drivers/cameras.py`, `src/core/wholebody_pose_pipeline.py`):
- DLT triangulation from multiple 2D views
- RANSAC outlier rejection
- Requires minimum 2 cameras for 3D

**Single-Camera Fallback** (`wholebody_pose_pipeline.py:1113-1122`):
```python
# Current implementation - VERY CRUDE
if len(pose_2d_results) < 2:
    # Single camera: use 2D with depth heuristics
    pose_3d.keypoints_3d[:len(kp_2d), :2] = kp_2d[:, :2] / 500.0  # Scale to meters
    pose_3d.keypoints_3d[:len(kp_2d), 2] = 2.0  # Assume 2m depth  ← HARDCODED!
```

**Budget Allocation** (`src/core/config_loader.py`, `src/pipeline/integrated_pipeline.py`):
- Depth estimation budgeted at 5.0 TFLOPS
- "Depth Anything v2" mentioned but **NOT IMPLEMENTED**
- Just a placeholder in the TFLOPS budget

### Critical Gap

The single-camera depth estimation is essentially broken:
- Assumes fixed 2m depth for ALL body parts
- No perspective/scale correction
- Motion capture accuracy degrades severely with 1 camera

---

## Depth Anything V3 TRT Capabilities

### Core Features

| Feature | Details |
|---------|---------|
| **Model** | DA3METRIC-LARGE (metric depth) |
| **Performance** | 50 FPS on RTX 6000, ~30 FPS expected on Jetson Orin |
| **Precision** | FP16/FP32 TensorRT engines |
| **Input** | [1, 3, 280, 504] normalized RGB |
| **Output** | Metric depth map + sky classification |
| **Point Cloud** | Native generation with camera intrinsics |

### Key Advantages

1. **Metric Depth Estimation**
   - Real depth values in meters (not relative/disparity)
   - Calibrated with camera focal length
   - Formula: `scale = (fx + fy) / 2 / 300.0`

2. **Sky Detection**
   - Separate sky classification head
   - Prevents infinity depth artifacts
   - Important for outdoor camera setups

3. **Point Cloud Generation**
   - Direct 3D point generation from depth
   - RGB colorization support
   - Configurable downsampling

4. **Production-Ready TensorRT**
   - Pre-compiled engines available
   - FP16 optimization
   - Engine caching for fast startup

---

## Integration Benefits for Motion Capture

### 1. Single-Camera Depth Estimation (HIGH VALUE)

**Current Problem:**
```
Camera → 2D Pose → Assume 2m depth → Wrong 3D
```

**With Depth Anything V3:**
```
Camera → 2D Pose → Query Depth Map → Accurate 3D
```

**Implementation:**
```python
class DepthEnhancedPoseEstimator:
    def estimate_3d_single_camera(self, image: np.ndarray, pose_2d: Pose2DResult) -> Pose3DResult:
        # Run depth estimation
        depth_map = self.depth_model.infer(image)  # [H, W] metric depth

        # Sample depth at keypoint locations
        for i, (x, y, conf) in enumerate(pose_2d.keypoints[0]):
            if conf > 0.3:
                # Bilinear interpolation for sub-pixel accuracy
                z = self._sample_depth(depth_map, x, y)

                # Back-project to 3D using camera intrinsics
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                pose_3d.keypoints_3d[i] = [X, Y, z]

        return pose_3d
```

**Expected Improvement:**
- Position error: 50cm → 5-10cm (single camera)
- Enables single-camera teleoperation fallback

### 2. Multi-Camera Triangulation Enhancement (MODERATE VALUE)

**Use Case:** Depth as additional constraint for triangulation

```python
def triangulate_with_depth_prior(
    points_2d: Dict[str, np.ndarray],
    depth_maps: Dict[str, np.ndarray],
    projection_matrices: Dict[str, np.ndarray],
) -> np.ndarray:
    # Standard DLT triangulation
    point_3d_dlt = triangulate_dlt(points_2d, projection_matrices)

    # Depth-based estimates from each camera
    depth_estimates = []
    for cam_id, depth_map in depth_maps.items():
        z = sample_depth(depth_map, points_2d[cam_id])
        point_3d_depth = back_project(points_2d[cam_id], z, intrinsics[cam_id])
        depth_estimates.append(point_3d_depth)

    # Weighted fusion (DLT more reliable with multiple views)
    n_cameras = len(points_2d)
    weight_dlt = min(0.9, n_cameras * 0.3)  # 0.3 per camera, max 0.9
    weight_depth = 1.0 - weight_dlt

    return weight_dlt * point_3d_dlt + weight_depth * np.mean(depth_estimates, axis=0)
```

### 3. Camera Extrinsic Calibration Assistance (MODERATE VALUE)

**Use Case:** Automatic camera placement verification

```python
def verify_camera_extrinsics(
    depth_map: np.ndarray,
    known_floor_region: np.ndarray,  # Mask of floor pixels
    camera_params: CameraParams,
) -> Tuple[bool, float]:
    """
    Verify camera height and orientation using floor depth.

    If depth to floor differs significantly from expected,
    the camera may have moved or extrinsics are wrong.
    """
    floor_depths = depth_map[known_floor_region]
    measured_height = np.median(floor_depths)

    expected_height = camera_params.t[2]  # Z component of translation

    error = abs(measured_height - expected_height)
    is_valid = error < 0.1  # 10cm tolerance

    return is_valid, error
```

### 4. Obstacle Detection for Safety (BONUS VALUE)

**Use Case:** Detect obstacles in teleoperation workspace

```python
def detect_workspace_obstacles(
    depth_map: np.ndarray,
    workspace_mask: np.ndarray,  # Expected clear workspace
    expected_depth_range: Tuple[float, float],
) -> List[ObstacleRegion]:
    """
    Find unexpected objects in the workspace.
    """
    # Points closer than expected = obstacles
    obstacles = depth_map < expected_depth_range[0]
    obstacles &= workspace_mask

    # Connected components analysis
    contours = find_contours(obstacles)

    return [ObstacleRegion(contour, centroid, depth) for contour in contours]
```

---

## Performance Analysis

### Jetson Orin AGX (Our Target Platform)

| Model | RTX 6000 (Benchmark) | Orin AGX (Estimated) |
|-------|---------------------|---------------------|
| DA3METRIC-LARGE | 50 FPS | 25-30 FPS |
| DA3METRIC-SMALL | 80+ FPS | 40-50 FPS |

**Our TFLOPS Budget:**
- Allocated: 5.0 TFLOPS for depth estimation
- DA V3 requirement: ~8.9 TFLOPS at full resolution
- Solution: Use smaller model or reduce resolution

### Memory Footprint

| Component | Size |
|-----------|------|
| ONNX Model | ~250 MB |
| TRT Engine (FP16) | ~150 MB |
| Runtime VRAM | ~500 MB |

**Fits within our Jetson Orin 64GB budget.**

---

## Integration Architecture

### Proposed Module Structure

```
src/core/
├── depth_estimation/
│   ├── __init__.py
│   ├── depth_anything_v3.py      # TensorRT inference wrapper
│   ├── depth_pose_fusion.py      # Combine depth with pose
│   └── depth_calibration.py      # Camera verification tools
```

### Key Classes

```python
@dataclass
class DepthEstimationConfig:
    """Configuration for Depth Anything V3."""
    model_path: str = "models/DA3METRIC-LARGE.onnx"
    engine_path: str = "models/DA3METRIC-LARGE.engine"
    precision: str = "fp16"
    input_size: Tuple[int, int] = (504, 280)  # W, H
    sky_threshold: float = 0.3
    sky_depth_cap: float = 200.0


class DepthAnythingV3:
    """TensorRT-accelerated depth estimation."""

    def __init__(self, config: DepthEstimationConfig):
        self.config = config
        self.engine = self._load_or_build_engine()

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate metric depth from RGB image.

        Args:
            image: [H, W, 3] BGR image

        Returns:
            depth: [H, W] metric depth in meters
        """
        # Preprocess
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.config.input_size)
        normalized = self._normalize_imagenet(resized)

        # TensorRT inference
        depth_raw, sky_logits = self._run_inference(normalized)

        # Postprocess
        depth = self._apply_focal_scaling(depth_raw)
        depth = self._handle_sky_regions(depth, sky_logits)
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        return depth
```

### Integration with Pose Pipeline

```python
class WholeBodyRetargetingPipelineV2:
    """Enhanced pipeline with depth estimation."""

    def __init__(self, ...):
        # Existing components
        self.pose_estimator = RTMPoseTensorRT(pose_config)
        self.triangulator = MultiViewTriangulator(cameras)

        # NEW: Depth estimation
        self.depth_estimator = DepthAnythingV3(depth_config)
        self.depth_pose_fusion = DepthPoseFusion()

    def process_frame(self, images: Dict[str, np.ndarray], ...):
        # 2D pose estimation (existing)
        pose_2d_results = {
            cam_id: self.pose_estimator.infer(img, cam_id)
            for cam_id, img in images.items()
        }

        # NEW: Depth estimation on all cameras
        depth_maps = {
            cam_id: self.depth_estimator.infer(img)
            for cam_id, img in images.items()
        }

        # 3D reconstruction (enhanced)
        if len(images) >= 2:
            # Multi-view: triangulation + depth refinement
            pose_3d = self.triangulator.triangulate(pose_2d_results)
            pose_3d = self.depth_pose_fusion.refine_with_depth(
                pose_3d, pose_2d_results, depth_maps
            )
        else:
            # Single view: depth-based 3D estimation
            cam_id = next(iter(images.keys()))
            pose_3d = self.depth_pose_fusion.estimate_from_depth(
                pose_2d_results[cam_id],
                depth_maps[cam_id],
                self.cameras[cam_id]
            )

        return pose_3d
```

---

## Implementation Roadmap

### Phase 1: Basic Integration (3-4 days)

1. Download DA V3 ONNX model
2. Create `DepthAnythingV3` class with TensorRT
3. Test on single camera
4. Verify depth accuracy

### Phase 2: Pose Fusion (2-3 days)

1. Implement `DepthPoseFusion` class
2. Single-camera 3D pose from depth
3. Multi-camera depth refinement
4. Unit tests

### Phase 3: Camera Calibration (2 days)

1. Implement extrinsic verification
2. Floor detection for height calibration
3. Integration with calibration tools

### Phase 4: Production (2-3 days)

1. Jetson Orin optimization
2. Memory profiling
3. Latency benchmarking
4. Integration tests

**Total Effort: 9-12 days**

---

## Required Model Downloads

```bash
# Option 1: Pre-converted ONNX from HuggingFace
mkdir -p models/depth_anything_v3
wget -O models/depth_anything_v3/DA3METRIC-LARGE.onnx \
    https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX/resolve/main/DA3METRIC-LARGE.onnx

# Option 2: Convert from PyTorch (requires depth_anything_v3 repo)
git clone https://github.com/DepthAnything/Depth-Anything-V3
cd Depth-Anything-V3
python export_onnx.py --model large --output ../models/depth_anything_v3/DA3METRIC-LARGE.onnx
```

---

## Comparison with Alternatives

| Method | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Depth Anything V3** | Metric depth, 50 FPS, TRT ready | Larger model | **Best for teleoperation** |
| ZoeDepth | Good metric accuracy | Slower, no TRT support | Fallback option |
| MiDaS | Widely used, stable | Relative depth only | Not suitable |
| Stereo matching | Very accurate | Requires stereo rig | Alternative for dual-cam |

---

## Conclusion

**Integrate Depth Anything V3 TRT** for these reasons:

1. **Massive improvement for single-camera scenarios** - From 50cm+ error to 5-10cm
2. **Production-ready TensorRT** - No need to optimize ourselves
3. **Fits within our TFLOPS budget** - 5 TFLOPS allocated, ~5-9 needed
4. **Point cloud capability** - Enables scene understanding
5. **Already budgeted** - `depth_estimation: 5.0 TFLOPS` in config

The current hardcoded `2.0m` depth assumption is a critical weakness that Depth Anything V3 directly addresses.

---

## References

- [Depth Anything V3 Paper](https://arxiv.org/abs/2406.09422)
- [ROS2 TRT Implementation](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt)
- [HuggingFace ONNX Models](https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX)
- [DepthAnything GitHub](https://github.com/DepthAnything/Depth-Anything-V3)

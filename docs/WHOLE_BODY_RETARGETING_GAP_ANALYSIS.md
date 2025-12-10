# Whole-Body Retargeting Gap Analysis

## MMPose + GMR Integration Assessment

**Date:** December 2025
**Current Implementation Status:** **~65% Complete**

**Files Analyzed:**
- `src/core/wholebody_pose_pipeline.py` (1254 lines)
- `src/core/whole_body_gmr.py` (451 lines)
- `src/core/retargeting.py` (999 lines)
- `src/core/human_state.py` (312 lines)
- `src/core/hand_retargeting.py` (780+ lines)
- `src/platform/calibration/mmpose_calibration.py` (1484 lines)
- `src/platform/calibration/dyglove_calibration.py` (69KB)

---

## Executive Summary

The architecture is solid, but the implementation relies heavily on **mock/placeholder code** rather than real inference and retargeting. The current state is approximately **65% complete** for a functional whole-body retargeting system.

### Overall Status

| Component | Status | Completeness |
|-----------|--------|--------------|
| COCO-WholeBody 133 keypoint definitions | ✅ Done | 100% |
| 2D Pose estimation abstraction | ✅ Done | 90% |
| Multi-view DLT triangulation | ✅ Done | 85% |
| Quality scoring (confidence, smoothness, biomechanics) | ✅ Done | 80% |
| Hand retargeting (21-DOF) | ✅ Done | 90% |
| GMR wrapper | ⚠️ Partial | 60% |
| **Real RTMPose model inference** | ❌ Missing | 10% |
| **GMR library integration** | ❌ Missing | 15% |
| **DeepStream multi-stream pipeline** | ❌ Missing | 0% |
| **Multi-camera sync** | ❌ Missing | 20% |
| **Body-glove calibration** | ⚠️ Partial | 40% |
| **SMPL body model** | ❌ Missing | 0% |
| **Production IK solver** | ⚠️ Partial | 30% |

---

## What Each Mock Currently Returns

### RTMPose Mock Inference

**Location:** `src/core/wholebody_pose_pipeline.py:329-387`

```python
def _mock_inference(self, camera_id: str, timestamp: float, image_size: Tuple[int, int]) -> Pose2DResult:
    """Generate mock pose result for testing."""
    w, h = image_size
    center_x, center_y = w // 2, h // 2

    # Returns hardcoded standing pose in image center - NO ACTUAL DETECTION!
    keypoints = np.zeros((1, n_keypoints, 3))
    keypoints[0, 0] = [center_x, center_y - 150, 0.95]  # nose
    keypoints[0, 5] = [center_x - 80, center_y - 100, 0.95]  # left_shoulder
    keypoints[0, 6] = [center_x + 80, center_y - 100, 0.95]  # right_shoulder
    # ... more hardcoded positions
```

### GMR Mock Retargeter

**Location:** `src/core/whole_body_gmr.py:200-244`

```python
class MockGMRRetargeter:
    def retarget(self, human_motion: np.ndarray) -> np.ndarray:
        """Mock retargeting - NOT real retargeting!"""
        q = np.zeros(self.n_joints)

        # Simple arctan2 on arm vectors - NOT real kinematic retargeting
        upper_arm = r_elbow - r_shoulder
        q[0] = np.arctan2(upper_arm[1], upper_arm[0])
        q[1] = np.arctan2(upper_arm[2], np.linalg.norm(upper_arm[:2]))

        # Random noise added for "realism" - SHOULD NOT BE IN PRODUCTION
        q += np.random.randn(self.n_joints) * 0.01

        return q
```

---

## Critical Gaps (P0) - Must Fix for Production

### 1. No Real RTMPose Inference

**Problem:** `RTMPoseONNX.infer()` always returns mock data - no actual model loading or inference.

**Current State:**
```python
# In wholebody_pose_pipeline.py line 306-320
def infer(self, image: np.ndarray, camera_id: str = "cam0") -> Pose2DResult:
    if not self._initialized:
        return self._mock_inference(...)  # Always returns mock data!
    # Real inference would go here - NOT IMPLEMENTED
    return self._mock_inference(...)
```

**Missing:**
- Actual ONNX model loading and session creation
- TensorRT engine compilation for Jetson
- Image pre-processing pipeline (resize, normalize)
- Post-processing (decode heatmaps/SimCC, NMS)
- Person detection integration (YOLO/RTMDet)

**Effort:** 3-4 days

---

### 2. No GMR Library Integration

**Problem:** GMR library doesn't exist as Python package - `MockGMRRetargeter` always used.

**Current State:**
```python
# In whole_body_gmr.py line 303-326
def _init_gmr(self):
    try:
        from gmr import GeneralMotionRetargeting  # DOES NOT EXIST
        ...
    except ImportError:
        self.gmr = MockGMRRetargeter(...)  # Always uses mock!
```

**Missing:**
- GMR library doesn't have a pip package
- Need to implement native retargeting algorithm OR build from source
- URDF loading and kinematic chain parsing
- Forward/inverse kinematics computation
- Joint limit enforcement

**Effort:** 5-7 days

---

### 3. Missing DeepStream Pipeline

**Problem:** No multi-stream GPU pipeline - using sequential ONNX Runtime.

**Current State:**
- Sequential inference on each camera frame
- No batched GPU processing
- No NVIDIA-optimized pipeline

**Missing:**
- GStreamer/DeepStream pipeline construction
- NvInfer plugin configuration for pose models
- Multi-stream batching (nvstreammux)
- GPU memory management
- INT8/FP16 calibration for inference

**Effort:** 4-5 days

---

### 4. Incomplete Body-Glove Calibration

**Problem:** `T_world_imu` transform is never calibrated - glove IMU not aligned to world frame.

**Current State:**
```python
# In wholebody_pose_pipeline.py line 938-1000
class BodyGloveFusion:
    def __init__(self):
        self.T_world_imu_left: Optional[np.ndarray] = None  # Never calibrated!
        self.T_world_imu_right: Optional[np.ndarray] = None
        self._calibrated = False  # Always False
```

**Missing:**
- Automatic calibration procedure
- Procrustes alignment algorithm
- IMU → world frame estimation from arm direction
- Temporal alignment (glove vs camera timestamps)

**Effort:** 2-3 days

---

## Major Gaps (P1)

### 5. No SMPL Body Model

**Problem:** Many retargeting methods require SMPL as intermediate representation.

**Why Needed:**
- GMR and similar methods use SMPL/SMPL-X internally
- Provides consistent body topology across different humans
- Enables mesh-based rendering for visualization
- Required for physics simulation and contact estimation

**Effort:** 3-4 days

---

### 6. No Multi-Camera Synchronization

**Problem:** Without hardware sync, triangulation errors increase 3-5cm at fast motions.

**Current State:**
- Assumes cameras are synchronized (they're not)
- No timestamp interpolation
- No hardware trigger support

**Impact:**
- 33ms desync @ 30fps → 3-5cm position error for fast arm motions
- Industrial teleoperation requires <10ms sync for safety

**Effort:** 3-4 days

---

### 7. Incomplete IK Solver

**Problem:** Placeholder numerical Jacobian, no URDF loading, no collision checking.

**Current State:**
- Simple transpose Jacobian in `retargeting.py`
- No robot model loading
- No joint limit handling
- No collision avoidance

**Effort:** 2-3 days

---

## What's Actually Implemented Well ✅

| Component | Status | Notes |
|-----------|--------|-------|
| COCO-WholeBody 133 keypoints | ✅ Complete | Proper joint definitions in `COCOWholeBodyKeypoints` class |
| DLT triangulation | ✅ Works | SVD-based multi-view triangulation with weighted least squares |
| Quality scoring | ✅ Works | Confidence, smoothness, biomechanics checks |
| Hand retargeting (21-DOF) | ✅ Complete | Multiple gripper targets (parallel jaw, Allegro, etc.) |
| Pipeline architecture | ✅ Solid | Clean abstractions with ABC/Protocol patterns |
| Dataclasses for state | ✅ Clean | Well-typed state objects throughout |
| Temporal smoothing | ✅ Works | Exponential smoothing on joint configs |

---

## Files Needed to Complete Implementation

| New File | Purpose | Est. Lines |
|----------|---------|------------|
| `src/core/rtmpose_inference.py` | Real ONNX/TensorRT inference | ~500 |
| `src/core/gmr_native.py` | Native retargeting algorithm | ~800 |
| `src/core/pinocchio_ik.py` | Production IK solver with Pinocchio | ~250 |
| `src/core/deepstream_pose.py` | Multi-camera GPU pipeline | ~400 |
| `src/core/multi_cam_sync.py` | Hardware/software synchronization | ~300 |
| `src/core/smpl_model.py` | SMPL body model wrapper | ~300 |
| `src/core/pose_tracker.py` | Kalman filter + multi-person tracking | ~400 |

---

## Required Model Downloads

```bash
# RTMPose WholeBody (133 keypoints) - ~250MB
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth

# Convert to ONNX (requires mmpose installed)
python tools/deployment/pytorch2onnx.py \
    configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb64-270e_coco-ubody-wholebody-384x288.py \
    rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth \
    --output-file rtmpose-l-wholebody.onnx

# Person detector (RTMDet or YOLOv8) - ~150MB
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# SMPL model (requires registration at smpl.is.tue.mpg.de)
# Download: SMPL_NEUTRAL.pkl (~10MB)
```

---

## Implementation Roadmap

### Week 1: Core Inference
```
├── Download RTMPose/RTMDet models from MMPose/Ultralytics
├── Implement RTMPoseRealInference class with ONNX Runtime
├── Add image pre-processing (resize, normalize, pad)
├── Add heatmap/SimCC post-processing
├── Integrate YOLOv8 person detection
└── Test on single camera with real video
```

### Week 2: Multi-Camera
```
├── Implement hardware sync (GPIO trigger) or software sync (PTP/NTP)
├── Add timestamp interpolation for pose alignment
├── Improve triangulation with RANSAC outlier rejection
├── Add Kalman filter for temporal smoothing
└── Test with 4-camera rig
```

### Week 3: GMR/Retargeting
```
├── Implement native GMR algorithm (position → joint angles)
├── Add URDF loading with yourdfpy/urdfpy
├── Integrate Pinocchio for IK
├── Create joint mapping configs for target robots
└── Test end-to-end retargeting
```

### Week 4: Integration
```
├── Complete body-glove calibration procedure
├── Add temporal alignment (camera vs glove timestamps)
├── Implement full teleoperation loop
├── Performance profiling and optimization
└── Add latency monitoring
```

**Total Effort:** ~20-25 engineering days

---

## Blocking Dependencies

| Dependency | Source | Notes |
|------------|--------|-------|
| RTMPose ONNX models | MMPose Model Zoo | Download or export from checkpoints |
| Person detector | Ultralytics/MMDet | YOLOv8 or RTMDet |
| Robot URDFs | Manufacturer | Daimon VTLA, etc. |
| SMPL model | smpl.is.tue.mpg.de | Requires registration |
| Camera sync hardware | GPIO or PTP support | Jetson GPIO or network PTP |
| Pinocchio | pip install pin | For production IK |

---

## Summary

The architecture is sound. The main work is **replacing mock implementations with real ones**:

| Gap | Mock Location | Real Implementation Needed |
|-----|---------------|---------------------------|
| RTMPose inference | `wholebody_pose_pipeline.py:329` | ONNX/TensorRT session with real model |
| GMR retargeting | `whole_body_gmr.py:211` | Native algorithm or GMR library build |
| Camera sync | Not implemented | GPIO trigger or PTP-based sync |
| Glove calibration | `wholebody_pose_pipeline.py:954` | Procrustes alignment procedure |

**Priority Order:**
1. **P0 (Critical):** Real inference, GMR native, DeepStream → 12-15 days
2. **P1 (Major):** SMPL, sync, IK solver → 8-10 days
3. **P2 (Minor):** Filtering, monitoring, visualization → 5 days

The gap analysis confirms that **~35% of the system needs to be implemented** to move from prototype to production.

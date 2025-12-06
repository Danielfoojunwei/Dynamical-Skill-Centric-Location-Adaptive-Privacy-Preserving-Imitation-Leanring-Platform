"""
ONVIF Camera Interface Module

This module handles RTSP stream ingestion from ONVIF-compliant IP cameras.
For 3D human pose estimation, we need multiple synchronized camera views
to triangulate body keypoints into world coordinates.

Key concepts:

1. ONVIF (Open Network Video Interface Forum) is a standard for IP cameras.
   Most industrial cameras support it, giving us a unified API regardless
   of manufacturer (Axis, Hikvision, Dahua, etc.).

2. RTSP (Real Time Streaming Protocol) is how we actually get video frames.
   The ONVIF camera exposes an RTSP URL, and we pull frames from it.

3. Multi-view synchronization is critical for triangulation. If cameras
   aren't capturing at the same moment, the 3D reconstruction will be wrong.
   We handle this with either hardware sync (genlock) or software sync
   (timestamp-based frame matching).

4. Calibration: each camera needs intrinsic parameters (focal length, 
   principal point, distortion) and extrinsic parameters (position and
   orientation in world frame). These come from a calibration procedure
   using a checkerboard or ChArUco board.

Architecture:
    OnvifCamera: Single camera interface
    MultiViewCameraRig: Manages multiple cameras with synchronization
    CameraCalibration: Stores and applies camera calibration data
"""

import cv2
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Camera Calibration Data Structure
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Stores intrinsic and extrinsic calibration for a single camera.
    
    Intrinsic parameters describe the camera's internal geometry:
        - camera_matrix (K): 3x3 matrix with focal lengths (fx, fy) and 
          principal point (cx, cy)
        - dist_coeffs: Distortion coefficients for lens correction
          (radial k1,k2,k3 and tangential p1,p2)
    
    Extrinsic parameters describe camera pose in world frame:
        - R: 3x3 rotation matrix (camera orientation)
        - t: 3x1 translation vector (camera position)
        - T_world_camera: 4x4 homogeneous transform combining R and t
    
    The projection matrix P = K @ [R | t] maps 3D world points to 2D pixels.
    """
    camera_id: str
    
    # Intrinsics
    camera_matrix: np.ndarray           # 3x3 intrinsic matrix K
    dist_coeffs: np.ndarray             # Distortion coefficients [k1,k2,p1,p2,k3]
    image_size: Tuple[int, int]         # (width, height)
    
    # Extrinsics (camera pose in world frame)
    R: np.ndarray                       # 3x3 rotation matrix
    t: np.ndarray                       # 3x1 translation vector
    
    # Derived quantities (computed on init)
    T_world_camera: np.ndarray = field(init=False)  # 4x4 transform
    projection_matrix: np.ndarray = field(init=False)  # 3x4 projection P
    
    def __post_init__(self):
        """Compute derived matrices from R, t, and K."""
        # Build 4x4 transform: T_world_camera transforms points from 
        # camera frame to world frame
        self.T_world_camera = np.eye(4)
        self.T_world_camera[:3, :3] = self.R
        self.T_world_camera[:3, 3] = self.t.flatten()
        
        # Projection matrix: maps world points to normalized image coordinates
        # P = K @ [R^T | -R^T @ t] (note: this is for world-to-camera)
        R_cam = self.R.T  # Camera rotation (world to camera)
        t_cam = -R_cam @ self.t  # Camera translation (world to camera)
        
        Rt = np.hstack([R_cam, t_cam.reshape(3, 1)])
        self.projection_matrix = self.camera_matrix @ Rt
    
    def project_point(self, point_world: np.ndarray) -> np.ndarray:
        """
        Project a 3D world point to 2D pixel coordinates.
        
        Args:
            point_world: [3] or [N, 3] array of 3D points in world frame
            
        Returns:
            [2] or [N, 2] array of pixel coordinates
        """
        point_world = np.atleast_2d(point_world)
        
        # Add homogeneous coordinate
        ones = np.ones((point_world.shape[0], 1))
        points_h = np.hstack([point_world, ones])  # [N, 4]
        
        # Project: p = P @ X (gives homogeneous 2D)
        points_2d_h = (self.projection_matrix @ points_h.T).T  # [N, 3]
        
        # Normalize by z to get pixel coordinates
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
        
        return points_2d.squeeze()
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from 2D pixel coordinates.
        
        This is important before triangulation since the camera model
        assumes an ideal pinhole camera.
        """
        points_2d = np.atleast_2d(points_2d).astype(np.float32)
        
        # OpenCV's undistortPoints returns normalized coordinates
        # We want pixel coordinates, so we reapply the camera matrix
        undistorted = cv2.undistortPoints(
            points_2d.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix
        )
        
        return undistorted.reshape(-1, 2)
    
    def save(self, path: Path):
        """Save calibration to JSON file."""
        data = {
            'camera_id': self.camera_id,
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_size': list(self.image_size),
            'R': self.R.tolist(),
            't': self.t.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'CameraCalibration':
        """Load calibration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            camera_id=data['camera_id'],
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['dist_coeffs']),
            image_size=tuple(data['image_size']),
            R=np.array(data['R']),
            t=np.array(data['t']),
        )
    
    @classmethod
    def create_default(cls, camera_id: str, image_size: Tuple[int, int] = (1920, 1080)) -> 'CameraCalibration':
        """
        Create a default calibration for testing.
        
        This assumes a typical wide-angle camera with no distortion.
        DO NOT use this for real 3D reconstruction - run actual calibration!
        """
        w, h = image_size
        
        # Assume 60° horizontal FOV
        fov_h = np.radians(60)
        fx = w / (2 * np.tan(fov_h / 2))
        fy = fx  # Square pixels
        cx, cy = w / 2, h / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return cls(
            camera_id=camera_id,
            camera_matrix=K,
            dist_coeffs=np.zeros(5),
            image_size=image_size,
            R=np.eye(3),
            t=np.zeros(3),
        )


# ---------------------------------------------------------------------------
# Single Camera Interface
# ---------------------------------------------------------------------------

@dataclass
class CameraFrame:
    """A captured frame with metadata."""
    image: np.ndarray                   # BGR image [H, W, 3]
    timestamp: float                    # Capture timestamp (Unix time)
    camera_id: str                      # Which camera captured this
    frame_number: int                   # Sequential frame counter
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape


class OnvifCamera:
    """
    Interface to a single ONVIF camera via RTSP.
    
    This class handles:
    - Connecting to the RTSP stream
    - Capturing frames in a background thread
    - Providing the latest frame on demand
    - Graceful reconnection on stream errors
    
    Usage:
        camera = OnvifCamera("cam1", "rtsp://user:pass@192.168.1.100:554/stream")
        camera.start()
        
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("Camera", frame.image)
        
        camera.stop()
    
    For ONVIF discovery and stream URL retrieval, use the onvif-zeep library.
    Here we assume the RTSP URL is already known.
    """
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        calibration: Optional[CameraCalibration] = None,
        buffer_size: int = 2,
        reconnect_delay: float = 2.0,
        simulation_mode: bool = False,
    ):
        """
        Initialize camera interface.
        
        Args:
            camera_id: Unique identifier for this camera
            rtsp_url: Full RTSP URL including credentials
            calibration: Camera calibration data (can be set later)
            buffer_size: Number of frames to buffer (small = low latency)
            reconnect_delay: Seconds to wait before reconnecting on error
            simulation_mode: If True, generates synthetic frames instead of connecting
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.calibration = calibration
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.simulation_mode = simulation_mode
        
        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_frame: Optional[CameraFrame] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._fps_history: List[float] = []
        self._last_frame_time: float = 0
    
    def _create_capture(self) -> cv2.VideoCapture:
        """
        Create OpenCV VideoCapture with optimized settings for RTSP.
        
        We use GStreamer backend when available for better performance,
        falling back to FFmpeg otherwise.
        """
        # Try GStreamer pipeline first (lower latency on Linux)
        gst_pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            # Fall back to FFmpeg
            logger.info(f"[{self.camera_id}] GStreamer failed, trying FFmpeg")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Optimize FFmpeg settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")
        
        return cap
    
    def _capture_loop(self):
        """
        Background thread that continuously captures frames.
        
        This runs in a separate thread so frame capture doesn't block
        the main processing pipeline. We keep only the most recent frames
        to minimize latency.
        """
        while self._running:
            try:
                if self.simulation_mode:
                    # Synthetic frame generation
                    time.sleep(1.0 / 30.0) # Simulate 30 FPS
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    # Add noise and timestamp to make it look alive
                    noise = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
                    cv2.add(frame, noise, frame)
                    cv2.putText(frame, f"{self.camera_id} SIMULATION {time.time():.2f}", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    ret = True
                elif self._cap is None or not self._cap.isOpened():
                    logger.info(f"[{self.camera_id}] Connecting to stream...")
                    self._cap = self._create_capture()
                    logger.info(f"[{self.camera_id}] Connected")
                    ret, frame = self._cap.read()
                else:
                    ret, frame = self._cap.read()
                
                if not ret:
                    logger.warning(f"[{self.camera_id}] Frame read failed, reconnecting...")
                    self._cap.release()
                    self._cap = None
                    time.sleep(self.reconnect_delay)
                    continue
                
                # Create frame object with timestamp
                now = time.time()
                camera_frame = CameraFrame(
                    image=frame,
                    timestamp=now,
                    camera_id=self.camera_id,
                    frame_number=self._frame_count,
                )
                self._frame_count += 1
                
                # Update FPS tracking
                if self._last_frame_time > 0:
                    dt = now - self._last_frame_time
                    if dt > 0:
                        fps = 1.0 / dt
                        self._fps_history.append(fps)
                        if len(self._fps_history) > 30:
                            self._fps_history.pop(0)
                self._last_frame_time = now
                
                # Store latest frame (thread-safe)
                with self._lock:
                    self._last_frame = camera_frame
                
                # Also put in queue for synchronized access
                try:
                    # Non-blocking put; drop oldest if full
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_queue.put_nowait(camera_frame)
                except queue.Full:
                    pass
                    
            except Exception as e:
                logger.error(f"[{self.camera_id}] Capture error: {e}")
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                time.sleep(self.reconnect_delay)
    
    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # Wait for first frame
        timeout = 5.0
        start = time.time()
        while self._last_frame is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        if self._last_frame is None:
            logger.warning(f"[{self.camera_id}] No frame received within {timeout}s")
    
    def stop(self):
        """Stop the capture thread and release resources."""
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def get_frame(self) -> Optional[CameraFrame]:
        """
        Get the most recent frame (non-blocking).
        
        Returns None if no frame is available yet.
        """
        with self._lock:
            return self._last_frame
    
    def get_frame_blocking(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """
        Wait for and return the next frame.
        
        This is useful for synchronized multi-camera capture.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    @property
    def fps(self) -> float:
        """Get estimated frames per second."""
        if not self._fps_history:
            return 0.0
        return np.mean(self._fps_history)
    
    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()
    
    def get_pi0_frame(self, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Get frame optimized for Pi0 VLA model.
        
        Performs:
        1. Resize to target_size (default 224x224)
        2. Convert BGR to RGB
        3. Normalize to [0, 1] range (float32)
        
        This offloads preprocessing from the main inference loop.
        """
        frame = self.get_frame()
        if frame is None:
            return None
            
        # Resize
        resized = cv2.resize(frame.image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()



# ---------------------------------------------------------------------------
# Multi-View Camera Rig
# ---------------------------------------------------------------------------

@dataclass
class SynchronizedFrameSet:
    """
    A set of frames from multiple cameras captured at approximately the same time.
    
    For triangulation to work correctly, all frames should be from the same
    instant in time. We use timestamps to find the best matching frames.
    """
    frames: Dict[str, CameraFrame]      # camera_id -> frame
    reference_timestamp: float          # The target timestamp
    max_time_delta: float               # Largest deviation from reference
    
    def get_frame(self, camera_id: str) -> Optional[CameraFrame]:
        return self.frames.get(camera_id)
    
    @property
    def camera_ids(self) -> List[str]:
        return list(self.frames.keys())
    
    @property
    def num_cameras(self) -> int:
        return len(self.frames)
    
    def get_images(self) -> Dict[str, np.ndarray]:
        """Get dictionary of camera_id -> image."""
        return {cid: f.image for cid, f in self.frames.items()}


class MultiViewCameraRig:
    """
    Manages multiple cameras for synchronized multi-view capture.
    
    This class is responsible for:
    1. Managing multiple OnvifCamera instances
    2. Synchronizing frame capture across cameras
    3. Providing calibration data for triangulation
    
    Synchronization Strategies:
    
    1. Hardware sync (genlock): All cameras share a sync signal and capture
       at exactly the same moment. This is ideal but requires compatible
       cameras and extra cabling.
    
    2. Software sync (timestamp matching): Each camera captures independently,
       and we match frames by timestamp. Works with any cameras but introduces
       some temporal error (typically a few ms).
    
    3. Triggered capture: We send a capture command to all cameras at once.
       Supported by some ONVIF cameras but adds latency.
    
    We implement software sync here since it works universally.
    
    Typical Setup for Human Pose Capture:
    
        Camera 1 (Front): Captures face and front of body
        Camera 2 (Side):  Captures profile view
        Camera 3 (Back/Other side): Captures back or other side
    
    Placement tips:
    - Cameras at ~45-90° angles to each other
    - All cameras should see the capture volume
    - Avoid placing cameras directly opposite (poor depth estimation)
    - Typical baseline: 2-4 meters between cameras
    """
    
    def __init__(
        self,
        sync_tolerance: float = 0.05,  # Max time difference between frames (50ms)
        simulation_mode: bool = False,
    ):
        """
        Initialize the camera rig.
        
        Args:
            sync_tolerance: Maximum acceptable time difference between
                           synchronized frames, in seconds.
            simulation_mode: If True, all cameras will run in simulation mode.
        """
        self.cameras: Dict[str, OnvifCamera] = {}
        self.calibrations: Dict[str, CameraCalibration] = {}
        self.sync_tolerance = sync_tolerance
        self.simulation_mode = simulation_mode
        
        self._running = False
    
    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        calibration: Optional[CameraCalibration] = None,
    ):
        """
        Add a camera to the rig.
        
        Args:
            camera_id: Unique identifier (e.g., "cam_front", "cam_left")
            rtsp_url: RTSP URL for the camera
            calibration: Camera calibration data
        """
        camera = OnvifCamera(camera_id, rtsp_url, calibration, simulation_mode=self.simulation_mode)
        self.cameras[camera_id] = camera
        
        if calibration is not None:
            self.calibrations[camera_id] = calibration
    
    def set_calibration(self, camera_id: str, calibration: CameraCalibration):
        """Set or update calibration for a camera."""
        if camera_id not in self.cameras:
            raise ValueError(f"Unknown camera: {camera_id}")
        
        self.calibrations[camera_id] = calibration
        self.cameras[camera_id].calibration = calibration
    
    def load_calibrations(self, directory: Path):
        """
        Load calibration files from a directory.
        
        Expects files named {camera_id}_calibration.json
        """
        directory = Path(directory)
        
        for camera_id in self.cameras:
            calib_path = directory / f"{camera_id}_calibration.json"
            if calib_path.exists():
                calib = CameraCalibration.load(calib_path)
                self.set_calibration(camera_id, calib)
                logger.info(f"Loaded calibration for {camera_id}")
            else:
                logger.warning(f"No calibration file for {camera_id}")
    
    def start(self):
        """Start all cameras."""
        if self._running:
            return
        
        logger.info(f"Starting {len(self.cameras)} cameras...")
        
        for camera_id, camera in self.cameras.items():
            camera.start()
            logger.info(f"  {camera_id}: started")
        
        self._running = True
        
        # Wait for all cameras to have at least one frame
        timeout = 10.0
        start = time.time()
        while time.time() - start < timeout:
            all_ready = all(cam.get_frame() is not None for cam in self.cameras.values())
            if all_ready:
                break
            time.sleep(0.1)
        
        ready_count = sum(1 for cam in self.cameras.values() if cam.get_frame() is not None)
        logger.info(f"Camera rig ready: {ready_count}/{len(self.cameras)} cameras online")
    
    def stop(self):
        """Stop all cameras."""
        self._running = False
        
        for camera in self.cameras.values():
            camera.stop()
    
    def get_synchronized_frames(self) -> Optional[SynchronizedFrameSet]:
        """
        Get a synchronized set of frames from all cameras.
        
        This method finds the best matching frames across cameras based on
        timestamps. If cameras are well-synchronized (hardware sync), the
        timestamps will be very close. With software sync, there may be
        some variation.
        
        Returns None if not enough cameras have frames or if synchronization
        fails (timestamps too far apart).
        """
        if not self._running:
            return None
        
        # Gather latest frame from each camera
        latest_frames = {}
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is not None:
                latest_frames[camera_id] = frame
        
        if len(latest_frames) < 2:
            # Need at least 2 cameras for triangulation
            return None
        
        # Use median timestamp as reference (robust to outliers)
        timestamps = [f.timestamp for f in latest_frames.values()]
        reference_time = np.median(timestamps)
        
        # Check if all frames are within tolerance
        max_delta = max(abs(t - reference_time) for t in timestamps)
        
        if max_delta > self.sync_tolerance:
            logger.debug(f"Frame sync failed: max delta = {max_delta*1000:.1f}ms")
            # Still return what we have, but flag the sync issue
        
        return SynchronizedFrameSet(
            frames=latest_frames,
            reference_timestamp=reference_time,
            max_time_delta=max_delta,
        )
    
    def get_projection_matrices(self) -> Dict[str, np.ndarray]:
        """Get projection matrices for all calibrated cameras."""
        return {
            cid: calib.projection_matrix 
            for cid, calib in self.calibrations.items()
        }
    
    @property
    def num_cameras(self) -> int:
        return len(self.cameras)
    
    @property
    def camera_ids(self) -> List[str]:
        return list(self.cameras.keys())
    
    @property
    def all_calibrated(self) -> bool:
        """Check if all cameras have calibration data."""
        return all(cid in self.calibrations for cid in self.cameras)
    
    def get_calibration(self) -> Dict[str, CameraCalibration]:
        """Get all calibration data."""
        return self.calibrations

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Alias for compatibility with integrated pipeline
MultiCameraManager = MultiViewCameraRig



# ---------------------------------------------------------------------------
# Triangulation Utilities
# ---------------------------------------------------------------------------

def triangulate_point(
    points_2d: Dict[str, np.ndarray],
    projection_matrices: Dict[str, np.ndarray],
    min_views: int = 2,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Triangulate a 3D point from multiple 2D observations.
    
    Uses Direct Linear Transform (DLT) to find the 3D point that best
    explains all 2D observations. This is a linear least-squares solution.
    
    Args:
        points_2d: camera_id -> [x, y] pixel coordinates (1D or 2D array)
        projection_matrices: camera_id -> 3x4 projection matrix
        min_views: Minimum number of views required
        
    Returns:
        point_3d: [x, y, z] in world coordinates, or None if failed
        reprojection_error: Mean pixel error across views
    """
    # Get common cameras (have both 2D point and projection matrix)
    common_cameras = set(points_2d.keys()) & set(projection_matrices.keys())
    
    if len(common_cameras) < min_views:
        return None, float('inf')
    
    # Build the DLT system: A @ X = 0
    # For each view, we get 2 equations from the constraint u × (P @ X) = 0
    A_rows = []
    
    for camera_id in common_cameras:
        p2d = np.atleast_1d(points_2d[camera_id]).flatten()
        P = projection_matrices[camera_id]
        
        if len(p2d) < 2:
            continue
            
        x, y = p2d[0], p2d[1]
        
        # Two equations per view
        A_rows.append(x * P[2, :] - P[0, :])
        A_rows.append(y * P[2, :] - P[1, :])
    
    if len(A_rows) < 4:  # Need at least 2 views (4 equations)
        return None, float('inf')
    
    A = np.array(A_rows)
    
    # Solve via SVD: X is the right singular vector with smallest singular value
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]  # Last row of V^T
    
    # Convert from homogeneous: [x, y, z, w] -> [x/w, y/w, z/w]
    if abs(X_homogeneous[3]) < 1e-10:
        return None, float('inf')
    
    point_3d = X_homogeneous[:3] / X_homogeneous[3]
    
    # Compute reprojection error
    errors = []
    for camera_id in common_cameras:
        P = projection_matrices[camera_id]
        p2d = np.atleast_1d(points_2d[camera_id]).flatten()
        
        if len(p2d) < 2:
            continue
            
        projected = P @ np.append(point_3d, 1)
        projected = projected[:2] / projected[2]
        
        error = np.linalg.norm(projected - p2d[:2])
        errors.append(error)
    
    mean_error = np.mean(errors) if errors else float('inf')
    
    return point_3d, mean_error


def triangulate_points_ransac(
    points_2d: Dict[str, np.ndarray],
    projection_matrices: Dict[str, np.ndarray],
    confidence_scores: Optional[Dict[str, float]] = None,
    ransac_iterations: int = 20,
    inlier_threshold: float = 10.0,  # pixels
) -> Tuple[Optional[np.ndarray], float, Dict[str, bool]]:
    """
    Robust triangulation using RANSAC to handle outlier detections.
    
    Pose detectors sometimes produce incorrect 2D detections (e.g., detecting
    someone else's body part, or hallucinating a joint). RANSAC finds the
    subset of views that agree on a consistent 3D point.
    
    Args:
        points_2d: camera_id -> [x, y] detections (1D or 2D arrays)
        projection_matrices: camera_id -> 3x4 projection matrix
        confidence_scores: camera_id -> detection confidence (0-1)
        ransac_iterations: Number of RANSAC iterations
        inlier_threshold: Max reprojection error to be considered inlier
        
    Returns:
        point_3d: Best 3D point estimate
        reprojection_error: Error of best estimate
        inliers: camera_id -> bool indicating which views were inliers
    """
    camera_ids = list(set(points_2d.keys()) & set(projection_matrices.keys()))
    
    if len(camera_ids) < 2:
        return None, float('inf'), {}
    
    best_point = None
    best_error = float('inf')
    best_inliers = {}
    best_inlier_count = 0
    
    # Weight sampling by confidence if available
    if confidence_scores:
        weights = np.array([confidence_scores.get(cid, 0.5) for cid in camera_ids])
        weights = weights / weights.sum()
    else:
        weights = None
    
    for _ in range(ransac_iterations):
        # Sample 2 cameras (minimum for triangulation)
        if weights is not None:
            sample_indices = np.random.choice(
                len(camera_ids), size=2, replace=False, p=weights
            )
        else:
            sample_indices = np.random.choice(len(camera_ids), size=2, replace=False)
        
        sample_cameras = [camera_ids[i] for i in sample_indices]
        
        # Triangulate from sample
        sample_points = {cid: points_2d[cid] for cid in sample_cameras}
        sample_P = {cid: projection_matrices[cid] for cid in sample_cameras}
        
        point_3d, _ = triangulate_point(sample_points, sample_P)
        
        if point_3d is None:
            continue
        
        # Count inliers (all cameras with small reprojection error)
        inliers = {}
        errors = []
        
        for cid in camera_ids:
            P = projection_matrices[cid]
            p2d = np.atleast_1d(points_2d[cid]).flatten()
            
            if len(p2d) < 2:
                inliers[cid] = False
                continue
            
            projected = P @ np.append(point_3d, 1)
            projected = projected[:2] / projected[2]
            
            error = np.linalg.norm(projected - p2d[:2])
            
            if error < inlier_threshold:
                inliers[cid] = True
                errors.append(error)
            else:
                inliers[cid] = False
        
        inlier_count = sum(inliers.values())
        
        if inlier_count > best_inlier_count or (
            inlier_count == best_inlier_count and 
            len(errors) > 0 and np.mean(errors) < best_error
        ):
            best_inlier_count = inlier_count
            best_inliers = inliers
            
            # Refine with all inliers
            inlier_cameras = [cid for cid, is_inlier in inliers.items() if is_inlier]
            if len(inlier_cameras) >= 2:
                refined_points = {cid: points_2d[cid] for cid in inlier_cameras}
                refined_P = {cid: projection_matrices[cid] for cid in inlier_cameras}
                best_point, best_error = triangulate_point(refined_points, refined_P)
    
    return best_point, best_error, best_inliers

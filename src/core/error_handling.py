"""
Dynamical.ai Error Handling and Graceful Degradation

This module provides comprehensive error handling with graceful degradation
for all system components. When components fail, the system continues
operating in a degraded but safe mode.

Error Handling Philosophy:
=========================
1. Safety NEVER degrades - if safety fails, stop immediately
2. Control can degrade to "hold position" mode
3. Perception can degrade to "use last known state"
4. Learning can degrade to "skip update"

Fallback Hierarchy:
==================
Each component has a defined fallback chain:

Perception Pipeline:
    DINOv3 fails    → Use cached features / zero features
    SAM3 fails      → Use bounding boxes without masks
    Depth fails     → Use default depth (2.0m) / last known depth
    Pose fails      → Use last known pose / zero pose
    Camera fails    → Use last frame / synthetic frame

Control Pipeline:
    VLA fails       → Use action buffer / hold position
    Skill fails     → Use fallback skill / safe stop
    Retargeting fails → Use identity mapping / freeze pose
    IK fails        → Use last solution / joint limits

Communication:
    Cloud fails     → Queue updates locally
    Edge fails      → Use cached model
    Glove fails     → Use last known hand state

Safety (NO fallback - immediate stop):
    E-stop triggered → Stop all motion
    Limit exceeded   → Stop all motion
    Watchdog timeout → Stop all motion
"""

import time
import logging
import threading
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, TypeVar, Generic
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Error States
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = auto()      # Informational, no action needed
    WARNING = auto()   # Degraded performance, continue with fallback
    ERROR = auto()     # Component failed, use fallback
    CRITICAL = auto()  # System-level failure, safe stop required


class ComponentState(Enum):
    """Component operational state."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ErrorEvent:
    """Record of an error event."""
    timestamp: float
    component: str
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Fallback Value Cache
# =============================================================================

class FallbackCache(Generic[T]):
    """
    Cache for fallback values with expiration.

    Stores the last N good values and provides the most recent
    when the primary source fails.
    """

    def __init__(
        self,
        max_age_s: float = 5.0,
        max_items: int = 10,
        default_value: Optional[T] = None,
    ):
        self.max_age_s = max_age_s
        self.max_items = max_items
        self.default_value = default_value

        self._values: deque = deque(maxlen=max_items)
        self._timestamps: deque = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def update(self, value: T):
        """Add a new good value to the cache."""
        with self._lock:
            self._values.append(value)
            self._timestamps.append(time.time())

    def get(self) -> Optional[T]:
        """Get the most recent valid cached value."""
        with self._lock:
            now = time.time()

            # Find most recent non-expired value
            for i in range(len(self._values) - 1, -1, -1):
                age = now - self._timestamps[i]
                if age <= self.max_age_s:
                    return self._values[i]

            # All expired, return default
            return self.default_value

    def get_or_default(self, default: T) -> T:
        """Get cached value or provided default."""
        value = self.get()
        return value if value is not None else default

    @property
    def age_s(self) -> float:
        """Age of most recent cached value in seconds."""
        with self._lock:
            if not self._timestamps:
                return float('inf')
            return time.time() - self._timestamps[-1]

    @property
    def is_valid(self) -> bool:
        """Check if cache has a valid (non-expired) value."""
        return self.age_s <= self.max_age_s


# =============================================================================
# Component Error Tracker
# =============================================================================

class ErrorTracker:
    """
    Tracks errors for a component and determines state.

    Uses a sliding window to track error rate and determine
    if component should be marked as failed.
    """

    def __init__(
        self,
        component_name: str,
        error_threshold: int = 5,
        window_s: float = 60.0,
        recovery_time_s: float = 10.0,
    ):
        self.component_name = component_name
        self.error_threshold = error_threshold
        self.window_s = window_s
        self.recovery_time_s = recovery_time_s

        self._errors: List[ErrorEvent] = []
        self._state = ComponentState.HEALTHY
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

        # Callbacks
        self._on_state_change: List[Callable[[ComponentState], None]] = []

    def record_error(
        self,
        severity: ErrorSeverity,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict] = None,
    ):
        """Record an error event."""
        event = ErrorEvent(
            timestamp=time.time(),
            component=self.component_name,
            severity=severity,
            message=message,
            exception=exception,
            context=context or {},
        )

        with self._lock:
            self._errors.append(event)
            self._prune_old_errors()
            self._update_state()

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"[{self.component_name}] {message}")
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"[{self.component_name}] {message}")
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"[{self.component_name}] {message}")
        else:
            logger.info(f"[{self.component_name}] {message}")

    def record_success(self):
        """Record a successful operation (for recovery tracking)."""
        with self._lock:
            if self._state == ComponentState.RECOVERING:
                # Check if enough time has passed for recovery
                if time.time() - self._last_failure_time > self.recovery_time_s:
                    self._set_state(ComponentState.HEALTHY)

    def _prune_old_errors(self):
        """Remove errors outside the sliding window."""
        cutoff = time.time() - self.window_s
        self._errors = [e for e in self._errors if e.timestamp > cutoff]

    def _update_state(self):
        """Update component state based on error count."""
        error_count = len([e for e in self._errors
                          if e.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]])

        old_state = self._state

        if error_count >= self.error_threshold:
            self._set_state(ComponentState.FAILED)
            self._last_failure_time = time.time()
        elif error_count > 0:
            self._set_state(ComponentState.DEGRADED)
        elif old_state == ComponentState.FAILED:
            self._set_state(ComponentState.RECOVERING)

    def _set_state(self, new_state: ComponentState):
        """Set state and notify callbacks."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.info(f"[{self.component_name}] State: {old_state.value} -> {new_state.value}")

            for callback in self._on_state_change:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.error(f"State change callback failed: {e}")

    def on_state_change(self, callback: Callable[[ComponentState], None]):
        """Register state change callback."""
        self._on_state_change.append(callback)

    @property
    def state(self) -> ComponentState:
        """Get current component state."""
        return self._state

    @property
    def error_count(self) -> int:
        """Get error count in current window."""
        with self._lock:
            return len(self._errors)

    @property
    def recent_errors(self) -> List[ErrorEvent]:
        """Get recent error events."""
        with self._lock:
            return list(self._errors)


# =============================================================================
# Graceful Degradation Decorators
# =============================================================================

def with_fallback(fallback_value: Any = None, cache: Optional[FallbackCache] = None):
    """
    Decorator that provides fallback on exception.

    Args:
        fallback_value: Value to return on failure
        cache: FallbackCache to use for dynamic fallback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if cache is not None:
                    cache.update(result)
                return result
            except Exception as e:
                logger.warning(f"{func.__name__} failed, using fallback: {e}")
                if cache is not None:
                    cached = cache.get()
                    if cached is not None:
                        return cached
                return fallback_value
        return wrapper
    return decorator


def with_retry(max_retries: int = 3, delay_s: float = 0.1, backoff: float = 2.0):
    """
    Decorator that retries on exception with exponential backoff.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay_s

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.debug(f"{func.__name__} attempt {attempt + 1} failed, retrying...")
                        time.sleep(current_delay)
                        current_delay *= backoff

            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            raise last_exception
        return wrapper
    return decorator


def with_timeout(timeout_s: float):
    """
    Decorator that enforces a timeout on function execution.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    logger.error(f"{func.__name__} timed out after {timeout_s}s")
                    raise TimeoutError(f"{func.__name__} timed out")
        return wrapper
    return decorator


# =============================================================================
# Component-Specific Fallback Handlers
# =============================================================================

class PerceptionFallback:
    """Fallback handlers for perception pipeline."""

    def __init__(self):
        # Caches for each perception component
        self.pose_cache = FallbackCache[np.ndarray](
            max_age_s=1.0,
            default_value=np.zeros((133, 3))
        )
        self.depth_cache = FallbackCache[np.ndarray](
            max_age_s=2.0,
            default_value=None
        )
        self.features_cache = FallbackCache[np.ndarray](
            max_age_s=5.0,
            default_value=np.zeros(768)
        )
        self.frame_cache = FallbackCache[np.ndarray](
            max_age_s=0.5,
            default_value=None
        )

        # Error trackers
        self.pose_tracker = ErrorTracker("pose_estimation")
        self.depth_tracker = ErrorTracker("depth_estimation")
        self.features_tracker = ErrorTracker("feature_extraction")
        self.camera_tracker = ErrorTracker("camera")

    def get_pose_fallback(self) -> np.ndarray:
        """Get fallback pose when estimation fails."""
        cached = self.pose_cache.get()
        if cached is not None:
            return cached
        # Return zero pose as last resort
        return np.zeros((133, 3))

    def get_depth_fallback(self, shape: tuple = (480, 640)) -> np.ndarray:
        """Get fallback depth map."""
        cached = self.depth_cache.get()
        if cached is not None:
            return cached
        # Return default 2m depth
        return np.ones(shape) * 2.0

    def get_features_fallback(self) -> np.ndarray:
        """Get fallback visual features."""
        return self.features_cache.get_or_default(np.zeros(768))


class ControlFallback:
    """Fallback handlers for control pipeline."""

    def __init__(self, n_joints: int = 7):
        self.n_joints = n_joints

        # Caches
        self.action_cache = FallbackCache[np.ndarray](
            max_age_s=0.5,
            default_value=np.zeros(n_joints)
        )
        self.joint_pos_cache = FallbackCache[np.ndarray](
            max_age_s=1.0,
            default_value=np.zeros(n_joints)
        )

        # Error trackers
        self.vla_tracker = ErrorTracker("vla_inference")
        self.skill_tracker = ErrorTracker("skill_execution")
        self.retarget_tracker = ErrorTracker("retargeting")

        # Safe stop positions
        self.safe_position: Optional[np.ndarray] = None

    def set_safe_position(self, position: np.ndarray):
        """Set the safe stop position."""
        self.safe_position = position.copy()

    def get_action_fallback(self, current_position: np.ndarray) -> np.ndarray:
        """Get fallback action - hold current position."""
        # First try action cache
        cached = self.action_cache.get()
        if cached is not None:
            return cached

        # Otherwise hold current position
        return current_position.copy()

    def get_safe_stop_action(self, current_position: np.ndarray) -> np.ndarray:
        """Get safe stop action - move slowly to safe position."""
        if self.safe_position is not None:
            # Blend toward safe position
            alpha = 0.1  # Slow blend
            return alpha * self.safe_position + (1 - alpha) * current_position
        return current_position.copy()


class CommunicationFallback:
    """Fallback handlers for communication failures."""

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size

        # Offline queues
        self._gradient_queue: deque = deque(maxlen=max_queue_size)
        self._update_queue: deque = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()

        # Error trackers
        self.cloud_tracker = ErrorTracker("cloud_connection")
        self.edge_tracker = ErrorTracker("edge_connection")

        # Cached models
        self._cached_model_version: Optional[str] = None
        self._cached_model_data: Optional[bytes] = None

    def queue_gradient(self, gradient: np.ndarray):
        """Queue gradient for later upload."""
        with self._lock:
            self._gradient_queue.append({
                'gradient': gradient.copy(),
                'timestamp': time.time(),
            })
        logger.debug(f"Gradient queued (queue size: {len(self._gradient_queue)})")

    def get_queued_gradients(self) -> List[Dict]:
        """Get all queued gradients for batch upload."""
        with self._lock:
            gradients = list(self._gradient_queue)
            self._gradient_queue.clear()
            return gradients

    def cache_model(self, version: str, data: bytes):
        """Cache model for offline use."""
        self._cached_model_version = version
        self._cached_model_data = data
        logger.info(f"Model cached: version={version}, size={len(data)} bytes")

    def get_cached_model(self) -> Optional[bytes]:
        """Get cached model."""
        return self._cached_model_data

    @property
    def queued_count(self) -> int:
        """Get number of queued items."""
        with self._lock:
            return len(self._gradient_queue)


# =============================================================================
# System-Wide Error Manager
# =============================================================================

class SystemErrorManager:
    """
    Central error manager for the entire system.

    Coordinates error handling across all components and manages
    system-wide degradation state.
    """

    def __init__(self):
        # Component handlers
        self.perception = PerceptionFallback()
        self.control = ControlFallback()
        self.communication = CommunicationFallback()

        # All component trackers
        self._trackers: Dict[str, ErrorTracker] = {}

        # Safety callback
        self._safety_callback: Optional[Callable[[], None]] = None

        # System state
        self._system_state = ComponentState.HEALTHY
        self._lock = threading.Lock()

        # Error history
        self._error_history: deque = deque(maxlen=1000)

    def register_tracker(self, name: str, tracker: ErrorTracker):
        """Register a component error tracker."""
        self._trackers[name] = tracker
        tracker.on_state_change(lambda state: self._on_component_state_change(name, state))

    def set_safety_callback(self, callback: Callable[[], None]):
        """Set callback for safety-critical failures."""
        self._safety_callback = callback

    def _on_component_state_change(self, component: str, state: ComponentState):
        """Handle component state change."""
        if state == ComponentState.FAILED:
            # Check if this is a safety-critical component
            if component in ['safety', 'estop', 'watchdog']:
                self._trigger_safety_stop(f"Safety component {component} failed")

        self._update_system_state()

    def _update_system_state(self):
        """Update overall system state."""
        with self._lock:
            failed_count = sum(1 for t in self._trackers.values()
                              if t.state == ComponentState.FAILED)
            degraded_count = sum(1 for t in self._trackers.values()
                                if t.state == ComponentState.DEGRADED)

            if failed_count > 0:
                self._system_state = ComponentState.FAILED
            elif degraded_count > 0:
                self._system_state = ComponentState.DEGRADED
            else:
                self._system_state = ComponentState.HEALTHY

    def _trigger_safety_stop(self, reason: str):
        """Trigger safety stop."""
        logger.critical(f"SAFETY STOP: {reason}")
        if self._safety_callback:
            try:
                self._safety_callback()
            except Exception as e:
                logger.critical(f"Safety callback failed: {e}")

    def record_error(
        self,
        component: str,
        severity: ErrorSeverity,
        message: str,
        exception: Optional[Exception] = None,
    ):
        """Record an error event."""
        event = ErrorEvent(
            timestamp=time.time(),
            component=component,
            severity=severity,
            message=message,
            exception=exception,
        )
        self._error_history.append(event)

        # Route to appropriate tracker
        if component in self._trackers:
            self._trackers[component].record_error(
                severity, message, exception
            )

        # Handle critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._trigger_safety_stop(message)

    def get_status(self) -> Dict[str, Any]:
        """Get system error status."""
        return {
            'system_state': self._system_state.value,
            'components': {
                name: {
                    'state': tracker.state.value,
                    'error_count': tracker.error_count,
                }
                for name, tracker in self._trackers.items()
            },
            'recent_errors': [
                {
                    'timestamp': e.timestamp,
                    'component': e.component,
                    'severity': e.severity.name,
                    'message': e.message,
                }
                for e in list(self._error_history)[-10:]
            ],
            'queued_gradients': self.communication.queued_count,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if system is fully healthy."""
        return self._system_state == ComponentState.HEALTHY

    @property
    def is_operational(self) -> bool:
        """Check if system can still operate (healthy or degraded)."""
        return self._system_state in [ComponentState.HEALTHY, ComponentState.DEGRADED]


# =============================================================================
# Global Error Manager Instance
# =============================================================================

_error_manager: Optional[SystemErrorManager] = None


def get_error_manager() -> SystemErrorManager:
    """Get the global error manager instance."""
    global _error_manager
    if _error_manager is None:
        _error_manager = SystemErrorManager()
    return _error_manager


def init_error_manager() -> SystemErrorManager:
    """Initialize the global error manager."""
    global _error_manager
    _error_manager = SystemErrorManager()
    return _error_manager


# =============================================================================
# Utility Functions
# =============================================================================

def safe_call(
    func: Callable[..., T],
    *args,
    fallback: T = None,
    component: str = "unknown",
    **kwargs
) -> T:
    """
    Safely call a function with error handling and fallback.

    Args:
        func: Function to call
        *args: Positional arguments
        fallback: Value to return on failure
        component: Component name for error tracking
        **kwargs: Keyword arguments

    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        manager = get_error_manager()
        manager.record_error(
            component=component,
            severity=ErrorSeverity.ERROR,
            message=f"{func.__name__} failed: {str(e)}",
            exception=e,
        )
        return fallback


def require_healthy(component: str):
    """
    Decorator that requires a component to be healthy.

    Raises exception if component is in failed state.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_error_manager()
            if component in manager._trackers:
                state = manager._trackers[component].state
                if state == ComponentState.FAILED:
                    raise RuntimeError(f"Component {component} is in failed state")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Initialize error manager
    manager = init_error_manager()

    # Register component trackers
    manager.register_tracker("perception", manager.perception.pose_tracker)
    manager.register_tracker("control", manager.control.vla_tracker)
    manager.register_tracker("communication", manager.communication.cloud_tracker)

    # Set safety callback
    def safety_stop():
        print("EMERGENCY STOP TRIGGERED!")

    manager.set_safety_callback(safety_stop)

    # Simulate some errors
    print("Simulating perception errors...")
    for i in range(3):
        manager.perception.pose_tracker.record_error(
            ErrorSeverity.ERROR,
            f"Pose estimation failed: attempt {i+1}"
        )
        time.sleep(0.1)

    # Check status
    status = manager.get_status()
    print(f"\nSystem Status: {status['system_state']}")
    for name, comp in status['components'].items():
        print(f"  {name}: {comp['state']} ({comp['error_count']} errors)")

    # Test fallback decorator
    @with_fallback(fallback_value=np.zeros(10))
    def failing_function():
        raise ValueError("Simulated failure")

    result = failing_function()
    print(f"\nFallback result shape: {result.shape}")

    print("\nError handling system initialized successfully!")

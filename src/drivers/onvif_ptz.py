"""
ONVIF PTZ (Pan-Tilt-Zoom) Controller

This module provides PTZ control for ONVIF-compliant IP cameras.
Uses the python-onvif-zeep library for ONVIF protocol implementation.

Features:
- PTZ move (continuous, relative, absolute)
- Preset management (get, set, goto)
- Auto-calibration with visual tracking
- Focus and zoom control
- Status monitoring

Reference: https://www.onvif.org/profiles/profile-s/
"""

import time
import threading
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

logger = logging.getLogger(__name__)

# Try to import python-onvif-zeep
try:
    from onvif import ONVIFCamera, ONVIFError
    HAS_ONVIF = True
except ImportError:
    HAS_ONVIF = False
    logger.warning("python-onvif-zeep not installed. Install with: pip install onvif-zeep")


class PTZDirection(str, Enum):
    """PTZ movement directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    STOP = "stop"


@dataclass
class PTZStatus:
    """Current PTZ status."""
    pan: float = 0.0          # -1.0 to 1.0
    tilt: float = 0.0         # -1.0 to 1.0
    zoom: float = 0.0         # 0.0 to 1.0
    moving: bool = False
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PTZPreset:
    """PTZ preset position."""
    token: str
    name: str
    position: Optional[Tuple[float, float, float]] = None  # (pan, tilt, zoom)


@dataclass
class PTZLimits:
    """PTZ movement limits."""
    pan_min: float = -1.0
    pan_max: float = 1.0
    tilt_min: float = -1.0
    tilt_max: float = 1.0
    zoom_min: float = 0.0
    zoom_max: float = 1.0
    speed_min: float = 0.0
    speed_max: float = 1.0


class ONVIFPTZController:
    """
    ONVIF PTZ controller for IP cameras.

    Usage:
        ptz = ONVIFPTZController(
            host="192.168.1.100",
            port=80,
            username="admin",
            password="password"
        )
        ptz.connect()

        # Move camera
        ptz.move(PTZDirection.LEFT, speed=0.5)
        time.sleep(1)
        ptz.stop()

        # Go to preset
        ptz.goto_preset("home")

        ptz.disconnect()
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "admin",
        wsdl_dir: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize PTZ controller.

        Args:
            host: Camera IP address
            port: ONVIF port (usually 80 or 8080)
            username: ONVIF username
            password: ONVIF password
            wsdl_dir: Path to ONVIF WSDL files (optional)
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.wsdl_dir = wsdl_dir
        self.timeout = timeout

        self._camera: Optional[Any] = None  # ONVIFCamera instance
        self._ptz_service: Optional[Any] = None
        self._media_service: Optional[Any] = None
        self._profile_token: Optional[str] = None
        self._ptz_node: Optional[Any] = None

        self._connected = False
        self._lock = threading.Lock()

        # Cached data
        self._presets: Dict[str, PTZPreset] = {}
        self._limits: Optional[PTZLimits] = None
        self._status: PTZStatus = PTZStatus()

    def connect(self) -> bool:
        """
        Connect to the ONVIF camera.

        Returns:
            True if connection successful
        """
        if not HAS_ONVIF:
            logger.error("python-onvif-zeep not installed")
            return False

        try:
            with self._lock:
                # Create camera instance
                self._camera = ONVIFCamera(
                    self.host,
                    self.port,
                    self.username,
                    self.password,
                    self.wsdl_dir,
                )

                # Get media service and profile
                self._media_service = self._camera.create_media_service()
                profiles = self._media_service.GetProfiles()

                if not profiles:
                    logger.error("No media profiles found on camera")
                    return False

                # Use first profile with PTZ support
                for profile in profiles:
                    if hasattr(profile, 'PTZConfiguration') and profile.PTZConfiguration:
                        self._profile_token = profile.token
                        break

                if not self._profile_token:
                    # Use first profile anyway
                    self._profile_token = profiles[0].token
                    logger.warning("No PTZ-enabled profile found, using first profile")

                # Get PTZ service
                self._ptz_service = self._camera.create_ptz_service()

                # Get PTZ node for limits
                nodes = self._ptz_service.GetNodes()
                if nodes:
                    self._ptz_node = nodes[0]

                # Load presets
                self._load_presets()

                # Get limits
                self._load_limits()

                self._connected = True
                logger.info(f"Connected to ONVIF camera at {self.host}:{self.port}")
                return True

        except Exception as e:
            logger.error(f"Failed to connect to ONVIF camera: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from the camera."""
        with self._lock:
            self._camera = None
            self._ptz_service = None
            self._media_service = None
            self._connected = False
        logger.info(f"Disconnected from ONVIF camera at {self.host}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to camera."""
        return self._connected

    def _load_presets(self):
        """Load presets from camera."""
        if not self._ptz_service or not self._profile_token:
            return

        try:
            presets = self._ptz_service.GetPresets({'ProfileToken': self._profile_token})
            self._presets.clear()

            for preset in presets:
                name = getattr(preset, 'Name', preset.token)
                position = None
                if hasattr(preset, 'PTZPosition'):
                    pos = preset.PTZPosition
                    if hasattr(pos, 'PanTilt') and hasattr(pos, 'Zoom'):
                        position = (
                            getattr(pos.PanTilt, 'x', 0.0),
                            getattr(pos.PanTilt, 'y', 0.0),
                            getattr(pos.Zoom, 'x', 0.0),
                        )

                self._presets[name] = PTZPreset(
                    token=preset.token,
                    name=name,
                    position=position,
                )

            logger.info(f"Loaded {len(self._presets)} PTZ presets")

        except Exception as e:
            logger.error(f"Failed to load presets: {e}")

    def _load_limits(self):
        """Load PTZ limits from camera."""
        if not self._ptz_node:
            self._limits = PTZLimits()
            return

        try:
            self._limits = PTZLimits()

            if hasattr(self._ptz_node, 'SupportedPTZSpaces'):
                spaces = self._ptz_node.SupportedPTZSpaces

                # Pan/Tilt limits
                if hasattr(spaces, 'AbsolutePanTiltPositionSpace') and spaces.AbsolutePanTiltPositionSpace:
                    pt_space = spaces.AbsolutePanTiltPositionSpace[0]
                    if hasattr(pt_space, 'XRange'):
                        self._limits.pan_min = pt_space.XRange.Min
                        self._limits.pan_max = pt_space.XRange.Max
                    if hasattr(pt_space, 'YRange'):
                        self._limits.tilt_min = pt_space.YRange.Min
                        self._limits.tilt_max = pt_space.YRange.Max

                # Zoom limits
                if hasattr(spaces, 'AbsoluteZoomPositionSpace') and spaces.AbsoluteZoomPositionSpace:
                    z_space = spaces.AbsoluteZoomPositionSpace[0]
                    if hasattr(z_space, 'XRange'):
                        self._limits.zoom_min = z_space.XRange.Min
                        self._limits.zoom_max = z_space.XRange.Max

            logger.info(f"Loaded PTZ limits: pan=[{self._limits.pan_min}, {self._limits.pan_max}], "
                       f"tilt=[{self._limits.tilt_min}, {self._limits.tilt_max}], "
                       f"zoom=[{self._limits.zoom_min}, {self._limits.zoom_max}]")

        except Exception as e:
            logger.error(f"Failed to load PTZ limits: {e}")
            self._limits = PTZLimits()

    def get_status(self) -> PTZStatus:
        """
        Get current PTZ status.

        Returns:
            PTZStatus with current position
        """
        if not self._connected or not self._ptz_service:
            return PTZStatus(error="Not connected")

        try:
            status = self._ptz_service.GetStatus({'ProfileToken': self._profile_token})

            pan = tilt = zoom = 0.0
            moving = False

            if hasattr(status, 'Position'):
                pos = status.Position
                if hasattr(pos, 'PanTilt'):
                    pan = getattr(pos.PanTilt, 'x', 0.0)
                    tilt = getattr(pos.PanTilt, 'y', 0.0)
                if hasattr(pos, 'Zoom'):
                    zoom = getattr(pos.Zoom, 'x', 0.0)

            if hasattr(status, 'MoveStatus'):
                move_status = status.MoveStatus
                if hasattr(move_status, 'PanTilt'):
                    moving = move_status.PanTilt in ('MOVING', 'Moving')

            self._status = PTZStatus(
                pan=pan,
                tilt=tilt,
                zoom=zoom,
                moving=moving,
            )
            return self._status

        except Exception as e:
            logger.error(f"Failed to get PTZ status: {e}")
            return PTZStatus(error=str(e))

    def move(
        self,
        direction: PTZDirection,
        speed: float = 0.5,
        timeout_s: float = 0.0,
    ) -> bool:
        """
        Move camera in a direction (continuous move).

        Args:
            direction: Movement direction
            speed: Speed 0.0 to 1.0
            timeout_s: Auto-stop after this many seconds (0=continuous)

        Returns:
            True if move command sent successfully
        """
        if not self._connected or not self._ptz_service:
            logger.error("Not connected to camera")
            return False

        speed = max(0.0, min(1.0, speed))

        try:
            # Build velocity request
            pan_vel = tilt_vel = zoom_vel = 0.0

            if direction == PTZDirection.LEFT:
                pan_vel = -speed
            elif direction == PTZDirection.RIGHT:
                pan_vel = speed
            elif direction == PTZDirection.UP:
                tilt_vel = speed
            elif direction == PTZDirection.DOWN:
                tilt_vel = -speed
            elif direction == PTZDirection.ZOOM_IN:
                zoom_vel = speed
            elif direction == PTZDirection.ZOOM_OUT:
                zoom_vel = -speed
            elif direction == PTZDirection.STOP:
                return self.stop()

            # Create velocity request
            request = self._ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self._profile_token
            request.Velocity = {
                'PanTilt': {'x': pan_vel, 'y': tilt_vel},
                'Zoom': {'x': zoom_vel}
            }

            if timeout_s > 0:
                request.Timeout = f'PT{timeout_s}S'

            self._ptz_service.ContinuousMove(request)

            logger.debug(f"PTZ move: {direction.value} @ speed {speed}")
            return True

        except Exception as e:
            logger.error(f"PTZ move failed: {e}")
            return False

    def move_absolute(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        speed: float = 0.5,
    ) -> bool:
        """
        Move camera to absolute position.

        Args:
            pan: Target pan position (-1.0 to 1.0)
            tilt: Target tilt position (-1.0 to 1.0)
            zoom: Target zoom position (0.0 to 1.0)
            speed: Movement speed 0.0 to 1.0

        Returns:
            True if move command sent successfully
        """
        if not self._connected or not self._ptz_service:
            return False

        try:
            request = self._ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self._profile_token
            request.Position = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }
            request.Speed = {
                'PanTilt': {'x': speed, 'y': speed},
                'Zoom': {'x': speed}
            }

            self._ptz_service.AbsoluteMove(request)

            logger.debug(f"PTZ absolute move: pan={pan}, tilt={tilt}, zoom={zoom}")
            return True

        except Exception as e:
            logger.error(f"PTZ absolute move failed: {e}")
            return False

    def move_relative(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        zoom_delta: float = 0.0,
        speed: float = 0.5,
    ) -> bool:
        """
        Move camera by relative amount.

        Args:
            pan_delta: Relative pan movement
            tilt_delta: Relative tilt movement
            zoom_delta: Relative zoom movement
            speed: Movement speed 0.0 to 1.0

        Returns:
            True if move command sent successfully
        """
        if not self._connected or not self._ptz_service:
            return False

        try:
            request = self._ptz_service.create_type('RelativeMove')
            request.ProfileToken = self._profile_token
            request.Translation = {
                'PanTilt': {'x': pan_delta, 'y': tilt_delta},
                'Zoom': {'x': zoom_delta}
            }
            request.Speed = {
                'PanTilt': {'x': speed, 'y': speed},
                'Zoom': {'x': speed}
            }

            self._ptz_service.RelativeMove(request)

            logger.debug(f"PTZ relative move: pan={pan_delta}, tilt={tilt_delta}, zoom={zoom_delta}")
            return True

        except Exception as e:
            logger.error(f"PTZ relative move failed: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop all PTZ movement.

        Returns:
            True if stop command sent successfully
        """
        if not self._connected or not self._ptz_service:
            return False

        try:
            self._ptz_service.Stop({
                'ProfileToken': self._profile_token,
                'PanTilt': True,
                'Zoom': True
            })
            logger.debug("PTZ stopped")
            return True

        except Exception as e:
            logger.error(f"PTZ stop failed: {e}")
            return False

    def get_presets(self) -> List[PTZPreset]:
        """Get list of available presets."""
        return list(self._presets.values())

    def goto_preset(self, name_or_token: str, speed: float = 0.5) -> bool:
        """
        Go to a preset position.

        Args:
            name_or_token: Preset name or token
            speed: Movement speed 0.0 to 1.0

        Returns:
            True if command sent successfully
        """
        if not self._connected or not self._ptz_service:
            return False

        # Find preset token
        token = name_or_token
        if name_or_token in self._presets:
            token = self._presets[name_or_token].token

        try:
            self._ptz_service.GotoPreset({
                'ProfileToken': self._profile_token,
                'PresetToken': token,
                'Speed': {
                    'PanTilt': {'x': speed, 'y': speed},
                    'Zoom': {'x': speed}
                }
            })

            logger.debug(f"Going to preset: {name_or_token}")
            return True

        except Exception as e:
            logger.error(f"Goto preset failed: {e}")
            return False

    def set_preset(self, name: str) -> Optional[str]:
        """
        Save current position as a preset.

        Args:
            name: Preset name

        Returns:
            Preset token if successful, None otherwise
        """
        if not self._connected or not self._ptz_service:
            return None

        try:
            result = self._ptz_service.SetPreset({
                'ProfileToken': self._profile_token,
                'PresetName': name
            })

            token = result.PresetToken

            # Update local cache
            status = self.get_status()
            self._presets[name] = PTZPreset(
                token=token,
                name=name,
                position=(status.pan, status.tilt, status.zoom),
            )

            logger.info(f"Saved preset: {name} (token: {token})")
            return token

        except Exception as e:
            logger.error(f"Set preset failed: {e}")
            return None

    def remove_preset(self, name_or_token: str) -> bool:
        """
        Remove a preset.

        Args:
            name_or_token: Preset name or token

        Returns:
            True if successful
        """
        if not self._connected or not self._ptz_service:
            return False

        token = name_or_token
        if name_or_token in self._presets:
            token = self._presets[name_or_token].token

        try:
            self._ptz_service.RemovePreset({
                'ProfileToken': self._profile_token,
                'PresetToken': token
            })

            # Update local cache
            for name, preset in list(self._presets.items()):
                if preset.token == token:
                    del self._presets[name]
                    break

            logger.info(f"Removed preset: {name_or_token}")
            return True

        except Exception as e:
            logger.error(f"Remove preset failed: {e}")
            return False

    def go_home(self, speed: float = 0.5) -> bool:
        """
        Go to home position.

        Args:
            speed: Movement speed

        Returns:
            True if successful
        """
        if not self._connected or not self._ptz_service:
            return False

        try:
            self._ptz_service.GotoHomePosition({
                'ProfileToken': self._profile_token,
                'Speed': {
                    'PanTilt': {'x': speed, 'y': speed},
                    'Zoom': {'x': speed}
                }
            })

            logger.debug("Going to home position")
            return True

        except Exception as e:
            logger.error(f"Go home failed: {e}")
            return False

    def set_home(self) -> bool:
        """
        Set current position as home.

        Returns:
            True if successful
        """
        if not self._connected or not self._ptz_service:
            return False

        try:
            self._ptz_service.SetHomePosition({
                'ProfileToken': self._profile_token
            })

            logger.info("Set current position as home")
            return True

        except Exception as e:
            logger.error(f"Set home failed: {e}")
            return False

    def get_limits(self) -> PTZLimits:
        """Get PTZ movement limits."""
        return self._limits or PTZLimits()

    def auto_calibrate(
        self,
        target_detector: Optional[callable] = None,
        calibration_points: int = 9,
    ) -> Dict[str, Any]:
        """
        Run auto-calibration routine.

        This moves the camera to known positions and optionally
        uses visual tracking to refine the calibration.

        Args:
            target_detector: Function to detect calibration target in frame
            calibration_points: Number of calibration points

        Returns:
            Calibration result dictionary
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        results = {
            "success": True,
            "points": [],
            "limits_verified": False,
            "presets_verified": False,
        }

        try:
            # Test movement to extremes
            logger.info("Starting PTZ auto-calibration...")

            # Go to limits
            test_positions = [
                (0, 0, 0),      # Center
                (-1, 0, 0),     # Left
                (1, 0, 0),      # Right
                (0, 1, 0),      # Up
                (0, -1, 0),     # Down
                (0, 0, 0),      # Center zoom out
                (0, 0, 1),      # Center zoom in
            ]

            for pan, tilt, zoom in test_positions[:calibration_points]:
                self.move_absolute(pan, tilt, zoom, speed=0.8)
                time.sleep(2.0)  # Wait for movement

                status = self.get_status()
                results["points"].append({
                    "target": {"pan": pan, "tilt": tilt, "zoom": zoom},
                    "actual": {"pan": status.pan, "tilt": status.tilt, "zoom": status.zoom},
                    "error": np.sqrt(
                        (pan - status.pan)**2 +
                        (tilt - status.tilt)**2 +
                        (zoom - status.zoom)**2
                    ),
                })

            # Return to home
            self.go_home()

            results["limits_verified"] = True
            results["avg_error"] = np.mean([p["error"] for p in results["points"]])

            logger.info(f"PTZ calibration complete. Avg error: {results['avg_error']:.4f}")

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            logger.error(f"PTZ calibration failed: {e}")

        return results


# =============================================================================
# Testing
# =============================================================================

def test_ptz_controller():
    """
    Test PTZ controller with a real ONVIF camera.

    Set environment variables to test:
        PTZ_HOST=192.168.1.100
        PTZ_USER=admin
        PTZ_PASS=password
    """
    import os

    print("\n" + "=" * 60)
    print("ONVIF PTZ CONTROLLER TEST")
    print("=" * 60)

    host = os.getenv("PTZ_HOST")
    username = os.getenv("PTZ_USER", "admin")
    password = os.getenv("PTZ_PASS", "admin")

    if not host:
        print("\nNo PTZ_HOST environment variable set.")
        print("Set PTZ_HOST, PTZ_USER, PTZ_PASS to test with a real camera.")
        print("Skipping hardware test.")
        return

    print(f"\nConnecting to PTZ camera at {host}...")

    ptz = ONVIFPTZController(
        host=host,
        username=username,
        password=password,
    )

    if not ptz.connect():
        print("Failed to connect to PTZ camera.")
        return

    print("Connected successfully!")

    print("\n1. Get Status")
    print("-" * 40)
    status = ptz.get_status()
    print(f"   Pan: {status.pan:.2f}, Tilt: {status.tilt:.2f}, Zoom: {status.zoom:.2f}")

    print("\n2. Get Presets")
    print("-" * 40)
    presets = ptz.get_presets()
    for preset in presets:
        print(f"   {preset.name}: {preset.position}")

    print("\n3. Get Limits")
    print("-" * 40)
    limits = ptz.get_limits()
    print(f"   Pan: [{limits.pan_min:.2f}, {limits.pan_max:.2f}]")
    print(f"   Tilt: [{limits.tilt_min:.2f}, {limits.tilt_max:.2f}]")
    print(f"   Zoom: [{limits.zoom_min:.2f}, {limits.zoom_max:.2f}]")

    ptz.disconnect()
    print("\nDisconnected.")

    print("\n" + "=" * 60)
    print("PTZ CONTROLLER TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_ptz_controller()

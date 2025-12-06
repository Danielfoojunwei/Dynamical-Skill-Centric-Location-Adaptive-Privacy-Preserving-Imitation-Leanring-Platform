"""
Dynamical.ai Edge Service

Microservice running on Jetson Orin for:
- Device discovery and management
- DOGlove haptic glove integration
- ONVIF camera integration
- Real-time status monitoring via WebSocket
- Platform connectivity and remote management
- Troubleshooting and diagnostics

Run with:
    python -m jetson.edge_service --config /etc/dynamical/edge.json
"""

import os
import sys
import json
import uuid
import asyncio
import socket
import struct
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
import xml.etree.ElementTree as ET

# Web framework
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Optional imports
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceListener, ServiceInfo
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False

try:
    import pyudev
    HAS_UDEV = True
except ImportError:
    HAS_UDEV = False

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import usb.core
    import usb.util
    HAS_USB = True
except ImportError:
    HAS_USB = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EdgeConfig:
    """Edge service configuration."""
    device_id: str = ""
    device_name: str = ""
    api_port: int = 8080
    websocket_port: int = 8081
    platform_url: str = ""
    api_key: str = ""
    data_dir: str = "/var/lib/dynamical"
    
    # Discovery settings
    discovery_enabled: bool = True
    discovery_interval_s: int = 30
    
    # DOGlove settings
    doglove_enabled: bool = True
    doglove_baud_rate: int = 115200
    
    # ONVIF settings
    onvif_enabled: bool = True
    onvif_discovery_timeout: float = 5.0
    
    def __post_init__(self):
        if not self.device_id:
            self.device_id = f"jetson_{uuid.uuid4().hex[:12]}"
        if not self.device_name:
            self.device_name = socket.gethostname()
    
    @classmethod
    def load(cls, path: str) -> "EdgeConfig":
        """Load configuration from file."""
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()
    
    def save(self, path: str):
        """Save configuration to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# =============================================================================
# Device Types
# =============================================================================

class DeviceType(str, Enum):
    """Connected device types."""
    DOGLOVE = "doglove"
    ONVIF_CAMERA = "onvif_camera"
    USB_CAMERA = "usb_camera"
    ROBOT_ARM = "robot_arm"
    GRIPPER = "gripper"
    SENSOR = "sensor"
    UNKNOWN = "unknown"


class DeviceStatus(str, Enum):
    """Device connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    STREAMING = "streaming"
    CALIBRATING = "calibrating"


class ConnectionType(str, Enum):
    """Device connection type."""
    USB = "usb"
    SERIAL = "serial"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"


@dataclass
class Device:
    """Connected device information."""
    id: str
    name: str
    type: DeviceType
    status: DeviceStatus
    connection_type: ConnectionType
    address: str  # IP, serial port, USB path, etc.
    
    # Optional fields
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    firmware_version: str = ""
    
    # Status info
    last_seen: str = ""
    error_message: str = ""
    
    # Device-specific config
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    
    # Real-time metrics
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DOGlove Driver
# =============================================================================

class DOGloveDriver:
    """
    Driver for DOGlove haptic glove.
    
    DOGlove features:
    - 21-DOF finger tracking
    - Force feedback on each finger
    - IMU for hand orientation
    - USB/Serial connection
    """
    
    # DOGlove USB identifiers
    VENDOR_ID = 0x1234  # Example - replace with actual
    PRODUCT_ID = 0x5678
    
    # Protocol constants
    CMD_START = 0xAA
    CMD_END = 0x55
    CMD_GET_DATA = 0x01
    CMD_SET_HAPTIC = 0x02
    CMD_CALIBRATE = 0x03
    CMD_GET_INFO = 0x04
    CMD_SET_CONFIG = 0x05
    
    # Finger indices
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4
    
    def __init__(self, port: str = None, baud_rate: int = 115200):
        self.port = port
        self.baud_rate = baud_rate
        self._serial: Optional[serial.Serial] = None
        self._connected = False
        self._streaming = False
        self._data_callback: Optional[Callable] = None
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Glove state
        self.finger_positions = [0.0] * 21  # 21 DOF
        self.finger_forces = [0.0] * 5  # Force per finger
        self.orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion
        self.accelerometer = [0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]
        
        # Device info
        self.device_info = {
            "manufacturer": "Dexta Robotics",
            "model": "DOGlove",
            "serial_number": "",
            "firmware_version": "",
            "hand": "right",  # or "left"
        }
    
    @classmethod
    def find_devices(cls) -> List[Dict[str, str]]:
        """Find connected DOGlove devices."""
        devices = []
        
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                # Check for DOGlove identifiers
                if port.vid == cls.VENDOR_ID and port.pid == cls.PRODUCT_ID:
                    devices.append({
                        "port": port.device,
                        "description": port.description,
                        "serial_number": port.serial_number or "",
                    })
                # Also check description for manual identification
                elif "doglove" in (port.description or "").lower():
                    devices.append({
                        "port": port.device,
                        "description": port.description,
                        "serial_number": port.serial_number or "",
                    })
        
        return devices
    
    def connect(self, port: str = None) -> bool:
        """Connect to DOGlove."""
        if port:
            self.port = port
        
        if not self.port:
            # Auto-detect
            devices = self.find_devices()
            if devices:
                self.port = devices[0]["port"]
            else:
                logger.error("No DOGlove device found")
                return False
        
        try:
            if HAS_SERIAL:
                self._serial = serial.Serial(
                    self.port,
                    self.baud_rate,
                    timeout=1.0
                )
                self._connected = True
                
                # Get device info
                self._request_info()
                
                logger.info(f"DOGlove connected on {self.port}")
                return True
            else:
                logger.error("pyserial not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to DOGlove: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from DOGlove."""
        self.stop_streaming()
        
        if self._serial:
            self._serial.close()
            self._serial = None
        
        self._connected = False
        logger.info("DOGlove disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._serial is not None
    
    def _send_command(self, cmd: int, data: bytes = b"") -> bool:
        """Send command to glove."""
        if not self._serial:
            return False
        
        try:
            # Build packet: START + CMD + LEN + DATA + CHECKSUM + END
            length = len(data)
            packet = bytes([self.CMD_START, cmd, length]) + data
            checksum = sum(packet) & 0xFF
            packet += bytes([checksum, self.CMD_END])
            
            self._serial.write(packet)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def _read_response(self, timeout: float = 1.0) -> Optional[bytes]:
        """Read response from glove."""
        if not self._serial:
            return None
        
        try:
            self._serial.timeout = timeout
            
            # Read until START byte
            while True:
                byte = self._serial.read(1)
                if not byte:
                    return None
                if byte[0] == self.CMD_START:
                    break
            
            # Read command and length
            header = self._serial.read(2)
            if len(header) < 2:
                return None
            
            cmd, length = header
            
            # Read data
            data = self._serial.read(length)
            if len(data) < length:
                return None
            
            # Read checksum and end
            footer = self._serial.read(2)
            if len(footer) < 2:
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to read response: {e}")
            return None
    
    def _request_info(self):
        """Request device information."""
        if self._send_command(self.CMD_GET_INFO):
            response = self._read_response()
            if response and len(response) >= 20:
                # Parse device info
                self.device_info["serial_number"] = response[0:8].decode('utf-8', errors='ignore').strip('\x00')
                self.device_info["firmware_version"] = f"{response[8]}.{response[9]}.{response[10]}"
                self.device_info["hand"] = "left" if response[11] == 0 else "right"
    
    def _parse_data_packet(self, data: bytes):
        """Parse incoming data packet."""
        if len(data) < 100:
            return
        
        # Parse finger positions (21 floats = 84 bytes)
        for i in range(21):
            offset = i * 4
            self.finger_positions[i] = struct.unpack('<f', data[offset:offset+4])[0]
        
        # Parse forces (5 floats = 20 bytes)
        offset = 84
        for i in range(5):
            self.finger_forces[i] = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
        
        # Parse orientation (4 floats = 16 bytes)
        for i in range(4):
            self.orientation[i] = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
        
        # Parse IMU (6 floats = 24 bytes)
        for i in range(3):
            self.accelerometer[i] = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
        for i in range(3):
            self.gyroscope[i] = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
    
    def start_streaming(self, callback: Callable[[Dict], None] = None):
        """Start streaming data from glove."""
        if not self._connected:
            logger.error("Not connected")
            return False
        
        self._data_callback = callback
        self._streaming = True
        self._stop_event.clear()
        
        self._read_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._read_thread.start()
        
        logger.info("DOGlove streaming started")
        return True
    
    def stop_streaming(self):
        """Stop streaming data."""
        self._streaming = False
        self._stop_event.set()
        
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
            self._read_thread = None
        
        logger.info("DOGlove streaming stopped")
    
    def _stream_loop(self):
        """Background thread for reading stream data."""
        while not self._stop_event.is_set():
            try:
                if self._send_command(self.CMD_GET_DATA):
                    response = self._read_response(timeout=0.1)
                    if response:
                        self._parse_data_packet(response)
                        
                        if self._data_callback:
                            self._data_callback(self.get_state())
                
                time.sleep(0.01)  # ~100 Hz
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(0.1)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current glove state."""
        return {
            "finger_positions": self.finger_positions.copy(),
            "finger_forces": self.finger_forces.copy(),
            "orientation": self.orientation.copy(),
            "accelerometer": self.accelerometer.copy(),
            "gyroscope": self.gyroscope.copy(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def set_haptic_feedback(self, finger: int, intensity: float):
        """
        Set haptic feedback for a finger.
        
        Args:
            finger: Finger index (0-4)
            intensity: Feedback intensity (0.0-1.0)
        """
        if not 0 <= finger <= 4:
            return False
        
        intensity = max(0.0, min(1.0, intensity))
        intensity_byte = int(intensity * 255)
        
        data = bytes([finger, intensity_byte])
        return self._send_command(self.CMD_SET_HAPTIC, data)
    
    def set_all_haptics(self, intensities: List[float]):
        """Set haptic feedback for all fingers."""
        if len(intensities) != 5:
            return False
        
        for i, intensity in enumerate(intensities):
            self.set_haptic_feedback(i, intensity)
        
        return True
    
    def calibrate(self) -> bool:
        """Run calibration routine."""
        logger.info("Starting DOGlove calibration...")
        
        if self._send_command(self.CMD_CALIBRATE):
            response = self._read_response(timeout=10.0)
            if response and response[0] == 0x01:
                logger.info("Calibration complete")
                return True
        
        logger.error("Calibration failed")
        return False
    
    def get_finger_angles(self) -> Dict[str, List[float]]:
        """Get finger angles in degrees."""
        # Convert raw positions to angles
        # Each finger has 4 DOF (MCP_flex, MCP_spread, PIP, DIP) except thumb (5 DOF)
        fingers = {
            "thumb": self.finger_positions[0:5],
            "index": self.finger_positions[5:9],
            "middle": self.finger_positions[9:13],
            "ring": self.finger_positions[13:17],
            "pinky": self.finger_positions[17:21],
        }
        return fingers


# =============================================================================
# ONVIF Camera Discovery
# =============================================================================

class ONVIFDiscovery:
    """
    ONVIF camera discovery and management.
    
    Uses WS-Discovery to find ONVIF-compliant cameras on the network.
    """
    
    # WS-Discovery multicast address
    MULTICAST_IP = "239.255.255.250"
    MULTICAST_PORT = 3702
    
    # ONVIF device service namespace
    ONVIF_NS = "http://www.onvif.org/ver10/network/wsdl"
    
    # WS-Discovery probe message
    PROBE_MESSAGE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
    <e:Header>
        <w:MessageID>uuid:{message_id}</w:MessageID>
        <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
        <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
    </e:Header>
    <e:Body>
        <d:Probe>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
        </d:Probe>
    </e:Body>
</e:Envelope>"""
    
    @dataclass
    class Camera:
        """Discovered ONVIF camera."""
        id: str
        name: str
        ip_address: str
        port: int
        manufacturer: str = ""
        model: str = ""
        hardware_id: str = ""
        firmware_version: str = ""
        service_url: str = ""
        profiles: List[str] = field(default_factory=list)
        stream_urls: Dict[str, str] = field(default_factory=dict)
        status: str = "discovered"
        username: str = ""
        password: str = ""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.cameras: Dict[str, ONVIFDiscovery.Camera] = {}
        self._lock = threading.Lock()
    
    async def discover(self) -> List['ONVIFDiscovery.Camera']:
        """Discover ONVIF cameras on the network."""
        self.cameras.clear()
        
        # Create UDP socket for WS-Discovery
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(0.5)
        
        try:
            # Bind to any interface
            sock.bind(('', 0))
            
            # Send probe message
            message_id = str(uuid.uuid4())
            probe = self.PROBE_MESSAGE.format(message_id=message_id)
            
            sock.sendto(
                probe.encode('utf-8'),
                (self.MULTICAST_IP, self.MULTICAST_PORT)
            )
            
            # Collect responses
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    data, addr = sock.recvfrom(65536)
                    await self._parse_probe_response(data.decode('utf-8'), addr[0])
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Error parsing response: {e}")
                    continue
        
        finally:
            sock.close()
        
        logger.info(f"Discovered {len(self.cameras)} ONVIF cameras")
        return list(self.cameras.values())
    
    async def _parse_probe_response(self, xml_str: str, source_ip: str):
        """Parse WS-Discovery probe response."""
        try:
            # Parse XML
            root = ET.fromstring(xml_str)
            
            # Find XAddrs (service URLs)
            for elem in root.iter():
                if elem.tag.endswith('XAddrs'):
                    xaddrs = elem.text
                    if xaddrs:
                        for url in xaddrs.split():
                            if 'onvif' in url.lower():
                                await self._add_camera_from_url(url, source_ip)
                                break
        
        except ET.ParseError as e:
            logger.debug(f"XML parse error: {e}")
    
    async def _add_camera_from_url(self, service_url: str, ip: str):
        """Add camera from service URL."""
        # Extract port from URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(service_url)
            port = parsed.port or 80
        except:
            port = 80
        
        camera_id = hashlib.md5(f"{ip}:{port}".encode()).hexdigest()[:12]
        
        if camera_id not in self.cameras:
            camera = self.Camera(
                id=camera_id,
                name=f"Camera {ip}",
                ip_address=ip,
                port=port,
                service_url=service_url,
            )
            
            with self._lock:
                self.cameras[camera_id] = camera
            
            logger.info(f"Discovered camera: {ip}:{port}")
    
    async def get_camera_info(self, camera: 'ONVIFDiscovery.Camera') -> bool:
        """Get detailed camera information via ONVIF."""
        if not HAS_AIOHTTP:
            return False
        
        try:
            # GetDeviceInformation SOAP request
            soap_request = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xmlns:xsd="http://www.w3.org/2001/XMLSchema">
        <GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>
    </s:Body>
</s:Envelope>"""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    camera.service_url,
                    data=soap_request,
                    headers={"Content-Type": "application/soap+xml"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        self._parse_device_info(camera, text)
                        return True
        
        except Exception as e:
            logger.debug(f"Failed to get camera info: {e}")
        
        return False
    
    def _parse_device_info(self, camera: 'ONVIFDiscovery.Camera', xml_str: str):
        """Parse GetDeviceInformation response."""
        try:
            root = ET.fromstring(xml_str)
            
            for elem in root.iter():
                tag = elem.tag.split('}')[-1]  # Remove namespace
                
                if tag == 'Manufacturer':
                    camera.manufacturer = elem.text or ""
                elif tag == 'Model':
                    camera.model = elem.text or ""
                elif tag == 'FirmwareVersion':
                    camera.firmware_version = elem.text or ""
                elif tag == 'HardwareId':
                    camera.hardware_id = elem.text or ""
            
            if camera.manufacturer or camera.model:
                camera.name = f"{camera.manufacturer} {camera.model}".strip()
        
        except Exception as e:
            logger.debug(f"Failed to parse device info: {e}")
    
    async def get_stream_uri(
        self,
        camera: 'ONVIFDiscovery.Camera',
        profile: str = "Profile_1"
    ) -> Optional[str]:
        """Get RTSP stream URI for camera."""
        if not HAS_AIOHTTP:
            return None
        
        try:
            # GetStreamUri SOAP request
            soap_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body>
        <GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">
            <StreamSetup>
                <Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>
                <Transport xmlns="http://www.onvif.org/ver10/schema">
                    <Protocol>RTSP</Protocol>
                </Transport>
            </StreamSetup>
            <ProfileToken>{profile}</ProfileToken>
        </GetStreamUri>
    </s:Body>
</s:Envelope>"""
            
            # Media service URL (usually on different path)
            media_url = camera.service_url.replace("/onvif/device_service", "/onvif/media_service")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    media_url,
                    data=soap_request,
                    headers={"Content-Type": "application/soap+xml"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        
                        # Parse URI from response
                        root = ET.fromstring(text)
                        for elem in root.iter():
                            if elem.tag.endswith('Uri'):
                                return elem.text
        
        except Exception as e:
            logger.debug(f"Failed to get stream URI: {e}")
        
        # Fallback: construct common RTSP URL
        return f"rtsp://{camera.ip_address}:554/stream1"
    
    def configure_camera(
        self,
        camera_id: str,
        username: str,
        password: str
    ) -> bool:
        """Configure camera credentials."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return False
        
        camera.username = username
        camera.password = password
        return True


# =============================================================================
# Device Manager
# =============================================================================

class DeviceManager:
    """
    Central manager for all connected devices.
    
    Handles:
    - Device discovery (USB, serial, network)
    - Device lifecycle management
    - Real-time status updates
    """
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.devices: Dict[str, Device] = {}
        self._lock = threading.Lock()
        
        # Device drivers
        self.doglove_driver: Optional[DOGloveDriver] = None
        self.onvif_discovery: Optional[ONVIFDiscovery] = None
        
        # Callbacks
        self._device_callbacks: List[Callable[[str, Device], None]] = []
        
        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._running = False
    
    def add_device_callback(self, callback: Callable[[str, Device], None]):
        """Add callback for device status changes."""
        self._device_callbacks.append(callback)
    
    def _notify_device_change(self, event: str, device: Device):
        """Notify callbacks of device change."""
        for callback in self._device_callbacks:
            try:
                callback(event, device)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def start(self):
        """Start device manager."""
        self._running = True
        
        # Initialize ONVIF discovery
        if self.config.onvif_enabled:
            self.onvif_discovery = ONVIFDiscovery(
                timeout=self.config.onvif_discovery_timeout
            )
        
        # Start background discovery
        if self.config.discovery_enabled:
            self._discovery_task = asyncio.create_task(self._discovery_loop())
        
        logger.info("Device manager started")
    
    async def stop(self):
        """Stop device manager."""
        self._running = False
        
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all devices
        await self.disconnect_all()
        
        logger.info("Device manager stopped")
    
    async def _discovery_loop(self):
        """Background device discovery."""
        while self._running:
            try:
                await self.discover_devices()
            except Exception as e:
                logger.error(f"Discovery error: {e}")
            
            await asyncio.sleep(self.config.discovery_interval_s)
    
    async def discover_devices(self) -> List[Device]:
        """Discover all connected devices."""
        discovered = []
        
        # Discover DOGlove devices
        if self.config.doglove_enabled:
            doglove_devices = await self._discover_doglove()
            discovered.extend(doglove_devices)
        
        # Discover ONVIF cameras
        if self.config.onvif_enabled and self.onvif_discovery:
            cameras = await self._discover_onvif()
            discovered.extend(cameras)
        
        # Discover USB cameras
        usb_cameras = await self._discover_usb_cameras()
        discovered.extend(usb_cameras)
        
        return discovered
    
    async def _discover_doglove(self) -> List[Device]:
        """Discover DOGlove devices."""
        devices = []
        
        found = DOGloveDriver.find_devices()
        for info in found:
            device_id = f"doglove_{hashlib.md5(info['port'].encode()).hexdigest()[:8]}"
            
            # Check if already known
            existing = self.devices.get(device_id)
            if existing:
                existing.last_seen = datetime.now().isoformat()
                devices.append(existing)
                continue
            
            device = Device(
                id=device_id,
                name=f"DOGlove ({info['port']})",
                type=DeviceType.DOGLOVE,
                status=DeviceStatus.DISCONNECTED,
                connection_type=ConnectionType.USB,
                address=info['port'],
                manufacturer="Dexta Robotics",
                model="DOGlove",
                serial_number=info.get('serial_number', ''),
                capabilities=["finger_tracking", "haptic_feedback", "imu"],
                last_seen=datetime.now().isoformat(),
            )
            
            with self._lock:
                self.devices[device_id] = device
            
            self._notify_device_change("discovered", device)
            devices.append(device)
        
        return devices
    
    async def _discover_onvif(self) -> List[Device]:
        """Discover ONVIF cameras."""
        devices = []
        
        if not self.onvif_discovery:
            return devices
        
        cameras = await self.onvif_discovery.discover()
        
        for camera in cameras:
            device_id = f"onvif_{camera.id}"
            
            # Check if already known
            existing = self.devices.get(device_id)
            if existing:
                existing.last_seen = datetime.now().isoformat()
                devices.append(existing)
                continue
            
            # Get detailed info
            await self.onvif_discovery.get_camera_info(camera)
            
            device = Device(
                id=device_id,
                name=camera.name or f"Camera {camera.ip_address}",
                type=DeviceType.ONVIF_CAMERA,
                status=DeviceStatus.DISCONNECTED,
                connection_type=ConnectionType.ETHERNET,
                address=camera.ip_address,
                manufacturer=camera.manufacturer,
                model=camera.model,
                firmware_version=camera.firmware_version,
                capabilities=["video_stream", "ptz", "events"],
                config={
                    "port": camera.port,
                    "service_url": camera.service_url,
                },
                last_seen=datetime.now().isoformat(),
            )
            
            with self._lock:
                self.devices[device_id] = device
            
            self._notify_device_change("discovered", device)
            devices.append(device)
        
        return devices
    
    async def _discover_usb_cameras(self) -> List[Device]:
        """Discover USB cameras via V4L2."""
        devices = []
        
        try:
            # List V4L2 devices
            v4l2_path = Path("/sys/class/video4linux")
            if v4l2_path.exists():
                for video_dev in v4l2_path.iterdir():
                    try:
                        name_path = video_dev / "name"
                        if name_path.exists():
                            name = name_path.read_text().strip()
                            dev_path = f"/dev/{video_dev.name}"
                            
                            device_id = f"usb_cam_{video_dev.name}"
                            
                            # Skip if already known
                            if device_id in self.devices:
                                self.devices[device_id].last_seen = datetime.now().isoformat()
                                devices.append(self.devices[device_id])
                                continue
                            
                            device = Device(
                                id=device_id,
                                name=name,
                                type=DeviceType.USB_CAMERA,
                                status=DeviceStatus.DISCONNECTED,
                                connection_type=ConnectionType.USB,
                                address=dev_path,
                                capabilities=["video_stream"],
                                last_seen=datetime.now().isoformat(),
                            )
                            
                            with self._lock:
                                self.devices[device_id] = device
                            
                            devices.append(device)
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"USB camera discovery error: {e}")
        
        return devices
    
    async def connect_device(self, device_id: str) -> bool:
        """Connect to a device."""
        device = self.devices.get(device_id)
        if not device:
            return False
        
        device.status = DeviceStatus.CONNECTING
        self._notify_device_change("connecting", device)
        
        try:
            if device.type == DeviceType.DOGLOVE:
                return await self._connect_doglove(device)
            elif device.type == DeviceType.ONVIF_CAMERA:
                return await self._connect_onvif_camera(device)
            elif device.type == DeviceType.USB_CAMERA:
                return await self._connect_usb_camera(device)
            else:
                device.status = DeviceStatus.ERROR
                device.error_message = "Unsupported device type"
                return False
                
        except Exception as e:
            device.status = DeviceStatus.ERROR
            device.error_message = str(e)
            self._notify_device_change("error", device)
            return False
    
    async def _connect_doglove(self, device: Device) -> bool:
        """Connect to DOGlove device."""
        self.doglove_driver = DOGloveDriver(
            port=device.address,
            baud_rate=self.config.doglove_baud_rate
        )
        
        if self.doglove_driver.connect():
            device.status = DeviceStatus.CONNECTED
            device.serial_number = self.doglove_driver.device_info.get("serial_number", "")
            device.firmware_version = self.doglove_driver.device_info.get("firmware_version", "")
            device.config["hand"] = self.doglove_driver.device_info.get("hand", "right")
            
            self._notify_device_change("connected", device)
            return True
        
        device.status = DeviceStatus.ERROR
        device.error_message = "Failed to connect"
        return False
    
    async def _connect_onvif_camera(self, device: Device) -> bool:
        """Connect to ONVIF camera."""
        if not self.onvif_discovery:
            return False
        
        camera = self.onvif_discovery.cameras.get(device.id.replace("onvif_", ""))
        if not camera:
            return False
        
        # Get stream URL
        stream_url = await self.onvif_discovery.get_stream_uri(camera)
        if stream_url:
            device.config["stream_url"] = stream_url
            device.status = DeviceStatus.CONNECTED
            self._notify_device_change("connected", device)
            return True
        
        device.status = DeviceStatus.ERROR
        device.error_message = "Failed to get stream URL"
        return False
    
    async def _connect_usb_camera(self, device: Device) -> bool:
        """Connect to USB camera."""
        # Verify device exists
        if Path(device.address).exists():
            device.status = DeviceStatus.CONNECTED
            self._notify_device_change("connected", device)
            return True
        
        device.status = DeviceStatus.ERROR
        device.error_message = "Device not found"
        return False
    
    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect from a device."""
        device = self.devices.get(device_id)
        if not device:
            return False
        
        if device.type == DeviceType.DOGLOVE and self.doglove_driver:
            self.doglove_driver.disconnect()
            self.doglove_driver = None
        
        device.status = DeviceStatus.DISCONNECTED
        self._notify_device_change("disconnected", device)
        return True
    
    async def disconnect_all(self):
        """Disconnect all devices."""
        for device_id in list(self.devices.keys()):
            await self.disconnect_device(device_id)
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> List[Device]:
        """Get all devices."""
        return list(self.devices.values())
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """Get devices by type."""
        return [d for d in self.devices.values() if d.type == device_type]


# =============================================================================
# Diagnostics and Troubleshooting
# =============================================================================

@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # pass, fail, warn
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: str = ""


class DiagnosticsService:
    """
    System diagnostics and troubleshooting.
    
    Provides:
    - Hardware checks
    - Network diagnostics
    - Device connectivity tests
    - Performance metrics
    """
    
    def __init__(self, device_manager: DeviceManager, config: EdgeConfig):
        self.device_manager = device_manager
        self.config = config
    
    async def run_all_diagnostics(self) -> List[DiagnosticResult]:
        """Run all diagnostic checks."""
        results = []
        
        results.append(await self._check_system_resources())
        results.append(await self._check_gpu())
        results.append(await self._check_network())
        results.append(await self._check_usb_ports())
        results.append(await self._check_platform_connection())
        
        # Device-specific checks
        for device in self.device_manager.get_all_devices():
            results.append(await self._check_device(device))
        
        return results
    
    async def _check_system_resources(self) -> DiagnosticResult:
        """Check CPU, memory, and disk."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"Low disk space: {100-disk.percent}% free")
            
            if issues:
                return DiagnosticResult(
                    name="System Resources",
                    status="warn",
                    message="; ".join(issues),
                    details={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": disk.percent,
                    },
                    fix_suggestion="Close unused applications or free up disk space"
                )
            
            return DiagnosticResult(
                name="System Resources",
                status="pass",
                message="All system resources OK",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                }
            )
            
        except ImportError:
            return DiagnosticResult(
                name="System Resources",
                status="warn",
                message="psutil not installed - cannot check resources",
                fix_suggestion="pip install psutil"
            )
    
    async def _check_gpu(self) -> DiagnosticResult:
        """Check GPU availability and status."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 4:
                    name = parts[0].strip()
                    mem_used = int(parts[1].strip())
                    mem_total = int(parts[2].strip())
                    temp = int(parts[3].strip())
                    
                    status = "pass"
                    message = f"GPU OK: {name}"
                    
                    if temp > 85:
                        status = "warn"
                        message = f"GPU temperature high: {temp}°C"
                    
                    return DiagnosticResult(
                        name="GPU",
                        status=status,
                        message=message,
                        details={
                            "name": name,
                            "memory_used_mb": mem_used,
                            "memory_total_mb": mem_total,
                            "temperature_c": temp,
                        }
                    )
            
            return DiagnosticResult(
                name="GPU",
                status="fail",
                message="GPU not detected or nvidia-smi failed",
                fix_suggestion="Ensure NVIDIA drivers are installed"
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="GPU",
                status="fail",
                message=f"GPU check failed: {e}",
                fix_suggestion="Ensure nvidia-smi is available"
            )
    
    async def _check_network(self) -> DiagnosticResult:
        """Check network connectivity."""
        results = {"local": False, "internet": False, "dns": False}
        
        # Check local network
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("10.255.255.255", 1))
            local_ip = sock.getsockname()[0]
            sock.close()
            results["local"] = True
            results["local_ip"] = local_ip
        except Exception:
            pass
        
        # Check internet
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            results["internet"] = True
        except Exception:
            pass
        
        # Check DNS
        try:
            socket.gethostbyname("google.com")
            results["dns"] = True
        except Exception:
            pass
        
        if all([results["local"], results["internet"], results["dns"]]):
            return DiagnosticResult(
                name="Network",
                status="pass",
                message=f"Network OK (IP: {results.get('local_ip', 'unknown')})",
                details=results
            )
        else:
            issues = []
            if not results["local"]:
                issues.append("No local network")
            if not results["internet"]:
                issues.append("No internet connection")
            if not results["dns"]:
                issues.append("DNS not working")
            
            return DiagnosticResult(
                name="Network",
                status="fail" if not results["local"] else "warn",
                message="; ".join(issues),
                details=results,
                fix_suggestion="Check network cable or WiFi connection"
            )
    
    async def _check_usb_ports(self) -> DiagnosticResult:
        """Check USB port availability."""
        try:
            usb_devices = []
            
            if HAS_SERIAL:
                ports = serial.tools.list_ports.comports()
                for port in ports:
                    usb_devices.append({
                        "port": port.device,
                        "description": port.description,
                        "vid": port.vid,
                        "pid": port.pid,
                    })
            
            return DiagnosticResult(
                name="USB Ports",
                status="pass",
                message=f"Found {len(usb_devices)} USB serial devices",
                details={"devices": usb_devices}
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="USB Ports",
                status="warn",
                message=f"USB check error: {e}"
            )
    
    async def _check_platform_connection(self) -> DiagnosticResult:
        """Check connection to Dynamical.ai platform."""
        if not self.config.platform_url:
            return DiagnosticResult(
                name="Platform Connection",
                status="warn",
                message="Platform URL not configured",
                fix_suggestion="Configure platform_url in settings"
            )
        
        try:
            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config.platform_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            return DiagnosticResult(
                                name="Platform Connection",
                                status="pass",
                                message="Connected to platform",
                                details={"url": self.config.platform_url}
                            )
                        else:
                            return DiagnosticResult(
                                name="Platform Connection",
                                status="fail",
                                message=f"Platform returned status {response.status}",
                                fix_suggestion="Check platform URL and API key"
                            )
            
            return DiagnosticResult(
                name="Platform Connection",
                status="warn",
                message="aiohttp not available",
                fix_suggestion="pip install aiohttp"
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="Platform Connection",
                status="fail",
                message=f"Connection failed: {e}",
                fix_suggestion="Check network and platform URL"
            )
    
    async def _check_device(self, device: Device) -> DiagnosticResult:
        """Check specific device health."""
        if device.status == DeviceStatus.CONNECTED:
            return DiagnosticResult(
                name=f"Device: {device.name}",
                status="pass",
                message="Connected and operational",
                details={"id": device.id, "type": device.type.value}
            )
        elif device.status == DeviceStatus.ERROR:
            return DiagnosticResult(
                name=f"Device: {device.name}",
                status="fail",
                message=device.error_message or "Device error",
                details={"id": device.id, "type": device.type.value},
                fix_suggestion="Try reconnecting the device"
            )
        else:
            return DiagnosticResult(
                name=f"Device: {device.name}",
                status="warn",
                message=f"Device status: {device.status.value}",
                details={"id": device.id, "type": device.type.value},
                fix_suggestion="Click 'Connect' to establish connection"
            )
    
    async def troubleshoot_device(self, device_id: str) -> List[DiagnosticResult]:
        """Run device-specific troubleshooting."""
        device = self.device_manager.get_device(device_id)
        if not device:
            return [DiagnosticResult(
                name="Device",
                status="fail",
                message="Device not found"
            )]
        
        results = []
        
        if device.type == DeviceType.DOGLOVE:
            results.extend(await self._troubleshoot_doglove(device))
        elif device.type == DeviceType.ONVIF_CAMERA:
            results.extend(await self._troubleshoot_onvif(device))
        elif device.type == DeviceType.USB_CAMERA:
            results.extend(await self._troubleshoot_usb_camera(device))
        
        return results
    
    async def _troubleshoot_doglove(self, device: Device) -> List[DiagnosticResult]:
        """Troubleshoot DOGlove device."""
        results = []
        
        # Check if port exists
        if Path(device.address).exists():
            results.append(DiagnosticResult(
                name="Serial Port",
                status="pass",
                message=f"Port {device.address} exists"
            ))
        else:
            results.append(DiagnosticResult(
                name="Serial Port",
                status="fail",
                message=f"Port {device.address} not found",
                fix_suggestion="Check USB connection and try different port"
            ))
            return results
        
        # Check permissions
        try:
            import os
            if os.access(device.address, os.R_OK | os.W_OK):
                results.append(DiagnosticResult(
                    name="Port Permissions",
                    status="pass",
                    message="Read/write access OK"
                ))
            else:
                results.append(DiagnosticResult(
                    name="Port Permissions",
                    status="fail",
                    message="No read/write permission",
                    fix_suggestion="Add user to dialout group: sudo usermod -a -G dialout $USER"
                ))
        except Exception as e:
            results.append(DiagnosticResult(
                name="Port Permissions",
                status="fail",
                message=str(e)
            ))
        
        # Try connection test
        try:
            driver = DOGloveDriver(port=device.address)
            if driver.connect():
                results.append(DiagnosticResult(
                    name="Connection Test",
                    status="pass",
                    message="Successfully connected to DOGlove"
                ))
                driver.disconnect()
            else:
                results.append(DiagnosticResult(
                    name="Connection Test",
                    status="fail",
                    message="Could not establish connection",
                    fix_suggestion="Try power cycling the glove"
                ))
        except Exception as e:
            results.append(DiagnosticResult(
                name="Connection Test",
                status="fail",
                message=str(e)
            ))
        
        return results
    
    async def _troubleshoot_onvif(self, device: Device) -> List[DiagnosticResult]:
        """Troubleshoot ONVIF camera."""
        results = []
        
        # Ping test
        try:
            import subprocess
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", device.address],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                results.append(DiagnosticResult(
                    name="Network Reachability",
                    status="pass",
                    message=f"Camera at {device.address} is reachable"
                ))
            else:
                results.append(DiagnosticResult(
                    name="Network Reachability",
                    status="fail",
                    message=f"Cannot reach {device.address}",
                    fix_suggestion="Check network connection and camera IP"
                ))
        except Exception as e:
            results.append(DiagnosticResult(
                name="Network Reachability",
                status="fail",
                message=str(e)
            ))
        
        # Port check
        port = device.config.get("port", 80)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((device.address, port))
            sock.close()
            
            if result == 0:
                results.append(DiagnosticResult(
                    name="ONVIF Port",
                    status="pass",
                    message=f"Port {port} is open"
                ))
            else:
                results.append(DiagnosticResult(
                    name="ONVIF Port",
                    status="fail",
                    message=f"Port {port} is closed",
                    fix_suggestion="Check camera ONVIF settings"
                ))
        except Exception as e:
            results.append(DiagnosticResult(
                name="ONVIF Port",
                status="fail",
                message=str(e)
            ))
        
        return results
    
    async def _troubleshoot_usb_camera(self, device: Device) -> List[DiagnosticResult]:
        """Troubleshoot USB camera."""
        results = []
        
        # Check device exists
        if Path(device.address).exists():
            results.append(DiagnosticResult(
                name="Device Node",
                status="pass",
                message=f"{device.address} exists"
            ))
        else:
            results.append(DiagnosticResult(
                name="Device Node",
                status="fail",
                message=f"{device.address} not found",
                fix_suggestion="Check USB connection"
            ))
            return results
        
        # Check V4L2 capabilities
        try:
            import subprocess
            result = subprocess.run(
                ["v4l2-ctl", "--device", device.address, "--all"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                results.append(DiagnosticResult(
                    name="V4L2 Query",
                    status="pass",
                    message="Camera responds to V4L2 commands"
                ))
            else:
                results.append(DiagnosticResult(
                    name="V4L2 Query",
                    status="warn",
                    message="V4L2 query failed"
                ))
        except FileNotFoundError:
            results.append(DiagnosticResult(
                name="V4L2 Query",
                status="warn",
                message="v4l2-ctl not installed",
                fix_suggestion="apt install v4l-utils"
            ))
        except Exception as e:
            results.append(DiagnosticResult(
                name="V4L2 Query",
                status="fail",
                message=str(e)
            ))
        
        return results


# =============================================================================
# WebSocket Manager
# =============================================================================

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Add new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        async with self._lock:
            self.connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        if not self.connections:
            return
        
        data = json.dumps(message)
        dead_connections = set()
        
        async with self._lock:
            for ws in self.connections:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead_connections.add(ws)
            
            self.connections -= dead_connections
    
    async def send_device_update(self, event: str, device: Device):
        """Send device status update."""
        await self.broadcast({
            "type": "device_update",
            "event": event,
            "device": asdict(device),
            "timestamp": datetime.now().isoformat(),
        })


# =============================================================================
# API Models
# =============================================================================

class DeviceResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    connection_type: str
    address: str
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    firmware_version: str = ""
    last_seen: str = ""
    error_message: str = ""
    capabilities: List[str] = []
    config: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}


class DeviceListResponse(BaseModel):
    devices: List[DeviceResponse]
    count: int


class DiagnosticResultResponse(BaseModel):
    name: str
    status: str
    message: str
    details: Dict[str, Any] = {}
    fix_suggestion: str = ""


class ConfigureDeviceRequest(BaseModel):
    username: str = ""
    password: str = ""
    config: Dict[str, Any] = {}


class DOGloveStateResponse(BaseModel):
    finger_positions: List[float]
    finger_forces: List[float]
    orientation: List[float]
    accelerometer: List[float]
    gyroscope: List[float]
    timestamp: str


class HapticFeedbackRequest(BaseModel):
    finger: int = Field(..., ge=0, le=4)
    intensity: float = Field(..., ge=0.0, le=1.0)


class SystemInfoResponse(BaseModel):
    device_id: str
    device_name: str
    version: str
    uptime_seconds: float
    platform_connected: bool
    device_count: int


# =============================================================================
# FastAPI Application
# =============================================================================

# Global instances
config: EdgeConfig = None
device_manager: DeviceManager = None
diagnostics: DiagnosticsService = None
ws_manager: WebSocketManager = None
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global device_manager, diagnostics, ws_manager
    
    # Startup
    logger.info("Starting Edge Service...")
    
    ws_manager = WebSocketManager()
    device_manager = DeviceManager(config)
    diagnostics = DiagnosticsService(device_manager, config)
    
    # Add WebSocket callback for device updates
    def device_callback(event: str, device: Device):
        asyncio.create_task(ws_manager.send_device_update(event, device))
    
    device_manager.add_device_callback(device_callback)
    
    await device_manager.start()
    
    # Initial device discovery
    await device_manager.discover_devices()
    
    logger.info("Edge Service started")
    
    yield
    
    # Shutdown
    logger.info("Stopping Edge Service...")
    await device_manager.stop()
    logger.info("Edge Service stopped")


app = FastAPI(
    title="Dynamical.ai Edge Service",
    description="Edge service for device management, DOGlove integration, and ONVIF cameras",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get edge system information."""
    return SystemInfoResponse(
        device_id=config.device_id,
        device_name=config.device_name,
        version=__version__,
        uptime_seconds=time.time() - start_time,
        platform_connected=bool(config.platform_url),
        device_count=len(device_manager.get_all_devices()),
    )


@app.get("/api/v1/system/jetson")
async def get_jetson_info():
    """Get Jetson hardware information."""
    from jetson.jetson_sdk import JetsonDetector
    info = JetsonDetector.detect()
    return asdict(info)


# Device Management
@app.get("/api/v1/devices", response_model=DeviceListResponse)
async def list_devices(
    type: Optional[str] = None,
    status: Optional[str] = None,
):
    """List all discovered devices."""
    devices = device_manager.get_all_devices()
    
    if type:
        devices = [d for d in devices if d.type.value == type]
    if status:
        devices = [d for d in devices if d.status.value == status]
    
    return DeviceListResponse(
        devices=[DeviceResponse(**asdict(d)) for d in devices],
        count=len(devices),
    )


@app.post("/api/v1/devices/discover")
async def discover_devices():
    """Trigger device discovery."""
    devices = await device_manager.discover_devices()
    return {
        "discovered": len(devices),
        "devices": [DeviceResponse(**asdict(d)) for d in devices],
    }


@app.get("/api/v1/devices/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str):
    """Get device details."""
    device = device_manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return DeviceResponse(**asdict(device))


@app.post("/api/v1/devices/{device_id}/connect")
async def connect_device(device_id: str):
    """Connect to a device."""
    success = await device_manager.connect_device(device_id)
    device = device_manager.get_device(device_id)
    
    return {
        "success": success,
        "device": DeviceResponse(**asdict(device)) if device else None,
    }


@app.post("/api/v1/devices/{device_id}/disconnect")
async def disconnect_device(device_id: str):
    """Disconnect from a device."""
    success = await device_manager.disconnect_device(device_id)
    return {"success": success}


@app.post("/api/v1/devices/{device_id}/configure")
async def configure_device(device_id: str, request: ConfigureDeviceRequest):
    """Configure device settings."""
    device = device_manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Update configuration
    if request.config:
        device.config.update(request.config)
    
    # Handle ONVIF camera credentials
    if device.type == DeviceType.ONVIF_CAMERA:
        if request.username and request.password:
            camera_id = device_id.replace("onvif_", "")
            if device_manager.onvif_discovery:
                device_manager.onvif_discovery.configure_camera(
                    camera_id, request.username, request.password
                )
    
    return {"success": True}


# DOGlove Specific Endpoints
@app.get("/api/v1/doglove/state", response_model=DOGloveStateResponse)
async def get_doglove_state():
    """Get current DOGlove state."""
    if not device_manager.doglove_driver:
        raise HTTPException(status_code=400, detail="DOGlove not connected")
    
    state = device_manager.doglove_driver.get_state()
    return DOGloveStateResponse(**state)


@app.post("/api/v1/doglove/haptic")
async def set_haptic_feedback(request: HapticFeedbackRequest):
    """Set haptic feedback for a finger."""
    if not device_manager.doglove_driver:
        raise HTTPException(status_code=400, detail="DOGlove not connected")
    
    success = device_manager.doglove_driver.set_haptic_feedback(
        request.finger, request.intensity
    )
    return {"success": success}


@app.post("/api/v1/doglove/calibrate")
async def calibrate_doglove():
    """Run DOGlove calibration."""
    if not device_manager.doglove_driver:
        raise HTTPException(status_code=400, detail="DOGlove not connected")
    
    success = device_manager.doglove_driver.calibrate()
    return {"success": success}


@app.post("/api/v1/doglove/stream/start")
async def start_doglove_stream():
    """Start DOGlove data streaming."""
    if not device_manager.doglove_driver:
        raise HTTPException(status_code=400, detail="DOGlove not connected")
    
    def stream_callback(state):
        asyncio.create_task(ws_manager.broadcast({
            "type": "doglove_data",
            "data": state,
        }))
    
    device_manager.doglove_driver.start_streaming(stream_callback)
    
    # Update device status
    devices = device_manager.get_devices_by_type(DeviceType.DOGLOVE)
    for d in devices:
        d.status = DeviceStatus.STREAMING
    
    return {"success": True}


@app.post("/api/v1/doglove/stream/stop")
async def stop_doglove_stream():
    """Stop DOGlove data streaming."""
    if device_manager.doglove_driver:
        device_manager.doglove_driver.stop_streaming()
        
        # Update device status
        devices = device_manager.get_devices_by_type(DeviceType.DOGLOVE)
        for d in devices:
            if d.status == DeviceStatus.STREAMING:
                d.status = DeviceStatus.CONNECTED
    
    return {"success": True}


# ONVIF Camera Endpoints
@app.get("/api/v1/cameras")
async def list_cameras():
    """List discovered cameras."""
    cameras = device_manager.get_devices_by_type(DeviceType.ONVIF_CAMERA)
    cameras += device_manager.get_devices_by_type(DeviceType.USB_CAMERA)
    return {
        "cameras": [DeviceResponse(**asdict(c)) for c in cameras],
        "count": len(cameras),
    }


@app.get("/api/v1/cameras/{device_id}/stream-url")
async def get_camera_stream_url(device_id: str):
    """Get camera stream URL."""
    device = device_manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if device.type == DeviceType.ONVIF_CAMERA:
        stream_url = device.config.get("stream_url")
        if not stream_url and device_manager.onvif_discovery:
            camera = device_manager.onvif_discovery.cameras.get(
                device_id.replace("onvif_", "")
            )
            if camera:
                stream_url = await device_manager.onvif_discovery.get_stream_uri(camera)
                device.config["stream_url"] = stream_url
        
        return {"stream_url": stream_url}
    
    elif device.type == DeviceType.USB_CAMERA:
        return {"device_path": device.address}
    
    raise HTTPException(status_code=400, detail="Not a camera device")


# Diagnostics Endpoints
@app.get("/api/v1/diagnostics", response_model=List[DiagnosticResultResponse])
async def run_diagnostics():
    """Run all system diagnostics."""
    results = await diagnostics.run_all_diagnostics()
    return [DiagnosticResultResponse(**asdict(r)) for r in results]


@app.get("/api/v1/diagnostics/device/{device_id}", response_model=List[DiagnosticResultResponse])
async def troubleshoot_device(device_id: str):
    """Run device-specific troubleshooting."""
    results = await diagnostics.troubleshoot_device(device_id)
    return [DiagnosticResultResponse(**asdict(r)) for r in results]


# Configuration Endpoints
@app.get("/api/v1/config")
async def get_config():
    """Get current configuration."""
    return asdict(config)


@app.patch("/api/v1/config")
async def update_config(updates: Dict[str, Any]):
    """Update configuration."""
    global config
    
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Save configuration
    config_path = os.getenv("CONFIG_PATH", "/etc/dynamical/edge.json")
    config.save(config_path)
    
    return {"success": True}


# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Receive messages (for commands)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle ping
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            # Handle device subscription
            elif message.get("type") == "subscribe":
                # Send current device list
                devices = device_manager.get_all_devices()
                await websocket.send_json({
                    "type": "device_list",
                    "devices": [asdict(d) for d in devices],
                })
    
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the edge service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamical.ai Edge Service")
    parser.add_argument("--config", default="/etc/dynamical/edge.json", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    
    args = parser.parse_args()
    
    global config
    config = EdgeConfig.load(args.config)
    
    # Override port from config
    port = config.api_port or args.port
    
    logger.info(f"Starting Edge Service on {args.host}:{port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

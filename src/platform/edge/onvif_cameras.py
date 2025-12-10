"""
ONVIF Camera Service

Complete ONVIF camera integration for Dynamical.ai platform.
Supports camera discovery, configuration, PTZ control, and video streaming.

ONVIF (Open Network Video Interface Forum) is an industry standard
for IP-based security cameras that enables interoperability.

Features:
- WS-Discovery for automatic camera detection
- Device information retrieval
- PTZ (Pan-Tilt-Zoom) control
- Video streaming (RTSP)
- Snapshot capture
- Event subscriptions
- Multi-camera management

Example:
    ```python
    from platform.edge.onvif_cameras import ONVIFCameraManager
    
    # Create manager
    manager = ONVIFCameraManager()
    
    # Discover cameras
    cameras = manager.discover()
    
    # Connect to camera
    camera = manager.connect("192.168.1.100", "admin", "password")
    
    # Get camera info
    info = camera.get_device_info()
    
    # PTZ control
    camera.ptz_move(pan=0.5, tilt=0.3, zoom=0.0)
    
    # Get stream URL
    stream_url = camera.get_stream_url()
    
    # Capture snapshot
    camera.capture_snapshot("snapshot.jpg")
    ```
"""

import os
import socket
import struct
import threading
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from zeep import Client, Settings
    from zeep.wsse.username import UsernameToken
    from zeep.transports import Transport
    HAS_ZEEP = True
except ImportError:
    HAS_ZEEP = False
    logger.debug("zeep not available - using basic ONVIF implementation")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# Enums and Constants
# =============================================================================

class CameraStatus(str, Enum):
    """Camera connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class StreamProfile(str, Enum):
    """Stream quality profiles."""
    MAIN = "main"
    SUB = "sub"
    THIRD = "third"


class PTZPreset(str, Enum):
    """Common PTZ presets."""
    HOME = "home"
    PATROL_1 = "patrol_1"
    PATROL_2 = "patrol_2"


# ONVIF WS-Discovery constants
class WSDiscovery:
    """WS-Discovery constants."""
    MULTICAST_IP = "239.255.255.250"
    MULTICAST_PORT = 3702
    PROBE_TIMEOUT = 3.0
    
    # SOAP namespaces
    NS_SOAP = "http://www.w3.org/2003/05/soap-envelope"
    NS_WSA = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
    NS_WSD = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
    NS_ONVIF = "http://www.onvif.org/ver10/network/wsdl"
    NS_TDS = "http://www.onvif.org/ver10/device/wsdl"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CameraInfo:
    """Camera device information."""
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    serial_number: str = ""
    hardware_id: str = ""
    onvif_version: str = ""
    
    # Network info
    ip_address: str = ""
    mac_address: str = ""
    hostname: str = ""
    
    # Capabilities
    has_ptz: bool = False
    has_audio: bool = False
    has_analytics: bool = False
    has_events: bool = False


@dataclass
class StreamInfo:
    """Video stream information."""
    profile_token: str = ""
    name: str = ""
    encoding: str = "H264"
    resolution_width: int = 1920
    resolution_height: int = 1080
    frame_rate: int = 30
    bitrate_kbps: int = 4000
    rtsp_url: str = ""
    snapshot_url: str = ""


@dataclass
class PTZStatus:
    """PTZ position status."""
    pan: float = 0.0  # -1 to 1
    tilt: float = 0.0  # -1 to 1
    zoom: float = 0.0  # 0 to 1
    moving: bool = False
    preset: Optional[str] = None


@dataclass
class DiscoveredCamera:
    """Discovered camera from WS-Discovery."""
    ip_address: str
    xaddrs: List[str]
    scopes: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    metadata_version: int = 1
    
    @property
    def name(self) -> str:
        """Extract name from scopes."""
        for scope in self.scopes:
            if "name/" in scope.lower():
                return scope.split("/")[-1]
            if "hardware/" in scope.lower():
                return scope.split("/")[-1]
        return f"Camera ({self.ip_address})"
    
    @property
    def manufacturer(self) -> str:
        """Extract manufacturer from scopes."""
        for scope in self.scopes:
            if "mfgr/" in scope.lower() or "manufacturer/" in scope.lower():
                return scope.split("/")[-1]
        return "Unknown"


@dataclass
class CameraEvent:
    """Camera event notification."""
    timestamp: str
    topic: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# WS-Discovery
# =============================================================================

class ONVIFDiscovery:
    """
    ONVIF camera discovery using WS-Discovery.
    
    Sends multicast probe messages and collects responses from
    ONVIF-compatible cameras on the network.
    """
    
    def __init__(self):
        self._discovered: Dict[str, DiscoveredCamera] = {}
    
    def discover(self, timeout: float = WSDiscovery.PROBE_TIMEOUT) -> List[DiscoveredCamera]:
        """
        Discover ONVIF cameras on the network.
        
        Args:
            timeout: Discovery timeout in seconds
        
        Returns:
            List of discovered cameras
        """
        self._discovered.clear()
        
        # Build probe message
        probe_message = self._build_probe_message()
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(timeout)
        
        try:
            # Send probe to multicast address
            sock.sendto(
                probe_message.encode('utf-8'),
                (WSDiscovery.MULTICAST_IP, WSDiscovery.MULTICAST_PORT)
            )
            
            # Collect responses
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    data, addr = sock.recvfrom(65535)
                    self._parse_probe_response(data.decode('utf-8', errors='ignore'), addr[0])
                except socket.timeout:
                    break
                except Exception as e:
                    logger.debug(f"Error receiving response: {e}")
        
        finally:
            sock.close()
        
        cameras = list(self._discovered.values())
        logger.info(f"Discovered {len(cameras)} ONVIF camera(s)")
        return cameras
    
    def _build_probe_message(self) -> str:
        """Build WS-Discovery probe message."""
        message_id = f"uuid:{uuid.uuid4()}"
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="{WSDiscovery.NS_SOAP}"
            xmlns:a="{WSDiscovery.NS_WSA}"
            xmlns:d="{WSDiscovery.NS_WSD}"
            xmlns:dn="{WSDiscovery.NS_ONVIF}">
    <s:Header>
        <a:Action s:mustUnderstand="1">{WSDiscovery.NS_WSD}/Probe</a:Action>
        <a:MessageID>{message_id}</a:MessageID>
        <a:ReplyTo>
            <a:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:Address>
        </a:ReplyTo>
        <a:To s:mustUnderstand="1">urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>
    </s:Header>
    <s:Body>
        <d:Probe>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
        </d:Probe>
    </s:Body>
</s:Envelope>"""
    
    def _parse_probe_response(self, response: str, ip_address: str):
        """Parse WS-Discovery probe match response."""
        try:
            # Parse XML
            root = ET.fromstring(response)
            
            # Find ProbeMatch elements
            ns = {
                's': WSDiscovery.NS_SOAP,
                'a': WSDiscovery.NS_WSA,
                'd': WSDiscovery.NS_WSD,
            }
            
            for match in root.findall('.//d:ProbeMatch', ns):
                # Extract XAddrs (service addresses)
                xaddrs_elem = match.find('.//d:XAddrs', ns)
                xaddrs = xaddrs_elem.text.split() if xaddrs_elem is not None and xaddrs_elem.text else []
                
                # Extract scopes
                scopes_elem = match.find('.//d:Scopes', ns)
                scopes = scopes_elem.text.split() if scopes_elem is not None and scopes_elem.text else []
                
                # Extract types
                types_elem = match.find('.//d:Types', ns)
                types = types_elem.text.split() if types_elem is not None and types_elem.text else []
                
                # Extract metadata version
                metadata_elem = match.find('.//d:MetadataVersion', ns)
                metadata_version = int(metadata_elem.text) if metadata_elem is not None and metadata_elem.text else 1
                
                # Use first XAddr to determine actual IP (might differ from sender)
                camera_ip = ip_address
                if xaddrs:
                    try:
                        parsed = urlparse(xaddrs[0])
                        camera_ip = parsed.hostname or ip_address
                    except:
                        pass
                
                # Add to discovered
                if camera_ip not in self._discovered:
                    self._discovered[camera_ip] = DiscoveredCamera(
                        ip_address=camera_ip,
                        xaddrs=xaddrs,
                        scopes=scopes,
                        types=types,
                        metadata_version=metadata_version,
                    )
                    logger.debug(f"Discovered camera: {camera_ip}")
        
        except ET.ParseError as e:
            logger.debug(f"Failed to parse response: {e}")
        except Exception as e:
            logger.debug(f"Error parsing response: {e}")


# =============================================================================
# ONVIF Camera Client
# =============================================================================

class ONVIFCamera:
    """
    ONVIF camera client.
    
    Provides control and streaming for a single ONVIF camera.
    """
    
    def __init__(
        self,
        ip_address: str,
        username: str = "admin",
        password: str = "",
        port: int = 80,
        wsdl_dir: str = None
    ):
        """
        Initialize ONVIF camera client.
        
        Args:
            ip_address: Camera IP address
            username: ONVIF username
            password: ONVIF password
            port: HTTP port (usually 80)
            wsdl_dir: Directory containing ONVIF WSDL files
        """
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.port = port
        self.wsdl_dir = wsdl_dir
        
        self._status = CameraStatus.DISCONNECTED
        self._info: Optional[CameraInfo] = None
        self._profiles: List[StreamInfo] = []
        self._ptz_status: Optional[PTZStatus] = None
        
        # Service endpoints (discovered via GetCapabilities)
        self._device_service_url = f"http://{ip_address}:{port}/onvif/device_service"
        self._media_service_url: Optional[str] = None
        self._ptz_service_url: Optional[str] = None
        self._imaging_service_url: Optional[str] = None
        self._events_service_url: Optional[str] = None
        
        # SOAP client (if zeep available)
        self._device_client = None
        self._media_client = None
        self._ptz_client = None
    
    @property
    def status(self) -> CameraStatus:
        """Get connection status."""
        return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status in [CameraStatus.CONNECTED, CameraStatus.STREAMING]
    
    @property
    def info(self) -> Optional[CameraInfo]:
        """Get camera info."""
        return self._info
    
    @property
    def profiles(self) -> List[StreamInfo]:
        """Get available stream profiles."""
        return self._profiles
    
    def connect(self) -> bool:
        """
        Connect to camera and retrieve capabilities.
        
        Returns:
            True if connected successfully
        """
        self._status = CameraStatus.CONNECTING
        
        try:
            # Get device information
            self._info = self._get_device_info()
            
            # Get capabilities and service URLs
            self._get_capabilities()
            
            # Get media profiles
            self._profiles = self._get_profiles()
            
            self._status = CameraStatus.CONNECTED
            logger.info(f"Connected to camera: {self.ip_address}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            self._status = CameraStatus.ERROR
            return False
    
    def disconnect(self):
        """Disconnect from camera."""
        self._status = CameraStatus.DISCONNECTED
        logger.info(f"Disconnected from camera: {self.ip_address}")
    
    def _get_device_info(self) -> CameraInfo:
        """Get device information via ONVIF GetDeviceInformation."""
        info = CameraInfo(ip_address=self.ip_address)
        
        try:
            if HAS_ZEEP:
                return self._get_device_info_zeep()
            else:
                return self._get_device_info_basic()
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
        
        return info
    
    def _get_device_info_basic(self) -> CameraInfo:
        """Get device info using basic SOAP."""
        info = CameraInfo(ip_address=self.ip_address)
        
        soap_body = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
    <s:Header>
        {security}
    </s:Header>
    <s:Body>
        <tds:GetDeviceInformation/>
    </s:Body>
</s:Envelope>"""
        
        security = self._build_security_header()
        soap_body = soap_body.format(security=security)
        
        try:
            response = requests.post(
                self._device_service_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # Parse response
                ns = {'tds': 'http://www.onvif.org/ver10/device/wsdl'}
                
                manufacturer = root.find('.//tds:Manufacturer', ns)
                info.manufacturer = manufacturer.text if manufacturer is not None else ""
                
                model = root.find('.//tds:Model', ns)
                info.model = model.text if model is not None else ""
                
                firmware = root.find('.//tds:FirmwareVersion', ns)
                info.firmware_version = firmware.text if firmware is not None else ""
                
                serial = root.find('.//tds:SerialNumber', ns)
                info.serial_number = serial.text if serial is not None else ""
                
                hardware = root.find('.//tds:HardwareId', ns)
                info.hardware_id = hardware.text if hardware is not None else ""
        
        except Exception as e:
            logger.warning(f"GetDeviceInformation failed: {e}")
        
        return info
    
    def _get_device_info_zeep(self) -> CameraInfo:
        """Get device info using zeep SOAP client."""
        info = CameraInfo(ip_address=self.ip_address)
        
        # Create zeep client with authentication
        settings = Settings(strict=False, xml_huge_tree=True)
        transport = Transport(timeout=10)
        
        wsdl_url = f"http://{self.ip_address}:{self.port}/onvif/device_service?wsdl"
        
        client = Client(
            wsdl_url,
            wsse=UsernameToken(self.username, self.password),
            settings=settings,
            transport=transport
        )
        
        result = client.service.GetDeviceInformation()
        
        info.manufacturer = result.Manufacturer or ""
        info.model = result.Model or ""
        info.firmware_version = result.FirmwareVersion or ""
        info.serial_number = result.SerialNumber or ""
        info.hardware_id = result.HardwareId or ""
        
        return info
    
    def _build_security_header(self) -> str:
        """Build WS-Security header for SOAP requests."""
        import hashlib
        import base64
        from datetime import datetime
        
        # Create nonce and timestamp
        nonce = base64.b64encode(os.urandom(16)).decode()
        created = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Create password digest (nonce + created + password)
        nonce_bytes = base64.b64decode(nonce)
        created_bytes = created.encode('utf-8')
        password_bytes = self.password.encode('utf-8')
        
        digest = hashlib.sha1(nonce_bytes + created_bytes + password_bytes).digest()
        password_digest = base64.b64encode(digest).decode()
        
        return f"""<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
                             xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">
        <wsse:UsernameToken>
            <wsse:Username>{self.username}</wsse:Username>
            <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{password_digest}</wsse:Password>
            <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce}</wsse:Nonce>
            <wsu:Created>{created}</wsu:Created>
        </wsse:UsernameToken>
    </wsse:Security>"""
    
    def _get_capabilities(self):
        """Get camera capabilities and service URLs."""
        soap_body = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
    <s:Header>{security}</s:Header>
    <s:Body>
        <tds:GetCapabilities>
            <tds:Category>All</tds:Category>
        </tds:GetCapabilities>
    </s:Body>
</s:Envelope>"""
        
        security = self._build_security_header()
        soap_body = soap_body.format(security=security)
        
        try:
            response = requests.post(
                self._device_service_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # Find service URLs in response
                content = response.content.decode('utf-8')
                
                # Parse Media service URL
                if 'Media' in content:
                    import re
                    media_match = re.search(r'<tt:Media[^>]*>.*?<tt:XAddr>([^<]+)</tt:XAddr>', content, re.DOTALL)
                    if media_match:
                        self._media_service_url = media_match.group(1)
                        self._info.has_audio = True
                
                # Parse PTZ service URL
                if 'PTZ' in content:
                    ptz_match = re.search(r'<tt:PTZ[^>]*>.*?<tt:XAddr>([^<]+)</tt:XAddr>', content, re.DOTALL)
                    if ptz_match:
                        self._ptz_service_url = ptz_match.group(1)
                        self._info.has_ptz = True
                
                # Parse Events service URL
                if 'Events' in content:
                    events_match = re.search(r'<tt:Events[^>]*>.*?<tt:XAddr>([^<]+)</tt:XAddr>', content, re.DOTALL)
                    if events_match:
                        self._events_service_url = events_match.group(1)
                        self._info.has_events = True
                
                # Parse Analytics
                if 'Analytics' in content:
                    self._info.has_analytics = True
        
        except Exception as e:
            logger.warning(f"GetCapabilities failed: {e}")
    
    def _get_profiles(self) -> List[StreamInfo]:
        """Get media profiles."""
        profiles = []
        
        media_url = self._media_service_url or f"http://{self.ip_address}:{self.port}/onvif/media"
        
        soap_body = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
    <s:Header>{security}</s:Header>
    <s:Body>
        <trt:GetProfiles/>
    </s:Body>
</s:Envelope>"""
        
        security = self._build_security_header()
        soap_body = soap_body.format(security=security)
        
        try:
            response = requests.post(
                media_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # Find all profiles
                for profile in root.iter():
                    if 'Profiles' in profile.tag:
                        token = profile.get('token', '')
                        if token:
                            stream_info = StreamInfo(profile_token=token)
                            
                            # Get profile name
                            name_elem = profile.find('.//{http://www.onvif.org/ver10/schema}Name')
                            if name_elem is not None:
                                stream_info.name = name_elem.text or token
                            
                            # Get stream URL for this profile
                            stream_info.rtsp_url = self._get_stream_uri(token)
                            stream_info.snapshot_url = self._get_snapshot_uri(token)
                            
                            profiles.append(stream_info)
        
        except Exception as e:
            logger.warning(f"GetProfiles failed: {e}")
        
        return profiles
    
    def _get_stream_uri(self, profile_token: str) -> str:
        """Get RTSP stream URI for profile."""
        media_url = self._media_service_url or f"http://{self.ip_address}:{self.port}/onvif/media"
        
        soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
    <s:Header>{self._build_security_header()}</s:Header>
    <s:Body>
        <trt:GetStreamUri>
            <trt:StreamSetup>
                <tt:Stream>RTP-Unicast</tt:Stream>
                <tt:Transport>
                    <tt:Protocol>RTSP</tt:Protocol>
                </tt:Transport>
            </trt:StreamSetup>
            <trt:ProfileToken>{profile_token}</trt:ProfileToken>
        </trt:GetStreamUri>
    </s:Body>
</s:Envelope>"""
        
        try:
            response = requests.post(
                media_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.content.decode('utf-8')
                import re
                match = re.search(r'<tt:Uri>([^<]+)</tt:Uri>', content)
                if match:
                    return match.group(1)
        except Exception as e:
            logger.debug(f"GetStreamUri failed: {e}")
        
        # Fallback to common RTSP URL format
        return f"rtsp://{self.username}:{self.password}@{self.ip_address}:554/stream1"
    
    def _get_snapshot_uri(self, profile_token: str) -> str:
        """Get snapshot URI for profile."""
        media_url = self._media_service_url or f"http://{self.ip_address}:{self.port}/onvif/media"
        
        soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
    <s:Header>{self._build_security_header()}</s:Header>
    <s:Body>
        <trt:GetSnapshotUri>
            <trt:ProfileToken>{profile_token}</trt:ProfileToken>
        </trt:GetSnapshotUri>
    </s:Body>
</s:Envelope>"""
        
        try:
            response = requests.post(
                media_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.content.decode('utf-8')
                import re
                match = re.search(r'<tt:Uri>([^<]+)</tt:Uri>', content)
                if match:
                    return match.group(1)
        except Exception as e:
            logger.debug(f"GetSnapshotUri failed: {e}")
        
        return f"http://{self.ip_address}/snapshot.jpg"
    
    def get_stream_url(self, profile: str = None) -> str:
        """
        Get RTSP stream URL.
        
        Args:
            profile: Profile token (uses first profile if None)
        
        Returns:
            RTSP URL string
        """
        if not self._profiles:
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:554/stream1"
        
        if profile:
            for p in self._profiles:
                if p.profile_token == profile or p.name == profile:
                    return p.rtsp_url
        
        return self._profiles[0].rtsp_url
    
    def capture_snapshot(self, save_path: str = None, profile: str = None) -> Optional[bytes]:
        """
        Capture snapshot from camera.
        
        Args:
            save_path: Path to save image (optional)
            profile: Profile token
        
        Returns:
            Image data as bytes, or None on failure
        """
        # Get snapshot URL
        snapshot_url = None
        if self._profiles:
            if profile:
                for p in self._profiles:
                    if p.profile_token == profile:
                        snapshot_url = p.snapshot_url
                        break
            else:
                snapshot_url = self._profiles[0].snapshot_url
        
        if not snapshot_url:
            snapshot_url = f"http://{self.ip_address}/snapshot.jpg"
        
        try:
            response = requests.get(
                snapshot_url,
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                image_data = response.content
                
                if save_path:
                    Path(save_path).write_bytes(image_data)
                    logger.info(f"Snapshot saved: {save_path}")
                
                return image_data
        
        except Exception as e:
            logger.error(f"Snapshot capture failed: {e}")
        
        return None
    
    # =========================================================================
    # PTZ Control
    # =========================================================================
    
    def ptz_move(
        self,
        pan: float = 0.0,
        tilt: float = 0.0,
        zoom: float = 0.0,
        speed: float = 0.5
    ) -> bool:
        """
        Move PTZ camera.
        
        Args:
            pan: Pan velocity (-1 to 1, negative=left, positive=right)
            tilt: Tilt velocity (-1 to 1, negative=down, positive=up)
            zoom: Zoom velocity (-1 to 1, negative=out, positive=in)
            speed: Movement speed (0 to 1)
        
        Returns:
            True if command sent successfully
        """
        if not self._info or not self._info.has_ptz:
            logger.warning("Camera does not support PTZ")
            return False
        
        ptz_url = self._ptz_service_url or f"http://{self.ip_address}:{self.port}/onvif/ptz"
        profile_token = self._profiles[0].profile_token if self._profiles else "profile1"
        
        soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
    <s:Header>{self._build_security_header()}</s:Header>
    <s:Body>
        <tptz:ContinuousMove>
            <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
            <tptz:Velocity>
                <tt:PanTilt x="{pan * speed}" y="{tilt * speed}"/>
                <tt:Zoom x="{zoom * speed}"/>
            </tptz:Velocity>
        </tptz:ContinuousMove>
    </s:Body>
</s:Envelope>"""
        
        try:
            response = requests.post(
                ptz_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"PTZ move failed: {e}")
            return False
    
    def ptz_stop(self) -> bool:
        """Stop PTZ movement."""
        ptz_url = self._ptz_service_url or f"http://{self.ip_address}:{self.port}/onvif/ptz"
        profile_token = self._profiles[0].profile_token if self._profiles else "profile1"
        
        soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
    <s:Header>{self._build_security_header()}</s:Header>
    <s:Body>
        <tptz:Stop>
            <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
            <tptz:PanTilt>true</tptz:PanTilt>
            <tptz:Zoom>true</tptz:Zoom>
        </tptz:Stop>
    </s:Body>
</s:Envelope>"""
        
        try:
            response = requests.post(
                ptz_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"PTZ stop failed: {e}")
            return False
    
    def ptz_go_to_preset(self, preset: str) -> bool:
        """Go to PTZ preset position."""
        ptz_url = self._ptz_service_url or f"http://{self.ip_address}:{self.port}/onvif/ptz"
        profile_token = self._profiles[0].profile_token if self._profiles else "profile1"
        
        soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
    <s:Header>{self._build_security_header()}</s:Header>
    <s:Body>
        <tptz:GotoPreset>
            <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
            <tptz:PresetToken>{preset}</tptz:PresetToken>
        </tptz:GotoPreset>
    </s:Body>
</s:Envelope>"""
        
        try:
            response = requests.post(
                ptz_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"PTZ goto preset failed: {e}")
            return False
    
    def ptz_home(self) -> bool:
        """Go to home position."""
        return self.ptz_go_to_preset("home")
    
    def get_ptz_status(self) -> PTZStatus:
        """Get current PTZ position."""
        ptz_url = self._ptz_service_url or f"http://{self.ip_address}:{self.port}/onvif/ptz"
        profile_token = self._profiles[0].profile_token if self._profiles else "profile1"
        
        soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
    <s:Header>{self._build_security_header()}</s:Header>
    <s:Body>
        <tptz:GetStatus>
            <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
        </tptz:GetStatus>
    </s:Body>
</s:Envelope>"""
        
        status = PTZStatus()
        
        try:
            response = requests.post(
                ptz_url,
                data=soap_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=(self.username, self.password),
                timeout=10
            )
            
            if response.status_code == 200:
                import re
                content = response.content.decode('utf-8')
                
                # Parse position
                pan_match = re.search(r'PanTilt[^>]*x="([^"]+)"', content)
                tilt_match = re.search(r'PanTilt[^>]*y="([^"]+)"', content)
                zoom_match = re.search(r'Zoom[^>]*x="([^"]+)"', content)
                
                if pan_match:
                    status.pan = float(pan_match.group(1))
                if tilt_match:
                    status.tilt = float(tilt_match.group(1))
                if zoom_match:
                    status.zoom = float(zoom_match.group(1))
        
        except Exception as e:
            logger.debug(f"GetStatus failed: {e}")
        
        self._ptz_status = status
        return status


# =============================================================================
# Camera Manager
# =============================================================================

class ONVIFCameraManager:
    """
    Manages multiple ONVIF cameras.
    
    Provides discovery, connection management, and centralized control.
    """
    
    def __init__(self):
        self._cameras: Dict[str, ONVIFCamera] = {}
        self._discovery = ONVIFDiscovery()
    
    @property
    def cameras(self) -> Dict[str, ONVIFCamera]:
        """Get connected cameras."""
        return self._cameras
    
    def discover(self, timeout: float = 3.0) -> List[DiscoveredCamera]:
        """
        Discover ONVIF cameras on network.
        
        Args:
            timeout: Discovery timeout
        
        Returns:
            List of discovered cameras
        """
        return self._discovery.discover(timeout)
    
    def connect(
        self,
        ip_address: str,
        username: str = "admin",
        password: str = "",
        port: int = 80
    ) -> Optional[ONVIFCamera]:
        """
        Connect to camera.
        
        Args:
            ip_address: Camera IP
            username: ONVIF username
            password: ONVIF password
            port: HTTP port
        
        Returns:
            ONVIFCamera instance if connected, None otherwise
        """
        camera = ONVIFCamera(ip_address, username, password, port)
        
        if camera.connect():
            self._cameras[ip_address] = camera
            return camera
        
        return None
    
    def disconnect(self, ip_address: str):
        """Disconnect camera by IP."""
        if ip_address in self._cameras:
            self._cameras[ip_address].disconnect()
            del self._cameras[ip_address]
    
    def disconnect_all(self):
        """Disconnect all cameras."""
        for camera in self._cameras.values():
            camera.disconnect()
        self._cameras.clear()
    
    def get_camera(self, ip_address: str) -> Optional[ONVIFCamera]:
        """Get connected camera by IP."""
        return self._cameras.get(ip_address)
    
    def get_all_cameras(self) -> List[ONVIFCamera]:
        """Get all connected cameras."""
        return list(self._cameras.values())
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all cameras."""
        return {
            ip: {
                "status": camera.status.value,
                "info": {
                    "manufacturer": camera.info.manufacturer if camera.info else "",
                    "model": camera.info.model if camera.info else "",
                } if camera.info else {},
                "profiles": len(camera.profiles),
                "has_ptz": camera.info.has_ptz if camera.info else False,
            }
            for ip, camera in self._cameras.items()
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ONVIF Camera Service")
    parser.add_argument("command", choices=["discover", "connect", "info", "snapshot", "ptz"])
    parser.add_argument("--ip", help="Camera IP address")
    parser.add_argument("--username", default="admin", help="ONVIF username")
    parser.add_argument("--password", default="", help="ONVIF password")
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--pan", type=float, default=0.0, help="PTZ pan (-1 to 1)")
    parser.add_argument("--tilt", type=float, default=0.0, help="PTZ tilt (-1 to 1)")
    parser.add_argument("--zoom", type=float, default=0.0, help="PTZ zoom (-1 to 1)")
    
    args = parser.parse_args()
    
    if args.command == "discover":
        discovery = ONVIFDiscovery()
        cameras = discovery.discover(args.timeout)
        print(f"Found {len(cameras)} camera(s):")
        for cam in cameras:
            print(f"  {cam.ip_address}: {cam.name}")
            print(f"    Manufacturer: {cam.manufacturer}")
            print(f"    XAddrs: {cam.xaddrs}")
    
    elif args.command == "connect":
        if not args.ip:
            print("Error: --ip required")
            return
        
        camera = ONVIFCamera(args.ip, args.username, args.password)
        if camera.connect():
            print(f"Connected to {args.ip}")
            print(f"  Profiles: {len(camera.profiles)}")
            camera.disconnect()
    
    elif args.command == "info":
        if not args.ip:
            print("Error: --ip required")
            return
        
        camera = ONVIFCamera(args.ip, args.username, args.password)
        if camera.connect():
            info = camera.info
            print(f"Camera: {args.ip}")
            print(f"  Manufacturer: {info.manufacturer}")
            print(f"  Model: {info.model}")
            print(f"  Firmware: {info.firmware_version}")
            print(f"  Serial: {info.serial_number}")
            print(f"  PTZ: {info.has_ptz}")
            print(f"  Audio: {info.has_audio}")
            print(f"\nProfiles:")
            for p in camera.profiles:
                print(f"  {p.name} ({p.profile_token})")
                print(f"    RTSP: {p.rtsp_url}")
            camera.disconnect()
    
    elif args.command == "snapshot":
        if not args.ip:
            print("Error: --ip required")
            return
        
        camera = ONVIFCamera(args.ip, args.username, args.password)
        if camera.connect():
            output = args.output or f"snapshot_{args.ip.replace('.', '_')}.jpg"
            data = camera.capture_snapshot(output)
            if data:
                print(f"Snapshot saved: {output} ({len(data)} bytes)")
            camera.disconnect()
    
    elif args.command == "ptz":
        if not args.ip:
            print("Error: --ip required")
            return
        
        camera = ONVIFCamera(args.ip, args.username, args.password)
        if camera.connect():
            if args.pan != 0 or args.tilt != 0 or args.zoom != 0:
                print(f"Moving PTZ: pan={args.pan}, tilt={args.tilt}, zoom={args.zoom}")
                camera.ptz_move(args.pan, args.tilt, args.zoom)
                time.sleep(2)
                camera.ptz_stop()
            else:
                status = camera.get_ptz_status()
                print(f"PTZ Status: pan={status.pan:.2f}, tilt={status.tilt:.2f}, zoom={status.zoom:.2f}")
            camera.disconnect()


if __name__ == "__main__":
    main()

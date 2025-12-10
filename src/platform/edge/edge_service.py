"""
Dynamical.ai Edge Device Microservice

FastAPI-based microservice running on edge devices (Jetson Orin) that handles:
- Device registration with platform
- Service lifecycle management
- Peripheral device discovery (DOGlove, cameras)
- Health reporting and heartbeat
- Remote configuration
- Troubleshooting and diagnostics

This service runs on the edge device and communicates with the central platform.
"""

import os
import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import SDK components
try:
    from .jetson_orin_sdk import (
        JetsonOrinSDK, EdgeDeviceConfig, HardwareInfo, SystemMetrics,
        NetworkInterface, ConnectedDevice, DeviceStatus, ServiceStatus
    )
except ImportError:
    from jetson_orin_sdk import (
        JetsonOrinSDK, EdgeDeviceConfig, HardwareInfo, SystemMetrics,
        NetworkInterface, ConnectedDevice, DeviceStatus, ServiceStatus
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# =============================================================================
# Configuration
# =============================================================================

class EdgeServiceConfig:
    """Edge service configuration."""
    SERVICE_PORT: int = int(os.getenv("EDGE_SERVICE_PORT", "8090"))
    HEARTBEAT_INTERVAL: int = 30  # seconds
    METRICS_INTERVAL: int = 5  # seconds
    DISCOVERY_INTERVAL: int = 60  # seconds
    PLATFORM_TIMEOUT: int = 10  # seconds


config = EdgeServiceConfig()


# =============================================================================
# Pydantic Models
# =============================================================================

class DeviceInfo(BaseModel):
    """Device information response."""
    device_id: str
    device_name: str
    model: str
    jetpack_version: str
    status: str
    uptime_seconds: float
    platform_connected: bool


class MetricsResponse(BaseModel):
    """System metrics response."""
    timestamp: str
    cpu_usage_percent: float
    gpu_usage_percent: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_cpu_c: float
    temperature_gpu_c: float
    power_usage_w: float


class NetworkInfo(BaseModel):
    """Network interface information."""
    name: str
    mac_address: str
    ip_address: Optional[str]
    is_up: bool
    is_wireless: bool
    ssid: Optional[str] = None
    signal_strength: Optional[int] = None


class PeripheralDevice(BaseModel):
    """Connected peripheral device."""
    device_type: str
    device_id: str
    name: str
    connection_type: str
    status: str
    port: Optional[str] = None
    ip_address: Optional[str] = None
    firmware_version: Optional[str] = None
    battery_level: Optional[int] = None
    last_seen: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ConfigureRequest(BaseModel):
    """Configuration update request."""
    device_name: Optional[str] = None
    site_id: Optional[str] = None
    platform_url: Optional[str] = None
    api_key: Optional[str] = None
    power_mode: Optional[str] = None
    enable_inference: Optional[bool] = None
    enable_recording: Optional[bool] = None


class WiFiConnectRequest(BaseModel):
    """WiFi connection request."""
    ssid: str
    password: str


class WiFiNetwork(BaseModel):
    """WiFi network info."""
    ssid: str
    signal: int
    security: str


class ServiceInfo(BaseModel):
    """Service status."""
    name: str
    status: str
    pid: Optional[int] = None
    uptime_seconds: Optional[float] = None
    memory_mb: Optional[float] = None


class DiagnosticResult(BaseModel):
    """Diagnostic test result."""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class PlatformRegistration(BaseModel):
    """Platform registration data."""
    platform_url: str
    api_key: str
    site_id: str
    organization_id: Optional[str] = None


# =============================================================================
# Peripheral Discovery
# =============================================================================

class PeripheralDiscovery:
    """Discovers and manages connected peripheral devices."""
    
    def __init__(self):
        self.devices: Dict[str, PeripheralDevice] = {}
        self._discovery_lock = threading.Lock()
    
    def discover_all(self) -> List[PeripheralDevice]:
        """Discover all connected peripherals."""
        with self._discovery_lock:
            self.devices.clear()
            
            # Discover DOGlove devices
            self._discover_doglove()
            
            # Discover ONVIF cameras
            self._discover_onvif_cameras()
            
            # Discover USB devices
            self._discover_usb_devices()
            
            # Discover Bluetooth devices
            self._discover_bluetooth()
            
            return list(self.devices.values())
    
    def _discover_doglove(self):
        """Discover DOGlove haptic gloves."""
        try:
            # Check for DOGlove USB devices
            import subprocess
            
            # Look for DOGlove USB vendor ID
            result = subprocess.run(
                ["lsusb"], capture_output=True, text=True
            )
            
            # DOGlove typically shows up as specific VID:PID
            # This is a placeholder - actual VID:PID would be from DOGlove specs
            doglove_patterns = ["DOGlove", "1234:5678", "Haptic"]
            
            for line in result.stdout.split("\n"):
                for pattern in doglove_patterns:
                    if pattern.lower() in line.lower():
                        device_id = f"doglove_{uuid.uuid4().hex[:8]}"
                        self.devices[device_id] = PeripheralDevice(
                            device_type="doglove",
                            device_id=device_id,
                            name="DOGlove Haptic Glove",
                            connection_type="usb",
                            status="connected",
                            port=self._extract_usb_port(line),
                            last_seen=datetime.utcnow().isoformat(),
                        )
                        break
            
            # Also check serial ports for DOGlove
            serial_ports = Path("/dev").glob("ttyUSB*")
            for port in serial_ports:
                try:
                    # Try to identify if this is a DOGlove
                    device_id = f"doglove_{port.name}"
                    if device_id not in self.devices:
                        # Mark as potential DOGlove - needs verification
                        self.devices[device_id] = PeripheralDevice(
                            device_type="doglove",
                            device_id=device_id,
                            name=f"DOGlove (unverified)",
                            connection_type="serial",
                            status="detected",
                            port=str(port),
                            last_seen=datetime.utcnow().isoformat(),
                        )
                except Exception as e:
                    logger.debug(f"Error checking serial port {port}: {e}")
            
            # Check Bluetooth for wireless DOGlove
            self._discover_doglove_bluetooth()
            
        except Exception as e:
            logger.error(f"DOGlove discovery failed: {e}")
    
    def _discover_doglove_bluetooth(self):
        """Discover DOGlove via Bluetooth."""
        try:
            import subprocess
            
            # Scan for Bluetooth devices
            result = subprocess.run(
                ["bluetoothctl", "devices"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "DOGlove" in line or "Haptic" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            mac = parts[1]
                            name = " ".join(parts[2:]) if len(parts) > 2 else "DOGlove"
                            device_id = f"doglove_bt_{mac.replace(':', '')}"
                            
                            self.devices[device_id] = PeripheralDevice(
                                device_type="doglove",
                                device_id=device_id,
                                name=name,
                                connection_type="bluetooth",
                                status="paired",
                                metadata={"mac_address": mac},
                                last_seen=datetime.utcnow().isoformat(),
                            )
        except Exception as e:
            logger.debug(f"Bluetooth DOGlove discovery failed: {e}")
    
    def _discover_onvif_cameras(self):
        """Discover ONVIF-compatible cameras on the network."""
        try:
            # WS-Discovery for ONVIF cameras
            import socket
            
            # ONVIF WS-Discovery multicast
            WS_DISCOVERY_MULTICAST = "239.255.255.250"
            WS_DISCOVERY_PORT = 3702
            
            # Discovery message
            discovery_msg = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
    <s:Header>
        <a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>
        <a:MessageID>uuid:""" + str(uuid.uuid4()) + """</a:MessageID>
        <a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>
    </s:Header>
    <s:Body>
        <d:Probe>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
        </d:Probe>
    </s:Body>
</s:Envelope>"""
            
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(3)
            
            # Send discovery probe
            sock.sendto(
                discovery_msg.encode(),
                (WS_DISCOVERY_MULTICAST, WS_DISCOVERY_PORT)
            )
            
            # Collect responses
            discovered_ips = set()
            try:
                while True:
                    data, addr = sock.recvfrom(65535)
                    if addr[0] not in discovered_ips:
                        discovered_ips.add(addr[0])
                        
                        # Parse response to get camera info
                        device_id = f"camera_onvif_{addr[0].replace('.', '_')}"
                        
                        # Extract device service address from response
                        response = data.decode('utf-8', errors='ignore')
                        xaddrs = self._extract_xaddrs(response)
                        
                        self.devices[device_id] = PeripheralDevice(
                            device_type="camera",
                            device_id=device_id,
                            name=f"ONVIF Camera ({addr[0]})",
                            connection_type="ethernet",
                            status="discovered",
                            ip_address=addr[0],
                            metadata={
                                "protocol": "onvif",
                                "xaddrs": xaddrs,
                            },
                            last_seen=datetime.utcnow().isoformat(),
                        )
            except socket.timeout:
                pass
            finally:
                sock.close()
            
            logger.info(f"Discovered {len(discovered_ips)} ONVIF cameras")
            
        except Exception as e:
            logger.error(f"ONVIF camera discovery failed: {e}")
    
    def _extract_xaddrs(self, response: str) -> List[str]:
        """Extract XAddrs from WS-Discovery response."""
        xaddrs = []
        try:
            import re
            matches = re.findall(r'<[^>]*XAddrs[^>]*>([^<]+)</[^>]*XAddrs>', response)
            for match in matches:
                xaddrs.extend(match.split())
        except:
            pass
        return xaddrs
    
    def _discover_usb_devices(self):
        """Discover USB devices."""
        try:
            import subprocess
            
            result = subprocess.run(
                ["lsusb", "-v"],
                capture_output=True, text=True
            )
            
            # Parse USB devices - look for known device types
            known_devices = {
                "webcam": ["Camera", "Webcam", "Video"],
                "sensor": ["IMU", "Accelerometer", "Gyro"],
            }
            
            current_device = None
            for line in result.stdout.split("\n"):
                if "Bus" in line and "Device" in line:
                    # Start of new device
                    for dev_type, patterns in known_devices.items():
                        for pattern in patterns:
                            if pattern.lower() in line.lower():
                                device_id = f"{dev_type}_usb_{uuid.uuid4().hex[:8]}"
                                self.devices[device_id] = PeripheralDevice(
                                    device_type=dev_type,
                                    device_id=device_id,
                                    name=line.split(":")[-1].strip() if ":" in line else dev_type,
                                    connection_type="usb",
                                    status="connected",
                                    last_seen=datetime.utcnow().isoformat(),
                                )
                                break
                                
        except Exception as e:
            logger.debug(f"USB discovery failed: {e}")
    
    def _discover_bluetooth(self):
        """Discover Bluetooth devices."""
        try:
            import subprocess
            
            result = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            mac = parts[1] if len(parts) > 1 else ""
                            name = " ".join(parts[2:]) if len(parts) > 2 else "Bluetooth Device"
                            
                            device_id = f"bt_{mac.replace(':', '')}"
                            if device_id not in self.devices:
                                self.devices[device_id] = PeripheralDevice(
                                    device_type="bluetooth",
                                    device_id=device_id,
                                    name=name,
                                    connection_type="bluetooth",
                                    status="connected",
                                    metadata={"mac_address": mac},
                                    last_seen=datetime.utcnow().isoformat(),
                                )
        except Exception as e:
            logger.debug(f"Bluetooth discovery failed: {e}")
    
    def _extract_usb_port(self, lsusb_line: str) -> Optional[str]:
        """Extract USB port from lsusb output."""
        try:
            parts = lsusb_line.split()
            if len(parts) >= 4:
                bus = parts[1]
                device = parts[3].rstrip(":")
                return f"/dev/bus/usb/{bus}/{device}"
        except:
            pass
        return None
    
    def get_device(self, device_id: str) -> Optional[PeripheralDevice]:
        """Get specific device by ID."""
        return self.devices.get(device_id)
    
    def get_devices_by_type(self, device_type: str) -> List[PeripheralDevice]:
        """Get devices by type."""
        return [d for d in self.devices.values() if d.device_type == device_type]


# =============================================================================
# Service Manager
# =============================================================================

class EdgeServiceManager:
    """Manages edge services (inference, recording, etc.)."""
    
    def __init__(self):
        self.services: Dict[str, Dict] = {
            "inference": {"status": ServiceStatus.STOPPED, "pid": None, "process": None},
            "recording": {"status": ServiceStatus.STOPPED, "pid": None, "process": None},
            "streaming": {"status": ServiceStatus.STOPPED, "pid": None, "process": None},
        }
    
    def get_service_status(self, service_name: str) -> ServiceInfo:
        """Get service status."""
        svc = self.services.get(service_name, {})
        return ServiceInfo(
            name=service_name,
            status=svc.get("status", ServiceStatus.STOPPED).value if isinstance(svc.get("status"), ServiceStatus) else str(svc.get("status", "unknown")),
            pid=svc.get("pid"),
        )
    
    def get_all_services(self) -> List[ServiceInfo]:
        """Get all service statuses."""
        return [self.get_service_status(name) for name in self.services]
    
    def start_service(self, service_name: str) -> bool:
        """Start a service."""
        if service_name not in self.services:
            return False
        
        # Placeholder - actual implementation would start real processes
        self.services[service_name]["status"] = ServiceStatus.RUNNING
        self.services[service_name]["pid"] = os.getpid()  # Placeholder
        return True
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a service."""
        if service_name not in self.services:
            return False
        
        self.services[service_name]["status"] = ServiceStatus.STOPPED
        self.services[service_name]["pid"] = None
        return True
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a service."""
        self.stop_service(service_name)
        return self.start_service(service_name)


# =============================================================================
# Platform Connector
# =============================================================================

class PlatformConnector:
    """Handles communication with central platform."""
    
    def __init__(self, sdk: JetsonOrinSDK):
        self.sdk = sdk
        self.connected = False
        self.last_heartbeat = None
        self._heartbeat_task = None
        self._stop_event = threading.Event()
    
    async def register(self) -> bool:
        """Register device with platform."""
        if not self.sdk.config.platform_url or not self.sdk.config.api_key:
            logger.warning("Platform URL or API key not configured")
            return False
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=config.PLATFORM_TIMEOUT) as client:
                # Register device
                response = await client.post(
                    f"{self.sdk.config.platform_url}/api/v1/robots",
                    headers={"Authorization": f"Bearer {self.sdk.config.api_key}"},
                    json={
                        "name": self.sdk.config.device_name or self.sdk.config.device_id,
                        "site_id": self.sdk.config.site_id,
                        "robot_type": f"jetson_{self.sdk.hardware_info.model.value}",
                        "hardware_config": asdict(self.sdk.hardware_info),
                        "software_version": __version__,
                    }
                )
                
                if response.status_code in [200, 201]:
                    self.connected = True
                    logger.info("Successfully registered with platform")
                    return True
                else:
                    logger.error(f"Registration failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Platform registration failed: {e}")
        
        return False
    
    async def send_heartbeat(self) -> bool:
        """Send heartbeat to platform."""
        if not self.connected:
            return False
        
        try:
            import httpx
            
            metrics = self.sdk.get_metrics()
            
            async with httpx.AsyncClient(timeout=config.PLATFORM_TIMEOUT) as client:
                response = await client.post(
                    f"{self.sdk.config.platform_url}/api/v1/robots/{self.sdk.config.device_id}/heartbeat",
                    headers={"Authorization": f"Bearer {self.sdk.config.api_key}"},
                    json={
                        "status": "online",
                        "metrics": asdict(metrics),
                    }
                )
                
                if response.status_code == 200:
                    self.last_heartbeat = datetime.utcnow().isoformat()
                    return True
                    
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
        
        return False
    
    def start_heartbeat_loop(self):
        """Start background heartbeat loop."""
        async def heartbeat_loop():
            while not self._stop_event.is_set():
                await self.send_heartbeat()
                await asyncio.sleep(config.HEARTBEAT_INTERVAL)
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    def stop_heartbeat_loop(self):
        """Stop heartbeat loop."""
        self._stop_event.set()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()


# =============================================================================
# Diagnostics
# =============================================================================

class DiagnosticsRunner:
    """Runs diagnostic tests on the edge device."""
    
    def __init__(self, sdk: JetsonOrinSDK):
        self.sdk = sdk
    
    async def run_all(self) -> List[DiagnosticResult]:
        """Run all diagnostic tests."""
        results = []
        
        results.append(await self.test_gpu())
        results.append(await self.test_network())
        results.append(await self.test_storage())
        results.append(await self.test_memory())
        results.append(await self.test_platform_connectivity())
        results.append(await self.test_peripherals())
        
        return results
    
    async def test_gpu(self) -> DiagnosticResult:
        """Test GPU availability and CUDA."""
        try:
            import subprocess
            
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return DiagnosticResult(
                    test_name="GPU",
                    passed=True,
                    message="GPU detected and working",
                    details={"cuda_version": self.sdk.hardware_info.cuda_version}
                )
            else:
                return DiagnosticResult(
                    test_name="GPU",
                    passed=False,
                    message="nvidia-smi failed",
                    details={"error": result.stderr}
                )
        except Exception as e:
            return DiagnosticResult(
                test_name="GPU",
                passed=False,
                message=f"GPU test failed: {e}"
            )
    
    async def test_network(self) -> DiagnosticResult:
        """Test network connectivity."""
        try:
            interfaces = self.sdk.get_network_interfaces()
            connected = [i for i in interfaces if i.is_up and i.ip_address]
            
            if connected:
                return DiagnosticResult(
                    test_name="Network",
                    passed=True,
                    message=f"{len(connected)} interface(s) connected",
                    details={"interfaces": [asdict(i) for i in connected]}
                )
            else:
                return DiagnosticResult(
                    test_name="Network",
                    passed=False,
                    message="No network interfaces connected"
                )
        except Exception as e:
            return DiagnosticResult(
                test_name="Network",
                passed=False,
                message=f"Network test failed: {e}"
            )
    
    async def test_storage(self) -> DiagnosticResult:
        """Test storage availability."""
        try:
            metrics = self.sdk.get_metrics()
            free_gb = metrics.disk_total_gb - metrics.disk_used_gb
            
            if free_gb > 5:
                return DiagnosticResult(
                    test_name="Storage",
                    passed=True,
                    message=f"{free_gb:.1f} GB free",
                    details={
                        "total_gb": metrics.disk_total_gb,
                        "used_gb": metrics.disk_used_gb,
                        "free_gb": free_gb
                    }
                )
            else:
                return DiagnosticResult(
                    test_name="Storage",
                    passed=False,
                    message=f"Low disk space: {free_gb:.1f} GB free"
                )
        except Exception as e:
            return DiagnosticResult(
                test_name="Storage",
                passed=False,
                message=f"Storage test failed: {e}"
            )
    
    async def test_memory(self) -> DiagnosticResult:
        """Test memory availability."""
        try:
            metrics = self.sdk.get_metrics()
            free_mb = metrics.memory_total_mb - metrics.memory_used_mb
            usage_percent = (metrics.memory_used_mb / metrics.memory_total_mb) * 100
            
            if usage_percent < 90:
                return DiagnosticResult(
                    test_name="Memory",
                    passed=True,
                    message=f"{free_mb} MB free ({100-usage_percent:.1f}%)",
                    details={
                        "total_mb": metrics.memory_total_mb,
                        "used_mb": metrics.memory_used_mb,
                        "free_mb": free_mb
                    }
                )
            else:
                return DiagnosticResult(
                    test_name="Memory",
                    passed=False,
                    message=f"High memory usage: {usage_percent:.1f}%"
                )
        except Exception as e:
            return DiagnosticResult(
                test_name="Memory",
                passed=False,
                message=f"Memory test failed: {e}"
            )
    
    async def test_platform_connectivity(self) -> DiagnosticResult:
        """Test connectivity to platform."""
        if not self.sdk.config.platform_url:
            return DiagnosticResult(
                test_name="Platform Connectivity",
                passed=False,
                message="Platform URL not configured"
            )
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.sdk.config.platform_url}/health")
                
                if response.status_code == 200:
                    return DiagnosticResult(
                        test_name="Platform Connectivity",
                        passed=True,
                        message="Platform reachable",
                        details={"url": self.sdk.config.platform_url}
                    )
                else:
                    return DiagnosticResult(
                        test_name="Platform Connectivity",
                        passed=False,
                        message=f"Platform returned {response.status_code}"
                    )
        except Exception as e:
            return DiagnosticResult(
                test_name="Platform Connectivity",
                passed=False,
                message=f"Cannot reach platform: {e}"
            )
    
    async def test_peripherals(self) -> DiagnosticResult:
        """Test peripheral discovery."""
        try:
            discovery = PeripheralDiscovery()
            devices = discovery.discover_all()
            
            return DiagnosticResult(
                test_name="Peripherals",
                passed=True,
                message=f"{len(devices)} device(s) found",
                details={
                    "devices": [d.dict() for d in devices]
                }
            )
        except Exception as e:
            return DiagnosticResult(
                test_name="Peripherals",
                passed=False,
                message=f"Peripheral test failed: {e}"
            )


# =============================================================================
# FastAPI Application
# =============================================================================

# Initialize components
sdk = JetsonOrinSDK()
peripheral_discovery = PeripheralDiscovery()
service_manager = EdgeServiceManager()
platform_connector = PlatformConnector(sdk)
diagnostics = DiagnosticsRunner(sdk)

# Create FastAPI app
app = FastAPI(
    title="Dynamical.ai Edge Service",
    description="Edge device management service for Jetson Orin",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track start time for uptime
start_time = time.time()

# WebSocket connections for real-time updates
websocket_connections: List[WebSocket] = []


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "version": __version__}


@app.get("/device", response_model=DeviceInfo)
async def get_device_info():
    """Get device information."""
    return DeviceInfo(
        device_id=sdk.config.device_id,
        device_name=sdk.config.device_name or sdk.config.device_id,
        model=sdk.hardware_info.model.value,
        jetpack_version=sdk.hardware_info.jetpack_version,
        status=DeviceStatus.RUNNING.value,
        uptime_seconds=time.time() - start_time,
        platform_connected=platform_connector.connected,
    )


@app.get("/device/status")
async def get_device_status():
    """Get comprehensive device status."""
    return sdk.get_status()


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current system metrics."""
    m = sdk.get_metrics()
    return MetricsResponse(
        timestamp=m.timestamp,
        cpu_usage_percent=m.cpu_usage_percent,
        gpu_usage_percent=m.gpu_usage_percent,
        memory_used_mb=m.memory_used_mb,
        memory_total_mb=m.memory_total_mb,
        temperature_cpu_c=m.temperature_cpu_c,
        temperature_gpu_c=m.temperature_gpu_c,
        power_usage_w=m.power_usage_w,
    )


@app.get("/network", response_model=List[NetworkInfo])
async def get_network_interfaces():
    """Get network interfaces."""
    interfaces = sdk.get_network_interfaces()
    return [
        NetworkInfo(
            name=i.name,
            mac_address=i.mac_address,
            ip_address=i.ip_address,
            is_up=i.is_up,
            is_wireless=i.is_wireless,
            ssid=i.ssid,
            signal_strength=i.signal_strength,
        )
        for i in interfaces
    ]


@app.get("/network/wifi/scan", response_model=List[WiFiNetwork])
async def scan_wifi():
    """Scan for WiFi networks."""
    networks = sdk.scan_wifi()
    return [WiFiNetwork(**n) for n in networks]


@app.post("/network/wifi/connect")
async def connect_wifi(request: WiFiConnectRequest):
    """Connect to WiFi network."""
    if sdk.connect_wifi(request.ssid, request.password):
        return {"status": "connected", "ssid": request.ssid}
    raise HTTPException(status_code=400, detail="Failed to connect to WiFi")


# =============================================================================
# Peripheral Endpoints
# =============================================================================

@app.get("/peripherals", response_model=List[PeripheralDevice])
async def list_peripherals():
    """List all discovered peripherals."""
    return peripheral_discovery.discover_all()


@app.post("/peripherals/discover")
async def discover_peripherals():
    """Trigger peripheral discovery."""
    devices = peripheral_discovery.discover_all()
    return {"count": len(devices), "devices": [d.dict() for d in devices]}


@app.get("/peripherals/{device_id}", response_model=PeripheralDevice)
async def get_peripheral(device_id: str):
    """Get specific peripheral device."""
    device = peripheral_discovery.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@app.get("/peripherals/type/{device_type}", response_model=List[PeripheralDevice])
async def get_peripherals_by_type(device_type: str):
    """Get peripherals by type (doglove, camera, etc.)."""
    return peripheral_discovery.get_devices_by_type(device_type)


# =============================================================================
# Configuration Endpoints
# =============================================================================

@app.get("/config")
async def get_config():
    """Get device configuration."""
    return {
        "device_id": sdk.config.device_id,
        "device_name": sdk.config.device_name,
        "site_id": sdk.config.site_id,
        "organization_id": sdk.config.organization_id,
        "platform_url": sdk.config.platform_url,
        "power_mode": sdk.get_power_mode(),
        "enable_inference": sdk.config.enable_inference,
        "enable_recording": sdk.config.enable_recording,
        "enable_streaming": sdk.config.enable_streaming,
    }


@app.post("/config")
async def update_config(request: ConfigureRequest):
    """Update device configuration."""
    sdk.configure(
        device_name=request.device_name,
        site_id=request.site_id,
        platform_url=request.platform_url,
        api_key=request.api_key,
    )
    
    if request.power_mode:
        sdk.set_power_mode(request.power_mode)
    
    if request.enable_inference is not None:
        sdk.config.enable_inference = request.enable_inference
    
    if request.enable_recording is not None:
        sdk.config.enable_recording = request.enable_recording
    
    sdk.config.save()
    return {"status": "updated"}


@app.post("/platform/register")
async def register_with_platform(request: PlatformRegistration):
    """Register device with central platform."""
    sdk.configure(
        platform_url=request.platform_url,
        api_key=request.api_key,
        site_id=request.site_id,
        organization_id=request.organization_id,
    )
    
    if await platform_connector.register():
        return {"status": "registered", "device_id": sdk.config.device_id}
    raise HTTPException(status_code=400, detail="Registration failed")


# =============================================================================
# Service Endpoints
# =============================================================================

@app.get("/services", response_model=List[ServiceInfo])
async def list_services():
    """List all services."""
    return service_manager.get_all_services()


@app.post("/services/{service_name}/start")
async def start_service(service_name: str):
    """Start a service."""
    if service_manager.start_service(service_name):
        return {"status": "started", "service": service_name}
    raise HTTPException(status_code=400, detail="Failed to start service")


@app.post("/services/{service_name}/stop")
async def stop_service(service_name: str):
    """Stop a service."""
    if service_manager.stop_service(service_name):
        return {"status": "stopped", "service": service_name}
    raise HTTPException(status_code=400, detail="Failed to stop service")


@app.post("/services/{service_name}/restart")
async def restart_service(service_name: str):
    """Restart a service."""
    if service_manager.restart_service(service_name):
        return {"status": "restarted", "service": service_name}
    raise HTTPException(status_code=400, detail="Failed to restart service")


# =============================================================================
# Diagnostics Endpoints
# =============================================================================

@app.get("/diagnostics", response_model=List[DiagnosticResult])
async def run_diagnostics():
    """Run all diagnostic tests."""
    return await diagnostics.run_all()


@app.get("/diagnostics/{test_name}", response_model=DiagnosticResult)
async def run_diagnostic(test_name: str):
    """Run specific diagnostic test."""
    test_map = {
        "gpu": diagnostics.test_gpu,
        "network": diagnostics.test_network,
        "storage": diagnostics.test_storage,
        "memory": diagnostics.test_memory,
        "platform": diagnostics.test_platform_connectivity,
        "peripherals": diagnostics.test_peripherals,
    }
    
    if test_name not in test_map:
        raise HTTPException(status_code=404, detail="Test not found")
    
    return await test_map[test_name]()


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            metrics = sdk.get_metrics()
            await websocket.send_json(asdict(metrics))
            await asyncio.sleep(config.METRICS_INTERVAL)
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


@app.websocket("/ws/peripherals")
async def websocket_peripherals(websocket: WebSocket):
    """WebSocket endpoint for peripheral updates."""
    await websocket.accept()
    
    try:
        while True:
            devices = peripheral_discovery.discover_all()
            await websocket.send_json([d.dict() for d in devices])
            await asyncio.sleep(config.DISCOVERY_INTERVAL)
    except WebSocketDisconnect:
        pass


# =============================================================================
# Startup/Shutdown
# =============================================================================

@app.on_event("startup")
async def startup():
    """Startup tasks."""
    logger.info(f"Edge service starting on device: {sdk.config.device_id}")
    logger.info(f"Hardware: {sdk.hardware_info.model.value}")
    
    # Try to register with platform
    if sdk.config.platform_url and sdk.config.api_key:
        await platform_connector.register()
        platform_connector.start_heartbeat_loop()
    
    # Initial peripheral discovery
    peripheral_discovery.discover_all()


@app.on_event("shutdown")
async def shutdown():
    """Shutdown tasks."""
    logger.info("Edge service shutting down")
    platform_connector.stop_heartbeat_loop()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the edge service."""
    uvicorn.run(
        "edge_service:app",
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()

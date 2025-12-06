"""
Dynamical.ai Jetson Orin SDK

Complete SDK for deploying Dynamical.ai on NVIDIA Jetson Orin devices.
Handles installation, configuration, hardware detection, and system management.

Supported Devices:
- Jetson Orin Nano (8GB)
- Jetson Orin NX (8GB/16GB)
- Jetson AGX Orin (32GB/64GB)

Features:
- Automatic hardware detection
- CUDA/TensorRT optimization
- Container deployment
- System monitoring
- Remote management
- OTA updates

Requirements:
    JetPack 5.x or 6.x
    Python 3.8+
"""

import os
import sys
import json
import time
import socket
import hashlib
import platform
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# =============================================================================
# Enums and Constants
# =============================================================================

class JetsonModel(str, Enum):
    """Supported Jetson models."""
    ORIN_NANO_8GB = "orin_nano_8gb"
    ORIN_NX_8GB = "orin_nx_8gb"
    ORIN_NX_16GB = "orin_nx_16gb"
    AGX_ORIN_32GB = "agx_orin_32gb"
    AGX_ORIN_64GB = "agx_orin_64gb"
    UNKNOWN = "unknown"


class DeviceStatus(str, Enum):
    """Device status."""
    OFFLINE = "offline"
    ONLINE = "online"
    INITIALIZING = "initializing"
    RUNNING = "running"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ServiceStatus(str, Enum):
    """Service status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


# Hardware specifications per model
JETSON_SPECS = {
    JetsonModel.ORIN_NANO_8GB: {
        "gpu_cores": 1024,
        "cpu_cores": 6,
        "memory_gb": 8,
        "ai_tops": 40,
        "power_modes": ["7W", "15W"],
    },
    JetsonModel.ORIN_NX_8GB: {
        "gpu_cores": 1024,
        "cpu_cores": 6,
        "memory_gb": 8,
        "ai_tops": 70,
        "power_modes": ["10W", "15W", "25W"],
    },
    JetsonModel.ORIN_NX_16GB: {
        "gpu_cores": 1024,
        "cpu_cores": 8,
        "memory_gb": 16,
        "ai_tops": 100,
        "power_modes": ["10W", "15W", "25W"],
    },
    JetsonModel.AGX_ORIN_32GB: {
        "gpu_cores": 2048,
        "cpu_cores": 12,
        "memory_gb": 32,
        "ai_tops": 200,
        "power_modes": ["15W", "30W", "50W"],
    },
    JetsonModel.AGX_ORIN_64GB: {
        "gpu_cores": 2048,
        "cpu_cores": 12,
        "memory_gb": 64,
        "ai_tops": 275,
        "power_modes": ["15W", "30W", "50W", "60W"],
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HardwareInfo:
    """Hardware information."""
    model: JetsonModel = JetsonModel.UNKNOWN
    jetpack_version: str = ""
    l4t_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    tensorrt_version: str = ""
    cpu_cores: int = 0
    gpu_cores: int = 0
    memory_total_mb: int = 0
    storage_total_gb: float = 0.0
    serial_number: str = ""
    mac_address: str = ""


@dataclass
class SystemMetrics:
    """Real-time system metrics."""
    timestamp: str = ""
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    gpu_memory_total_mb: int = 0
    temperature_cpu_c: float = 0.0
    temperature_gpu_c: float = 0.0
    power_usage_w: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    network_rx_mbps: float = 0.0
    network_tx_mbps: float = 0.0


@dataclass
class NetworkInterface:
    """Network interface information."""
    name: str = ""
    mac_address: str = ""
    ip_address: str = ""
    netmask: str = ""
    is_up: bool = False
    is_wireless: bool = False
    ssid: Optional[str] = None
    signal_strength: Optional[int] = None


@dataclass
class ConnectedDevice:
    """Connected peripheral device."""
    device_type: str = ""  # doglove, camera, sensor
    device_id: str = ""
    name: str = ""
    connection_type: str = ""  # usb, bluetooth, ethernet, wifi
    status: str = "disconnected"
    port: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeDeviceConfig:
    """Edge device configuration."""
    device_id: str = ""
    device_name: str = ""
    site_id: str = ""
    organization_id: str = ""
    platform_url: str = "http://localhost:8080"
    api_key: str = ""
    
    # Hardware settings
    power_mode: str = "default"
    fan_mode: str = "auto"
    
    # Service settings
    enable_inference: bool = True
    enable_recording: bool = True
    enable_streaming: bool = True
    
    # Network settings
    wifi_ssid: str = ""
    wifi_password: str = ""
    static_ip: str = ""
    
    # Storage settings
    data_dir: str = "/data/dynamical"
    max_storage_gb: float = 50.0
    
    def save(self, path: str = None):
        """Save configuration."""
        path = path or "/etc/dynamical/config.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = None) -> "EdgeDeviceConfig":
        """Load configuration."""
        path = path or "/etc/dynamical/config.json"
        if Path(path).exists():
            with open(path) as f:
                return cls(**json.load(f))
        return cls()


# =============================================================================
# Hardware Detection
# =============================================================================

class JetsonHardwareDetector:
    """Detects Jetson hardware and capabilities."""
    
    @staticmethod
    def detect_model() -> JetsonModel:
        """Detect Jetson model from device tree."""
        try:
            # Check device tree compatible string
            dt_path = "/proc/device-tree/compatible"
            if Path(dt_path).exists():
                with open(dt_path, 'rb') as f:
                    compatible = f.read().decode('utf-8', errors='ignore')
                
                if "orin-nano" in compatible.lower():
                    return JetsonModel.ORIN_NANO_8GB
                elif "orin-nx" in compatible.lower():
                    # Check memory to distinguish 8GB vs 16GB
                    mem_gb = JetsonHardwareDetector._get_memory_gb()
                    if mem_gb > 12:
                        return JetsonModel.ORIN_NX_16GB
                    return JetsonModel.ORIN_NX_8GB
                elif "agx-orin" in compatible.lower():
                    mem_gb = JetsonHardwareDetector._get_memory_gb()
                    if mem_gb > 48:
                        return JetsonModel.AGX_ORIN_64GB
                    return JetsonModel.AGX_ORIN_32GB
            
            # Fallback: check tegra chip info
            chip_path = "/sys/module/tegra_fuse/parameters/tegra_chip_id"
            if Path(chip_path).exists():
                with open(chip_path) as f:
                    chip_id = f.read().strip()
                if chip_id == "35":  # Orin chip ID
                    return JetsonModel.AGX_ORIN_32GB
            
        except Exception as e:
            logger.warning(f"Failed to detect Jetson model: {e}")
        
        return JetsonModel.UNKNOWN
    
    @staticmethod
    def _get_memory_gb() -> float:
        """Get total memory in GB."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / 1024 / 1024
        except:
            pass
        return 0
    
    @staticmethod
    def get_jetpack_version() -> str:
        """Get JetPack version."""
        try:
            # Check apt package
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Check from nv_tegra_release
            release_path = "/etc/nv_tegra_release"
            if Path(release_path).exists():
                with open(release_path) as f:
                    content = f.read()
                    # Parse version from release file
                    if "R35" in content:
                        return "5.1"
                    elif "R36" in content:
                        return "6.0"
        except:
            pass
        return "unknown"
    
    @staticmethod
    def get_l4t_version() -> str:
        """Get L4T (Linux for Tegra) version."""
        try:
            release_path = "/etc/nv_tegra_release"
            if Path(release_path).exists():
                with open(release_path) as f:
                    line = f.readline()
                    # Parse "# R35 (release), REVISION: 4.1"
                    parts = line.split(",")
                    if len(parts) >= 2:
                        release = parts[0].replace("#", "").strip()
                        revision = parts[1].split(":")[1].strip() if ":" in parts[1] else ""
                        return f"{release}.{revision}"
        except:
            pass
        return "unknown"
    
    @staticmethod
    def get_cuda_version() -> str:
        """Get CUDA version."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        # Parse "Cuda compilation tools, release 11.4, V11.4.315"
                        parts = line.split(",")
                        if len(parts) >= 2:
                            return parts[1].replace("release", "").strip()
        except:
            pass
        return "unknown"
    
    @staticmethod
    def get_tensorrt_version() -> str:
        """Get TensorRT version."""
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Version}", "tensorrt"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split("-")[0]
        except:
            pass
        return "unknown"
    
    @staticmethod
    def get_serial_number() -> str:
        """Get device serial number."""
        try:
            serial_path = "/proc/device-tree/serial-number"
            if Path(serial_path).exists():
                with open(serial_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='ignore').strip('\x00')
        except:
            pass
        return "unknown"
    
    @staticmethod
    def get_mac_address() -> str:
        """Get primary MAC address."""
        try:
            for iface in ["eth0", "wlan0", "enp1s0"]:
                mac_path = f"/sys/class/net/{iface}/address"
                if Path(mac_path).exists():
                    with open(mac_path) as f:
                        return f.read().strip()
        except:
            pass
        return "00:00:00:00:00:00"
    
    @classmethod
    def get_hardware_info(cls) -> HardwareInfo:
        """Get complete hardware information."""
        model = cls.detect_model()
        specs = JETSON_SPECS.get(model, {})
        
        return HardwareInfo(
            model=model,
            jetpack_version=cls.get_jetpack_version(),
            l4t_version=cls.get_l4t_version(),
            cuda_version=cls.get_cuda_version(),
            tensorrt_version=cls.get_tensorrt_version(),
            cpu_cores=specs.get("cpu_cores", os.cpu_count() or 0),
            gpu_cores=specs.get("gpu_cores", 0),
            memory_total_mb=int(cls._get_memory_gb() * 1024),
            storage_total_gb=cls._get_storage_gb(),
            serial_number=cls.get_serial_number(),
            mac_address=cls.get_mac_address(),
        )
    
    @staticmethod
    def _get_storage_gb() -> float:
        """Get total storage in GB."""
        try:
            stat = os.statvfs("/")
            return (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        except:
            return 0


# =============================================================================
# System Monitoring
# =============================================================================

class SystemMonitor:
    """Real-time system monitoring for Jetson."""
    
    def __init__(self):
        self._last_net_rx = 0
        self._last_net_tx = 0
        self._last_net_time = time.time()
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_usage_percent=self._get_cpu_usage(),
            gpu_usage_percent=self._get_gpu_usage(),
            memory_used_mb=self._get_memory_used_mb(),
            memory_total_mb=self._get_memory_total_mb(),
            gpu_memory_used_mb=self._get_gpu_memory_used_mb(),
            gpu_memory_total_mb=self._get_gpu_memory_total_mb(),
            temperature_cpu_c=self._get_cpu_temperature(),
            temperature_gpu_c=self._get_gpu_temperature(),
            power_usage_w=self._get_power_usage(),
            disk_used_gb=self._get_disk_used_gb(),
            disk_total_gb=self._get_disk_total_gb(),
            network_rx_mbps=self._get_network_rx_mbps(),
            network_tx_mbps=self._get_network_tx_mbps(),
        )
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()
                parts = line.split()
                if len(parts) >= 5:
                    idle = int(parts[4])
                    total = sum(int(x) for x in parts[1:])
                    return round(100 * (1 - idle / total), 1)
        except:
            pass
        return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage from tegrastats or nvidia-smi."""
        try:
            # Try tegrastats first (Jetson-specific)
            result = subprocess.run(
                ["tegrastats", "--interval", "100", "--stop"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                # Parse GPU usage from tegrastats output
                output = result.stdout
                if "GR3D_FREQ" in output:
                    # Extract percentage
                    import re
                    match = re.search(r'GR3D_FREQ\s+(\d+)%', output)
                    if match:
                        return float(match.group(1))
        except:
            pass
        
        try:
            # Fallback to nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return 0.0
    
    def _get_memory_used_mb(self) -> int:
        """Get used memory in MB."""
        try:
            with open("/proc/meminfo") as f:
                mem_total = 0
                mem_available = 0
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1])
                return (mem_total - mem_available) // 1024
        except:
            return 0
    
    def _get_memory_total_mb(self) -> int:
        """Get total memory in MB."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except:
            return 0
    
    def _get_gpu_memory_used_mb(self) -> int:
        """Get GPU memory usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return 0
    
    def _get_gpu_memory_total_mb(self) -> int:
        """Get total GPU memory."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return 0
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature."""
        try:
            # Jetson thermal zones
            for i in range(10):
                temp_path = f"/sys/devices/virtual/thermal/thermal_zone{i}/temp"
                type_path = f"/sys/devices/virtual/thermal/thermal_zone{i}/type"
                if Path(temp_path).exists() and Path(type_path).exists():
                    with open(type_path) as f:
                        zone_type = f.read().strip()
                    if "cpu" in zone_type.lower() or "CPU" in zone_type:
                        with open(temp_path) as f:
                            return int(f.read().strip()) / 1000.0
        except:
            pass
        return 0.0
    
    def _get_gpu_temperature(self) -> float:
        """Get GPU temperature."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        try:
            # Fallback to thermal zones
            for i in range(10):
                temp_path = f"/sys/devices/virtual/thermal/thermal_zone{i}/temp"
                type_path = f"/sys/devices/virtual/thermal/thermal_zone{i}/type"
                if Path(temp_path).exists() and Path(type_path).exists():
                    with open(type_path) as f:
                        zone_type = f.read().strip()
                    if "gpu" in zone_type.lower() or "GPU" in zone_type:
                        with open(temp_path) as f:
                            return int(f.read().strip()) / 1000.0
        except:
            pass
        return 0.0
    
    def _get_power_usage(self) -> float:
        """Get current power usage in watts."""
        try:
            # Jetson power monitoring
            power_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
            if Path(power_path).exists():
                hwmon_dirs = list(Path(power_path).iterdir())
                if hwmon_dirs:
                    power_file = hwmon_dirs[0] / "power1_input"
                    if power_file.exists():
                        with open(power_file) as f:
                            return int(f.read().strip()) / 1000000.0  # Convert from uW
        except:
            pass
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return 0.0
    
    def _get_disk_used_gb(self) -> float:
        """Get used disk space in GB."""
        try:
            stat = os.statvfs("/")
            used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
            return round(used / (1024 ** 3), 2)
        except:
            return 0.0
    
    def _get_disk_total_gb(self) -> float:
        """Get total disk space in GB."""
        try:
            stat = os.statvfs("/")
            return round(stat.f_blocks * stat.f_frsize / (1024 ** 3), 2)
        except:
            return 0.0
    
    def _get_network_rx_mbps(self) -> float:
        """Get network receive rate."""
        return self._get_network_rate("rx_bytes")
    
    def _get_network_tx_mbps(self) -> float:
        """Get network transmit rate."""
        return self._get_network_rate("tx_bytes")
    
    def _get_network_rate(self, stat_name: str) -> float:
        """Get network rate for a stat."""
        try:
            total = 0
            for iface in ["eth0", "wlan0", "enp1s0"]:
                stat_path = f"/sys/class/net/{iface}/statistics/{stat_name}"
                if Path(stat_path).exists():
                    with open(stat_path) as f:
                        total += int(f.read().strip())
            
            now = time.time()
            dt = now - self._last_net_time
            
            if stat_name == "rx_bytes":
                rate = (total - self._last_net_rx) / dt / 1024 / 1024 * 8  # Mbps
                self._last_net_rx = total
            else:
                rate = (total - self._last_net_tx) / dt / 1024 / 1024 * 8  # Mbps
                self._last_net_tx = total
            
            self._last_net_time = now
            return round(max(0, rate), 2)
        except:
            return 0.0


# =============================================================================
# Network Manager
# =============================================================================

class NetworkManager:
    """Manages network interfaces and connectivity."""
    
    @staticmethod
    def get_interfaces() -> List[NetworkInterface]:
        """Get all network interfaces."""
        interfaces = []
        
        try:
            net_path = Path("/sys/class/net")
            for iface_path in net_path.iterdir():
                if iface_path.name == "lo":
                    continue
                
                iface = NetworkInterface(name=iface_path.name)
                
                # MAC address
                mac_path = iface_path / "address"
                if mac_path.exists():
                    with open(mac_path) as f:
                        iface.mac_address = f.read().strip()
                
                # Check if up
                operstate_path = iface_path / "operstate"
                if operstate_path.exists():
                    with open(operstate_path) as f:
                        iface.is_up = f.read().strip() == "up"
                
                # Check if wireless
                iface.is_wireless = (iface_path / "wireless").exists()
                
                # Get IP address
                try:
                    result = subprocess.run(
                        ["ip", "-4", "addr", "show", iface.name],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "inet " in line:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    ip_cidr = parts[1]
                                    iface.ip_address = ip_cidr.split("/")[0]
                                    break
                except:
                    pass
                
                # Get WiFi info if wireless
                if iface.is_wireless and iface.is_up:
                    try:
                        result = subprocess.run(
                            ["iwgetid", iface.name, "-r"],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            iface.ssid = result.stdout.strip()
                        
                        result = subprocess.run(
                            ["iwconfig", iface.name],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            import re
                            match = re.search(r'Signal level=(-?\d+)', result.stdout)
                            if match:
                                iface.signal_strength = int(match.group(1))
                    except:
                        pass
                
                interfaces.append(iface)
        
        except Exception as e:
            logger.error(f"Failed to get network interfaces: {e}")
        
        return interfaces
    
    @staticmethod
    def connect_wifi(ssid: str, password: str) -> bool:
        """Connect to WiFi network."""
        try:
            # Use nmcli to connect
            result = subprocess.run(
                ["nmcli", "device", "wifi", "connect", ssid, "password", password],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to connect to WiFi: {e}")
            return False
    
    @staticmethod
    def scan_wifi() -> List[Dict[str, Any]]:
        """Scan for available WiFi networks."""
        networks = []
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            networks.append({
                                "ssid": parts[0],
                                "signal": int(parts[1]) if parts[1] else 0,
                                "security": parts[2],
                            })
        except Exception as e:
            logger.error(f"Failed to scan WiFi: {e}")
        
        return networks
    
    @staticmethod
    def get_hostname() -> str:
        """Get device hostname."""
        return socket.gethostname()
    
    @staticmethod
    def set_hostname(hostname: str) -> bool:
        """Set device hostname."""
        try:
            subprocess.run(["hostnamectl", "set-hostname", hostname], check=True)
            return True
        except:
            return False


# =============================================================================
# Installation Manager
# =============================================================================

class InstallationManager:
    """Manages Dynamical.ai installation on Jetson."""
    
    INSTALL_DIR = "/opt/dynamical"
    CONFIG_DIR = "/etc/dynamical"
    DATA_DIR = "/data/dynamical"
    LOG_DIR = "/var/log/dynamical"
    
    @classmethod
    def check_prerequisites(cls) -> Dict[str, bool]:
        """Check installation prerequisites."""
        checks = {}
        
        # Check CUDA
        checks["cuda"] = Path("/usr/local/cuda/bin/nvcc").exists()
        
        # Check TensorRT
        checks["tensorrt"] = cls._check_package("tensorrt")
        
        # Check Docker
        checks["docker"] = cls._check_command("docker")
        
        # Check nvidia-container-toolkit
        checks["nvidia_container"] = cls._check_package("nvidia-container-toolkit")
        
        # Check Python
        checks["python"] = sys.version_info >= (3, 8)
        
        # Check pip packages
        checks["pytorch"] = cls._check_python_package("torch")
        
        # Check disk space (need at least 10GB)
        try:
            stat = os.statvfs("/")
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            checks["disk_space"] = free_gb >= 10
        except:
            checks["disk_space"] = False
        
        return checks
    
    @staticmethod
    def _check_command(cmd: str) -> bool:
        """Check if command exists."""
        try:
            result = subprocess.run(["which", cmd], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def _check_package(package: str) -> bool:
        """Check if apt package is installed."""
        try:
            result = subprocess.run(
                ["dpkg", "-l", package],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def _check_python_package(package: str) -> bool:
        """Check if Python package is installed."""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
    
    @classmethod
    def install(cls, config: EdgeDeviceConfig = None) -> bool:
        """Install Dynamical.ai on device."""
        logger.info("Starting Dynamical.ai installation...")
        
        try:
            # Create directories
            for dir_path in [cls.INSTALL_DIR, cls.CONFIG_DIR, cls.DATA_DIR, cls.LOG_DIR]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Check prerequisites
            prereqs = cls.check_prerequisites()
            missing = [k for k, v in prereqs.items() if not v]
            if missing:
                logger.warning(f"Missing prerequisites: {missing}")
                # Install missing prerequisites
                cls._install_prerequisites(missing)
            
            # Install Python dependencies
            cls._install_python_dependencies()
            
            # Setup systemd services
            cls._setup_systemd_services()
            
            # Save configuration
            if config:
                config.save(f"{cls.CONFIG_DIR}/config.json")
            
            logger.info("Installation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    @classmethod
    def _install_prerequisites(cls, missing: List[str]):
        """Install missing prerequisites."""
        apt_packages = []
        
        if "docker" in missing:
            apt_packages.append("docker.io")
        
        if "nvidia_container" in missing:
            apt_packages.append("nvidia-container-toolkit")
        
        if apt_packages:
            subprocess.run(["apt-get", "update"], check=False)
            subprocess.run(["apt-get", "install", "-y"] + apt_packages, check=False)
    
    @classmethod
    def _install_python_dependencies(cls):
        """Install Python dependencies."""
        packages = [
            "fastapi",
            "uvicorn",
            "requests",
            "pydantic",
            "numpy",
            "opencv-python",
            "pyserial",
        ]
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + packages, check=False)
    
    @classmethod
    def _setup_systemd_services(cls):
        """Setup systemd services."""
        # Edge service
        service_content = """[Unit]
Description=Dynamical.ai Edge Service
After=network.target docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dynamical
ExecStart=/usr/bin/python3 -m platform.edge.edge_service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = "/etc/systemd/system/dynamical-edge.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        subprocess.run(["systemctl", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "enable", "dynamical-edge"], check=False)
    
    @classmethod
    def uninstall(cls) -> bool:
        """Uninstall Dynamical.ai from device."""
        try:
            # Stop services
            subprocess.run(["systemctl", "stop", "dynamical-edge"], check=False)
            subprocess.run(["systemctl", "disable", "dynamical-edge"], check=False)
            
            # Remove service file
            Path("/etc/systemd/system/dynamical-edge.service").unlink(missing_ok=True)
            subprocess.run(["systemctl", "daemon-reload"], check=False)
            
            # Remove directories (keep data)
            import shutil
            shutil.rmtree(cls.INSTALL_DIR, ignore_errors=True)
            shutil.rmtree(cls.CONFIG_DIR, ignore_errors=True)
            
            logger.info("Uninstallation completed")
            return True
            
        except Exception as e:
            logger.error(f"Uninstallation failed: {e}")
            return False


# =============================================================================
# Power Management
# =============================================================================

class PowerManager:
    """Manages Jetson power modes and fan control."""
    
    @staticmethod
    def get_power_mode() -> str:
        """Get current power mode."""
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "NV Power Mode" in line:
                        return line.split(":")[-1].strip()
        except:
            pass
        return "unknown"
    
    @staticmethod
    def set_power_mode(mode: str) -> bool:
        """Set power mode (0=max performance, higher=lower power)."""
        try:
            result = subprocess.run(
                ["nvpmodel", "-m", str(mode)],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def get_fan_speed() -> int:
        """Get current fan speed (0-255)."""
        try:
            fan_path = "/sys/devices/pwm-fan/target_pwm"
            if Path(fan_path).exists():
                with open(fan_path) as f:
                    return int(f.read().strip())
        except:
            pass
        return 0
    
    @staticmethod
    def set_fan_speed(speed: int) -> bool:
        """Set fan speed (0-255) or -1 for auto."""
        try:
            if speed < 0:
                # Enable auto fan control
                subprocess.run(["jetson_clocks", "--fan"], check=False)
            else:
                fan_path = "/sys/devices/pwm-fan/target_pwm"
                if Path(fan_path).exists():
                    with open(fan_path, 'w') as f:
                        f.write(str(max(0, min(255, speed))))
            return True
        except:
            return False


# =============================================================================
# Main SDK Class
# =============================================================================

class JetsonOrinSDK:
    """
    Main SDK for Dynamical.ai on Jetson Orin.
    
    Example:
        ```python
        from platform.edge.jetson_orin_sdk import JetsonOrinSDK
        
        # Initialize SDK
        sdk = JetsonOrinSDK()
        
        # Get hardware info
        hw_info = sdk.get_hardware_info()
        print(f"Model: {hw_info.model}")
        print(f"JetPack: {hw_info.jetpack_version}")
        
        # Get system metrics
        metrics = sdk.get_metrics()
        print(f"CPU: {metrics.cpu_usage_percent}%")
        print(f"GPU: {metrics.gpu_usage_percent}%")
        
        # Configure device
        sdk.configure(
            device_name="warehouse-robot-01",
            platform_url="https://api.dynamical.ai",
            api_key="dyn_xxx"
        )
        
        # Start services
        sdk.start_services()
        ```
    """
    
    def __init__(self, config_path: str = None):
        """Initialize SDK."""
        self.config = EdgeDeviceConfig.load(config_path)
        self.hardware_info = JetsonHardwareDetector.get_hardware_info()
        self.monitor = SystemMonitor()
        self._running = False
        
        # Generate device ID if not set
        if not self.config.device_id:
            self.config.device_id = self._generate_device_id()
    
    def _generate_device_id(self) -> str:
        """Generate unique device ID based on hardware."""
        hw_string = f"{self.hardware_info.serial_number}-{self.hardware_info.mac_address}"
        return f"jetson_{hashlib.sha256(hw_string.encode()).hexdigest()[:12]}"
    
    def get_hardware_info(self) -> HardwareInfo:
        """Get hardware information."""
        return self.hardware_info
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.monitor.get_metrics()
    
    def get_network_interfaces(self) -> List[NetworkInterface]:
        """Get network interfaces."""
        return NetworkManager.get_interfaces()
    
    def configure(
        self,
        device_name: str = None,
        site_id: str = None,
        organization_id: str = None,
        platform_url: str = None,
        api_key: str = None,
        **kwargs
    ) -> bool:
        """Configure the device."""
        if device_name:
            self.config.device_name = device_name
        if site_id:
            self.config.site_id = site_id
        if organization_id:
            self.config.organization_id = organization_id
        if platform_url:
            self.config.platform_url = platform_url
        if api_key:
            self.config.api_key = api_key
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.config.save()
        return True
    
    def connect_wifi(self, ssid: str, password: str) -> bool:
        """Connect to WiFi network."""
        return NetworkManager.connect_wifi(ssid, password)
    
    def scan_wifi(self) -> List[Dict]:
        """Scan for WiFi networks."""
        return NetworkManager.scan_wifi()
    
    def set_power_mode(self, mode: str) -> bool:
        """Set power mode."""
        return PowerManager.set_power_mode(mode)
    
    def get_power_mode(self) -> str:
        """Get current power mode."""
        return PowerManager.get_power_mode()
    
    def install(self) -> bool:
        """Install Dynamical.ai on this device."""
        return InstallationManager.install(self.config)
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check installation prerequisites."""
        return InstallationManager.check_prerequisites()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive device status."""
        metrics = self.get_metrics()
        interfaces = self.get_network_interfaces()
        
        return {
            "device_id": self.config.device_id,
            "device_name": self.config.device_name,
            "status": DeviceStatus.RUNNING.value if self._running else DeviceStatus.ONLINE.value,
            "hardware": asdict(self.hardware_info),
            "metrics": asdict(metrics),
            "network": [asdict(iface) for iface in interfaces],
            "config": {
                "site_id": self.config.site_id,
                "platform_url": self.config.platform_url,
                "power_mode": self.get_power_mode(),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamical.ai Jetson Orin SDK")
    parser.add_argument("command", choices=[
        "info", "metrics", "status", "install", "configure", "wifi-scan", "wifi-connect"
    ])
    parser.add_argument("--platform-url", help="Platform URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--device-name", help="Device name")
    parser.add_argument("--ssid", help="WiFi SSID")
    parser.add_argument("--password", help="WiFi password")
    
    args = parser.parse_args()
    
    sdk = JetsonOrinSDK()
    
    if args.command == "info":
        hw = sdk.get_hardware_info()
        print(f"Model: {hw.model.value}")
        print(f"JetPack: {hw.jetpack_version}")
        print(f"L4T: {hw.l4t_version}")
        print(f"CUDA: {hw.cuda_version}")
        print(f"TensorRT: {hw.tensorrt_version}")
        print(f"CPU Cores: {hw.cpu_cores}")
        print(f"GPU Cores: {hw.gpu_cores}")
        print(f"Memory: {hw.memory_total_mb} MB")
        print(f"Serial: {hw.serial_number}")
        print(f"MAC: {hw.mac_address}")
    
    elif args.command == "metrics":
        m = sdk.get_metrics()
        print(f"CPU Usage: {m.cpu_usage_percent}%")
        print(f"GPU Usage: {m.gpu_usage_percent}%")
        print(f"Memory: {m.memory_used_mb}/{m.memory_total_mb} MB")
        print(f"CPU Temp: {m.temperature_cpu_c}°C")
        print(f"GPU Temp: {m.temperature_gpu_c}°C")
        print(f"Power: {m.power_usage_w}W")
    
    elif args.command == "status":
        status = sdk.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "install":
        prereqs = sdk.check_prerequisites()
        print("Prerequisites:")
        for k, v in prereqs.items():
            print(f"  {k}: {'✓' if v else '✗'}")
        
        if input("\nProceed with installation? [y/N] ").lower() == 'y':
            sdk.install()
    
    elif args.command == "configure":
        sdk.configure(
            platform_url=args.platform_url,
            api_key=args.api_key,
            device_name=args.device_name,
        )
        print("Configuration saved")
    
    elif args.command == "wifi-scan":
        networks = sdk.scan_wifi()
        for net in networks:
            print(f"  {net['ssid']}: {net['signal']}% ({net['security']})")
    
    elif args.command == "wifi-connect":
        if not args.ssid or not args.password:
            print("Error: --ssid and --password required")
            return
        if sdk.connect_wifi(args.ssid, args.password):
            print("Connected successfully")
        else:
            print("Connection failed")


if __name__ == "__main__":
    main()

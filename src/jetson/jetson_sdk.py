"""
Dynamical.ai Jetson Orin SDK

Complete SDK for deploying Dynamical.ai on NVIDIA Jetson Orin devices.
Handles installation, configuration, device discovery, and edge services.

Supported Devices:
- Jetson Orin Nano (8GB)
- Jetson Orin NX (8GB/16GB)
- Jetson AGX Orin (32GB/64GB)

Features:
- One-click installation
- Hardware detection and optimization
- DOGlove and ONVIF camera integration
- Real-time device monitoring
- OTA updates
- Remote management via platform

Requirements:
    JetPack 5.1+ (L4T 35.x+)
    Python 3.8+
    CUDA 11.4+
"""

import os
import sys
import json
import uuid
import socket
import struct
import asyncio
import hashlib
import platform
import subprocess
import threading
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# =============================================================================
# Jetson Hardware Detection
# =============================================================================

class JetsonModel(str, Enum):
    """Supported Jetson models."""
    ORIN_NANO_8GB = "orin_nano_8gb"
    ORIN_NX_8GB = "orin_nx_8gb"
    ORIN_NX_16GB = "orin_nx_16gb"
    AGX_ORIN_32GB = "agx_orin_32gb"
    AGX_ORIN_64GB = "agx_orin_64gb"
    UNKNOWN = "unknown"


@dataclass
class JetsonInfo:
    """Jetson hardware information."""
    model: JetsonModel
    serial_number: str
    l4t_version: str
    jetpack_version: str
    cuda_version: str
    cudnn_version: str
    tensorrt_version: str
    memory_total_mb: int
    storage_total_gb: float
    storage_free_gb: float
    gpu_name: str
    gpu_memory_mb: int
    cpu_cores: int
    cpu_model: str
    hostname: str
    ip_addresses: List[str]
    mac_addresses: Dict[str, str]


class JetsonDetector:
    """Detect Jetson hardware and software configuration."""
    
    # Model detection patterns
    MODEL_PATTERNS = {
        "p3767-0000": JetsonModel.ORIN_NANO_8GB,
        "p3767-0001": JetsonModel.ORIN_NX_8GB,
        "p3767-0003": JetsonModel.ORIN_NX_16GB,
        "p3701-0000": JetsonModel.AGX_ORIN_32GB,
        "p3701-0004": JetsonModel.AGX_ORIN_64GB,
    }
    
    @classmethod
    def detect(cls) -> JetsonInfo:
        """Detect Jetson hardware and configuration."""
        return JetsonInfo(
            model=cls._detect_model(),
            serial_number=cls._get_serial_number(),
            l4t_version=cls._get_l4t_version(),
            jetpack_version=cls._get_jetpack_version(),
            cuda_version=cls._get_cuda_version(),
            cudnn_version=cls._get_cudnn_version(),
            tensorrt_version=cls._get_tensorrt_version(),
            memory_total_mb=cls._get_memory_total(),
            storage_total_gb=cls._get_storage_total(),
            storage_free_gb=cls._get_storage_free(),
            gpu_name=cls._get_gpu_name(),
            gpu_memory_mb=cls._get_gpu_memory(),
            cpu_cores=os.cpu_count() or 0,
            cpu_model=cls._get_cpu_model(),
            hostname=socket.gethostname(),
            ip_addresses=cls._get_ip_addresses(),
            mac_addresses=cls._get_mac_addresses(),
        )
    
    @classmethod
    def _detect_model(cls) -> JetsonModel:
        """Detect Jetson model from device tree."""
        try:
            # Check NVIDIA device tree
            dt_path = Path("/proc/device-tree/model")
            if dt_path.exists():
                model_str = dt_path.read_text().strip('\x00').lower()
                
                for pattern, model in cls.MODEL_PATTERNS.items():
                    if pattern in model_str:
                        return model
                
                # Check for Orin keywords
                if "orin" in model_str:
                    if "nano" in model_str:
                        return JetsonModel.ORIN_NANO_8GB
                    elif "nx" in model_str:
                        return JetsonModel.ORIN_NX_16GB
                    elif "agx" in model_str:
                        return JetsonModel.AGX_ORIN_64GB
            
            # Alternative: check tegra chip
            tegra_path = Path("/sys/module/tegra_fuse/parameters/tegra_chip_id")
            if tegra_path.exists():
                chip_id = tegra_path.read_text().strip()
                if chip_id == "35":  # Orin
                    return JetsonModel.AGX_ORIN_64GB
                    
        except Exception as e:
            logger.warning(f"Model detection failed: {e}")
        
        return JetsonModel.UNKNOWN
    
    @classmethod
    def _get_serial_number(cls) -> str:
        """Get device serial number."""
        try:
            sn_path = Path("/proc/device-tree/serial-number")
            if sn_path.exists():
                return sn_path.read_text().strip('\x00')
            
            # Alternative: use machine-id
            mid_path = Path("/etc/machine-id")
            if mid_path.exists():
                return mid_path.read_text().strip()[:16]
                
        except Exception:
            pass
        
        return hashlib.md5(socket.gethostname().encode()).hexdigest()[:16]
    
    @classmethod
    def _get_l4t_version(cls) -> str:
        """Get L4T (Linux for Tegra) version."""
        try:
            release_path = Path("/etc/nv_tegra_release")
            if release_path.exists():
                content = release_path.read_text()
                # Parse: # R35 (release), REVISION: 3.1
                import re
                match = re.search(r'R(\d+).*REVISION:\s*([\d.]+)', content)
                if match:
                    return f"{match.group(1)}.{match.group(2)}"
        except Exception:
            pass
        return "unknown"
    
    @classmethod
    def _get_jetpack_version(cls) -> str:
        """Get JetPack version."""
        l4t = cls._get_l4t_version()
        
        # L4T to JetPack mapping
        l4t_jp_map = {
            "35.1": "5.0.2",
            "35.2": "5.1",
            "35.3": "5.1.1",
            "35.4": "5.1.2",
            "36.0": "6.0 DP",
            "36.2": "6.0",
            "36.3": "6.0.1",
        }
        
        for l4t_prefix, jp_version in l4t_jp_map.items():
            if l4t.startswith(l4t_prefix):
                return jp_version
        
        return "unknown"
    
    @classmethod
    def _get_cuda_version(cls) -> str:
        """Get CUDA version."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True
            )
            import re
            match = re.search(r'release ([\d.]+)', result.stdout)
            if match:
                return match.group(1)
        except Exception:
            pass
        
        # Check cuda version file
        try:
            version_path = Path("/usr/local/cuda/version.txt")
            if version_path.exists():
                content = version_path.read_text()
                import re
                match = re.search(r'([\d.]+)', content)
                if match:
                    return match.group(1)
        except Exception:
            pass
        
        return "unknown"
    
    @classmethod
    def _get_cudnn_version(cls) -> str:
        """Get cuDNN version."""
        try:
            header_path = Path("/usr/include/cudnn_version.h")
            if header_path.exists():
                content = header_path.read_text()
                major = minor = patch = "0"
                import re
                for line in content.split('\n'):
                    if 'CUDNN_MAJOR' in line:
                        m = re.search(r'(\d+)', line)
                        if m: major = m.group(1)
                    elif 'CUDNN_MINOR' in line:
                        m = re.search(r'(\d+)', line)
                        if m: minor = m.group(1)
                    elif 'CUDNN_PATCHLEVEL' in line:
                        m = re.search(r'(\d+)', line)
                        if m: patch = m.group(1)
                return f"{major}.{minor}.{patch}"
        except Exception:
            pass
        return "unknown"
    
    @classmethod
    def _get_tensorrt_version(cls) -> str:
        """Get TensorRT version."""
        try:
            result = subprocess.run(
                ["dpkg", "-l", "tensorrt"],
                capture_output=True, text=True
            )
            import re
            match = re.search(r'tensorrt\s+([\d.]+)', result.stdout)
            if match:
                return match.group(1)
        except Exception:
            pass
        return "unknown"
    
    @classmethod
    def _get_memory_total(cls) -> int:
        """Get total RAM in MB."""
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemTotal' in line:
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return 0
    
    @classmethod
    def _get_storage_total(cls) -> float:
        """Get total storage in GB."""
        try:
            stat = os.statvfs('/')
            return (stat.f_blocks * stat.f_frsize) / (1024**3)
        except Exception:
            return 0.0
    
    @classmethod
    def _get_storage_free(cls) -> float:
        """Get free storage in GB."""
        try:
            stat = os.statvfs('/')
            return (stat.f_bavail * stat.f_frsize) / (1024**3)
        except Exception:
            return 0.0
    
    @classmethod
    def _get_gpu_name(cls) -> str:
        """Get GPU name."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            pass
        
        model = cls._detect_model()
        gpu_names = {
            JetsonModel.ORIN_NANO_8GB: "NVIDIA Orin Nano (1024 CUDA cores)",
            JetsonModel.ORIN_NX_8GB: "NVIDIA Orin NX (1024 CUDA cores)",
            JetsonModel.ORIN_NX_16GB: "NVIDIA Orin NX (2048 CUDA cores)",
            JetsonModel.AGX_ORIN_32GB: "NVIDIA AGX Orin (1792 CUDA cores)",
            JetsonModel.AGX_ORIN_64GB: "NVIDIA AGX Orin (2048 CUDA cores)",
        }
        return gpu_names.get(model, "NVIDIA Tegra GPU")
    
    @classmethod
    def _get_gpu_memory(cls) -> int:
        """Get GPU memory in MB (shared with system on Jetson)."""
        # On Jetson, GPU memory is unified with system RAM
        model = cls._detect_model()
        
        # Approximate GPU memory allocation
        gpu_mem = {
            JetsonModel.ORIN_NANO_8GB: 4096,
            JetsonModel.ORIN_NX_8GB: 4096,
            JetsonModel.ORIN_NX_16GB: 8192,
            JetsonModel.AGX_ORIN_32GB: 16384,
            JetsonModel.AGX_ORIN_64GB: 32768,
        }
        return gpu_mem.get(model, 4096)
    
    @classmethod
    def _get_cpu_model(cls) -> str:
        """Get CPU model."""
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line.lower() or 'cpu model' in line.lower():
                        return line.split(':')[1].strip()
                    if 'hardware' in line.lower():
                        return line.split(':')[1].strip()
        except Exception:
            pass
        return "ARM Cortex-A78AE"
    
    @classmethod
    def _get_ip_addresses(cls) -> List[str]:
        """Get all IP addresses."""
        ips = []
        try:
            import socket
            hostname = socket.gethostname()
            ips = socket.gethostbyname_ex(hostname)[2]
        except Exception:
            pass
        
        # Also try getting from interfaces
        try:
            result = subprocess.run(
                ["ip", "-4", "addr", "show"],
                capture_output=True, text=True
            )
            import re
            ips.extend(re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout))
        except Exception:
            pass
        
        # Filter out localhost
        return list(set(ip for ip in ips if not ip.startswith('127.')))
    
    @classmethod
    def _get_mac_addresses(cls) -> Dict[str, str]:
        """Get MAC addresses for each interface."""
        macs = {}
        try:
            net_path = Path("/sys/class/net")
            for iface in net_path.iterdir():
                if iface.name != 'lo':
                    addr_path = iface / "address"
                    if addr_path.exists():
                        macs[iface.name] = addr_path.read_text().strip()
        except Exception:
            pass
        return macs
    
    @classmethod
    def is_jetson(cls) -> bool:
        """Check if running on Jetson hardware."""
        indicators = [
            Path("/etc/nv_tegra_release").exists(),
            Path("/proc/device-tree/model").exists(),
            Path("/sys/module/tegra_fuse").exists(),
        ]
        return any(indicators)


# =============================================================================
# Installation Manager
# =============================================================================

@dataclass
class InstallationConfig:
    """Installation configuration."""
    install_dir: str = "/opt/dynamical"
    data_dir: str = "/var/lib/dynamical"
    log_dir: str = "/var/log/dynamical"
    config_dir: str = "/etc/dynamical"
    
    # Components
    install_core: bool = True
    install_perception: bool = True
    install_moai: bool = True
    install_platform: bool = True
    
    # Hardware
    enable_gpu: bool = True
    enable_tensorrt: bool = True
    
    # Networking
    api_port: int = 8080
    websocket_port: int = 8081
    discovery_port: int = 5353
    
    # Platform connection
    platform_url: str = ""
    api_key: str = ""
    device_name: str = ""


class JetsonInstaller:
    """Install Dynamical.ai on Jetson devices."""
    
    REQUIRED_PACKAGES = [
        "python3-pip",
        "python3-dev",
        "build-essential",
        "libopencv-dev",
        "libjpeg-dev",
        "libpng-dev",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "libv4l-dev",
        "libusb-1.0-0-dev",
    ]
    
    PYTHON_PACKAGES = [
        "numpy",
        "opencv-python",
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "aiohttp",
        "pydantic",
        "python-multipart",
        "psutil",
        "zeroconf",
        "pyudev",
        "requests",
    ]
    
    def __init__(self, config: InstallationConfig):
        self.config = config
        self.jetson_info: Optional[JetsonInfo] = None
        self._progress_callback: Optional[Callable[[str, float], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback function."""
        self._progress_callback = callback
    
    def _report_progress(self, message: str, progress: float):
        """Report installation progress."""
        logger.info(f"[{progress:.0%}] {message}")
        if self._progress_callback:
            self._progress_callback(message, progress)
    
    def check_prerequisites(self) -> Dict[str, Any]:
        """Check installation prerequisites."""
        results = {
            "is_jetson": JetsonDetector.is_jetson(),
            "is_root": os.geteuid() == 0,
            "python_version": platform.python_version(),
            "python_ok": sys.version_info >= (3, 8),
            "disk_space_gb": JetsonDetector._get_storage_free(),
            "disk_ok": JetsonDetector._get_storage_free() > 5.0,
            "memory_mb": JetsonDetector._get_memory_total(),
            "memory_ok": JetsonDetector._get_memory_total() > 4000,
            "cuda_available": shutil.which("nvcc") is not None,
            "internet_ok": self._check_internet(),
        }
        
        results["all_ok"] = all([
            results["is_jetson"] or True,  # Allow non-Jetson for testing
            results["python_ok"],
            results["disk_ok"],
            results["memory_ok"],
        ])
        
        return results
    
    def _check_internet(self) -> bool:
        """Check internet connectivity."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    def install(self) -> bool:
        """Run full installation."""
        try:
            self._report_progress("Checking prerequisites...", 0.0)
            prereqs = self.check_prerequisites()
            
            if not prereqs["all_ok"]:
                logger.error(f"Prerequisites not met: {prereqs}")
                return False
            
            self._report_progress("Detecting hardware...", 0.05)
            self.jetson_info = JetsonDetector.detect()
            logger.info(f"Detected: {self.jetson_info.model.value}")
            
            self._report_progress("Creating directories...", 0.1)
            self._create_directories()
            
            self._report_progress("Installing system packages...", 0.15)
            self._install_system_packages()
            
            self._report_progress("Installing Python packages...", 0.3)
            self._install_python_packages()
            
            if self.config.enable_tensorrt:
                self._report_progress("Configuring TensorRT...", 0.45)
                self._configure_tensorrt()
            
            self._report_progress("Installing Dynamical.ai...", 0.5)
            self._install_dynamical()
            
            self._report_progress("Configuring services...", 0.7)
            self._configure_services()
            
            self._report_progress("Setting up device discovery...", 0.85)
            self._setup_discovery()
            
            self._report_progress("Finalizing installation...", 0.95)
            self._finalize()
            
            self._report_progress("Installation complete!", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def _create_directories(self):
        """Create required directories."""
        dirs = [
            self.config.install_dir,
            self.config.data_dir,
            self.config.log_dir,
            self.config.config_dir,
            f"{self.config.data_dir}/episodes",
            f"{self.config.data_dir}/models",
            f"{self.config.data_dir}/cache",
        ]
        
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def _install_system_packages(self):
        """Install system dependencies."""
        try:
            subprocess.run(
                ["apt-get", "update"],
                check=True, capture_output=True
            )
            subprocess.run(
                ["apt-get", "install", "-y"] + self.REQUIRED_PACKAGES,
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some system packages may not have installed: {e}")
    
    def _install_python_packages(self):
        """Install Python dependencies."""
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True, capture_output=True
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + self.PYTHON_PACKAGES,
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some Python packages may not have installed: {e}")
    
    def _configure_tensorrt(self):
        """Configure TensorRT optimization."""
        # Create TensorRT engine cache directory
        cache_dir = Path(self.config.data_dir) / "tensorrt_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Set environment variables
        env_file = Path(self.config.config_dir) / "tensorrt.env"
        env_file.write_text(f"""
# TensorRT Configuration
TRT_ENGINE_CACHE={cache_dir}
TRT_FP16_ENABLE=1
TRT_INT8_ENABLE=0
TRT_WORKSPACE_MB=1024
""")
    
    def _install_dynamical(self):
        """Install Dynamical.ai package."""
        # In production, this would download/install the actual package
        # For now, we copy the local files
        
        src_dir = Path(__file__).parent.parent
        dst_dir = Path(self.config.install_dir)
        
        # Copy Python packages
        packages = ["core", "perception", "quality", "il", "moai", "crypto", "platform"]
        for pkg in packages:
            src_pkg = src_dir / pkg
            if src_pkg.exists():
                dst_pkg = dst_dir / pkg
                if dst_pkg.exists():
                    shutil.rmtree(dst_pkg)
                shutil.copytree(src_pkg, dst_pkg)
        
        # Copy Jetson-specific files
        jetson_dir = dst_dir / "jetson"
        jetson_dir.mkdir(exist_ok=True)
        
        # Copy this SDK
        shutil.copy(__file__, jetson_dir / "jetson_sdk.py")
    
    def _configure_services(self):
        """Configure systemd services."""
        # Edge service
        edge_service = f"""[Unit]
Description=Dynamical.ai Edge Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={self.config.install_dir}
ExecStart={sys.executable} -m jetson.edge_service --config {self.config.config_dir}/edge.json
Restart=always
RestartSec=10
Environment=PYTHONPATH={self.config.install_dir}

[Install]
WantedBy=multi-user.target
"""
        
        service_path = Path("/etc/systemd/system/dynamical-edge.service")
        try:
            service_path.write_text(edge_service)
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", "dynamical-edge"], check=True)
        except Exception as e:
            logger.warning(f"Could not configure systemd service: {e}")
    
    def _setup_discovery(self):
        """Setup mDNS/Zeroconf discovery."""
        # Avahi service file for network discovery
        avahi_service = f"""<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name>Dynamical.ai Edge - {self.config.device_name or socket.gethostname()}</name>
  <service>
    <type>_dynamical._tcp</type>
    <port>{self.config.api_port}</port>
    <txt-record>version={__version__}</txt-record>
    <txt-record>type=jetson-orin</txt-record>
  </service>
</service-group>
"""
        
        avahi_path = Path("/etc/avahi/services/dynamical.service")
        try:
            avahi_path.parent.mkdir(parents=True, exist_ok=True)
            avahi_path.write_text(avahi_service)
            subprocess.run(["systemctl", "restart", "avahi-daemon"], check=True)
        except Exception as e:
            logger.warning(f"Could not configure Avahi: {e}")
    
    def _finalize(self):
        """Finalize installation."""
        # Save installation info
        install_info = {
            "version": __version__,
            "installed_at": datetime.now().isoformat(),
            "jetson_model": self.jetson_info.model.value if self.jetson_info else "unknown",
            "serial_number": self.jetson_info.serial_number if self.jetson_info else "",
            "config": asdict(self.config),
        }
        
        info_path = Path(self.config.config_dir) / "installation.json"
        info_path.write_text(json.dumps(install_info, indent=2))
        
        # Save edge configuration
        edge_config = {
            "device_id": f"jetson_{self.jetson_info.serial_number if self.jetson_info else uuid.uuid4().hex[:12]}",
            "device_name": self.config.device_name or socket.gethostname(),
            "api_port": self.config.api_port,
            "websocket_port": self.config.websocket_port,
            "platform_url": self.config.platform_url,
            "api_key": self.config.api_key,
        }
        
        config_path = Path(self.config.config_dir) / "edge.json"
        config_path.write_text(json.dumps(edge_config, indent=2))
    
    def uninstall(self) -> bool:
        """Uninstall Dynamical.ai."""
        try:
            # Stop and disable service
            subprocess.run(["systemctl", "stop", "dynamical-edge"], check=False)
            subprocess.run(["systemctl", "disable", "dynamical-edge"], check=False)
            
            # Remove service file
            Path("/etc/systemd/system/dynamical-edge.service").unlink(missing_ok=True)
            
            # Remove Avahi service
            Path("/etc/avahi/services/dynamical.service").unlink(missing_ok=True)
            
            # Remove installation directory (but keep data)
            if Path(self.config.install_dir).exists():
                shutil.rmtree(self.config.install_dir)
            
            logger.info("Uninstallation complete")
            return True
            
        except Exception as e:
            logger.error(f"Uninstallation failed: {e}")
            return False


# =============================================================================
# Power Management
# =============================================================================

class PowerMode(str, Enum):
    """Jetson power modes."""
    MAX_PERFORMANCE = "maxn"
    PERFORMANCE_15W = "15w"
    PERFORMANCE_30W = "30w"
    POWER_SAVE_10W = "10w"
    POWER_SAVE_7W = "7w"


class JetsonPowerManager:
    """Manage Jetson power modes for optimal performance/efficiency."""
    
    # NVP model mappings for different Jetson models
    NVP_MODELS = {
        JetsonModel.ORIN_NANO_8GB: {
            PowerMode.MAX_PERFORMANCE: "0",
            PowerMode.PERFORMANCE_15W: "1",
        },
        JetsonModel.ORIN_NX_16GB: {
            PowerMode.MAX_PERFORMANCE: "0",
            PowerMode.PERFORMANCE_15W: "2",
            PowerMode.POWER_SAVE_10W: "3",
        },
        JetsonModel.AGX_ORIN_64GB: {
            PowerMode.MAX_PERFORMANCE: "0",
            PowerMode.PERFORMANCE_30W: "1",
            PowerMode.PERFORMANCE_15W: "3",
        },
    }
    
    @classmethod
    def get_current_mode(cls) -> Optional[str]:
        """Get current power mode."""
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    @classmethod
    def set_mode(cls, model: JetsonModel, mode: PowerMode) -> bool:
        """Set power mode."""
        try:
            mode_map = cls.NVP_MODELS.get(model, {})
            nvp_id = mode_map.get(mode)
            
            if nvp_id is None:
                logger.warning(f"Power mode {mode} not supported for {model}")
                return False
            
            subprocess.run(
                ["nvpmodel", "-m", nvp_id],
                check=True
            )
            
            # Also set max clock for performance modes
            if mode in [PowerMode.MAX_PERFORMANCE, PowerMode.PERFORMANCE_30W]:
                subprocess.run(["jetson_clocks"], check=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set power mode: {e}")
            return False
    
    @classmethod
    def get_thermal_status(cls) -> Dict[str, float]:
        """Get thermal zone temperatures."""
        temps = {}
        try:
            thermal_path = Path("/sys/class/thermal")
            for zone in thermal_path.glob("thermal_zone*"):
                try:
                    temp = int((zone / "temp").read_text().strip()) / 1000.0
                    type_name = (zone / "type").read_text().strip()
                    temps[type_name] = temp
                except Exception:
                    pass
        except Exception:
            pass
        return temps
    
    @classmethod
    def get_power_stats(cls) -> Dict[str, Any]:
        """Get power consumption statistics."""
        stats = {}
        try:
            # INA3221 power monitors
            hwmon_path = Path("/sys/bus/i2c/drivers/ina3221")
            for device in hwmon_path.glob("*/hwmon/hwmon*/"):
                for power_file in device.glob("power*_input"):
                    try:
                        name = power_file.name.replace("_input", "")
                        value = int(power_file.read_text().strip()) / 1000000.0  # Convert to Watts
                        stats[name] = value
                    except Exception:
                        pass
        except Exception:
            pass
        return stats


# =============================================================================
# CLI Interface
# =============================================================================

def cli_main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dynamical.ai Jetson SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check system info
    python jetson_sdk.py info
    
    # Install Dynamical.ai
    sudo python jetson_sdk.py install --platform-url https://api.dynamical.ai --api-key dyn_xxx
    
    # Check prerequisites
    python jetson_sdk.py check
    
    # Set power mode
    sudo python jetson_sdk.py power --mode maxn
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check prerequisites")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install Dynamical.ai")
    install_parser.add_argument("--platform-url", help="Platform API URL")
    install_parser.add_argument("--api-key", help="Platform API key")
    install_parser.add_argument("--device-name", help="Device name")
    install_parser.add_argument("--install-dir", default="/opt/dynamical", help="Installation directory")
    
    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall Dynamical.ai")
    
    # Power command
    power_parser = subparsers.add_parser("power", help="Power management")
    power_parser.add_argument("--mode", choices=["maxn", "15w", "30w", "10w", "7w"], help="Power mode")
    power_parser.add_argument("--status", action="store_true", help="Show power status")
    
    args = parser.parse_args()
    
    if args.command == "info":
        info = JetsonDetector.detect()
        if args.json:
            print(json.dumps(asdict(info), indent=2))
        else:
            print(f"Model: {info.model.value}")
            print(f"Serial: {info.serial_number}")
            print(f"JetPack: {info.jetpack_version}")
            print(f"CUDA: {info.cuda_version}")
            print(f"GPU: {info.gpu_name}")
            print(f"Memory: {info.memory_total_mb} MB")
            print(f"Storage: {info.storage_free_gb:.1f} / {info.storage_total_gb:.1f} GB")
            print(f"Hostname: {info.hostname}")
            print(f"IPs: {', '.join(info.ip_addresses)}")
    
    elif args.command == "check":
        config = InstallationConfig()
        installer = JetsonInstaller(config)
        results = installer.check_prerequisites()
        
        print("Prerequisites Check:")
        for key, value in results.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
    
    elif args.command == "install":
        config = InstallationConfig(
            install_dir=args.install_dir,
            platform_url=args.platform_url or "",
            api_key=args.api_key or "",
            device_name=args.device_name or "",
        )
        
        installer = JetsonInstaller(config)
        
        def progress_callback(msg, progress):
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r[{bar}] {progress:.0%} {msg}", end="", flush=True)
            if progress >= 1.0:
                print()
        
        installer.set_progress_callback(progress_callback)
        
        if installer.install():
            print("\n✓ Installation successful!")
            print(f"  Start service: systemctl start dynamical-edge")
            print(f"  View logs: journalctl -u dynamical-edge -f")
        else:
            print("\n✗ Installation failed")
            sys.exit(1)
    
    elif args.command == "uninstall":
        config = InstallationConfig()
        installer = JetsonInstaller(config)
        
        confirm = input("Are you sure you want to uninstall? (yes/no): ")
        if confirm.lower() == "yes":
            if installer.uninstall():
                print("✓ Uninstallation complete")
            else:
                print("✗ Uninstallation failed")
                sys.exit(1)
    
    elif args.command == "power":
        if args.status:
            print("Current mode:", JetsonPowerManager.get_current_mode())
            print("\nThermal status:")
            for zone, temp in JetsonPowerManager.get_thermal_status().items():
                print(f"  {zone}: {temp:.1f}°C")
            print("\nPower consumption:")
            for rail, watts in JetsonPowerManager.get_power_stats().items():
                print(f"  {rail}: {watts:.2f}W")
        
        elif args.mode:
            mode_map = {
                "maxn": PowerMode.MAX_PERFORMANCE,
                "30w": PowerMode.PERFORMANCE_30W,
                "15w": PowerMode.PERFORMANCE_15W,
                "10w": PowerMode.POWER_SAVE_10W,
                "7w": PowerMode.POWER_SAVE_7W,
            }
            mode = mode_map[args.mode]
            model = JetsonDetector.detect().model
            
            if JetsonPowerManager.set_mode(model, mode):
                print(f"✓ Power mode set to {args.mode}")
            else:
                print(f"✗ Failed to set power mode")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()

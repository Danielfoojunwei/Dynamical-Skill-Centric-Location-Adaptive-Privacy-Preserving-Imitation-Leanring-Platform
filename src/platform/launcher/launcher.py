"""
Dynamical.ai Platform Launcher

Windows desktop application for launching and managing the Dynamical.ai platform.
Provides system tray integration, service management, and easy access.

Features:
- System tray icon with quick actions
- Auto-start on Windows login
- Service status monitoring
- Log viewing
- Settings management

Requirements:
    pip install pystray pillow psutil requests

Usage:
    python launcher.py
    
Build executable:
    pip install pyinstaller
    pyinstaller --onefile --windowed --icon=dynamical.ico launcher.py
"""

import os
import sys
import json
import time
import subprocess
import threading
import webbrowser
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dynamical_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False
    logger.warning("pystray/PIL not available - system tray disabled")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import winreg
    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LauncherConfig:
    """Launcher configuration."""
    
    # Paths
    install_dir: str = ""
    data_dir: str = ""
    log_dir: str = ""
    
    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    
    # Frontend settings
    frontend_port: int = 3000
    
    # Startup
    auto_start: bool = False
    start_minimized: bool = False
    
    # Services
    auto_start_api: bool = True
    auto_start_frontend: bool = True
    
    def __post_init__(self):
        if not self.install_dir:
            self.install_dir = str(Path(__file__).parent.absolute())
        if not self.data_dir:
            self.data_dir = str(Path.home() / ".dynamical")
        if not self.log_dir:
            self.log_dir = str(Path(self.data_dir) / "logs")
    
    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"
    
    @property
    def frontend_url(self) -> str:
        return f"http://127.0.0.1:{self.frontend_port}"
    
    def save(self, path: str = None):
        """Save configuration to file."""
        path = path or str(Path(self.data_dir) / "launcher_config.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = None) -> "LauncherConfig":
        """Load configuration from file."""
        if path is None:
            path = str(Path.home() / ".dynamical" / "launcher_config.json")
        
        if Path(path).exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return cls()


# =============================================================================
# Service Manager
# =============================================================================

class ServiceStatus:
    """Service status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class Service:
    """Managed service."""
    name: str
    command: list
    working_dir: str = ""
    status: str = ServiceStatus.STOPPED
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    started_at: Optional[str] = None
    log_file: Optional[str] = None


class ServiceManager:
    """Manages platform services."""
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.services: Dict[str, Service] = {}
        self._stop_event = threading.Event()
        
        # Ensure directories exist
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self._init_services()
    
    def _init_services(self):
        """Initialize service definitions."""
        # API Server
        self.services["api"] = Service(
            name="API Server",
            command=[
                sys.executable, "-m", "uvicorn",
                "src.platform.api.main:app",
                "--host", self.config.api_host,
                "--port", str(self.config.api_port),
            ],
            working_dir=self.config.install_dir,
            log_file=str(Path(self.config.log_dir) / "api.log"),
        )
        
        # Frontend Server (simple HTTP server for static files)
        frontend_dir = str(Path(self.config.install_dir) / "platform" / "frontend")
        self.services["frontend"] = Service(
            name="Frontend Server",
            command=[
                sys.executable, "-m", "http.server",
                str(self.config.frontend_port),
                "--directory", frontend_dir,
            ],
            working_dir=frontend_dir,
            log_file=str(Path(self.config.log_dir) / "frontend.log"),
        )
    
    def start_service(self, service_id: str) -> bool:
        """Start a service."""
        service = self.services.get(service_id)
        if not service:
            logger.error(f"Service {service_id} not found")
            return False
        
        if service.status == ServiceStatus.RUNNING:
            logger.info(f"Service {service_id} already running")
            return True
        
        try:
            service.status = ServiceStatus.STARTING
            logger.info(f"Starting {service.name}...")
            
            # Open log file
            log_file = None
            if service.log_file:
                log_file = open(service.log_file, 'a')
            
            # Start process
            service.process = subprocess.Popen(
                service.command,
                cwd=service.working_dir or None,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )
            
            service.pid = service.process.pid
            service.started_at = datetime.now().isoformat()
            service.status = ServiceStatus.RUNNING
            
            logger.info(f"{service.name} started (PID: {service.pid})")
            return True
            
        except Exception as e:
            service.status = ServiceStatus.ERROR
            logger.error(f"Failed to start {service.name}: {e}")
            return False
    
    def stop_service(self, service_id: str) -> bool:
        """Stop a service."""
        service = self.services.get(service_id)
        if not service:
            return False
        
        if service.status != ServiceStatus.RUNNING:
            return True
        
        try:
            service.status = ServiceStatus.STOPPING
            logger.info(f"Stopping {service.name}...")
            
            if service.process:
                service.process.terminate()
                try:
                    service.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    service.process.kill()
            
            service.status = ServiceStatus.STOPPED
            service.process = None
            service.pid = None
            
            logger.info(f"{service.name} stopped")
            return True
            
        except Exception as e:
            service.status = ServiceStatus.ERROR
            logger.error(f"Failed to stop {service.name}: {e}")
            return False
    
    def restart_service(self, service_id: str) -> bool:
        """Restart a service."""
        self.stop_service(service_id)
        time.sleep(1)
        return self.start_service(service_id)
    
    def start_all(self):
        """Start all services."""
        if self.config.auto_start_api:
            self.start_service("api")
            time.sleep(2)  # Wait for API to initialize
        
        if self.config.auto_start_frontend:
            self.start_service("frontend")
    
    def stop_all(self):
        """Stop all services."""
        for service_id in self.services:
            self.stop_service(service_id)
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all services."""
        status = {}
        for service_id, service in self.services.items():
            # Check if process is still running
            if service.process and service.status == ServiceStatus.RUNNING:
                if service.process.poll() is not None:
                    service.status = ServiceStatus.STOPPED
                    service.process = None
                    service.pid = None
            
            status[service_id] = {
                "name": service.name,
                "status": service.status,
                "pid": service.pid,
                "started_at": service.started_at,
            }
        return status
    
    def check_api_health(self) -> bool:
        """Check if API is healthy."""
        if not HAS_REQUESTS:
            return False
        
        try:
            response = requests.get(
                f"{self.config.api_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# =============================================================================
# System Tray Application
# =============================================================================

class TrayApp:
    """System tray application."""
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.service_manager = ServiceManager(config)
        self.icon = None
        self._running = False
    
    def create_icon_image(self, size: int = 64) -> 'Image':
        """Create tray icon image."""
        # Create a simple icon (purple circle with 'D')
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw purple circle
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill='#7c3aed'
        )
        
        # Draw 'D' letter
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", int(size * 0.5))
        except:
            font = ImageFont.load_default()
        
        text = "D"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 2
        draw.text((x, y), text, fill='white', font=font)
        
        return image
    
    def create_menu(self) -> 'pystray.Menu':
        """Create tray menu."""
        status = self.service_manager.get_status()
        api_running = status.get("api", {}).get("status") == ServiceStatus.RUNNING
        frontend_running = status.get("frontend", {}).get("status") == ServiceStatus.RUNNING
        
        return pystray.Menu(
            pystray.MenuItem(
                "Dynamical.ai Platform",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Open Dashboard",
                self.open_dashboard,
                default=True
            ),
            pystray.MenuItem(
                "Open API Docs",
                self.open_api_docs
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                f"API Server {'✓' if api_running else '✗'}",
                pystray.Menu(
                    pystray.MenuItem("Start", lambda: self.start_service("api")),
                    pystray.MenuItem("Stop", lambda: self.stop_service("api")),
                    pystray.MenuItem("Restart", lambda: self.restart_service("api")),
                )
            ),
            pystray.MenuItem(
                f"Frontend Server {'✓' if frontend_running else '✗'}",
                pystray.Menu(
                    pystray.MenuItem("Start", lambda: self.start_service("frontend")),
                    pystray.MenuItem("Stop", lambda: self.stop_service("frontend")),
                    pystray.MenuItem("Restart", lambda: self.restart_service("frontend")),
                )
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Start All Services",
                self.start_all_services
            ),
            pystray.MenuItem(
                "Stop All Services",
                self.stop_all_services
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "View Logs",
                self.open_logs
            ),
            pystray.MenuItem(
                "Settings",
                pystray.Menu(
                    pystray.MenuItem(
                        "Start with Windows",
                        self.toggle_autostart,
                        checked=lambda _: self.config.auto_start
                    ),
                    pystray.MenuItem(
                        "Open Config Folder",
                        self.open_config_folder
                    ),
                )
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Exit",
                self.quit
            ),
        )
    
    def open_dashboard(self):
        """Open dashboard in browser."""
        webbrowser.open(f"http://127.0.0.1:{self.config.frontend_port}")
    
    def open_api_docs(self):
        """Open API documentation."""
        webbrowser.open(f"{self.config.api_url}/docs")
    
    def start_service(self, service_id: str):
        """Start a service."""
        self.service_manager.start_service(service_id)
        self.update_menu()
    
    def stop_service(self, service_id: str):
        """Stop a service."""
        self.service_manager.stop_service(service_id)
        self.update_menu()
    
    def restart_service(self, service_id: str):
        """Restart a service."""
        self.service_manager.restart_service(service_id)
        self.update_menu()
    
    def start_all_services(self):
        """Start all services."""
        self.service_manager.start_all()
        self.update_menu()
    
    def stop_all_services(self):
        """Stop all services."""
        self.service_manager.stop_all()
        self.update_menu()
    
    def open_logs(self):
        """Open log folder."""
        log_dir = Path(self.config.log_dir)
        if sys.platform == 'win32':
            os.startfile(str(log_dir))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(log_dir)])
        else:
            subprocess.run(['xdg-open', str(log_dir)])
    
    def open_config_folder(self):
        """Open config folder."""
        config_dir = Path(self.config.data_dir)
        if sys.platform == 'win32':
            os.startfile(str(config_dir))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(config_dir)])
        else:
            subprocess.run(['xdg-open', str(config_dir)])
    
    def toggle_autostart(self):
        """Toggle auto-start setting."""
        self.config.auto_start = not self.config.auto_start
        self.config.save()
        
        if HAS_WINREG and sys.platform == 'win32':
            self._set_windows_autostart(self.config.auto_start)
        
        self.update_menu()
    
    def _set_windows_autostart(self, enable: bool):
        """Set Windows auto-start registry."""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE
            )
            
            if enable:
                exe_path = sys.executable
                if getattr(sys, 'frozen', False):
                    exe_path = sys.executable
                else:
                    exe_path = f'"{sys.executable}" "{__file__}"'
                
                winreg.SetValueEx(
                    key, "DynamicalLauncher", 0,
                    winreg.REG_SZ, exe_path
                )
            else:
                try:
                    winreg.DeleteValue(key, "DynamicalLauncher")
                except FileNotFoundError:
                    pass
            
            winreg.CloseKey(key)
            logger.info(f"Auto-start {'enabled' if enable else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to set auto-start: {e}")
    
    def update_menu(self):
        """Update tray menu."""
        if self.icon:
            self.icon.menu = self.create_menu()
    
    def quit(self):
        """Quit application."""
        logger.info("Shutting down...")
        self._running = False
        self.service_manager.stop_all()
        if self.icon:
            self.icon.stop()
    
    def run(self):
        """Run the tray application."""
        if not HAS_TRAY:
            logger.error("pystray not available - cannot run tray app")
            return self.run_headless()
        
        self._running = True
        
        # Start services
        if self.config.auto_start_api or self.config.auto_start_frontend:
            threading.Thread(target=self.service_manager.start_all, daemon=True).start()
        
        # Create and run tray icon
        self.icon = pystray.Icon(
            "dynamical",
            self.create_icon_image(),
            "Dynamical.ai Platform",
            self.create_menu()
        )
        
        # Start menu update thread
        threading.Thread(target=self._menu_updater, daemon=True).start()
        
        logger.info("Launcher started")
        self.icon.run()
    
    def run_headless(self):
        """Run without tray (headless mode)."""
        logger.info("Running in headless mode")
        self._running = True
        self.service_manager.start_all()
        
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.service_manager.stop_all()
    
    def _menu_updater(self):
        """Periodically update menu to reflect service status."""
        while self._running:
            time.sleep(5)
            self.update_menu()


# =============================================================================
# Console Application (No GUI)
# =============================================================================

class ConsoleLauncher:
    """Console-based launcher for systems without GUI."""
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.service_manager = ServiceManager(config)
    
    def run(self):
        """Run console interface."""
        print("=" * 60)
        print("Dynamical.ai Platform Launcher")
        print("=" * 60)
        print()
        print(f"API URL: {self.config.api_url}")
        print(f"Frontend URL: http://127.0.0.1:{self.config.frontend_port}")
        print()
        print("Commands: start, stop, status, logs, quit")
        print()
        
        # Auto-start services
        if self.config.auto_start_api or self.config.auto_start_frontend:
            print("Starting services...")
            self.service_manager.start_all()
            print()
        
        while True:
            try:
                cmd = input("> ").strip().lower()
                
                if cmd == "start":
                    self.service_manager.start_all()
                    print("Services started")
                
                elif cmd == "stop":
                    self.service_manager.stop_all()
                    print("Services stopped")
                
                elif cmd == "status":
                    status = self.service_manager.get_status()
                    for service_id, info in status.items():
                        print(f"  {info['name']}: {info['status']}")
                        if info['pid']:
                            print(f"    PID: {info['pid']}")
                
                elif cmd == "logs":
                    print(f"Logs directory: {self.config.log_dir}")
                
                elif cmd in ("quit", "exit", "q"):
                    print("Shutting down...")
                    self.service_manager.stop_all()
                    break
                
                elif cmd == "help":
                    print("Commands:")
                    print("  start  - Start all services")
                    print("  stop   - Stop all services")
                    print("  status - Show service status")
                    print("  logs   - Show logs directory")
                    print("  quit   - Exit launcher")
                
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.service_manager.stop_all()
                break
            except EOFError:
                break


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamical.ai Platform Launcher")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--console", action="store_true", help="Run in console mode")
    parser.add_argument("--api-port", type=int, default=8080, help="API port")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port")
    parser.add_argument("--no-auto-start", action="store_true", help="Don't auto-start services")
    
    args = parser.parse_args()
    
    # Load configuration
    config = LauncherConfig.load()
    
    # Override with command line args
    if args.api_port:
        config.api_port = args.api_port
    if args.frontend_port:
        config.frontend_port = args.frontend_port
    if args.no_auto_start:
        config.auto_start_api = False
        config.auto_start_frontend = False
    
    # Save configuration
    config.save()
    
    # Run appropriate mode
    if args.console:
        launcher = ConsoleLauncher(config)
        launcher.run()
    elif args.headless or not HAS_TRAY:
        app = TrayApp(config)
        app.run_headless()
    else:
        app = TrayApp(config)
        app.run()


if __name__ == "__main__":
    main()

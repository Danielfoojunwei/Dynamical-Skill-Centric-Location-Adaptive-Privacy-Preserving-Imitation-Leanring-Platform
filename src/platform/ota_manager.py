import os
import json
import logging
import hashlib
import shutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
@dataclass
class FirmwarePackage:
    version: str
    target_device: str
    file_path: str
    checksum: str
    release_date: str

class OTAManager:
    """
    Manages Over-The-Air (OTA) updates for connected devices.
    Handles firmware versions and policy synchronization.
    """
    
    STORAGE_DIR = "ota_packages"
    
    def __init__(self):
        self._packages: Dict[str, List[FirmwarePackage]] = {}
        self._ensure_storage()
        self._load_manifest()
        
    def _ensure_storage(self):
        if not os.path.exists(self.STORAGE_DIR):
            os.makedirs(self.STORAGE_DIR)
            
    def _load_manifest(self):
        """Load available firmware packages from manifest."""
        manifest_path = os.path.join(self.STORAGE_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                    for target, pkgs in data.items():
                        self._packages[target] = [FirmwarePackage(**p) for p in pkgs]
            except Exception as e:
                logger.error(f"Failed to load OTA manifest: {e}")

    def _save_manifest(self):
        """Save current packages to manifest."""
        data = {}
        for target, pkgs in self._packages.items():
            data[target] = [p.__dict__ for p in pkgs]
            
        with open(os.path.join(self.STORAGE_DIR, "manifest.json"), 'w') as f:
            json.dump(data, f, indent=2)

    def register_package(self, file_path: str, target_device: str, version: str) -> bool:
        """Register a new firmware package."""
        try:
            # Calculate checksum
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()
            
            # Move to storage
            filename = f"{target_device}_{version}.bin"
            dest_path = os.path.join(self.STORAGE_DIR, filename)
            shutil.copy2(file_path, dest_path)
            
            pkg = FirmwarePackage(
                version=version,
                target_device=target_device,
                file_path=dest_path,
                checksum=checksum,
                release_date=datetime.utcnow().isoformat()
            )
            
            if target_device not in self._packages:
                self._packages[target_device] = []
            
            self._packages[target_device].append(pkg)
            self._save_manifest()
            logger.info(f"Registered FW {version} for {target_device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register package: {e}")
            return False

    def get_latest_version(self, target_device: str) -> Optional[FirmwarePackage]:
        """Get latest firmware for a target."""
        pkgs = self._packages.get(target_device, [])
        if not pkgs:
            return None
        # Simple semantic version sort (assumes x.y.z)
        return sorted(pkgs, key=lambda p: p.version, reverse=True)[0]

    def get_update_url(self, target_device: str, host_ip: str) -> Optional[str]:
        """Get the download URL for the latest firmware."""
        pkg = self.get_latest_version(target_device)
        if not pkg:
            return None
        filename = os.path.basename(pkg.file_path)
        return f"http://{host_ip}:8000/ota/{filename}"

# Global instance
ota_manager = OTAManager()

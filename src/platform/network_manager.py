import socket
import threading
import time
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("edge_platform.network")

@dataclass
class NetworkDevice:
    ip: str
    hostname: str
    type: str  # "DOGLOVE", "CAMERA", "UNKNOWN"
    last_seen: float
    metadata: Dict

class NetworkManager:
    """
    Manages network discovery and device connections.
    Uses UDP Broadcast for discovery.
    """
    DISCOVERY_PORT = 9875
    BROADCAST_INTERVAL = 5.0
    
    def __init__(self):
        self._devices: Dict[str, NetworkDevice] = {}
        self._stop_event = threading.Event()
        self._discovery_thread = None
        self._broadcast_thread = None
        self._sock = None
        
    def start(self):
        """Start discovery services."""
        self._stop_event.clear()
        
        # Setup UDP Broadcast Socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(('', self.DISCOVERY_PORT))
        except Exception as e:
            logger.error(f"Failed to bind discovery port: {e}")
            return

        # Start threads
        self._discovery_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._discovery_thread.start()
        
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._broadcast_thread.start()
        
        logger.info("Network Manager started")

    def stop(self):
        """Stop discovery services."""
        self._stop_event.set()
        if self._sock:
            self._sock.close()
        logger.info("Network Manager stopped")

    def get_devices(self) -> List[NetworkDevice]:
        """Get list of discovered devices."""
        # Prune old devices (> 30s)
        now = time.time()
        active_devices = []
        for ip, dev in list(self._devices.items()):
            if now - dev.last_seen < 30.0:
                active_devices.append(dev)
            else:
                del self._devices[ip]
        return active_devices

    def _broadcast_loop(self):
        """Periodically broadcast 'WHOIS' packet."""
        while not self._stop_event.is_set():
            try:
                msg = json.dumps({"cmd": "WHOIS", "source": "DYNAMICAL_EDGE_HOST"}).encode('utf-8')
                self._sock.sendto(msg, ('<broadcast>', self.DISCOVERY_PORT))
                time.sleep(self.BROADCAST_INTERVAL)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                time.sleep(5)

    def _listen_loop(self):
        """Listen for 'IAM' responses."""
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(1024)
                try:
                    msg = json.loads(data.decode('utf-8'))
                    # Assuming 'ip', 'hostname', 'dev_type' are extracted from 'msg'
                    # For now, using placeholder values or deriving from 'addr'
                    ip = addr[0]
                    hostname = msg.get("hostname", "unknown")
                    dev_type = msg.get("type", "UNKNOWN")

                    device = NetworkDevice(
                        ip=ip,
                        hostname=hostname,
                        type=dev_type,
                        last_seen=time.time(),
                        metadata=msg
                    )
                    
                    if ip not in self._devices:
                        logger.info(f"New device discovered: {hostname} ({ip}) [{dev_type}]")
                    
                    self._devices[ip] = device
                except json.JSONDecodeError:
                    logger.warning(f"Received malformed JSON from {addr[0]}")
                except Exception as e:
                    logger.error(f"Error processing received message from {addr[0]}: {e}")
            except socket.timeout:
                # Socket timeout is expected if no data is received
                pass
            except Exception as e:
                logger.error(f"Listen loop error: {e}")
                time.sleep(1) # Prevent busy-waiting on persistent errors

# Global instance
network_manager = NetworkManager()

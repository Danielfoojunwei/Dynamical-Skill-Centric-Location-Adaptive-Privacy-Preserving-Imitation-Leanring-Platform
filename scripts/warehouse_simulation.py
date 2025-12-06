import time
import threading
import json
import urllib.request
import urllib.error
import random
import sys
import statistics
from dataclasses import dataclass
from typing import List

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "default_insecure_key"
DURATION_SECONDS = 60  # Longer test for stability
LOG_INTERVAL = 1.0

# Warehouse Scenario
NUM_GLOVES = 3      # Wireless Users
NUM_CAMERAS = 12    # Wireless Cameras
NUM_ROBOTS = 3      # Mobile Robots

# Simulation Specs
GLOVE_RATE_HZ = 60  # 60Hz updates (simulating 120Hz might be too heavy for python threads)
CAMERA_RATE_HZ = 30 # 30Hz frames (metadata only)
ROBOT_RATE_HZ = 10  # 10Hz telemetry

# Metrics
metrics = {
    "requests": 0,
    "errors": 0,
    "latencies": [],
    "bytes_sent": 0
}
metrics_lock = threading.Lock()

def make_request(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers, method=method)
        if data:
            req.data = json.dumps(data).encode('utf-8')
            
        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            latency = (time.time() - start_time) * 1000
            
            with metrics_lock:
                metrics["requests"] += 1
                metrics["latencies"].append(latency)
                if data:
                    metrics["bytes_sent"] += len(req.data)
            
            return json.loads(response.read().decode('utf-8'))
            
    except Exception as e:
        with metrics_lock:
            metrics["errors"] += 1
        return None

class SimulatedDevice(threading.Thread):
    def __init__(self, dev_type, dev_id, rate_hz, stop_event):
        super().__init__()
        self.dev_type = dev_type
        self.dev_id = dev_id
        self.rate_hz = rate_hz
        self.stop_event = stop_event
        self.daemon = True

    def run(self):
        interval = 1.0 / self.rate_hz
        while not self.stop_event.is_set():
            start = time.time()
            
            # Simulate Device Activity
            if self.dev_type == "DOGLOVE":
                # Simulate Glove State (Joints + IMU)
                payload = {
                    "id": self.dev_id,
                    "timestamp": time.time(),
                    "joints": [random.random() for _ in range(21)],
                    "imu": {"acc": [0,0,9.8], "quat": [1,0,0,0]}
                }
                # In real system this is UDP, here we hit API to stress backend logic
                # or we could hit a specific endpoint if we had one for ingestion
                # For now, we'll simulate "keepalive/heartbeat" to the devices endpoint
                # to avoid overwhelming the HTTP server with 60Hz * 3 requests
                if random.random() < 0.1: # 10% chance to send full update via HTTP
                     make_request(f"/devices", "GET") 
                
            elif self.dev_type == "CAMERA":
                # Simulate Camera Metadata / Discovery
                # Cameras usually stream RTSP, but they also ping for status
                if random.random() < 0.05: # Occasional status check
                    make_request(f"/devices", "GET")

            elif self.dev_type == "ROBOT":
                # Robots check for OTA and Settings frequently
                if random.random() < 0.2:
                    make_request(f"/api/ota/check/ROBOT_VLA", "GET")
            
            # Sleep to maintain rate
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

def main():
    print(f"Starting Warehouse Simulation...")
    print(f"Scenario: {NUM_GLOVES} Gloves, {NUM_CAMERAS} Cameras, {NUM_ROBOTS} Robots")
    
    # 1. Start System
    print("Initializing Backend...")
    make_request("/system/start", "POST", {})
    time.sleep(2)
    
    stop_event = threading.Event()
    threads = []
    
    # 2. Spawn Devices
    print("Spawning Devices...")
    
    # Gloves
    for i in range(NUM_GLOVES):
        t = SimulatedDevice("DOGLOVE", f"glove_{i}", GLOVE_RATE_HZ, stop_event)
        t.start()
        threads.append(t)
        
    # Cameras
    for i in range(NUM_CAMERAS):
        t = SimulatedDevice("CAMERA", f"cam_{i}", CAMERA_RATE_HZ, stop_event)
        t.start()
        threads.append(t)
        
    # Robots
    for i in range(NUM_ROBOTS):
        t = SimulatedDevice("ROBOT", f"robot_{i}", ROBOT_RATE_HZ, stop_event)
        t.start()
        threads.append(t)
        
    print(f"Simulation Running with {len(threads)} active device threads.")
    
    # 3. Monitor Loop
    start_time = time.time()
    try:
        while time.time() - start_time < DURATION_SECONDS:
            time.sleep(LOG_INTERVAL)
            elapsed = time.time() - start_time
            
            # Get System Stats
            sys_stats = make_request("/system/stats")
            tflops = sys_stats.get("tflops_used", 0) if sys_stats else 0
            
            with metrics_lock:
                req_count = metrics["requests"]
                err_count = metrics["errors"]
                avg_lat = statistics.mean(metrics["latencies"][-100:]) if metrics["latencies"] else 0
                
            sys.stdout.write(f"\r[{elapsed:.0f}s] TFLOPS: {tflops:.1f} | Req: {req_count} | Err: {err_count} | Latency: {avg_lat:.1f}ms")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\nStopping simulation...")
        for t in threads:
            t.join()
            
    # 4. Final Report
    print("\n" + "="*40)
    print("WAREHOUSE SIMULATION REPORT")
    print("="*40)
    print(f"Duration:       {DURATION_SECONDS}s")
    print(f"Total Requests: {metrics['requests']}")
    print(f"Total Errors:   {metrics['errors']}")
    print(f"Avg Latency:    {statistics.mean(metrics['latencies']) if metrics['latencies'] else 0:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()

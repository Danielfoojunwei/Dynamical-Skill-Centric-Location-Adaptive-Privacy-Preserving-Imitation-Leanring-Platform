import sys
import os
import time
import threading
import requests
import subprocess
try:
    from uvicorn import Config, Server
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    print("Warning: uvicorn not installed. Skipping server runtime check.")

# Add parent to path
sys.path.insert(0, os.getcwd())

# Mock Pi0
from unittest.mock import MagicMock
sys.modules["spatial_intelligence.pi0.model"] = MagicMock()
sys.modules["spatial_intelligence.pi0.model"].Pi0 = MagicMock()

# Import main app to check syntax
try:
    from src.platform.api.main import app
    BACKEND_IMPORT_SUCCESS = True
except ImportError as e:
    BACKEND_IMPORT_SUCCESS = False
    BACKEND_IMPORT_ERROR = str(e)

def run_server():
    if not UVICORN_AVAILABLE: return
    config = Config(app=app, host="127.0.0.1", port=8000, log_level="error")
    server = Server(config)
    server.run()

def verify_platform():
    print("=" * 60)
    print("VERIFYING DYNAMICAL EDGE PLATFORM")
    print("=" * 60)
    
    # 1. Verify Backend
    print("\n[Backend] Verifying API...")
    if BACKEND_IMPORT_SUCCESS:
        print("[PASS] Backend Import: PASSED")
    else:
        print(f"[FAIL] Backend Import: FAILED ({BACKEND_IMPORT_ERROR})")
        
    if UVICORN_AVAILABLE:
        print("[Backend] Starting API Server...")
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server
        time.sleep(2)
        
        try:
            resp = requests.get("http://127.0.0.1:8000/health")
            if resp.status_code == 200:
                print("[PASS] API Health Check: PASSED")
            else:
                print(f"[FAIL] API Health Check: FAILED ({resp.status_code})")
                
            # Check Stats Endpoint
            headers = {"X-API-Key": "neuracore_live_8f4a2b9c1d3e5f6g7h8i9j0k1l2m3n4o"}
            resp = requests.get("http://127.0.0.1:8000/system/stats", headers=headers)
            if resp.status_code == 200:
                stats = resp.json()
                print(f"[PASS] Stats Endpoint: PASSED (Status: {stats['status']})")
            else:
                print(f"[FAIL] Stats Endpoint: FAILED ({resp.status_code})")
                
        except Exception as e:
            print(f"[FAIL] Backend Verification Failed: {e}")
    else:
        print("[WARN] Skipping Server Runtime Check (uvicorn missing)")
        
    # 2. Verify Frontend Build
    print("\n[Frontend] Building React App...")
    frontend_dir = os.path.join(os.getcwd(), "src", "platform", "ui")
    
    try:
        # Use shell=True for npm on Windows
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command", "npm run build"], 
            cwd=frontend_dir, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("[PASS] Frontend Build: PASSED")
            dist_path = os.path.join(frontend_dir, "dist", "index.html")
            if os.path.exists(dist_path):
                print(f"[PASS] Build Artifacts Found: {dist_path}")
            else:
                print("[FAIL] Build Artifacts Missing")
        else:
            print("[FAIL] Frontend Build: FAILED")
            print(result.stderr)
            
    except Exception as e:
        print(f"[FAIL] Frontend Verification Failed: {e}")

if __name__ == "__main__":
    verify_platform()

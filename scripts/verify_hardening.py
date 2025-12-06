import sys
import os
import time
import threading
import requests
import logging

# Add parent to path
sys.path.insert(0, os.getcwd())

# Mock Pi0
from unittest.mock import MagicMock
sys.modules["spatial_intelligence.pi0.model"] = MagicMock()
sys.modules["spatial_intelligence.pi0.model"].Pi0 = MagicMock()

def verify_hardening():
    print("=" * 60)
    print("VERIFYING SYSTEM HARDENING")
    print("=" * 60)
    
    # 1. Check .env
    if os.path.exists(".env"):
        print("[PASS] .env file found")
        try:
            with open(".env", "r") as f:
                content = f.read()
                if "API_KEY=" in content:
                    print("[PASS] API_KEY found in .env")
                else:
                    print("[FAIL] API_KEY missing in .env")
        except Exception as e:
            print(f"[FAIL] Could not read .env: {e}")
    else:
        print("[FAIL] .env file missing")
        return

    # 2. Check Logging Config
    # We can't easily check the running app's logging without starting it,
    # but we can check if the log file is created if we were to run it.
    # Instead, let's verify the code structure in main.py
    try:
        with open("edge_platform/api/main.py", "r") as f:
            content = f.read()
            if "logging.basicConfig" in content and "RotatingFileHandler" in content:
                print("[PASS] Structured logging configured in main.py")
            else:
                print("[FAIL] Structured logging missing in main.py")
            
            if "APIKeyHeader" in content and "Depends(get_api_key)" in content:
                print("[PASS] API Key Auth configured in main.py")
            else:
                print("[FAIL] API Key Auth missing in main.py")
    except Exception as e:
        print(f"[FAIL] Could not read main.py: {e}")

    # 3. Verify Frontend Auth
    try:
        with open("edge_platform/ui/src/api.js", "r") as f:
            content = f.read()
            if "X-API-Key" in content:
                print("[PASS] Frontend api.js sends X-API-Key header")
            else:
                print("[FAIL] Frontend api.js missing Auth header")
    except Exception as e:
        print(f"[FAIL] Could not read api.js: {e}")

if __name__ == "__main__":
    verify_hardening()

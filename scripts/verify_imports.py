import sys
import os

# Add root to path
sys.path.insert(0, os.getcwd())

print("Verifying imports...")

try:
    from src.pipeline.integrated_pipeline import IntegratedDataPipeline
    print("[OK] src.pipeline.integrated_pipeline imported")
except ImportError as e:
    print(f"[FAIL] Failed to import src.pipeline.integrated_pipeline: {e}")

try:
    from src.platform.api.main import app
    print("[OK] src.platform.api.main imported")
except ImportError as e:
    print(f"[FAIL] Failed to import src.platform.api.main: {e}")

try:
    from src.drivers.cameras import MultiViewCameraRig
    print("[OK] src.drivers.cameras imported")
except ImportError as e:
    print(f"[FAIL] Failed to import src.drivers.cameras: {e}")

try:
    from src.core.neuracore_client import get_client
    print("[OK] src.core.neuracore_client imported")
except ImportError as e:
    print(f"[FAIL] Failed to import src.core.neuracore_client: {e}")

print("Verification complete.")

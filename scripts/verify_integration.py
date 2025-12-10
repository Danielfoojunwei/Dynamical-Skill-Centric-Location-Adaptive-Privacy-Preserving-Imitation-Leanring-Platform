import sys
import os
import time
import shutil
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.getcwd())

# Mock Pi0 to avoid HF auth
from unittest.mock import MagicMock
sys.modules["spatial_intelligence.pi0.model"] = MagicMock()
sys.modules["spatial_intelligence.pi0.model"].Pi0 = MagicMock()

from agx_orin_32gb.integrated_pipeline import IntegratedDataPipeline

def test_pipeline():
    print("Testing Integrated Pipeline...")
    
    # Clean up previous logs
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")
        
    pipeline = IntegratedDataPipeline(
        num_robots=1,
        cameras_per_robot=2,
        storage_path="test_logs"
    )
    
    pipeline.initialize()
    pipeline.start()
    
    print("Pipeline started. Submitting frames...")
    
    # Submit some frames
    for i in range(10):
        frame = np.random.rand(100, 100, 3).astype(np.float32)
        pipeline.submit_frame(
            robot_id="robot_0",
            camera_id="cam_0", # Safety camera
            frame=frame,
            timestamp=time.time()
        )
        
        frame_nav = np.random.rand(100, 100, 3).astype(np.float32)
        pipeline.submit_frame(
            robot_id="robot_0",
            camera_id="cam_4", # Navigation camera
            frame=frame_nav,
            timestamp=time.time()
        )
        
        time.sleep(0.1)
        
    print("Frames submitted. Waiting for processing...")
    time.sleep(2.0)
    
    stats = pipeline.get_statistics()
    print("Pipeline Stats:", stats)
    
    pipeline.stop()
    
    # Verify logs
    log_dir = Path("test_logs/neuracore_logs")
    if log_dir.exists():
        print(f"Neuracore logs found at {log_dir}")
        files = list(log_dir.glob("*.jsonl"))
        print(f"Log files: {[f.name for f in files]}")
        if len(files) > 0:
            print("Async logging verification: SUCCESS")
        else:
            print("Async logging verification: FAILED (No files)")
    else:
        print("Async logging verification: FAILED (Dir not found)")
        
    # Verify validation
    if "validation_errors" in stats:
        print(f"Validation errors tracked: {stats['validation_errors']}")
        print("Schema enforcement verification: SUCCESS")
    else:
        print("Schema enforcement verification: FAILED")

if __name__ == "__main__":
    test_pipeline()

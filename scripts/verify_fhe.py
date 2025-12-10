import sys
import os
import time
import shutil
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.getcwd())

# Mock Pi0
from unittest.mock import MagicMock
sys.modules["spatial_intelligence.pi0.model"] = MagicMock()
sys.modules["spatial_intelligence.pi0.model"].Pi0 = MagicMock()

from agx_orin_32gb.integrated_pipeline import IntegratedDataPipeline
from moai.moai_fhe import MoaiFHESystem
from moai.n2he import Ciphertext

class MockMoaiSystem:
    def __init__(self):
        self.frames = []
        
    def start(self):
        pass
        
    def stop(self):
        pass
        
    def submit_frame(self, robot_id, camera_id, frame, timestamp):
        self.frames.append({
            "robot_id": robot_id,
            "camera_id": camera_id,
            "frame": frame,
            "timestamp": timestamp
        })

def test_fhe_pipeline():
    print("Testing FHE Integrated Pipeline...")
    
    # Clean up
    if os.path.exists("test_fhe_logs"):
        try:
            shutil.rmtree("test_fhe_logs")
        except PermissionError:
            print("Warning: Could not clean up test_fhe_logs")
        
    # Setup FHE System
    fhe_system = MoaiFHESystem(use_mock=True)
    moai_system = MockMoaiSystem()
    
    pipeline = IntegratedDataPipeline(
        num_robots=1,
        cameras_per_robot=2,
        storage_path="test_fhe_logs",
        learning_batch_size=1, # Process immediately
        learning_batch_timeout=0.1
    )
    
    # Inject systems
    pipeline.initialize(moai_system=moai_system, fhe_system=fhe_system)
    pipeline.start()
    
    print("Pipeline started. Submitting frames for Learning (Tier 3)...")
    
    # Submit frames that should go to learning queue (every 3rd frame)
    # We submit 10 frames to ensure some get routed to learning
    for i in range(10):
        frame = np.random.rand(100, 100, 3).astype(np.float32)
        pipeline.submit_frame(
            robot_id="robot_0",
            camera_id="cam_10", # Learning camera (not safety/nav)
            frame=frame,
            timestamp=time.time()
        )
        time.sleep(0.05)
        
    print("Frames submitted. Waiting for processing...")
    time.sleep(3.0) # Wait for batch processing
    
    pipeline.stop()
    
    # Verify Encryption
    print("\nVerifying MOAI System received encrypted frames...")
    encrypted_count = 0
    for f in moai_system.frames:
        if isinstance(f["frame"], Ciphertext):
            encrypted_count += 1
            
    if encrypted_count > 0:
        print(f"SUCCESS: {encrypted_count} frames were encrypted before reaching MOAI.")
    else:
        print("FAILED: No encrypted frames found in MOAI system.")
        
    # Verify Logs
    print("\nVerifying Neuracore Logs...")
    log_dir = Path("test_fhe_logs/neuracore_logs")
    if log_dir.exists():
        files = list(log_dir.glob("*.jsonl"))
        if len(files) > 0:
             print(f"SUCCESS: Logs generated: {[f.name for f in files]}")
        else:
             print("FAILED: No log files found.")
    else:
        print("FAILED: Log directory not found.")

if __name__ == "__main__":
    test_fhe_pipeline()

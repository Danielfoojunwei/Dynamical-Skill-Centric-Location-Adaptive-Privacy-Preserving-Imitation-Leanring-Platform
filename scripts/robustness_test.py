import time
import requests
import random
import sys
import os

# Ensure we can import src
sys.path.append(os.getcwd())

from src.platform.edge.dyglove_sdk import DYGloveSimulator
from src.platform.cloud.vendor_adapter import SimulatedVendorAdapter as MockVendorAdapter

def run_robustness_test():
    print("=== Starting Full-Stack Robustness Test (Glove -> Cloud) ===")
    
    # 1. Simulate Sensor Input (Glove)
    print("\n[1/4] Simulating DYGlove Input...")
    glove = DYGloveSimulator()
    glove.connect()
    state = glove.read_state()
    print(f"   Glove State: {state}")
    
    # 2. Simulate Local Inference (Vendor Adapter)
    print("\n[2/4] Running Local Inference (Mock Vendor Adapter)...")
    adapter = MockVendorAdapter()
    adapter.load_weights("mock_model.bin")
    
    # Create mock observation
    obs = {
        "joint_positions": state.joint_angles,
        "camera_rgb": "mock_frame_data"
    }
    
    action = adapter.predict(obs)
    print(f"   Predicted Action: {action['action'][:3]}... (Confidence: {action['confidence']})")
    
    # 3. Simulate "Intervention" & Cloud Upload
    print("\n[3/4] Triggering Cloud Upload (Federated Learning)...")
    
    # We'll hit the local API endpoint (assuming server is running, or just mock the call)
    # Since we can't easily spin up the uvicorn server in this script without blocking,
    # we will instantiate the modules directly to verify the logic flow.
    
    from src.platform.cloud.secure_aggregator import SecureAggregator
    aggregator = SecureAggregator()
    
    gradients = adapter.get_gradient_buffer()
    print(f"   Got Gradients: Shape {gradients.shape}")
    
    encrypted_blob = aggregator.encrypt_gradients(gradients)
    print(f"   Encrypted Blob Size: {len(encrypted_blob)} bytes")
    
    upload_success = aggregator.upload_update(encrypted_blob)
    
    if upload_success:
        print("\n[4/4] Cloud Upload SUCCESS")
    else:
        print("\n[4/4] Cloud Upload FAILED")
        sys.exit(1)

    print("\n=== Robustness Test PASSED ===")

if __name__ == "__main__":
    run_robustness_test()

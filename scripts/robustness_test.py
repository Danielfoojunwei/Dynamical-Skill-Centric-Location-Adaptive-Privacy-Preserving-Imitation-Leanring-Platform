import time
import requests
import random
import sys
import os
import numpy as np

# Ensure we can import src
sys.path.append(os.getcwd())

from src.platform.edge.dyglove_sdk import DYGloveSimulator
from src.platform.cloud.vendor_adapter_real import VendorAdapter


class MockVendorAdapter(VendorAdapter):
    """Mock vendor adapter for testing (v0.9.0)."""

    def __init__(self):
        self._gradient_buffer = np.random.randn(1000).astype(np.float32)
        self._model_loaded = False

    def connect(self):
        return True

    def disconnect(self):
        pass

    def load_weights(self, weights_path: str):
        self._model_loaded = True

    def infer(self, observation: dict) -> dict:
        return {
            "action": np.random.randn(7).tolist(),
            "confidence": 0.85
        }

    def predict(self, observation: dict) -> dict:
        return self.infer(observation)

    def get_gradient_buffer(self) -> np.ndarray:
        return self._gradient_buffer

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

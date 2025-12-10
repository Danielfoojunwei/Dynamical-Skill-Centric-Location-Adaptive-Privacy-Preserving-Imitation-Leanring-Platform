"""Run Pi0 VLA model on AGX Orin 32GB."""

import os
import sys
import time
import argparse
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: Torch not found. Running in dry-run verification mode.")
    # Mock for dry run
    class MockObj:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self
        def to(self, *args): return self
        @property
        def shape(self): return (1, 16, 7)
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return "float32"
        
    class MockLib:
        pass
        
    torch = MockLib()
    torch.cuda = MockLib()
    torch.cuda.is_available = lambda: False
    torch.device = lambda x: x
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.randn = lambda *args, **kwargs: MockObj()
    torch.ones = lambda *args, **kwargs: MockObj()
    torch.randint = lambda *args, **kwargs: MockObj()
    torch.no_grad = lambda: type('obj', (object,), {'__enter__': lambda x: None, '__exit__': lambda x,y,z: None})()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.memory_monitor import MemoryMonitor
from src.spatial_intelligence.pi0.model import Pi0
from src.platform.logging_utils import get_logger

logger = get_logger(__name__)

# Configuration
NVME_PATH = "/data"
HF_HOME = os.path.join(NVME_PATH, "huggingface")
os.environ["HF_HOME"] = HF_HOME

def run_pi0_inference(test_load_only: bool = False):
    """Run Pi0 inference."""
    logger.info("=" * 60)
    logger.info("Pi0 VLA Model - AGX Orin 32GB")
    logger.info("=" * 60)
    logger.info(f"Storage Path: {HF_HOME}")
    
    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN not set. Model download may fail if not cached.")
    
    # Initialize memory monitor
    monitor = MemoryMonitor()
    try:
        monitor.check_memory(log=True)
    except Exception as e:
        logger.error(f"Memory check failed: {e}")

    logger.info("Initializing Pi0 model...")
    try:
        model = Pi0(
            action_dim=7,
            action_horizon=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.to(model.device)
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    if test_load_only:
        logger.info("Load test complete.")
        return

    logger.info("Running dummy inference...")
    batch_size = 1
    num_cams = 3
    
    # Create dummy inputs
    images = torch.randn(batch_size, num_cams, 3, 224, 224, device=model.device, dtype=model.dtype)
    image_masks = torch.ones(batch_size, num_cams, device=model.device, dtype=model.dtype)
    
    # Dummy language tokens (would come from tokenizer)
    language_tokens = torch.randint(0, 1000, (batch_size, 128), device=model.device)
    language_masks = torch.ones(batch_size, 128, device=model.device, dtype=model.dtype)
    
    # Dummy proprioception
    proprio = torch.randn(batch_size, 21, device=model.device, dtype=model.dtype)
    
    start_time = time.time()
    with torch.no_grad():
        actions = model(images, image_masks, language_tokens, language_masks, proprio)
    end_time = time.time()
    
    logger.info(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
    logger.info(f"Output shape: {actions.shape}")
    
    monitor.check_memory(log=True)
    logger.info("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-load", action="store_true", help="Only test model loading")
    args = parser.parse_args()
    
    run_pi0_inference(test_load_only=args.test_load)

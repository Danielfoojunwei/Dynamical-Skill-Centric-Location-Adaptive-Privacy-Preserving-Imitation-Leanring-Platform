import sys
import os
import time
import logging
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.integrated_pipeline import IntegratedDataPipeline as IntegratedPipeline
from src.platform.logging_utils import get_logger

logger = get_logger("smoke_test")

def test_pipeline_simulation():
    """
    Smoke test to verify pipeline initializes and runs in simulation mode.
    """
    logger.info("Starting Pipeline Smoke Test...")
    
    # Force simulation mode via env var (override config for this test)
    os.environ["SIMULATION_MODE"] = "true"
    
    try:
        pipeline = IntegratedPipeline()
        logger.info("Pipeline initialized successfully.")
        
        # Run pipeline for a few seconds
        logger.info("Starting pipeline...")
        pipeline.start()
        time.sleep(2)
        logger.info("Stopping pipeline...")
        pipeline.stop()
            
        logger.info("Pipeline ran successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
        pytest.fail(f"Pipeline crashed: {e}")
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    # Manual run
    test_pipeline_simulation()

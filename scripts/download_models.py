#!/usr/bin/env python3
"""
Model Downloader for Dynamical Edge System
Downloads required model weights from Hugging Face or other sources.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_file(url: str, dest: Path):
    import requests
    logger.info(f"Downloading {url} to {dest}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Don't exit, just log error so we can try others
        
def setup_paligemma():
    """Setup PaliGemma weights (placeholder for now)."""
    target_dir = MODELS_DIR / "paligemma"
    target_dir.mkdir(exist_ok=True)
    
    # In reality, this would likely use huggingface_hub
    # For now, we'll create a dummy file to simulate presence if not exists
    # or print instructions
    
    logger.info("Checking PaliGemma weights...")
    if not (target_dir / "config.json").exists():
        logger.warning("PaliGemma weights not found.")
        logger.info("Please run: huggingface-cli download google/paligemma-3b-pt-224 --local-dir models/paligemma")
        # Create a marker file so system knows we checked
        with open(target_dir / "README.txt", "w") as f:
            f.write("Download weights here.")

def setup_safety_models():
    """Setup Safety/Hazard models."""
    target_dir = MODELS_DIR / "safety"
    target_dir.mkdir(exist_ok=True)
    # Placeholder for future safety models

def main():
    logger.info("Starting Model Setup...")
    
    # Check dependencies
    try:
        import requests
    except ImportError:
        logger.error("requests library not found. Run scripts/install_python_deps.sh first.")
        sys.exit(1)
        
    setup_paligemma()
    setup_safety_models()
    
    logger.info("Model setup check complete.")

if __name__ == "__main__":
    main()

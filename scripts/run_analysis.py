import sys
import os

# Add parent to path
sys.path.insert(0, os.getcwd())

# Mock Pi0
from unittest.mock import MagicMock
sys.modules["spatial_intelligence.pi0.model"] = MagicMock()
sys.modules["spatial_intelligence.pi0.model"].Pi0 = MagicMock()

from agx_orin_32gb.integrated_pipeline import analyze_complete_system

if __name__ == "__main__":
    analyze_complete_system()

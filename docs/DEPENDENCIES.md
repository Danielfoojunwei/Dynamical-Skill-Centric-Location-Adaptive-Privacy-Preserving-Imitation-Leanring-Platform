# System Dependencies & Setup Guide

## Overview
This system is designed to run on NVIDIA Jetson AGX Orin but supports a **Simulation Mode** for development on standard Linux/Windows machines.

## 1. Quick Start
```bash
# 1. Install OS dependencies (requires sudo)
sudo ./scripts/install_os_deps.sh

# 2. Install Python dependencies
./scripts/install_python_deps.sh

# 3. Download Models (or setup placeholders)
./scripts/download_models.py

# 4. Configure
# Edit config/config.yaml to set 'simulation_mode: true' if needed.

# 5. Launch
./launch_orin.sh
```

## 2. Python Dependencies
Managed via `requirements.txt`.
Key packages:
- `torch`, `torchvision` (ML Core)
- `transformers` (Pi0 / VLA)
- `fastapi`, `uvicorn` (Backend)
- `opencv-python`, `scipy` (Vision/Processing)

**Note for Jetson:**
PyTorch is best installed from the NVIDIA JetPack index. The install script attempts standard pip install, which works for x86 but might need adjustment for ARM64/Jetson if not using the official container.

## 3. External Models
Models are stored in `models/`.
- **PaliGemma**: Required for VLA. Run `scripts/download_models.py` to see instructions.
- **Safety Models**: (Future) Custom hazard detectors.

## 4. Vendor SDKs
- **Daimon VTLA**: Proprietary. If SDK is missing, the system automatically falls back to `SIMULATION_MODE`.
- **DYGlove**: Uses standard `pyserial` / `websockets`.

## 5. Troubleshooting
- **Missing `libopencv`**: Run `install_os_deps.sh`.
- **ImportError**: Ensure you activated the venv: `source venv/bin/activate`.

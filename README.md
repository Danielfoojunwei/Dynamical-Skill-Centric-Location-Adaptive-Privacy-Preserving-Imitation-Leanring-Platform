# Dynamical Robotics Edge Platform (AGX Orin 32GB)

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-Production%20Prototype-orange)
![CI](https://github.com/dynamical-ai/edge-platform/actions/workflows/ci.yml/badge.svg)

> [!IMPORTANT]
> **SAFETY DISCLAIMER**: This software is a **prototype** and is **NOT a certified safety controller**. It provides "safety-adjacent" features (hazard detection, reflex stops) but **MUST** be used in conjunction with hardware E-stops, physical barriers, and standard industrial safety practices. Do not rely solely on this software for human safety.

## Overview
This is the production-ready software stack for the **Dynamical Robotics VTLA System**. It runs on the NVIDIA Jetson AGX Orin (32GB) and orchestrates the entire robotic pipeline, including:

-   **Perception**: Multi-view camera triangulation (OpenCV).
-   **Control**: Daimon Robotics VTLA Robot & DYGlove Haptic Interface.
-   **Compute**: Real-time inference on AGX Orin.
-   **Security**: FHE-audited data pipeline.
-   **UI**: React-based dashboard for monitoring and control.

## System Requirements
-   **Hardware**: NVIDIA Jetson AGX Orin (32GB), Daimon VTLA Robot, DYGlove, 2x IP Cameras.
-   **OS**: Ubuntu 20.04 (JetPack 5.x) or Windows (for Dev/Sim).
-   **Dependencies**: Python 3.8+, Node.js 16+, OpenCV, PyTorch.

## Quick Start
1.  **Install Dependencies**:
    ```bash
    ./setup_orin.sh  # On Linux/Orin (Installs OS + Python deps)
    # OR
    scripts/install_python_deps.sh  # Python only
    scripts/install_vision_deps.sh  # Vision SDKs (MMPose, RTMW3D)
    ```

2.  **Download Models**:
    ```bash
    python scripts/download_models.py
    ```

2.  **Launch System**:
    ```bash
    ./launch_orin.sh  # On Linux/Orin
    # OR
    ./launch_dev.bat  # On Windows
    ```

3.  **Access Dashboard**:
    Open `http://localhost:3000` in your browser.

## Documentation
-   [Setup Guide & FAQ](docs/SETUP_GUIDE.md): Step-by-step installation and troubleshooting.
-   [Developer Notes](docs/DEVELOPER_NOTES.md): Architecture, config, and testing.
-   [Dependencies](docs/DEPENDENCIES.md): Detailed dependency list.
-   [System Overview](docs/SYSTEM_OVERVIEW.md): Detailed explanation of how it works.
-   [Bill of Materials](BOM.md): Hardware list.

## Hardware Integration
-   **Robot**: Uses `src/drivers/daimon_vtla.py`. Requires Daimon SDK.
-   **Glove**: Uses `src/drivers/dyglove.py`. Requires WiFi 6E.
-   **Cameras**: Uses `src/drivers/cameras.py`. Requires ONVIF calibration.

## Support
## Configuration
System settings are in `config/config.yaml`. You can enable **Simulation Mode** there or via `SIMULATION_MODE=true` environment variable.

## Testing
Run the full test suite:
```bash
python -m pytest
```

## Support
For issues, check the logs in `platform_logs/` or run `tests/smoke_test.py` to diagnose.

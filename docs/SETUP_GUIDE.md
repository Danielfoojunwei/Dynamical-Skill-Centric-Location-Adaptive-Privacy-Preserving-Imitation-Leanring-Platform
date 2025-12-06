# Setup Guide - Dynamical Edge Platform

This guide covers the installation and setup of the Dynamical Edge system on an NVIDIA Jetson AGX Orin (32GB) or a development machine (Windows/Linux).

## Prerequisites

-   **Hardware**: NVIDIA Jetson AGX Orin (32GB) OR PC with NVIDIA GPU.
-   **OS**: Ubuntu 20.04 (JetPack 5.x) or Windows 10/11 (for dev).
-   **Python**: 3.8 or higher.
-   **Network**: WiFi 6E router (for DYGlove).

## 1. Installation

### Option A: Quick Start (AGX Orin)
Use the automated setup script:
```bash
./setup_orin.sh
```
This will:
1.  Install OS dependencies (OpenCV, GStreamer).
2.  Create a Python virtual environment.
3.  Install Python dependencies.
4.  Create a default `.env` file.

### Option B: Manual Setup (Windows/Dev)
1.  **Install Python Dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  **Install Vision Dependencies** (Optional for Sim):
    ```bash
    # Only if you have a GPU and want to run full perception
    pip install openmim
    mim install mmengine mmcv mmdet mmpose
    ```

## 2. Configuration

The system is configured via `config/config.yaml`.

### Simulation Mode
To run without hardware, enable simulation mode:
```yaml
# config/config.yaml
system:
  simulation_mode: true
```
Or use an environment variable:
```bash
export SIMULATION_MODE=true
```

### API Key
Set your API key in `.env`:
```bash
API_KEY=your_secure_key_here
```

## 3. Running the System

### Start the Pipeline
```bash
./launch_orin.sh
# OR
python src/platform/api/main.py
```

### Verify Status
Check the health endpoint:
```bash
curl http://localhost:8000/health
# Output: {"status": "ok", "version": "0.1.0"}
```

## 4. Troubleshooting

### "No module named src"
Ensure your `PYTHONPATH` is set:
```bash
export PYTHONPATH=.
```

### "403 Forbidden"
Check your `X-API-Key` header matches the `API_KEY` in `.env`.

### "TFLOPS Budget Exceeded"
Adjust `config/config.yaml` to allocate fewer resources or reduce camera frame rates.

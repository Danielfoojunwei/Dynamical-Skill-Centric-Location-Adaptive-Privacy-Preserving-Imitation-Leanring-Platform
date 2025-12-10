# Developer Notes - Dynamical Edge (AGX Orin 32GB)

## System Architecture

The Dynamical Edge system is designed for high-performance robotics and AI on the NVIDIA AGX Orin 32GB platform. It integrates perception, safety, imitation learning, and encryption into a unified pipeline.

### Core Components

1.  **Integrated Pipeline (`src/pipeline/integrated_pipeline.py`)**:
    -   Orchestrates data flow between cameras, models, and cloud.
    -   Manages TFLOPS budget dynamically.
    -   Handles safety-critical and learning loops.

2.  **Safety Manager (`src/platform/safety_manager.py`)**:
    -   Real-time hazard detection and response.
    -   Uses `HazardRegistry` (`src/core/environment_hazards.py`) for extensible hazard definitions.
    -   Enforces safety zones and stops robots if necessary.

3.  **Retargeting (`src/core/retargeting.py`)**:
    -   Object-centric human-to-robot motion retargeting.
    -   Uses Inverse Kinematics (IK) and optional Whole-Body Control (GMR).
    -   Configurable via `config/config.yaml`.

4.  **DYGlove SDK (`src/platform/edge/dyglove_sdk.py`)**:
    -   Interface for the DYGlove haptic glove.
    -   Supports WiFi 6E (UDP) and USB communication.
    -   Includes simulation mode for testing without hardware.

### Configuration

The system is configured via `config/config.yaml`. Key sections:

-   `system`: Global settings (simulation mode, logging).
-   `tflops_budget`: Allocation of compute resources.
-   `pipeline`: Queue sizes and camera routing.
-   `retargeting`: Workspace bounds and IK parameters.
-   `safety`: Safety thresholds and sensitivity.

### Simulation Mode

To run without hardware, set `SIMULATION_MODE=true` in `.env` or `config.yaml`. This enables:
-   Simulated cameras (static images or noise).
-   Simulated DYGlove (random or recorded motion).
-   Simulated robot vendor adapter.

### Testing

Run tests using `pytest`:
```bash
python -m pytest
```
Key tests:
-   `tests/smoke_test.py`: Verifies pipeline initialization.
-   `tests/test_hazards.py`: Verifies hazard registry and safety logic.
-   `tests/test_whole_body_gmr.py`: Verifies retargeting logic.

### Dependencies

-   **Python**: Managed via `requirements.txt`. Install with `scripts/install_python_deps.sh`.
-   **Vision**: MMPose, RTMW3D. Install with `scripts/install_vision_deps.sh`.
-   **OS**: System libraries. Install with `scripts/install_os_deps.sh`.

### Troubleshooting

-   **ImportError**: Ensure `PYTHONPATH` includes the project root (`.`).
-   **403 Forbidden**: Check `API_KEY` in `.env` and client headers.
-   **TFLOPS Warning**: Check `config.yaml` budget allocations.

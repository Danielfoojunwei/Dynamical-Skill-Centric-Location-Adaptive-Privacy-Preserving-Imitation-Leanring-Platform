# System Overview

## For the Layman (Simple Explanation)
Imagine you want to teach a robot how to cook. You wear a special glove (DYGlove) and move your hands. The robot watches you with cameras and feels what you feel through the glove.

1.  **The Eyes (Cameras)**: Two cameras watch your movements and turn them into 3D data.
2.  **The Hands (DYGlove)**: The glove captures your finger movements and vibrations.
3.  **The Brain (AGX Orin)**: A powerful computer takes all this data (sight + touch) and uses Artificial Intelligence to understand what you are doing.
4.  **The Robot (Daimon VTLA)**: The robot mimics your actions in real-time.

This system is "Dynamical" because it adapts. If you move faster, it moves faster. If you stop, it stops. It learns from you.

## For the Engineer (Technical Deep Dive)

### Architecture
The system follows a micro-modular architecture running on the NVIDIA Jetson AGX Orin.

1.  **Hardware Layer**:
    *   **Compute**: Jetson AGX Orin (32GB Shared Memory).
    *   **Sensors**: ONVIF IP Cameras (RTSP), DYGlove (UDP/WiFi), Robot Proprioception (Ethernet).

2.  **Driver Layer (`src/drivers`)**:
    *   `cameras.py`: Handles RTSP ingestion, decoding (NVDEC), and synchronization.
    *   `dyglove.py`: UDP server for high-frequency haptic data (1kHz).
    *   `daimon_vtla.py`: Adapter pattern implementation for Daimon Robotics SDK.

3.  **Core Pipeline (`src/pipeline`)**:
    *   **Data Ingestion**: Aggregates data from all drivers.
    *   **Preprocessing**: Resizes images, normalizes tensors.
    *   **Inference**: Runs the "Pi0 VLA" model (Vision-Language-Action) using TensorRT or PyTorch.
    *   **Post-processing**: Converts model output (logits) into robot joint commands.

4.  **Platform Services (`src/platform`)**:
    *   **API**: FastAPI backend for control and monitoring.
    *   **Security**: FHE (Fully Homomorphic Encryption) Auditor ensures data privacy.
    *   **OTA**: Over-The-Air update manager.

5.  **Frontend (`src/platform/ui`)**:
    *   React + Vite application.
    *   Real-time WebSocket connection for low-latency status updates.

### Data Flow
`[Sensors] -> [Drivers] -> [Integrated Pipeline] -> [AI Model] -> [Robot Command]`

### Security
*   **API Key**: All REST endpoints are protected by a strong API Key (Bearer Token).
*   **Audit**: Sensitive data operations are logged via the FHE Auditor.

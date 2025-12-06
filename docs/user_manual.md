# Dynamical Edge Platform - User Manual

## Overview
The **Dynamical Edge Platform** is the control center for your AGX Orin-powered robot. It manages hardware devices, safety systems, and cloud integration.

## 1. Dashboard
**Goal**: Monitor system health and control the main loop.
*   **Start System**: Click the green **Start System** button to initialize the robot pipeline (Drivers -> Perception -> Control).
*   **Stop System**: Click the red **Stop System** button to safely halt all operations.
*   **Stats**: View real-time TFLOPS usage, memory consumption, and active components.

## 2. Device Manager
**Goal**: Manage connected hardware (Cameras, Gloves, etc.).
*   **Scan Wireless**: Click to discover new WiFi-enabled devices (like the Wireless DYGlove).
*   **OTA Updates**: Click the **Refresh Icon** next to a device to check for firmware updates. If an update is available, you will be prompted to install it.
*   **Status**: Green dots indicate online devices; red dots indicate offline.

## 3. Safety System
**Goal**: Define safe operating zones and behavior.
*   **Zones**:
    1.  Click **Draw Zone**.
    2.  Click points on the map to define a polygon.
    3.  Select **Keep Out** (Robot stops) or **Slow Down** (Robot reduces speed).
*   **Configuration**:
    *   **Human Sensitivity**: Adjust how aggressively the robot detects humans (0.1 - 1.0).
    *   **Stop Distance**: Set the minimum safe distance (0.5m - 5.0m).

## 4. Cloud Integration (Value Chain)
**Goal**: Sync with Foundation Field Models (FFMs) while maintaining privacy.
*   **Check for Updates**: Downloads the latest signed model weights from the vendor (e.g., Tesla, Pi0).
*   **Upload Gradients**: Encrypts local learning progress using **MOAI (FHE)** and uploads it for federated learning. **Your raw data never leaves the device.**
*   **Traceability**: View a log of all sync and upload actions for audit purposes.

## 5. Settings
**Goal**: Configure system-wide parameters.
*   **RTSP URL**: Set the video stream source for the robot's primary camera.

---

## FAQ

**Q: How do I update the robot's software?**
A: Go to **Device Manager** and click the update icon next to the "AGX Orin" device (if listed) or use the `ota_manager` CLI.

**Q: Is my data safe when uploading to the cloud?**
A: **Yes.** We use Homomorphic Encryption (FHE). The cloud server receives encrypted "noise" that it can mathematically add to other updates, but it cannot decrypt your specific data.

**Q: What happens if a human enters a "Keep Out" zone?**
A: The Safety Manager immediately overrides the control loop and halts the robot.

**Q: Why is the "Start System" button disabled?**
A: Check if the system is already running or if the API connection is lost (refresh the page).

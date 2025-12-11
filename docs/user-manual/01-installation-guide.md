# Dynamical Edge Platform - Installation Guide

## Part 1: Installation

Welcome to the Dynamical Edge Platform! This guide will walk you through setting up your robotic control system step by step. No technical expertise required - just follow along.

---

## Table of Contents

1. [What You'll Need](#what-youll-need)
2. [Unboxing & Hardware Setup](#unboxing--hardware-setup)
3. [Network Configuration](#network-configuration)
4. [Software Installation](#software-installation)
5. [First-Time Startup](#first-time-startup)
6. [Connecting Your Devices](#connecting-your-devices)
7. [Verifying Your Installation](#verifying-your-installation)

---

## What You'll Need

### Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **NVIDIA Jetson AGX Orin 32GB** | Provided in kit | The brain of your robot system |
| **Power Supply** | 65W USB-C adapter (included) | Powers the Orin device |
| **MicroSD Card** | 64GB+ (included) | Boot and system storage |
| **NVMe SSD** | 500GB (pre-installed) | Training data and skill storage |
| **Ethernet Cable** | Cat6 or better | Network connection |
| **Monitor + HDMI Cable** | Any HDMI monitor | Initial setup only |
| **USB Keyboard/Mouse** | Standard USB | Initial setup only |

### Optional Equipment

| Component | Purpose |
|-----------|---------|
| **ONVIF IP Cameras** | Vision for the robot (up to 12 supported) |
| **DYGlove Haptic Gloves** | Hand tracking for teleoperation |
| **Humanoid Robot** | Daimon VTLA or compatible |
| **Wi-Fi 6E Router** | Wireless connectivity |

---

## Unboxing & Hardware Setup

### Step 1: Prepare Your Workspace

1. Clear a flat, stable surface near a power outlet
2. Ensure good ventilation - the Orin device needs airflow
3. Have your network router nearby for the ethernet connection

### Step 2: Connect the Hardware

```
┌─────────────────────────────────────────────────────────────┐
│                    CONNECTION DIAGRAM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [Power Outlet] ──── [65W Adapter] ──── [Orin USB-C]       │
│                                                              │
│   [Router] ──────────── [Ethernet] ──── [Orin LAN Port]     │
│                                                              │
│   [Monitor] ────────────── [HDMI] ──── [Orin HDMI Port]     │
│                                                              │
│   [Keyboard/Mouse] ────────[USB] ──── [Orin USB Ports]      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Connection Order (Important!):**

1. ✅ Connect the Ethernet cable FIRST
2. ✅ Connect the HDMI monitor
3. ✅ Connect keyboard and mouse
4. ✅ Connect power adapter LAST

### Step 3: Power On

1. Press the power button on the Orin device
2. Wait for the boot screen (about 30 seconds)
3. You should see the Ubuntu desktop appear

> **💡 Tip:** If nothing appears on screen, check that the HDMI cable is firmly connected and the monitor is set to the correct input.

---

## Network Configuration

Your Orin device needs to connect to your local network to communicate with cameras, robots, and the cloud.

### Wired Connection (Recommended)

If you connected an Ethernet cable, you're likely already connected!

**To verify:**
1. Click the network icon in the top-right corner
2. Look for "Wired Connected" with a green checkmark

### Wireless Connection (Optional)

If you need Wi-Fi instead:

1. Click the network icon in the top-right corner
2. Select your Wi-Fi network from the list
3. Enter your Wi-Fi password
4. Wait for "Connected" to appear

### Finding Your Device's IP Address

You'll need this address to access the control panel from other computers:

1. Open Terminal (press `Ctrl+Alt+T`)
2. Type: `hostname -I`
3. Write down the number that appears (e.g., `192.168.1.100`)

> **📝 Note:** This is your device's IP address. You'll use it to access the web interface.

---

## Software Installation

The Dynamical Edge Platform comes pre-installed on your device. However, if you need to install or update it:

### Option A: Fresh Installation

Open Terminal and run these commands:

```bash
# Step 1: Download the platform
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform

# Step 2: Install dependencies (takes about 10-15 minutes)
./install.sh

# Step 3: Initialize the database
python -m src.platform.api.database init
```

### Option B: Update Existing Installation

```bash
# Navigate to platform folder
cd ~/edge-platform

# Pull latest updates
git pull origin main

# Update dependencies
./update.sh
```

### Installing Camera Drivers (If Using ONVIF Cameras)

```bash
pip install onvif-zeep
```

### Installing Glove Support (If Using DYGlove)

```bash
pip install pyserial
```

---

## First-Time Startup

### Starting the Platform

1. Open Terminal (`Ctrl+Alt+T`)
2. Navigate to the platform folder:
   ```bash
   cd ~/edge-platform
   ```
3. Start the platform:
   ```bash
   python -m src.platform.api.main
   ```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Network manager started
INFO:     Dynamical Edge Platform v0.3.2 ready
```

### Accessing the Control Panel

Open a web browser (on any computer on your network) and go to:

```
http://[YOUR-IP-ADDRESS]:8000
```

For example: `http://192.168.1.100:8000`

You should see the Dynamical Edge Platform dashboard!

---

## Connecting Your Devices

### Adding IP Cameras

1. Click **"Devices"** in the left sidebar
2. Click **"Scan Network"** button
3. Your ONVIF cameras should appear automatically

**If cameras don't appear automatically:**

1. Click **"Add Camera Manually"**
2. Enter the camera details:
   - **Camera ID:** A name you choose (e.g., "front-camera")
   - **Host:** Camera's IP address (e.g., 192.168.1.50)
   - **Port:** Usually 80 or 8080
   - **Username:** Camera's login username
   - **Password:** Camera's login password
3. Click **"Connect"**

### Connecting DYGlove

1. Power on your DYGlove
2. Connect it via USB or ensure it's on the same Wi-Fi network
3. Go to **"Devices"** in the sidebar
4. Click **"Scan Network"**
5. Your glove should appear as "DYGLOVE"

### Connecting Your Robot

1. Ensure your robot is powered on and networked
2. Go to **"Devices"** in the sidebar
3. Click **"Scan Network"**
4. Select your robot from the list
5. Click **"Configure"** to set up communication

---

## Verifying Your Installation

Let's make sure everything is working correctly.

### System Health Checklist

Open the Dashboard and verify these items:

| Check | What to Look For | Status |
|-------|------------------|--------|
| **System Status** | Shows "IDLE" or "OPERATIONAL" | ✅ / ❌ |
| **TFLOPS Display** | Shows "0 / 137 TFLOPS" | ✅ / ❌ |
| **Memory** | Shows memory usage in GB | ✅ / ❌ |
| **Components** | Lists active components | ✅ / ❌ |

### Device Connection Test

Go to **Devices** and verify:

| Check | What to Look For | Status |
|-------|------------------|--------|
| **Device Count** | Shows total connected devices | ✅ / ❌ |
| **Camera Status** | Shows "Online" with green dot | ✅ / ❌ |
| **Glove Status** | Shows "Online" with green dot | ✅ / ❌ |

### API Health Test

Open a new browser tab and go to:
```
http://[YOUR-IP-ADDRESS]:8000/health
```

You should see:
```json
{"status": "healthy", "version": "0.3.2"}
```

---

## Troubleshooting Common Installation Issues

### "Cannot connect to platform"

**Problem:** Web browser shows "Connection refused" or "Cannot reach page"

**Solutions:**
1. Verify the platform is running (check Terminal for errors)
2. Check you're using the correct IP address
3. Make sure you're on the same network as the Orin device
4. Try: `http://localhost:8000` if browsing from the Orin itself

### "Cameras not detected"

**Problem:** Network scan doesn't find cameras

**Solutions:**
1. Verify cameras are powered on and networked
2. Check cameras are on the same network subnet
3. Try adding camera manually with its IP address
4. Ensure ONVIF is enabled on your camera (check camera settings)

### "Database error on startup"

**Problem:** Error message about database when starting

**Solution:**
```bash
# Reset and reinitialize the database
rm platform.db
python -m src.platform.api.database init
```

### "Permission denied" errors

**Problem:** Commands fail with permission errors

**Solution:**
```bash
# Add yourself to the docker group (if using Docker)
sudo usermod -aG docker $USER

# Log out and log back in, then try again
```

---

## What's Next?

Congratulations! Your Dynamical Edge Platform is installed and ready to use.

**Continue to:**
- **[Part 2: Operation Guide](./02-operation-guide.md)** - Learn how to use all the features
- **[Part 3: Post-Deployment Management](./03-post-deployment-guide.md)** - Maintenance and monitoring

---

## Quick Reference Card

| Task | Command / Action |
|------|------------------|
| Start Platform | `python -m src.platform.api.main` |
| Access Dashboard | `http://[IP]:8000` |
| Stop Platform | Press `Ctrl+C` in Terminal |
| Check Status | Visit `http://[IP]:8000/health` |
| View Logs | Check Terminal output |
| Restart Platform | Stop and Start again |

---

*Dynamical Edge Platform v0.3.2 - Installation Guide*
*For support, visit: https://dynamical.ai/support*

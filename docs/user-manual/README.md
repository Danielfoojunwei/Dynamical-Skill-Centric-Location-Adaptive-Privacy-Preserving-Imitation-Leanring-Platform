# Dynamical Edge Platform - User Manual

## Welcome

Welcome to the Dynamical Edge Platform documentation. This manual will guide you through every aspect of using your robotic control system, from initial setup to day-to-day operations and long-term maintenance.

---

## Who Is This Manual For?

This manual is written for **operators and supervisors** who work with the Dynamical Edge Platform. No programming experience is required - we explain everything in plain language with step-by-step instructions.

---

## Documentation Structure

This manual is divided into three parts:

### Part 1: Installation Guide
**[→ Read the Installation Guide](./01-installation-guide.md)**

Everything you need to set up your system for the first time:
- Hardware requirements and unboxing
- Network configuration
- Software installation
- Connecting cameras, gloves, and robots
- Verifying your installation

**Time Required:** 30-60 minutes for first-time setup

---

### Part 2: Operation Guide
**[→ Read the Operation Guide](./02-operation-guide.md)**

How to use every feature of the platform:
- Understanding the dashboard
- Managing devices (cameras, gloves, robots)
- Using and managing robot skills
- Training new capabilities
- Monitoring and debugging with observability tools
- Configuring safety zones
- Cloud connectivity

**Time Required:** 1-2 hours to read through; reference as needed

---

### Part 3: Post-Deployment Management
**[→ Read the Post-Deployment Guide](./03-post-deployment-guide.md)**

Keeping your system healthy over time:
- Daily, weekly, and monthly maintenance tasks
- Performance monitoring
- Software updates
- Backup and recovery procedures
- Troubleshooting common problems
- Security best practices
- Getting support

**Time Required:** 30 minutes to read; reference as needed

---

## Quick Start

If you're in a hurry, here's the minimum path to get started:

1. **Set up hardware** - Connect Orin, cameras, network
2. **Start the platform** - `python -m src.platform.api.main`
3. **Open the dashboard** - `http://[your-ip]:8000`
4. **Scan for devices** - Devices → Scan Network
5. **Configure safety zones** - Safety → Draw zones
6. **Start the system** - Dashboard → START SYSTEM

For full details on each step, see the Installation Guide.

---

## Platform Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMICAL EDGE PLATFORM v0.3.2                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │  Dashboard  │   │   Devices   │   │   Skills    │   │  Training   │     │
│  │  System     │   │  Cameras    │   │  MoE Router │   │  Datasets   │     │
│  │  Status     │   │  Gloves     │   │  Library    │   │  Jobs       │     │
│  │  Metrics    │   │  Robots     │   │  Upload     │   │  Versions   │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │Observability│   │   Safety    │   │    Cloud    │   │  Settings   │     │
│  │  Logs       │   │  Zones      │   │  Sync       │   │  Config     │     │
│  │  Debugging  │   │  Hazards    │   │  Status     │   │  System     │     │
│  │  RCA        │   │  Config     │   │  Audit      │   │  Params     │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements Summary

| Component | Specification |
|-----------|--------------|
| **Compute** | NVIDIA Jetson AGX Orin 32GB |
| **Storage** | 500GB NVMe SSD |
| **Network** | Ethernet or Wi-Fi 6E |
| **Cameras** | ONVIF-compatible IP cameras (up to 12) |
| **Gloves** | DYGlove haptic gloves (optional) |
| **Robot** | Daimon VTLA or compatible humanoid |

---

## Key Concepts

Before diving in, understand these key concepts:

### Skills
Pre-trained robot behaviors (like "grasp object" or "pour liquid") that can be combined to accomplish tasks. You don't program them - the system learns them.

### MoE (Mixture of Experts)
The system that automatically picks the right combination of skills for any task you describe. Just tell it what you want, and it figures out how.

### Safety Zones
Areas you define on a map where the robot should stop (KEEP_OUT) or slow down (SLOW_DOWN). Critical for human safety.

### Federated Learning
How robots learn from each other while keeping your data private. Your robot benefits from others' experiences without sharing your raw data.

### FHE (Fully Homomorphic Encryption)
The technology that keeps your data private even when learning from the cloud. 128-bit security means your data stays yours.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.3.2 | Dec 2024 | Current release |
| 0.3.1 | Nov 2024 | Added glove calibration UI |
| 0.3.0 | Nov 2024 | MoE skill architecture |
| 0.2.0 | Oct 2024 | Safety zones and FL |
| 0.1.0 | Sep 2024 | Initial release |

---

## Need Help?

- **Documentation Issues:** If you find errors in this manual, please report them
- **Technical Support:** support@dynamical.ai
- **Emergency (Safety):** emergency@dynamical.ai

---

*Dynamical Edge Platform v0.3.2*
*© 2024 Dynamical.ai - All rights reserved*

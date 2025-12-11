# Dynamical Edge Platform v0.3.3

![Version](https://img.shields.io/badge/version-0.3.3-blue)
![Status](https://img.shields.io/badge/status-Production-green)
![License](https://img.shields.io/badge/license-Proprietary-red)

> **Skill-Centric Federated Learning for Vision-Language-Action Models on Edge Robotics**

The Dynamical Edge Platform is a production-ready software stack for humanoid robots running on NVIDIA Jetson AGX Orin 32GB. It features a novel **Mixture-of-Experts (MoE) skill architecture** with **privacy-preserving federated learning** using homomorphic encryption.

---

## Key Features

### MoE Skill Architecture
- **Frozen Base VLA Models**: Pi0/OpenVLA-7B remain read-only (IP-safe)
- **Trainable Skill Experts**: Lightweight skills that augment base models
- **Dynamic Task Routing**: Natural language → automatic skill selection
- **Skill Blending**: Combine multiple skills for complex tasks

### Meta AI Foundation Models (NEW in v0.3.3)
- **DINOv2/DINOv3**: Self-supervised visual features (8 TFLOPS)
  - Patch-based feature extraction at multiple scales
  - Cross-frame temporal consistency
- **SAM 3 (Segment Anything)**: Zero-shot segmentation (15 TFLOPS)
  - Automatic mask generation for objects/humans
  - Privacy-aware face/hand region masking
- **V-JEPA 2**: Video prediction & world modeling (10 TFLOPS)
  - Latent video prediction (replaces trajectory_prediction)
  - Action-conditioned future state estimation
- **Unified Perception Pipeline**: Multi-model fusion with privacy wrapper

### Privacy-Preserving Learning
- **N2HE Encryption**: 128-bit homomorphic encryption for gradients
- **Federated Aggregation**: Learn from fleet without sharing raw data
- **Encrypted Skill Storage**: Skills encrypted at rest and in transit

### Real-Time Control
- **4-Tier Timing System**:
  - Tier 1: Safety Loop @ 1kHz (never throttled)
  - Tier 2: Control Loop @ 100Hz
  - Tier 3: Learning Loop @ 10Hz
  - Tier 4: Cloud Sync @ 0.1Hz
- **137 TFLOPS Budget Management**: 127 TFLOPS allocated (92.7% utilization)
  - Meta AI Models: 33 TFLOPS (DINOv3 8 + SAM3 15 + V-JEPA 2 10)
  - Safety & Perception: 65 TFLOPS
  - VLA & Skills: 29 TFLOPS

### Comprehensive UI
- **Dashboard**: System status, TFLOPS usage, component monitoring
- **Device Manager**: ONVIF PTZ cameras, DYGlove calibration, robot control
- **Skills Manager**: MoE task routing, skill upload/download
- **Training Manager**: Datasets, jobs, version control, FL status
- **Observability**: Flight recorder, VLA status, FHE audit, RCA
- **Safety**: Interactive zone drawing, hazard configuration

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMICAL EDGE PLATFORM v0.3.3                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    PERCEPTION LAYER (Meta AI Enhanced)                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ DINOv3   │  │  SAM 3   │  │ V-JEPA 2 │  │  Multi-  │             │   │
│  │  │ Features │  │ Segment  │  │  World   │  │  Camera  │             │   │
│  │  │ (8 TF)   │  │ (15 TF)  │  │ (10 TF)  │  │  Fusion  │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │   │
│  │        │              │             │             │                  │   │
│  │        └──────────────┴─────────────┴─────────────┘                  │   │
│  │                    Unified Perception Pipeline                        │   │
│  │                    (Privacy Wrapper + N2HE)                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         VLA + MoE SKILLS                              │   │
│  │  ┌────────────────────┐    ┌─────────────────────────────────────┐   │   │
│  │  │  Base VLA (Frozen) │    │        MoE Skill Router             │   │   │
│  │  │  ┌──────┐ ┌──────┐ │    │  ┌───────┐ ┌───────┐ ┌───────┐     │   │   │
│  │  │  │ Pi0  │ │OpenVLA│ │───►│  │Grasp  │ │ Pour  │ │Navigate│    │   │   │
│  │  │  │ 7B   │ │  7B   │ │    │  │ Skill │ │ Skill │ │ Skill │    │   │   │
│  │  │  └──────┘ └──────┘ │    │  └───────┘ └───────┘ └───────┘     │   │   │
│  │  └────────────────────┘    └─────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CONTROL LAYER                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │  Robot   │  │  Glove   │  │  Safety  │  │Retargeting│            │   │
│  │  │ Invoker  │  │  Driver  │  │  Manager │  │   GMR    │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CLOUD LAYER (Encrypted)                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │  N2HE    │  │  FedAvg  │  │  Skill   │  │  MOAI    │             │   │
│  │  │ Encrypt  │  │ Aggregator│ │  Library │  │ Compress │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Compute** | NVIDIA Jetson AGX Orin 32GB | Edge inference (137 TFLOPS FP16) |
| **Storage** | 500GB NVMe SSD | Training data, skill cache |
| **Memory** | 32GB Unified | VLA models, perception |
| **Network** | Ethernet / Wi-Fi 6E | Cloud sync, device comms |
| **Cameras** | ONVIF IP cameras (up to 12) | Multi-view perception |
| **Gloves** | DYGlove 21-DOF haptic | Teleoperation input |
| **Robot** | Daimon VTLA or compatible | Humanoid control |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform

# Install dependencies
./install.sh

# Initialize database
python -m src.platform.api.database init
```

### 2. Start the Platform

```bash
python -m src.platform.api.main
```

### 3. Access the Dashboard

Open in browser: `http://localhost:8000`

### 4. Connect Devices

1. Go to **Devices** → Click **Scan Network**
2. Configure cameras with PTZ controls
3. Calibrate gloves (4-step process)

### 5. Start Operations

1. Go to **Safety** → Draw safety zones
2. Go to **Dashboard** → Click **START SYSTEM**
3. Use **Skills** → Enter task description → Click **Route**

---

## Documentation

### User Manual (For Operators)

| Guide | Description |
|-------|-------------|
| [**Installation Guide**](docs/user-manual/01-installation-guide.md) | Hardware setup, network, first boot |
| [**Operation Guide**](docs/user-manual/02-operation-guide.md) | All features, step-by-step |
| [**Post-Deployment Guide**](docs/user-manual/03-post-deployment-guide.md) | Maintenance, updates, troubleshooting |

### Technical Documentation

| Document | Description |
|----------|-------------|
| [Setup Guide](docs/SETUP_GUIDE.md) | Technical installation |
| [Developer Notes](docs/DEVELOPER_NOTES.md) | Architecture, APIs |
| [Research Paper](docs/research/dynamical_moe_skills_paper.md) | Skill-Centric Federated Learning |

---

## Project Structure

```
├── src/
│   ├── core/                    # Core algorithms
│   │   ├── robot_skill_invoker.py   # Skill invocation pipeline
│   │   ├── gmr_retargeting.py       # Motion retargeting
│   │   └── wholebody_pose_pipeline.py
│   │
│   ├── drivers/                 # Hardware interfaces
│   │   ├── cameras.py               # ONVIF camera driver
│   │   ├── onvif_ptz.py             # PTZ camera control
│   │   ├── dyglove.py               # Haptic glove driver
│   │   ├── glove_calibration.py     # 21-DOF calibration
│   │   └── daimon_vtla.py           # Robot driver
│   │
│   ├── models/                  # Meta AI Foundation Models (NEW)
│   │   ├── meta_ai/
│   │   │   ├── dinov3.py            # DINOv3 visual features (8 TF)
│   │   │   ├── sam3.py              # SAM 3 segmentation (15 TF)
│   │   │   ├── vjepa2.py            # V-JEPA 2 world model (10 TF)
│   │   │   ├── privacy_wrapper.py   # N2HE encryption wrapper
│   │   │   └── unified_perception.py # Multi-model fusion pipeline
│   │
│   ├── platform/
│   │   ├── api/                     # REST API (FastAPI)
│   │   │   └── main.py              # All endpoints
│   │   ├── cloud/                   # Cloud integration
│   │   │   ├── moe_skill_router.py  # MoE routing
│   │   │   └── secure_aggregator.py # FL aggregation
│   │   ├── edge/
│   │   │   └── skill_client.py      # Edge skill execution
│   │   ├── ui/                      # React dashboard
│   │   │   └── src/
│   │   │       ├── App.jsx
│   │   │       ├── Dashboard.jsx
│   │   │       ├── DeviceManager.jsx
│   │   │       ├── SkillsManager.jsx
│   │   │       ├── TrainingManager.jsx
│   │   │       ├── Observability.jsx
│   │   │       └── Safety.jsx
│   │   └── safety_manager.py
│   │
│   ├── moai/                    # Compression & encryption
│   │   ├── n2he.py                  # Homomorphic encryption
│   │   └── moai_pt.py               # Neural compression
│   │
│   └── spatial_intelligence/    # VLA models
│       └── pi0/
│
├── docs/
│   ├── user-manual/             # Operator documentation
│   └── research/                # Research papers
│
├── config/
│   └── config.yaml              # System configuration
│
└── tests/                       # Test suite
```

---

## API Reference

### Skills API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/skills` | GET | List all skills |
| `/api/v1/skills/request` | POST | Route task to skills |
| `/api/v1/skills/upload` | POST | Upload new skill |
| `/api/v1/robot/invoke_skill` | POST | Execute skills on robot |

### Device API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/devices` | GET | List connected devices |
| `/devices/scan` | POST | Scan network for devices |
| `/api/devices/ptz/{id}/move` | POST | Control PTZ camera |
| `/api/devices/glove/{id}/calibration/start` | POST | Start glove calibration |

### Safety API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/safety/zones` | GET/POST | Manage safety zones |
| `/api/safety/config` | GET/POST | Safety configuration |
| `/api/safety/hazards` | GET | Active hazard detection |

### Observability API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/observability/blackbox` | GET | Flight recorder data |
| `/api/observability/fhe/audit` | GET | Encryption audit log |
| `/api/observability/incident/{id}/analyze` | GET | Root cause analysis |

---

## Configuration

Edit `config/config.yaml`:

```yaml
# System Settings
system:
  device_id: "orin_001"
  simulation_mode: false

# TFLOPS Budget
compute:
  total_tflops: 137
  safe_utilization: 0.85

# Safety
safety:
  human_sensitivity: 0.8
  stop_distance_m: 1.5

# Cloud
cloud:
  api_url: "https://api.dynamical.ai"
  sync_interval_s: 600
  encryption: "n2he_128"

# Cameras
cameras:
  - id: "front"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
  - id: "side"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
```

---

## Safety

> [!IMPORTANT]
> **SAFETY DISCLAIMER**: This software provides safety-adjacent features but is **NOT a certified safety controller**. Always use in conjunction with:
> - Hardware emergency stops (E-stops)
> - Physical barriers and guards
> - Trained personnel supervision
> - Standard industrial safety practices

### Safety Features
- **KEEP_OUT Zones**: Robot stops immediately if it enters
- **SLOW_DOWN Zones**: Robot reduces speed in area
- **Human Detection**: Camera-based person detection
- **Configurable Stop Distance**: 0.5m to 5.0m range

---

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest tests/test_skills.py
python -m pytest tests/test_safety.py

# Run with coverage
python -m pytest --cov=src
```

---

## Research

This platform implements novel research in robotics AI:

**Skill-Centric Federated Learning for Vision-Language-Action Models**

- Frozen base VLA models preserve vendor IP
- Trainable skill experts enable customization
- N2HE encryption ensures privacy during FL
- MoE routing enables dynamic skill composition

See: [Research Paper](docs/research/dynamical_moe_skills_paper.md)

---

## Open Source Dependencies

| Library | License | Purpose |
|---------|---------|---------|
| PyTorch | BSD | Deep learning |
| FastAPI | MIT | REST API |
| React | MIT | Dashboard UI |
| OpenCV | Apache 2.0 | Computer vision |
| MMPose | Apache 2.0 | Pose estimation |
| Pinocchio | BSD | Robot kinematics |
| DINOv2/v3 | Apache 2.0 | Self-supervised visual features (Meta AI) |
| SAM 2/3 | Apache 2.0 | Segment Anything model (Meta AI) |
| V-JEPA | CC-BY-NC | Video prediction (Meta AI, non-commercial) |

---

## Support

- **Documentation**: [docs.dynamical.ai](https://docs.dynamical.ai)
- **Email**: support@dynamical.ai
- **Emergency**: emergency@dynamical.ai (safety issues)

---

## License

Proprietary - Dynamical.ai © 2024

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.3.3 | Dec 2024 | **Meta AI Integration**: DINOv3, SAM3, V-JEPA 2 models; unified perception pipeline; mock code cleanup |
| 0.3.2 | Dec 2024 | User documentation, ONVIF PTZ, glove calibration |
| 0.3.1 | Dec 2024 | Comprehensive UI (Skills, Training, Observability) |
| 0.3.0 | Dec 2024 | MoE skill architecture, N2HE encryption |
| 0.2.0 | Nov 2024 | Safety zones, federated learning |
| 0.1.0 | Oct 2024 | Initial release |

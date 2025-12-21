# Dynamical Edge Platform v0.5.0

![Version](https://img.shields.io/badge/version-0.5.0-blue)
![Status](https://img.shields.io/badge/status-Production-green)
![License](https://img.shields.io/badge/license-Proprietary-red)

> **On-Device Runtime and Training Engine for Humanoid Robots with Meta AI Foundation Models**

The Dynamical Edge Platform is an on-device runtime and training engine for humanoid robots. It runs on **NVIDIA Jetson Thor** (Blackwell architecture) with:

- **2070 FP4 TFLOPS** - 7.5x more compute than previous generation
- **128 GB LPDDR5X** - All models loaded simultaneously
- **10Hz Control Loop** - 5x faster than previous generation
- **Giant Model Support** - DINOv3 ViT-G, SAM3 Large, V-JEPA 2 Giant

### Key Features
- **Meta AI Foundation Models**: DINOv3, SAM3, V-JEPA 2 for state-of-the-art perception
- **Mixture-of-Experts (MoE) Skill Architecture** with privacy-preserving federated learning
- **NVIDIA Isaac Lab Integration** for sim-to-real transfer
- **128-bit FHE Encryption** for secure gradient aggregation

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DYNAMICAL EDGE PLATFORM                                │
│                    (On-Device Runtime & Training Engine)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    SKILL INVOCATION API (Unified)                          │ │
│  │                                                                            │ │
│  │    POST /api/v1/robot/invoke_skill                                         │ │
│  │    POST /api/v1/robot/execute_skill                                        │ │
│  │                                                                            │ │
│  │    Inputs: robot_id, skill_id, role, goal, coordination_id                 │ │
│  │    Returns: actions[], skill_ids_used, blend_weights, safety_status        │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   PERCEPTION    │  │  VLA + MoE      │  │    CONTROL      │                  │
│  │   ───────────   │  │  ─────────      │  │    ───────      │                  │
│  │  • RTMPose      │  │  • Pi0/OpenVLA  │  │  • Safety Mgr   │                  │
│  │  • Depth        │  │  • Skill Router │  │  • Retargeting  │                  │
│  │  • Multi-Cam    │  │  • Skill Blend  │  │  • Robot Cmds   │                  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│                                      │                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    SHARED CRYPTO LIBRARY                                   │ │
│  │                                                                            │ │
│  │    src/shared/crypto/fhe_backend.py                                        │ │
│  │    • TenSEALBackend (CKKS, 128-bit) - Production                          │ │
│  │    • N2HEBackend (LWE) - Research fallback                                │ │
│  │    • MockFHEBackend - Development                                         │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    FEDERATED LEARNING                                      │ │
│  │                                                                            │ │
│  │    • Gradient encryption (homomorphic)                                     │ │
│  │    • Flower-based FL server                                               │ │
│  │    • Differential privacy (epsilon tracking)                              │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL ORCHESTRATION                                    │
│                     (SwarmBridge / SwarmBrain)                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│  • Multi-robot coordination                                                      │
│  • Cross-site orchestration                                                      │
│  • Central FL service                                                            │
│  • Skill registry                                                                │
│                                                                                  │
│  Communicates via Dynamical's standardized APIs with:                            │
│    - robot_id, role (leader/follower/observer/independent/assistant)            │
│    - coordination_id for multi-robot tasks                                       │
│    - CoordinationMetadata in skill artifacts                                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Meta AI Foundation Models

State-of-the-art perception powered by Meta's latest foundation models:

| Model | Purpose | Key Features |
|-------|---------|--------------|
| **DINOv2/v3** | Self-supervised vision | Dense feature extraction, zero-shot classification |
| **SAM 2/3** | Segment Anything | Real-time object segmentation, video tracking |
| **V-JEPA 2** | Video understanding | Temporal reasoning, action prediction |

All models include privacy-preserving wrappers for federated learning:
- Differential privacy for feature extraction
- Secure aggregation for model updates
- Audit logging for compliance

### NVIDIA Isaac Lab Simulation

Sim-to-real transfer with high-fidelity physics simulation:
- Real-time robot visualization
- Domain randomization for robust policies
- Parallel environment training
- Direct deployment to Jetson hardware

### Unified Skill Invocation API

Single API for all skill execution, supporting both autonomous operation and coordinated multi-robot scenarios:

```python
POST /api/v1/robot/invoke_skill
{
    "robot_id": "thor_001",
    "role": "leader",           # leader, follower, observer, independent, assistant
    "goal": "pick up the cup",
    "coordination_id": "task_123",
    "skill_ids": ["grasp_v2", "pour_v1"],
    "mode": "blended",
    "max_skills": 3
}
```

### MoE Skill Architecture
- **Frozen Base VLA Models**: Pi0/OpenVLA-7B remain read-only (IP-safe)
- **Trainable Skill Experts**: Lightweight skills that augment base models
- **Dynamic Task Routing**: Natural language → automatic skill selection
- **Skill Blending**: Combine multiple skills with MoE routing weights

### Shared Crypto Library
Unified FHE interface used by both edge and cloud components:

```python
from src.shared.crypto import create_fhe_backend

# Auto-selects best available backend
backend = create_fhe_backend()

# Encrypt gradients for federated learning
encrypted = backend.encrypt(gradients)

# Homomorphic aggregation (server-side)
aggregated = backend.homomorphic_sum([enc1, enc2, enc3])
```

### Coordination Metadata Support
Skills include coordination metadata for SwarmBridge integration:

```python
@dataclass
class CoordinationMetadata:
    supported_roles: List[str]     # ["leader", "follower"]
    requires_leader: bool          # True for coordinated skills
    sync_mode: str                 # "none", "loose", "strict", "realtime"
    min_robots: int
    max_robots: int
```

### 4-Tier Timing Architecture (Jetson Thor)
- **Tier 1**: Safety Loop @ 1kHz (hardware watchdog, never throttled)
- **Tier 2**: Control Loop @ 10Hz (100ms - robot motion, skill execution)
- **Tier 3**: Perception Loop @ 10Hz (100ms - DINOv3, SAM3, V-JEPA 2)
- **Tier 4**: Learning Loop (offline - FHE encryption, FL aggregation)

### Privacy-Preserving Learning
- **TenSEAL/N2HE**: 128-bit homomorphic encryption for gradients
- **Flower Framework**: Production federated learning
- **Encrypted Skill Storage**: Skills encrypted at rest and in transit

---

## Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Compute** | NVIDIA Jetson Thor (Blackwell) | Edge inference (2070 FP4 TFLOPS) |
| **Storage** | 500GB NVMe SSD | Training data, skill cache |
| **Memory** | 128GB LPDDR5X Unified | Giant VLA models, full perception |
| **Network** | Ethernet / Wi-Fi 6E | Cloud sync, device comms |
| **Cameras** | ONVIF IP cameras (up to 12) | Multi-view perception |
| **Gloves** | DYGlove 21-DOF haptic | Teleoperation input |
| **Robot** | Daimon VTLA or compatible | Humanoid control |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/dynamical-ai/edge-platform.git
cd edge-platform
./install.sh
python -m src.platform.api.database init
```

### 2. Start the Platform

```bash
python -m src.platform.api.main
```

### 3. Access Dashboard

Open: `http://localhost:8000`

### 4. Invoke a Skill

```bash
curl -X POST http://localhost:8000/api/v1/robot/invoke_skill \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "robot_id": "thor_001",
    "task_description": "pick up the red cup",
    "mode": "autonomous"
  }'
```

---

## Project Structure

```
├── src/
│   ├── core/                        # Core algorithms
│   │   ├── timing_architecture.py       # 4-tier timing (Thor optimized)
│   │   ├── system_robustness.py         # Reliability & safety
│   │   ├── meta_ai_models.py            # DINOv3, SAM3, V-JEPA 2
│   │   ├── gmr_retargeting.py           # Motion retargeting
│   │   └── robot_skill_invoker.py       # Unified skill invocation
│   │
│   ├── platform/
│   │   ├── jetson_thor.py               # Thor hardware config (NEW)
│   │   ├── api/                         # REST API (FastAPI)
│   │   ├── cloud/                       # FL, MoE routing
│   │   ├── edge/                        # Edge components
│   │   └── ui/                          # React dashboard
│   │
│   ├── drivers/                     # Hardware interfaces
│   │   ├── dyglove.py                   # WiFi 6E haptic gloves
│   │   ├── onvif_ptz.py                 # PTZ camera control
│   │   └── daimon_vtla.py               # Robot driver
│   │
│   └── shared/crypto/              # FHE (128-bit security)
│
├── config/config.yaml              # System configuration
└── tests/                          # Test suite
```

---

## API Reference

### Skill Invocation API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/robot/invoke_skill` | POST | Execute skills with role coordination |
| `/api/v1/robot/execute_skill` | POST | Alias for invoke_skill |
| `/api/v1/robot/invoker/stats` | GET | Invoker statistics |

**Request Body (invoke_skill)**:
```json
{
    "robot_id": "thor_001",
    "role": "independent",
    "goal": "pick up the cup",
    "task_description": "pick up the red cup",
    "mode": "autonomous",
    "max_skills": 3
}
```

**Response**:
```json
{
    "success": true,
    "robot_id": "thor_001",
    "actions": [...],
    "skill_ids_used": ["grasp_v2"],
    "safety_status": "OK",
    "total_time_ms": 8.5
}
```

### Skills API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/skills` | GET | List all skills |
| `/api/v1/skills` | POST | Register new skill |
| `/api/v1/skills/{id}` | GET | Get skill details |
| `/api/v1/skills/request` | POST | Route task to skills (MoE) |
| `/api/v1/skills/deploy` | POST | Deploy skills to device |

### Federated Learning API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/fl/register` | POST | Register FL client |
| `/api/v1/fl/update` | POST | Submit encrypted gradients |
| `/api/v1/fl/model` | GET | Download global model |
| `/api/v1/fl/status` | GET | FL server status |
| `/api/v1/fl/privacy` | GET | Differential privacy budget |

### Device API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/edge-devices` | GET/POST | Manage edge devices |
| `/api/v1/peripherals` | GET/POST | Manage peripherals |
| `/api/devices/ptz/{id}/move` | POST | Control PTZ camera |
| `/api/devices/glove/{id}/calibration/start` | POST | Start calibration |

### Safety API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/safety/zones` | GET/POST | Manage safety zones |
| `/api/safety/hazards` | GET | Active hazard detection |
| `/health` | GET | System health check |

---

## Role Coordination

Dynamical supports role-based skill execution for SwarmBridge integration:

| Role | Description |
|------|-------------|
| `independent` | Default - acts autonomously |
| `leader` | Primary actor, initiates coordination |
| `follower` | Follows leader commands/trajectory |
| `observer` | Monitors but doesn't act |
| `assistant` | Supports leader with secondary tasks |

The role is passed via the unified skill invocation API and affects skill execution behavior.

---

## Shared Crypto Library

The shared crypto library (`src/shared/crypto/`) provides a unified FHE interface:

```python
from src.shared.crypto import create_fhe_backend, FHEConfig, get_available_backends

# Check available backends
print(get_available_backends())  # ['tenseal', 'n2he', 'mock']

# Create with custom config
config = FHEConfig(
    backend='auto',
    poly_modulus_degree=8192,
    security_bits=128
)
backend = create_fhe_backend(config=config)

# Encrypt/decrypt
encrypted = backend.encrypt(gradients)
decrypted = backend.decrypt(encrypted)

# Homomorphic operations
summed = backend.homomorphic_sum([enc1, enc2, enc3])
```

---

## UI Dashboard

The React dashboard manages edge-specific functionality:

| Component | Description |
|-----------|-------------|
| **Dashboard** | System status, TFLOPS usage, component health |
| **Device Manager** | ONVIF cameras, DYGlove calibration, robot control |
| **Skills Manager** | MoE routing, skill upload/download |
| **Training Manager** | Datasets, training jobs, FL status |
| **Observability** | Flight recorder, VLA status, FHE audit |
| **Safety** | Interactive zone drawing, hazard configuration |

> **Note**: Cross-site orchestration and multi-robot coordination UI are handled by SwarmBridge/SwarmBrain, not Dynamical.

---

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_config_loader.py
python -m pytest tests/test_whole_body_gmr.py

# Run with coverage
python -m pytest --cov=src
```

---

## What Dynamical Does NOT Include

The following are handled by external services (SwarmBridge/SwarmBrain):

- Multi-robot coordination logic
- Cross-site orchestration
- OpenFL integration (uses Flower instead)
- CSA registry (uses encrypted storage directly)
- Central FL service (exposes API for external FL service)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **0.5.0** | Dec 2024 | **Jetson Thor upgrade** - 7.5x compute (2070 TFLOPS), 128GB memory, 10Hz control, giant model variants |
| 0.4.0 | Dec 2024 | Meta AI Foundation Models (DINOv3, SAM3, V-JEPA 2), Isaac Lab simulation |
| 0.3.0 | Dec 2024 | MoE skill architecture, N2HE encryption |
| 0.2.0 | Nov 2024 | Safety zones, federated learning |
| 0.1.0 | Oct 2024 | Initial release |

---

## License

Proprietary - Dynamical.ai © 2024

---

## Support

- **Documentation**: [docs.dynamical.ai](https://docs.dynamical.ai)
- **Email**: support@dynamical.ai
- **Emergency**: emergency@dynamical.ai (safety issues)

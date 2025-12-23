# Dynamical Edge Platform v0.6.0

![Version](https://img.shields.io/badge/version-0.6.0-blue)
![Status](https://img.shields.io/badge/status-Production-green)
![License](https://img.shields.io/badge/license-Proprietary-red)
![React](https://img.shields.io/badge/React-18.3-61DAFB)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB)

> **On-Device Runtime and Training Engine for Humanoid Robots with Meta AI Foundation Models**

The Dynamical Edge Platform is an on-device runtime and training engine for humanoid robots. It runs on **NVIDIA Jetson Thor** (Blackwell architecture) with:

- **2070 FP4 TFLOPS** - Full utilization via INT4/FP4 quantization
- **128 GB LPDDR5X** - All models loaded simultaneously
- **200Hz Control Loop** - 2x faster with FP4 quantization
- **Giant Model Support** - DINOv3 ViT-G, SAM3 Huge, V-JEPA 2 Giant
- **Redundant Safety** - Multi-model ensemble with human intent prediction

---

## What's New in v0.6.0

### Backend Consolidation
- **Unified Observability**: Merged TraceManager, RootCauseAnalyzer, and FHEAuditor into single `observability.py`
- **Consolidated MOAI**: Combined PyTorch and FHE implementations into unified `moai_fhe.py`
- **Clean Module Exports**: Added `__init__.py` files across all major packages for cleaner imports
- **Simplified Architecture**: Reduced code duplication while maintaining all functionality

### Modern Frontend (React 18.3)
- **Zustand State Management**: Reactive stores for system, robot, and UI state
- **React Router v7**: SPA navigation with lazy-loaded routes
- **React Three Fiber**: 3D robot visualization with safety zones
- **Radix UI Components**: Accessible toast notifications and dialogs
- **TanStack React Query**: Server state management with caching
- **Enhanced WebSocket**: Exponential backoff reconnection, heartbeat monitoring
- **Keyboard Shortcuts**: Global shortcuts for power users (Ctrl+S, Ctrl+R, etc.)
- **Comprehensive Tests**: Vitest with 19+ test cases for state management

### Robot Runtime Agent (v0.7.0 Preview)
- **Deployment-Grade Architecture**: Robot-resident runtime with optional cloud
- **Three-Tier Compute Model**: Robot CPU (1kHz) → Robot GPU (30-100Hz) → Cloud (async)
- **Cascaded Perception**: Tiny→Medium→Giant models triggered on-demand
- **Offline Capable**: Robot functions without cloud connectivity
- **See**: [`docs/DEPLOYMENT_ARCHITECTURE.md`](docs/DEPLOYMENT_ARCHITECTURE.md)

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

## Unified Skill Orchestration (7-Layer Architecture)

The entire system follows a layered architecture where each layer has ONE responsibility:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CLOUD                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 3: TASK DECOMPOSITION (async, 100ms-1s)                              ││
│  │  Cloud LLM decomposes "pick up cube" → [locate, approach, grasp, place]     ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 4: SKILL SELECTION (MoE, <10ms)                                      ││
│  │  MoESkillRouter: task_embedding → gating network → skill_ids + weights      ││
│  │  ONE decision: WHAT skill to use?                                           ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 5: ROBOT ASSIGNMENT (Spatial, <1ms)                                  ││
│  │  TaskRouter: strategies = NEAREST, LEAST_BUSY, CAPABILITY, ROUND_ROBIN      ││
│  │  ONE decision: WHICH robot executes?                                        ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 6: SKILL ADAPTATION (Location, <1ms)                                 ││
│  │  LocationAdaptiveSkillManager: workspace → table_height, obstacles, etc.    ││
│  │  ONE decision: HOW to adapt skill params?                                   ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                      │                                           │
└──────────────────────────────────────┼───────────────────────────────────────────┘
                                       │ SkillExecutionPlan (gRPC)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EDGE (Jetson Thor)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 1: PERCEPTION (1kHz safety, 200Hz control)                           ││
│  │  ONVIF Cameras → DINOv3/SAM3 → Robot Detection + Object Detection           ││
│  │  Updates: RobotLocationTracker with positions                               ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 2: SPATIAL AWARENESS                                                 ││
│  │  CameraWorkspaceMapper: camera_id → workspace_ids                           ││
│  │  RobotLocationTracker: robot_id → position, workspace, visible_cameras      ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  Layer 7: EXECUTION (200Hz real-time)                                       ││
│  │  EdgeSkillClient: download → cache → VLA inference → joint actions          ││
│  │  Safety @ 1kHz: V-JEPA 2 + backup model (NEVER bypassed)                    ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Unified Entry Point

All skill orchestration goes through ONE function:

```python
from src.core.unified_skill_orchestrator import get_orchestrator, OrchestrationRequest

orchestrator = get_orchestrator()
result = orchestrator.orchestrate(OrchestrationRequest(
    task_description="Pick up the red cube and place on shelf",
    assignment_strategy=AssignmentStrategy.NEAREST,
    target_position=np.array([1.2, 0.8, 0.8]),
))

# Result contains complete execution plan:
# - plan.steps[]: skill_ids, weights, robot_id, workspace_id, location_params
# - Each step has tier (REALTIME/CONTROL/PLANNING) and dependencies
```

### ONE Decision Per Layer

| Layer | Question | Component | Output |
|-------|----------|-----------|--------|
| **4** | WHAT skill? | `MoESkillRouter.route()` | `skill_ids, weights` |
| **5** | WHICH robot? | `TaskRouter.assign_task()` | `robot_id` |
| **6** | HOW to adapt? | `LocationAdaptiveSkillManager` | `location_params` |
| **7** | WHEN? | `_determine_tier()` | `ExecutionTier` |

---

## Key Features

### Meta AI Foundation Models

State-of-the-art perception powered by Meta's latest foundation models with FP4 quantization:

| Model | Variant | Purpose | FP4 TFLOPS | FP16 TFLOPS |
|-------|---------|---------|------------|-------------|
| **DINOv3** | ViT-Giant | Dense features, zero-shot classification | 120.0 | 8.0 |
| **SAM3** | Huge | Real-time segmentation, video tracking | 200.0 | 15.0 |
| **V-JEPA 2** | Giant | World model, collision prediction | 330.0 | 10.0 |

**Meta AI Total: 650.0 FP4 TFLOPS (31% of compute budget)**

### Unified Observability System (v0.6.0)

Consolidated observability with single import:

```python
from src.platform.observability import (
    get_observability_system,
    TraceManager,
    RootCauseAnalyzer,
    FHEAuditor
)

# Initialize unified observability
obs = get_observability_system()

# Trace recording
obs.trace_manager.start_trace("skill_execution")
obs.trace_manager.add_event("inference", {"model": "pi0", "latency_ms": 4.2})

# Root cause analysis
analysis = obs.root_cause_analyzer.analyze_failure(error_trace)

# FHE auditing
obs.fhe_auditor.log_encryption("gradient_update", metadata)
```

### NVIDIA Isaac Lab Simulation

Sim-to-real transfer with high-fidelity physics simulation:
- Real-time robot visualization
- Domain randomization for robust policies
- Parallel environment training
- Direct deployment to Jetson hardware

### MoE Skill Architecture
- **Frozen Base VLA Models**: Pi0/OpenVLA-7B remain read-only (IP-safe)
- **Trainable Skill Experts**: Lightweight skills that augment base models
- **Dynamic Task Routing**: Natural language → automatic skill selection
- **Skill Blending**: Combine multiple skills with MoE routing weights

### 4-Tier Timing Architecture (Jetson Thor + FP4)
- **Tier 1**: Safety Loop @ 1kHz (V-JEPA 2 Giant + backup model, never throttled)
- **Tier 2**: Control/Perception @ 200Hz (5ms - UPGRADED with FP4 quantization!)
- **Tier 3**: Learning Loop @ 10Hz (IL training, skill distillation)
- **Tier 4**: Background (anomaly detection, FL aggregation)

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
| **Cameras** | ONVIF IP cameras (up to 22) | Multi-view perception |
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
# Start backend API
python -m src.platform.api.main

# Start frontend (development)
cd src/platform/ui
npm install
npm run dev
```

### 3. Access Dashboard

Open: `http://localhost:5173` (Vite dev server) or `http://localhost:8000` (production)

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
│   ├── core/                           # Core algorithms
│   │   ├── __init__.py                     # v0.6.0: Clean exports
│   │   ├── unified_skill_orchestrator.py   # UNIFIED: Single entry point
│   │   ├── timing_architecture.py          # 4-tier timing (Thor optimized)
│   │   ├── system_robustness.py            # Reliability & safety
│   │   ├── meta_ai_models.py               # DINOv3, SAM3, V-JEPA 2
│   │   ├── gmr_retargeting.py              # Motion retargeting
│   │   └── robot_skill_invoker.py          # Control loop skill execution
│   │
│   ├── moai/                           # MOAI Architecture
│   │   └── moai_fhe.py                     # v0.6.0: Unified FHE + PyTorch
│   │
│   ├── platform/
│   │   ├── jetson_thor.py                  # Thor hardware config
│   │   ├── api/                            # REST API (FastAPI)
│   │   ├── cloud/                          # FL, MoE routing
│   │   │   ├── __init__.py                     # v0.6.0: Clean exports
│   │   │   ├── ffm_client.py                   # Federated learning client
│   │   │   └── moe_skill_router.py             # MoE gating network
│   │   ├── edge/                           # Edge components
│   │   ├── observability/                  # v0.6.0: Unified observability
│   │   │   └── observability.py                # TraceManager + RCA + FHE Audit
│   │   └── ui/                             # v0.6.0: Modern React dashboard
│   │
│   ├── drivers/                        # Hardware interfaces
│   │   ├── __init__.py                     # v0.6.0: Clean exports
│   │   ├── dyglove.py                      # WiFi 6E haptic gloves
│   │   ├── onvif_ptz.py                    # PTZ camera control
│   │   └── daimon_vtla.py                  # Robot driver
│   │
│   ├── pipeline/                       # Data pipeline
│   │   └── __init__.py                     # v0.6.0: Clean exports
│   │
│   ├── spatial_intelligence/           # Pi0 VLA Models
│   │   ├── __init__.py                     # v0.6.0: Clean exports
│   │   └── pi0/
│   │       └── __init__.py                     # Pi0 model exports
│   │
│   └── shared/crypto/                  # FHE (128-bit security)
│
├── config/config.yaml                  # System configuration
└── tests/                              # Test suite
```

---

## Frontend Architecture (v0.6.0)

The React dashboard has been completely modernized with a component-based architecture:

### State Management (Zustand)

```javascript
// Three reactive stores for separation of concerns
import { useSystemStore, useRobotStore, useUIStore } from './stores';

// System state (status, TFLOPS, memory, models)
const { status, tflopsUsage, metaAIModels } = useSystemStore();

// Robot state (joints, pose, teleoperation, recording)
const { joints, eePose, startTeleop, stopRecording } = useRobotStore();

// UI state (theme, toasts, notifications)
const { theme, showToast, toggleTheme } = useUIStore();
```

### Component Hierarchy

```
App.jsx
├── Providers (QueryClient, Router, Toast)
├── Routes
│   ├── /dashboard      → Dashboard (lazy)
│   ├── /devices        → DeviceManager (lazy)
│   ├── /skills         → SkillsManager (lazy)
│   ├── /training       → TrainingManager (lazy)
│   ├── /observability  → Observability (lazy)
│   └── /safety         → SafetyZones (lazy)
└── Global Components
    ├── AlertCenter     → Notification panel
    └── Toast           → Toast notifications
```

### 3D Visualization

```javascript
import { RobotVisualizer3D } from './components/visualization/RobotVisualizer3D';

// Real-time 3D robot arm with safety zones
<RobotVisualizer3D
  showSafetyZone={true}
  showTrajectory={true}
  showControls={true}
/>
```

### WebSocket Connection

```javascript
import { useWebSocket } from './hooks/useWebSocket';

// Auto-reconnecting WebSocket with heartbeat
const { isConnected, sendMessage, connectionState } = useWebSocket();

// Connection states: connecting, connected, disconnecting, disconnected
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Toggle sidebar |
| `Ctrl+R` | Start/stop recording |
| `Ctrl+T` | Toggle teleop mode |
| `Ctrl+E` | Emergency stop |
| `Ctrl+D` | Toggle theme |
| `Ctrl+?` | Show help |

### Running Frontend Tests

```bash
cd src/platform/ui

# Run all tests
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage
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

## Compute Budget (2070 FP4 TFLOPS)

### Fixed Allocations (750 TFLOPS)

| Component | TFLOPS | Purpose |
|-----------|--------|---------|
| **Safety V-JEPA 2 Giant** | 150 | Collision prediction @ 1kHz |
| **Safety Ensemble** | 100 | Multi-model safety |
| **Safety Backup** | 100 | Redundant safety model |
| **Force/Torque Prediction** | 50 | Predictive force limiting |
| **Core VLA + ACT** | 80 | Base action models |
| **Learning Pipeline** | 170 | IL, MOAI, distillation |
| **Background** | 100 | Anomaly, spatial, FHE |

### Per-Peripheral Scaling (1320 TFLOPS Available)

| Peripheral | TFLOPS Each | Components |
|------------|-------------|------------|
| **ONVIF Camera** | 60.0 | DINOv3 (15) + SAM3 (25) + Depth (10) + Pose (7.5) + Fusion (2.5) |
| **Glove (DYGlove/MANUS)** | 30.0 | Retargeting (15) + Haptics (5) + Grasp Planning (10) |

### Maximum Peripherals

| Configuration | Cameras | Gloves | Compute Used | Headroom |
|---------------|---------|--------|--------------|----------|
| **Minimal** | 4 | 2 | 990 TFLOPS | 1080 |
| **Recommended** | 12 | 2 | 1470 TFLOPS | 600 |
| **Maximum** | 22 | 4 | 1870 TFLOPS | 200 |

---

## Testing

### Backend Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_config_loader.py
python -m pytest tests/test_whole_body_gmr.py

# Run with coverage
python -m pytest --cov=src
```

### Frontend Tests

```bash
cd src/platform/ui

# Run all tests
npm test

# Watch mode
npm test -- --watch

# With coverage
npm run test:coverage

# Visual UI
npm run test:ui
```

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

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **0.6.0** | Dec 2024 | **Frontend Modernization** - React 18.3, Zustand stores, React Router v7, 3D visualization, unified observability, consolidated Python codebase |
| 0.5.0 | Dec 2024 | Jetson Thor upgrade - 7.5x compute (2070 TFLOPS), 128GB memory, 200Hz control, giant model variants |
| 0.4.0 | Dec 2024 | Meta AI Foundation Models (DINOv3, SAM3, V-JEPA 2), Isaac Lab simulation |
| 0.3.0 | Dec 2024 | MoE skill architecture, N2HE encryption |
| 0.2.0 | Nov 2024 | Safety zones, federated learning |
| 0.1.0 | Oct 2024 | Initial release |

---

## Dependencies

### Backend
- Python 3.8+
- FastAPI 0.100+
- SQLAlchemy 2.0+
- TenSEAL (FHE)
- Flower (FL)
- PyTorch 2.0+

### Frontend
- React 18.3
- Vite 7.2
- Zustand 5.0
- React Router 7.1
- React Three Fiber 8.17
- TanStack React Query 5.62
- Radix UI Components
- Vitest 2.1

---

## License

Proprietary - Dynamical.ai 2024

---

## Support

- **Documentation**: [docs.dynamical.ai](https://docs.dynamical.ai)
- **Email**: support@dynamical.ai
- **Emergency**: emergency@dynamical.ai (safety issues)

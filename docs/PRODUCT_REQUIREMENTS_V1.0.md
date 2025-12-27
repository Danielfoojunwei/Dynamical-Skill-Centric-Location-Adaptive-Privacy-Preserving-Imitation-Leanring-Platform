# Product Requirements Document: GUI Platform Update v1.0

## Executive Summary

This document defines acceptance criteria for new product features and workflows required to expose recent backend implementations (ARM module, trajectory visualization, skill orchestration, cross-robot transfer) in the Dynamical Platform GUI.

**Document Version:** 1.0
**Date:** 2025-12-27
**Status:** Draft for Review

---

## Gap Analysis Summary

| Category | Backend Endpoints | GUI Pages | Coverage |
|----------|------------------|-----------|----------|
| Core System | 10 | Dashboard | 100% |
| Safety | 6 | Safety.jsx | 100% |
| PTZ Cameras | 10 | DeviceManager | 90% |
| Skills | 9 | SkillsManager | 60% |
| ARM Module | 0 (NEW) | None | 0% |
| Perception | 10 | PerceptionManager | 80% |
| Observability | 7 | Observability | 70% |
| Integrator | 41 | DeploymentManager | 60% |

**Critical Gaps Identified:**
1. ARM Pipeline - Zero API/GUI exposure
2. Trajectory Visualization - No UI component
3. Action Reasoning Display - Not shown to users
4. Skill Orchestration - API exists, no UI
5. Cross-Robot Transfer - Backend only
6. Privacy/FHE Transparency - Hidden from users

---

## Feature 1: ARM Pipeline Integration

### 1.1 Overview

**Feature Name:** Action Reasoning Model (ARM) Pipeline
**Priority:** P0 - Critical
**Backend Status:** Complete (`src/spatial_intelligence/arm/`)
**Target Components:** New API endpoints + New GUI page

### 1.2 User Stories

#### US-1.1: Execute ARM Pipeline
**As a** robot operator
**I want to** execute the ARM pipeline on a camera image with natural language instruction
**So that** I can generate interpretable robot manipulation plans

#### US-1.2: View Trajectory Visualization
**As a** system integrator
**I want to** see the predicted trajectory overlaid on the camera image
**So that** I can verify the robot's planned path before execution

#### US-1.3: Apply User Guidance (Steerability)
**As a** safety engineer
**I want to** modify the predicted trajectory by adding waypoints or avoid regions
**So that** I can ensure the robot avoids obstacles or follows preferred paths

#### US-1.4: Cross-Robot Execution
**As a** fleet manager
**I want to** execute the same task plan on different robot types
**So that** I can deploy skills across heterogeneous robot fleets

### 1.3 Acceptance Criteria

#### AC-1.1: ARM Pipeline Execution API

```gherkin
Feature: ARM Pipeline API

  Scenario: Execute ARM pipeline with image and instruction
    Given a valid API authentication token
    And a camera image (JPEG/PNG, max 4MB)
    And a natural language instruction (max 500 chars)
    When I POST to /api/v1/arm/execute with:
      | Field       | Value                          |
      | image       | base64-encoded image           |
      | instruction | "pick up the red cup"          |
      | robot_id    | "ur10e"                        |
    Then the response status should be 200
    And the response should contain:
      | Field                  | Type    |
      | result_id              | string  |
      | trajectory_trace       | object  |
      | decoded_actions        | object  |
      | reasoning              | object  |
      | total_time_ms          | number  |
    And trajectory_trace should have:
      | Field           | Constraint        |
      | waypoints       | array, len >= 4   |
      | confidences     | array, same len   |
      | mean_confidence | number, 0.0-1.0   |

  Scenario: Execute ARM with depth map
    Given valid authentication
    And a camera image and depth map
    When I POST to /api/v1/arm/execute with depth_map included
    Then the response should include depth_tokens
    And depth_time_ms should be > 0

  Scenario: Execute ARM with user guidance
    Given valid authentication
    And a previous ARM result
    When I POST to /api/v1/arm/steer with:
      | Field            | Value                     |
      | result_id        | previous result ID        |
      | guidance_type    | "add_waypoint"            |
      | waypoint         | {"x": 150, "y": 200}      |
    Then the response should contain modified trajectory_trace
    And user_guidance_applied should be true

  Scenario: Reject invalid image format
    Given valid authentication
    When I POST to /api/v1/arm/execute with invalid image
    Then the response status should be 400
    And the error message should indicate "Invalid image format"

  Scenario: Handle unsupported robot type
    Given valid authentication
    When I POST with robot_id = "unsupported_robot"
    Then the response status should be 400
    And the error should list supported robots
```

#### AC-1.2: ARM Pipeline GUI Page

```gherkin
Feature: ARM Planning Page

  Background:
    Given I am logged in as a robot operator
    And I navigate to the ARM Planning page

  Scenario: ARM page displays correctly
    Then I should see the following sections:
      | Section              |
      | Image Upload         |
      | Instruction Input    |
      | Robot Selector       |
      | Execute Button       |
      | Trajectory Canvas    |
      | Reasoning Panel      |
      | Timing Metrics       |

  Scenario: Upload and execute ARM pipeline
    When I upload a camera image
    And I enter instruction "pick up the blue block"
    And I select robot "UR10e"
    And I click "Execute ARM"
    Then the Execute button should show loading state
    And within 5 seconds the Trajectory Canvas should display:
      | Element                | Description                    |
      | Source image           | Uploaded image as background   |
      | Waypoint markers       | Numbered circles on trajectory |
      | Trajectory line        | Curved line connecting points  |
      | Confidence heatmap     | Color gradient on waypoints    |
    And the Reasoning Panel should show:
      | Section               |
      | Perception Reasoning  |
      | Spatial Reasoning     |
      | Action Reasoning      |
    And the Timing Metrics should display:
      | Metric          | Unit |
      | Total Time      | ms   |
      | Depth Time      | ms   |
      | Trajectory Time | ms   |
      | Decoding Time   | ms   |

  Scenario: Interactive trajectory editing (Steerability)
    Given an ARM result is displayed
    When I click "Edit Mode"
    Then I should enter trajectory editing mode
    And I should be able to:
      | Action                    |
      | Click to add waypoint     |
      | Drag waypoint to move     |
      | Double-click to delete    |
      | Draw avoid region         |
    When I click "Apply Changes"
    Then the trajectory should update
    And the guidance_applied indicator should be true

  Scenario: Cross-robot comparison
    Given an ARM result for UR10e
    When I click "Compare Robots"
    And I select multiple robots: ["UR10e", "Franka", "Custom"]
    Then I should see a comparison table with:
      | Column               |
      | Robot Name           |
      | Action Horizon       |
      | IK Success Rate      |
      | Estimated Duration   |

  Scenario: Export ARM result
    Given an ARM result is displayed
    When I click "Export"
    Then I should be able to download:
      | Format     | Contents                        |
      | JSON       | Full result object              |
      | Image      | Trajectory visualization        |
      | Actions    | Joint action sequence (CSV)     |
```

#### AC-1.3: ARM API Endpoints Specification

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/v1/arm/execute` | POST | Execute ARM pipeline | Required |
| `/api/v1/arm/result/{result_id}` | GET | Get cached result | Required |
| `/api/v1/arm/visualization/{result_id}` | GET | Get visualization image | Required |
| `/api/v1/arm/steer` | POST | Apply user guidance | Required |
| `/api/v1/arm/robots` | GET | List supported robots | Public |
| `/api/v1/arm/stats` | GET | Get pipeline statistics | Required |

---

## Feature 2: Trajectory Visualization Component

### 2.1 Overview

**Feature Name:** TrajectoryVisualizer Component
**Priority:** P0 - Critical
**Target Components:** New React component + Integration with Observability

### 2.2 User Stories

#### US-2.1: View Trajectory Overlay
**As a** safety engineer
**I want to** see predicted trajectories overlaid on camera feeds
**So that** I can verify robot paths are safe

#### US-2.2: Inspect Waypoint Details
**As a** debugging engineer
**I want to** click on waypoints to see confidence scores and reasoning
**So that** I can diagnose planning issues

### 2.3 Acceptance Criteria

```gherkin
Feature: Trajectory Visualization Component

  Scenario: Render trajectory on image
    Given a TrajectoryTrace object with 8 waypoints
    And a source image (640x480)
    When the TrajectoryVisualizer component renders
    Then I should see:
      | Element            | Count/Description            |
      | Waypoint markers   | 8 numbered circles           |
      | Trajectory line    | Smooth bezier curve          |
      | Start indicator    | Green circle at waypoint 0   |
      | End indicator      | Red circle at last waypoint  |

  Scenario: Confidence-based coloring
    Given waypoints with varying confidence (0.6 to 0.95)
    Then waypoint markers should be colored:
      | Confidence Range | Color  |
      | 0.0 - 0.5        | Red    |
      | 0.5 - 0.7        | Orange |
      | 0.7 - 0.85       | Yellow |
      | 0.85 - 1.0       | Green  |

  Scenario: Hover waypoint for details
    When I hover over waypoint 3
    Then a tooltip should display:
      | Field        | Example Value  |
      | Index        | 3              |
      | Position     | (156, 234)     |
      | Confidence   | 0.92           |
      | Depth        | 0.45m          |
      | Reasoning    | "Approach..."  |

  Scenario: Click waypoint for full details
    When I click on waypoint 3
    Then a side panel should expand showing:
      | Section                |
      | Waypoint Coordinates   |
      | Confidence Score       |
      | Associated Reasoning   |
      | 3D Position (if depth) |
      | Joint Configuration    |

  Scenario: Animate trajectory playback
    When I click "Play Trajectory"
    Then an animation should show:
      | Frame | Description                          |
      | 1-N   | Marker moving along trajectory path  |
    And playback controls should include:
      | Control      |
      | Play/Pause   |
      | Speed (0.5x-2x) |
      | Step Forward |
      | Step Back    |

  Scenario: Zoom and pan
    When I scroll on the trajectory canvas
    Then the view should zoom in/out
    When I drag on the canvas
    Then the view should pan
    And a minimap should show current viewport position

  Scenario: Multiple trajectory comparison
    Given two TrajectoryTrace objects (original and steered)
    When comparison mode is enabled
    Then both trajectories should render with:
      | Trajectory | Style         |
      | Original   | Solid line    |
      | Steered    | Dashed line   |
    And differences should be highlighted
```

---

## Feature 3: Skill Orchestration UI

### 3.1 Overview

**Feature Name:** Skill Orchestration Dashboard
**Priority:** P1 - High
**Backend Status:** Complete (`/api/v1/skills/orchestrate`)
**Target Components:** New section in SkillsManager.jsx

### 3.2 User Stories

#### US-3.1: View Task Decomposition
**As a** robot programmer
**I want to** see how complex tasks are decomposed into skill sequences
**So that** I can understand and debug orchestration logic

#### US-3.2: Monitor Skill Execution
**As a** operations manager
**I want to** see real-time progress of skill orchestration
**So that** I can monitor production tasks

### 3.3 Acceptance Criteria

```gherkin
Feature: Skill Orchestration UI

  Background:
    Given I am on the Skills Manager page
    And I click the "Orchestration" tab

  Scenario: Submit task for orchestration
    When I enter task: "Pick up the red cup and place it on the shelf"
    And I click "Orchestrate"
    Then the system should display:
      | Section               | Content                      |
      | Task Analysis         | Parsed task requirements     |
      | Skill Decomposition   | List of required skills      |
      | Execution Plan        | Ordered skill sequence       |
      | Estimated Duration    | Total time estimate          |

  Scenario: View skill decomposition graph
    Given an orchestration result
    Then I should see a directed graph showing:
      | Node Type     | Description            |
      | Task Node     | Root task (top)        |
      | Skill Nodes   | Individual skills      |
      | Dependency    | Arrows between nodes   |
    And each skill node should display:
      | Field         |
      | Skill Name    |
      | Skill ID      |
      | Confidence    |
      | Dependencies  |

  Scenario: Execute orchestrated plan
    Given an orchestration result
    When I click "Execute Plan"
    Then each skill node should show status:
      | Status     | Color   |
      | Pending    | Gray    |
      | Running    | Blue    |
      | Completed  | Green   |
      | Failed     | Red     |
    And progress bar should update in real-time
    And logs should stream for current skill

  Scenario: View MoE routing decisions
    Given an orchestration with multiple skill options
    Then I should see the MoE routing panel showing:
      | Column          | Description              |
      | Skill Options   | Available skills         |
      | Router Score    | MoE confidence score     |
      | Selected        | Checkmark on chosen      |
      | Rationale       | Why skill was chosen     |

  Scenario: Manual skill override
    Given an orchestration result before execution
    When I click on a skill node
    Then I should be able to:
      | Action                      |
      | Replace with different skill|
      | Add prerequisite skill      |
      | Remove skill                |
      | Modify parameters           |
    When I confirm changes
    Then the execution plan should update
```

---

## Feature 4: Cross-Robot Transfer UI

### 4.1 Overview

**Feature Name:** Cross-Robot Transfer Dashboard
**Priority:** P1 - High
**Backend Status:** Complete (ARM + RobotRegistry)
**Target Components:** New section in DeploymentManager.jsx

### 4.2 User Stories

#### US-4.1: View Robot Compatibility
**As a** fleet manager
**I want to** see which skills can transfer between robot types
**So that** I can plan deployments efficiently

#### US-4.2: Execute Cross-Robot Transfer
**As a** integrator
**I want to** deploy a skill trained on one robot to a different robot
**So that** I can maximize skill reuse

### 4.3 Acceptance Criteria

```gherkin
Feature: Cross-Robot Transfer

  Background:
    Given I am on the Deployment Manager page
    And I click the "Cross-Robot Transfer" tab

  Scenario: View robot fleet
    Then I should see a table of registered robots:
      | Column          | Description              |
      | Robot ID        | Unique identifier        |
      | Robot Type      | UR10e, Franka, etc.      |
      | DOF             | Degrees of freedom       |
      | Status          | Online/Offline           |
      | Skill Count     | Deployed skills          |

  Scenario: View transfer compatibility matrix
    When I click "Compatibility Matrix"
    Then I should see a matrix showing:
      | Row/Column    | Description                    |
      | Source Robot  | Robot skill was trained on     |
      | Target Robot  | Robot to transfer to           |
      | Cell Value    | Compatibility score (0-100%)   |
    And cells should be color-coded:
      | Score Range | Color  |
      | 0-50%       | Red    |
      | 50-75%      | Yellow |
      | 75-100%     | Green  |

  Scenario: Execute skill transfer
    When I select a skill from source robot "UR10e"
    And I select target robot "Franka"
    And I click "Transfer Skill"
    Then the system should:
      | Step                        | Status |
      | Load skill weights          | ✓      |
      | Adapt to target embodiment  | ✓      |
      | Validate IK feasibility     | ✓      |
      | Deploy to target            | ✓      |
    And display transfer results:
      | Metric                   |
      | IK Success Rate          |
      | Workspace Coverage       |
      | Estimated Performance    |

  Scenario: View transfer preview
    Given a pending skill transfer
    When I click "Preview Transfer"
    Then I should see side-by-side:
      | Left Panel       | Right Panel       |
      | Source robot viz | Target robot viz  |
      | Original actions | Adapted actions   |
    And trajectory comparison overlay

  Scenario: Batch transfer
    When I select multiple skills
    And I select multiple target robots
    And I click "Batch Transfer"
    Then I should see a queue of transfers
    And progress for each transfer
    And summary when complete
```

---

## Feature 5: Action Reasoning Display

### 5.1 Overview

**Feature Name:** Chain-of-Thought Reasoning Panel
**Priority:** P1 - High
**Backend Status:** Complete (`action_reasoning.py`)
**Target Components:** Integration with Observability.jsx

### 5.2 User Stories

#### US-5.1: View Action Reasoning
**As a** safety engineer
**I want to** see the reasoning behind robot actions
**So that** I can verify decisions are appropriate

#### US-5.2: Debug Reasoning Failures
**As a** ML engineer
**I want to** inspect reasoning traces during failures
**So that** I can improve the reasoning model

### 5.3 Acceptance Criteria

```gherkin
Feature: Action Reasoning Display

  Background:
    Given I am on the Observability page

  Scenario: View reasoning for current action
    When the robot is executing an action
    Then the Reasoning Panel should display:
      | Section               | Content                        |
      | Perception Reasoning  | What the robot sees            |
      | Spatial Reasoning     | Spatial relationships          |
      | Action Reasoning      | Why this action was chosen     |
      | Confidence Scores     | Per-section confidence         |

  Scenario: View reasoning timeline
    Given an episode with multiple actions
    When I click "Reasoning Timeline"
    Then I should see a timeline with:
      | Element          | Description                   |
      | Time markers     | Action timestamps             |
      | Reasoning cards  | Expandable reasoning per step |
      | Confidence graph | Confidence over time          |

  Scenario: Filter by reasoning type
    When I filter by "Low Confidence" (< 0.7)
    Then only actions with low confidence should display
    And low-confidence sections should be highlighted

  Scenario: Link reasoning to trajectory
    When I click on a reasoning card
    Then the associated trajectory waypoint should highlight
    And the camera view should show that moment
    And I should see:
      | Field                 |
      | Waypoint index        |
      | Timestamp             |
      | Sensor readings       |
      | Joint positions       |

  Scenario: Export reasoning trace
    When I click "Export Reasoning"
    Then I should be able to download:
      | Format     | Contents                    |
      | JSON       | Full reasoning trace        |
      | Markdown   | Human-readable report       |
      | CSV        | Confidence scores over time |
```

---

## Feature 6: Privacy/FHE Transparency

### 6.1 Overview

**Feature Name:** Privacy Dashboard
**Priority:** P2 - Medium
**Backend Status:** Complete (N2HE integration)
**Target Components:** New section in PerceptionManager.jsx

### 6.2 Acceptance Criteria

```gherkin
Feature: Privacy Dashboard

  Background:
    Given I am on the Perception Manager page
    And I click the "Privacy" tab

  Scenario: View FHE status
    Then I should see:
      | Metric                    | Description             |
      | FHE Enabled               | Yes/No                  |
      | Encryption Scheme         | CKKS/BFV                |
      | Security Level            | 128-bit                 |
      | Operations Count          | Total FHE ops today     |
      | Average Latency           | Per-operation latency   |

  Scenario: View N2HE routing decisions
    When I click "Routing Decisions"
    Then I should see recent routing decisions:
      | Column           | Description              |
      | Timestamp        | When decision was made   |
      | Data Type        | Image/Depth/Pose         |
      | Privacy Level    | Low/Medium/High          |
      | Route Selected   | FHE/TEE/Plaintext        |
      | Latency          | Processing time          |

  Scenario: Configure privacy levels
    When I click "Configure"
    Then I should be able to set:
      | Setting                   | Options            |
      | Default Privacy Level    | Low/Medium/High    |
      | Data Type Overrides      | Per-type settings  |
      | FHE Threshold            | Latency limit      |
    When I save changes
    Then the new configuration should apply

  Scenario: View FHE audit logs
    When I click "Audit Logs"
    Then I should see:
      | Column           |
      | Timestamp        |
      | Operation Type   |
      | Input Size       |
      | Output Size      |
      | Latency          |
      | Requestor        |
```

---

## Feature 7: Model Strategy Tiers Visualization

### 7.1 Overview

**Feature Name:** Perception Model Strategy Display
**Priority:** P2 - Medium
**Backend Status:** Complete (L1/L2/L3 tiers)
**Target Components:** Enhancement to PerceptionManager.jsx

### 7.2 Acceptance Criteria

```gherkin
Feature: Model Strategy Tiers

  Scenario: View current tier
    Given I am on the Perception Manager page
    Then I should see a tier indicator showing:
      | Tier | Models Active    | Trigger Condition      |
      | L1   | DINOv3           | Default                |
      | L2   | DINOv3 + SAM3    | Complex scene          |
      | L3   | All + V-JEPA2    | Dynamic/Safety-critical|
    And the current tier should be highlighted

  Scenario: View tier transition history
    When I click "Tier History"
    Then I should see:
      | Column          |
      | Timestamp       |
      | From Tier       |
      | To Tier         |
      | Trigger Reason  |
      | Duration in Tier|

  Scenario: Manual tier override
    Given operator privileges
    When I click "Override Tier"
    And I select "L3 - Maximum"
    And I confirm the override
    Then the system should switch to L3
    And a warning should display about increased compute
    And an audit log entry should be created

  Scenario: View TFLOPS allocation
    Then I should see TFLOPS allocation per tier:
      | Tier | TFLOPS Allocation |
      | L1   | 2.5 TFLOPS        |
      | L2   | 7.0 TFLOPS        |
      | L3   | 15.0 TFLOPS       |
    And current usage should be displayed
```

---

## Implementation Roadmap

### Phase 1: P0 Critical Features (Sprint 1-2)

| Feature | Components | Effort |
|---------|------------|--------|
| ARM API Endpoints | 6 endpoints | 3 days |
| ARM Planning Page | New page | 5 days |
| TrajectoryVisualizer | New component | 4 days |

**Deliverables:**
- `/api/v1/arm/*` endpoints operational
- ARMPlanning.jsx page with full functionality
- TrajectoryVisualizer.jsx component

### Phase 2: P1 High Features (Sprint 3-4)

| Feature | Components | Effort |
|---------|------------|--------|
| Skill Orchestration UI | Tab in SkillsManager | 4 days |
| Cross-Robot Transfer UI | Tab in DeploymentManager | 4 days |
| Action Reasoning Display | Panel in Observability | 3 days |

**Deliverables:**
- OrchestrationTab.jsx component
- CrossRobotTransferTab.jsx component
- ReasoningPanel.jsx component

### Phase 3: P2 Medium Features (Sprint 5-6)

| Feature | Components | Effort |
|---------|------------|--------|
| Privacy Dashboard | Tab in PerceptionManager | 3 days |
| Model Strategy Tiers | Enhancement to PerceptionManager | 2 days |
| Config Diff Viewer | Enhancement to VersionControl | 2 days |

---

## API Schema Definitions

### ARM Execute Request

```typescript
interface ARMExecuteRequest {
  image: string;           // Base64 encoded image
  instruction: string;     // Natural language instruction
  robot_id: string;        // Target robot identifier
  depth_map?: string;      // Optional base64 depth map
  user_guidance?: {
    type: "add_waypoint" | "delete_waypoint" | "avoid_region";
    data: object;
  };
}
```

### ARM Execute Response

```typescript
interface ARMExecuteResponse {
  result_id: string;
  trajectory_trace: {
    waypoints: [number, number][];
    confidences: number[];
    mean_confidence: number;
    source_image_shape: [number, number, number];
    instruction: string;
  };
  decoded_actions: {
    joint_actions: number[][];
    action_horizon: number;
    success_rate: number;
  };
  reasoning: {
    perception_reasoning: string;
    spatial_reasoning: string;
    action_reasoning: string;
    perception_confidence: number;
    spatial_confidence: number;
    action_confidence: number;
  };
  timing: {
    total_ms: number;
    depth_ms: number;
    trajectory_ms: number;
    decoding_ms: number;
    reasoning_ms: number;
  };
}
```

---

## Testing Requirements

### Unit Tests

- ARM API endpoint input validation
- TrajectoryVisualizer rendering
- Reasoning panel data parsing

### Integration Tests

- ARM pipeline end-to-end execution
- Cross-robot transfer workflow
- Skill orchestration with MoE routing

### E2E Tests

- Full ARM workflow from upload to visualization
- Cross-robot comparison UI
- Privacy settings persistence

### Performance Benchmarks

| Operation | Target Latency |
|-----------|---------------|
| ARM Execute | < 500ms |
| Trajectory Render | < 100ms |
| Reasoning Parse | < 50ms |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| GUI Feature Coverage | 95% | Backend features exposed in UI |
| ARM Pipeline Adoption | 50% | Users using ARM within 30 days |
| Trajectory Visualization Usage | 80% | Sessions with visualization viewed |
| User Steerability Usage | 30% | Trajectories manually edited |
| Cross-Robot Transfer Success | 85% | Successful skill transfers |

---

## Appendix A: Route Inventory

See exploration results for complete list of 156 API endpoints.

## Appendix B: Component Hierarchy

```
App.jsx
├── Dashboard.jsx
├── DeviceManager.jsx
├── Safety.jsx
├── SkillsManager.jsx
│   └── OrchestrationTab.jsx (NEW)
├── Observability.jsx
│   └── ReasoningPanel.jsx (NEW)
├── PerceptionManager.jsx
│   ├── PrivacyTab.jsx (NEW)
│   └── ModelStrategyTiers.jsx (NEW)
├── DeploymentManager.jsx
│   └── CrossRobotTransferTab.jsx (NEW)
├── ARMPlanning.jsx (NEW)
│   ├── TrajectoryVisualizer.jsx (NEW)
│   └── SteerabilityControls.jsx (NEW)
├── TrainingManager.jsx
├── SimulationDashboard.jsx
├── CloudIntegration.jsx
├── Settings.jsx
├── VersionControl.jsx
└── AuditLog.jsx
```

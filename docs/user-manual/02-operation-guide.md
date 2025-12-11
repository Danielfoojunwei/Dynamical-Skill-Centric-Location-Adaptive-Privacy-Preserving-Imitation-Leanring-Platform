# Dynamical Edge Platform - Operation Guide

## Part 2: Operation

This guide teaches you how to use every feature of the Dynamical Edge Platform. We'll go screen by screen, explaining what everything does and how to use it.

---

## Table of Contents

1. [Understanding the Interface](#understanding-the-interface)
2. [Dashboard - Your Control Center](#dashboard---your-control-center)
3. [Devices - Managing Your Hardware](#devices---managing-your-hardware)
4. [Skills - Teaching Your Robot](#skills---teaching-your-robot)
5. [Training - Creating New Skills](#training---creating-new-skills)
6. [Observability - Monitoring & Debugging](#observability---monitoring--debugging)
7. [Safety - Protecting People & Equipment](#safety---protecting-people--equipment)
8. [Cloud - Connecting to the Cloud](#cloud---connecting-to-the-cloud)
9. [Settings - Configuration](#settings---configuration)

---

## Understanding the Interface

When you open the platform, you'll see this layout:

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Dynamical Edge  v0.3.2                                  │
├──────────────┬──────────────────────────────────────────────┤
│              │                                               │
│  Dashboard   │                                               │
│  Devices     │                                               │
│  Skills      │           MAIN CONTENT AREA                   │
│  Observability│          (changes based on                   │
│  Training    │           which menu item                     │
│  Safety      │           you click)                          │
│  Cloud       │                                               │
│  Settings    │                                               │
│              │                                               │
├──────────────┤                                               │
│ SYSTEM STATUS│                                               │
│ ● IDLE       │                                               │
│ 0/137 TFLOPS │                                               │
└──────────────┴──────────────────────────────────────────────┘
```

**Left Sidebar:** Navigation menu and system status
**Main Area:** The content for whatever page you're viewing

---

## Dashboard - Your Control Center

The Dashboard is your home screen. It shows you the overall health of your system at a glance.

### What You'll See

```
┌─────────────────────────────────────────────────────────────┐
│  📊 Dashboard                                    [START]     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│   │   STATUS    │   │   TFLOPS    │   │   MEMORY    │       │
│   │    IDLE     │   │   0/137     │   │   2.1 GB    │       │
│   └─────────────┘   └─────────────┘   └─────────────┘       │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              TFLOPS USAGE CHART                      │   │
│   │                    ┌───┐                             │   │
│   │                  ┌─┤   ├─┐                           │   │
│   │                ┌─┤ Free ├─┐                          │   │
│   │                │ └───────┘ │                         │   │
│   │                └───────────┘                         │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   Active Components: SafetyLoop, Navigation, VLA            │
│   Uptime: 45 minutes                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Metrics Explained

| Metric | What It Means | Good Range |
|--------|---------------|------------|
| **Status** | Is the system running? | OPERATIONAL = good |
| **TFLOPS** | Computing power in use | Below 110 is safe |
| **Memory** | RAM being used | Below 28 GB is safe |
| **Uptime** | How long system has been running | Any value |

### Starting and Stopping the System

**To Start:**
1. Click the big green **"START SYSTEM"** button
2. Wait for status to change to "OPERATIONAL"
3. The TFLOPS meter will show current usage

**To Stop:**
1. Click the red **"STOP SYSTEM"** button
2. Wait for status to change to "IDLE"
3. All processing will halt

> **⚠️ Warning:** Stopping the system will halt all robot operations immediately. Make sure the robot is in a safe position first!

---

## Devices - Managing Your Hardware

The Devices page lets you see, configure, and control all connected equipment.

### Device List Overview

```
┌─────────────────────────────────────────────────────────────┐
│  🖥️ Device Manager                          [Scan Network]   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Total: 5    Online: 4    Cameras: 2    Robots: 1          │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 📹 front-camera     │ ONVIF   │ ● Online │ [PTZ]    │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │ 📹 side-camera      │ ONVIF   │ ● Online │ [PTZ]    │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │ 🧤 right-glove      │ DYGLOVE │ ● Online │ [Calibrate]│  │
│   ├─────────────────────────────────────────────────────┤   │
│   │ 🤖 robot-arm        │ VTLA    │ ● Online │ [Config]  │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │ 🧤 left-glove       │ DYGLOVE │ ○ Offline│ [Calibrate]│  │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Controlling PTZ Cameras

PTZ stands for "Pan-Tilt-Zoom" - cameras that can move and zoom.

**To control a camera:**

1. Click the **[PTZ]** button next to a camera
2. A control panel opens:

```
┌─────────────────────────────────────────┐
│  Camera: front-camera                    │
├─────────────────────────────────────────┤
│                                          │
│   [Camera Preview Image Here]            │
│                                          │
├─────────────────────────────────────────┤
│         Movement Controls                │
│                                          │
│              [ ↑ ]                       │
│         [ ← ] [⌂] [ → ]                 │
│              [ ↓ ]                       │
│                                          │
│   Zoom: [ - ] ═══════○═══ [ + ]         │
│   Speed: ═══○═══════════                 │
│                                          │
├─────────────────────────────────────────┤
│  Presets: [Home] [Position 1] [Position 2]│
│           [Save Current Position]         │
└─────────────────────────────────────────┘
```

**Movement buttons:**
- **↑ ↓ ← →** Move the camera in that direction
- **⌂** Return to home/center position
- **+ -** Zoom in or out
- **Speed slider** How fast the camera moves

**Using Presets:**
- Click a preset name to move camera to that saved position
- Click "Save Current Position" to create a new preset

### Calibrating Gloves

The DYGlove needs calibration to accurately track your hand movements.

**To calibrate a glove:**

1. Click **[Calibrate]** next to the glove
2. Follow the on-screen instructions:

```
┌─────────────────────────────────────────┐
│  Glove Calibration: right-glove          │
├─────────────────────────────────────────┤
│                                          │
│  Step 1 of 4: FLAT HAND                  │
│  ═══════════○───────────  25%            │
│                                          │
│  📋 Instructions:                        │
│  Hold your hand flat with fingers        │
│  extended and together.                  │
│  Keep still for 2 seconds.               │
│                                          │
│  [🖐️ Hand Illustration]                  │
│                                          │
│            [CAPTURE]                     │
│                                          │
└─────────────────────────────────────────┘
```

**The 4 calibration steps:**

| Step | Pose | Purpose |
|------|------|---------|
| 1 | Flat Hand | Sets the "zero" position |
| 2 | Full Fist | Captures maximum finger curl |
| 3 | Pinch Grip | Calibrates thumb-finger coordination |
| 4 | Spread Fingers | Calibrates finger spread |

After each pose, click **[CAPTURE]** when you're ready.

**Testing Haptic Feedback:**

After calibration, test the motors:
1. Click **[Test Haptics]**
2. You'll feel a gentle vibration in each finger
3. Verify you feel feedback in: Thumb, Index, Middle, Ring, Pinky

---

## Skills - Teaching Your Robot

Skills are the movements and actions your robot can perform. Think of them like apps for your robot.

### Understanding the Skills Page

```
┌─────────────────────────────────────────────────────────────┐
│  ⚡ MoE Skill Library                        [Upload Skill]  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Architecture: MoE_Skills    Base Model: 🔒 FROZEN           │
│  Total Skills: 12            MoE Balance: 94.2%              │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  🧠 MoE Task Router                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Describe your task...                        [Route]│    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Filter: [All Types ▼]    Search: [________________]         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 🧠 Grasp    │  │ 🧠 Place    │  │ 🧠 Pour     │          │
│  │ Object     │  │ Object     │  │ Liquid     │          │
│  │             │  │             │  │             │          │
│  │ manipulation│  │ manipulation│  │ manipulation│          │
│  │ v1.2.0     │  │ v1.0.0     │  │ v1.1.0     │          │
│  │ ● active    │  │ ● active    │  │ ● active    │          │
│  │             │  │             │  │             │          │
│  │ [Invoke]    │  │ [Invoke]    │  │ [Invoke]    │          │
│  │ [Download]  │  │ [Download]  │  │ [Download]  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Using the Task Router

The Task Router is like a smart assistant that picks the best skill(s) for your task.

**How to use it:**

1. Type what you want the robot to do:
   - "Pick up the red cube"
   - "Pour water into the glass"
   - "Hand me the screwdriver"

2. Click **[Route]**

3. See the results:
```
Routed in 3.45ms
┌────────────────────────────────────────┐
│ grasp_object       │ 65.2% weight      │
│ approach_target    │ 24.8% weight      │
│ precision_grip     │ 10.0% weight      │
└────────────────────────────────────────┘
```

The system shows which skills it would combine, with percentages showing how much each skill contributes.

### Skill Types Explained

| Type | Description | Examples |
|------|-------------|----------|
| **Manipulation** | Moving and handling objects | Grasp, Place, Pour, Stack |
| **Navigation** | Moving around the environment | Go To, Avoid, Follow |
| **Perception** | Understanding what's around | Detect, Track, Identify |
| **Interaction** | Working with humans | Handover, Point, Gesture |

### Invoking a Skill Directly

To make the robot perform a specific skill immediately:

1. Find the skill card
2. Click **[Invoke]**
3. The robot will execute that skill

### Uploading New Skills

If you've trained a new skill, you can add it to the library:

1. Click **[Upload Skill]** in the top right
2. Fill in the details:
   - **Name:** What to call this skill
   - **Description:** What it does
   - **Type:** Manipulation/Navigation/Perception/Interaction
   - **Tags:** Keywords for searching (e.g., "gripper", "precision")
   - **Version:** Start with "1.0.0"
3. Click **[Upload]**

---

## Training - Creating New Skills

The Training page is where you manage datasets and train new skills for your robot.

### The Four Training Tabs

```
┌─────────────────────────────────────────────────────────────┐
│  📊 Training & Data Management                               │
├─────────────────────────────────────────────────────────────┤
│  [Datasets]  [Training Jobs]  [Version Control]  [FL Status] │
├─────────────────────────────────────────────────────────────┤
```

### Tab 1: Datasets

Datasets are collections of examples showing the robot what to do.

```
┌─────────────────────────────────────────────────────────────┐
│  Total: 3    Samples: 2,572    Storage: 0.85 GB    Processing: 1│
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Dataset          │ Type         │ Samples │ Status         │
│  ─────────────────┼──────────────┼─────────┼───────────     │
│  Grasp Demos v1   │ demonstration│ 1,250   │ ● ready        │
│  Navigation Routes│ trajectory   │ 890     │ ● ready        │
│  Tool Use Collection│demonstration│ 432     │ ◐ processing   │
│                                                              │
│                                        [Upload New Dataset]  │
└─────────────────────────────────────────────────────────────┘
```

**Dataset Types:**
- **Demonstration:** Videos of a human doing the task
- **Trajectory:** Recorded robot movements

**To train from a dataset:**
1. Find a dataset with "ready" status
2. Click **[Train]** next to it
3. A new training job will start

### Tab 2: Training Jobs

See all ongoing and completed training runs.

```
┌─────────────────────────────────────────────────────────────┐
│  precise_grasp                                               │
│  ══════════════════════════════════○──  67%                 │
│  Epoch: 34/100    Loss: 0.0234    Status: ● running         │
├─────────────────────────────────────────────────────────────┤
│  pour_liquid                                                 │
│  ══════════════════════════════════════  100%               │
│  Epoch: 100/100   Loss: 0.0089    Status: ● completed       │
├─────────────────────────────────────────────────────────────┤
│  stack_blocks                                                │
│  ──────────────────────────────────────  0%                 │
│  Epoch: 0/100     Loss: --        Status: ◌ queued          │
└─────────────────────────────────────────────────────────────┘
```

**Understanding the metrics:**
- **Epoch:** How many times through the training data (higher = more training)
- **Loss:** How well it's learning (lower = better)
- **Progress:** Percentage complete

### Tab 3: Version Control

Track different versions of your skills - like save points in a video game.

```
┌─────────────────────────────────────────────────────────────┐
│  Skill Version History: precise_grasp                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ● v1.2.0  │ 2024-12-10 │ 94.2% success │ ✓ DEPLOYED        │
│  │                                                           │
│  ● v1.1.0  │ 2024-12-08 │ 89.1% success │   archived        │
│  │                         [Rollback to this version]        │
│  ● v1.0.0  │ 2024-12-05 │ 82.3% success │   archived        │
│                           [Rollback to this version]        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**If a new version isn't working well:**
1. Find a previous version with good success rate
2. Click **[Rollback to this version]**
3. That version becomes active again

### Tab 4: Federated Learning Status

See how your robot is learning alongside others (while keeping data private).

```
┌─────────────────────────────────────────────────────────────┐
│  Current Round: 42     Participants: 8                       │
│  Aggregation: FedAvg   Encryption: N2HE (128-bit)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✓ All skill updates encrypted with N2HE (128-bit)          │
│  ✓ Base VLA models remain frozen (IP-safe)                  │
│                                                              │
│  Last Aggregation: 45 minutes ago                            │
│  Skills Updated: 3                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What this means:**
- Your robot learns from other robots' experiences
- Your data stays private (encrypted)
- You benefit from collective learning without sharing raw data

---

## Observability - Monitoring & Debugging

When something goes wrong (or you want to understand what's happening), this is where you look.

### The Four Observability Tabs

```
┌─────────────────────────────────────────────────────────────┐
│  👁️ Observability                                            │
├─────────────────────────────────────────────────────────────┤
│  [Flight Recorder]  [VLA Model]  [FHE Audit]  [Root Cause]   │
├─────────────────────────────────────────────────────────────┘
```

### Tab 1: Flight Recorder (Blackbox)

Like an airplane's black box - records everything that happens.

```
┌─────────────────────────────────────────────────────────────┐
│  Recording: ● ACTIVE    Events: 1,247    Last: 2 sec ago     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  14:32:05 │ info    │ SafetyLoop │ Human detected at 2.3m   │
│  14:32:04 │ info    │ Navigation │ Moving to waypoint 3     │
│  14:32:03 │ warning │ VLA        │ Low confidence: 62%      │
│  14:32:01 │ info    │ SkillRouter│ Routing to grasp_object  │
│  14:31:58 │ error   │ Camera     │ Frame dropped on cam-2   │
│                                                              │
│  [Export Full Log]    [Filter: All ▼]                        │
└─────────────────────────────────────────────────────────────┘
```

**Event levels:**
- **Info** (gray): Normal operations
- **Warning** (yellow): Something unusual, but not critical
- **Error** (red): Something went wrong

### Tab 2: VLA Model Status

Shows the status of the robot's "brain" (Vision-Language-Action model).

```
┌─────────────────────────────────────────────────────────────┐
│  Model Status                                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Loaded:       YES                                     │
│  Inference Confidence: 87%                                   │
│  Inference Latency:   12.3 ms                               │
│                                                              │
│  Base Model: Pi0 / OpenVLA-7B                               │
│  Mode: READ-ONLY (frozen)                                    │
│  MoE Augmentation: ACTIVE                                    │
│  Encryption: N2HE 128-bit                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key things to check:**
- **Confidence** should be above 70% for reliable operation
- **Latency** should be below 50ms for real-time control

### Tab 3: FHE Audit

Shows all encryption operations - proof that your data stays private.

```
┌─────────────────────────────────────────────────────────────┐
│  Encryption: N2HE    Security: 128-bit    Entries: 156       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Time     │ Operation │ Data Type  │ Size   │ Status        │
│  ─────────┼───────────┼────────────┼────────┼───────        │
│  14:30:00 │ encrypt   │ gradients  │ 2.4 MB │ ✓ success     │
│  14:29:45 │ aggregate │ weights    │ 1.2 MB │ ✓ success     │
│  14:29:30 │ decrypt   │ skill      │ 456 KB │ ✓ success     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Tab 4: Root Cause Analysis

When something goes wrong, this helps you figure out why.

**To analyze an incident:**

1. Click **[Trigger Manual Incident]** if you want to capture current state
2. Or click on a past incident from the list
3. Click **[Analyze]**

The system will show:
```
┌─────────────────────────────────────────────────────────────┐
│  Root Cause Analysis Report                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Status: ⚠️ WARNING                                          │
│                                                              │
│  Root Cause:                                                 │
│  Camera frame drops detected on camera-2, causing           │
│  degraded pose estimation confidence.                        │
│                                                              │
│  Recommendations:                                            │
│  1. Check network connection to camera-2                     │
│  2. Reduce camera resolution if bandwidth limited            │
│  3. Consider adding lighting to reduce motion blur           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Safety - Protecting People & Equipment

The Safety page is **critical** - it defines where the robot can and cannot go.

### Safety Zones

```
┌─────────────────────────────────────────────────────────────┐
│  🛡️ Safety Configuration                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                      │    │
│  │     [Workspace Map - Interactive Canvas]             │    │
│  │                                                      │    │
│  │        ┌────────┐                                    │    │
│  │        │ KEEP   │  <-- Red zone: Robot STOPS        │    │
│  │        │ OUT    │                                    │    │
│  │        └────────┘                                    │    │
│  │                     ┌──────────┐                     │    │
│  │                     │ SLOW     │  <-- Orange zone:  │    │
│  │                     │ DOWN     │      Robot slows   │    │
│  │                     └──────────┘                     │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  [Draw New Zone]    [Delete Selected]                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Zone Types

| Zone Type | Color | Robot Behavior |
|-----------|-------|----------------|
| **KEEP_OUT** | Red | Robot stops immediately if it enters |
| **SLOW_DOWN** | Orange | Robot reduces speed in this area |

### Creating a Safety Zone

1. Click **[Draw New Zone]**
2. Click on the map to place points (at least 3)
3. Double-click to finish the shape
4. Choose zone type: KEEP_OUT or SLOW_DOWN
5. Give it a name (e.g., "Near electrical panel")
6. Click **[Save Zone]**

### Safety Settings

```
┌─────────────────────────────────────────────────────────────┐
│  Safety Parameters                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Human Detection Sensitivity                                 │
│  Low ═══════════○═════════════ High                         │
│         (Current: 0.8)                                       │
│                                                              │
│  Emergency Stop Distance                                     │
│  0.5m ═════○═══════════════════ 5.0m                        │
│       (Current: 1.5m)                                        │
│                                                              │
│                                            [Save Settings]   │
└─────────────────────────────────────────────────────────────┘
```

**What these mean:**
- **Human Detection Sensitivity:** Higher = more cautious (may stop for false alarms)
- **Stop Distance:** How close a human can get before robot stops

> **⚠️ Important:** Always err on the side of caution. Start with high sensitivity and large stop distances, then adjust if needed.

---

## Cloud - Connecting to the Cloud

The Cloud page manages your connection to Dynamical's cloud services.

### Cloud Status

```
┌─────────────────────────────────────────────────────────────┐
│  ☁️ Cloud Integration                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Connection: ● Connected                                     │
│  Provider: Dynamical Cloud                                   │
│  Last Sync: 5 minutes ago                                    │
│                                                              │
│  Architecture: MoE_Skills                                    │
│  Base Model: 🔒 FROZEN (protected)                           │
│  Skill Library: 42 skills available                          │
│                                                              │
│  [Check for Updates]    [Sync Now]                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### What Syncs with the Cloud

| Synced | Not Synced (Private) |
|--------|---------------------|
| Skill updates | Your raw video data |
| Encrypted gradients | Personal information |
| Performance metrics | Detailed logs |

### Activity Log

See what's been happening with cloud communication:

```
┌─────────────────────────────────────────────────────────────┐
│  Recent Activity                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  14:30:00 │ Sync     │ Downloaded 2 new skills │ ✓ success  │
│  14:25:00 │ Upload   │ Sent encrypted gradients│ ✓ success  │
│  14:20:00 │ Sync     │ Skill library update    │ ✓ success  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Settings - Configuration

Basic system configuration options.

### Camera Settings

```
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ Settings                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Camera RTSP URL                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ rtsp://192.168.1.100:554/stream                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│                                            [Save Settings]   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**When to change this:**
- If you have a main camera at a different address
- If your camera uses a different RTSP path

---

## Quick Reference: Common Tasks

| Task | Steps |
|------|-------|
| **Start the robot system** | Dashboard → Click "START SYSTEM" |
| **Stop the robot system** | Dashboard → Click "STOP SYSTEM" |
| **Move a camera** | Devices → Click [PTZ] → Use arrow buttons |
| **Calibrate a glove** | Devices → Click [Calibrate] → Follow 4 steps |
| **Run a skill** | Skills → Type task → Click [Route] |
| **Check for problems** | Observability → Flight Recorder |
| **Add a safety zone** | Safety → Draw New Zone → Save |
| **Sync with cloud** | Cloud → Click [Sync Now] |

---

## What's Next?

Now you know how to operate the platform. Continue to:

- **[Part 3: Post-Deployment Management](./03-post-deployment-guide.md)** - Maintenance, updates, and troubleshooting

---

*Dynamical Edge Platform v0.3.2 - Operation Guide*
*For support, visit: https://dynamical.ai/support*

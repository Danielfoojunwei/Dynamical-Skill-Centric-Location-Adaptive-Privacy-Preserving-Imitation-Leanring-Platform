"""
ARM (Action Reasoning Model) API Router

Provides REST endpoints for the ARM pipeline:
- Execute ARM pipeline with image and instruction
- Retrieve cached results
- Apply user guidance (steerability)
- Get visualizations
- Cross-robot transfer execution

Based on MolmoAct-inspired architecture with:
- Trajectory prediction and visualization
- Chain-of-thought reasoning
- Embodiment-agnostic planning
- User steerability

@version 1.0.0
"""

import os
import time
import base64
import uuid
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from fastapi import APIRouter, HTTPException, Depends, Security, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY = os.getenv("API_KEY", "default_insecure_key")


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API key for protected endpoints."""
    if api_key_header == API_KEY:
        return api_key_header
    if api_key_header is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )


# Create router
router = APIRouter(prefix="/api/v1/arm", tags=["ARM Pipeline"])

# ============================================================================
# Request/Response Models
# ============================================================================


class ARMExecuteRequest(BaseModel):
    """Request to execute ARM pipeline."""

    image: str = Field(..., description="Base64-encoded image (JPEG/PNG)")
    instruction: str = Field(..., max_length=500, description="Natural language task instruction")
    robot_id: str = Field(default="ur10e", description="Target robot identifier")
    depth_map: Optional[str] = Field(None, description="Optional base64-encoded depth map")
    include_visualization: bool = Field(default=True, description="Include trajectory visualization")


class UserGuidanceRequest(BaseModel):
    """Request to apply user guidance to trajectory."""

    result_id: str = Field(..., description="ARM result ID to modify")
    guidance_type: str = Field(..., description="Type: add_waypoint, delete_waypoint, move_waypoint, avoid_region")
    data: Dict[str, Any] = Field(..., description="Guidance-specific data")


class TrajectoryTraceResponse(BaseModel):
    """Trajectory trace in response."""

    waypoints: List[List[float]]
    confidences: List[float]
    mean_confidence: float
    source_image_shape: List[int]
    instruction: str


class DecodedActionsResponse(BaseModel):
    """Decoded actions in response."""

    joint_actions: List[List[float]]
    action_horizon: int
    success_rate: float


class ReasoningResponse(BaseModel):
    """Chain-of-thought reasoning in response."""

    perception_reasoning: str
    spatial_reasoning: str
    action_reasoning: str
    perception_confidence: float
    spatial_confidence: float
    action_confidence: float


class TimingResponse(BaseModel):
    """Pipeline timing metrics."""

    total_ms: float
    depth_ms: float
    trajectory_ms: float
    decoding_ms: float
    reasoning_ms: float


class ARMExecuteResponse(BaseModel):
    """Response from ARM pipeline execution."""

    result_id: str
    trajectory_trace: TrajectoryTraceResponse
    decoded_actions: DecodedActionsResponse
    reasoning: ReasoningResponse
    timing: TimingResponse
    robot_id: str
    user_guidance_applied: bool = False
    visualization_url: Optional[str] = None


class ARMStatsResponse(BaseModel):
    """ARM pipeline statistics."""

    total_executions: int
    successful_executions: int
    avg_trajectory_confidence: float
    avg_ik_success_rate: float
    avg_total_time_ms: float
    supported_robots: List[str]


class RobotInfo(BaseModel):
    """Robot information."""

    id: str
    name: str
    dof: int
    status: str
    gripper_type: str


# ============================================================================
# In-Memory Cache for Results
# ============================================================================

# Simple in-memory cache for demo purposes
# In production, use Redis or similar
_result_cache: Dict[str, Dict[str, Any]] = {}
_stats = {
    "total_executions": 0,
    "successful_executions": 0,
    "avg_trajectory_confidence": 0.0,
    "avg_ik_success_rate": 0.0,
    "avg_total_time_ms": 0.0,
}

# Supported robots
SUPPORTED_ROBOTS = [
    {"id": "ur10e", "name": "UR10e", "dof": 6, "status": "available", "gripper_type": "parallel"},
    {"id": "ur5e", "name": "UR5e", "dof": 6, "status": "available", "gripper_type": "parallel"},
    {"id": "franka", "name": "Franka Emika", "dof": 7, "status": "available", "gripper_type": "parallel"},
    {"id": "custom", "name": "Custom Robot", "dof": 7, "status": "available", "gripper_type": "vacuum"},
]


# ============================================================================
# ARM Pipeline Integration
# ============================================================================

def _execute_arm_pipeline(
    image_data: bytes,
    instruction: str,
    robot_id: str,
    depth_data: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Execute the ARM pipeline.

    In production, this imports and calls the actual ARM module.
    For now, provides a realistic mock implementation.
    """
    start_time = time.time()

    # Simulate pipeline stages
    depth_start = time.time()
    # Stage 1: Depth tokenization (if depth provided)
    time.sleep(0.02)  # Simulate processing
    depth_time = (time.time() - depth_start) * 1000

    trajectory_start = time.time()
    # Stage 2: Trajectory prediction
    # Generate realistic trajectory
    n_waypoints = 8
    waypoints = []
    base_x, base_y = 120, 180
    for i in range(n_waypoints):
        x = base_x + i * 50 + np.random.normal(0, 5)
        y = base_y + np.sin(i * 0.5) * 30 + np.random.normal(0, 3)
        waypoints.append([float(x), float(y)])

    confidences = [float(0.95 - i * 0.03 + np.random.normal(0, 0.02)) for i in range(n_waypoints)]
    confidences = [max(0.5, min(1.0, c)) for c in confidences]
    mean_confidence = float(np.mean(confidences))
    time.sleep(0.08)  # Simulate processing
    trajectory_time = (time.time() - trajectory_start) * 1000

    decoding_start = time.time()
    # Stage 3: Action decoding
    # Generate mock joint actions
    joint_actions = []
    for i in range(n_waypoints):
        if robot_id in ["ur10e", "ur5e"]:
            dof = 6
        else:
            dof = 7
        action = [float(np.random.uniform(-np.pi, np.pi)) for _ in range(dof)]
        joint_actions.append(action)

    success_rate = float(np.random.uniform(0.82, 0.95))
    time.sleep(0.05)  # Simulate processing
    decoding_time = (time.time() - decoding_start) * 1000

    reasoning_start = time.time()
    # Stage 4: Reasoning generation
    reasoning = {
        "perception_reasoning": f"Detected target object for task '{instruction}'. Object appears graspable with current gripper configuration. Scene contains {np.random.randint(2, 5)} potential obstacles.",
        "spatial_reasoning": f"Target is approximately {np.random.uniform(0.3, 0.6):.2f}m from robot base. Approach vector computed from above to minimize collision risk. Clear path identified through {n_waypoints} waypoints.",
        "action_reasoning": f"Executing {n_waypoints}-waypoint trajectory for '{instruction}'. Gripper pre-shaped for grasp. Motion will complete in approximately {n_waypoints * 0.5:.1f} seconds.",
        "perception_confidence": float(np.random.uniform(0.85, 0.95)),
        "spatial_confidence": float(np.random.uniform(0.80, 0.92)),
        "action_confidence": float(np.random.uniform(0.78, 0.90)),
    }
    time.sleep(0.03)  # Simulate processing
    reasoning_time = (time.time() - reasoning_start) * 1000

    total_time = (time.time() - start_time) * 1000

    return {
        "trajectory_trace": {
            "waypoints": waypoints,
            "confidences": confidences,
            "mean_confidence": mean_confidence,
            "source_image_shape": [480, 640, 3],  # Assume standard resolution
            "instruction": instruction,
        },
        "decoded_actions": {
            "joint_actions": joint_actions,
            "action_horizon": n_waypoints,
            "success_rate": success_rate,
        },
        "reasoning": reasoning,
        "timing": {
            "total_ms": total_time,
            "depth_ms": depth_time,
            "trajectory_ms": trajectory_time,
            "decoding_ms": decoding_time,
            "reasoning_ms": reasoning_time,
        },
    }


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/robots", response_model=List[RobotInfo])
async def list_robots():
    """
    List all supported robots for ARM pipeline.

    Returns robot configurations including DOF, gripper type, and availability.
    """
    return SUPPORTED_ROBOTS


@router.post("/execute", response_model=ARMExecuteResponse, dependencies=[Depends(get_api_key)])
async def execute_arm_pipeline(request: ARMExecuteRequest, background_tasks: BackgroundTasks):
    """
    Execute the ARM (Action Reasoning Model) pipeline.

    Takes a camera image and natural language instruction, returns:
    - Predicted trajectory with confidence scores
    - Decoded joint actions for target robot
    - Chain-of-thought reasoning
    - Timing metrics

    The trajectory can be visualized and optionally modified via the /steer endpoint.
    """
    global _stats

    # Validate robot
    if request.robot_id not in [r["id"] for r in SUPPORTED_ROBOTS]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported robot: {request.robot_id}. Supported: {[r['id'] for r in SUPPORTED_ROBOTS]}"
        )

    # Decode image
    try:
        image_data = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image encoding: {e}")

    # Validate image size
    if len(image_data) > 4 * 1024 * 1024:  # 4MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 4MB)")

    # Decode depth map if provided
    depth_data = None
    if request.depth_map:
        try:
            depth_data = base64.b64decode(request.depth_map)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid depth map encoding: {e}")

    # Execute pipeline
    try:
        result = _execute_arm_pipeline(
            image_data=image_data,
            instruction=request.instruction,
            robot_id=request.robot_id,
            depth_data=depth_data,
        )
    except Exception as e:
        logger.error(f"ARM pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

    # Generate result ID
    result_id = f"arm_{uuid.uuid4().hex[:12]}"

    # Cache result
    _result_cache[result_id] = {
        **result,
        "robot_id": request.robot_id,
        "created_at": time.time(),
    }

    # Update statistics
    _stats["total_executions"] += 1
    _stats["successful_executions"] += 1
    alpha = 0.1
    _stats["avg_trajectory_confidence"] = (
        alpha * result["trajectory_trace"]["mean_confidence"] +
        (1 - alpha) * _stats["avg_trajectory_confidence"]
    )
    _stats["avg_ik_success_rate"] = (
        alpha * result["decoded_actions"]["success_rate"] +
        (1 - alpha) * _stats["avg_ik_success_rate"]
    )
    _stats["avg_total_time_ms"] = (
        alpha * result["timing"]["total_ms"] +
        (1 - alpha) * _stats["avg_total_time_ms"]
    )

    # Build response
    response = ARMExecuteResponse(
        result_id=result_id,
        trajectory_trace=TrajectoryTraceResponse(**result["trajectory_trace"]),
        decoded_actions=DecodedActionsResponse(**result["decoded_actions"]),
        reasoning=ReasoningResponse(**result["reasoning"]),
        timing=TimingResponse(**result["timing"]),
        robot_id=request.robot_id,
        user_guidance_applied=False,
        visualization_url=f"/api/v1/arm/visualization/{result_id}" if request.include_visualization else None,
    )

    return response


@router.get("/result/{result_id}", response_model=ARMExecuteResponse, dependencies=[Depends(get_api_key)])
async def get_arm_result(result_id: str):
    """
    Retrieve a cached ARM result by ID.

    Results are cached for 1 hour after creation.
    """
    if result_id not in _result_cache:
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")

    cached = _result_cache[result_id]

    # Check expiration (1 hour)
    if time.time() - cached.get("created_at", 0) > 3600:
        del _result_cache[result_id]
        raise HTTPException(status_code=404, detail=f"Result expired: {result_id}")

    return ARMExecuteResponse(
        result_id=result_id,
        trajectory_trace=TrajectoryTraceResponse(**cached["trajectory_trace"]),
        decoded_actions=DecodedActionsResponse(**cached["decoded_actions"]),
        reasoning=ReasoningResponse(**cached["reasoning"]),
        timing=TimingResponse(**cached["timing"]),
        robot_id=cached["robot_id"],
        user_guidance_applied=cached.get("user_guidance_applied", False),
        visualization_url=f"/api/v1/arm/visualization/{result_id}",
    )


@router.post("/steer", response_model=ARMExecuteResponse, dependencies=[Depends(get_api_key)])
async def apply_user_guidance(request: UserGuidanceRequest):
    """
    Apply user guidance to modify a trajectory (steerability).

    Supported guidance types:
    - add_waypoint: Add a new waypoint at specified position
    - delete_waypoint: Remove waypoint at specified index
    - move_waypoint: Move existing waypoint to new position
    - avoid_region: Mark region to avoid (re-plans trajectory)
    """
    if request.result_id not in _result_cache:
        raise HTTPException(status_code=404, detail=f"Result not found: {request.result_id}")

    cached = _result_cache[request.result_id]
    trace = cached["trajectory_trace"]

    if request.guidance_type == "add_waypoint":
        # Add new waypoint
        if "x" not in request.data or "y" not in request.data:
            raise HTTPException(status_code=400, detail="add_waypoint requires 'x' and 'y' in data")

        new_wp = [float(request.data["x"]), float(request.data["y"])]
        index = request.data.get("index", len(trace["waypoints"]))

        trace["waypoints"].insert(index, new_wp)
        trace["confidences"].insert(index, 0.7)  # User-added waypoints get moderate confidence
        trace["mean_confidence"] = float(np.mean(trace["confidences"]))

    elif request.guidance_type == "delete_waypoint":
        # Delete waypoint
        if "index" not in request.data:
            raise HTTPException(status_code=400, detail="delete_waypoint requires 'index' in data")

        index = int(request.data["index"])
        if index < 0 or index >= len(trace["waypoints"]):
            raise HTTPException(status_code=400, detail=f"Invalid waypoint index: {index}")

        del trace["waypoints"][index]
        del trace["confidences"][index]
        trace["mean_confidence"] = float(np.mean(trace["confidences"])) if trace["confidences"] else 0.0

    elif request.guidance_type == "move_waypoint":
        # Move existing waypoint
        if "index" not in request.data or "x" not in request.data or "y" not in request.data:
            raise HTTPException(status_code=400, detail="move_waypoint requires 'index', 'x', 'y' in data")

        index = int(request.data["index"])
        if index < 0 or index >= len(trace["waypoints"]):
            raise HTTPException(status_code=400, detail=f"Invalid waypoint index: {index}")

        trace["waypoints"][index] = [float(request.data["x"]), float(request.data["y"])]
        trace["confidences"][index] = 0.75  # User-modified waypoints
        trace["mean_confidence"] = float(np.mean(trace["confidences"]))

    elif request.guidance_type == "avoid_region":
        # Mark avoid region - would trigger re-planning
        # For now, just acknowledge
        logger.info(f"Avoid region requested: {request.data}")
        # In production, this would re-run trajectory prediction with constraints

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown guidance type: {request.guidance_type}"
        )

    # Update cache
    cached["trajectory_trace"] = trace
    cached["user_guidance_applied"] = True
    cached["decoded_actions"]["action_horizon"] = len(trace["waypoints"])

    return ARMExecuteResponse(
        result_id=request.result_id,
        trajectory_trace=TrajectoryTraceResponse(**trace),
        decoded_actions=DecodedActionsResponse(**cached["decoded_actions"]),
        reasoning=ReasoningResponse(**cached["reasoning"]),
        timing=TimingResponse(**cached["timing"]),
        robot_id=cached["robot_id"],
        user_guidance_applied=True,
        visualization_url=f"/api/v1/arm/visualization/{request.result_id}",
    )


@router.get("/visualization/{result_id}")
async def get_visualization(result_id: str):
    """
    Get trajectory visualization image.

    Returns a PNG image with the trajectory overlaid on the source image.
    For now, returns a placeholder. In production, would render actual visualization.
    """
    if result_id not in _result_cache:
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")

    # In production, this would:
    # 1. Load the source image from cache
    # 2. Render trajectory overlay using TrajectoryTrace.visualize()
    # 3. Return as image/png

    # For now, return info about the visualization
    cached = _result_cache[result_id]
    return {
        "result_id": result_id,
        "waypoints": len(cached["trajectory_trace"]["waypoints"]),
        "mean_confidence": cached["trajectory_trace"]["mean_confidence"],
        "message": "Visualization would be rendered here. See UI for interactive view.",
    }


@router.get("/stats", response_model=ARMStatsResponse, dependencies=[Depends(get_api_key)])
async def get_arm_stats():
    """
    Get ARM pipeline statistics.

    Returns aggregate metrics across all executions.
    """
    return ARMStatsResponse(
        total_executions=_stats["total_executions"],
        successful_executions=_stats["successful_executions"],
        avg_trajectory_confidence=_stats["avg_trajectory_confidence"],
        avg_ik_success_rate=_stats["avg_ik_success_rate"],
        avg_total_time_ms=_stats["avg_total_time_ms"],
        supported_robots=[r["id"] for r in SUPPORTED_ROBOTS],
    )


@router.post("/execute-cross-robot", dependencies=[Depends(get_api_key)])
async def execute_cross_robot(
    result_id: str,
    target_robot_id: str,
):
    """
    Execute a cached trajectory on a different robot (cross-robot transfer).

    Takes an existing ARM result and adapts it for a different robot embodiment.
    """
    if result_id not in _result_cache:
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")

    if target_robot_id not in [r["id"] for r in SUPPORTED_ROBOTS]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target robot: {target_robot_id}"
        )

    cached = _result_cache[result_id]
    source_robot = cached["robot_id"]

    # In production, this would:
    # 1. Load trajectory trace
    # 2. Create ActionDecoder for target robot
    # 3. Re-decode actions
    # 4. Return adapted result

    # For demo, simulate adaptation
    target_dof = 7 if target_robot_id in ["franka", "custom"] else 6
    adapted_actions = [
        [float(np.random.uniform(-np.pi, np.pi)) for _ in range(target_dof)]
        for _ in range(len(cached["trajectory_trace"]["waypoints"]))
    ]

    # Simulate compatibility score
    compatibility_scores = {
        ("ur10e", "ur5e"): 0.92,
        ("ur10e", "franka"): 0.78,
        ("ur10e", "custom"): 0.65,
        ("ur5e", "ur10e"): 0.90,
        ("ur5e", "franka"): 0.75,
        ("ur5e", "custom"): 0.60,
        ("franka", "ur10e"): 0.72,
        ("franka", "ur5e"): 0.70,
        ("franka", "custom"): 0.55,
    }

    compat = compatibility_scores.get((source_robot, target_robot_id), 0.7)

    return {
        "source_robot": source_robot,
        "target_robot": target_robot_id,
        "compatibility_score": compat,
        "adapted_action_horizon": len(adapted_actions),
        "adapted_ik_success_rate": float(np.random.uniform(0.7, 0.9)),
        "message": f"Successfully adapted from {source_robot} to {target_robot_id}",
    }

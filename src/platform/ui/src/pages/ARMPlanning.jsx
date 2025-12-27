/**
 * ARM Planning Page - Action Reasoning Model Interface
 *
 * Enterprise interface for MolmoAct-inspired spatial reasoning:
 * - Image upload with instruction input
 * - Trajectory visualization with confidence heatmaps
 * - User steerability controls (edit waypoints, avoid regions)
 * - Action reasoning display (chain-of-thought)
 * - Cross-robot execution preview
 *
 * @version 1.0.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Target,
  Upload,
  Play,
  Pause,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Move,
  Pencil,
  MousePointer,
  Circle,
  Square,
  Trash2,
  Download,
  Share2,
  ChevronRight,
  ChevronDown,
  Clock,
  Cpu,
  Activity,
  Brain,
  Eye,
  Layers,
  Bot,
  RefreshCw,
  Check,
  AlertCircle,
  Info,
  Maximize2,
  Settings,
  Image as ImageIcon,
} from 'lucide-react';
import { clsx } from 'clsx';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Badge,
  Tabs,
  Tab,
  ProgressBar,
  Alert,
  DataRow,
  Divider,
  KPICard,
  StatusIndicator,
  EmptyState,
} from '../design-system/components';
import { fetchWithAuth } from '../api';

// Mock data for demonstration
const MOCK_ROBOTS = [
  { id: 'ur10e', name: 'UR10e', dof: 6, status: 'online' },
  { id: 'ur5e', name: 'UR5e', dof: 6, status: 'online' },
  { id: 'franka', name: 'Franka Emika', dof: 7, status: 'offline' },
  { id: 'custom', name: 'Custom Robot', dof: 7, status: 'online' },
];

// Trajectory Visualizer Component
function TrajectoryVisualizer({
  imageUrl,
  trajectory,
  editMode,
  onWaypointClick,
  onCanvasClick,
  selectedWaypoint,
}) {
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [hoveredWaypoint, setHoveredWaypoint] = useState(null);

  // Draw trajectory on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trajectory) return;

    const ctx = canvas.getContext('2d');
    const img = new window.Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw trajectory line
      if (trajectory.waypoints && trajectory.waypoints.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        trajectory.waypoints.forEach((wp, i) => {
          if (i === 0) {
            ctx.moveTo(wp[0], wp[1]);
          } else {
            ctx.lineTo(wp[0], wp[1]);
          }
        });
        ctx.stroke();

        // Draw waypoints
        trajectory.waypoints.forEach((wp, i) => {
          const confidence = trajectory.confidences?.[i] || 0.8;
          const isSelected = selectedWaypoint === i;
          const isHovered = hoveredWaypoint === i;

          // Confidence-based color
          let color;
          if (confidence >= 0.85) color = '#22c55e';
          else if (confidence >= 0.7) color = '#eab308';
          else if (confidence >= 0.5) color = '#f97316';
          else color = '#ef4444';

          // Draw waypoint circle
          ctx.beginPath();
          ctx.arc(wp[0], wp[1], isSelected || isHovered ? 12 : 8, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.5)';
          ctx.lineWidth = isSelected ? 3 : 2;
          ctx.stroke();

          // Draw waypoint number
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 10px Inter';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(i + 1), wp[0], wp[1]);
        });

        // Draw start/end indicators
        if (trajectory.waypoints.length > 0) {
          const start = trajectory.waypoints[0];
          const end = trajectory.waypoints[trajectory.waypoints.length - 1];

          // Start indicator
          ctx.beginPath();
          ctx.arc(start[0], start[1], 16, 0, 2 * Math.PI);
          ctx.strokeStyle = '#22c55e';
          ctx.lineWidth = 2;
          ctx.stroke();

          // End indicator
          ctx.beginPath();
          ctx.arc(end[0], end[1], 16, 0, 2 * Math.PI);
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    };

    img.src = imageUrl || 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect fill="%231e293b" width="100%" height="100%"/><text x="50%" y="50%" fill="%2364748b" text-anchor="middle" dy=".3em" font-family="Inter">No Image</text></svg>';
  }, [imageUrl, trajectory, selectedWaypoint, hoveredWaypoint]);

  const handleCanvasClick = (e) => {
    if (!editMode) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Check if clicked on existing waypoint
    if (trajectory?.waypoints) {
      for (let i = 0; i < trajectory.waypoints.length; i++) {
        const wp = trajectory.waypoints[i];
        const dist = Math.sqrt((x - wp[0]) ** 2 + (y - wp[1]) ** 2);
        if (dist < 15) {
          onWaypointClick?.(i);
          return;
        }
      }
    }

    // Clicked on empty space - add waypoint
    onCanvasClick?.({ x, y });
  };

  return (
    <div className="relative bg-gray-900 rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-gray-900/80 backdrop-blur rounded-lg p-1">
        <button
          onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
          className="p-1.5 hover:bg-gray-700 rounded"
        >
          <ZoomIn size={16} className="text-gray-400" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
          className="p-1.5 hover:bg-gray-700 rounded"
        >
          <ZoomOut size={16} className="text-gray-400" />
        </button>
        <div className="w-px h-4 bg-gray-700" />
        <span className="text-xs text-gray-400 px-1">{(zoom * 100).toFixed(0)}%</span>
      </div>

      {/* Edit mode indicator */}
      {editMode && (
        <div className="absolute top-3 right-3 z-10">
          <Badge variant="warning" dot>Edit Mode</Badge>
        </div>
      )}

      {/* Canvas */}
      <div
        className="overflow-auto"
        style={{ maxHeight: '500px' }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className={clsx(
            'max-w-full transition-transform',
            editMode && 'cursor-crosshair'
          )}
          style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}
        />
      </div>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex items-center gap-4 text-xs bg-gray-900/80 backdrop-blur rounded-lg px-3 py-2">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="text-gray-400">High Conf</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span className="text-gray-400">Medium</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-400">Low</span>
        </div>
      </div>
    </div>
  );
}

// Reasoning Panel Component
function ReasoningPanel({ reasoning, expanded, onToggle }) {
  if (!reasoning) return null;

  const sections = [
    {
      key: 'perception',
      label: 'Perception Reasoning',
      icon: Eye,
      content: reasoning.perception_reasoning,
      confidence: reasoning.perception_confidence,
    },
    {
      key: 'spatial',
      label: 'Spatial Reasoning',
      icon: Layers,
      content: reasoning.spatial_reasoning,
      confidence: reasoning.spatial_confidence,
    },
    {
      key: 'action',
      label: 'Action Reasoning',
      icon: Target,
      content: reasoning.action_reasoning,
      confidence: reasoning.action_confidence,
    },
  ];

  return (
    <Card variant="elevated">
      <CardHeader
        action={
          <button onClick={onToggle} className="text-gray-400 hover:text-white">
            {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>
        }
      >
        <CardTitle icon={Brain} iconColor="text-purple-400">
          Chain-of-Thought Reasoning
        </CardTitle>
      </CardHeader>

      {expanded && (
        <CardContent>
          <div className="space-y-4">
            {sections.map((section) => (
              <div key={section.key} className="p-3 bg-gray-900/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <section.icon size={14} className="text-gray-400" />
                    <span className="text-sm font-medium text-white">{section.label}</span>
                  </div>
                  <Badge
                    variant={section.confidence >= 0.8 ? 'success' : section.confidence >= 0.6 ? 'warning' : 'danger'}
                    size="sm"
                  >
                    {(section.confidence * 100).toFixed(0)}%
                  </Badge>
                </div>
                <p className="text-sm text-gray-400 leading-relaxed">
                  {section.content || 'No reasoning available'}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

// Timing Metrics Component
function TimingMetrics({ timing }) {
  if (!timing) return null;

  const metrics = [
    { label: 'Total', value: timing.total_ms, color: 'text-blue-400' },
    { label: 'Depth', value: timing.depth_ms, color: 'text-purple-400' },
    { label: 'Trajectory', value: timing.trajectory_ms, color: 'text-green-400' },
    { label: 'Decoding', value: timing.decoding_ms, color: 'text-amber-400' },
    { label: 'Reasoning', value: timing.reasoning_ms, color: 'text-cyan-400' },
  ];

  return (
    <Card variant="default" padding="sm">
      <div className="flex items-center gap-2 mb-3">
        <Clock size={14} className="text-gray-400" />
        <span className="text-sm font-medium text-white">Timing</span>
      </div>
      <div className="grid grid-cols-5 gap-2">
        {metrics.map((m) => (
          <div key={m.label} className="text-center">
            <div className={clsx('text-lg font-bold', m.color)}>
              {m.value?.toFixed(0) || '-'}
            </div>
            <div className="text-xs text-gray-500">{m.label}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// Robot Selector Component
function RobotSelector({ robots, selected, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {robots.map((robot) => (
        <button
          key={robot.id}
          onClick={() => onChange(robot.id)}
          className={clsx(
            'p-3 rounded-lg border transition-all text-left',
            selected === robot.id
              ? 'bg-blue-600/20 border-blue-500/50 text-blue-400'
              : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
          )}
        >
          <div className="flex items-center justify-between mb-1">
            <span className="font-medium text-white">{robot.name}</span>
            <StatusIndicator status={robot.status} showLabel={false} size="sm" />
          </div>
          <div className="text-xs text-gray-500">{robot.dof} DOF</div>
        </button>
      ))}
    </div>
  );
}

// Main ARM Planning Page
export default function ARMPlanning() {
  // State
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [instruction, setInstruction] = useState('');
  const [selectedRobot, setSelectedRobot] = useState('ur10e');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [selectedWaypoint, setSelectedWaypoint] = useState(null);
  const [reasoningExpanded, setReasoningExpanded] = useState(true);
  const [activeTab, setActiveTab] = useState('trajectory');

  const fileInputRef = useRef(null);

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setImageUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  // Execute ARM pipeline
  const handleExecute = async () => {
    if (!image || !instruction.trim()) {
      setError('Please upload an image and enter an instruction');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = reader.result.split(',')[1];

        // Mock API call - in production, this would call /api/v1/arm/execute
        // Simulating ARM pipeline execution
        await new Promise((resolve) => setTimeout(resolve, 1500));

        // Mock result
        const mockResult = {
          result_id: `arm_${Date.now()}`,
          trajectory_trace: {
            waypoints: [
              [120, 180],
              [160, 200],
              [220, 220],
              [280, 200],
              [340, 180],
              [400, 160],
              [450, 150],
              [500, 140],
            ],
            confidences: [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.72],
            mean_confidence: 0.83,
          },
          decoded_actions: {
            joint_actions: [],
            action_horizon: 8,
            success_rate: 0.87,
          },
          reasoning: {
            perception_reasoning: `Analyzing the scene, I detect a ${instruction.includes('cup') ? 'cup' : 'target object'} in the center-right region of the image. The object appears to be graspable with the current gripper configuration.`,
            spatial_reasoning: 'The target object is approximately 0.45m from the robot base. The approach vector should be from above to avoid collision with nearby objects. Clear path identified with no obstacles in the planned trajectory.',
            action_reasoning: `To execute "${instruction}", I will: 1) Move to pre-grasp position above the target, 2) Descend to grasp height, 3) Close gripper, 4) Lift object, 5) Move to destination. Total 8 waypoints planned.`,
            perception_confidence: 0.91,
            spatial_confidence: 0.85,
            action_confidence: 0.82,
          },
          timing: {
            total_ms: 342,
            depth_ms: 45,
            trajectory_ms: 156,
            decoding_ms: 89,
            reasoning_ms: 52,
          },
        };

        setResult(mockResult);
        setLoading(false);
      };
      reader.readAsDataURL(image);
    } catch (err) {
      setError('Failed to execute ARM pipeline');
      setLoading(false);
    }
  };

  // Handle waypoint editing
  const handleWaypointClick = (index) => {
    if (editMode) {
      setSelectedWaypoint(index);
    }
  };

  const handleCanvasClick = ({ x, y }) => {
    if (editMode && result) {
      // Add new waypoint
      const newWaypoints = [...result.trajectory_trace.waypoints, [x, y]];
      const newConfidences = [...result.trajectory_trace.confidences, 0.7];

      setResult({
        ...result,
        trajectory_trace: {
          ...result.trajectory_trace,
          waypoints: newWaypoints,
          confidences: newConfidences,
        },
      });
    }
  };

  const handleDeleteWaypoint = () => {
    if (selectedWaypoint !== null && result) {
      const newWaypoints = result.trajectory_trace.waypoints.filter((_, i) => i !== selectedWaypoint);
      const newConfidences = result.trajectory_trace.confidences.filter((_, i) => i !== selectedWaypoint);

      setResult({
        ...result,
        trajectory_trace: {
          ...result.trajectory_trace,
          waypoints: newWaypoints,
          confidences: newConfidences,
        },
      });
      setSelectedWaypoint(null);
    }
  };

  // Export result
  const handleExport = () => {
    if (!result) return;

    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `arm_result_${result.result_id}.json`;
    a.click();
  };

  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <Target className="text-blue-400" />
            ARM Planning
          </h1>
          <p className="text-gray-400 mt-1">
            Action Reasoning Model for interpretable robot manipulation
          </p>
        </div>
        <div className="flex items-center gap-2">
          {result && (
            <>
              <Button variant="ghost" size="sm" icon={Download} onClick={handleExport}>
                Export
              </Button>
              <Button variant="ghost" size="sm" icon={Share2}>
                Share
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="error" dismissible onDismiss={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Input */}
        <div className="col-span-4 space-y-4">
          {/* Image Upload */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle icon={ImageIcon} iconColor="text-cyan-400">
                Input Image
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                onClick={() => fileInputRef.current?.click()}
                className={clsx(
                  'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
                  imageUrl
                    ? 'border-blue-500/50 bg-blue-500/5'
                    : 'border-gray-700 hover:border-gray-600'
                )}
              >
                {imageUrl ? (
                  <img
                    src={imageUrl}
                    alt="Uploaded"
                    className="max-h-40 mx-auto rounded"
                  />
                ) : (
                  <div className="py-4">
                    <Upload className="w-8 h-8 mx-auto text-gray-500 mb-2" />
                    <p className="text-sm text-gray-400">
                      Click to upload image
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      PNG, JPG up to 4MB
                    </p>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </CardContent>
          </Card>

          {/* Instruction Input */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle icon={Brain} iconColor="text-purple-400">
                Instruction
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                placeholder="Describe the task (e.g., 'Pick up the red cup and place it on the shelf')"
                className="w-full h-24 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none resize-none"
              />
            </CardContent>
          </Card>

          {/* Robot Selection */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle icon={Bot} iconColor="text-green-400">
                Target Robot
              </CardTitle>
            </CardHeader>
            <CardContent>
              <RobotSelector
                robots={MOCK_ROBOTS}
                selected={selectedRobot}
                onChange={setSelectedRobot}
              />
            </CardContent>
          </Card>

          {/* Execute Button */}
          <Button
            variant="primary"
            size="lg"
            icon={loading ? RefreshCw : Play}
            onClick={handleExecute}
            disabled={!image || !instruction.trim() || loading}
            className="w-full"
          >
            {loading ? 'Executing...' : 'Execute ARM Pipeline'}
          </Button>
        </div>

        {/* Right Panel - Results */}
        <div className="col-span-8 space-y-4">
          {!result ? (
            <Card variant="elevated" className="h-full min-h-[600px]">
              <EmptyState
                icon={Target}
                title="No trajectory generated"
                description="Upload an image, enter an instruction, and execute the ARM pipeline to see the trajectory visualization."
              />
            </Card>
          ) : (
            <>
              {/* Tabs */}
              <Tabs value={activeTab} onChange={setActiveTab}>
                <Tab value="trajectory" icon={Target}>Trajectory</Tab>
                <Tab value="actions" icon={Activity}>Actions</Tab>
                <Tab value="compare" icon={Bot}>Compare Robots</Tab>
              </Tabs>

              {/* Trajectory Tab */}
              {activeTab === 'trajectory' && (
                <div className="space-y-4">
                  {/* Toolbar */}
                  <Card variant="default" padding="sm">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Button
                          variant={editMode ? 'warning' : 'secondary'}
                          size="sm"
                          icon={editMode ? Check : Pencil}
                          onClick={() => {
                            setEditMode(!editMode);
                            setSelectedWaypoint(null);
                          }}
                        >
                          {editMode ? 'Done' : 'Edit'}
                        </Button>
                        {editMode && selectedWaypoint !== null && (
                          <Button
                            variant="danger"
                            size="sm"
                            icon={Trash2}
                            onClick={handleDeleteWaypoint}
                          >
                            Delete
                          </Button>
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-gray-400">Waypoints:</span>
                          <span className="text-white font-medium">
                            {result.trajectory_trace.waypoints.length}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-gray-400">Confidence:</span>
                          <Badge
                            variant={
                              result.trajectory_trace.mean_confidence >= 0.8
                                ? 'success'
                                : result.trajectory_trace.mean_confidence >= 0.6
                                ? 'warning'
                                : 'danger'
                            }
                          >
                            {(result.trajectory_trace.mean_confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </Card>

                  {/* Trajectory Visualization */}
                  <TrajectoryVisualizer
                    imageUrl={imageUrl}
                    trajectory={result.trajectory_trace}
                    editMode={editMode}
                    selectedWaypoint={selectedWaypoint}
                    onWaypointClick={handleWaypointClick}
                    onCanvasClick={handleCanvasClick}
                  />

                  {/* Timing Metrics */}
                  <TimingMetrics timing={result.timing} />

                  {/* Reasoning Panel */}
                  <ReasoningPanel
                    reasoning={result.reasoning}
                    expanded={reasoningExpanded}
                    onToggle={() => setReasoningExpanded(!reasoningExpanded)}
                  />
                </div>
              )}

              {/* Actions Tab */}
              {activeTab === 'actions' && (
                <Card variant="elevated">
                  <CardHeader>
                    <CardTitle icon={Activity} iconColor="text-green-400">
                      Decoded Actions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <KPICard
                        title="Action Horizon"
                        value={result.decoded_actions.action_horizon}
                        subvalue="steps"
                        icon={Target}
                        iconColor="text-blue-400"
                        iconBg="bg-blue-500/10"
                      />
                      <KPICard
                        title="IK Success Rate"
                        value={`${(result.decoded_actions.success_rate * 100).toFixed(0)}%`}
                        subvalue="valid configs"
                        icon={Check}
                        iconColor="text-green-400"
                        iconBg="bg-green-500/10"
                      />
                      <KPICard
                        title="Robot"
                        value={selectedRobot.toUpperCase()}
                        subvalue={MOCK_ROBOTS.find((r) => r.id === selectedRobot)?.dof + ' DOF'}
                        icon={Bot}
                        iconColor="text-purple-400"
                        iconBg="bg-purple-500/10"
                      />
                    </div>
                    <Alert variant="info">
                      Joint actions ready for execution. Connect to robot controller to deploy.
                    </Alert>
                  </CardContent>
                </Card>
              )}

              {/* Compare Robots Tab */}
              {activeTab === 'compare' && (
                <Card variant="elevated">
                  <CardHeader>
                    <CardTitle icon={Bot} iconColor="text-purple-400">
                      Cross-Robot Comparison
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-gray-700">
                            <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Robot</th>
                            <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                            <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">DOF</th>
                            <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">IK Success</th>
                            <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Compatibility</th>
                          </tr>
                        </thead>
                        <tbody>
                          {MOCK_ROBOTS.map((robot) => (
                            <tr key={robot.id} className="border-b border-gray-800">
                              <td className="py-3 px-4">
                                <span className="font-medium text-white">{robot.name}</span>
                              </td>
                              <td className="py-3 px-4">
                                <StatusIndicator status={robot.status} />
                              </td>
                              <td className="py-3 px-4 text-gray-300">{robot.dof}</td>
                              <td className="py-3 px-4">
                                <Badge variant={robot.status === 'online' ? 'success' : 'default'}>
                                  {robot.status === 'online' ? '87%' : 'N/A'}
                                </Badge>
                              </td>
                              <td className="py-3 px-4">
                                <ProgressBar
                                  value={robot.status === 'online' ? 85 : 0}
                                  variant={robot.status === 'online' ? 'success' : 'default'}
                                  size="sm"
                                />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

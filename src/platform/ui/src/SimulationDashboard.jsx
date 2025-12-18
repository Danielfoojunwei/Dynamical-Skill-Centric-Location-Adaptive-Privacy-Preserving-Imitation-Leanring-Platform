/**
 * Simulation Dashboard Component
 *
 * Real-time visualization dashboard for Isaac Lab simulation including:
 * - 3D robot visualization (using SVG for lightweight rendering)
 * - Camera feeds
 * - Task progress
 * - Federated learning metrics
 * - Safety status
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, RadarChart,
    PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
    Play, Pause, RotateCcw, Camera, Activity, Cpu, Shield, CloudLightning,
    Target, Sliders, Eye, Layers, AlertTriangle, CheckCircle, XCircle
} from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import { fetchWithAuth } from './api';

const COLORS = {
    primary: '#3b82f6',
    success: '#22c55e',
    warning: '#eab308',
    danger: '#ef4444',
    purple: '#8b5cf6',
    cyan: '#06b6d4',
    gray: '#6b7280',
};

// ============= Robot 3D Visualization =============

const Robot3DView = ({ jointPositions = [], eePosition = [], gripperState = 1.0 }) => {
    // Simple 2.5D robot arm visualization using SVG
    // This provides a lightweight alternative to full 3D rendering

    const joints = jointPositions.length > 0 ? jointPositions : [0, -0.785, 0, -2.356, 0, 1.571, 0.785];

    // Convert joint angles to link positions (simplified kinematics)
    const linkLengths = [0, 0.333, 0.316, 0, 0.384, 0, 0.107];
    const points = useMemo(() => {
        let x = 200, y = 350; // Base position
        let angle = -Math.PI / 2; // Start pointing up

        const pts = [{ x, y, joint: 0 }];

        for (let i = 0; i < Math.min(7, joints.length); i++) {
            angle += joints[i] * 0.3; // Scale down for visualization
            const len = (linkLengths[i] || 0.3) * 200;
            x += Math.cos(angle) * len;
            y += Math.sin(angle) * len;
            pts.push({ x, y, joint: i + 1, angle });
        }

        return pts;
    }, [joints]);

    // End effector position indicator
    const eePos = eePosition.length >= 3 ? eePosition : [0.5, 0, 0.5];

    return (
        <div className="robot-3d-view" style={{ background: '#0f172a', borderRadius: '12px', padding: '1rem' }}>
            <svg width="100%" height="300" viewBox="0 0 400 400">
                {/* Grid */}
                <defs>
                    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1e293b" strokeWidth="0.5" />
                    </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />

                {/* Robot base */}
                <rect x="180" y="340" width="40" height="20" fill="#475569" rx="4" />
                <circle cx="200" cy="350" r="15" fill="#64748b" />

                {/* Robot links */}
                {points.slice(0, -1).map((pt, i) => (
                    <g key={i}>
                        <line
                            x1={pt.x}
                            y1={pt.y}
                            x2={points[i + 1].x}
                            y2={points[i + 1].y}
                            stroke={i < 4 ? '#3b82f6' : '#8b5cf6'}
                            strokeWidth="8"
                            strokeLinecap="round"
                        />
                    </g>
                ))}

                {/* Joints */}
                {points.map((pt, i) => (
                    <circle
                        key={`joint-${i}`}
                        cx={pt.x}
                        cy={pt.y}
                        r={i === 0 ? 12 : 8}
                        fill="#1e293b"
                        stroke="#64748b"
                        strokeWidth="2"
                    />
                ))}

                {/* End effector */}
                <g transform={`translate(${points[points.length - 1]?.x || 200}, ${points[points.length - 1]?.y || 100})`}>
                    {/* Gripper */}
                    <rect
                        x={-15 + gripperState * 5}
                        y={-5}
                        width="8"
                        height="25"
                        fill="#22c55e"
                        rx="2"
                    />
                    <rect
                        x={7 - gripperState * 5}
                        y={-5}
                        width="8"
                        height="25"
                        fill="#22c55e"
                        rx="2"
                    />
                </g>

                {/* EE Position label */}
                <text x="10" y="20" fill="#94a3b8" fontSize="10">
                    EE: [{eePos[0].toFixed(3)}, {eePos[1].toFixed(3)}, {eePos[2].toFixed(3)}]
                </text>

                {/* Gripper state */}
                <text x="10" y="35" fill="#94a3b8" fontSize="10">
                    Gripper: {(gripperState * 100).toFixed(0)}% open
                </text>
            </svg>
        </div>
    );
};

// ============= Camera Feed Component =============

const CameraFeed = ({ cameraId, frame, isLive = true }) => {
    return (
        <div className="camera-feed" style={{
            background: '#1e293b',
            borderRadius: '8px',
            padding: '0.5rem',
            position: 'relative'
        }}>
            <div style={{
                position: 'absolute',
                top: '0.75rem',
                left: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                background: 'rgba(0,0,0,0.7)',
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                fontSize: '0.75rem'
            }}>
                <Camera size={12} />
                {cameraId}
                {isLive && <span style={{ color: '#22c55e' }}>LIVE</span>}
            </div>

            {frame?.rgb_b64 ? (
                <img
                    src={`data:image/jpeg;base64,${frame.rgb_b64}`}
                    alt={cameraId}
                    style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '4px' }}
                />
            ) : (
                <div style={{
                    width: '100%',
                    height: '150px',
                    background: '#0f172a',
                    borderRadius: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#64748b'
                }}>
                    <Eye size={24} />
                </div>
            )}
        </div>
    );
};

// ============= Task Progress Component =============

const TaskProgress = ({ taskState }) => {
    if (!taskState) {
        return (
            <div className="card">
                <h3><Target size={18} /> Task Progress</h3>
                <p style={{ color: '#64748b' }}>No active task</p>
            </div>
        );
    }

    const { task_type, phase, step, reward, success, objects } = taskState;

    const phases = ['approach', 'descend', 'grasp', 'lift', 'transport', 'lower', 'release', 'done'];
    const currentPhaseIndex = phases.indexOf(phase?.toLowerCase() || '');

    return (
        <div className="card">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Target size={18} /> Task: {task_type || 'Unknown'}
            </h3>

            {/* Phase progress bar */}
            <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.875rem' }}>Phase: {phase}</span>
                    <span style={{ fontSize: '0.875rem' }}>Step: {step}</span>
                </div>
                <div style={{
                    display: 'flex',
                    gap: '2px',
                    height: '8px'
                }}>
                    {phases.map((p, i) => (
                        <div
                            key={p}
                            style={{
                                flex: 1,
                                borderRadius: '2px',
                                background: i <= currentPhaseIndex
                                    ? (success ? COLORS.success : COLORS.primary)
                                    : '#1e293b'
                            }}
                        />
                    ))}
                </div>
            </div>

            {/* Reward */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Reward</span>
                <span style={{
                    color: reward >= 0 ? COLORS.success : COLORS.danger,
                    fontWeight: 'bold'
                }}>
                    {(reward || 0).toFixed(3)}
                </span>
            </div>

            {/* Success indicator */}
            {success && (
                <div style={{
                    marginTop: '1rem',
                    padding: '0.5rem',
                    background: '#22c55e20',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    color: COLORS.success
                }}>
                    <CheckCircle size={18} /> Task Completed!
                </div>
            )}
        </div>
    );
};

// ============= Federated Learning Metrics =============

const FederatedLearningPanel = ({ flState }) => {
    const [lossHistory, setLossHistory] = useState([]);

    useEffect(() => {
        if (flState?.global_loss !== undefined) {
            setLossHistory(prev => {
                const newHistory = [...prev, {
                    round: flState.round || prev.length,
                    loss: flState.global_loss,
                    timestamp: Date.now()
                }].slice(-50); // Keep last 50 points
                return newHistory;
            });
        }
    }, [flState?.round]);

    if (!flState) {
        return (
            <div className="card">
                <h3><CloudLightning size={18} /> Federated Learning</h3>
                <p style={{ color: '#64748b' }}>Not active</p>
            </div>
        );
    }

    return (
        <div className="card">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <CloudLightning size={18} /> Federated Learning
            </h3>

            <div className="grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
                <div>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Round</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{flState.round || 0}</div>
                </div>
                <div>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Clients</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{flState.num_clients || 0}</div>
                </div>
                <div>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Global Loss</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: COLORS.primary }}>
                        {(flState.global_loss || 0).toFixed(4)}
                    </div>
                </div>
                <div>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Convergence</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: COLORS.success }}>
                        {((flState.convergence_rate || 0) * 100).toFixed(1)}%
                    </div>
                </div>
            </div>

            {/* Loss chart */}
            {lossHistory.length > 1 && (
                <div style={{ height: '120px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={lossHistory}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="round" stroke="#64748b" fontSize={10} />
                            <YAxis stroke="#64748b" fontSize={10} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px' }}
                                labelStyle={{ color: '#94a3b8' }}
                            />
                            <Area
                                type="monotone"
                                dataKey="loss"
                                stroke={COLORS.primary}
                                fill={`${COLORS.primary}40`}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
};

// ============= Performance Metrics Panel =============

const PerformanceMetrics = ({ metrics }) => {
    if (!metrics) {
        return (
            <div className="card">
                <h3><Activity size={18} /> Performance</h3>
                <p style={{ color: '#64748b' }}>No data</p>
            </div>
        );
    }

    const {
        physics_fps = 0,
        render_fps = 0,
        step_time_ms = 0,
        real_time_factor = 0,
        total_steps = 0,
        simulation_time = 0
    } = metrics;

    return (
        <div className="card">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Activity size={18} /> Simulation Performance
            </h3>

            <div className="grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
                <div style={{ textAlign: 'center', padding: '0.5rem', background: '#1e293b', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: COLORS.success }}>
                        {physics_fps.toFixed(0)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Physics FPS</div>
                </div>
                <div style={{ textAlign: 'center', padding: '0.5rem', background: '#1e293b', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: COLORS.primary }}>
                        {render_fps.toFixed(0)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Render FPS</div>
                </div>
                <div style={{ textAlign: 'center', padding: '0.5rem', background: '#1e293b', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: COLORS.cyan }}>
                        {real_time_factor.toFixed(2)}x
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Real-time</div>
                </div>
            </div>

            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                <span style={{ color: '#64748b' }}>Step time: {step_time_ms.toFixed(1)}ms</span>
                <span style={{ color: '#64748b' }}>Steps: {total_steps}</span>
            </div>
            <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                <span style={{ color: '#64748b' }}>Sim time: {simulation_time.toFixed(1)}s</span>
            </div>
        </div>
    );
};

// ============= Safety Status Panel =============

const SafetyStatusPanel = ({ safetyState }) => {
    const levels = {
        'SAFE': { color: COLORS.success, icon: CheckCircle },
        'WARN': { color: COLORS.warning, icon: AlertTriangle },
        'SLOW': { color: COLORS.warning, icon: Sliders },
        'CRAWL': { color: COLORS.danger, icon: AlertTriangle },
        'STOP': { color: COLORS.danger, icon: XCircle },
    };

    const level = safetyState?.level || 'SAFE';
    const { color, icon: Icon } = levels[level] || levels['SAFE'];

    return (
        <div className="card">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Shield size={18} /> Safety Status
            </h3>

            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                padding: '1rem',
                background: `${color}20`,
                borderRadius: '8px',
                marginBottom: '1rem'
            }}>
                <Icon size={32} color={color} />
                <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color }}>{level}</div>
                    <div style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
                        {safetyState?.hazards?.length || 0} active hazards
                    </div>
                </div>
            </div>

            {safetyState?.zone_violations?.length > 0 && (
                <div style={{ fontSize: '0.875rem', color: COLORS.warning }}>
                    Zone violations: {safetyState.zone_violations.join(', ')}
                </div>
            )}
        </div>
    );
};

// ============= Main Simulation Dashboard =============

const SimulationDashboard = () => {
    const [simulationRunning, setSimulationRunning] = useState(false);
    const [config, setConfig] = useState({
        robot_type: 'franka_panda',
        scene_type: 'tabletop',
        task_type: 'pick_place',
    });

    // WebSocket connection
    const wsUrl = `ws://${window.location.hostname}:8000/ws/simulation`;
    const {
        isConnected,
        robotState,
        taskState,
        metrics,
        flState,
        safetyState,
        cameraFrames,
        sendCommand,
    } = useWebSocket(wsUrl, ['all']);

    // Start simulation
    const handleStart = async () => {
        try {
            const res = await fetchWithAuth('/ws/simulation/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            const data = await res.json();
            if (data.status === 'started') {
                setSimulationRunning(true);
            }
        } catch (e) {
            console.error('Failed to start simulation:', e);
        }
    };

    // Stop simulation
    const handleStop = async () => {
        try {
            await fetchWithAuth('/ws/simulation/stop', { method: 'POST' });
            setSimulationRunning(false);
        } catch (e) {
            console.error('Failed to stop simulation:', e);
        }
    };

    // Reset simulation
    const handleReset = async () => {
        try {
            await fetchWithAuth('/ws/simulation/reset', { method: 'POST' });
        } catch (e) {
            console.error('Failed to reset simulation:', e);
        }
    };

    return (
        <div className="simulation-dashboard">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold">Isaac Lab Simulation</h1>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                        <div style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: isConnected ? COLORS.success : COLORS.danger
                        }} />
                        <span style={{ fontSize: '0.875rem', color: '#64748b' }}>
                            {isConnected ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    {!simulationRunning ? (
                        <button
                            onClick={handleStart}
                            className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded flex items-center gap-2 transition"
                        >
                            <Play size={20} /> Start Simulation
                        </button>
                    ) : (
                        <>
                            <button
                                onClick={handleReset}
                                className="bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded flex items-center gap-2 transition"
                            >
                                <RotateCcw size={18} /> Reset
                            </button>
                            <button
                                onClick={handleStop}
                                className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded flex items-center gap-2 transition"
                            >
                                <Pause size={20} /> Stop
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* Main Grid */}
            <div className="grid" style={{ gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
                {/* Left Column */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    {/* Robot Visualization */}
                    <div className="card">
                        <h3 style={{ marginBottom: '1rem' }}>
                            <Layers size={18} style={{ display: 'inline', marginRight: '0.5rem' }} />
                            Robot State
                        </h3>
                        <Robot3DView
                            jointPositions={robotState?.joint_positions}
                            eePosition={robotState?.ee_position}
                            gripperState={robotState?.gripper_state}
                        />
                    </div>

                    {/* Camera Feeds */}
                    <div className="card">
                        <h3 style={{ marginBottom: '1rem' }}>
                            <Camera size={18} style={{ display: 'inline', marginRight: '0.5rem' }} />
                            Camera Feeds
                        </h3>
                        <div className="grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                            {['front', 'left', 'right', 'wrist'].map(camId => (
                                <CameraFeed
                                    key={camId}
                                    cameraId={camId}
                                    frame={cameraFrames[camId]}
                                    isLive={simulationRunning}
                                />
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    {/* Task Progress */}
                    <TaskProgress taskState={taskState} />

                    {/* Performance Metrics */}
                    <PerformanceMetrics metrics={metrics} />

                    {/* Safety Status */}
                    <SafetyStatusPanel safetyState={safetyState} />

                    {/* Federated Learning */}
                    <FederatedLearningPanel flState={flState} />
                </div>
            </div>
        </div>
    );
};

export default SimulationDashboard;

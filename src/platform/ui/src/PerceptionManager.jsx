import React, { useState, useEffect, useRef } from 'react';
import { Brain, Eye, Layers, Shield, Settings, Play, Pause, Square, RefreshCw, AlertTriangle, CheckCircle, Lock, Unlock, Cpu, Activity, Camera, Video, Target, Plus, Trash2, Circle, MousePointer, Type, Crosshair, Film, Download, Upload } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { fetchWithAuth } from './api';

// =============================================================================
// TAB 1: Camera & Segmentation (SAM 3)
// =============================================================================
const CameraSegmentationTab = () => {
    const [cameras, setCameras] = useState([
        { id: 'cam_front', name: 'Front Camera', status: 'online' },
        { id: 'cam_left', name: 'Left Camera', status: 'online' },
        { id: 'cam_right', name: 'Right Camera', status: 'online' },
        { id: 'cam_overhead', name: 'Overhead Camera', status: 'online' },
    ]);
    const [selectedCamera, setSelectedCamera] = useState('cam_front');
    const [segmentMode, setSegmentMode] = useState('click'); // 'click', 'box', 'text'
    const [textPrompt, setTextPrompt] = useState('');
    const [segments, setSegments] = useState([]);
    const [watchlist, setWatchlist] = useState([
        { id: 1, label: 'Red Cup', color: '#ef4444', tracking: true },
        { id: 2, label: 'Human Hand', color: '#3b82f6', tracking: true },
    ]);
    const [isSegmenting, setIsSegmenting] = useState(false);
    const canvasRef = useRef(null);

    const handleCanvasClick = async (e) => {
        if (segmentMode !== 'click') return;

        const rect = e.target.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;

        setIsSegmenting(true);

        // Simulate SAM segmentation
        setTimeout(() => {
            const newSegment = {
                id: Date.now(),
                type: 'click',
                points: [{ x, y }],
                mask: `polygon(${x-10}% ${y-10}%, ${x+10}% ${y-10}%, ${x+10}% ${y+10}%, ${x-10}% ${y+10}%)`,
                color: `hsl(${Math.random() * 360}, 70%, 50%)`,
                label: `Object ${segments.length + 1}`
            };
            setSegments([...segments, newSegment]);
            setIsSegmenting(false);
        }, 500);
    };

    const handleTextSegment = async () => {
        if (!textPrompt.trim()) return;

        setIsSegmenting(true);

        // Simulate text-based SAM segmentation
        setTimeout(() => {
            const newSegment = {
                id: Date.now(),
                type: 'text',
                prompt: textPrompt,
                mask: `polygon(30% 30%, 70% 30%, 70% 70%, 30% 70%)`,
                color: `hsl(${Math.random() * 360}, 70%, 50%)`,
                label: textPrompt
            };
            setSegments([...segments, newSegment]);
            setTextPrompt('');
            setIsSegmenting(false);
        }, 800);
    };

    const addToWatchlist = (segment) => {
        setWatchlist([...watchlist, {
            id: Date.now(),
            label: segment.label,
            color: segment.color,
            tracking: true
        }]);
    };

    const removeFromWatchlist = (id) => {
        setWatchlist(watchlist.filter(w => w.id !== id));
    };

    const clearSegments = () => setSegments([]);

    return (
        <div className="grid grid-cols-4 gap-4 h-full">
            {/* Left: Camera Feed with Segmentation */}
            <div className="col-span-3 space-y-4">
                {/* Camera Selector */}
                <div className="flex gap-2">
                    {cameras.map(cam => (
                        <button
                            key={cam.id}
                            onClick={() => setSelectedCamera(cam.id)}
                            className={`px-3 py-2 rounded flex items-center gap-2 ${
                                selectedCamera === cam.id ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                            }`}
                        >
                            <Camera size={16} />
                            {cam.name}
                            <div className={`w-2 h-2 rounded-full ${cam.status === 'online' ? 'bg-green-500' : 'bg-red-500'}`} />
                        </button>
                    ))}
                </div>

                {/* Segmentation Mode Selector */}
                <div className="flex gap-2 items-center">
                    <span className="text-gray-400 text-sm">Segment by:</span>
                    <button
                        onClick={() => setSegmentMode('click')}
                        className={`px-3 py-1.5 rounded flex items-center gap-1 text-sm ${
                            segmentMode === 'click' ? 'bg-purple-600' : 'bg-gray-700'
                        }`}
                    >
                        <MousePointer size={14} /> Click
                    </button>
                    <button
                        onClick={() => setSegmentMode('box')}
                        className={`px-3 py-1.5 rounded flex items-center gap-1 text-sm ${
                            segmentMode === 'box' ? 'bg-purple-600' : 'bg-gray-700'
                        }`}
                    >
                        <Square size={14} /> Box
                    </button>
                    <button
                        onClick={() => setSegmentMode('text')}
                        className={`px-3 py-1.5 rounded flex items-center gap-1 text-sm ${
                            segmentMode === 'text' ? 'bg-purple-600' : 'bg-gray-700'
                        }`}
                    >
                        <Type size={14} /> Text Prompt
                    </button>

                    {segmentMode === 'text' && (
                        <div className="flex gap-2 ml-4">
                            <input
                                type="text"
                                value={textPrompt}
                                onChange={e => setTextPrompt(e.target.value)}
                                placeholder="e.g., 'red cup', 'human hand', 'robot gripper'"
                                className="bg-gray-800 border border-gray-600 rounded px-3 py-1.5 w-64 text-sm"
                                onKeyPress={e => e.key === 'Enter' && handleTextSegment()}
                            />
                            <button
                                onClick={handleTextSegment}
                                disabled={isSegmenting}
                                className="bg-purple-600 hover:bg-purple-700 px-3 py-1.5 rounded text-sm"
                            >
                                {isSegmenting ? 'Segmenting...' : 'Segment'}
                            </button>
                        </div>
                    )}

                    <button
                        onClick={clearSegments}
                        className="ml-auto bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-sm flex items-center gap-1"
                    >
                        <Trash2 size={14} /> Clear All
                    </button>
                </div>

                {/* Camera View with Overlays */}
                <div
                    className="relative bg-gray-900 rounded-lg aspect-video cursor-crosshair overflow-hidden"
                    onClick={handleCanvasClick}
                    ref={canvasRef}
                >
                    {/* Simulated camera feed */}
                    <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
                        <div className="text-center">
                            <Video size={48} className="mx-auto mb-2 text-gray-600" />
                            <p className="text-gray-500">Camera: {selectedCamera}</p>
                            <p className="text-gray-600 text-sm">Click to segment objects</p>
                        </div>
                    </div>

                    {/* Segmentation Overlays */}
                    {segments.map(seg => (
                        <div
                            key={seg.id}
                            className="absolute border-2 rounded"
                            style={{
                                left: `${seg.points?.[0]?.x - 10 || 30}%`,
                                top: `${seg.points?.[0]?.y - 10 || 30}%`,
                                width: '20%',
                                height: '20%',
                                borderColor: seg.color,
                                backgroundColor: `${seg.color}20`
                            }}
                        >
                            <div
                                className="absolute -top-6 left-0 px-2 py-0.5 rounded text-xs font-medium"
                                style={{ backgroundColor: seg.color }}
                            >
                                {seg.label}
                            </div>
                            <button
                                onClick={(e) => { e.stopPropagation(); addToWatchlist(seg); }}
                                className="absolute -top-6 right-0 p-1 bg-green-600 rounded hover:bg-green-700"
                                title="Add to watchlist"
                            >
                                <Plus size={12} />
                            </button>
                        </div>
                    ))}

                    {/* Loading indicator */}
                    {isSegmenting && (
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                            <div className="flex items-center gap-2 bg-purple-600 px-4 py-2 rounded">
                                <RefreshCw className="animate-spin" size={16} />
                                SAM 3 Segmenting...
                            </div>
                        </div>
                    )}
                </div>

                {/* Segment List */}
                {segments.length > 0 && (
                    <div className="bg-gray-800 p-3 rounded-lg">
                        <h4 className="text-sm font-medium mb-2">Detected Segments</h4>
                        <div className="flex flex-wrap gap-2">
                            {segments.map(seg => (
                                <div
                                    key={seg.id}
                                    className="flex items-center gap-2 bg-gray-700 px-2 py-1 rounded text-sm"
                                >
                                    <div className="w-3 h-3 rounded" style={{ backgroundColor: seg.color }} />
                                    {seg.label}
                                    <button
                                        onClick={() => addToWatchlist(seg)}
                                        className="text-green-400 hover:text-green-300"
                                        title="Track this"
                                    >
                                        <Target size={14} />
                                    </button>
                                    <button
                                        onClick={() => setSegments(segments.filter(s => s.id !== seg.id))}
                                        className="text-red-400 hover:text-red-300"
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Right: Watchlist Panel */}
            <div className="space-y-4">
                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Target className="text-green-400" /> Watch For
                    </h3>
                    <p className="text-xs text-gray-400 mb-3">
                        Objects being tracked across all cameras. Alerts when detected.
                    </p>

                    <div className="space-y-2">
                        {watchlist.map(item => (
                            <div
                                key={item.id}
                                className="flex items-center gap-2 bg-gray-700 p-2 rounded"
                            >
                                <div
                                    className="w-4 h-4 rounded"
                                    style={{ backgroundColor: item.color }}
                                />
                                <span className="flex-1 text-sm">{item.label}</span>
                                <button
                                    onClick={() => {
                                        setWatchlist(watchlist.map(w =>
                                            w.id === item.id ? {...w, tracking: !w.tracking} : w
                                        ));
                                    }}
                                    className={`p-1 rounded ${item.tracking ? 'bg-green-600' : 'bg-gray-600'}`}
                                    title={item.tracking ? 'Tracking' : 'Paused'}
                                >
                                    {item.tracking ? <Eye size={12} /> : <Pause size={12} />}
                                </button>
                                <button
                                    onClick={() => removeFromWatchlist(item.id)}
                                    className="p-1 rounded bg-red-600/50 hover:bg-red-600"
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        ))}
                    </div>

                    <button className="w-full mt-3 bg-gray-700 hover:bg-gray-600 py-2 rounded text-sm flex items-center justify-center gap-2">
                        <Plus size={14} /> Add Custom Watch Item
                    </button>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Layers className="text-purple-400" /> SAM 3 Status
                    </h3>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-400">Model</span>
                            <span className="text-green-400">SAM3-Large</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">TFLOPS</span>
                            <span>15.0</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Latency</span>
                            <span>25ms</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Tracking</span>
                            <span className="text-green-400">Enabled</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// =============================================================================
// TAB 2: Imitation Learning (DINOv3 + V-JEPA 2)
// =============================================================================
const ImitationLearningTab = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [demonstrations, setDemonstrations] = useState([
        { id: 1, name: 'Pick up cup', frames: 120, duration: '4.0s', features: 1024, status: 'ready' },
        { id: 2, name: 'Pour water', frames: 180, duration: '6.0s', features: 1536, status: 'ready' },
        { id: 3, name: 'Place on table', frames: 90, duration: '3.0s', features: 768, status: 'processing' },
    ]);
    const [selectedDemo, setSelectedDemo] = useState(null);
    const [predictionHorizon, setPredictionHorizon] = useState(16);
    const [showPrediction, setShowPrediction] = useState(false);

    useEffect(() => {
        let interval;
        if (isRecording) {
            interval = setInterval(() => {
                setRecordingTime(t => t + 0.1);
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isRecording]);

    const startRecording = () => {
        setIsRecording(true);
        setRecordingTime(0);
    };

    const stopRecording = () => {
        setIsRecording(false);
        const newDemo = {
            id: Date.now(),
            name: `Demo ${demonstrations.length + 1}`,
            frames: Math.floor(recordingTime * 30),
            duration: `${recordingTime.toFixed(1)}s`,
            features: Math.floor(recordingTime * 256),
            status: 'processing'
        };
        setDemonstrations([...demonstrations, newDemo]);

        // Simulate processing
        setTimeout(() => {
            setDemonstrations(demos =>
                demos.map(d => d.id === newDemo.id ? {...d, status: 'ready'} : d)
            );
        }, 3000);
    };

    return (
        <div className="grid grid-cols-3 gap-4 h-full">
            {/* Left: Recording & Preview */}
            <div className="col-span-2 space-y-4">
                {/* Recording Controls */}
                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-4 flex items-center gap-2">
                        <Film className="text-blue-400" /> Record Demonstration
                    </h3>

                    <div className="grid grid-cols-2 gap-4">
                        {/* Camera preview during recording */}
                        <div className="bg-gray-900 rounded-lg aspect-video relative">
                            <div className="absolute inset-0 flex items-center justify-center">
                                <Video size={32} className="text-gray-600" />
                            </div>
                            {isRecording && (
                                <div className="absolute top-2 left-2 flex items-center gap-2 bg-red-600 px-2 py-1 rounded text-sm">
                                    <Circle size={8} className="animate-pulse fill-current" />
                                    REC {recordingTime.toFixed(1)}s
                                </div>
                            )}
                            <div className="absolute bottom-2 left-2 text-xs text-gray-400">
                                DINOv3 extracting features...
                            </div>
                        </div>

                        {/* V-JEPA prediction preview */}
                        <div className="bg-gray-900 rounded-lg aspect-video relative">
                            <div className="absolute inset-0 flex items-center justify-center">
                                {showPrediction ? (
                                    <div className="text-center">
                                        <Brain size={32} className="text-green-400 mx-auto mb-2" />
                                        <p className="text-sm text-gray-400">V-JEPA 2 Prediction</p>
                                        <p className="text-xs text-gray-500">+{predictionHorizon} frames</p>
                                    </div>
                                ) : (
                                    <p className="text-gray-600 text-sm">Prediction preview</p>
                                )}
                            </div>
                            <div className="absolute top-2 right-2">
                                <button
                                    onClick={() => setShowPrediction(!showPrediction)}
                                    className={`px-2 py-1 rounded text-xs ${showPrediction ? 'bg-green-600' : 'bg-gray-700'}`}
                                >
                                    {showPrediction ? 'Hide' : 'Show'} Prediction
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-4 mt-4">
                        {!isRecording ? (
                            <button
                                onClick={startRecording}
                                className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded flex items-center gap-2"
                            >
                                <Circle size={16} className="fill-current" /> Start Recording
                            </button>
                        ) : (
                            <button
                                onClick={stopRecording}
                                className="bg-gray-600 hover:bg-gray-500 px-6 py-2 rounded flex items-center gap-2"
                            >
                                <Square size={16} /> Stop Recording
                            </button>
                        )}

                        <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-400">Prediction Horizon:</span>
                            <input
                                type="range"
                                min={4}
                                max={32}
                                value={predictionHorizon}
                                onChange={e => setPredictionHorizon(parseInt(e.target.value))}
                                className="w-32"
                            />
                            <span className="text-sm w-16">{predictionHorizon} frames</span>
                        </div>
                    </div>
                </div>

                {/* Demonstration Playback */}
                {selectedDemo && (
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="font-semibold mb-4 flex items-center gap-2">
                            <Play className="text-green-400" /> Playback: {selectedDemo.name}
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-900 rounded-lg aspect-video relative">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <p className="text-gray-500">Original Recording</p>
                                </div>
                            </div>
                            <div className="bg-gray-900 rounded-lg aspect-video relative">
                                <div className="absolute inset-0 flex items-center justify-center flex-col">
                                    <Brain size={24} className="text-green-400 mb-2" />
                                    <p className="text-gray-500 text-sm">V-JEPA 2 Predicted Trajectory</p>
                                </div>
                            </div>
                        </div>
                        <div className="mt-4 flex items-center gap-2">
                            <button className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded flex items-center gap-2">
                                <Play size={16} /> Play
                            </button>
                            <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded flex items-center gap-2">
                                <Upload size={16} /> Train Skill
                            </button>
                            <div className="flex-1 bg-gray-700 rounded h-2">
                                <div className="bg-green-500 h-2 rounded" style={{width: '0%'}} />
                            </div>
                            <span className="text-sm text-gray-400">0:00 / {selectedDemo.duration}</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Right: Demonstrations List */}
            <div className="space-y-4">
                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Film className="text-blue-400" /> Demonstrations
                    </h3>

                    <div className="space-y-2">
                        {demonstrations.map(demo => (
                            <div
                                key={demo.id}
                                onClick={() => setSelectedDemo(demo)}
                                className={`p-3 rounded cursor-pointer transition ${
                                    selectedDemo?.id === demo.id
                                        ? 'bg-blue-600/30 border border-blue-500'
                                        : 'bg-gray-700 hover:bg-gray-600'
                                }`}
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className="font-medium">{demo.name}</span>
                                    <span className={`text-xs px-2 py-0.5 rounded ${
                                        demo.status === 'ready' ? 'bg-green-600' : 'bg-yellow-600'
                                    }`}>
                                        {demo.status}
                                    </span>
                                </div>
                                <div className="grid grid-cols-3 gap-2 text-xs text-gray-400">
                                    <span>{demo.frames} frames</span>
                                    <span>{demo.duration}</span>
                                    <span>{demo.features} features</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Eye className="text-blue-400" /> DINOv3 Features
                    </h3>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-400">Model</span>
                            <span>ViT-L/14</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Feature Dim</span>
                            <span>1024</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Patch Size</span>
                            <span>14x14</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">TFLOPS</span>
                            <span>8.0</span>
                        </div>
                    </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Brain className="text-green-400" /> V-JEPA 2 World Model
                    </h3>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-400">Model</span>
                            <span>V-JEPA2-Large</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Input Frames</span>
                            <span>16</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Prediction</span>
                            <span>{predictionHorizon} frames</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">TFLOPS</span>
                            <span>10.0</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// =============================================================================
// TAB 3: Safety Monitor (V-JEPA 2)
// =============================================================================
const SafetyMonitorTab = () => {
    const [predictions, setPredictions] = useState({
        collision_prob: 0.12,
        time_to_impact: -1,
        predicted_trajectory: [],
        hazards: []
    });

    useEffect(() => {
        const interval = setInterval(() => {
            // Simulate safety predictions updating
            setPredictions(p => ({
                ...p,
                collision_prob: Math.max(0, Math.min(1, p.collision_prob + (Math.random() - 0.5) * 0.1))
            }));
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    const collisionLevel = predictions.collision_prob > 0.7 ? 'danger' :
                          predictions.collision_prob > 0.3 ? 'warning' : 'safe';

    return (
        <div className="grid grid-cols-3 gap-4">
            {/* Left: Visualization */}
            <div className="col-span-2 space-y-4">
                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-4 flex items-center gap-2">
                        <Shield className="text-red-400" /> Collision Prediction Visualization
                    </h3>

                    {/* Main visualization area */}
                    <div className="bg-gray-900 rounded-lg aspect-video relative">
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-center">
                                <Brain size={48} className="mx-auto mb-2 text-gray-600" />
                                <p className="text-gray-500">V-JEPA 2 Trajectory Prediction</p>
                                <p className="text-gray-600 text-sm">Predicted path overlay on camera feed</p>
                            </div>
                        </div>

                        {/* Collision indicator overlay */}
                        <div className={`absolute top-4 right-4 px-4 py-2 rounded-lg ${
                            collisionLevel === 'danger' ? 'bg-red-600' :
                            collisionLevel === 'warning' ? 'bg-yellow-600' : 'bg-green-600'
                        }`}>
                            <div className="text-sm font-medium">Collision Risk</div>
                            <div className="text-2xl font-bold">{(predictions.collision_prob * 100).toFixed(0)}%</div>
                        </div>

                        {/* Trajectory legend */}
                        <div className="absolute bottom-4 left-4 bg-gray-800/80 p-2 rounded text-xs">
                            <div className="flex items-center gap-2 mb-1">
                                <div className="w-8 h-0.5 bg-green-500" />
                                <span>Predicted Safe Path</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-0.5 bg-red-500" />
                                <span>Collision Zone</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Timeline */}
                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3">Prediction Timeline (16 frames / 0.53s)</h3>
                    <div className="flex gap-1">
                        {Array.from({length: 16}).map((_, i) => {
                            const risk = Math.sin(i * 0.3) * 0.3 + 0.2;
                            return (
                                <div
                                    key={i}
                                    className="flex-1 rounded"
                                    style={{
                                        height: '60px',
                                        backgroundColor: risk > 0.4 ? '#ef4444' : risk > 0.2 ? '#eab308' : '#22c55e',
                                        opacity: 0.3 + risk
                                    }}
                                    title={`Frame ${i+1}: ${(risk * 100).toFixed(0)}% risk`}
                                />
                            );
                        })}
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Now</span>
                        <span>+0.53s</span>
                    </div>
                </div>
            </div>

            {/* Right: Safety Stats */}
            <div className="space-y-4">
                <div className={`p-4 rounded-lg border-2 ${
                    collisionLevel === 'danger' ? 'bg-red-900/30 border-red-500' :
                    collisionLevel === 'warning' ? 'bg-yellow-900/30 border-yellow-500' :
                    'bg-green-900/30 border-green-500'
                }`}>
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        {collisionLevel === 'danger' ? <AlertTriangle className="text-red-400" /> :
                         collisionLevel === 'warning' ? <AlertTriangle className="text-yellow-400" /> :
                         <CheckCircle className="text-green-400" />}
                        Safety Status
                    </h3>
                    <div className="text-3xl font-bold mb-2">
                        {collisionLevel === 'danger' ? 'DANGER' :
                         collisionLevel === 'warning' ? 'CAUTION' : 'SAFE'}
                    </div>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-400">Collision Prob</span>
                            <span className="font-mono">{(predictions.collision_prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Time to Impact</span>
                            <span className="font-mono">
                                {predictions.time_to_impact > 0 ? `${predictions.time_to_impact.toFixed(2)}s` : 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3">Thresholds</h3>
                    <div className="space-y-3">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-gray-400">Slow Down</span>
                                <span>30%</span>
                            </div>
                            <div className="h-2 bg-gray-700 rounded">
                                <div className="h-2 bg-yellow-500 rounded" style={{width: '30%'}} />
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-gray-400">Warning</span>
                                <span>70%</span>
                            </div>
                            <div className="h-2 bg-gray-700 rounded">
                                <div className="h-2 bg-orange-500 rounded" style={{width: '70%'}} />
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-gray-400">E-Stop</span>
                                <span>90%</span>
                            </div>
                            <div className="h-2 bg-gray-700 rounded">
                                <div className="h-2 bg-red-500 rounded" style={{width: '90%'}} />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Brain className="text-green-400" /> V-JEPA 2 Config
                    </h3>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-400">Status</span>
                            <span className="text-green-400">Active</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Horizon</span>
                            <span>16 frames</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Rate</span>
                            <span>1kHz (Safety Tier)</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">TFLOPS</span>
                            <span>10.0</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// =============================================================================
// TAB 4: Model Settings (Advanced)
// =============================================================================
const ModelSettingsTab = () => {
    const [models, setModels] = useState([
        { id: 'dinov3', name: 'DINOv3', enabled: true, tflops: 8.0 },
        { id: 'sam3', name: 'SAM 3', enabled: true, tflops: 15.0 },
        { id: 'vjepa2', name: 'V-JEPA 2', enabled: true, tflops: 10.0 },
    ]);
    const [privacy, setPrivacy] = useState({
        enabled: true,
        security_bits: 128,
        homomorphic_routing: true
    });

    const totalTflops = models.filter(m => m.enabled).reduce((sum, m) => sum + m.tflops, 0);

    return (
        <div className="grid grid-cols-2 gap-6">
            {/* Models */}
            <div className="space-y-4">
                <h3 className="text-lg font-semibold">Foundation Models</h3>
                {models.map(model => (
                    <div key={model.id} className="bg-gray-800 p-4 rounded-lg">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-3">
                                {model.id === 'dinov3' && <Eye className="text-blue-400" />}
                                {model.id === 'sam3' && <Layers className="text-purple-400" />}
                                {model.id === 'vjepa2' && <Brain className="text-green-400" />}
                                <span className="font-medium">{model.name}</span>
                            </div>
                            <button
                                onClick={() => setModels(models.map(m =>
                                    m.id === model.id ? {...m, enabled: !m.enabled} : m
                                ))}
                                className={`w-12 h-6 rounded-full transition ${
                                    model.enabled ? 'bg-green-600' : 'bg-gray-600'
                                }`}
                            >
                                <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                                    model.enabled ? 'translate-x-6' : 'translate-x-1'
                                }`} />
                            </button>
                        </div>
                        <div className="flex justify-between text-sm text-gray-400">
                            <span>TFLOPS: {model.tflops}</span>
                            <span className={model.enabled ? 'text-green-400' : 'text-gray-500'}>
                                {model.enabled ? 'Active' : 'Disabled'}
                            </span>
                        </div>
                    </div>
                ))}

                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="flex justify-between mb-2">
                        <span className="text-gray-400">Total TFLOPS</span>
                        <span className="font-bold">{totalTflops} / 33.0</span>
                    </div>
                    <div className="h-3 bg-gray-700 rounded">
                        <div
                            className="h-3 bg-blue-500 rounded transition-all"
                            style={{width: `${(totalTflops / 33) * 100}%`}}
                        />
                    </div>
                </div>
            </div>

            {/* Privacy */}
            <div className="space-y-4">
                <h3 className="text-lg font-semibold">Privacy Settings (N2HE)</h3>
                <div className="bg-gray-800 p-4 rounded-lg space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="font-medium">Encryption Enabled</div>
                            <div className="text-xs text-gray-400">128-bit N2HE homomorphic encryption</div>
                        </div>
                        <button
                            onClick={() => setPrivacy({...privacy, enabled: !privacy.enabled})}
                            className={`w-12 h-6 rounded-full transition ${
                                privacy.enabled ? 'bg-green-600' : 'bg-gray-600'
                            }`}
                        >
                            <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                                privacy.enabled ? 'translate-x-6' : 'translate-x-1'
                            }`} />
                        </button>
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <div className="font-medium">Homomorphic Routing</div>
                            <div className="text-xs text-gray-400">Process encrypted features in-place</div>
                        </div>
                        <button
                            onClick={() => setPrivacy({...privacy, homomorphic_routing: !privacy.homomorphic_routing})}
                            className={`w-12 h-6 rounded-full transition ${
                                privacy.homomorphic_routing ? 'bg-green-600' : 'bg-gray-600'
                            }`}
                        >
                            <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                                privacy.homomorphic_routing ? 'translate-x-6' : 'translate-x-1'
                            }`} />
                        </button>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gray-900 p-3 rounded">
                            <div className="text-xs text-gray-500">Security Bits</div>
                            <div className="text-xl font-bold text-green-400">{privacy.security_bits}</div>
                        </div>
                        <div className="bg-gray-900 p-3 rounded">
                            <div className="text-xs text-gray-500">LWE Dimension</div>
                            <div className="text-xl font-bold">1024</div>
                        </div>
                    </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg">
                    <h4 className="font-medium mb-3">Pipeline Rates</h4>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-400">Safety Tier</span>
                            <span>1000 Hz</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Control Tier</span>
                            <span>100 Hz</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Learning Tier</span>
                            <span>10 Hz</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-400">Cloud Sync</span>
                            <span>0.1 Hz</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================
const PerceptionManager = () => {
    const [activeTab, setActiveTab] = useState('segmentation');

    const tabs = [
        { id: 'segmentation', label: 'Camera & Segmentation', icon: Layers, description: 'SAM 3' },
        { id: 'learning', label: 'Imitation Learning', icon: Film, description: 'DINOv3 + V-JEPA' },
        { id: 'safety', label: 'Safety Monitor', icon: Shield, description: 'V-JEPA 2' },
        { id: 'settings', label: 'Model Settings', icon: Settings, description: 'Advanced' },
    ];

    return (
        <div className="p-6 h-full flex flex-col">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Brain className="text-purple-400" /> Meta AI Perception
                </h1>
            </div>

            {/* Tab Navigation */}
            <div className="flex gap-2 mb-6">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-3 rounded-lg flex items-center gap-2 transition ${
                            activeTab === tab.id
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-800 hover:bg-gray-700'
                        }`}
                    >
                        <tab.icon size={18} />
                        <div className="text-left">
                            <div className="font-medium">{tab.label}</div>
                            <div className="text-xs opacity-70">{tab.description}</div>
                        </div>
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 min-h-0">
                {activeTab === 'segmentation' && <CameraSegmentationTab />}
                {activeTab === 'learning' && <ImitationLearningTab />}
                {activeTab === 'safety' && <SafetyMonitorTab />}
                {activeTab === 'settings' && <ModelSettingsTab />}
            </div>
        </div>
    );
};

export default PerceptionManager;

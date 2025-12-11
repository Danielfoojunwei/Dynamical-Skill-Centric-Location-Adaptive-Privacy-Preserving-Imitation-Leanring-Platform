import React, { useState, useEffect } from 'react';
import { Brain, Eye, Layers, Shield, Settings, Play, Pause, RefreshCw, AlertTriangle, CheckCircle, Lock, Unlock, Cpu, Activity, Camera, Box, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { fetchWithAuth } from './api';

const COLORS = {
    dinov3: '#3b82f6',
    sam3: '#8b5cf6',
    vjepa2: '#22c55e',
    unused: '#1e293b'
};

// Model Card Component
const ModelCard = ({ model, onToggle, onConfigure }) => {
    const icons = {
        dinov3: Eye,
        sam3: Layers,
        vjepa2: Brain
    };
    const Icon = icons[model.id] || Brain;

    const statusColors = {
        running: 'bg-green-500',
        stopped: 'bg-gray-500',
        error: 'bg-red-500',
        loading: 'bg-yellow-500'
    };

    return (
        <div className={`bg-gray-800 rounded-lg border ${model.enabled ? 'border-blue-500' : 'border-gray-700'} overflow-hidden`}>
            <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                        <div className={`p-3 rounded-lg ${model.enabled ? 'bg-blue-600/20' : 'bg-gray-700'}`}>
                            <Icon size={24} className={model.enabled ? 'text-blue-400' : 'text-gray-500'} />
                        </div>
                        <div>
                            <h3 className="font-bold text-lg">{model.name}</h3>
                            <p className="text-xs text-gray-400">{model.version}</p>
                        </div>
                    </div>
                    <div className={`w-3 h-3 rounded-full ${statusColors[model.status]}`} />
                </div>

                <p className="text-sm text-gray-400 mb-4">{model.description}</p>

                <div className="grid grid-cols-2 gap-3 text-sm mb-4">
                    <div className="bg-gray-900 p-2 rounded">
                        <div className="text-gray-500 text-xs">TFLOPS</div>
                        <div className="font-bold" style={{ color: COLORS[model.id] }}>{model.tflops}</div>
                    </div>
                    <div className="bg-gray-900 p-2 rounded">
                        <div className="text-gray-500 text-xs">Latency</div>
                        <div className="font-bold">{model.latency_ms}ms</div>
                    </div>
                    <div className="bg-gray-900 p-2 rounded">
                        <div className="text-gray-500 text-xs">Model Size</div>
                        <div className="font-bold">{model.model_size}</div>
                    </div>
                    <div className="bg-gray-900 p-2 rounded">
                        <div className="text-gray-500 text-xs">Input</div>
                        <div className="font-bold">{model.input_size}px</div>
                    </div>
                </div>

                <div className="flex gap-2">
                    <button
                        onClick={() => onToggle(model.id)}
                        className={`flex-1 py-2 rounded flex items-center justify-center gap-2 transition ${
                            model.enabled
                                ? 'bg-red-600 hover:bg-red-700'
                                : 'bg-green-600 hover:bg-green-700'
                        }`}
                    >
                        {model.enabled ? <Pause size={16} /> : <Play size={16} />}
                        {model.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button
                        onClick={() => onConfigure(model)}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                    >
                        <Settings size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
};

// TFLOPS Allocation Chart
const TFLOPSChart = ({ models }) => {
    const data = models.map(m => ({
        name: m.name,
        value: m.enabled ? m.tflops : 0,
        fill: COLORS[m.id]
    }));

    const totalUsed = data.reduce((sum, d) => sum + d.value, 0);
    data.push({ name: 'Available', value: 33 - totalUsed, fill: COLORS.unused });

    return (
        <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Cpu className="text-blue-400" /> Meta AI TFLOPS Allocation
            </h3>
            <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="50%"
                            innerRadius={50}
                            outerRadius={70}
                            paddingAngle={3}
                            dataKey="value"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                    </PieChart>
                </ResponsiveContainer>
            </div>
            <div className="text-center text-sm text-gray-400 mt-2">
                {totalUsed.toFixed(1)} / 33.0 TFLOPS ({((totalUsed / 33) * 100).toFixed(1)}%)
            </div>
        </div>
    );
};

// Privacy Settings Panel
const PrivacyPanel = ({ settings, onUpdate }) => {
    return (
        <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Lock className="text-purple-400" /> Privacy Wrapper (N2HE)
            </h3>

            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <div>
                        <div className="font-medium">Encryption Enabled</div>
                        <div className="text-xs text-gray-400">128-bit N2HE homomorphic encryption</div>
                    </div>
                    <button
                        onClick={() => onUpdate({ ...settings, enabled: !settings.enabled })}
                        className={`w-12 h-6 rounded-full transition ${
                            settings.enabled ? 'bg-green-600' : 'bg-gray-600'
                        }`}
                    >
                        <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                            settings.enabled ? 'translate-x-6' : 'translate-x-1'
                        }`} />
                    </button>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-gray-500 text-xs">Security Bits</div>
                        <div className="font-bold text-green-400">{settings.security_bits}</div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-gray-500 text-xs">LWE Dimension</div>
                        <div className="font-bold">{settings.lwe_dimension}</div>
                    </div>
                </div>

                <div className="flex items-center justify-between">
                    <div>
                        <div className="font-medium">Homomorphic Routing</div>
                        <div className="text-xs text-gray-400">Process data without decryption</div>
                    </div>
                    <button
                        onClick={() => onUpdate({ ...settings, enable_homomorphic_routing: !settings.enable_homomorphic_routing })}
                        className={`w-12 h-6 rounded-full transition ${
                            settings.enable_homomorphic_routing ? 'bg-green-600' : 'bg-gray-600'
                        }`}
                    >
                        <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                            settings.enable_homomorphic_routing ? 'translate-x-6' : 'translate-x-1'
                        }`} />
                    </button>
                </div>
            </div>
        </div>
    );
};

// Unified Pipeline Status
const PipelineStatus = ({ status }) => {
    const tiers = [
        { name: 'Safety', rate: '1kHz', color: 'red', component: 'V-JEPA 2' },
        { name: 'Control', rate: '100Hz', color: 'yellow', component: 'DINOv3 + SAM3' },
        { name: 'Learning', rate: '10Hz', color: 'blue', component: 'All Models' },
        { name: 'Cloud', rate: '0.1Hz', color: 'purple', component: 'Encrypted Sync' },
    ];

    return (
        <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Activity className="text-green-400" /> Unified Pipeline Status
            </h3>

            <div className="space-y-3">
                {tiers.map((tier, i) => (
                    <div key={i} className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full bg-${tier.color}-500 ${
                            status.running ? 'animate-pulse' : ''
                        }`} />
                        <div className="flex-1">
                            <div className="flex justify-between">
                                <span className="font-medium">{tier.name} Tier</span>
                                <span className="text-gray-400 text-sm">{tier.rate}</span>
                            </div>
                            <div className="text-xs text-gray-500">{tier.component}</div>
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-4 p-3 bg-gray-900 rounded">
                <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Fusion Method</span>
                    <span className="font-mono">{status.fusion_method}</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                    <span className="text-gray-400">Pipeline State</span>
                    <span className={status.running ? 'text-green-400' : 'text-gray-400'}>
                        {status.running ? 'ACTIVE' : 'STOPPED'}
                    </span>
                </div>
            </div>
        </div>
    );
};

// Safety Prediction Panel (V-JEPA 2)
const SafetyPredictionPanel = ({ predictions }) => {
    return (
        <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Shield className="text-red-400" /> Safety Predictions (V-JEPA 2)
            </h3>

            <div className="space-y-3">
                <div className="bg-gray-900 p-3 rounded">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">Collision Probability</span>
                        <span className={`font-bold ${
                            predictions.collision_prob > 0.7 ? 'text-red-400' :
                            predictions.collision_prob > 0.3 ? 'text-yellow-400' : 'text-green-400'
                        }`}>
                            {(predictions.collision_prob * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                            className={`h-2 rounded-full transition-all ${
                                predictions.collision_prob > 0.7 ? 'bg-red-500' :
                                predictions.collision_prob > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${predictions.collision_prob * 100}%` }}
                        />
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-xs text-gray-500">Prediction Horizon</div>
                        <div className="font-bold">{predictions.horizon} frames</div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-xs text-gray-500">Time to Impact</div>
                        <div className="font-bold">
                            {predictions.time_to_impact > 0
                                ? `${predictions.time_to_impact.toFixed(2)}s`
                                : 'N/A'}
                        </div>
                    </div>
                </div>

                {predictions.collision_prob > 0.9 && (
                    <div className="flex items-center gap-2 p-3 bg-red-900/30 border border-red-500 rounded">
                        <AlertTriangle className="text-red-400" />
                        <span className="text-red-400 font-medium">Emergency Stop Threshold Exceeded</span>
                    </div>
                )}
            </div>
        </div>
    );
};

// Model Configuration Modal
const ConfigModal = ({ model, onClose, onSave }) => {
    const [config, setConfig] = useState(model?.config || {});

    if (!model) return null;

    const configFields = {
        dinov3: [
            { key: 'model_size', label: 'Model Size', type: 'select', options: ['vit_small', 'vit_base', 'vit_large', 'vit_giant'] },
            { key: 'input_size', label: 'Input Size', type: 'number' },
            { key: 'use_fp16', label: 'Use FP16', type: 'boolean' },
        ],
        sam3: [
            { key: 'model_size', label: 'Model Size', type: 'select', options: ['sam3_tiny', 'sam3_small', 'sam3_base', 'sam3_large'] },
            { key: 'input_size', label: 'Input Size', type: 'number' },
            { key: 'max_objects', label: 'Max Objects', type: 'number' },
            { key: 'confidence_threshold', label: 'Confidence Threshold', type: 'range', min: 0, max: 1, step: 0.1 },
            { key: 'enable_tracking', label: 'Enable Tracking', type: 'boolean' },
        ],
        vjepa2: [
            { key: 'model_size', label: 'Model Size', type: 'select', options: ['vjepa2_small', 'vjepa2_base', 'vjepa2_large', 'vjepa2_huge'] },
            { key: 'num_frames', label: 'Input Frames', type: 'number' },
            { key: 'prediction_horizon', label: 'Prediction Horizon', type: 'number' },
            { key: 'enable_safety_prediction', label: 'Safety Prediction', type: 'boolean' },
            { key: 'collision_threshold', label: 'Collision Threshold', type: 'range', min: 0, max: 1, step: 0.1 },
            { key: 'emergency_stop_threshold', label: 'E-Stop Threshold', type: 'range', min: 0, max: 1, step: 0.1 },
        ],
    };

    const fields = configFields[model.id] || [];

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-lg">
                <div className="p-4 border-b border-gray-700 flex justify-between items-center">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Settings className="text-blue-400" /> Configure {model.name}
                    </h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">x</button>
                </div>

                <div className="p-4 space-y-4">
                    {fields.map(field => (
                        <div key={field.key}>
                            <label className="block text-sm text-gray-400 mb-1">{field.label}</label>
                            {field.type === 'select' && (
                                <select
                                    value={config[field.key] || ''}
                                    onChange={e => setConfig({ ...config, [field.key]: e.target.value })}
                                    className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                                >
                                    {field.options.map(opt => (
                                        <option key={opt} value={opt}>{opt}</option>
                                    ))}
                                </select>
                            )}
                            {field.type === 'number' && (
                                <input
                                    type="number"
                                    value={config[field.key] || ''}
                                    onChange={e => setConfig({ ...config, [field.key]: parseInt(e.target.value) })}
                                    className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                                />
                            )}
                            {field.type === 'boolean' && (
                                <button
                                    onClick={() => setConfig({ ...config, [field.key]: !config[field.key] })}
                                    className={`w-12 h-6 rounded-full transition ${
                                        config[field.key] ? 'bg-green-600' : 'bg-gray-600'
                                    }`}
                                >
                                    <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                                        config[field.key] ? 'translate-x-6' : 'translate-x-1'
                                    }`} />
                                </button>
                            )}
                            {field.type === 'range' && (
                                <div className="flex items-center gap-3">
                                    <input
                                        type="range"
                                        min={field.min}
                                        max={field.max}
                                        step={field.step}
                                        value={config[field.key] || 0.5}
                                        onChange={e => setConfig({ ...config, [field.key]: parseFloat(e.target.value) })}
                                        className="flex-1"
                                    />
                                    <span className="w-12 text-right">{(config[field.key] || 0.5).toFixed(1)}</span>
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                <div className="p-4 border-t border-gray-700 flex gap-2 justify-end">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={() => { onSave(model.id, config); onClose(); }}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
                    >
                        Save Changes
                    </button>
                </div>
            </div>
        </div>
    );
};

// Main PerceptionManager Component
const PerceptionManager = () => {
    const [models, setModels] = useState([]);
    const [privacySettings, setPrivacySettings] = useState({
        enabled: true,
        security_bits: 128,
        lwe_dimension: 1024,
        enable_homomorphic_routing: true
    });
    const [pipelineStatus, setPipelineStatus] = useState({
        running: false,
        fusion_method: 'concat'
    });
    const [safetyPredictions, setSafetyPredictions] = useState({
        collision_prob: 0.0,
        horizon: 16,
        time_to_impact: -1
    });
    const [loading, setLoading] = useState(true);
    const [configModel, setConfigModel] = useState(null);

    // Fetch perception data
    const fetchData = async () => {
        try {
            const [modelsRes, privacyRes, pipelineRes, safetyRes] = await Promise.all([
                fetchWithAuth('/api/perception/models'),
                fetchWithAuth('/api/perception/privacy'),
                fetchWithAuth('/api/perception/pipeline/status'),
                fetchWithAuth('/api/perception/safety/predictions')
            ]);

            if (modelsRes.ok) setModels(await modelsRes.json());
            if (privacyRes.ok) setPrivacySettings(await privacyRes.json());
            if (pipelineRes.ok) setPipelineStatus(await pipelineRes.json());
            if (safetyRes.ok) setSafetyPredictions(await safetyRes.json());
        } catch (e) {
            console.error('Failed to fetch perception data', e);
            // Use defaults with mock data for demo
            setModels([
                {
                    id: 'dinov3',
                    name: 'DINOv3',
                    description: 'Self-supervised visual features with patch-based extraction',
                    version: 'ViT-L/14',
                    enabled: true,
                    status: 'running',
                    tflops: 8.0,
                    latency_ms: 12,
                    model_size: 'vit_large',
                    input_size: 518,
                    config: { model_size: 'vit_large', input_size: 518, use_fp16: true }
                },
                {
                    id: 'sam3',
                    name: 'SAM 3',
                    description: 'Zero-shot segmentation with text prompts and tracking',
                    version: 'SAM3-Large',
                    enabled: true,
                    status: 'running',
                    tflops: 15.0,
                    latency_ms: 25,
                    model_size: 'sam3_large',
                    input_size: 1024,
                    config: { model_size: 'sam3_large', input_size: 1024, max_objects: 10, confidence_threshold: 0.5, enable_tracking: true }
                },
                {
                    id: 'vjepa2',
                    name: 'V-JEPA 2',
                    description: 'Video prediction and world modeling for safety',
                    version: 'V-JEPA2-Large',
                    enabled: true,
                    status: 'running',
                    tflops: 10.0,
                    latency_ms: 18,
                    model_size: 'vjepa2_large',
                    input_size: 224,
                    config: { model_size: 'vjepa2_large', num_frames: 16, prediction_horizon: 16, enable_safety_prediction: true, collision_threshold: 0.7, emergency_stop_threshold: 0.9 }
                }
            ]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 2000);
        return () => clearInterval(interval);
    }, []);

    const handleToggleModel = async (modelId) => {
        try {
            const model = models.find(m => m.id === modelId);
            await fetchWithAuth(`/api/perception/models/${modelId}/enable`, {
                method: 'POST',
                body: JSON.stringify({ enabled: !model.enabled })
            });
            setModels(models.map(m =>
                m.id === modelId ? { ...m, enabled: !m.enabled, status: !m.enabled ? 'loading' : 'stopped' } : m
            ));
        } catch (e) {
            console.error('Failed to toggle model', e);
            // Toggle locally for demo
            setModels(models.map(m =>
                m.id === modelId ? { ...m, enabled: !m.enabled, status: !m.enabled ? 'running' : 'stopped' } : m
            ));
        }
    };

    const handleSaveConfig = async (modelId, config) => {
        try {
            await fetchWithAuth(`/api/perception/models/${modelId}/config`, {
                method: 'POST',
                body: JSON.stringify(config)
            });
            setModels(models.map(m =>
                m.id === modelId ? { ...m, config } : m
            ));
        } catch (e) {
            console.error('Failed to save config', e);
        }
    };

    const handleUpdatePrivacy = async (newSettings) => {
        try {
            await fetchWithAuth('/api/perception/privacy', {
                method: 'POST',
                body: JSON.stringify(newSettings)
            });
            setPrivacySettings(newSettings);
        } catch (e) {
            console.error('Failed to update privacy settings', e);
            setPrivacySettings(newSettings); // Update locally for demo
        }
    };

    if (loading) {
        return (
            <div className="p-6 flex items-center justify-center">
                <RefreshCw className="animate-spin" size={32} />
            </div>
        );
    }

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Brain className="text-purple-400" /> Meta AI Perception Manager
                </h1>
                <div className="flex gap-2">
                    <button
                        onClick={fetchData}
                        className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <RefreshCw size={18} /> Refresh
                    </button>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Total TFLOPS</div>
                    <div className="text-2xl font-bold text-blue-400">
                        {models.filter(m => m.enabled).reduce((sum, m) => sum + m.tflops, 0).toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-500">of 33.0 allocated</div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Active Models</div>
                    <div className="text-2xl font-bold text-green-400">
                        {models.filter(m => m.enabled && m.status === 'running').length}
                    </div>
                    <div className="text-xs text-gray-500">of {models.length} total</div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Avg Latency</div>
                    <div className="text-2xl font-bold">
                        {(models.filter(m => m.enabled).reduce((sum, m) => sum + m.latency_ms, 0) / Math.max(1, models.filter(m => m.enabled).length)).toFixed(0)}ms
                    </div>
                    <div className="text-xs text-gray-500">per inference</div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Privacy</div>
                    <div className="text-2xl font-bold flex items-center gap-2">
                        {privacySettings.enabled ? (
                            <><Lock size={20} className="text-green-400" /> Enabled</>
                        ) : (
                            <><Unlock size={20} className="text-yellow-400" /> Disabled</>
                        )}
                    </div>
                    <div className="text-xs text-gray-500">{privacySettings.security_bits}-bit N2HE</div>
                </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-3 gap-6">
                {/* Left Column - Model Cards */}
                <div className="col-span-2 space-y-4">
                    <h2 className="text-lg font-semibold text-gray-300">Foundation Models</h2>
                    <div className="grid grid-cols-2 gap-4">
                        {models.map(model => (
                            <ModelCard
                                key={model.id}
                                model={model}
                                onToggle={handleToggleModel}
                                onConfigure={setConfigModel}
                            />
                        ))}
                    </div>
                </div>

                {/* Right Column - Status Panels */}
                <div className="space-y-4">
                    <TFLOPSChart models={models} />
                    <PipelineStatus status={pipelineStatus} />
                    <SafetyPredictionPanel predictions={safetyPredictions} />
                    <PrivacyPanel settings={privacySettings} onUpdate={handleUpdatePrivacy} />
                </div>
            </div>

            {/* Config Modal */}
            {configModel && (
                <ConfigModal
                    model={configModel}
                    onClose={() => setConfigModel(null)}
                    onSave={handleSaveConfig}
                />
            )}
        </div>
    );
};

export default PerceptionManager;

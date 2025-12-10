import React, { useState, useEffect } from 'react';
import { Zap, Upload, Download, Search, RefreshCw, Brain, Lock, Unlock, Play, BarChart3 } from 'lucide-react';
import { fetchWithAuth } from './api';

const SkillCard = ({ skill, onInvoke, onDownload }) => {
    const statusColor = {
        'active': 'bg-green-500',
        'pending': 'bg-yellow-500',
        'deprecated': 'bg-red-500',
    };

    return (
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 hover:border-blue-500 transition">
            <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                    <Brain size={20} className="text-blue-400" />
                    <h3 className="font-semibold">{skill.name}</h3>
                </div>
                <span className={`px-2 py-1 rounded text-xs ${statusColor[skill.status] || 'bg-gray-600'}`}>
                    {skill.status}
                </span>
            </div>
            <p className="text-gray-400 text-sm mb-3">{skill.description}</p>
            <div className="flex flex-wrap gap-1 mb-3">
                {skill.tags?.map((tag, i) => (
                    <span key={i} className="bg-gray-700 px-2 py-0.5 rounded text-xs">{tag}</span>
                ))}
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs text-gray-400 mb-3">
                <div>Type: <span className="text-white">{skill.skill_type}</span></div>
                <div>Version: <span className="text-white">{skill.version}</span></div>
            </div>
            <div className="flex gap-2">
                <button
                    onClick={() => onInvoke(skill)}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 py-1.5 rounded text-sm flex items-center justify-center gap-1"
                >
                    <Play size={14} /> Invoke
                </button>
                <button
                    onClick={() => onDownload(skill)}
                    className="flex-1 bg-gray-700 hover:bg-gray-600 py-1.5 rounded text-sm flex items-center justify-center gap-1"
                >
                    <Download size={14} /> Download
                </button>
            </div>
        </div>
    );
};

const SkillsManager = () => {
    const [skills, setSkills] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterType, setFilterType] = useState('all');
    const [cloudStatus, setCloudStatus] = useState(null);
    const [routingResult, setRoutingResult] = useState(null);
    const [taskDescription, setTaskDescription] = useState('');

    // Upload form state
    const [showUpload, setShowUpload] = useState(false);
    const [uploadForm, setUploadForm] = useState({
        name: '',
        description: '',
        skill_type: 'manipulation',
        version: '1.0.0',
        tags: '',
    });

    useEffect(() => {
        fetchSkills();
        fetchCloudStatus();
    }, []);

    const fetchSkills = async () => {
        setLoading(true);
        try {
            const res = await fetchWithAuth('/api/v1/skills');
            const data = await res.json();
            setSkills(data.skills || []);
        } catch (e) {
            console.error("Failed to fetch skills", e);
        } finally {
            setLoading(false);
        }
    };

    const fetchCloudStatus = async () => {
        try {
            const res = await fetchWithAuth('/cloud/status');
            const data = await res.json();
            setCloudStatus(data);
        } catch (e) {
            console.error("Failed to fetch cloud status", e);
        }
    };

    const handleRouteTask = async () => {
        if (!taskDescription.trim()) return;
        try {
            const res = await fetchWithAuth('/api/v1/skills/request', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_description: taskDescription,
                    max_skills: 3
                })
            });
            const data = await res.json();
            setRoutingResult(data);
        } catch (e) {
            console.error("Routing failed", e);
        }
    };

    const handleUploadSkill = async () => {
        try {
            // Create dummy weights for demo (in real app, this would be trained weights)
            const dummyWeights = new Float32Array(1000);
            for (let i = 0; i < 1000; i++) dummyWeights[i] = Math.random() * 2 - 1;
            const weightsB64 = btoa(String.fromCharCode(...new Uint8Array(dummyWeights.buffer)));

            const res = await fetchWithAuth('/api/v1/skills/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...uploadForm,
                    tags: uploadForm.tags.split(',').map(t => t.trim()).filter(t => t),
                    weights: weightsB64,
                    config: {}
                })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Skill uploaded! ID: ${data.skill_id}`);
                setShowUpload(false);
                fetchSkills();
            } else {
                alert(`Upload failed: ${data.error}`);
            }
        } catch (e) {
            alert(`Upload error: ${e.message}`);
        }
    };

    const filteredSkills = skills.filter(skill => {
        const matchesSearch = skill.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            skill.description?.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesType = filterType === 'all' || skill.skill_type === filterType;
        return matchesSearch && matchesType;
    });

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Zap className="text-yellow-400" /> MoE Skill Library
                </h1>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowUpload(true)}
                        className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Upload size={18} /> Upload Skill
                    </button>
                    <button
                        onClick={fetchSkills}
                        className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
                        disabled={loading}
                    >
                        <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
                    </button>
                </div>
            </div>

            {/* Architecture Status */}
            {cloudStatus && (
                <div className="bg-gray-800 p-4 rounded-lg mb-6 grid grid-cols-4 gap-4">
                    <div>
                        <div className="text-gray-400 text-sm">Architecture</div>
                        <div className="text-xl font-bold text-blue-400">{cloudStatus.architecture}</div>
                    </div>
                    <div>
                        <div className="text-gray-400 text-sm">Base Model Mode</div>
                        <div className="text-xl font-bold flex items-center gap-2">
                            <Lock size={16} className="text-yellow-400" />
                            {cloudStatus.base_model_mode?.toUpperCase()}
                        </div>
                    </div>
                    <div>
                        <div className="text-gray-400 text-sm">Total Skills</div>
                        <div className="text-xl font-bold">{cloudStatus.skill_library?.total_skills || 0}</div>
                    </div>
                    <div>
                        <div className="text-gray-400 text-sm">MoE Load Balance</div>
                        <div className="text-xl font-bold">
                            {(cloudStatus.moe_router?.load_balance * 100)?.toFixed(1) || 0}%
                        </div>
                    </div>
                </div>
            )}

            {/* Task Router */}
            <div className="bg-gray-800 p-4 rounded-lg mb-6">
                <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Brain className="text-purple-400" /> MoE Task Router
                </h2>
                <div className="flex gap-2 mb-3">
                    <input
                        type="text"
                        placeholder="Describe your task (e.g., 'Pick up the red cube and place it on the plate')"
                        value={taskDescription}
                        onChange={(e) => setTaskDescription(e.target.value)}
                        className="flex-1 bg-gray-900 border border-gray-700 rounded px-4 py-2"
                    />
                    <button
                        onClick={handleRouteTask}
                        className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded flex items-center gap-2"
                    >
                        <Search size={18} /> Route
                    </button>
                </div>
                {routingResult && (
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-sm text-gray-400 mb-2">
                            Routed in {routingResult.inference_time_ms?.toFixed(2)}ms
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                            {routingResult.skill_ids?.map((id, i) => (
                                <div key={id} className="bg-gray-800 p-2 rounded flex justify-between">
                                    <span className="font-mono text-sm">{id.slice(0, 12)}...</span>
                                    <span className="text-blue-400">
                                        {(routingResult.weights[i] * 100).toFixed(1)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Filters */}
            <div className="flex gap-4 mb-4">
                <div className="relative flex-1">
                    <Search size={18} className="absolute left-3 top-2.5 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search skills..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-gray-800 border border-gray-700 rounded pl-10 pr-4 py-2"
                    />
                </div>
                <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="all">All Types</option>
                    <option value="manipulation">Manipulation</option>
                    <option value="navigation">Navigation</option>
                    <option value="perception">Perception</option>
                    <option value="interaction">Interaction</option>
                </select>
            </div>

            {/* Skills Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredSkills.map(skill => (
                    <SkillCard
                        key={skill.id}
                        skill={skill}
                        onInvoke={(s) => {
                            setTaskDescription(`Execute skill: ${s.name}`);
                            handleRouteTask();
                        }}
                        onDownload={(s) => alert(`Downloading skill ${s.id}...`)}
                    />
                ))}
                {filteredSkills.length === 0 && !loading && (
                    <div className="col-span-full text-center text-gray-500 py-12">
                        No skills found. Upload your first skill!
                    </div>
                )}
            </div>

            {/* Upload Modal */}
            {showUpload && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-gray-800 p-6 rounded-lg w-full max-w-md">
                        <h2 className="text-xl font-bold mb-4">Upload New Skill</h2>
                        <div className="space-y-4">
                            <input
                                type="text"
                                placeholder="Skill Name"
                                value={uploadForm.name}
                                onChange={(e) => setUploadForm({...uploadForm, name: e.target.value})}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2"
                            />
                            <textarea
                                placeholder="Description"
                                value={uploadForm.description}
                                onChange={(e) => setUploadForm({...uploadForm, description: e.target.value})}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2 h-24"
                            />
                            <select
                                value={uploadForm.skill_type}
                                onChange={(e) => setUploadForm({...uploadForm, skill_type: e.target.value})}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2"
                            >
                                <option value="manipulation">Manipulation</option>
                                <option value="navigation">Navigation</option>
                                <option value="perception">Perception</option>
                                <option value="interaction">Interaction</option>
                            </select>
                            <input
                                type="text"
                                placeholder="Tags (comma separated)"
                                value={uploadForm.tags}
                                onChange={(e) => setUploadForm({...uploadForm, tags: e.target.value})}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2"
                            />
                            <input
                                type="text"
                                placeholder="Version (e.g., 1.0.0)"
                                value={uploadForm.version}
                                onChange={(e) => setUploadForm({...uploadForm, version: e.target.value})}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2"
                            />
                        </div>
                        <div className="flex gap-2 mt-6">
                            <button
                                onClick={handleUploadSkill}
                                className="flex-1 bg-purple-600 hover:bg-purple-700 py-2 rounded"
                            >
                                Upload
                            </button>
                            <button
                                onClick={() => setShowUpload(false)}
                                className="flex-1 bg-gray-700 hover:bg-gray-600 py-2 rounded"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SkillsManager;

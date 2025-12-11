import React, { useState, useEffect } from 'react';
import { Database, Upload, Trash2, Play, Pause, RefreshCw, GitBranch, Clock, HardDrive, FileText, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { fetchWithAuth } from './api';

const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6'];

const TrainingManager = () => {
    const [activeTab, setActiveTab] = useState('datasets');
    const [datasets, setDatasets] = useState([]);
    const [trainingJobs, setTrainingJobs] = useState([]);
    const [versions, setVersions] = useState([]);
    const [flStatus, setFlStatus] = useState(null);
    const [loading, setLoading] = useState(false);

    // Demo data for UI (in production, this would come from API)
    useEffect(() => {
        // Simulated datasets
        setDatasets([
            { id: 'ds-001', name: 'Grasp Demos v1', type: 'demonstration', samples: 1250, size_mb: 456, created: '2024-12-08', status: 'ready' },
            { id: 'ds-002', name: 'Navigation Routes', type: 'trajectory', samples: 890, size_mb: 234, created: '2024-12-07', status: 'ready' },
            { id: 'ds-003', name: 'Tool Use Collection', type: 'demonstration', samples: 432, size_mb: 178, created: '2024-12-09', status: 'processing' },
        ]);

        setTrainingJobs([
            { id: 'job-001', skill_name: 'precise_grasp', status: 'running', progress: 67, epoch: 34, loss: 0.0234, created: '2024-12-10 08:30' },
            { id: 'job-002', skill_name: 'pour_liquid', status: 'completed', progress: 100, epoch: 100, loss: 0.0089, created: '2024-12-09 14:00' },
            { id: 'job-003', skill_name: 'stack_blocks', status: 'queued', progress: 0, epoch: 0, loss: null, created: '2024-12-10 09:45' },
        ]);

        setVersions([
            { id: 'v1.2.0', skill: 'precise_grasp', date: '2024-12-10', author: 'FL-Aggregator', status: 'deployed', success_rate: 94.2 },
            { id: 'v1.1.0', skill: 'precise_grasp', date: '2024-12-08', author: 'FL-Aggregator', status: 'archived', success_rate: 89.1 },
            { id: 'v1.0.0', skill: 'precise_grasp', date: '2024-12-05', author: 'Initial', status: 'archived', success_rate: 82.3 },
        ]);

        setFlStatus({
            round: 42,
            participants: 8,
            aggregation_method: 'FedAvg',
            encryption: 'N2HE',
            last_aggregation: Date.now() / 1000 - 3600,
            skills_updated: 3
        });
    }, []);

    const tabs = [
        { id: 'datasets', label: 'Datasets', icon: Database },
        { id: 'training', label: 'Training Jobs', icon: Play },
        { id: 'versions', label: 'Version Control', icon: GitBranch },
        { id: 'federated', label: 'Federated Learning', icon: RefreshCw },
    ];

    const handleDeleteDataset = (id) => {
        if (confirm('Are you sure you want to delete this dataset?')) {
            setDatasets(prev => prev.filter(d => d.id !== id));
        }
    };

    const handleStartTraining = (datasetId) => {
        const newJob = {
            id: `job-${Date.now()}`,
            skill_name: 'new_skill',
            status: 'queued',
            progress: 0,
            epoch: 0,
            loss: null,
            created: new Date().toLocaleString()
        };
        setTrainingJobs(prev => [newJob, ...prev]);
        alert('Training job queued!');
    };

    const lossHistory = [
        { epoch: 0, loss: 0.8 },
        { epoch: 10, loss: 0.4 },
        { epoch: 20, loss: 0.2 },
        { epoch: 30, loss: 0.1 },
        { epoch: 40, loss: 0.05 },
        { epoch: 50, loss: 0.03 },
    ];

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Database className="text-blue-400" /> Training & Data Management
                </h1>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mb-6 border-b border-gray-700 pb-2">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 rounded-t flex items-center gap-2 transition ${
                            activeTab === tab.id
                                ? 'bg-gray-800 text-white'
                                : 'text-gray-400 hover:text-white'
                        }`}
                    >
                        <tab.icon size={18} /> {tab.label}
                    </button>
                ))}
            </div>

            {/* Datasets Tab */}
            {activeTab === 'datasets' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Total Datasets</div>
                            <div className="text-2xl font-bold">{datasets.length}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Total Samples</div>
                            <div className="text-2xl font-bold">
                                {datasets.reduce((sum, d) => sum + d.samples, 0).toLocaleString()}
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Storage Used</div>
                            <div className="text-2xl font-bold">
                                {(datasets.reduce((sum, d) => sum + d.size_mb, 0) / 1024).toFixed(2)} GB
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Processing</div>
                            <div className="text-2xl font-bold text-yellow-400">
                                {datasets.filter(d => d.status === 'processing').length}
                            </div>
                        </div>
                    </div>

                    <div className="bg-gray-800 rounded-lg overflow-hidden">
                        <table className="w-full">
                            <thead className="bg-gray-900">
                                <tr>
                                    <th className="text-left p-4">Dataset</th>
                                    <th className="text-left p-4">Type</th>
                                    <th className="text-left p-4">Samples</th>
                                    <th className="text-left p-4">Size</th>
                                    <th className="text-left p-4">Created</th>
                                    <th className="text-left p-4">Status</th>
                                    <th className="text-left p-4">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {datasets.map(dataset => (
                                    <tr key={dataset.id} className="border-t border-gray-700">
                                        <td className="p-4">
                                            <div className="font-semibold">{dataset.name}</div>
                                            <div className="text-gray-400 text-xs font-mono">{dataset.id}</div>
                                        </td>
                                        <td className="p-4">
                                            <span className="bg-gray-700 px-2 py-1 rounded text-sm">{dataset.type}</span>
                                        </td>
                                        <td className="p-4">{dataset.samples.toLocaleString()}</td>
                                        <td className="p-4">{dataset.size_mb} MB</td>
                                        <td className="p-4 text-gray-400">{dataset.created}</td>
                                        <td className="p-4">
                                            <span className={`px-2 py-1 rounded text-xs ${
                                                dataset.status === 'ready' ? 'bg-green-900 text-green-400' :
                                                dataset.status === 'processing' ? 'bg-yellow-900 text-yellow-400' :
                                                'bg-red-900 text-red-400'
                                            }`}>
                                                {dataset.status}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => handleStartTraining(dataset.id)}
                                                    className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm flex items-center gap-1"
                                                    disabled={dataset.status !== 'ready'}
                                                >
                                                    <Play size={14} /> Train
                                                </button>
                                                <button
                                                    onClick={() => handleDeleteDataset(dataset.id)}
                                                    className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <button className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded flex items-center gap-2">
                        <Upload size={18} /> Upload New Dataset
                    </button>
                </div>
            )}

            {/* Training Jobs Tab */}
            {activeTab === 'training' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-6">
                        {/* Jobs List */}
                        <div className="space-y-4">
                            {trainingJobs.map(job => (
                                <div key={job.id} className="bg-gray-800 p-4 rounded-lg">
                                    <div className="flex justify-between items-start mb-3">
                                        <div>
                                            <h3 className="font-semibold">{job.skill_name}</h3>
                                            <div className="text-gray-400 text-xs font-mono">{job.id}</div>
                                        </div>
                                        <span className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${
                                            job.status === 'running' ? 'bg-blue-900 text-blue-400' :
                                            job.status === 'completed' ? 'bg-green-900 text-green-400' :
                                            job.status === 'queued' ? 'bg-yellow-900 text-yellow-400' :
                                            'bg-red-900 text-red-400'
                                        }`}>
                                            {job.status === 'running' && <RefreshCw size={12} className="animate-spin" />}
                                            {job.status === 'completed' && <CheckCircle size={12} />}
                                            {job.status}
                                        </span>
                                    </div>
                                    <div className="mb-2">
                                        <div className="flex justify-between text-sm text-gray-400 mb-1">
                                            <span>Progress</span>
                                            <span>{job.progress}%</span>
                                        </div>
                                        <div className="w-full bg-gray-700 rounded-full h-2">
                                            <div
                                                className="bg-blue-500 h-2 rounded-full transition-all"
                                                style={{ width: `${job.progress}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-3 gap-2 text-sm">
                                        <div><span className="text-gray-400">Epoch:</span> {job.epoch}</div>
                                        <div><span className="text-gray-400">Loss:</span> {job.loss?.toFixed(4) || '--'}</div>
                                        <div><span className="text-gray-400">Started:</span> {job.created}</div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Loss Chart */}
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-4">Training Loss</h3>
                            <div className="h-64">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={lossHistory}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="epoch" stroke="#9ca3af" />
                                        <YAxis stroke="#9ca3af" />
                                        <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                                        <Line type="monotone" dataKey="loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Version Control Tab */}
            {activeTab === 'versions' && (
                <div className="space-y-6">
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <GitBranch className="text-purple-400" /> Skill Version History
                        </h3>
                        <div className="relative">
                            {/* Version timeline */}
                            <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-700" />
                            {versions.map((version, i) => (
                                <div key={version.id} className="relative flex items-start gap-4 mb-6">
                                    <div className={`w-12 h-12 rounded-full flex items-center justify-center z-10 ${
                                        version.status === 'deployed' ? 'bg-green-600' :
                                        version.status === 'archived' ? 'bg-gray-600' : 'bg-blue-600'
                                    }`}>
                                        <GitBranch size={20} />
                                    </div>
                                    <div className="flex-1 bg-gray-900 p-4 rounded-lg">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <h4 className="font-semibold">{version.id}</h4>
                                                <div className="text-gray-400 text-sm">{version.skill}</div>
                                            </div>
                                            <span className={`px-2 py-1 rounded text-xs ${
                                                version.status === 'deployed' ? 'bg-green-900 text-green-400' :
                                                'bg-gray-700 text-gray-400'
                                            }`}>
                                                {version.status}
                                            </span>
                                        </div>
                                        <div className="mt-2 grid grid-cols-3 gap-4 text-sm">
                                            <div><span className="text-gray-400">Date:</span> {version.date}</div>
                                            <div><span className="text-gray-400">Author:</span> {version.author}</div>
                                            <div><span className="text-gray-400">Success Rate:</span> <span className="text-green-400">{version.success_rate}%</span></div>
                                        </div>
                                        {version.status !== 'deployed' && (
                                            <button className="mt-3 bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm">
                                                Rollback to this version
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Federated Learning Tab */}
            {activeTab === 'federated' && flStatus && (
                <div className="space-y-6">
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Current Round</div>
                            <div className="text-2xl font-bold text-blue-400">{flStatus.round}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Active Participants</div>
                            <div className="text-2xl font-bold text-green-400">{flStatus.participants}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Aggregation Method</div>
                            <div className="text-2xl font-bold">{flStatus.aggregation_method}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Encryption</div>
                            <div className="text-2xl font-bold text-purple-400">{flStatus.encryption}</div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-4">FL Architecture</h3>
                            <div className="bg-gray-900 p-4 rounded font-mono text-sm">
                                <pre>{`
┌─────────────────────────────────────┐
│         CLOUD (Aggregator)          │
│  ┌─────────────────────────────┐   │
│  │  FedAvg + N2HE Encryption   │   │
│  └─────────────────────────────┘   │
│              ▲ ▲ ▲                  │
│              │ │ │  Encrypted       │
│              │ │ │  Skill Updates   │
├──────────────┼─┼─┼──────────────────┤
│  Edge 1      │ │ │      Edge N      │
│  ┌────┐    ┌─┴─┴─┴─┐    ┌────┐     │
│  │Orin│◄──►│ Skill │◄──►│Orin│     │
│  └────┘    │Library│    └────┘     │
│            └───────┘               │
└─────────────────────────────────────┘
                `}</pre>
                            </div>
                        </div>

                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-4">Recent Updates</h3>
                            <div className="space-y-3">
                                <div className="bg-gray-900 p-3 rounded flex justify-between items-center">
                                    <div>
                                        <div className="font-semibold">Skills Updated</div>
                                        <div className="text-gray-400 text-sm">Last aggregation</div>
                                    </div>
                                    <div className="text-2xl font-bold text-green-400">{flStatus.skills_updated}</div>
                                </div>
                                <div className="bg-gray-900 p-3 rounded flex justify-between items-center">
                                    <div>
                                        <div className="font-semibold">Last Aggregation</div>
                                        <div className="text-gray-400 text-sm">Time since last FL round</div>
                                    </div>
                                    <div className="text-xl font-bold">
                                        {Math.round((Date.now() / 1000 - flStatus.last_aggregation) / 60)} min ago
                                    </div>
                                </div>
                                <div className="bg-gray-900 p-3 rounded">
                                    <div className="font-semibold mb-2">Security Status</div>
                                    <div className="flex items-center gap-2 text-green-400">
                                        <CheckCircle size={16} />
                                        <span>All skill updates encrypted with N2HE (128-bit)</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-green-400 mt-1">
                                        <CheckCircle size={16} />
                                        <span>Base VLA models remain frozen (IP-safe)</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default TrainingManager;

/**
 * Version Control - Configuration Version Management
 *
 * Enables system integrators to:
 * - View configuration history
 * - Compare versions
 * - Rollback to previous configurations
 * - Create configuration snapshots
 *
 * @version 0.7.0
 */

import React, { useState, useEffect } from 'react';
import {
    GitBranch, Plus, RefreshCw, Clock, CheckCircle, GitCompare,
    RotateCcw, FileText, Settings, Shield, Zap, User, Eye,
    ChevronDown, ChevronRight, Copy, Download
} from 'lucide-react';
import { fetchWithAuth } from './api';

const configTypeIcons = {
    safety: <Shield size={16} className="text-red-400" />,
    perception: <Eye size={16} className="text-blue-400" />,
    skills: <Zap size={16} className="text-yellow-400" />,
    system: <Settings size={16} className="text-green-400" />,
    preset_applied: <Copy size={16} className="text-purple-400" />,
};

const ConfigVersionCard = ({ version, onView, onActivate, onCompare, isSelected }) => {
    const timeString = new Date(version.created_at * 1000).toLocaleString();

    return (
        <div
            className={`bg-gray-800 rounded-lg border transition-all cursor-pointer ${
                version.is_active ? 'border-green-500' :
                isSelected ? 'border-blue-500' : 'border-gray-700 hover:border-gray-600'
            }`}
            onClick={() => onView(version)}
        >
            <div className="p-4">
                <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                        {configTypeIcons[version.config_type] || <FileText size={16} className="text-gray-400" />}
                        <span className="font-semibold text-white capitalize">
                            {version.config_type.replace('_', ' ')}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        {version.is_active && (
                            <span className="bg-green-900 text-green-400 px-2 py-0.5 rounded text-xs flex items-center gap-1">
                                <CheckCircle size={12} /> Active
                            </span>
                        )}
                        <span className="bg-gray-700 px-2 py-0.5 rounded text-xs font-mono">
                            v{version.version}
                        </span>
                    </div>
                </div>

                {version.comment && (
                    <p className="text-gray-400 text-sm mb-3">{version.comment}</p>
                )}

                <div className="flex justify-between items-center text-sm">
                    <div className="flex items-center gap-2 text-gray-500">
                        <Clock size={14} />
                        {timeString}
                    </div>
                    <div className="flex items-center gap-2 text-gray-500">
                        <User size={14} />
                        {version.created_by}
                    </div>
                </div>

                <div className="flex gap-2 mt-3 pt-3 border-t border-gray-700">
                    {!version.is_active && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onActivate(version); }}
                            className="flex-1 bg-green-600 hover:bg-green-700 py-1.5 rounded text-sm flex items-center justify-center gap-1"
                        >
                            <RotateCcw size={14} /> Rollback
                        </button>
                    )}
                    <button
                        onClick={(e) => { e.stopPropagation(); onCompare(version); }}
                        className="flex-1 bg-gray-700 hover:bg-gray-600 py-1.5 rounded text-sm flex items-center justify-center gap-1"
                    >
                        <GitCompare size={14} /> Compare
                    </button>
                </div>
            </div>
        </div>
    );
};

const CreateVersionModal = ({ isOpen, onClose, onCreated }) => {
    const [form, setForm] = useState({
        config_type: 'system',
        config_data: '{}',
        comment: ''
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    if (!isOpen) return null;

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            // Parse JSON
            let configData;
            try {
                configData = JSON.parse(form.config_data);
            } catch (e) {
                setError('Invalid JSON in configuration data');
                setLoading(false);
                return;
            }

            const res = await fetchWithAuth('/api/v1/integrator/config-versions', {
                method: 'POST',
                body: JSON.stringify({
                    config_type: form.config_type,
                    config_data: configData,
                    comment: form.comment
                })
            });
            const data = await res.json();
            if (data.success) {
                onCreated(data);
                onClose();
            } else {
                setError(data.error || 'Failed to create version');
            }
        } catch (e) {
            setError('Error creating version: ' + e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-lg">
                <div className="p-6 border-b border-gray-700">
                    <h2 className="text-xl font-bold">Create Configuration Snapshot</h2>
                    <p className="text-gray-400 text-sm mt-1">Save current configuration as a new version</p>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-4">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Configuration Type *</label>
                        <select
                            value={form.config_type}
                            onChange={(e) => setForm({ ...form, config_type: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                        >
                            <option value="system">System</option>
                            <option value="safety">Safety</option>
                            <option value="perception">Perception</option>
                            <option value="skills">Skills</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Configuration Data (JSON) *</label>
                        <textarea
                            value={form.config_data}
                            onChange={(e) => setForm({ ...form, config_data: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 h-40 font-mono text-sm"
                            placeholder='{"key": "value"}'
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Comment</label>
                        <input
                            type="text"
                            value={form.comment}
                            onChange={(e) => setForm({ ...form, comment: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                            placeholder="Describe what changed..."
                        />
                    </div>

                    {error && (
                        <div className="bg-red-900/30 border border-red-700 rounded p-3 text-red-400 text-sm">
                            {error}
                        </div>
                    )}

                    <div className="flex gap-3 pt-4">
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex-1 bg-blue-600 hover:bg-blue-700 py-2 rounded font-medium"
                        >
                            {loading ? 'Creating...' : 'Create Version'}
                        </button>
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 bg-gray-700 hover:bg-gray-600 py-2 rounded"
                        >
                            Cancel
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

const VersionDetailsModal = ({ version, onClose }) => {
    if (!version) return null;

    const timeString = new Date(version.created_at * 1000).toLocaleString();

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
                <div className="p-6 border-b border-gray-700 flex justify-between items-start">
                    <div>
                        <div className="flex items-center gap-2">
                            {configTypeIcons[version.config_type]}
                            <h2 className="text-xl font-bold capitalize">{version.config_type.replace('_', ' ')}</h2>
                            <span className="bg-gray-700 px-2 py-0.5 rounded text-sm font-mono">
                                v{version.version}
                            </span>
                            {version.is_active && (
                                <span className="bg-green-900 text-green-400 px-2 py-0.5 rounded text-xs">
                                    Active
                                </span>
                            )}
                        </div>
                        <p className="text-gray-400 text-sm mt-1">{version.comment || 'No comment'}</p>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-white text-xl">
                        &times;
                    </button>
                </div>

                <div className="p-6 space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                        <div className="bg-gray-900 p-3 rounded">
                            <div className="text-gray-400 text-sm">Created</div>
                            <div className="text-white">{timeString}</div>
                        </div>
                        <div className="bg-gray-900 p-3 rounded">
                            <div className="text-gray-400 text-sm">Created By</div>
                            <div className="text-white">{version.created_by}</div>
                        </div>
                        <div className="bg-gray-900 p-3 rounded">
                            <div className="text-gray-400 text-sm">Checksum</div>
                            <div className="text-white font-mono text-sm">{version.checksum || 'N/A'}</div>
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <h4 className="text-gray-400 text-sm">Configuration Data</h4>
                            <button
                                onClick={() => navigator.clipboard.writeText(JSON.stringify(version.config_data, null, 2))}
                                className="text-gray-400 hover:text-white text-sm flex items-center gap-1"
                            >
                                <Copy size={14} /> Copy
                            </button>
                        </div>
                        <pre className="bg-gray-900 p-4 rounded overflow-auto max-h-96 text-sm font-mono">
                            {JSON.stringify(version.config_data, null, 2)}
                        </pre>
                    </div>
                </div>
            </div>
        </div>
    );
};

const CompareModal = ({ versionA, versionB, onClose }) => {
    const [comparison, setComparison] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (versionA && versionB) {
            fetchComparison();
        }
    }, [versionA, versionB]);

    const fetchComparison = async () => {
        setLoading(true);
        try {
            const res = await fetchWithAuth(
                `/api/v1/integrator/config-versions/compare/${versionA.id}/${versionB.id}`
            );
            const data = await res.json();
            setComparison(data);
        } catch (e) {
            console.error('Failed to compare', e);
        } finally {
            setLoading(false);
        }
    };

    if (!versionA || !versionB) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto">
                <div className="p-6 border-b border-gray-700 flex justify-between items-start">
                    <div>
                        <h2 className="text-xl font-bold flex items-center gap-2">
                            <GitCompare className="text-blue-400" /> Compare Versions
                        </h2>
                        <p className="text-gray-400 text-sm mt-1">
                            v{versionA.version} vs v{versionB.version}
                        </p>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-white text-xl">
                        &times;
                    </button>
                </div>

                <div className="p-6">
                    {loading ? (
                        <div className="flex items-center justify-center h-32">
                            <RefreshCw className="animate-spin" size={32} />
                        </div>
                    ) : comparison ? (
                        <div>
                            <div className="grid grid-cols-2 gap-4 mb-6">
                                <div className="bg-gray-900 p-4 rounded">
                                    <h4 className="text-gray-400 text-sm mb-1">Version A</h4>
                                    <div className="font-mono">v{comparison.version_a.version}</div>
                                    <div className="text-gray-500 text-sm">
                                        {new Date(comparison.version_a.created_at * 1000).toLocaleString()}
                                    </div>
                                </div>
                                <div className="bg-gray-900 p-4 rounded">
                                    <h4 className="text-gray-400 text-sm mb-1">Version B</h4>
                                    <div className="font-mono">v{comparison.version_b.version}</div>
                                    <div className="text-gray-500 text-sm">
                                        {new Date(comparison.version_b.created_at * 1000).toLocaleString()}
                                    </div>
                                </div>
                            </div>

                            <div className="mb-4">
                                <span className="text-lg font-semibold">{comparison.total_changes}</span>
                                <span className="text-gray-400 ml-2">changes detected</span>
                            </div>

                            {Object.keys(comparison.differences || {}).length > 0 ? (
                                <div className="space-y-3">
                                    {Object.entries(comparison.differences).map(([key, diff]) => (
                                        <div key={key} className="bg-gray-900 rounded p-4">
                                            <div className="font-semibold text-white mb-2">{key}</div>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div>
                                                    <div className="text-red-400 text-sm mb-1">Old Value</div>
                                                    <pre className="bg-red-900/20 p-2 rounded text-sm overflow-auto">
                                                        {JSON.stringify(diff.old, null, 2)}
                                                    </pre>
                                                </div>
                                                <div>
                                                    <div className="text-green-400 text-sm mb-1">New Value</div>
                                                    <pre className="bg-green-900/20 p-2 rounded text-sm overflow-auto">
                                                        {JSON.stringify(diff.new, null, 2)}
                                                    </pre>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-center text-gray-500 py-8">
                                    No differences found between these versions.
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center text-gray-500 py-8">
                            Failed to load comparison.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

const VersionControl = () => {
    const [versions, setVersions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [typeFilter, setTypeFilter] = useState('all');
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [selectedVersion, setSelectedVersion] = useState(null);
    const [compareVersions, setCompareVersions] = useState({ a: null, b: null });

    useEffect(() => {
        fetchVersions();
    }, [typeFilter]);

    const fetchVersions = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams();
            if (typeFilter !== 'all') params.append('config_type', typeFilter);

            const res = await fetchWithAuth(`/api/v1/integrator/config-versions?${params}`);
            const data = await res.json();
            setVersions(data.versions || []);
        } catch (e) {
            console.error('Failed to fetch versions', e);
        } finally {
            setLoading(false);
        }
    };

    const handleActivate = async (version) => {
        if (!confirm(`Rollback to version ${version.version}? This will activate this configuration.`)) return;

        try {
            const res = await fetchWithAuth(`/api/v1/integrator/config-versions/${version.id}/activate`, {
                method: 'POST'
            });
            const data = await res.json();
            if (data.success) {
                fetchVersions();
            } else {
                alert('Failed to activate version');
            }
        } catch (e) {
            alert('Error activating version: ' + e.message);
        }
    };

    const handleCompare = (version) => {
        if (!compareVersions.a) {
            setCompareVersions({ a: version, b: null });
        } else if (!compareVersions.b && compareVersions.a.id !== version.id) {
            setCompareVersions({ ...compareVersions, b: version });
        } else {
            setCompareVersions({ a: version, b: null });
        }
    };

    const handleViewDetails = async (version) => {
        try {
            const res = await fetchWithAuth(`/api/v1/integrator/config-versions/${version.id}`);
            const data = await res.json();
            setSelectedVersion(data);
        } catch (e) {
            console.error('Failed to fetch version details', e);
        }
    };

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <GitBranch className="text-green-400" /> Version Control
                    </h1>
                    <p className="text-gray-400 mt-1">Manage configuration versions and rollbacks</p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Plus size={18} /> New Snapshot
                    </button>
                    <button
                        onClick={fetchVersions}
                        className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded"
                        disabled={loading}
                    >
                        <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
                    </button>
                </div>
            </div>

            {compareVersions.a && (
                <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-4 mb-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <GitCompare size={20} className="text-blue-400" />
                            <span>
                                Comparing: <strong>v{compareVersions.a.version}</strong>
                                {compareVersions.b && <> vs <strong>v{compareVersions.b.version}</strong></>}
                            </span>
                        </div>
                        <div className="flex gap-2">
                            {compareVersions.b ? (
                                <>
                                    <button
                                        onClick={() => setCompareVersions({ a: null, b: null })}
                                        className="bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded text-sm"
                                    >
                                        Clear
                                    </button>
                                </>
                            ) : (
                                <span className="text-blue-400 text-sm">Select another version to compare</span>
                            )}
                        </div>
                    </div>
                </div>
            )}

            <div className="flex gap-4 mb-6">
                <select
                    value={typeFilter}
                    onChange={(e) => setTypeFilter(e.target.value)}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="all">All Types</option>
                    <option value="system">System</option>
                    <option value="safety">Safety</option>
                    <option value="perception">Perception</option>
                    <option value="skills">Skills</option>
                    <option value="preset_applied">Applied Presets</option>
                </select>
            </div>

            {loading ? (
                <div className="flex items-center justify-center h-64">
                    <RefreshCw className="animate-spin" size={32} />
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {versions.map(version => (
                        <ConfigVersionCard
                            key={version.id}
                            version={version}
                            onView={handleViewDetails}
                            onActivate={handleActivate}
                            onCompare={handleCompare}
                            isSelected={compareVersions.a?.id === version.id || compareVersions.b?.id === version.id}
                        />
                    ))}
                    {versions.length === 0 && (
                        <div className="col-span-full text-center text-gray-500 py-12">
                            No configuration versions found. Create a snapshot to get started.
                        </div>
                    )}
                </div>
            )}

            <CreateVersionModal
                isOpen={showCreateModal}
                onClose={() => setShowCreateModal(false)}
                onCreated={() => fetchVersions()}
            />

            {selectedVersion && (
                <VersionDetailsModal
                    version={selectedVersion}
                    onClose={() => setSelectedVersion(null)}
                />
            )}

            {compareVersions.a && compareVersions.b && (
                <CompareModal
                    versionA={compareVersions.a}
                    versionB={compareVersions.b}
                    onClose={() => setCompareVersions({ a: null, b: null })}
                />
            )}
        </div>
    );
};

export default VersionControl;

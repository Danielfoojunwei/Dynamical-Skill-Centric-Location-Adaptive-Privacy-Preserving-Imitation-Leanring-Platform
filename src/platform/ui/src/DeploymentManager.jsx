/**
 * Deployment Manager - Fleet Management for System Integrators
 *
 * Enables system integrators to:
 * - View and manage all deployment sites
 * - Monitor health across deployments
 * - Create and configure new deployments
 * - Track deployment status and metrics
 *
 * @version 0.7.0
 */

import React, { useState, useEffect } from 'react';
import {
    Building2, Plus, RefreshCw, MapPin, Activity,
    Settings, Trash2, AlertTriangle, CheckCircle,
    Clock, Server, ChevronRight, Filter, Search,
    MoreVertical, Edit2, Eye
} from 'lucide-react';
import { fetchWithAuth } from './api';

const statusColors = {
    active: 'bg-green-500',
    inactive: 'bg-gray-500',
    maintenance: 'bg-yellow-500',
    error: 'bg-red-500',
};

const environmentBadges = {
    production: 'bg-red-900 text-red-300',
    staging: 'bg-yellow-900 text-yellow-300',
    dev: 'bg-blue-900 text-blue-300',
};

const DeploymentCard = ({ deployment, onSelect, onEdit, onDelete }) => {
    const healthColor = deployment.health_score >= 90 ? 'text-green-400' :
        deployment.health_score >= 70 ? 'text-yellow-400' : 'text-red-400';

    const lastSeen = deployment.last_heartbeat
        ? new Date(deployment.last_heartbeat * 1000).toLocaleString()
        : 'Never';

    return (
        <div
            className="bg-gray-800 rounded-lg border border-gray-700 hover:border-blue-500 transition-all cursor-pointer"
            onClick={() => onSelect(deployment)}
        >
            <div className="p-4">
                <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${statusColors[deployment.status] || 'bg-gray-500'}`} />
                        <div>
                            <h3 className="font-semibold text-white">{deployment.name}</h3>
                            <div className="text-gray-400 text-sm flex items-center gap-1">
                                <Building2 size={12} />
                                {deployment.site_name}
                            </div>
                        </div>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs ${environmentBadges[deployment.environment] || 'bg-gray-700'}`}>
                        {deployment.environment}
                    </span>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-3">
                    <div>
                        <div className="text-gray-500 text-xs">Health Score</div>
                        <div className={`text-lg font-bold ${healthColor}`}>
                            {deployment.health_score?.toFixed(0)}%
                        </div>
                    </div>
                    <div>
                        <div className="text-gray-500 text-xs">Robots</div>
                        <div className="text-lg font-bold text-white">{deployment.robot_count}</div>
                    </div>
                </div>

                {deployment.location && (
                    <div className="flex items-center gap-1 text-gray-400 text-sm mb-3">
                        <MapPin size={12} />
                        {deployment.location}
                    </div>
                )}

                <div className="flex flex-wrap gap-1 mb-3">
                    {deployment.tags?.map((tag, i) => (
                        <span key={i} className="bg-gray-700 px-2 py-0.5 rounded text-xs">{tag}</span>
                    ))}
                </div>

                <div className="flex justify-between items-center pt-3 border-t border-gray-700">
                    <div className="text-gray-500 text-xs flex items-center gap-1">
                        <Clock size={12} />
                        Last seen: {lastSeen}
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={(e) => { e.stopPropagation(); onEdit(deployment); }}
                            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white"
                        >
                            <Edit2 size={14} />
                        </button>
                        <button
                            onClick={(e) => { e.stopPropagation(); onDelete(deployment); }}
                            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-red-400"
                        >
                            <Trash2 size={14} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

const CreateDeploymentModal = ({ isOpen, onClose, onCreated }) => {
    const [form, setForm] = useState({
        name: '',
        site_name: '',
        location: '',
        description: '',
        environment: 'production',
        robot_count: 1,
        tags: '',
        contact_email: '',
        notes: ''
    });
    const [loading, setLoading] = useState(false);

    if (!isOpen) return null;

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const res = await fetchWithAuth('/api/v1/integrator/deployments', {
                method: 'POST',
                body: JSON.stringify({
                    ...form,
                    tags: form.tags.split(',').map(t => t.trim()).filter(Boolean)
                })
            });
            const data = await res.json();
            if (data.success) {
                onCreated(data);
                onClose();
            } else {
                alert('Failed to create deployment: ' + (data.error || 'Unknown error'));
            }
        } catch (e) {
            alert('Error creating deployment: ' + e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto">
                <div className="p-6 border-b border-gray-700">
                    <h2 className="text-xl font-bold">Create New Deployment</h2>
                    <p className="text-gray-400 text-sm mt-1">Add a new client site or deployment</p>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Deployment Name *</label>
                            <input
                                type="text"
                                required
                                value={form.name}
                                onChange={(e) => setForm({ ...form, name: e.target.value })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                                placeholder="Main Production Line"
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Client Site Name *</label>
                            <input
                                type="text"
                                required
                                value={form.site_name}
                                onChange={(e) => setForm({ ...form, site_name: e.target.value })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                                placeholder="Acme Corp Factory"
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Location</label>
                            <input
                                type="text"
                                value={form.location}
                                onChange={(e) => setForm({ ...form, location: e.target.value })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                                placeholder="Building A, Floor 2"
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Environment *</label>
                            <select
                                value={form.environment}
                                onChange={(e) => setForm({ ...form, environment: e.target.value })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                            >
                                <option value="production">Production</option>
                                <option value="staging">Staging</option>
                                <option value="dev">Development</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Description</label>
                        <textarea
                            value={form.description}
                            onChange={(e) => setForm({ ...form, description: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 h-20"
                            placeholder="Brief description of the deployment..."
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Robot Count</label>
                            <input
                                type="number"
                                min="1"
                                value={form.robot_count}
                                onChange={(e) => setForm({ ...form, robot_count: parseInt(e.target.value) || 1 })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Contact Email</label>
                            <input
                                type="email"
                                value={form.contact_email}
                                onChange={(e) => setForm({ ...form, contact_email: e.target.value })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                                placeholder="contact@client.com"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Tags (comma separated)</label>
                        <input
                            type="text"
                            value={form.tags}
                            onChange={(e) => setForm({ ...form, tags: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2"
                            placeholder="warehouse, autonomous, high-priority"
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Notes</label>
                        <textarea
                            value={form.notes}
                            onChange={(e) => setForm({ ...form, notes: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 h-20"
                            placeholder="Internal notes about this deployment..."
                        />
                    </div>

                    <div className="flex gap-3 pt-4">
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex-1 bg-blue-600 hover:bg-blue-700 py-2 rounded font-medium"
                        >
                            {loading ? 'Creating...' : 'Create Deployment'}
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

const DeploymentDetails = ({ deployment, onClose, onRefresh }) => {
    if (!deployment) return null;

    const healthColor = deployment.health_score >= 90 ? 'bg-green-500' :
        deployment.health_score >= 70 ? 'bg-yellow-500' : 'bg-red-500';

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
                <div className="p-6 border-b border-gray-700">
                    <div className="flex justify-between items-start">
                        <div>
                            <h2 className="text-xl font-bold flex items-center gap-2">
                                <div className={`w-3 h-3 rounded-full ${statusColors[deployment.status]}`} />
                                {deployment.name}
                            </h2>
                            <p className="text-gray-400 text-sm mt-1">{deployment.site_name}</p>
                        </div>
                        <button onClick={onClose} className="text-gray-400 hover:text-white">
                            &times;
                        </button>
                    </div>
                </div>

                <div className="p-6 space-y-6">
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-gray-900 p-4 rounded-lg text-center">
                            <div className="text-gray-400 text-sm">Health</div>
                            <div className="text-2xl font-bold flex items-center justify-center gap-2">
                                <div className={`w-3 h-3 rounded-full ${healthColor}`} />
                                {deployment.health_score?.toFixed(0)}%
                            </div>
                        </div>
                        <div className="bg-gray-900 p-4 rounded-lg text-center">
                            <div className="text-gray-400 text-sm">Robots</div>
                            <div className="text-2xl font-bold">{deployment.robot_count}</div>
                        </div>
                        <div className="bg-gray-900 p-4 rounded-lg text-center">
                            <div className="text-gray-400 text-sm">Environment</div>
                            <div className="text-lg font-bold capitalize">{deployment.environment}</div>
                        </div>
                        <div className="bg-gray-900 p-4 rounded-lg text-center">
                            <div className="text-gray-400 text-sm">Status</div>
                            <div className="text-lg font-bold capitalize">{deployment.status}</div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Location</h4>
                            <p className="text-white">{deployment.location || 'Not specified'}</p>
                        </div>
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Contact</h4>
                            <p className="text-white">{deployment.contact_email || 'Not specified'}</p>
                        </div>
                    </div>

                    {deployment.description && (
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Description</h4>
                            <p className="text-white">{deployment.description}</p>
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Platform Version</h4>
                            <p className="text-white font-mono">{deployment.platform_version || 'Unknown'}</p>
                        </div>
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Config Version</h4>
                            <p className="text-white font-mono">{deployment.config_version || 'Default'}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Created</h4>
                            <p className="text-white">
                                {new Date(deployment.created_at * 1000).toLocaleString()}
                            </p>
                        </div>
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Last Updated</h4>
                            <p className="text-white">
                                {new Date(deployment.updated_at * 1000).toLocaleString()}
                            </p>
                        </div>
                    </div>

                    {deployment.tags?.length > 0 && (
                        <div>
                            <h4 className="text-gray-400 text-sm mb-2">Tags</h4>
                            <div className="flex flex-wrap gap-2">
                                {deployment.tags.map((tag, i) => (
                                    <span key={i} className="bg-gray-700 px-3 py-1 rounded">{tag}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    {deployment.notes && (
                        <div>
                            <h4 className="text-gray-400 text-sm mb-1">Notes</h4>
                            <p className="text-gray-300 bg-gray-900 p-3 rounded">{deployment.notes}</p>
                        </div>
                    )}

                    <div className="flex gap-3 pt-4 border-t border-gray-700">
                        <button className="flex-1 bg-blue-600 hover:bg-blue-700 py-2 rounded flex items-center justify-center gap-2">
                            <Eye size={16} /> View Dashboard
                        </button>
                        <button className="flex-1 bg-gray-700 hover:bg-gray-600 py-2 rounded flex items-center justify-center gap-2">
                            <Settings size={16} /> Configure
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

const DeploymentManager = () => {
    const [deployments, setDeployments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [envFilter, setEnvFilter] = useState('all');
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [selectedDeployment, setSelectedDeployment] = useState(null);
    const [summary, setSummary] = useState(null);

    useEffect(() => {
        fetchDeployments();
        fetchSummary();
    }, []);

    const fetchDeployments = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams();
            if (statusFilter !== 'all') params.append('status', statusFilter);
            if (envFilter !== 'all') params.append('environment', envFilter);

            const res = await fetchWithAuth(`/api/v1/integrator/deployments?${params}`);
            const data = await res.json();
            setDeployments(data.deployments || []);
        } catch (e) {
            console.error('Failed to fetch deployments', e);
        } finally {
            setLoading(false);
        }
    };

    const fetchSummary = async () => {
        try {
            const res = await fetchWithAuth('/api/v1/integrator/dashboard/summary');
            const data = await res.json();
            setSummary(data);
        } catch (e) {
            console.error('Failed to fetch summary', e);
        }
    };

    const handleDelete = async (deployment) => {
        if (!confirm(`Are you sure you want to delete "${deployment.name}"?`)) return;
        try {
            await fetchWithAuth(`/api/v1/integrator/deployments/${deployment.id}`, { method: 'DELETE' });
            fetchDeployments();
        } catch (e) {
            alert('Failed to delete deployment');
        }
    };

    const filteredDeployments = deployments.filter(d =>
        d.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        d.site_name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Building2 className="text-blue-400" /> Deployment Manager
                    </h1>
                    <p className="text-gray-400 mt-1">Manage your client sites and robot deployments</p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Plus size={18} /> New Deployment
                    </button>
                    <button
                        onClick={() => { fetchDeployments(); fetchSummary(); }}
                        className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded"
                        disabled={loading}
                    >
                        <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
                    </button>
                </div>
            </div>

            {summary && (
                <div className="grid grid-cols-4 gap-4 mb-6">
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Total Deployments</div>
                        <div className="text-2xl font-bold text-white">{summary.deployments.total}</div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Active</div>
                        <div className="text-2xl font-bold text-green-400">{summary.deployments.active}</div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Needs Attention</div>
                        <div className="text-2xl font-bold text-yellow-400">{summary.deployments.unhealthy}</div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Unresolved Alerts</div>
                        <div className="text-2xl font-bold text-red-400">{summary.alerts.unresolved}</div>
                    </div>
                </div>
            )}

            <div className="flex gap-4 mb-4">
                <div className="relative flex-1">
                    <Search size={18} className="absolute left-3 top-2.5 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search deployments..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-gray-800 border border-gray-700 rounded pl-10 pr-4 py-2"
                    />
                </div>
                <select
                    value={statusFilter}
                    onChange={(e) => { setStatusFilter(e.target.value); fetchDeployments(); }}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                    <option value="maintenance">Maintenance</option>
                    <option value="error">Error</option>
                </select>
                <select
                    value={envFilter}
                    onChange={(e) => { setEnvFilter(e.target.value); fetchDeployments(); }}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="all">All Environments</option>
                    <option value="production">Production</option>
                    <option value="staging">Staging</option>
                    <option value="dev">Development</option>
                </select>
            </div>

            {loading ? (
                <div className="flex items-center justify-center h-64">
                    <RefreshCw className="animate-spin" size={32} />
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filteredDeployments.map(deployment => (
                        <DeploymentCard
                            key={deployment.id}
                            deployment={deployment}
                            onSelect={setSelectedDeployment}
                            onEdit={(d) => { /* TODO: Edit modal */ }}
                            onDelete={handleDelete}
                        />
                    ))}
                    {filteredDeployments.length === 0 && (
                        <div className="col-span-full text-center text-gray-500 py-12">
                            {searchTerm ? 'No deployments match your search.' : 'No deployments yet. Create your first one!'}
                        </div>
                    )}
                </div>
            )}

            <CreateDeploymentModal
                isOpen={showCreateModal}
                onClose={() => setShowCreateModal(false)}
                onCreated={() => fetchDeployments()}
            />

            {selectedDeployment && (
                <DeploymentDetails
                    deployment={selectedDeployment}
                    onClose={() => setSelectedDeployment(null)}
                    onRefresh={fetchDeployments}
                />
            )}
        </div>
    );
};

export default DeploymentManager;

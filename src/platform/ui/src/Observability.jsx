import React, { useState, useEffect } from 'react';
import {
    Activity, AlertTriangle, Eye, FileText, Clock, Search, RefreshCw,
    Brain, Shield, Zap, CheckCircle, XCircle, TrendingUp, BarChart3,
    Filter, Download, Play
} from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, Legend
} from 'recharts';
import { fetchWithAuth } from './api';

const COLORS = ['#22c55e', '#ef4444', '#3b82f6', '#eab308', '#8b5cf6'];

const Observability = () => {
    const [activeTab, setActiveTab] = useState('timeline');
    const [blackboxData, setBlackboxData] = useState(null);
    const [vlaStatus, setVlaStatus] = useState(null);
    const [fheAudit, setFheAudit] = useState([]);
    const [incidents, setIncidents] = useState([]);
    const [loading, setLoading] = useState(false);

    // RCA
    const [selectedIncident, setSelectedIncident] = useState(null);
    const [rcaReport, setRcaReport] = useState(null);

    // New: Skill execution data
    const [skillStats, setSkillStats] = useState(null);
    const [skillExecutions, setSkillExecutions] = useState([]);
    const [systemHistory, setSystemHistory] = useState([]);
    const [alerts, setAlerts] = useState([]);

    useEffect(() => {
        if (activeTab === 'blackbox') fetchBlackbox();
        if (activeTab === 'vla') fetchVlaStatus();
        if (activeTab === 'fhe') fetchFheAudit();
        if (activeTab === 'timeline') {
            fetchSkillStats();
            fetchSkillExecutions();
            fetchSystemHistory();
        }
        if (activeTab === 'alerts') fetchAlerts();
    }, [activeTab]);

    // Polling for live data
    useEffect(() => {
        const interval = setInterval(() => {
            if (activeTab === 'vla') fetchVlaStatus();
            if (activeTab === 'blackbox') fetchBlackbox();
            if (activeTab === 'timeline') {
                fetchSkillStats();
                fetchSystemHistory();
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [activeTab]);

    const fetchSkillStats = async () => {
        try {
            const res = await fetchWithAuth('/api/v1/integrator/skill-executions/stats?hours=24');
            const data = await res.json();
            setSkillStats(data);
        } catch (e) {
            console.error("Failed to fetch skill stats", e);
        }
    };

    const fetchSkillExecutions = async () => {
        try {
            const res = await fetchWithAuth('/api/v1/integrator/skill-executions?limit=50');
            const data = await res.json();
            setSkillExecutions(data.executions || []);
        } catch (e) {
            console.error("Failed to fetch skill executions", e);
        }
    };

    const fetchSystemHistory = async () => {
        try {
            const res = await fetchWithAuth('/system/history?limit=50');
            const data = await res.json();
            setSystemHistory(data || []);
        } catch (e) {
            console.error("Failed to fetch system history", e);
        }
    };

    const fetchAlerts = async () => {
        try {
            const res = await fetchWithAuth('/api/v1/integrator/alerts?limit=50');
            const data = await res.json();
            setAlerts(data.alerts || []);
        } catch (e) {
            console.error("Failed to fetch alerts", e);
        }
    };

    const acknowledgeAlert = async (alertId) => {
        try {
            await fetchWithAuth(`/api/v1/integrator/alerts/${alertId}/acknowledge`, { method: 'POST' });
            fetchAlerts();
        } catch (e) {
            console.error("Failed to acknowledge alert", e);
        }
    };

    const resolveAlert = async (alertId) => {
        try {
            await fetchWithAuth(`/api/v1/integrator/alerts/${alertId}/resolve`, { method: 'POST' });
            fetchAlerts();
        } catch (e) {
            console.error("Failed to resolve alert", e);
        }
    };

    const fetchBlackbox = async () => {
        try {
            const res = await fetchWithAuth('/api/observability/blackbox');
            const data = await res.json();
            setBlackboxData(data);
        } catch (e) {
            console.error("Failed to fetch blackbox", e);
        }
    };

    const fetchVlaStatus = async () => {
        try {
            const res = await fetchWithAuth('/api/observability/vla/status');
            const data = await res.json();
            setVlaStatus(data);
        } catch (e) {
            console.error("Failed to fetch VLA status", e);
        }
    };

    const fetchFheAudit = async () => {
        try {
            const res = await fetchWithAuth('/api/observability/fhe/audit');
            const data = await res.json();
            setFheAudit(Array.isArray(data) ? data : []);
        } catch (e) {
            console.error("Failed to fetch FHE audit", e);
        }
    };

    const triggerIncident = async (type, description) => {
        try {
            const res = await fetchWithAuth('/api/observability/incident/trigger', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, description })
            });
            const data = await res.json();
            setIncidents(prev => [{ id: data.incident_id, type, description, time: new Date() }, ...prev]);
            return data.incident_id;
        } catch (e) {
            console.error("Failed to trigger incident", e);
        }
    };

    const analyzeIncident = async (incidentId) => {
        setLoading(true);
        try {
            const res = await fetchWithAuth(`/api/observability/incident/${incidentId}/analyze`);
            const data = await res.json();
            setRcaReport(data);
            setSelectedIncident(incidentId);
        } catch (e) {
            console.error("Failed to analyze incident", e);
        } finally {
            setLoading(false);
        }
    };

    const tabs = [
        { id: 'timeline', label: 'Skill Timeline', icon: TrendingUp },
        { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
        { id: 'blackbox', label: 'Flight Recorder', icon: FileText },
        { id: 'vla', label: 'VLA Model', icon: Brain },
        { id: 'fhe', label: 'FHE Audit', icon: Shield },
        { id: 'rca', label: 'Root Cause Analysis', icon: Search },
    ];

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Eye className="text-green-400" /> Observability & Traceability
                </h1>
                <button
                    onClick={() => triggerIncident('manual', 'Manual incident trigger')}
                    className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded flex items-center gap-2"
                >
                    <AlertTriangle size={18} /> Trigger Incident
                </button>
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

            {/* Skill Timeline */}
            {activeTab === 'timeline' && (
                <div className="space-y-6">
                    {/* Stats Summary */}
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Total Executions (24h)</div>
                            <div className="text-2xl font-bold text-white">{skillStats?.total_executions || 0}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Success Rate</div>
                            <div className={`text-2xl font-bold ${
                                (skillStats?.success_rate || 1) >= 0.95 ? 'text-green-400' :
                                (skillStats?.success_rate || 1) >= 0.8 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                                {((skillStats?.success_rate || 1) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Avg Execution Time</div>
                            <div className="text-2xl font-bold text-blue-400">
                                {(skillStats?.avg_execution_time_ms || 0).toFixed(0)}ms
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Active Skills</div>
                            <div className="text-2xl font-bold text-purple-400">
                                {Object.keys(skillStats?.by_skill || {}).length}
                            </div>
                        </div>
                    </div>

                    {/* Charts Row */}
                    <div className="grid grid-cols-2 gap-4">
                        {/* System Metrics */}
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-3">System Performance (TFLOPS)</h3>
                            <div style={{ height: 200 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={systemHistory.slice(-20)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis
                                            dataKey="timestamp"
                                            tickFormatter={(v) => new Date(v * 1000).toLocaleTimeString()}
                                            stroke="#9ca3af"
                                        />
                                        <YAxis stroke="#9ca3af" />
                                        <Tooltip
                                            labelFormatter={(v) => new Date(v * 1000).toLocaleString()}
                                            contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="tflops_used"
                                            stroke="#3b82f6"
                                            fill="#3b82f6"
                                            fillOpacity={0.3}
                                            name="TFLOPS Used"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Skill Distribution */}
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-3">Skill Usage Distribution</h3>
                            <div style={{ height: 200 }}>
                                {Object.keys(skillStats?.by_skill || {}).length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={Object.entries(skillStats?.by_skill || {}).map(([name, data], i) => ({
                                                    name,
                                                    value: data.count,
                                                    success: data.success
                                                }))}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={40}
                                                outerRadius={70}
                                                paddingAngle={2}
                                                dataKey="value"
                                            >
                                                {Object.entries(skillStats?.by_skill || {}).map((_, i) => (
                                                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                                            <Legend />
                                        </PieChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="flex items-center justify-center h-full text-gray-500">
                                        No skill data available
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Execution Timeline */}
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-3">
                            <h3 className="text-lg font-semibold">Recent Skill Executions</h3>
                            <button
                                onClick={fetchSkillExecutions}
                                className="text-gray-400 hover:text-white"
                            >
                                <RefreshCw size={18} />
                            </button>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead className="bg-gray-900">
                                    <tr>
                                        <th className="text-left p-3">Time</th>
                                        <th className="text-left p-3">Skill</th>
                                        <th className="text-left p-3">Task</th>
                                        <th className="text-left p-3">Mode</th>
                                        <th className="text-left p-3">Duration</th>
                                        <th className="text-left p-3">Confidence</th>
                                        <th className="text-left p-3">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {skillExecutions.slice(0, 15).map((exec, i) => (
                                        <tr key={exec.id || i} className="border-t border-gray-700">
                                            <td className="p-3 font-mono text-gray-400">
                                                {new Date(exec.timestamp * 1000).toLocaleTimeString()}
                                            </td>
                                            <td className="p-3">
                                                <div className="flex items-center gap-2">
                                                    <Zap size={14} className="text-yellow-400" />
                                                    {exec.skill_name || exec.skill_id?.slice(0, 12) || 'Unknown'}
                                                </div>
                                            </td>
                                            <td className="p-3 text-gray-400 max-w-xs truncate">
                                                {exec.task_description || '-'}
                                            </td>
                                            <td className="p-3 capitalize">{exec.execution_mode || '-'}</td>
                                            <td className="p-3">{exec.execution_time_ms?.toFixed(0) || '-'}ms</td>
                                            <td className="p-3">
                                                {exec.confidence ? (
                                                    <span className={
                                                        exec.confidence >= 0.8 ? 'text-green-400' :
                                                        exec.confidence >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                                                    }>
                                                        {(exec.confidence * 100).toFixed(0)}%
                                                    </span>
                                                ) : '-'}
                                            </td>
                                            <td className="p-3">
                                                {exec.success ? (
                                                    <span className="flex items-center gap-1 text-green-400">
                                                        <CheckCircle size={14} /> Success
                                                    </span>
                                                ) : (
                                                    <span className="flex items-center gap-1 text-red-400">
                                                        <XCircle size={14} /> Failed
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                    {skillExecutions.length === 0 && (
                                        <tr>
                                            <td colSpan="7" className="p-8 text-center text-gray-500">
                                                No skill executions recorded yet.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* Alerts */}
            {activeTab === 'alerts' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Total Alerts</div>
                            <div className="text-2xl font-bold text-white">{alerts.length}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Critical</div>
                            <div className="text-2xl font-bold text-red-400">
                                {alerts.filter(a => a.alert_type === 'critical' && !a.resolved).length}
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Unacknowledged</div>
                            <div className="text-2xl font-bold text-yellow-400">
                                {alerts.filter(a => !a.acknowledged).length}
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Resolved</div>
                            <div className="text-2xl font-bold text-green-400">
                                {alerts.filter(a => a.resolved).length}
                            </div>
                        </div>
                    </div>

                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-3">
                            <h3 className="text-lg font-semibold">Alert Feed</h3>
                            <button onClick={fetchAlerts} className="text-gray-400 hover:text-white">
                                <RefreshCw size={18} />
                            </button>
                        </div>
                        <div className="space-y-3 max-h-[500px] overflow-y-auto">
                            {alerts.map(alert => (
                                <div
                                    key={alert.id}
                                    className={`p-4 rounded-lg border ${
                                        alert.resolved ? 'bg-gray-900 border-gray-700 opacity-60' :
                                        alert.alert_type === 'critical' ? 'bg-red-900/30 border-red-700' :
                                        alert.alert_type === 'error' ? 'bg-red-900/20 border-red-800' :
                                        alert.alert_type === 'warning' ? 'bg-yellow-900/20 border-yellow-700' :
                                        'bg-blue-900/20 border-blue-700'
                                    }`}
                                >
                                    <div className="flex justify-between items-start mb-2">
                                        <div className="flex items-center gap-2">
                                            {alert.alert_type === 'critical' && <AlertTriangle className="text-red-400" size={18} />}
                                            {alert.alert_type === 'error' && <XCircle className="text-red-400" size={18} />}
                                            {alert.alert_type === 'warning' && <AlertTriangle className="text-yellow-400" size={18} />}
                                            {alert.alert_type === 'info' && <Activity className="text-blue-400" size={18} />}
                                            <span className="font-semibold">{alert.title}</span>
                                            <span className={`px-2 py-0.5 rounded text-xs ${
                                                alert.alert_type === 'critical' ? 'bg-red-900 text-red-300' :
                                                alert.alert_type === 'error' ? 'bg-red-800 text-red-300' :
                                                alert.alert_type === 'warning' ? 'bg-yellow-800 text-yellow-300' :
                                                'bg-blue-800 text-blue-300'
                                            }`}>
                                                {alert.alert_type}
                                            </span>
                                            <span className="bg-gray-700 px-2 py-0.5 rounded text-xs">
                                                {alert.category}
                                            </span>
                                        </div>
                                        <span className="text-gray-500 text-sm">
                                            {new Date(alert.timestamp * 1000).toLocaleString()}
                                        </span>
                                    </div>
                                    <p className="text-gray-300 mb-3">{alert.message}</p>
                                    <div className="flex gap-2">
                                        {!alert.acknowledged && (
                                            <button
                                                onClick={() => acknowledgeAlert(alert.id)}
                                                className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm"
                                            >
                                                Acknowledge
                                            </button>
                                        )}
                                        {!alert.resolved && (
                                            <button
                                                onClick={() => resolveAlert(alert.id)}
                                                className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm"
                                            >
                                                Resolve
                                            </button>
                                        )}
                                        {alert.resolved && (
                                            <span className="text-green-400 text-sm flex items-center gap-1">
                                                <CheckCircle size={14} /> Resolved
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {alerts.length === 0 && (
                                <div className="text-center text-gray-500 py-8">
                                    No alerts. System is healthy.
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Blackbox / Flight Recorder */}
            {activeTab === 'blackbox' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-3 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Recording Status</div>
                            <div className="text-2xl font-bold text-green-400">ACTIVE</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Events Recorded</div>
                            <div className="text-2xl font-bold">{blackboxData?.events?.length || 0}</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Last Update</div>
                            <div className="text-2xl font-bold">
                                {blackboxData?.timestamp ? new Date(blackboxData.timestamp * 1000).toLocaleTimeString() : '--'}
                            </div>
                        </div>
                    </div>

                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-3">Recent Events</h3>
                        <div className="bg-gray-900 rounded p-4 h-64 overflow-y-auto font-mono text-sm">
                            {blackboxData?.events?.slice(-20).reverse().map((event, i) => (
                                <div key={i} className="mb-2 pb-2 border-b border-gray-800">
                                    <span className="text-gray-500">[{new Date(event.timestamp * 1000).toLocaleTimeString()}]</span>
                                    <span className={`ml-2 ${
                                        event.level === 'error' ? 'text-red-400' :
                                        event.level === 'warning' ? 'text-yellow-400' : 'text-gray-300'
                                    }`}>
                                        {event.component}: {event.message}
                                    </span>
                                </div>
                            )) || <div className="text-gray-500">No events recorded</div>}
                        </div>
                    </div>

                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-3">VLA State Snapshot</h3>
                        <pre className="bg-gray-900 rounded p-4 overflow-auto text-sm">
                            {JSON.stringify(blackboxData?.vla_state, null, 2) || 'No VLA state'}
                        </pre>
                    </div>
                </div>
            )}

            {/* VLA Model Status */}
            {activeTab === 'vla' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Model Loaded</div>
                            <div className={`text-2xl font-bold ${vlaStatus?.model_loaded ? 'text-green-400' : 'text-red-400'}`}>
                                {vlaStatus?.model_loaded ? 'YES' : 'NO'}
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Confidence</div>
                            <div className="text-2xl font-bold text-blue-400">
                                {((vlaStatus?.confidence || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Inference Latency</div>
                            <div className="text-2xl font-bold">
                                {vlaStatus?.inference_latency_ms?.toFixed(1) || '--'} ms
                            </div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Attention Active</div>
                            <div className={`text-2xl font-bold ${vlaStatus?.attention_map ? 'text-green-400' : 'text-gray-400'}`}>
                                {vlaStatus?.attention_map ? 'YES' : 'NO'}
                            </div>
                        </div>
                    </div>

                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-3">VLA Attention Visualization</h3>
                        <div className="bg-gray-900 rounded p-4 h-64 flex items-center justify-center">
                            {vlaStatus?.attention_map ? (
                                <div className="text-center">
                                    <div className="text-green-400 mb-2">Attention Map Available</div>
                                    <div className="text-gray-400 text-sm">
                                        Shape: {JSON.stringify(vlaStatus.attention_map.shape || 'N/A')}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-gray-500">
                                    No attention data available. Run VLA inference to see attention maps.
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-3">Base Model Info (FROZEN)</h3>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <span className="text-gray-400">Model: </span>
                                <span className="font-mono">Pi0 / OpenVLA-7B</span>
                            </div>
                            <div>
                                <span className="text-gray-400">Mode: </span>
                                <span className="text-yellow-400">READ-ONLY (No gradient updates)</span>
                            </div>
                            <div>
                                <span className="text-gray-400">Skills Augmentation: </span>
                                <span className="text-green-400">ENABLED (MoE)</span>
                            </div>
                            <div>
                                <span className="text-gray-400">Encryption: </span>
                                <span className="text-blue-400">N2HE 128-bit</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* FHE Audit */}
            {activeTab === 'fhe' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-3 gap-4">
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Encryption Backend</div>
                            <div className="text-2xl font-bold text-purple-400">N2HE</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Security Level</div>
                            <div className="text-2xl font-bold">128-bit</div>
                        </div>
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <div className="text-gray-400 text-sm">Audit Entries</div>
                            <div className="text-2xl font-bold">{fheAudit.length}</div>
                        </div>
                    </div>

                    <div className="bg-gray-800 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-3">Encryption Audit Log</h3>
                        <div className="bg-gray-900 rounded overflow-hidden">
                            <table className="w-full text-sm">
                                <thead className="bg-gray-800">
                                    <tr>
                                        <th className="text-left p-3">Timestamp</th>
                                        <th className="text-left p-3">Operation</th>
                                        <th className="text-left p-3">Data Type</th>
                                        <th className="text-left p-3">Size</th>
                                        <th className="text-left p-3">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {fheAudit.slice(0, 20).map((entry, i) => (
                                        <tr key={i} className="border-t border-gray-800">
                                            <td className="p-3 font-mono text-gray-400">
                                                {new Date(entry.timestamp * 1000).toLocaleString()}
                                            </td>
                                            <td className="p-3">{entry.operation}</td>
                                            <td className="p-3">{entry.data_type}</td>
                                            <td className="p-3">{entry.size_bytes} bytes</td>
                                            <td className="p-3">
                                                <span className={`px-2 py-1 rounded text-xs ${
                                                    entry.status === 'success' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
                                                }`}>
                                                    {entry.status}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                    {fheAudit.length === 0 && (
                                        <tr>
                                            <td colSpan="5" className="p-8 text-center text-gray-500">
                                                No audit entries. FHE operations will appear here.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* Root Cause Analysis */}
            {activeTab === 'rca' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-6">
                        {/* Incidents List */}
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-3">Recent Incidents</h3>
                            <div className="space-y-2 max-h-96 overflow-y-auto">
                                {incidents.map((incident, i) => (
                                    <div
                                        key={incident.id}
                                        onClick={() => analyzeIncident(incident.id)}
                                        className={`p-3 rounded cursor-pointer transition ${
                                            selectedIncident === incident.id
                                                ? 'bg-blue-900 border border-blue-500'
                                                : 'bg-gray-900 hover:bg-gray-700'
                                        }`}
                                    >
                                        <div className="flex justify-between">
                                            <span className="font-semibold">{incident.type}</span>
                                            <span className="text-gray-400 text-sm">
                                                {incident.time.toLocaleTimeString()}
                                            </span>
                                        </div>
                                        <div className="text-gray-400 text-sm">{incident.description}</div>
                                        <div className="text-xs text-gray-500 mt-1 font-mono">{incident.id}</div>
                                    </div>
                                ))}
                                {incidents.length === 0 && (
                                    <div className="text-gray-500 text-center py-8">
                                        No incidents recorded. Trigger an incident to test RCA.
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* RCA Report */}
                        <div className="bg-gray-800 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-3">Analysis Report</h3>
                            {loading ? (
                                <div className="flex items-center justify-center h-64">
                                    <RefreshCw className="animate-spin" size={32} />
                                </div>
                            ) : rcaReport ? (
                                <div className="space-y-4">
                                    <div className={`p-3 rounded ${
                                        rcaReport.status === 'critical' ? 'bg-red-900' :
                                        rcaReport.status === 'warning' ? 'bg-yellow-900' : 'bg-green-900'
                                    }`}>
                                        <div className="font-semibold">Status: {rcaReport.status?.toUpperCase()}</div>
                                    </div>
                                    <div>
                                        <div className="text-gray-400 text-sm mb-1">Root Cause</div>
                                        <div className="bg-gray-900 p-3 rounded">{rcaReport.root_cause || 'Unknown'}</div>
                                    </div>
                                    <div>
                                        <div className="text-gray-400 text-sm mb-1">Recommendations</div>
                                        <ul className="bg-gray-900 p-3 rounded list-disc list-inside">
                                            {rcaReport.recommendations?.map((rec, i) => (
                                                <li key={i}>{rec}</li>
                                            )) || <li>No recommendations</li>}
                                        </ul>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-gray-500 text-center py-8">
                                    Select an incident to view analysis.
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Observability;

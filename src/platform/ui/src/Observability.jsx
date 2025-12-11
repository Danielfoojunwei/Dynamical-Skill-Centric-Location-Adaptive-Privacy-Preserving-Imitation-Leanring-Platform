import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, Eye, FileText, Clock, Search, RefreshCw, Brain, Shield, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { fetchWithAuth } from './api';

const Observability = () => {
    const [activeTab, setActiveTab] = useState('blackbox');
    const [blackboxData, setBlackboxData] = useState(null);
    const [vlaStatus, setVlaStatus] = useState(null);
    const [fheAudit, setFheAudit] = useState([]);
    const [incidents, setIncidents] = useState([]);
    const [loading, setLoading] = useState(false);

    // RCA
    const [selectedIncident, setSelectedIncident] = useState(null);
    const [rcaReport, setRcaReport] = useState(null);

    useEffect(() => {
        if (activeTab === 'blackbox') fetchBlackbox();
        if (activeTab === 'vla') fetchVlaStatus();
        if (activeTab === 'fhe') fetchFheAudit();
    }, [activeTab]);

    // Polling for live data
    useEffect(() => {
        const interval = setInterval(() => {
            if (activeTab === 'vla') fetchVlaStatus();
            if (activeTab === 'blackbox') fetchBlackbox();
        }, 2000);
        return () => clearInterval(interval);
    }, [activeTab]);

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

/**
 * Audit Log - Complete Change History for System Integrators
 *
 * Tracks all system changes including:
 * - Configuration changes
 * - Skill deployments
 * - Safety updates
 * - User actions
 *
 * @version 0.7.0
 */

import React, { useState, useEffect } from 'react';
import {
    History, Filter, Search, RefreshCw, Download, ChevronDown,
    ChevronRight, User, Settings, Zap, Shield, Server, Clock,
    CheckCircle, XCircle, AlertTriangle, FileText, GitBranch
} from 'lucide-react';
import { fetchWithAuth } from './api';

const actionIcons = {
    create: <CheckCircle size={14} className="text-green-400" />,
    update: <Settings size={14} className="text-blue-400" />,
    delete: <XCircle size={14} className="text-red-400" />,
    invoke: <Zap size={14} className="text-yellow-400" />,
    rollback: <GitBranch size={14} className="text-purple-400" />,
    apply: <CheckCircle size={14} className="text-blue-400" />,
};

const resourceIcons = {
    skill: <Zap size={14} className="text-yellow-400" />,
    deployment: <Server size={14} className="text-blue-400" />,
    config_version: <FileText size={14} className="text-purple-400" />,
    safety: <Shield size={14} className="text-red-400" />,
    preset: <Settings size={14} className="text-green-400" />,
};

const AuditLogEntry = ({ log, expanded, onToggle }) => {
    const timeString = new Date(log.timestamp * 1000).toLocaleString();
    const hasDetails = log.old_value || log.new_value;

    return (
        <div className="bg-gray-800 rounded-lg border border-gray-700 mb-2">
            <div
                className={`p-4 flex items-center gap-4 ${hasDetails ? 'cursor-pointer hover:bg-gray-750' : ''}`}
                onClick={() => hasDetails && onToggle(log.id)}
            >
                <div className="flex-shrink-0">
                    {hasDetails ? (
                        expanded ? <ChevronDown size={16} className="text-gray-400" /> :
                            <ChevronRight size={16} className="text-gray-400" />
                    ) : <div className="w-4" />}
                </div>

                <div className="flex items-center gap-2 w-24">
                    {actionIcons[log.action] || <Settings size={14} className="text-gray-400" />}
                    <span className="text-sm font-medium capitalize">{log.action}</span>
                </div>

                <div className="flex items-center gap-2 w-32">
                    {resourceIcons[log.resource_type] || <FileText size={14} className="text-gray-400" />}
                    <span className="text-sm text-gray-400 capitalize">{log.resource_type.replace('_', ' ')}</span>
                </div>

                <div className="flex-1">
                    <span className="text-white">{log.resource_name || log.resource_id || 'N/A'}</span>
                </div>

                <div className="flex items-center gap-2 text-gray-400 text-sm">
                    <User size={14} />
                    <span>{log.user_id}</span>
                </div>

                <div className="flex items-center gap-2 text-gray-500 text-sm w-48 justify-end">
                    <Clock size={14} />
                    <span>{timeString}</span>
                </div>

                <div className="w-6 flex justify-center">
                    {log.success ? (
                        <CheckCircle size={16} className="text-green-400" />
                    ) : (
                        <XCircle size={16} className="text-red-400" />
                    )}
                </div>
            </div>

            {expanded && hasDetails && (
                <div className="px-4 pb-4 border-t border-gray-700 pt-3">
                    <div className="grid grid-cols-2 gap-4">
                        {log.old_value && (
                            <div>
                                <h4 className="text-gray-400 text-sm mb-2">Previous Value</h4>
                                <pre className="bg-gray-900 p-3 rounded text-xs overflow-auto max-h-48">
                                    {JSON.stringify(log.old_value, null, 2)}
                                </pre>
                            </div>
                        )}
                        {log.new_value && (
                            <div>
                                <h4 className="text-gray-400 text-sm mb-2">New Value</h4>
                                <pre className="bg-gray-900 p-3 rounded text-xs overflow-auto max-h-48">
                                    {JSON.stringify(log.new_value, null, 2)}
                                </pre>
                            </div>
                        )}
                    </div>
                    {log.error_message && (
                        <div className="mt-3 bg-red-900/30 border border-red-700 rounded p-3">
                            <span className="text-red-400 text-sm">{log.error_message}</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const AuditLog = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [summary, setSummary] = useState(null);
    const [expandedLog, setExpandedLog] = useState(null);

    // Filters
    const [searchTerm, setSearchTerm] = useState('');
    const [resourceFilter, setResourceFilter] = useState('all');
    const [actionFilter, setActionFilter] = useState('all');
    const [hoursFilter, setHoursFilter] = useState('24');
    const [offset, setOffset] = useState(0);
    const limit = 50;

    useEffect(() => {
        fetchLogs();
        fetchSummary();
    }, [resourceFilter, actionFilter, hoursFilter, offset]);

    const fetchLogs = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams();
            if (resourceFilter !== 'all') params.append('resource_type', resourceFilter);
            if (actionFilter !== 'all') params.append('action', actionFilter);
            if (hoursFilter !== 'all') {
                const cutoff = Date.now() / 1000 - parseInt(hoursFilter) * 3600;
                params.append('start_time', cutoff);
            }
            params.append('limit', limit);
            params.append('offset', offset);

            const res = await fetchWithAuth(`/api/v1/integrator/audit-logs?${params}`);
            const data = await res.json();
            setLogs(data.logs || []);
        } catch (e) {
            console.error('Failed to fetch logs', e);
        } finally {
            setLoading(false);
        }
    };

    const fetchSummary = async () => {
        try {
            const res = await fetchWithAuth(`/api/v1/integrator/audit-logs/summary?hours=${hoursFilter === 'all' ? 168 : hoursFilter}`);
            const data = await res.json();
            setSummary(data);
        } catch (e) {
            console.error('Failed to fetch summary', e);
        }
    };

    const exportLogs = () => {
        const csv = [
            ['Timestamp', 'Action', 'Resource Type', 'Resource Name', 'User', 'Success'].join(','),
            ...logs.map(log => [
                new Date(log.timestamp * 1000).toISOString(),
                log.action,
                log.resource_type,
                log.resource_name || log.resource_id || '',
                log.user_id,
                log.success ? 'Yes' : 'No'
            ].join(','))
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audit-log-${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
    };

    const filteredLogs = logs.filter(log =>
        (log.resource_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
        (log.resource_id || '').toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <History className="text-purple-400" /> Audit Log
                    </h1>
                    <p className="text-gray-400 mt-1">Complete history of all system changes</p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={exportLogs}
                        className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Download size={18} /> Export CSV
                    </button>
                    <button
                        onClick={() => { fetchLogs(); fetchSummary(); }}
                        className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded"
                        disabled={loading}
                    >
                        <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
                    </button>
                </div>
            </div>

            {summary && (
                <div className="grid grid-cols-5 gap-4 mb-6">
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Total Actions</div>
                        <div className="text-2xl font-bold text-white">{summary.total_actions}</div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Success Rate</div>
                        <div className="text-2xl font-bold text-green-400">
                            {(summary.success_rate * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Top Action</div>
                        <div className="text-xl font-bold text-blue-400 capitalize">
                            {Object.entries(summary.by_action || {}).sort((a, b) => b[1] - a[1])[0]?.[0] || '-'}
                        </div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Top Resource</div>
                        <div className="text-xl font-bold text-purple-400 capitalize">
                            {Object.entries(summary.by_resource || {}).sort((a, b) => b[1] - a[1])[0]?.[0]?.replace('_', ' ') || '-'}
                        </div>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                        <div className="text-gray-400 text-sm">Active Users</div>
                        <div className="text-2xl font-bold text-yellow-400">
                            {Object.keys(summary.by_user || {}).length}
                        </div>
                    </div>
                </div>
            )}

            <div className="flex gap-4 mb-4">
                <div className="relative flex-1">
                    <Search size={18} className="absolute left-3 top-2.5 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search by resource name..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-gray-800 border border-gray-700 rounded pl-10 pr-4 py-2"
                    />
                </div>
                <select
                    value={actionFilter}
                    onChange={(e) => setActionFilter(e.target.value)}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="all">All Actions</option>
                    <option value="create">Create</option>
                    <option value="update">Update</option>
                    <option value="delete">Delete</option>
                    <option value="invoke">Invoke</option>
                    <option value="rollback">Rollback</option>
                </select>
                <select
                    value={resourceFilter}
                    onChange={(e) => setResourceFilter(e.target.value)}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="all">All Resources</option>
                    <option value="deployment">Deployment</option>
                    <option value="skill">Skill</option>
                    <option value="config_version">Config Version</option>
                    <option value="preset">Preset</option>
                    <option value="safety">Safety</option>
                </select>
                <select
                    value={hoursFilter}
                    onChange={(e) => setHoursFilter(e.target.value)}
                    className="bg-gray-800 border border-gray-700 rounded px-4 py-2"
                >
                    <option value="1">Last Hour</option>
                    <option value="24">Last 24 Hours</option>
                    <option value="168">Last Week</option>
                    <option value="720">Last 30 Days</option>
                    <option value="all">All Time</option>
                </select>
            </div>

            {loading ? (
                <div className="flex items-center justify-center h-64">
                    <RefreshCw className="animate-spin" size={32} />
                </div>
            ) : (
                <div>
                    {filteredLogs.map(log => (
                        <AuditLogEntry
                            key={log.id}
                            log={log}
                            expanded={expandedLog === log.id}
                            onToggle={(id) => setExpandedLog(expandedLog === id ? null : id)}
                        />
                    ))}
                    {filteredLogs.length === 0 && (
                        <div className="text-center text-gray-500 py-12">
                            No audit entries found for the selected filters.
                        </div>
                    )}

                    {logs.length >= limit && (
                        <div className="flex justify-center gap-4 mt-6">
                            <button
                                onClick={() => setOffset(Math.max(0, offset - limit))}
                                disabled={offset === 0}
                                className="bg-gray-700 hover:bg-gray-600 disabled:opacity-50 px-4 py-2 rounded"
                            >
                                Previous
                            </button>
                            <span className="py-2 text-gray-400">
                                Showing {offset + 1} - {offset + logs.length}
                            </span>
                            <button
                                onClick={() => setOffset(offset + limit)}
                                className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
                            >
                                Next
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default AuditLog;

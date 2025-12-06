import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function CloudIntegration() {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState([]);
    const [apiKey, setApiKey] = useState(localStorage.getItem('api_key') || '');

    useEffect(() => {
        fetchStatus();
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch('/cloud/status', {
                headers: { 'X-API-Key': apiKey }
            });
            if (res.ok) {
                const data = await res.json();
                setStatus(data);
            }
        } catch (err) {
            console.error("Failed to fetch status", err);
        }
    };

    const handleSync = async () => {
        setLoading(true);
        try {
            const res = await fetch('/cloud/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey
                },
                body: JSON.stringify({ current_version: status?.version || "v1.0.0" })
            });
            const data = await res.json();
            addLog(`Sync Check: ${data.status} ${data.version ? '(' + data.version + ')' : ''}`);
            fetchStatus();
        } catch (err) {
            addLog(`Sync Failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async () => {
        setLoading(true);
        try {
            const res = await fetch('/cloud/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey
                },
                body: JSON.stringify({ gradient_id: "latest" })
            });
            const data = await res.json();
            addLog(`Upload: ${data.status} (${data.bytes || 0} bytes encrypted)`);
        } catch (err) {
            addLog(`Upload Failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const addLog = (msg) => {
        setLogs(prev => [{ time: new Date().toLocaleTimeString(), msg }, ...prev]);
    };

    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-6">Cloud Integration & Value Chain</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Status Card */}
                <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 className="text-xl font-semibold mb-4">Connection Status</h2>
                    {status ? (
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Provider:</span>
                                <span className="font-mono text-green-400">{status.ffm_provider}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Status:</span>
                                <span className={`font-bold ${status.connection === 'connected' ? 'text-green-500' : 'text-red-500'}`}>
                                    {status.connection.toUpperCase()}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Last Sync:</span>
                                <span>{new Date(status.last_sync * 1000).toLocaleString()}</span>
                            </div>
                        </div>
                    ) : (
                        <p className="text-gray-500">Loading status...</p>
                    )}
                </div>

                {/* Actions Card */}
                <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 className="text-xl font-semibold mb-4">Actions</h2>
                    <div className="space-y-4">
                        <button
                            onClick={handleSync}
                            disabled={loading}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded disabled:opacity-50 transition"
                        >
                            {loading ? 'Processing...' : 'Check for Model Updates'}
                        </button>
                        <button
                            onClick={handleUpload}
                            disabled={loading}
                            className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded disabled:opacity-50 transition"
                        >
                            {loading ? 'Processing...' : 'Upload Encrypted Gradients (FHE)'}
                        </button>
                    </div>
                    <p className="mt-4 text-xs text-gray-400">
                        * Gradient uploads are encrypted using MOAI (N2HE) and cannot be decrypted by the server.
                    </p>
                </div>
            </div>

            {/* Traceability Logs */}
            <div className="mt-8 bg-gray-800 p-6 rounded-lg shadow-lg">
                <h2 className="text-xl font-semibold mb-4">Traceability Logs</h2>
                <div className="bg-gray-900 p-4 rounded h-64 overflow-y-auto font-mono text-sm">
                    {logs.length === 0 ? (
                        <p className="text-gray-600">No activity recorded.</p>
                    ) : (
                        logs.map((log, i) => (
                            <div key={i} className="mb-1 border-b border-gray-800 pb-1">
                                <span className="text-gray-500">[{log.time}]</span> <span className="text-gray-300">{log.msg}</span>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}

export default CloudIntegration;

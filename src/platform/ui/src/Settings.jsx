/**
 * Enhanced Settings Page - System Configuration for Integrators
 *
 * Features:
 * - System configuration
 * - Preset management
 * - Notification settings
 * - Quick actions
 *
 * @version 0.7.0
 */

import React, { useState, useEffect } from 'react';
import {
    Settings, Save, RefreshCw, Plus, Trash2, Copy, Download,
    Upload, Bell, Mail, Wifi, Camera, Shield, Zap, Check,
    AlertTriangle, ChevronDown, ChevronRight
} from 'lucide-react';
import { fetchWithAuth } from './api';

const SettingsSection = ({ title, description, icon: Icon, children, defaultOpen = true }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className="bg-gray-800 rounded-lg border border-gray-700 mb-4">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full p-4 flex items-center justify-between text-left"
            >
                <div className="flex items-center gap-3">
                    {Icon && <Icon size={20} className="text-blue-400" />}
                    <div>
                        <h3 className="font-semibold text-white">{title}</h3>
                        {description && <p className="text-gray-400 text-sm">{description}</p>}
                    </div>
                </div>
                {isOpen ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {isOpen && (
                <div className="px-4 pb-4 border-t border-gray-700 pt-4">
                    {children}
                </div>
            )}
        </div>
    );
};

const SettingsPage = () => {
    const [settings, setSettings] = useState({
        camera_rtsp_url: '',
    });
    const [presets, setPresets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState({ type: '', text: '' });
    const [activePreset, setActivePreset] = useState(null);

    // Notification settings (local state for demo)
    const [notifications, setNotifications] = useState({
        email_enabled: false,
        email_address: '',
        alert_on_error: true,
        alert_on_safety: true,
        alert_on_skill_failure: true,
        daily_digest: false,
    });

    // New preset form
    const [showNewPreset, setShowNewPreset] = useState(false);
    const [newPresetForm, setNewPresetForm] = useState({
        name: '',
        description: '',
        preset_type: 'full',
        tags: '',
    });

    useEffect(() => {
        loadSettings();
        loadPresets();
    }, []);

    const loadSettings = async () => {
        try {
            const res = await fetchWithAuth('/api/settings');
            if (res.ok) {
                const data = await res.json();
                setSettings(data);
            }
        } catch (e) {
            console.error("Failed to load settings", e);
        }
    };

    const loadPresets = async () => {
        try {
            const res = await fetchWithAuth('/api/v1/integrator/presets');
            if (res.ok) {
                const data = await res.json();
                setPresets(data.presets || []);
            }
        } catch (e) {
            console.error("Failed to load presets", e);
        }
    };

    const handleSaveSettings = async () => {
        setLoading(true);
        setMessage({ type: '', text: '' });
        try {
            const res = await fetchWithAuth('/api/settings', {
                method: 'POST',
                body: JSON.stringify(settings)
            });
            if (res.ok) {
                setMessage({ type: 'success', text: 'Settings saved successfully' });
            } else {
                setMessage({ type: 'error', text: 'Failed to save settings' });
            }
        } catch (e) {
            setMessage({ type: 'error', text: 'Error saving settings' });
        } finally {
            setLoading(false);
        }
    };

    const handleCreatePreset = async () => {
        if (!newPresetForm.name.trim()) {
            setMessage({ type: 'error', text: 'Preset name is required' });
            return;
        }

        setLoading(true);
        try {
            const res = await fetchWithAuth('/api/v1/integrator/presets', {
                method: 'POST',
                body: JSON.stringify({
                    name: newPresetForm.name,
                    description: newPresetForm.description,
                    preset_type: newPresetForm.preset_type,
                    config_data: {
                        ...settings,
                        notifications,
                    },
                    tags: newPresetForm.tags.split(',').map(t => t.trim()).filter(Boolean),
                })
            });
            const data = await res.json();
            if (data.success) {
                setMessage({ type: 'success', text: `Preset "${newPresetForm.name}" created` });
                setShowNewPreset(false);
                setNewPresetForm({ name: '', description: '', preset_type: 'full', tags: '' });
                loadPresets();
            } else {
                setMessage({ type: 'error', text: data.error || 'Failed to create preset' });
            }
        } catch (e) {
            setMessage({ type: 'error', text: 'Error creating preset' });
        } finally {
            setLoading(false);
        }
    };

    const handleApplyPreset = async (preset) => {
        if (!confirm(`Apply preset "${preset.name}"? This will update your current settings.`)) return;

        setLoading(true);
        try {
            const res = await fetchWithAuth(`/api/v1/integrator/presets/${preset.id}/apply`, {
                method: 'POST'
            });
            const data = await res.json();
            if (data.success) {
                setMessage({ type: 'success', text: `Preset "${preset.name}" applied` });
                setActivePreset(preset.id);
                loadSettings();
            } else {
                setMessage({ type: 'error', text: 'Failed to apply preset' });
            }
        } catch (e) {
            setMessage({ type: 'error', text: 'Error applying preset' });
        } finally {
            setLoading(false);
        }
    };

    const handleDeletePreset = async (preset) => {
        if (!confirm(`Delete preset "${preset.name}"?`)) return;

        try {
            await fetchWithAuth(`/api/v1/integrator/presets/${preset.id}`, { method: 'DELETE' });
            setMessage({ type: 'success', text: 'Preset deleted' });
            loadPresets();
        } catch (e) {
            setMessage({ type: 'error', text: 'Failed to delete preset' });
        }
    };

    const handleExportConfig = () => {
        const config = {
            settings,
            notifications,
            exported_at: new Date().toISOString(),
            version: '0.7.0',
        };
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dynamical-config-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
    };

    const handleImportConfig = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const config = JSON.parse(event.target.result);
                if (config.settings) setSettings(config.settings);
                if (config.notifications) setNotifications(config.notifications);
                setMessage({ type: 'success', text: 'Configuration imported' });
            } catch (err) {
                setMessage({ type: 'error', text: 'Invalid configuration file' });
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="p-6 max-w-4xl mx-auto">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Settings className="text-blue-400" /> System Settings
                    </h1>
                    <p className="text-gray-400 mt-1">Configure your Dynamical Edge platform</p>
                </div>
                <div className="flex gap-2">
                    <label className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center gap-2 cursor-pointer">
                        <Upload size={18} /> Import
                        <input type="file" accept=".json" onChange={handleImportConfig} className="hidden" />
                    </label>
                    <button
                        onClick={handleExportConfig}
                        className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Download size={18} /> Export
                    </button>
                </div>
            </div>

            {message.text && (
                <div className={`mb-4 p-4 rounded-lg flex items-center gap-2 ${
                    message.type === 'success' ? 'bg-green-900/30 border border-green-700 text-green-400' :
                    'bg-red-900/30 border border-red-700 text-red-400'
                }`}>
                    {message.type === 'success' ? <Check size={18} /> : <AlertTriangle size={18} />}
                    {message.text}
                </div>
            )}

            {/* Camera Configuration */}
            <SettingsSection
                title="Camera Configuration"
                description="Configure RTSP camera streams"
                icon={Camera}
            >
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Primary Camera RTSP URL</label>
                        <input
                            type="text"
                            value={settings.camera_rtsp_url}
                            onChange={(e) => setSettings({ ...settings, camera_rtsp_url: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2"
                            placeholder="rtsp://192.168.1.100:554/stream"
                        />
                    </div>
                    <button
                        onClick={handleSaveSettings}
                        disabled={loading}
                        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Save size={18} />
                        {loading ? 'Saving...' : 'Save Camera Settings'}
                    </button>
                </div>
            </SettingsSection>

            {/* Notification Settings */}
            <SettingsSection
                title="Notifications"
                description="Configure alert notifications"
                icon={Bell}
            >
                <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-gray-900 rounded">
                        <div>
                            <div className="font-medium">Email Notifications</div>
                            <div className="text-gray-400 text-sm">Receive alerts via email</div>
                        </div>
                        <button
                            onClick={() => setNotifications({ ...notifications, email_enabled: !notifications.email_enabled })}
                            className={`w-12 h-6 rounded-full transition ${
                                notifications.email_enabled ? 'bg-blue-600' : 'bg-gray-600'
                            }`}
                        >
                            <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                                notifications.email_enabled ? 'translate-x-6' : 'translate-x-0.5'
                            }`} />
                        </button>
                    </div>

                    {notifications.email_enabled && (
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Email Address</label>
                            <input
                                type="email"
                                value={notifications.email_address}
                                onChange={(e) => setNotifications({ ...notifications, email_address: e.target.value })}
                                className="w-full bg-gray-900 border border-gray-700 rounded px-4 py-2"
                                placeholder="alerts@yourcompany.com"
                            />
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-4">
                        <label className="flex items-center gap-2 p-3 bg-gray-900 rounded cursor-pointer">
                            <input
                                type="checkbox"
                                checked={notifications.alert_on_error}
                                onChange={(e) => setNotifications({ ...notifications, alert_on_error: e.target.checked })}
                                className="w-4 h-4"
                            />
                            <span>Alert on System Errors</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-gray-900 rounded cursor-pointer">
                            <input
                                type="checkbox"
                                checked={notifications.alert_on_safety}
                                onChange={(e) => setNotifications({ ...notifications, alert_on_safety: e.target.checked })}
                                className="w-4 h-4"
                            />
                            <span>Alert on Safety Events</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-gray-900 rounded cursor-pointer">
                            <input
                                type="checkbox"
                                checked={notifications.alert_on_skill_failure}
                                onChange={(e) => setNotifications({ ...notifications, alert_on_skill_failure: e.target.checked })}
                                className="w-4 h-4"
                            />
                            <span>Alert on Skill Failures</span>
                        </label>
                        <label className="flex items-center gap-2 p-3 bg-gray-900 rounded cursor-pointer">
                            <input
                                type="checkbox"
                                checked={notifications.daily_digest}
                                onChange={(e) => setNotifications({ ...notifications, daily_digest: e.target.checked })}
                                className="w-4 h-4"
                            />
                            <span>Daily Digest Email</span>
                        </label>
                    </div>
                </div>
            </SettingsSection>

            {/* Configuration Presets */}
            <SettingsSection
                title="Configuration Presets"
                description="Save and apply configuration templates"
                icon={Copy}
            >
                <div className="space-y-4">
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400 text-sm">{presets.length} presets saved</span>
                        <button
                            onClick={() => setShowNewPreset(true)}
                            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded flex items-center gap-2"
                        >
                            <Plus size={18} /> New Preset
                        </button>
                    </div>

                    {showNewPreset && (
                        <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                            <h4 className="font-semibold mb-3">Create New Preset</h4>
                            <div className="grid grid-cols-2 gap-4 mb-4">
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">Name *</label>
                                    <input
                                        type="text"
                                        value={newPresetForm.name}
                                        onChange={(e) => setNewPresetForm({ ...newPresetForm, name: e.target.value })}
                                        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                                        placeholder="Production Default"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">Type</label>
                                    <select
                                        value={newPresetForm.preset_type}
                                        onChange={(e) => setNewPresetForm({ ...newPresetForm, preset_type: e.target.value })}
                                        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                                    >
                                        <option value="full">Full Configuration</option>
                                        <option value="partial">Partial</option>
                                        <option value="template">Template</option>
                                    </select>
                                </div>
                            </div>
                            <div className="mb-4">
                                <label className="block text-sm text-gray-400 mb-1">Description</label>
                                <input
                                    type="text"
                                    value={newPresetForm.description}
                                    onChange={(e) => setNewPresetForm({ ...newPresetForm, description: e.target.value })}
                                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                                    placeholder="Standard production configuration"
                                />
                            </div>
                            <div className="mb-4">
                                <label className="block text-sm text-gray-400 mb-1">Tags (comma separated)</label>
                                <input
                                    type="text"
                                    value={newPresetForm.tags}
                                    onChange={(e) => setNewPresetForm({ ...newPresetForm, tags: e.target.value })}
                                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                                    placeholder="production, default"
                                />
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={handleCreatePreset}
                                    disabled={loading}
                                    className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
                                >
                                    {loading ? 'Creating...' : 'Create Preset'}
                                </button>
                                <button
                                    onClick={() => setShowNewPreset(false)}
                                    className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    )}

                    <div className="space-y-2">
                        {presets.map(preset => (
                            <div
                                key={preset.id}
                                className={`p-4 rounded-lg border ${
                                    activePreset === preset.id
                                        ? 'bg-blue-900/20 border-blue-700'
                                        : 'bg-gray-900 border-gray-700'
                                }`}
                            >
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <span className="font-semibold">{preset.name}</span>
                                            {preset.is_default && (
                                                <span className="bg-blue-900 text-blue-400 px-2 py-0.5 rounded text-xs">
                                                    Default
                                                </span>
                                            )}
                                            {activePreset === preset.id && (
                                                <span className="bg-green-900 text-green-400 px-2 py-0.5 rounded text-xs">
                                                    Active
                                                </span>
                                            )}
                                        </div>
                                        {preset.description && (
                                            <p className="text-gray-400 text-sm mt-1">{preset.description}</p>
                                        )}
                                        <div className="flex gap-1 mt-2">
                                            {preset.tags?.map((tag, i) => (
                                                <span key={i} className="bg-gray-700 px-2 py-0.5 rounded text-xs">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => handleApplyPreset(preset)}
                                            className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm"
                                        >
                                            Apply
                                        </button>
                                        <button
                                            onClick={() => handleDeletePreset(preset)}
                                            className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                        {presets.length === 0 && (
                            <div className="text-center text-gray-500 py-8">
                                No presets saved. Create one to save your current configuration.
                            </div>
                        )}
                    </div>
                </div>
            </SettingsSection>

            {/* Quick Actions */}
            <SettingsSection
                title="Quick Actions"
                description="Common maintenance operations"
                icon={Zap}
                defaultOpen={false}
            >
                <div className="grid grid-cols-2 gap-4">
                    <button className="bg-gray-900 hover:bg-gray-700 p-4 rounded-lg text-left border border-gray-700">
                        <div className="font-semibold flex items-center gap-2">
                            <RefreshCw size={18} className="text-blue-400" />
                            Restart Services
                        </div>
                        <div className="text-gray-400 text-sm mt-1">Restart all platform services</div>
                    </button>
                    <button className="bg-gray-900 hover:bg-gray-700 p-4 rounded-lg text-left border border-gray-700">
                        <div className="font-semibold flex items-center gap-2">
                            <Shield size={18} className="text-red-400" />
                            Reset Safety Defaults
                        </div>
                        <div className="text-gray-400 text-sm mt-1">Restore default safety parameters</div>
                    </button>
                    <button className="bg-gray-900 hover:bg-gray-700 p-4 rounded-lg text-left border border-gray-700">
                        <div className="font-semibold flex items-center gap-2">
                            <Wifi size={18} className="text-green-400" />
                            Test Connectivity
                        </div>
                        <div className="text-gray-400 text-sm mt-1">Check all device connections</div>
                    </button>
                    <button className="bg-gray-900 hover:bg-gray-700 p-4 rounded-lg text-left border border-gray-700">
                        <div className="font-semibold flex items-center gap-2">
                            <Download size={18} className="text-purple-400" />
                            Export Diagnostics
                        </div>
                        <div className="text-gray-400 text-sm mt-1">Download system diagnostic report</div>
                    </button>
                </div>
            </SettingsSection>

            {/* System Info */}
            <SettingsSection
                title="System Information"
                description="Platform version and status"
                icon={Settings}
                defaultOpen={false}
            >
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-gray-400 text-sm">Platform Version</div>
                        <div className="font-mono">v0.7.0</div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-gray-400 text-sm">API Version</div>
                        <div className="font-mono">v1</div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-gray-400 text-sm">Target Hardware</div>
                        <div>Jetson Thor</div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded">
                        <div className="text-gray-400 text-sm">Architecture</div>
                        <div>MoE Skills + VLA</div>
                    </div>
                </div>
            </SettingsSection>
        </div>
    );
};

export default SettingsPage;

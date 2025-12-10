import React, { useState, useEffect } from 'react';
import { Camera, Hand, RefreshCw, Settings, Wifi, Server, Play, Pause, Video, Sliders, CheckCircle, XCircle, AlertCircle, Crosshair, RotateCw, ZoomIn, ZoomOut } from 'lucide-react';
import { fetchWithAuth } from './api';

const DeviceCard = ({ device, onCalibrate, onConfigure }) => {
    const statusColor = {
        'ONLINE': 'bg-green-500',
        'OFFLINE': 'bg-red-500',
        'CONNECTING': 'bg-yellow-500',
        'CALIBRATING': 'bg-blue-500',
    };

    const getIcon = (type) => {
        switch (type) {
            case 'CAMERA':
            case 'ONVIF':
                return <Camera size={24} />;
            case 'DOGLOVE':
            case 'DYGLOVE':
                return <Hand size={24} />;
            case 'DAIMON_VTLA':
                return <Server size={24} />;
            default:
                return <Server size={24} />;
        }
    };

    return (
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 hover:border-blue-500 transition">
            <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-gray-700 rounded-lg">
                        {getIcon(device.type)}
                    </div>
                    <div>
                        <h3 className="font-semibold">{device.type}</h3>
                        <p className="text-xs text-gray-400 font-mono">{device.id}</p>
                    </div>
                </div>
                <div className={`w-3 h-3 rounded-full ${statusColor[device.status] || 'bg-gray-500'}`} />
            </div>

            <div className="grid grid-cols-2 gap-2 text-sm mb-3">
                <div className="text-gray-400">Status</div>
                <div className={device.status === 'ONLINE' ? 'text-green-400' : 'text-gray-300'}>{device.status}</div>
                <div className="text-gray-400">Last Seen</div>
                <div>{device.last_seen ? new Date(device.last_seen * 1000).toLocaleTimeString() : '--'}</div>
            </div>

            <div className="flex gap-2">
                <button
                    onClick={() => onCalibrate(device)}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 py-1.5 rounded text-sm flex items-center justify-center gap-1"
                >
                    <Sliders size={14} /> Calibrate
                </button>
                <button
                    onClick={() => onConfigure(device)}
                    className="flex-1 bg-gray-700 hover:bg-gray-600 py-1.5 rounded text-sm flex items-center justify-center gap-1"
                >
                    <Settings size={14} /> Configure
                </button>
            </div>
        </div>
    );
};

const ONVIFCameraPanel = ({ camera, onClose }) => {
    const [ptzEnabled, setPtzEnabled] = useState(false);
    const [streaming, setStreaming] = useState(false);
    const [presets, setPresets] = useState(['Home', 'Workbench', 'Entry']);
    const [selectedPreset, setSelectedPreset] = useState(null);

    const handlePTZ = async (direction) => {
        console.log(`PTZ: ${direction}`);
        // In production: await fetchWithAuth(`/api/onvif/${camera.id}/ptz`, { method: 'POST', body: JSON.stringify({ direction }) });
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden">
                <div className="p-4 border-b border-gray-700 flex justify-between items-center">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Camera className="text-blue-400" /> ONVIF Camera Control
                    </h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
                </div>

                <div className="p-4 grid grid-cols-3 gap-4">
                    {/* Video Preview */}
                    <div className="col-span-2">
                        <div className="bg-gray-900 rounded-lg aspect-video flex items-center justify-center relative">
                            {streaming ? (
                                <div className="text-green-400">RTSP Stream Active</div>
                            ) : (
                                <div className="text-center">
                                    <Video size={48} className="mx-auto mb-2 text-gray-600" />
                                    <p className="text-gray-500">Stream not active</p>
                                </div>
                            )}
                            <div className="absolute top-2 right-2 flex gap-2">
                                <button
                                    onClick={() => setStreaming(!streaming)}
                                    className={`p-2 rounded ${streaming ? 'bg-red-600' : 'bg-green-600'}`}
                                >
                                    {streaming ? <Pause size={16} /> : <Play size={16} />}
                                </button>
                            </div>
                        </div>

                        {/* PTZ Controls */}
                        <div className="mt-4 flex justify-center">
                            <div className="grid grid-cols-3 gap-1">
                                <div />
                                <button onClick={() => handlePTZ('up')} className="bg-gray-700 hover:bg-gray-600 p-3 rounded">↑</button>
                                <div />
                                <button onClick={() => handlePTZ('left')} className="bg-gray-700 hover:bg-gray-600 p-3 rounded">←</button>
                                <button onClick={() => handlePTZ('home')} className="bg-blue-600 hover:bg-blue-700 p-3 rounded">
                                    <Crosshair size={16} className="mx-auto" />
                                </button>
                                <button onClick={() => handlePTZ('right')} className="bg-gray-700 hover:bg-gray-600 p-3 rounded">→</button>
                                <div />
                                <button onClick={() => handlePTZ('down')} className="bg-gray-700 hover:bg-gray-600 p-3 rounded">↓</button>
                                <div />
                            </div>
                            <div className="ml-4 flex flex-col gap-1">
                                <button onClick={() => handlePTZ('zoom_in')} className="bg-gray-700 hover:bg-gray-600 p-3 rounded">
                                    <ZoomIn size={16} />
                                </button>
                                <button onClick={() => handlePTZ('zoom_out')} className="bg-gray-700 hover:bg-gray-600 p-3 rounded">
                                    <ZoomOut size={16} />
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Settings Panel */}
                    <div className="space-y-4">
                        <div className="bg-gray-900 p-4 rounded-lg">
                            <h3 className="font-semibold mb-3">Camera Info</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-gray-400">ID</span>
                                    <span className="font-mono">{camera?.id || 'N/A'}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Protocol</span>
                                    <span>ONVIF</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">PTZ</span>
                                    <span className="text-green-400">Supported</span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-gray-900 p-4 rounded-lg">
                            <h3 className="font-semibold mb-3">Presets</h3>
                            <div className="space-y-2">
                                {presets.map((preset, i) => (
                                    <button
                                        key={i}
                                        onClick={() => {
                                            setSelectedPreset(preset);
                                            handlePTZ(`preset_${i}`);
                                        }}
                                        className={`w-full text-left p-2 rounded ${
                                            selectedPreset === preset ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                                        }`}
                                    >
                                        {preset}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="bg-gray-900 p-4 rounded-lg">
                            <h3 className="font-semibold mb-3">Calibration</h3>
                            <button className="w-full bg-purple-600 hover:bg-purple-700 py-2 rounded flex items-center justify-center gap-2">
                                <RotateCw size={16} /> Auto-Calibrate
                            </button>
                            <p className="text-xs text-gray-400 mt-2">
                                Runs intrinsic/extrinsic calibration for multi-camera triangulation.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const GloveCalibrationPanel = ({ glove, onClose }) => {
    const [calibrationStep, setCalibrationStep] = useState(0);
    const [fingerData, setFingerData] = useState({
        thumb: { mcp: 0, pip: 0, dip: 0, abd: 0 },
        index: { mcp: 0, pip: 0, dip: 0 },
        middle: { mcp: 0, pip: 0, dip: 0 },
        ring: { mcp: 0, pip: 0, dip: 0 },
        pinky: { mcp: 0, pip: 0, dip: 0 },
    });
    const [hapticTest, setHapticTest] = useState(false);

    const calibrationSteps = [
        { title: 'Flat Hand', instruction: 'Place your hand flat on a surface with fingers extended.' },
        { title: 'Full Fist', instruction: 'Make a tight fist with all fingers curled.' },
        { title: 'Pinch Grip', instruction: 'Touch your thumb to your index finger.' },
        { title: 'Spread Fingers', instruction: 'Spread all fingers as wide as possible.' },
    ];

    const handleNextStep = () => {
        if (calibrationStep < calibrationSteps.length - 1) {
            setCalibrationStep(prev => prev + 1);
        } else {
            alert('Calibration complete!');
            onClose();
        }
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg w-full max-w-3xl">
                <div className="p-4 border-b border-gray-700 flex justify-between items-center">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Hand className="text-purple-400" /> DOGlove Calibration (21-DOF)
                    </h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
                </div>

                <div className="p-6">
                    {/* Progress */}
                    <div className="flex items-center mb-6">
                        {calibrationSteps.map((step, i) => (
                            <React.Fragment key={i}>
                                <div className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${
                                    i < calibrationStep ? 'bg-green-600' :
                                    i === calibrationStep ? 'bg-blue-600' : 'bg-gray-700'
                                }`}>
                                    {i < calibrationStep ? <CheckCircle size={16} /> : i + 1}
                                </div>
                                {i < calibrationSteps.length - 1 && (
                                    <div className={`flex-1 h-1 mx-2 ${i < calibrationStep ? 'bg-green-600' : 'bg-gray-700'}`} />
                                )}
                            </React.Fragment>
                        ))}
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        {/* Instructions */}
                        <div className="bg-gray-900 p-6 rounded-lg">
                            <h3 className="text-lg font-semibold mb-2">
                                Step {calibrationStep + 1}: {calibrationSteps[calibrationStep].title}
                            </h3>
                            <p className="text-gray-400 mb-4">
                                {calibrationSteps[calibrationStep].instruction}
                            </p>
                            <div className="flex gap-2">
                                <button
                                    onClick={handleNextStep}
                                    className="flex-1 bg-blue-600 hover:bg-blue-700 py-2 rounded"
                                >
                                    {calibrationStep < calibrationSteps.length - 1 ? 'Capture & Next' : 'Complete'}
                                </button>
                                <button
                                    onClick={() => setCalibrationStep(0)}
                                    className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
                                >
                                    Restart
                                </button>
                            </div>
                        </div>

                        {/* Live Data */}
                        <div className="bg-gray-900 p-4 rounded-lg">
                            <h3 className="font-semibold mb-3">Live Joint Angles (degrees)</h3>
                            <div className="space-y-2 text-sm font-mono">
                                {Object.entries(fingerData).map(([finger, joints]) => (
                                    <div key={finger} className="flex justify-between">
                                        <span className="text-gray-400 capitalize">{finger}</span>
                                        <span>{Object.values(joints).map(v => v.toFixed(0)).join(' / ')}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Haptic Test */}
                    <div className="mt-6 bg-gray-900 p-4 rounded-lg">
                        <div className="flex justify-between items-center">
                            <div>
                                <h3 className="font-semibold">Haptic Feedback Test</h3>
                                <p className="text-gray-400 text-sm">Test force feedback on each finger (5-DOF).</p>
                            </div>
                            <button
                                onClick={() => setHapticTest(!hapticTest)}
                                className={`px-4 py-2 rounded ${hapticTest ? 'bg-red-600' : 'bg-green-600'}`}
                            >
                                {hapticTest ? 'Stop Test' : 'Start Test'}
                            </button>
                        </div>
                        {hapticTest && (
                            <div className="mt-4 grid grid-cols-5 gap-2">
                                {['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].map((finger, i) => (
                                    <div key={finger} className="bg-gray-800 p-3 rounded text-center">
                                        <div className="text-sm text-gray-400">{finger}</div>
                                        <div className="text-green-400 mt-1 animate-pulse">●</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

const DeviceManager = () => {
    const [devices, setDevices] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedDevice, setSelectedDevice] = useState(null);
    const [showONVIF, setShowONVIF] = useState(false);
    const [showGlove, setShowGlove] = useState(false);

    const refreshDevices = async () => {
        setLoading(true);
        try {
            const res = await fetchWithAuth('/devices');
            const data = await res.json();
            setDevices(data);
        } catch (e) {
            console.error("Failed to fetch devices", e);
        } finally {
            setLoading(false);
        }
    };

    const handleScan = async () => {
        setLoading(true);
        try {
            await fetchWithAuth('/devices/scan', { method: 'POST' });
            setTimeout(refreshDevices, 2000);
        } catch (e) {
            console.error("Scan failed", e);
            setLoading(false);
        }
    };

    const handleCalibrate = (device) => {
        setSelectedDevice(device);
        if (device.type === 'CAMERA' || device.type === 'ONVIF') {
            setShowONVIF(true);
        } else if (device.type === 'DOGLOVE' || device.type === 'DYGLOVE') {
            setShowGlove(true);
        } else {
            alert(`Calibration not available for ${device.type}`);
        }
    };

    const handleConfigure = (device) => {
        alert(`Configure ${device.type}: ${device.id}`);
    };

    useEffect(() => {
        refreshDevices();
    }, []);

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Server className="text-blue-400" /> Device Manager
                </h1>
                <div className="flex gap-2">
                    <button
                        onClick={handleScan}
                        disabled={loading}
                        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded flex items-center gap-2"
                    >
                        <Wifi size={18} /> Scan Network
                    </button>
                    <button
                        onClick={refreshDevices}
                        disabled={loading}
                        className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
                    >
                        <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
                    </button>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Total Devices</div>
                    <div className="text-2xl font-bold">{devices.length}</div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Online</div>
                    <div className="text-2xl font-bold text-green-400">
                        {devices.filter(d => d.status === 'ONLINE').length}
                    </div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Cameras</div>
                    <div className="text-2xl font-bold">
                        {devices.filter(d => d.type === 'CAMERA' || d.type === 'ONVIF').length}
                    </div>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm">Robots</div>
                    <div className="text-2xl font-bold">
                        {devices.filter(d => d.type.includes('VTLA') || d.type.includes('ROBOT')).length}
                    </div>
                </div>
            </div>

            {/* Device Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {devices.map(device => (
                    <DeviceCard
                        key={device.id}
                        device={device}
                        onCalibrate={handleCalibrate}
                        onConfigure={handleConfigure}
                    />
                ))}
                {devices.length === 0 && !loading && (
                    <div className="col-span-full text-center text-gray-500 py-12">
                        No devices found. Try scanning the network.
                    </div>
                )}
            </div>

            {/* ONVIF Panel */}
            {showONVIF && (
                <ONVIFCameraPanel
                    camera={selectedDevice}
                    onClose={() => setShowONVIF(false)}
                />
            )}

            {/* Glove Panel */}
            {showGlove && (
                <GloveCalibrationPanel
                    glove={selectedDevice}
                    onClose={() => setShowGlove(false)}
                />
            )}
        </div>
    );
};

export default DeviceManager;

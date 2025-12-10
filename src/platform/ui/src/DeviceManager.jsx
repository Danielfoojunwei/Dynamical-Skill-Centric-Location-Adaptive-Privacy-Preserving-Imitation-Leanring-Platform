
import React, { useState, useEffect } from 'react'
import { Camera, Hand, RefreshCw, Settings, Wifi, Server } from 'lucide-react'
import { fetchWithAuth } from './api'

const DeviceManager = () => {
    const [devices, setDevices] = useState([])
    const [loading, setLoading] = useState(true)

    const refreshDevices = async () => {
        setLoading(true)
        try {
            const res = await fetchWithAuth('/devices')
            const data = await res.json()
            setDevices(data)
        } catch (e) {
            console.error("Failed to fetch devices", e)
        } finally {
            setLoading(false)
        }
    }

    const handleScan = async () => {
        setLoading(true)
        try {
            await fetchWithAuth('/devices/scan', { method: 'POST' })
            // Wait a bit for responses
            setTimeout(refreshDevices, 2000)
        } catch (e) {
            console.error("Scan failed", e)
            setLoading(false)
        }
    }

    const checkOTA = async (type) => {
        try {
            const res = await fetchWithAuth(`/api/ota/check/${type}`)
            const data = await res.json()
            if (data.update_available) {
                if (confirm(`Update available for ${type} (v${data.version}). Install now?`)) {
                    // In a real app, this would trigger the install endpoint
                    alert(`Installing update for ${type}...`)
                }
            } else {
                alert(`No updates available for ${type}`)
            }
        } catch (e) {
            console.error("OTA check failed", e)
            alert("Failed to check for updates")
        }
    }

    useEffect(() => {
        refreshDevices()
    }, [])

    return (
        <div className="card">
            <div className="flex justify-between items-center" style={{ marginBottom: '1.5rem' }}>
                <h2>Connected Devices</h2>
                <div className="flex gap-2">
                    <button className="btn" onClick={handleScan} disabled={loading}>
                        <Wifi size={18} style={{ marginRight: '0.5rem' }} />
                        {loading ? 'Scanning...' : 'Scan Wireless'}
                    </button>
                    <button className="btn" onClick={refreshDevices} disabled={loading}>
                        <RefreshCw size={18} className={loading ? 'spin' : ''} />
                    </button>
                </div>
            </div>

            <div className="device-grid">
                {devices.map(device => (
                    <div key={device.id} className="device-item">
                        <div className="device-icon">
                            {device.type === 'CAMERA' ? <Camera size={24} /> :
                                device.type === 'DOGLOVE' ? <Hand size={24} /> : <Server size={24} />}
                        </div>
                        <div className="device-info">
                            <h3>{device.type}</h3>
                            <p className="text-sm text-muted">ID: {device.id}</p>
                            <div className="flex items-center gap-2" style={{ marginTop: '0.25rem' }}>
                                <div className={`status-dot ${device.status === 'ONLINE' ? 'running' : ''}`} />
                                <span className="text-sm">{device.status}</span>
                            </div>
                        </div>
                    </div>
                ))}

                {devices.length === 0 && (
                    <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                        No devices found. Try scanning.
                    </div>
                )}
            </div>
        </div>
    )
}

export default DeviceManager

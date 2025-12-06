import React, { useState, useEffect } from 'react'
import { Save } from 'lucide-react'
import { fetchWithAuth } from './api'

const Settings = () => {
    const [settings, setSettings] = useState({
        camera_rtsp_url: ''
    })
    const [loading, setLoading] = useState(false)
    const [message, setMessage] = useState('')

    useEffect(() => {
        loadSettings()
    }, [])

    const loadSettings = async () => {
        try {
            const res = await fetchWithAuth('/api/settings')
            if (res.ok) {
                const data = await res.json()
                setSettings(data)
            }
        } catch (e) {
            console.error("Failed to load settings", e)
        }
    }

    const handleSave = async () => {
        setLoading(true)
        setMessage('')
        try {
            const res = await fetchWithAuth('/api/settings', {
                method: 'POST',
                body: JSON.stringify(settings)
            })
            if (res.ok) {
                setMessage('Settings saved successfully')
            } else {
                setMessage('Failed to save settings')
            }
        } catch (e) {
            console.error("Failed to save", e)
            setMessage('Error saving settings')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="card">
            <h2>System Configuration</h2>

            <div style={{ marginTop: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                    Camera RTSP URL
                </label>
                <input
                    type="text"
                    value={settings.camera_rtsp_url}
                    onChange={(e) => setSettings({ ...settings, camera_rtsp_url: e.target.value })}
                    style={{
                        width: '100%',
                        padding: '0.75rem',
                        borderRadius: '8px',
                        border: '1px solid var(--border)',
                        backgroundColor: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        marginBottom: '1rem'
                    }}
                    placeholder="rtsp://..."
                />

                <div style={{ marginTop: '1rem' }}>
                    <button
                        className="btn"
                        onClick={handleSave}
                        disabled={loading}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                    >
                        <Save size={18} />
                        {loading ? 'Saving...' : 'Save Configuration'}
                    </button>
                </div>

                {message && (
                    <div style={{
                        marginTop: '1rem',
                        padding: '0.75rem',
                        borderRadius: '8px',
                        backgroundColor: message.includes('success') ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        color: message.includes('success') ? '#22c55e' : '#ef4444'
                    }}>
                        {message}
                    </div>
                )}
            </div>
        </div>
    )
}

export default Settings

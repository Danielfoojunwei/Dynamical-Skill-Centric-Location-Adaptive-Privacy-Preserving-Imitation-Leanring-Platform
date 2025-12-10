import React, { useState, useEffect, useRef } from 'react'
import { Shield, AlertTriangle, Save, Trash2 } from 'lucide-react'
import { fetchWithAuth } from './api'

const SafetyPage = () => {
    const [zones, setZones] = useState([])
    const [config, setConfig] = useState({ human_sensitivity: 0.8, stop_distance_m: 1.5 })
    const [isDrawing, setIsDrawing] = useState(false)
    const [currentPoints, setCurrentPoints] = useState([])
    const canvasRef = useRef(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadData()
    }, [])

    useEffect(() => {
        drawCanvas()
    }, [zones, currentPoints])

    const loadData = async () => {
        try {
            const [zRes, cRes] = await Promise.all([
                fetchWithAuth('/api/safety/zones'),
                fetchWithAuth('/api/safety/config')
            ])
            setZones(await zRes.json())
            setConfig(await cRes.json())
        } catch (e) {
            console.error("Failed to load safety data", e)
        } finally {
            setLoading(false)
        }
    }

    const drawCanvas = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Draw Grid/Floorplan placeholder
        ctx.strokeStyle = '#333'
        ctx.lineWidth = 1
        ctx.beginPath()
        for (let i = 0; i < canvas.width; i += 50) {
            ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height)
        }
        for (let i = 0; i < canvas.height; i += 50) {
            ctx.moveTo(0, i); ctx.lineTo(canvas.width, i)
        }
        ctx.stroke()

        // Draw Existing Zones
        zones.forEach(zone => {
            const coords = JSON.parse(zone.coordinates_json)
            if (coords.length < 2) return

            ctx.beginPath()
            ctx.moveTo(coords[0][0], coords[0][1])
            coords.forEach(p => ctx.lineTo(p[0], p[1]))
            ctx.closePath()

            ctx.fillStyle = zone.zone_type === 'KEEP_OUT' ? 'rgba(255, 0, 0, 0.3)' : 'rgba(255, 165, 0, 0.3)'
            ctx.fill()
            ctx.strokeStyle = zone.zone_type === 'KEEP_OUT' ? 'red' : 'orange'
            ctx.stroke()
        })

        // Draw Current Drawing
        if (currentPoints.length > 0) {
            ctx.beginPath()
            ctx.moveTo(currentPoints[0][0], currentPoints[0][1])
            currentPoints.forEach(p => ctx.lineTo(p[0], p[1]))
            ctx.strokeStyle = 'cyan'
            ctx.lineWidth = 2
            ctx.stroke()
        }
    }

    const handleCanvasClick = (e) => {
        if (!isDrawing) return
        const rect = canvasRef.current.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top
        setCurrentPoints([...currentPoints, [x, y]])
    }

    const startDrawing = () => {
        setIsDrawing(true)
        setCurrentPoints([])
    }

    const finishDrawing = async (type) => {
        setIsDrawing(false)
        if (currentPoints.length < 3) {
            alert("Zone must have at least 3 points")
            setCurrentPoints([])
            return
        }

        const name = `Zone ${zones.length + 1}`
        try {
            await fetchWithAuth('/api/safety/zones', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    zone_type: type,
                    coordinates: currentPoints
                })
            })
            loadData()
        } catch (e) {
            console.error("Failed to save zone", e)
        }
        setCurrentPoints([])
    }

    const deleteZone = async (id) => {
        if (!confirm("Are you sure?")) return
        try {
            await fetchWithAuth(`/api/safety/zones/${id}`, { method: 'DELETE' })
            loadData()
        } catch (e) {
            console.error("Failed to delete zone", e)
        }
    }

    const updateConfig = async (key, value) => {
        const newConfig = { ...config, [key]: parseFloat(value) }
        setConfig(newConfig)
        try {
            await fetchWithAuth('/api/safety/config', {
                method: 'POST',
                body: JSON.stringify(newConfig)
            })
        } catch (e) {
            console.error("Failed to update config", e)
        }
    }

    return (
        <div className="grid" style={{ gridTemplateColumns: '300px 1fr', gap: '2rem' }}>
            {/* Controls */}
            <div className="card">
                <h2><Shield size={20} style={{ marginRight: '0.5rem' }} /> Safety Controls</h2>

                <div style={{ marginTop: '2rem' }}>
                    <label>Human Detection Sensitivity</label>
                    <input
                        type="range" min="0.1" max="1.0" step="0.1"
                        value={config.human_sensitivity}
                        onChange={(e) => updateConfig('human_sensitivity', e.target.value)}
                        style={{ width: '100%' }}
                    />
                    <div className="text-right text-muted">{config.human_sensitivity}</div>
                </div>

                <div style={{ marginTop: '1rem' }}>
                    <label>Emergency Stop Distance (m)</label>
                    <input
                        type="range" min="0.5" max="5.0" step="0.5"
                        value={config.stop_distance_m}
                        onChange={(e) => updateConfig('stop_distance_m', e.target.value)}
                        style={{ width: '100%' }}
                    />
                    <div className="text-right text-muted">{config.stop_distance_m}m</div>
                </div>

                <div style={{ marginTop: '2rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                    <h3>Zones</h3>
                    <div className="flex gap-2" style={{ marginBottom: '1rem' }}>
                        <button className="btn" onClick={startDrawing} disabled={isDrawing}>
                            {isDrawing ? 'Click on Map...' : 'Draw Zone'}
                        </button>
                        {isDrawing && (
                            <>
                                <button className="btn btn-danger" onClick={() => finishDrawing('KEEP_OUT')}>
                                    Keep Out
                                </button>
                                <button className="btn" onClick={() => finishDrawing('SLOW_DOWN')}>
                                    Slow Down
                                </button>
                            </>
                        )}
                    </div>

                    <div className="zone-list">
                        {zones.map(z => (
                            <div key={z.id} className="flex justify-between items-center" style={{ padding: '0.5rem', background: 'var(--bg-secondary)', marginBottom: '0.5rem', borderRadius: '4px' }}>
                                <div>
                                    <div style={{ fontWeight: 'bold' }}>{z.name}</div>
                                    <div className="text-sm text-muted">{z.zone_type}</div>
                                </div>
                                <button onClick={() => deleteZone(z.id)} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                                    <Trash2 size={16} />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>


                <div style={{ marginTop: '2rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                    <h3>Hazard Definitions</h3>
                    <HazardRegistry />
                </div>
            </div>

            {/* Map Canvas */}
            <div className="card" style={{ padding: 0, overflow: 'hidden', position: 'relative' }}>
                <div style={{ position: 'absolute', top: '1rem', left: '1rem', background: 'rgba(0,0,0,0.7)', padding: '0.5rem', borderRadius: '4px', pointerEvents: 'none' }}>
                    Floor Plan View
                </div>
                <canvas
                    ref={canvasRef}
                    width={800}
                    height={600}
                    onClick={handleCanvasClick}
                    style={{ width: '100%', height: '100%', cursor: isDrawing ? 'crosshair' : 'default', background: '#1a1a1a' }}
                />
            </div>
        </div >
    )
}

const HazardRegistry = () => {
    const [hazards, setHazards] = useState([])
    const [showForm, setShowForm] = useState(false)
    const [newHazard, setNewHazard] = useState({
        type_key: '',
        display_name: '',
        description: '',
        default_severity: 0.5,
        action: 'WARN',
        clearance_m: 1.0
    })

    useEffect(() => {
        loadHazards()
    }, [])

    const loadHazards = async () => {
        try {
            const res = await fetchWithAuth('/api/safety/hazards/types')
            if (res.ok) {
                setHazards(await res.json())
            }
        } catch (e) {
            console.error("Failed to load hazards", e)
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        try {
            const payload = {
                type_key: newHazard.type_key.toUpperCase(),
                display_name: newHazard.display_name,
                description: newHazard.description,
                default_severity: parseFloat(newHazard.default_severity),
                default_behaviour: {
                    action: newHazard.action,
                    clearance_m: parseFloat(newHazard.clearance_m)
                }
            }

            await fetchWithAuth('/api/safety/hazards/types', {
                method: 'POST',
                body: JSON.stringify(payload)
            })

            setShowForm(false)
            setNewHazard({
                type_key: '',
                display_name: '',
                description: '',
                default_severity: 0.5,
                action: 'WARN',
                clearance_m: 1.0
            })
            loadHazards()
        } catch (e) {
            console.error("Failed to create hazard", e)
            alert("Failed to create hazard")
        }
    }

    return (
        <div>
            <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-muted">{hazards.length} Types</span>
                <button className="btn btn-sm" onClick={() => setShowForm(!showForm)}>
                    {showForm ? 'Cancel' : '+ New'}
                </button>
            </div>

            {showForm && (
                <form onSubmit={handleSubmit} className="card p-2 mb-2" style={{ background: 'var(--bg-secondary)' }}>
                    <div className="grid gap-2">
                        <input
                            placeholder="Key (e.g. FALLING_BOX)"
                            value={newHazard.type_key}
                            onChange={e => setNewHazard({ ...newHazard, type_key: e.target.value })}
                            required
                            style={{ width: '100%' }}
                        />
                        <input
                            placeholder="Display Name"
                            value={newHazard.display_name}
                            onChange={e => setNewHazard({ ...newHazard, display_name: e.target.value })}
                            required
                            style={{ width: '100%' }}
                        />
                        <input
                            placeholder="Description"
                            value={newHazard.description}
                            onChange={e => setNewHazard({ ...newHazard, description: e.target.value })}
                            style={{ width: '100%' }}
                        />
                        <div className="flex gap-2">
                            <select
                                value={newHazard.action}
                                onChange={e => setNewHazard({ ...newHazard, action: e.target.value })}
                                style={{ flex: 1 }}
                            >
                                <option value="WARN">WARN</option>
                                <option value="SLOW">SLOW</option>
                                <option value="STOP">STOP</option>
                                <option value="DUCK">DUCK</option>
                                <option value="CRAWL">CRAWL</option>
                            </select>
                            <input
                                type="number" step="0.1" placeholder="Clearance (m)"
                                value={newHazard.clearance_m}
                                onChange={e => setNewHazard({ ...newHazard, clearance_m: e.target.value })}
                                style={{ width: '80px' }}
                            />
                        </div>
                        <button type="submit" className="btn btn-primary width-full">Save</button>
                    </div>
                </form>
            )}

            <div className="hazard-list" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {hazards.map(h => (
                    <div key={h.type_key} style={{
                        padding: '0.5rem',
                        borderBottom: '1px solid var(--border)',
                        opacity: h.category === 'builtin' ? 0.7 : 1.0
                    }}>
                        <div className="flex justify-between">
                            <strong>{h.display_name}</strong>
                            <span className="badge" style={{
                                background: h.category === 'builtin' ? '#333' : 'var(--primary)',
                                fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px'
                            }}>
                                {h.category}
                            </span>
                        </div>
                        <div className="text-xs text-muted">{h.description}</div>
                        <div className="text-xs mt-1">
                            Action: <strong>{h.default_behaviour.action}</strong>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

export default SafetyPage

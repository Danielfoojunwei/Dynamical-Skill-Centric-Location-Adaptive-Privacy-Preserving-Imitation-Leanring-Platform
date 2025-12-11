import React, { useState, useEffect } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import { Activity, Cpu, HardDrive, Clock, Eye, Layers, Brain, Shield } from 'lucide-react'
import { fetchWithAuth } from './api'

const COLORS = ['#3b82f6', '#1e293b']
const META_AI_COLORS = {
    dinov3: '#3b82f6',
    sam3: '#8b5cf6',
    vjepa2: '#22c55e'
}

const StatCard = ({ title, value, subtext, icon: Icon, color }) => (
    <div className="card">
        <div className="flex justify-between items-start">
            <div>
                <div className="text-sm text-muted">{title}</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0.5rem 0' }}>{value}</div>
                {subtext && <div className="text-sm text-muted">{subtext}</div>}
            </div>
            <div style={{ padding: '0.5rem', borderRadius: '8px', backgroundColor: `${color}20` }}>
                <Icon size={24} color={color} />
            </div>
        </div>
    </div>
)

// Meta AI Model Mini Card
const MetaAIModelCard = ({ name, icon: Icon, color, tflops, status, latency }) => (
    <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
        <div className="flex items-center gap-2 mb-2">
            <Icon size={18} style={{ color }} />
            <span className="font-medium">{name}</span>
            <div className={`ml-auto w-2 h-2 rounded-full ${status === 'running' ? 'bg-green-500' : 'bg-gray-500'}`} />
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
                <span className="text-gray-500">TFLOPS</span>
                <div className="font-bold" style={{ color }}>{tflops}</div>
            </div>
            <div>
                <span className="text-gray-500">Latency</span>
                <div className="font-bold">{latency}ms</div>
            </div>
        </div>
    </div>
)

const Dashboard = ({ state, onStart, onStop }) => {
    const [metaAIModels, setMetaAIModels] = useState([
        { id: 'dinov3', name: 'DINOv3', tflops: 8.0, status: 'running', latency: 12 },
        { id: 'sam3', name: 'SAM 3', tflops: 15.0, status: 'running', latency: 25 },
        { id: 'vjepa2', name: 'V-JEPA 2', tflops: 10.0, status: 'running', latency: 18 }
    ])

    // Fetch Meta AI status
    useEffect(() => {
        const fetchMetaAI = async () => {
            try {
                const res = await fetchWithAuth('/api/perception/models')
                if (res.ok) {
                    const data = await res.json()
                    if (Array.isArray(data)) {
                        setMetaAIModels(data)
                    }
                }
            } catch (e) {
                // Use defaults
            }
        }
        fetchMetaAI()
        const interval = setInterval(fetchMetaAI, 5000)
        return () => clearInterval(interval)
    }, [])

    const metaAITflops = metaAIModels.filter(m => m.status === 'running').reduce((sum, m) => sum + m.tflops, 0)

    const tflopsData = [
        { name: 'Used', value: state.tflops_used },
        { name: 'Free', value: state.tflops_total - state.tflops_used }
    ]

    const tflopsBreakdown = [
        { name: 'Meta AI', value: metaAITflops, fill: '#8b5cf6' },
        { name: 'Safety', value: 15.0, fill: '#ef4444' },
        { name: 'Perception', value: 50.0, fill: '#3b82f6' },
        { name: 'VLA/Skills', value: 29.0, fill: '#22c55e' },
    ]

    const modelIcons = {
        dinov3: Eye,
        sam3: Layers,
        vjepa2: Brain
    }

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold">System Dashboard</h1>
                <div className="flex gap-4">
                    {state.status === 'IDLE' ? (
                        <button
                            onClick={onStart}
                            className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded flex items-center gap-2 transition"
                        >
                            <Activity size={20} /> Start System
                        </button>
                    ) : (
                        <button
                            onClick={onStop}
                            className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded flex items-center gap-2 transition"
                        >
                            <Activity size={20} /> Stop System
                        </button>
                    )}
                </div>
            </div>

            <div className="grid">
                <StatCard
                    title="TFLOPS Usage"
                    value={`${state.tflops_used.toFixed(1)} / ${state.tflops_total}`}
                    subtext={`${state.utilization_percent.toFixed(1)}% Utilization`}
                    icon={Cpu}
                    color="#3b82f6"
                />
                <StatCard
                    title="Memory Usage"
                    value={`${state.memory_used_gb} GB`}
                    subtext="of 32 GB (AGX Orin)"
                    icon={HardDrive}
                    color="#8b5cf6"
                />
                <StatCard
                    title="System Uptime"
                    value={`${(state.uptime_seconds / 60).toFixed(0)}m`}
                    subtext="Since last start"
                    icon={Clock}
                    color="#22c55e"
                />
                <StatCard
                    title="Active Components"
                    value={state.active_components.length}
                    subtext="Running processes"
                    icon={Activity}
                    color="#eab308"
                />

                {/* Meta AI Models Section */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <h3 className="flex items-center gap-2 mb-4">
                        <Brain size={20} className="text-purple-400" />
                        Meta AI Foundation Models
                        <span className="ml-auto text-sm text-gray-400">{metaAITflops} TFLOPS</span>
                    </h3>
                    <div className="grid grid-cols-3 gap-3">
                        {metaAIModels.map(model => (
                            <MetaAIModelCard
                                key={model.id}
                                name={model.name}
                                icon={modelIcons[model.id] || Brain}
                                color={META_AI_COLORS[model.id]}
                                tflops={model.tflops}
                                status={model.status}
                                latency={model.latency || model.latency_ms || 0}
                            />
                        ))}
                    </div>
                </div>

                {/* TFLOPS Breakdown */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <h3>TFLOPS Allocation by Category</h3>
                    <div style={{ height: 200 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={tflopsBreakdown} layout="vertical">
                                <XAxis type="number" domain={[0, 50]} />
                                <YAxis dataKey="name" type="category" width={80} />
                                <Tooltip />
                                <Bar dataKey="value" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Overall Utilization Pie Chart */}
                <div className="card" style={{ gridColumn: '1/-1' }}>
                    <h3>Overall TFLOPS Utilization</h3>
                    <div style={{ height: 300 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={tflopsData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {tflopsData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>

    )
}

export default Dashboard

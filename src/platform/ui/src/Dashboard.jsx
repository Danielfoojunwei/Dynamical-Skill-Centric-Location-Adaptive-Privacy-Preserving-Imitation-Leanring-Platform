import React from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import { Activity, Cpu, HardDrive, Clock } from 'lucide-react'
import { fetchWithAuth } from './api'

const COLORS = ['#3b82f6', '#1e293b']

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

const Dashboard = ({ state, onStart, onStop }) => {
    const tflopsData = [
        { name: 'Used', value: state.tflops_used },
        { name: 'Free', value: state.tflops_total - state.tflops_used }
    ]

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
                <div className="card" style={{ gridColumn: '1/-1' }}>
                    <h3>Utilization</h3>
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

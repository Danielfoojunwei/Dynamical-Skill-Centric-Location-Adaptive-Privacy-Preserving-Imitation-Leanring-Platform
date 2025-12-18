import { useState, useEffect } from 'react'
import { Activity, Server, Shield, LayoutDashboard, Settings, Cloud, Zap, Eye, Database, Bot, Brain, Play } from 'lucide-react'
import Dashboard from './Dashboard'
import DeviceManager from './DeviceManager'
import SettingsPage from './Settings'
import SafetyPage from './Safety'
import CloudIntegration from './CloudIntegration'
import SkillsManager from './SkillsManager'
import Observability from './Observability'
import TrainingManager from './TrainingManager'
import PerceptionManager from './PerceptionManager'
import SimulationDashboard from './SimulationDashboard'
import { fetchWithAuth } from './api'


function App() {
  const [view, setView] = useState('dashboard')
  const [systemState, setSystemState] = useState({
    status: 'IDLE',
    tflops_used: 0,
    tflops_total: 137.0,
    utilization_percent: 0,
    memory_used_gb: 0,
    uptime_seconds: 0,
    active_components: []
  })

  // Poll system stats
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetchWithAuth('/system/stats')
        if (res.ok) {
          const data = await res.json()
          setSystemState(data)
        }
      } catch (e) {
        console.error("Failed to fetch stats", e)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  const handleStart = async () => {
    await fetchWithAuth('/system/start', { method: 'POST' })
  }

  const handleStop = async () => {
    await fetchWithAuth('/system/stop', { method: 'POST' })
  }

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'perception', label: 'Perception', icon: Brain },
    { id: 'simulation', label: 'Simulation', icon: Play },
    { id: 'devices', label: 'Devices', icon: Server },
    { id: 'skills', label: 'Skills', icon: Zap },
    { id: 'observability', label: 'Observability', icon: Eye },
    { id: 'training', label: 'Training', icon: Database },
    { id: 'safety', label: 'Safety', icon: Shield },
    { id: 'cloud', label: 'Cloud', icon: Cloud },
    { id: 'settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className="app-container">
      <nav className="sidebar">
        <div className="logo">
          <Bot size={24} color="#3b82f6" />
          <span>Dynamical Edge</span>
          <span className="text-xs text-gray-500 ml-1">v0.4.0</span>
        </div>

        <div className="nav-items">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={`nav-item ${view === item.id ? 'active' : ''}`}
            >
              <item.icon size={20} /> {item.label}
            </button>
          ))}
        </div>

        {/* System Status Mini */}
        <div className="mt-auto p-4 border-t border-gray-700">
          <div className="text-xs text-gray-500 mb-2">SYSTEM STATUS</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              systemState.status === 'OPERATIONAL' ? 'bg-green-500' :
              systemState.status === 'IDLE' ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
            <span className="text-sm">{systemState.status}</span>
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {systemState.tflops_used.toFixed(1)} / {systemState.tflops_total} TFLOPS
          </div>
        </div>
      </nav>

      <main className="main-content">
        {view === 'dashboard' && <Dashboard state={systemState} onStart={handleStart} onStop={handleStop} />}
        {view === 'perception' && <PerceptionManager />}
        {view === 'simulation' && <SimulationDashboard />}
        {view === 'devices' && <DeviceManager />}
        {view === 'skills' && <SkillsManager />}
        {view === 'observability' && <Observability />}
        {view === 'training' && <TrainingManager />}
        {view === 'safety' && <SafetyPage />}
        {view === 'cloud' && <CloudIntegration />}
        {view === 'settings' && <SettingsPage />}
      </main>
    </div>
  )
}

export default App

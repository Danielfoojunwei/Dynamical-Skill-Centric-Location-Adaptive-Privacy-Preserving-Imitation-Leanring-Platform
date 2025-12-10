import { useState, useEffect } from 'react'
import { Activity, Server, Shield, LayoutDashboard, Settings, Cloud } from 'lucide-react'
import Dashboard from './Dashboard'
import DeviceManager from './DeviceManager'
import SettingsPage from './Settings'
import SafetyPage from './Safety'
import CloudIntegration from './CloudIntegration'
import { fetchWithAuth } from './api'


function App() {
  const [view, setView] = useState('dashboard')
  const [systemState, setSystemState] = useState({
    status: 'IDLE',
    tflops_used: 0,
    tflops_total: 137.0,
    utilization_percent: 0,
    active_components: []
  })

  // Poll system stats
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetchWithAuth('/api/system/stats')
        const data = await res.json()
        setSystemState(data)
      } catch (e) {
        console.error("Failed to fetch stats", e)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  const handleStart = async () => {
    await fetchWithAuth('/api/system/start', { method: 'POST' })
  }

  const handleStop = async () => {
    await fetchWithAuth('/api/system/stop', { method: 'POST' })
  }

  return (
    <div className="app-container">
      <nav className="sidebar">
        <div className="logo">
          <Activity size={24} color="#3b82f6" />
          <span>Dynamical Edge</span>
        </div>

        <div className="nav-items">
          <button onClick={() => setView('dashboard')} className={`nav-item ${view === 'dashboard' ? 'active' : ''}`}>
            <LayoutDashboard size={20} /> Dashboard
          </button>
          <button onClick={() => setView('devices')} className={`nav-item ${view === 'devices' ? 'active' : ''}`}>
            <Server size={20} /> Devices
          </button>
          <button onClick={() => setView('safety')} className={`nav-item ${view === 'safety' ? 'active' : ''}`}>
            <Shield size={20} /> Safety
          </button>
          <button onClick={() => setView('cloud')} className={`nav-item ${view === 'cloud' ? 'active' : ''}`}>
            <Cloud size={20} /> Cloud
          </button>
          <button onClick={() => setView('settings')} className={`nav-item ${view === 'settings' ? 'active' : ''}`}>
            <Settings size={20} /> Settings
          </button>
        </div>
      </nav>

      <main className="main-content">
        {view === 'dashboard' && <Dashboard state={systemState} onStart={handleStart} onStop={handleStop} />}
        {view === 'devices' && <DeviceManager />}
        {view === 'safety' && <SafetyPage />}
        {view === 'cloud' && <CloudIntegration />}
        {view === 'settings' && <SettingsPage />}
      </main>
    </div>
  )
}

export default App

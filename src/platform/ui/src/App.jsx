/**
 * Dynamical Edge Platform - Main Application
 *
 * v0.6.0 - Complete rewrite with:
 * - React Router for navigation
 * - Zustand for state management
 * - Radix UI for accessible components
 * - React Three Fiber for 3D visualization
 * - Enhanced WebSocket with auto-reconnect
 *
 * @version 0.6.0
 */

import React, { useEffect, Suspense, lazy } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
  useLocation,
} from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  Activity,
  Server,
  Shield,
  LayoutDashboard,
  Settings,
  Cloud,
  Zap,
  Eye,
  Database,
  Bot,
  Brain,
  Play,
  Wifi,
  WifiOff,
  Menu,
  ChevronLeft,
  Building2,
  History,
  GitBranch,
} from 'lucide-react';
import { clsx } from 'clsx';

// Stores
import {
  useSystemStore,
  useUiStore,
  selectTflopsPercent,
} from './stores';

// Components
import { ToastProvider } from './components/ui/Toast';
import { AlertCenter } from './components/ui/AlertCenter';

// Hooks
import { useWebSocket, ConnectionState } from './hooks/useWebSocket';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

// API
import { fetchWithAuth } from './api';

// Lazy-loaded feature pages
const Dashboard = lazy(() => import('./Dashboard'));
const DeviceManager = lazy(() => import('./DeviceManager'));
const SettingsPage = lazy(() => import('./Settings'));
const SafetyPage = lazy(() => import('./Safety'));
const CloudIntegration = lazy(() => import('./CloudIntegration'));
const SkillsManager = lazy(() => import('./SkillsManager'));
const Observability = lazy(() => import('./Observability'));
const TrainingManager = lazy(() => import('./TrainingManager'));
const PerceptionManager = lazy(() => import('./PerceptionManager'));
const SimulationDashboard = lazy(() => import('./SimulationDashboard'));

// New integrator pages
const DeploymentManager = lazy(() => import('./DeploymentManager'));
const AuditLog = lazy(() => import('./AuditLog'));
const VersionControl = lazy(() => import('./VersionControl'));

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
});

// Navigation items - organized by category for system integrators
const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/deployments', label: 'Deployments', icon: Building2 },
  { path: '/observability', label: 'Observability', icon: Eye },
  { path: '/skills', label: 'Skills', icon: Zap },
  { path: '/devices', label: 'Devices', icon: Server },
  { path: '/perception', label: 'Perception', icon: Brain },
  { path: '/safety', label: 'Safety', icon: Shield },
  { path: '/simulation', label: 'Simulation', icon: Play },
  { path: '/training', label: 'Training', icon: Database },
  { path: '/versions', label: 'Versions', icon: GitBranch },
  { path: '/audit', label: 'Audit Log', icon: History },
  { path: '/cloud', label: 'Cloud', icon: Cloud },
  { path: '/settings', label: 'Settings', icon: Settings },
];

// Loading spinner
function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
    </div>
  );
}

// Connection indicator
function ConnectionIndicator({ state, latency }) {
  const isConnected = state === ConnectionState.CONNECTED;
  const isReconnecting = state === ConnectionState.RECONNECTING;

  return (
    <div
      className={clsx(
        'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium',
        isConnected && 'bg-green-500/20 text-green-400',
        isReconnecting && 'bg-yellow-500/20 text-yellow-400',
        !isConnected && !isReconnecting && 'bg-red-500/20 text-red-400'
      )}
    >
      {isConnected ? (
        <Wifi className="w-3.5 h-3.5" />
      ) : (
        <WifiOff className="w-3.5 h-3.5" />
      )}
      <span>
        {isConnected
          ? `Connected ${latency > 0 ? `(${latency}ms)` : ''}`
          : isReconnecting
          ? 'Reconnecting...'
          : 'Disconnected'}
      </span>
    </div>
  );
}

// Sidebar component
function Sidebar() {
  const location = useLocation();
  const sidebarCollapsed = useUiStore((state) => state.sidebarCollapsed);
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);
  const status = useSystemStore((state) => state.status);
  const tflopsPercent = useSystemStore(selectTflopsPercent);

  return (
    <nav
      className={clsx(
        'sidebar flex flex-col h-full bg-gray-900 border-r border-gray-800 transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-56'
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-gray-800">
        <Bot size={24} className="text-blue-500 flex-shrink-0" />
        {!sidebarCollapsed && (
          <div className="flex-1 min-w-0">
            <span className="font-bold text-white">Dynamical Edge</span>
            <span className="text-xs text-gray-500 ml-2">v0.6.0</span>
          </div>
        )}
        <button
          onClick={toggleSidebar}
          className="p-1 rounded hover:bg-gray-800 text-gray-500"
        >
          {sidebarCollapsed ? (
            <Menu className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-4">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-4 py-2.5 mx-2 rounded-lg transition-colors',
                isActive
                  ? 'bg-blue-600/20 text-blue-400'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              )
            }
            title={sidebarCollapsed ? item.label : undefined}
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {!sidebarCollapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </div>

      {/* System status mini */}
      <div className="p-4 border-t border-gray-800">
        {!sidebarCollapsed && (
          <div className="text-xs text-gray-500 mb-2 uppercase tracking-wide">
            System Status
          </div>
        )}
        <div className="flex items-center gap-2">
          <div
            className={clsx(
              'w-2.5 h-2.5 rounded-full flex-shrink-0',
              status === 'OPERATIONAL' && 'bg-green-500',
              status === 'IDLE' && 'bg-yellow-500',
              status === 'ERROR' && 'bg-red-500',
              !['OPERATIONAL', 'IDLE', 'ERROR'].includes(status) && 'bg-gray-500'
            )}
          />
          {!sidebarCollapsed && (
            <div>
              <div className="text-sm text-white">{status}</div>
              <div className="text-xs text-gray-500">
                {tflopsPercent.toFixed(0)}% TFLOPS
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}

// Header component
function Header({ wsState }) {
  const addToast = useUiStore((state) => state.addToast);

  const handleSystemToggle = async () => {
    const status = useSystemStore.getState().status;
    const endpoint = status === 'OPERATIONAL' ? '/system/stop' : '/system/start';
    try {
      await fetchWithAuth(endpoint, { method: 'POST' });
      addToast({
        type: 'success',
        message: status === 'OPERATIONAL' ? 'System stopping...' : 'System starting...',
      });
    } catch (err) {
      addToast({ type: 'error', message: 'Failed to toggle system' });
    }
  };

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800">
      <div className="flex items-center gap-4">
        <ConnectionIndicator
          state={wsState.connectionState}
          latency={wsState.latency}
        />
      </div>

      <div className="flex items-center gap-3">
        <AlertCenter />

        <button
          onClick={handleSystemToggle}
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
            useSystemStore.getState().status === 'OPERATIONAL'
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white'
          )}
        >
          <Activity className="w-4 h-4" />
          {useSystemStore.getState().status === 'OPERATIONAL' ? 'Stop' : 'Start'}
        </button>
      </div>
    </header>
  );
}

// Main app content with routing
function AppContent() {
  // WebSocket connection
  const wsUrl = `ws://${window.location.hostname}:8000/ws/simulation`;
  const wsState = useWebSocket(wsUrl, {
    channels: ['all'],
    autoConnect: true,
  });

  // Keyboard shortcuts
  const handleSystemToggle = async () => {
    const status = useSystemStore.getState().status;
    const endpoint = status === 'OPERATIONAL' ? '/system/stop' : '/system/start';
    await fetchWithAuth(endpoint, { method: 'POST' });
  };

  const handleEmergencyStop = async () => {
    await fetchWithAuth('/system/emergency_stop', { method: 'POST' });
  };

  useKeyboardShortcuts({
    onSystemToggle: handleSystemToggle,
    onEmergencyStop: handleEmergencyStop,
  });

  // Apply theme
  const theme = useUiStore((state) => state.theme);
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <div className="app-container flex h-screen bg-gray-950 text-white">
      <Sidebar />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header wsState={wsState} />

        <main className="flex-1 overflow-auto p-6">
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/deployments" element={<DeploymentManager />} />
              <Route path="/perception" element={<PerceptionManager />} />
              <Route path="/simulation" element={<SimulationDashboard />} />
              <Route path="/devices" element={<DeviceManager />} />
              <Route path="/skills" element={<SkillsManager />} />
              <Route path="/observability" element={<Observability />} />
              <Route path="/training" element={<TrainingManager />} />
              <Route path="/safety" element={<SafetyPage />} />
              <Route path="/versions" element={<VersionControl />} />
              <Route path="/audit" element={<AuditLog />} />
              <Route path="/cloud" element={<CloudIntegration />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </div>
  );
}

// Root App component
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ToastProvider>
          <AppContent />
        </ToastProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;

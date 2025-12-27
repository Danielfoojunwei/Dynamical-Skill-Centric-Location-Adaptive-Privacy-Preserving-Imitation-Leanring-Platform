/**
 * Dynamical Edge Platform - Enterprise Management Console
 *
 * v1.0.0 - Enterprise Edition with:
 * - Hierarchical navigation with grouped sections
 * - Command palette (Cmd+K)
 * - Breadcrumb navigation
 * - ARM Pipeline integration
 * - Cross-robot transfer workflows
 * - Action reasoning display
 *
 * Enterprise design patterns:
 * - Consistent page layouts
 * - Unified component library
 * - Role-based navigation
 * - Contextual actions
 *
 * @version 1.0.0
 */

import React, { useEffect, Suspense, lazy, useState, useCallback } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
  useLocation,
  useNavigate,
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
  ChevronRight,
  Building2,
  History,
  GitBranch,
  Command,
  Search,
  Bell,
  User,
  HelpCircle,
  Gauge,
  Cpu,
  Network,
  Workflow,
  Target,
  FlaskConical,
  Users,
  AlertTriangle,
  X,
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
import { Badge, StatusIndicator, Button } from './design-system/components';

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
const DeploymentManager = lazy(() => import('./DeploymentManager'));
const AuditLog = lazy(() => import('./AuditLog'));
const VersionControl = lazy(() => import('./VersionControl'));

// New enterprise pages
const ARMPlanning = lazy(() => import('./pages/ARMPlanning'));
const FleetManagement = lazy(() => import('./pages/FleetManagement'));
const SkillOrchestration = lazy(() => import('./pages/SkillOrchestration'));

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
});

// Navigation structure - Enterprise grouped navigation
const navGroups = [
  {
    id: 'overview',
    label: 'Overview',
    items: [
      { path: '/', label: 'Dashboard', icon: LayoutDashboard, description: 'System metrics & status' },
      { path: '/observability', label: 'Observability', icon: Eye, description: 'Monitoring & RCA' },
    ],
  },
  {
    id: 'operations',
    label: 'Operations',
    items: [
      { path: '/arm-planning', label: 'ARM Planning', icon: Target, badge: 'NEW', description: 'Action Reasoning Model' },
      { path: '/skills', label: 'Skills & Orchestration', icon: Workflow, description: 'MoE skill library' },
      { path: '/simulation', label: 'Simulation', icon: Play, description: 'Isaac Lab integration' },
    ],
  },
  {
    id: 'fleet',
    label: 'Fleet & Devices',
    items: [
      { path: '/fleet', label: 'Fleet Management', icon: Users, badge: 'NEW', description: 'Cross-robot transfer' },
      { path: '/deployments', label: 'Deployments', icon: Building2, description: 'Site management' },
      { path: '/devices', label: 'Devices', icon: Server, description: 'Hardware & peripherals' },
    ],
  },
  {
    id: 'intelligence',
    label: 'AI & Perception',
    items: [
      { path: '/perception', label: 'Perception', icon: Brain, description: 'Meta AI models' },
      { path: '/training', label: 'Training', icon: Database, description: 'Data & episodes' },
    ],
  },
  {
    id: 'safety',
    label: 'Safety & Compliance',
    items: [
      { path: '/safety', label: 'Safety', icon: Shield, description: 'Zones & hazards' },
      { path: '/audit', label: 'Audit Log', icon: History, description: 'Activity history' },
      { path: '/versions', label: 'Config Versions', icon: GitBranch, description: 'Version control' },
    ],
  },
  {
    id: 'system',
    label: 'System',
    items: [
      { path: '/cloud', label: 'Cloud', icon: Cloud, description: 'Federated learning' },
      { path: '/settings', label: 'Settings', icon: Settings, description: 'Configuration' },
    ],
  },
];

// Flatten for route matching
const allNavItems = navGroups.flatMap((g) => g.items);

// Loading spinner with skeleton
function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          <div className="w-12 h-12 border-4 border-gray-700 rounded-full" />
          <div className="absolute top-0 left-0 w-12 h-12 border-4 border-blue-500 rounded-full border-t-transparent animate-spin" />
        </div>
        <p className="text-sm text-gray-400">Loading...</p>
      </div>
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
        'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all',
        isConnected && 'bg-green-500/10 text-green-400 border border-green-500/20',
        isReconnecting && 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20',
        !isConnected && !isReconnecting && 'bg-red-500/10 text-red-400 border border-red-500/20'
      )}
    >
      {isConnected ? (
        <Wifi className="w-3.5 h-3.5" />
      ) : (
        <WifiOff className="w-3.5 h-3.5" />
      )}
      <span>
        {isConnected
          ? `Live ${latency > 0 ? `· ${latency}ms` : ''}`
          : isReconnecting
          ? 'Reconnecting...'
          : 'Disconnected'}
      </span>
    </div>
  );
}

// Breadcrumb navigation
function Breadcrumbs() {
  const location = useLocation();
  const currentItem = allNavItems.find((item) => item.path === location.pathname);
  const currentGroup = navGroups.find((g) => g.items.some((item) => item.path === location.pathname));

  if (!currentItem) return null;

  return (
    <nav className="flex items-center gap-2 text-sm">
      <span className="text-gray-500">Dynamical</span>
      <ChevronRight className="w-4 h-4 text-gray-600" />
      {currentGroup && (
        <>
          <span className="text-gray-400">{currentGroup.label}</span>
          <ChevronRight className="w-4 h-4 text-gray-600" />
        </>
      )}
      <span className="text-white font-medium">{currentItem.label}</span>
    </nav>
  );
}

// Command Palette
function CommandPalette({ isOpen, onClose }) {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');

  const filteredItems = allNavItems.filter(
    (item) =>
      item.label.toLowerCase().includes(search.toLowerCase()) ||
      item.description?.toLowerCase().includes(search.toLowerCase())
  );

  const handleSelect = (path) => {
    navigate(path);
    onClose();
    setSearch('');
  };

  useEffect(() => {
    if (isOpen) {
      setSearch('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-xl bg-gray-900 rounded-xl shadow-2xl border border-gray-700 overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-700">
          <Search className="w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search pages, actions..."
            className="flex-1 bg-transparent text-white placeholder-gray-500 focus:outline-none"
            autoFocus
          />
          <kbd className="px-2 py-0.5 text-xs bg-gray-800 rounded text-gray-400">ESC</kbd>
        </div>
        <div className="max-h-80 overflow-y-auto p-2">
          {filteredItems.length === 0 ? (
            <div className="py-8 text-center text-gray-500">No results found</div>
          ) : (
            filteredItems.map((item) => (
              <button
                key={item.path}
                onClick={() => handleSelect(item.path)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-gray-800 text-left transition-colors"
              >
                <item.icon className="w-5 h-5 text-gray-400" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-medium">{item.label}</span>
                    {item.badge && (
                      <Badge variant="primary" size="sm">{item.badge}</Badge>
                    )}
                  </div>
                  {item.description && (
                    <p className="text-xs text-gray-500 truncate">{item.description}</p>
                  )}
                </div>
                <ChevronRight className="w-4 h-4 text-gray-600" />
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// Sidebar component with grouped navigation
function Sidebar() {
  const location = useLocation();
  const sidebarCollapsed = useUiStore((state) => state.sidebarCollapsed);
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);
  const status = useSystemStore((state) => state.status);
  const tflopsPercent = useSystemStore(selectTflopsPercent);
  const [expandedGroups, setExpandedGroups] = useState(['overview', 'operations']);

  const toggleGroup = (groupId) => {
    setExpandedGroups((prev) =>
      prev.includes(groupId) ? prev.filter((id) => id !== groupId) : [...prev, groupId]
    );
  };

  const statusColors = {
    OPERATIONAL: 'bg-green-500',
    IDLE: 'bg-yellow-500',
    ERROR: 'bg-red-500',
    STARTING: 'bg-blue-500',
  };

  return (
    <nav
      className={clsx(
        'flex flex-col h-full bg-gray-900/95 backdrop-blur border-r border-gray-800 transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-gray-800">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
          <Bot size={18} className="text-white" />
        </div>
        {!sidebarCollapsed && (
          <div className="flex-1 min-w-0">
            <span className="font-bold text-white">Dynamical</span>
            <span className="text-xs text-gray-500 ml-1.5">Enterprise</span>
          </div>
        )}
        <button
          onClick={toggleSidebar}
          className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-500 transition-colors"
        >
          {sidebarCollapsed ? <Menu className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 overflow-y-auto py-3">
        {navGroups.map((group) => (
          <div key={group.id} className="mb-2">
            {!sidebarCollapsed && (
              <button
                onClick={() => toggleGroup(group.id)}
                className="w-full flex items-center justify-between px-4 py-1.5 text-xs font-medium text-gray-500 uppercase tracking-wider hover:text-gray-400"
              >
                {group.label}
                <ChevronRight
                  className={clsx(
                    'w-3 h-3 transition-transform',
                    expandedGroups.includes(group.id) && 'rotate-90'
                  )}
                />
              </button>
            )}
            {(sidebarCollapsed || expandedGroups.includes(group.id)) && (
              <div className="mt-1">
                {group.items.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      clsx(
                        'group flex items-center gap-3 px-4 py-2 mx-2 rounded-lg transition-all',
                        isActive
                          ? 'bg-blue-600/20 text-blue-400'
                          : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                      )
                    }
                    title={sidebarCollapsed ? item.label : undefined}
                  >
                    <item.icon className="w-5 h-5 flex-shrink-0" />
                    {!sidebarCollapsed && (
                      <>
                        <span className="flex-1">{item.label}</span>
                        {item.badge && (
                          <Badge variant="primary" size="sm">
                            {item.badge}
                          </Badge>
                        )}
                      </>
                    )}
                  </NavLink>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* System Status */}
      <div className="p-3 border-t border-gray-800">
        <div
          className={clsx(
            'flex items-center gap-3 p-3 rounded-lg bg-gray-800/50',
            sidebarCollapsed && 'justify-center'
          )}
        >
          <div className="relative">
            <div
              className={clsx(
                'w-2.5 h-2.5 rounded-full',
                statusColors[status] || 'bg-gray-500'
              )}
            />
            <div
              className={clsx(
                'absolute inset-0 w-2.5 h-2.5 rounded-full animate-ping opacity-75',
                status === 'OPERATIONAL' && 'bg-green-500'
              )}
            />
          </div>
          {!sidebarCollapsed && (
            <div className="flex-1 min-w-0">
              <div className="text-sm text-white font-medium">{status}</div>
              <div className="text-xs text-gray-500">{tflopsPercent.toFixed(0)}% TFLOPS</div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}

// Header component with enterprise actions
function Header({ wsState, onOpenCommandPalette }) {
  const addToast = useUiStore((state) => state.addToast);
  const status = useSystemStore((state) => state.status);

  const handleSystemToggle = async () => {
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

  const handleEmergencyStop = async () => {
    try {
      await fetchWithAuth('/api/v1/safety/estop', { method: 'POST' });
      addToast({ type: 'warning', message: 'Emergency stop triggered!' });
    } catch (err) {
      addToast({ type: 'error', message: 'E-Stop failed!' });
    }
  };

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-gray-900/95 backdrop-blur border-b border-gray-800">
      <div className="flex items-center gap-6">
        <Breadcrumbs />
      </div>

      <div className="flex items-center gap-3">
        {/* Command Palette Trigger */}
        <button
          onClick={onOpenCommandPalette}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-400 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
        >
          <Search className="w-4 h-4" />
          <span className="hidden md:inline">Search</span>
          <kbd className="hidden md:inline px-1.5 py-0.5 text-xs bg-gray-700 rounded">⌘K</kbd>
        </button>

        <ConnectionIndicator state={wsState.connectionState} latency={wsState.latency} />

        <AlertCenter />

        {/* E-Stop Button */}
        <Button
          variant="danger"
          size="sm"
          icon={AlertTriangle}
          onClick={handleEmergencyStop}
          className="hidden md:flex"
        >
          E-Stop
        </Button>

        {/* System Toggle */}
        <Button
          variant={status === 'OPERATIONAL' ? 'danger' : 'success'}
          size="sm"
          icon={Activity}
          onClick={handleSystemToggle}
        >
          {status === 'OPERATIONAL' ? 'Stop' : 'Start'}
        </Button>

        {/* User Menu */}
        <button className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center hover:bg-gray-700 transition-colors">
          <User className="w-4 h-4 text-gray-400" />
        </button>
      </div>
    </header>
  );
}

// Page wrapper with consistent layout
function PageWrapper({ children }) {
  return (
    <div className="min-h-full">
      {children}
    </div>
  );
}

// Main app content with routing
function AppContent() {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

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
    await fetchWithAuth('/api/v1/safety/estop', { method: 'POST' });
  };

  useKeyboardShortcuts({
    onSystemToggle: handleSystemToggle,
    onEmergencyStop: handleEmergencyStop,
  });

  // Command palette keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
      if (e.key === 'Escape') {
        setCommandPaletteOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Apply theme
  const theme = useUiStore((state) => state.theme);
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <div className="app-container flex h-screen bg-gray-950 text-white">
      <Sidebar />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header wsState={wsState} onOpenCommandPalette={() => setCommandPaletteOpen(true)} />

        <main className="flex-1 overflow-auto">
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              {/* Overview */}
              <Route path="/" element={<PageWrapper><Dashboard /></PageWrapper>} />
              <Route path="/observability" element={<PageWrapper><Observability /></PageWrapper>} />

              {/* Operations */}
              <Route path="/arm-planning" element={<PageWrapper><ARMPlanning /></PageWrapper>} />
              <Route path="/skills" element={<PageWrapper><SkillOrchestration /></PageWrapper>} />
              <Route path="/simulation" element={<PageWrapper><SimulationDashboard /></PageWrapper>} />

              {/* Fleet & Devices */}
              <Route path="/fleet" element={<PageWrapper><FleetManagement /></PageWrapper>} />
              <Route path="/deployments" element={<PageWrapper><DeploymentManager /></PageWrapper>} />
              <Route path="/devices" element={<PageWrapper><DeviceManager /></PageWrapper>} />

              {/* AI & Perception */}
              <Route path="/perception" element={<PageWrapper><PerceptionManager /></PageWrapper>} />
              <Route path="/training" element={<PageWrapper><TrainingManager /></PageWrapper>} />

              {/* Safety & Compliance */}
              <Route path="/safety" element={<PageWrapper><SafetyPage /></PageWrapper>} />
              <Route path="/audit" element={<PageWrapper><AuditLog /></PageWrapper>} />
              <Route path="/versions" element={<PageWrapper><VersionControl /></PageWrapper>} />

              {/* System */}
              <Route path="/cloud" element={<PageWrapper><CloudIntegration /></PageWrapper>} />
              <Route path="/settings" element={<PageWrapper><SettingsPage /></PageWrapper>} />
            </Routes>
          </Suspense>
        </main>
      </div>

      <CommandPalette isOpen={commandPaletteOpen} onClose={() => setCommandPaletteOpen(false)} />
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

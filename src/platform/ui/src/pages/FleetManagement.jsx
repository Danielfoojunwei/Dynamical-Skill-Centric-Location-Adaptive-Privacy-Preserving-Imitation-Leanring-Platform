/**
 * Fleet Management Page - Cross-Robot Transfer & Fleet Operations
 *
 * Enterprise interface for fleet operations:
 * - Robot fleet overview with status monitoring
 * - Cross-robot skill transfer with compatibility matrix
 * - Batch transfer operations
 * - Fleet health scoring
 *
 * @version 1.0.0
 */

import React, { useState, useEffect } from 'react';
import {
  Users,
  Bot,
  ArrowRight,
  ArrowLeftRight,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Download,
  Upload,
  Settings,
  Activity,
  Cpu,
  Zap,
  Clock,
  Target,
  Layers,
  Play,
  Pause,
  BarChart3,
  TrendingUp,
  TrendingDown,
  ChevronRight,
  Filter,
  Search,
  Plus,
  MoreVertical,
  Eye,
} from 'lucide-react';
import { clsx } from 'clsx';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Badge,
  Tabs,
  Tab,
  ProgressBar,
  Alert,
  DataRow,
  Divider,
  KPICard,
  StatusIndicator,
  EmptyState,
  DataGrid,
} from '../design-system/components';
import { fetchWithAuth } from '../api';

// Mock fleet data
const MOCK_FLEET = [
  {
    id: 'ur10e-001',
    name: 'UR10e Primary',
    type: 'ur10e',
    site: 'Factory Floor A',
    status: 'online',
    dof: 6,
    skills: 12,
    uptime: 98.5,
    health: 95,
    lastSeen: '2 min ago',
  },
  {
    id: 'ur10e-002',
    name: 'UR10e Secondary',
    type: 'ur10e',
    site: 'Factory Floor A',
    status: 'online',
    dof: 6,
    skills: 8,
    uptime: 97.2,
    health: 92,
    lastSeen: '1 min ago',
  },
  {
    id: 'ur5e-001',
    name: 'UR5e Assembly',
    type: 'ur5e',
    site: 'Factory Floor B',
    status: 'online',
    dof: 6,
    skills: 6,
    uptime: 99.1,
    health: 98,
    lastSeen: '30 sec ago',
  },
  {
    id: 'franka-001',
    name: 'Franka Research',
    type: 'franka',
    site: 'R&D Lab',
    status: 'maintenance',
    dof: 7,
    skills: 4,
    uptime: 85.0,
    health: 78,
    lastSeen: '15 min ago',
  },
  {
    id: 'custom-001',
    name: 'Custom Arm',
    type: 'custom',
    site: 'Test Cell',
    status: 'offline',
    dof: 7,
    skills: 2,
    uptime: 45.0,
    health: 60,
    lastSeen: '2 hours ago',
  },
];

// Mock skills data
const MOCK_SKILLS = [
  { id: 'skill-001', name: 'Pick and Place', type: 'manipulation', robots: ['ur10e', 'ur5e', 'franka'] },
  { id: 'skill-002', name: 'Precision Assembly', type: 'manipulation', robots: ['ur10e', 'franka'] },
  { id: 'skill-003', name: 'Bin Picking', type: 'perception', robots: ['ur10e', 'ur5e'] },
  { id: 'skill-004', name: 'Palletizing', type: 'manipulation', robots: ['ur10e'] },
  { id: 'skill-005', name: 'Welding Path', type: 'manipulation', robots: ['ur10e', 'custom'] },
];

// Compatibility matrix data
const COMPATIBILITY_MATRIX = {
  ur10e: { ur10e: 100, ur5e: 92, franka: 78, custom: 65 },
  ur5e: { ur10e: 90, ur5e: 100, franka: 75, custom: 60 },
  franka: { ur10e: 72, ur5e: 70, franka: 100, custom: 55 },
  custom: { ur10e: 58, ur5e: 55, franka: 50, custom: 100 },
};

// Robot Card Component
function RobotCard({ robot, selected, onClick }) {
  const statusColors = {
    online: 'border-green-500/50 bg-green-500/5',
    maintenance: 'border-yellow-500/50 bg-yellow-500/5',
    offline: 'border-gray-500/50 bg-gray-500/5',
  };

  const healthColor = robot.health >= 90 ? 'text-green-400' : robot.health >= 70 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div
      onClick={onClick}
      className={clsx(
        'p-4 rounded-xl border-2 cursor-pointer transition-all',
        selected
          ? 'border-blue-500 bg-blue-500/10'
          : statusColors[robot.status] || 'border-gray-700 bg-gray-800/50',
        'hover:shadow-lg'
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gray-700 flex items-center justify-center">
            <Bot className="w-5 h-5 text-gray-300" />
          </div>
          <div>
            <h3 className="font-medium text-white">{robot.name}</h3>
            <p className="text-xs text-gray-500">{robot.type.toUpperCase()} · {robot.site}</p>
          </div>
        </div>
        <StatusIndicator status={robot.status} showLabel={false} />
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div>
          <div className="text-xs text-gray-500">Skills</div>
          <div className="text-lg font-bold text-white">{robot.skills}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Uptime</div>
          <div className="text-lg font-bold text-white">{robot.uptime}%</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Health</div>
          <div className={clsx('text-lg font-bold', healthColor)}>{robot.health}%</div>
        </div>
      </div>

      <ProgressBar value={robot.health} variant={robot.health >= 90 ? 'success' : robot.health >= 70 ? 'warning' : 'danger'} size="sm" />

      <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-700">
        <span className="text-xs text-gray-500">Last seen: {robot.lastSeen}</span>
        <Button variant="ghost" size="xs" icon={Eye}>
          Details
        </Button>
      </div>
    </div>
  );
}

// Compatibility Cell Component
function CompatibilityCell({ value }) {
  const color = value >= 80 ? 'bg-green-500' : value >= 60 ? 'bg-yellow-500' : 'bg-red-500';
  const opacity = value / 100;

  return (
    <div
      className={clsx('w-full h-12 flex items-center justify-center font-medium text-white rounded', color)}
      style={{ opacity: Math.max(0.3, opacity) }}
    >
      {value}%
    </div>
  );
}

// Transfer Modal Component
function TransferModal({ isOpen, onClose, sourceRobot, targetRobot, skill }) {
  const [transferring, setTransferring] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('idle');

  const stages = [
    { id: 'load', label: 'Loading skill weights', icon: Download },
    { id: 'adapt', label: 'Adapting to target embodiment', icon: ArrowLeftRight },
    { id: 'validate', label: 'Validating IK feasibility', icon: Target },
    { id: 'deploy', label: 'Deploying to target', icon: Upload },
  ];

  const handleTransfer = async () => {
    setTransferring(true);

    for (let i = 0; i < stages.length; i++) {
      setStage(stages[i].id);
      await new Promise((r) => setTimeout(r, 800));
      setProgress((i + 1) * 25);
    }

    setTimeout(() => {
      setTransferring(false);
      onClose();
    }, 500);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-gray-900 rounded-xl shadow-2xl border border-gray-700 p-6">
        <h2 className="text-xl font-bold text-white mb-4">Transfer Skill</h2>

        <div className="flex items-center justify-center gap-4 mb-6 p-4 bg-gray-800 rounded-lg">
          <div className="text-center">
            <div className="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center mx-auto mb-2">
              <Bot className="w-6 h-6 text-blue-400" />
            </div>
            <div className="font-medium text-white">{sourceRobot?.name || 'Source'}</div>
            <div className="text-xs text-gray-500">{sourceRobot?.type?.toUpperCase()}</div>
          </div>

          <ArrowRight className="w-8 h-8 text-gray-500" />

          <div className="text-center">
            <div className="w-12 h-12 rounded-lg bg-green-500/20 flex items-center justify-center mx-auto mb-2">
              <Bot className="w-6 h-6 text-green-400" />
            </div>
            <div className="font-medium text-white">{targetRobot?.name || 'Target'}</div>
            <div className="text-xs text-gray-500">{targetRobot?.type?.toUpperCase()}</div>
          </div>
        </div>

        {skill && (
          <div className="mb-6 p-3 bg-gray-800 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span className="font-medium text-white">{skill.name}</span>
            </div>
            <span className="text-xs text-gray-500">{skill.type}</span>
          </div>
        )}

        {transferring && (
          <div className="mb-6">
            <ProgressBar value={progress} variant="gradient" showLabel />
            <div className="mt-4 space-y-2">
              {stages.map((s) => (
                <div
                  key={s.id}
                  className={clsx(
                    'flex items-center gap-3 p-2 rounded',
                    stage === s.id && 'bg-blue-500/10'
                  )}
                >
                  {stage === s.id ? (
                    <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
                  ) : progress >= stages.indexOf(s) * 25 + 25 ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <div className="w-4 h-4 rounded-full border border-gray-600" />
                  )}
                  <span
                    className={clsx(
                      'text-sm',
                      stage === s.id ? 'text-white' : 'text-gray-500'
                    )}
                  >
                    {s.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex gap-3">
          <Button
            variant="primary"
            className="flex-1"
            onClick={handleTransfer}
            disabled={transferring}
            icon={transferring ? RefreshCw : ArrowRight}
          >
            {transferring ? 'Transferring...' : 'Start Transfer'}
          </Button>
          <Button variant="secondary" onClick={onClose} disabled={transferring}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}

// Main Fleet Management Page
export default function FleetManagement() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedRobots, setSelectedRobots] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [transferModal, setTransferModal] = useState({ open: false, source: null, target: null, skill: null });

  // Filter robots
  const filteredRobots = MOCK_FLEET.filter((robot) => {
    const matchesSearch = robot.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      robot.type.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || robot.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  // Fleet statistics
  const fleetStats = {
    total: MOCK_FLEET.length,
    online: MOCK_FLEET.filter((r) => r.status === 'online').length,
    totalSkills: MOCK_FLEET.reduce((sum, r) => sum + r.skills, 0),
    avgHealth: (MOCK_FLEET.reduce((sum, r) => sum + r.health, 0) / MOCK_FLEET.length).toFixed(0),
  };

  const handleRobotSelect = (robotId) => {
    setSelectedRobots((prev) =>
      prev.includes(robotId) ? prev.filter((id) => id !== robotId) : [...prev, robotId].slice(-2)
    );
  };

  const openTransferModal = (skill) => {
    if (selectedRobots.length === 2) {
      const source = MOCK_FLEET.find((r) => r.id === selectedRobots[0]);
      const target = MOCK_FLEET.find((r) => r.id === selectedRobots[1]);
      setTransferModal({ open: true, source, target, skill });
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <Users className="text-purple-400" />
            Fleet Management
          </h1>
          <p className="text-gray-400 mt-1">
            Cross-robot skill transfer and fleet operations
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" icon={RefreshCw}>
            Refresh
          </Button>
          <Button variant="primary" size="sm" icon={Plus}>
            Add Robot
          </Button>
        </div>
      </div>

      {/* Fleet Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <KPICard
          title="Total Robots"
          value={fleetStats.total}
          subvalue={`${fleetStats.online} online`}
          icon={Bot}
          iconColor="text-blue-400"
          iconBg="bg-blue-500/10"
        />
        <KPICard
          title="Online Rate"
          value={`${((fleetStats.online / fleetStats.total) * 100).toFixed(0)}%`}
          change="+2.5% from last week"
          changeType="positive"
          icon={Activity}
          iconColor="text-green-400"
          iconBg="bg-green-500/10"
        />
        <KPICard
          title="Total Skills Deployed"
          value={fleetStats.totalSkills}
          subvalue="across fleet"
          icon={Zap}
          iconColor="text-yellow-400"
          iconBg="bg-yellow-500/10"
        />
        <KPICard
          title="Avg Fleet Health"
          value={`${fleetStats.avgHealth}%`}
          change="-1.2% from last week"
          changeType="negative"
          icon={TrendingUp}
          iconColor="text-purple-400"
          iconBg="bg-purple-500/10"
        />
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onChange={setActiveTab}>
        <Tab value="overview" icon={LayoutDashboard}>Fleet Overview</Tab>
        <Tab value="transfer" icon={ArrowLeftRight}>Cross-Robot Transfer</Tab>
        <Tab value="matrix" icon={Layers}>Compatibility Matrix</Tab>
      </Tabs>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-4">
          {/* Filters */}
          <Card variant="default" padding="sm">
            <div className="flex items-center gap-4">
              <div className="relative flex-1 max-w-xs">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search robots..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All Status</option>
                <option value="online">Online</option>
                <option value="maintenance">Maintenance</option>
                <option value="offline">Offline</option>
              </select>
            </div>
          </Card>

          {/* Robot Grid */}
          <div className="grid grid-cols-3 gap-4">
            {filteredRobots.map((robot) => (
              <RobotCard
                key={robot.id}
                robot={robot}
                selected={selectedRobots.includes(robot.id)}
                onClick={() => handleRobotSelect(robot.id)}
              />
            ))}
          </div>

          {selectedRobots.length > 0 && (
            <Alert variant="info">
              {selectedRobots.length} robot(s) selected. Select 2 robots to enable skill transfer.
            </Alert>
          )}
        </div>
      )}

      {/* Transfer Tab */}
      {activeTab === 'transfer' && (
        <div className="grid grid-cols-12 gap-6">
          {/* Robot Selection */}
          <div className="col-span-8">
            <Card variant="elevated">
              <CardHeader>
                <CardTitle icon={Bot} iconColor="text-blue-400">
                  Select Robots for Transfer
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-400 mb-4">
                  Select a source robot and a target robot to transfer skills between them.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  {MOCK_FLEET.filter((r) => r.status === 'online').map((robot) => (
                    <RobotCard
                      key={robot.id}
                      robot={robot}
                      selected={selectedRobots.includes(robot.id)}
                      onClick={() => handleRobotSelect(robot.id)}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Skills List */}
          <div className="col-span-4">
            <Card variant="elevated">
              <CardHeader>
                <CardTitle icon={Zap} iconColor="text-yellow-400">
                  Available Skills
                </CardTitle>
              </CardHeader>
              <CardContent>
                {selectedRobots.length < 2 ? (
                  <EmptyState
                    icon={ArrowLeftRight}
                    title="Select 2 robots"
                    description="Choose a source and target robot to see transferable skills"
                  />
                ) : (
                  <div className="space-y-2">
                    {MOCK_SKILLS.map((skill) => (
                      <div
                        key={skill.id}
                        className="p-3 bg-gray-800 rounded-lg flex items-center justify-between"
                      >
                        <div>
                          <div className="font-medium text-white">{skill.name}</div>
                          <div className="text-xs text-gray-500">{skill.type}</div>
                        </div>
                        <Button
                          variant="primary"
                          size="xs"
                          icon={ArrowRight}
                          onClick={() => openTransferModal(skill)}
                        >
                          Transfer
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Matrix Tab */}
      {activeTab === 'matrix' && (
        <Card variant="elevated">
          <CardHeader>
            <CardTitle icon={Layers} iconColor="text-cyan-400">
              Compatibility Matrix
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-400 mb-4">
              Skill transfer compatibility between robot types. Higher percentages indicate better transfer success rates.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-sm text-gray-400">From / To</th>
                    {Object.keys(COMPATIBILITY_MATRIX).map((type) => (
                      <th key={type} className="p-2 text-center text-sm text-gray-400">
                        {type.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(COMPATIBILITY_MATRIX).map(([source, targets]) => (
                    <tr key={source}>
                      <td className="p-2 text-sm font-medium text-white">{source.toUpperCase()}</td>
                      {Object.values(targets).map((value, i) => (
                        <td key={i} className="p-2">
                          <CompatibilityCell value={value} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Divider />

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-green-500" />
                <span className="text-sm text-gray-400">High (80%+)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-yellow-500" />
                <span className="text-sm text-gray-400">Medium (60-79%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-red-500" />
                <span className="text-sm text-gray-400">Low (&lt;60%)</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Transfer Modal */}
      <TransferModal
        isOpen={transferModal.open}
        onClose={() => setTransferModal({ open: false, source: null, target: null, skill: null })}
        sourceRobot={transferModal.source}
        targetRobot={transferModal.target}
        skill={transferModal.skill}
      />
    </div>
  );
}

// Add missing LayoutDashboard import reference
const LayoutDashboard = ({ className }) => (
  <svg className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="7" height="7" rx="1"/>
    <rect x="14" y="3" width="7" height="4" rx="1"/>
    <rect x="14" y="10" width="7" height="11" rx="1"/>
    <rect x="3" y="13" width="7" height="8" rx="1"/>
  </svg>
);

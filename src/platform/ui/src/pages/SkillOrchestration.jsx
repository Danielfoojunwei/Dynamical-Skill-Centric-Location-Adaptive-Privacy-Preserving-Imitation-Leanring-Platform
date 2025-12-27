/**
 * Skill Orchestration Page - MoE Skills & Task Decomposition
 *
 * Enterprise interface for skill management:
 * - MoE skill library with search and filtering
 * - Task decomposition visualization
 * - Skill orchestration graph
 * - MoE routing decisions display
 * - Real-time execution monitoring
 *
 * @version 1.0.0
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Workflow,
  Zap,
  Search,
  Filter,
  Upload,
  Download,
  Play,
  Pause,
  RefreshCw,
  Brain,
  Target,
  ChevronRight,
  ChevronDown,
  Plus,
  MoreVertical,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Activity,
  Layers,
  GitBranch,
  ArrowRight,
  Cpu,
  BarChart3,
  TrendingUp,
  Lock,
  Unlock,
  Star,
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
} from '../design-system/components';
import { fetchWithAuth } from '../api';

// Mock skills data
const MOCK_SKILLS = [
  {
    id: 'skill-001',
    name: 'Pick and Place',
    description: 'Generic pick and place manipulation for various objects',
    type: 'manipulation',
    version: '2.1.0',
    status: 'active',
    tags: ['picking', 'placing', 'gripper'],
    accuracy: 94.5,
    latency: 45,
    invocations: 12450,
  },
  {
    id: 'skill-002',
    name: 'Precision Assembly',
    description: 'High-precision assembly operations with tolerance < 0.1mm',
    type: 'manipulation',
    version: '1.3.0',
    status: 'active',
    tags: ['assembly', 'precision', 'insertion'],
    accuracy: 98.2,
    latency: 120,
    invocations: 5230,
  },
  {
    id: 'skill-003',
    name: 'Bin Picking',
    description: 'Vision-guided bin picking with object detection',
    type: 'perception',
    version: '3.0.0',
    status: 'active',
    tags: ['vision', 'detection', 'picking'],
    accuracy: 91.8,
    latency: 85,
    invocations: 8920,
  },
  {
    id: 'skill-004',
    name: 'Palletizing',
    description: 'Automated palletizing with pattern optimization',
    type: 'manipulation',
    version: '1.0.0',
    status: 'pending',
    tags: ['palletizing', 'stacking', 'logistics'],
    accuracy: 96.0,
    latency: 60,
    invocations: 3100,
  },
  {
    id: 'skill-005',
    name: 'Surface Inspection',
    description: 'AI-powered visual inspection for defect detection',
    type: 'perception',
    version: '2.0.1',
    status: 'active',
    tags: ['inspection', 'quality', 'vision'],
    accuracy: 99.1,
    latency: 35,
    invocations: 45000,
  },
  {
    id: 'skill-006',
    name: 'Welding Path',
    description: 'Automated welding path planning and execution',
    type: 'manipulation',
    version: '1.5.0',
    status: 'deprecated',
    tags: ['welding', 'path', 'industrial'],
    accuracy: 88.5,
    latency: 200,
    invocations: 890,
  },
];

// Skill Card Component
function SkillCard({ skill, onInvoke, onView }) {
  const statusColors = {
    active: 'success',
    pending: 'warning',
    deprecated: 'danger',
  };

  const typeColors = {
    manipulation: 'text-blue-400',
    perception: 'text-purple-400',
    navigation: 'text-green-400',
    interaction: 'text-yellow-400',
  };

  return (
    <Card variant="elevated" className="hover:border-gray-600 transition-all">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={clsx('w-10 h-10 rounded-lg flex items-center justify-center',
            skill.type === 'manipulation' ? 'bg-blue-500/20' : 'bg-purple-500/20'
          )}>
            {skill.type === 'manipulation' ? (
              <Target className="w-5 h-5 text-blue-400" />
            ) : (
              <Eye className="w-5 h-5 text-purple-400" />
            )}
          </div>
          <div>
            <h3 className="font-semibold text-white">{skill.name}</h3>
            <p className={clsx('text-xs', typeColors[skill.type])}>{skill.type}</p>
          </div>
        </div>
        <Badge variant={statusColors[skill.status]} size="sm" dot>
          {skill.status}
        </Badge>
      </div>

      <p className="text-sm text-gray-400 mb-3 line-clamp-2">{skill.description}</p>

      <div className="flex flex-wrap gap-1 mb-3">
        {skill.tags.slice(0, 3).map((tag) => (
          <span key={tag} className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300">
            {tag}
          </span>
        ))}
        {skill.tags.length > 3 && (
          <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-500">
            +{skill.tags.length - 3}
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-2 mb-4 text-center">
        <div className="p-2 bg-gray-800 rounded">
          <div className="text-sm font-bold text-green-400">{skill.accuracy}%</div>
          <div className="text-xs text-gray-500">Accuracy</div>
        </div>
        <div className="p-2 bg-gray-800 rounded">
          <div className="text-sm font-bold text-blue-400">{skill.latency}ms</div>
          <div className="text-xs text-gray-500">Latency</div>
        </div>
        <div className="p-2 bg-gray-800 rounded">
          <div className="text-sm font-bold text-purple-400">
            {skill.invocations > 1000 ? `${(skill.invocations / 1000).toFixed(1)}k` : skill.invocations}
          </div>
          <div className="text-xs text-gray-500">Invocations</div>
        </div>
      </div>

      <div className="flex gap-2">
        <Button variant="primary" size="sm" icon={Play} onClick={() => onInvoke(skill)} className="flex-1">
          Invoke
        </Button>
        <Button variant="secondary" size="sm" icon={Eye} onClick={() => onView(skill)}>
          View
        </Button>
      </div>
    </Card>
  );
}

// Task Decomposition Graph Component
function TaskDecompositionGraph({ task, decomposition }) {
  if (!decomposition) return null;

  return (
    <div className="p-4 bg-gray-900 rounded-lg">
      {/* Root Task */}
      <div className="flex items-center gap-3 mb-6 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
        <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
          <Target className="w-5 h-5 text-blue-400" />
        </div>
        <div className="flex-1">
          <div className="font-medium text-white">{task}</div>
          <div className="text-xs text-gray-400">Root Task</div>
        </div>
        <Badge variant="primary">Planning</Badge>
      </div>

      {/* Decomposition Tree */}
      <div className="ml-8 border-l-2 border-gray-700 pl-6 space-y-4">
        {decomposition.steps?.map((step, index) => (
          <div key={index} className="relative">
            {/* Connection dot */}
            <div className="absolute -left-8 top-4 w-3 h-3 rounded-full bg-gray-700 border-2 border-gray-600" />

            <div
              className={clsx(
                'p-3 rounded-lg border transition-all',
                step.status === 'completed'
                  ? 'bg-green-500/10 border-green-500/30'
                  : step.status === 'running'
                  ? 'bg-blue-500/10 border-blue-500/30'
                  : step.status === 'failed'
                  ? 'bg-red-500/10 border-red-500/30'
                  : 'bg-gray-800 border-gray-700'
              )}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Step {index + 1}</span>
                  <span className="font-medium text-white">{step.skill_name}</span>
                </div>
                <div className="flex items-center gap-2">
                  {step.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-400" />}
                  {step.status === 'running' && <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />}
                  {step.status === 'failed' && <XCircle className="w-4 h-4 text-red-400" />}
                  {step.status === 'pending' && <Clock className="w-4 h-4 text-gray-400" />}
                  <Badge
                    variant={
                      step.status === 'completed'
                        ? 'success'
                        : step.status === 'running'
                        ? 'primary'
                        : step.status === 'failed'
                        ? 'danger'
                        : 'default'
                    }
                    size="sm"
                  >
                    {step.status}
                  </Badge>
                </div>
              </div>

              {step.moe_weight && (
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs text-gray-500">MoE Weight:</span>
                  <ProgressBar value={step.moe_weight * 100} variant="gradient" size="sm" className="flex-1" />
                  <span className="text-xs text-blue-400">{(step.moe_weight * 100).toFixed(1)}%</span>
                </div>
              )}

              {step.dependencies?.length > 0 && (
                <div className="flex items-center gap-1 mt-2 text-xs text-gray-500">
                  <GitBranch className="w-3 h-3" />
                  Depends on: {step.dependencies.join(', ')}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// MoE Routing Panel Component
function MoERoutingPanel({ routingResult }) {
  if (!routingResult) return null;

  return (
    <Card variant="elevated">
      <CardHeader>
        <CardTitle icon={Brain} iconColor="text-purple-400">
          MoE Routing Decision
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 mb-4 text-sm">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-gray-400" />
            <span className="text-gray-400">Inference:</span>
            <span className="text-white font-medium">{routingResult.inference_time_ms?.toFixed(2)}ms</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-gray-400" />
            <span className="text-gray-400">Candidates:</span>
            <span className="text-white font-medium">{routingResult.candidates?.length || 0}</span>
          </div>
        </div>

        <div className="space-y-2">
          {routingResult.candidates?.map((candidate, index) => (
            <div
              key={candidate.skill_id}
              className={clsx(
                'p-3 rounded-lg border flex items-center justify-between',
                index === 0
                  ? 'bg-green-500/10 border-green-500/30'
                  : 'bg-gray-800 border-gray-700'
              )}
            >
              <div className="flex items-center gap-3">
                {index === 0 && <Star className="w-4 h-4 text-yellow-400" />}
                <div>
                  <div className="font-medium text-white">{candidate.skill_name}</div>
                  <div className="text-xs text-gray-500">{candidate.skill_id}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <ProgressBar value={candidate.weight * 100} variant={index === 0 ? 'success' : 'default'} size="sm" className="w-24" />
                <span className={clsx('text-sm font-medium', index === 0 ? 'text-green-400' : 'text-gray-400')}>
                  {(candidate.weight * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>

        {routingResult.reasoning && (
          <Alert variant="info" className="mt-4">
            <strong>Routing Rationale:</strong> {routingResult.reasoning}
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}

// Main Skill Orchestration Page
export default function SkillOrchestration() {
  const [activeTab, setActiveTab] = useState('library');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [taskInput, setTaskInput] = useState('');
  const [orchestrating, setOrchestrating] = useState(false);
  const [orchestrationResult, setOrchestrationResult] = useState(null);
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);

  // Filter skills
  const filteredSkills = MOCK_SKILLS.filter((skill) => {
    const matchesSearch =
      skill.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      skill.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      skill.tags.some((tag) => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesType = filterType === 'all' || skill.type === filterType;
    const matchesStatus = filterStatus === 'all' || skill.status === filterStatus;
    return matchesSearch && matchesType && matchesStatus;
  });

  // Library statistics
  const libraryStats = {
    total: MOCK_SKILLS.length,
    active: MOCK_SKILLS.filter((s) => s.status === 'active').length,
    totalInvocations: MOCK_SKILLS.reduce((sum, s) => sum + s.invocations, 0),
    avgAccuracy: (MOCK_SKILLS.reduce((sum, s) => sum + s.accuracy, 0) / MOCK_SKILLS.length).toFixed(1),
  };

  // Handle task orchestration
  const handleOrchestrate = async () => {
    if (!taskInput.trim()) return;

    setOrchestrating(true);
    setOrchestrationResult(null);

    // Simulate API call
    await new Promise((r) => setTimeout(r, 1500));

    // Mock orchestration result
    setOrchestrationResult({
      task: taskInput,
      decomposition: {
        steps: [
          { skill_name: 'Bin Picking', skill_id: 'skill-003', status: 'completed', moe_weight: 0.92, dependencies: [] },
          { skill_name: 'Surface Inspection', skill_id: 'skill-005', status: 'completed', moe_weight: 0.88, dependencies: ['skill-003'] },
          { skill_name: 'Pick and Place', skill_id: 'skill-001', status: 'running', moe_weight: 0.95, dependencies: ['skill-005'] },
          { skill_name: 'Precision Assembly', skill_id: 'skill-002', status: 'pending', moe_weight: 0.85, dependencies: ['skill-001'] },
        ],
      },
      routing: {
        inference_time_ms: 12.5,
        candidates: [
          { skill_id: 'skill-001', skill_name: 'Pick and Place', weight: 0.45 },
          { skill_id: 'skill-002', skill_name: 'Precision Assembly', weight: 0.32 },
          { skill_id: 'skill-003', skill_name: 'Bin Picking', weight: 0.23 },
        ],
        reasoning: 'Selected Pick and Place as primary skill due to high accuracy for the target object type and optimal latency characteristics.',
      },
      estimated_duration: '45s',
      confidence: 0.89,
    });

    setOrchestrating(false);
  };

  const handleInvokeSkill = (skill) => {
    setTaskInput(`Execute skill: ${skill.name}`);
    setActiveTab('orchestration');
  };

  const handleViewSkill = (skill) => {
    setSelectedSkill(skill);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <Workflow className="text-yellow-400" />
            Skills & Orchestration
          </h1>
          <p className="text-gray-400 mt-1">
            MoE skill library and task decomposition
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" icon={Download}>
            Export
          </Button>
          <Button variant="primary" size="sm" icon={Upload} onClick={() => setShowUploadModal(true)}>
            Upload Skill
          </Button>
        </div>
      </div>

      {/* Library Statistics */}
      <div className="grid grid-cols-4 gap-4">
        <KPICard
          title="Total Skills"
          value={libraryStats.total}
          subvalue={`${libraryStats.active} active`}
          icon={Zap}
          iconColor="text-yellow-400"
          iconBg="bg-yellow-500/10"
        />
        <KPICard
          title="Total Invocations"
          value={libraryStats.totalInvocations > 1000 ? `${(libraryStats.totalInvocations / 1000).toFixed(1)}k` : libraryStats.totalInvocations}
          subvalue="all time"
          icon={Activity}
          iconColor="text-blue-400"
          iconBg="bg-blue-500/10"
        />
        <KPICard
          title="Average Accuracy"
          value={`${libraryStats.avgAccuracy}%`}
          change="+0.5% this week"
          changeType="positive"
          icon={Target}
          iconColor="text-green-400"
          iconBg="bg-green-500/10"
        />
        <KPICard
          title="MoE Router"
          value="Active"
          subvalue="Load balanced"
          icon={Brain}
          iconColor="text-purple-400"
          iconBg="bg-purple-500/10"
        />
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onChange={setActiveTab}>
        <Tab value="library" icon={Layers} badge={MOCK_SKILLS.length}>Skill Library</Tab>
        <Tab value="orchestration" icon={Workflow}>Task Orchestration</Tab>
        <Tab value="monitoring" icon={BarChart3}>Execution Monitor</Tab>
      </Tabs>

      {/* Library Tab */}
      {activeTab === 'library' && (
        <div className="space-y-4">
          {/* Filters */}
          <Card variant="default" padding="sm">
            <div className="flex items-center gap-4">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search skills by name, description, or tags..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All Types</option>
                <option value="manipulation">Manipulation</option>
                <option value="perception">Perception</option>
                <option value="navigation">Navigation</option>
              </select>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="pending">Pending</option>
                <option value="deprecated">Deprecated</option>
              </select>
            </div>
          </Card>

          {/* Skills Grid */}
          <div className="grid grid-cols-3 gap-4">
            {filteredSkills.map((skill) => (
              <SkillCard
                key={skill.id}
                skill={skill}
                onInvoke={handleInvokeSkill}
                onView={handleViewSkill}
              />
            ))}
          </div>

          {filteredSkills.length === 0 && (
            <EmptyState
              icon={Zap}
              title="No skills found"
              description="Try adjusting your filters or upload a new skill"
              action={
                <Button variant="primary" icon={Upload} onClick={() => setShowUploadModal(true)}>
                  Upload Skill
                </Button>
              }
            />
          )}
        </div>
      )}

      {/* Orchestration Tab */}
      {activeTab === 'orchestration' && (
        <div className="grid grid-cols-12 gap-6">
          {/* Task Input */}
          <div className="col-span-12">
            <Card variant="elevated">
              <CardHeader>
                <CardTitle icon={Target} iconColor="text-blue-400">
                  Task Input
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4">
                  <input
                    type="text"
                    placeholder="Describe your task (e.g., 'Pick up parts from bin, inspect for defects, and assemble on pallet')"
                    value={taskInput}
                    onChange={(e) => setTaskInput(e.target.value)}
                    className="flex-1 px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  />
                  <Button
                    variant="primary"
                    size="lg"
                    icon={orchestrating ? RefreshCw : Play}
                    onClick={handleOrchestrate}
                    disabled={!taskInput.trim() || orchestrating}
                  >
                    {orchestrating ? 'Planning...' : 'Orchestrate'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Decomposition Graph */}
          <div className="col-span-7">
            <Card variant="elevated">
              <CardHeader>
                <CardTitle icon={GitBranch} iconColor="text-green-400">
                  Task Decomposition
                </CardTitle>
              </CardHeader>
              <CardContent>
                {orchestrationResult ? (
                  <TaskDecompositionGraph
                    task={orchestrationResult.task}
                    decomposition={orchestrationResult.decomposition}
                  />
                ) : (
                  <EmptyState
                    icon={Workflow}
                    title="No orchestration result"
                    description="Enter a task and click 'Orchestrate' to see the decomposition"
                  />
                )}
              </CardContent>
            </Card>
          </div>

          {/* MoE Routing */}
          <div className="col-span-5">
            {orchestrationResult && (
              <>
                <MoERoutingPanel routingResult={orchestrationResult.routing} />
                <Card variant="default" padding="sm" className="mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Estimated Duration</div>
                      <div className="text-lg font-bold text-white">{orchestrationResult.estimated_duration}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Confidence</div>
                      <div className="text-lg font-bold text-green-400">{(orchestrationResult.confidence * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </Card>
              </>
            )}
          </div>
        </div>
      )}

      {/* Monitoring Tab */}
      {activeTab === 'monitoring' && (
        <Card variant="elevated">
          <CardHeader>
            <CardTitle icon={BarChart3} iconColor="text-cyan-400">
              Execution Monitor
            </CardTitle>
          </CardHeader>
          <CardContent>
            <EmptyState
              icon={Activity}
              title="No active executions"
              description="Start a task orchestration to monitor execution progress"
              action={
                <Button variant="primary" icon={Play} onClick={() => setActiveTab('orchestration')}>
                  Start Orchestration
                </Button>
              }
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

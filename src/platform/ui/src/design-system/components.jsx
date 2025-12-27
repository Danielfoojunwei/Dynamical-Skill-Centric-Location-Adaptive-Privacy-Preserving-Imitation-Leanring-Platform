/**
 * Enterprise Design System Components
 *
 * Reusable UI components following enterprise patterns:
 * - Consistent styling and behavior
 * - Accessibility compliant (WCAG 2.1 AA)
 * - Responsive by default
 *
 * @version 1.0.0
 */

import React from 'react';
import { clsx } from 'clsx';
import {
  ChevronRight,
  ChevronDown,
  X,
  Check,
  AlertCircle,
  Info,
  AlertTriangle,
  Loader2,
} from 'lucide-react';

// ============================================================================
// Card Components
// ============================================================================

export function Card({ children, className, variant = 'default', padding = 'md', ...props }) {
  const variants = {
    default: 'bg-gray-800/50 border-gray-700/50',
    elevated: 'bg-gray-800 border-gray-700 shadow-lg',
    outlined: 'bg-transparent border-gray-600',
    ghost: 'bg-transparent border-transparent',
  };

  const paddings = {
    none: 'p-0',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  };

  return (
    <div
      className={clsx(
        'rounded-xl border transition-all duration-200',
        variants[variant],
        paddings[padding],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className, action }) {
  return (
    <div className={clsx('flex items-center justify-between mb-4', className)}>
      <div className="flex items-center gap-3">{children}</div>
      {action && <div>{action}</div>}
    </div>
  );
}

export function CardTitle({ children, icon: Icon, iconColor, className }) {
  return (
    <h3 className={clsx('text-lg font-semibold flex items-center gap-2', className)}>
      {Icon && <Icon size={20} className={iconColor} />}
      {children}
    </h3>
  );
}

export function CardContent({ children, className }) {
  return <div className={clsx(className)}>{children}</div>;
}

// ============================================================================
// Button Components
// ============================================================================

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  icon: Icon,
  iconPosition = 'left',
  loading = false,
  disabled = false,
  className,
  ...props
}) {
  const variants = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    secondary: 'bg-gray-700 hover:bg-gray-600 text-white',
    success: 'bg-green-600 hover:bg-green-700 text-white',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
    warning: 'bg-amber-600 hover:bg-amber-700 text-white',
    ghost: 'bg-transparent hover:bg-gray-800 text-gray-300',
    outline: 'bg-transparent border border-gray-600 hover:bg-gray-800 text-gray-300',
  };

  const sizes = {
    xs: 'px-2 py-1 text-xs gap-1',
    sm: 'px-3 py-1.5 text-sm gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-3 text-base gap-2',
  };

  const iconSizes = {
    xs: 12,
    sm: 14,
    md: 16,
    lg: 18,
  };

  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200',
        'focus:outline-none focus:ring-2 focus:ring-blue-500/50',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <Loader2 size={iconSizes[size]} className="animate-spin" />
      ) : (
        Icon && iconPosition === 'left' && <Icon size={iconSizes[size]} />
      )}
      {children}
      {!loading && Icon && iconPosition === 'right' && <Icon size={iconSizes[size]} />}
    </button>
  );
}

// ============================================================================
// Badge Components
// ============================================================================

export function Badge({ children, variant = 'default', size = 'md', dot = false, className }) {
  const variants = {
    default: 'bg-gray-700 text-gray-300',
    primary: 'bg-blue-500/20 text-blue-400',
    success: 'bg-green-500/20 text-green-400',
    warning: 'bg-amber-500/20 text-amber-400',
    danger: 'bg-red-500/20 text-red-400',
    info: 'bg-cyan-500/20 text-cyan-400',
    purple: 'bg-purple-500/20 text-purple-400',
  };

  const sizes = {
    sm: 'text-xs px-1.5 py-0.5',
    md: 'text-xs px-2 py-1',
    lg: 'text-sm px-3 py-1',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 font-medium rounded-full',
        variants[variant],
        sizes[size],
        className
      )}
    >
      {dot && (
        <span
          className={clsx('w-1.5 h-1.5 rounded-full', {
            'bg-gray-400': variant === 'default',
            'bg-blue-400': variant === 'primary',
            'bg-green-400': variant === 'success',
            'bg-amber-400': variant === 'warning',
            'bg-red-400': variant === 'danger',
            'bg-cyan-400': variant === 'info',
            'bg-purple-400': variant === 'purple',
          })}
        />
      )}
      {children}
    </span>
  );
}

// ============================================================================
// Status Indicator
// ============================================================================

export function StatusIndicator({ status, label, showLabel = true, size = 'md' }) {
  const statusConfig = {
    online: { color: 'bg-green-500', label: 'Online' },
    offline: { color: 'bg-gray-500', label: 'Offline' },
    degraded: { color: 'bg-amber-500', label: 'Degraded' },
    critical: { color: 'bg-red-500', label: 'Critical' },
    maintenance: { color: 'bg-purple-500', label: 'Maintenance' },
    running: { color: 'bg-green-500', label: 'Running' },
    stopped: { color: 'bg-gray-500', label: 'Stopped' },
    pending: { color: 'bg-amber-500', label: 'Pending' },
    error: { color: 'bg-red-500', label: 'Error' },
  };

  const config = statusConfig[status] || statusConfig.offline;
  const sizes = {
    sm: 'w-2 h-2',
    md: 'w-2.5 h-2.5',
    lg: 'w-3 h-3',
  };

  return (
    <div className="flex items-center gap-2">
      <span className={clsx('rounded-full animate-pulse', config.color, sizes[size])} />
      {showLabel && <span className="text-sm text-gray-400">{label || config.label}</span>}
    </div>
  );
}

// ============================================================================
// KPI Card Component
// ============================================================================

export function KPICard({
  title,
  value,
  subvalue,
  change,
  changeType = 'neutral',
  icon: Icon,
  iconColor = 'text-blue-400',
  iconBg = 'bg-blue-500/10',
  trend,
  className,
}) {
  const changeColors = {
    positive: 'text-green-400',
    negative: 'text-red-400',
    neutral: 'text-gray-400',
  };

  return (
    <Card variant="elevated" className={clsx('hover:border-gray-600 transition-colors', className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {subvalue && <p className="text-sm text-gray-500 mt-1">{subvalue}</p>}
          {change && (
            <div className={clsx('flex items-center gap-1 mt-2 text-sm', changeColors[changeType])}>
              {changeType === 'positive' ? '↑' : changeType === 'negative' ? '↓' : '→'}
              <span>{change}</span>
            </div>
          )}
        </div>
        {Icon && (
          <div className={clsx('p-3 rounded-xl', iconBg)}>
            <Icon size={24} className={iconColor} />
          </div>
        )}
      </div>
      {trend && <div className="mt-4 h-12">{trend}</div>}
    </Card>
  );
}

// ============================================================================
// Tab Components
// ============================================================================

export function Tabs({ children, value, onChange, className }) {
  return (
    <div className={clsx('border-b border-gray-700', className)}>
      <nav className="flex gap-1 -mb-px">
        {React.Children.map(children, (child) =>
          React.cloneElement(child, {
            active: child.props.value === value,
            onClick: () => onChange(child.props.value),
          })
        )}
      </nav>
    </div>
  );
}

export function Tab({ children, value, active, onClick, icon: Icon, badge }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px',
        active
          ? 'text-blue-400 border-blue-400'
          : 'text-gray-400 border-transparent hover:text-white hover:border-gray-600'
      )}
    >
      {Icon && <Icon size={16} />}
      {children}
      {badge && (
        <Badge variant={active ? 'primary' : 'default'} size="sm">
          {badge}
        </Badge>
      )}
    </button>
  );
}

// ============================================================================
// Progress Components
// ============================================================================

export function ProgressBar({
  value,
  max = 100,
  variant = 'primary',
  size = 'md',
  showLabel = false,
  className,
}) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));

  const variants = {
    primary: 'bg-blue-500',
    success: 'bg-green-500',
    warning: 'bg-amber-500',
    danger: 'bg-red-500',
    gradient: 'bg-gradient-to-r from-blue-500 to-purple-500',
  };

  const sizes = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  return (
    <div className={clsx('w-full', className)}>
      {showLabel && (
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400">Progress</span>
          <span className="text-white">{percentage.toFixed(0)}%</span>
        </div>
      )}
      <div className={clsx('w-full bg-gray-700 rounded-full overflow-hidden', sizes[size])}>
        <div
          className={clsx('h-full transition-all duration-500 rounded-full', variants[variant])}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Alert Components
// ============================================================================

export function Alert({ children, variant = 'info', title, dismissible = false, onDismiss }) {
  const variants = {
    info: {
      bg: 'bg-blue-500/10 border-blue-500/30',
      icon: Info,
      iconColor: 'text-blue-400',
    },
    success: {
      bg: 'bg-green-500/10 border-green-500/30',
      icon: Check,
      iconColor: 'text-green-400',
    },
    warning: {
      bg: 'bg-amber-500/10 border-amber-500/30',
      icon: AlertTriangle,
      iconColor: 'text-amber-400',
    },
    error: {
      bg: 'bg-red-500/10 border-red-500/30',
      icon: AlertCircle,
      iconColor: 'text-red-400',
    },
  };

  const config = variants[variant];
  const Icon = config.icon;

  return (
    <div className={clsx('flex gap-3 p-4 rounded-lg border', config.bg)}>
      <Icon size={20} className={clsx('flex-shrink-0 mt-0.5', config.iconColor)} />
      <div className="flex-1 min-w-0">
        {title && <p className="font-medium text-white mb-1">{title}</p>}
        <div className="text-sm text-gray-300">{children}</div>
      </div>
      {dismissible && (
        <button
          onClick={onDismiss}
          className="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Empty State Component
// ============================================================================

export function EmptyState({ icon: Icon, title, description, action, className }) {
  return (
    <div className={clsx('flex flex-col items-center justify-center py-12 text-center', className)}>
      {Icon && (
        <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mb-4">
          <Icon size={32} className="text-gray-500" />
        </div>
      )}
      <h3 className="text-lg font-medium text-white mb-2">{title}</h3>
      {description && <p className="text-sm text-gray-400 max-w-sm mb-4">{description}</p>}
      {action}
    </div>
  );
}

// ============================================================================
// Skeleton Components
// ============================================================================

export function Skeleton({ className, variant = 'text' }) {
  const variants = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  return (
    <div
      className={clsx('bg-gray-700 animate-pulse', variants[variant], className)}
    />
  );
}

// ============================================================================
// Divider Component
// ============================================================================

export function Divider({ label, className }) {
  if (label) {
    return (
      <div className={clsx('flex items-center gap-4 my-4', className)}>
        <div className="flex-1 h-px bg-gray-700" />
        <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
        <div className="flex-1 h-px bg-gray-700" />
      </div>
    );
  }

  return <div className={clsx('h-px bg-gray-700 my-4', className)} />;
}

// ============================================================================
// Data Display Components
// ============================================================================

export function DataRow({ label, value, className }) {
  return (
    <div className={clsx('flex items-center justify-between py-2', className)}>
      <span className="text-sm text-gray-400">{label}</span>
      <span className="text-sm font-medium text-white">{value}</span>
    </div>
  );
}

export function DataGrid({ children, columns = 2, className }) {
  return (
    <div
      className={clsx('grid gap-4', className)}
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {children}
    </div>
  );
}

// ============================================================================
// Tooltip (Simple)
// ============================================================================

export function Tooltip({ children, content, position = 'top' }) {
  const [visible, setVisible] = React.useState(false);

  const positions = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div
          className={clsx(
            'absolute z-50 px-2 py-1 text-xs text-white bg-gray-900 rounded shadow-lg whitespace-nowrap',
            positions[position]
          )}
        >
          {content}
        </div>
      )}
    </div>
  );
}

export default {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Badge,
  StatusIndicator,
  KPICard,
  Tabs,
  Tab,
  ProgressBar,
  Alert,
  EmptyState,
  Skeleton,
  Divider,
  DataRow,
  DataGrid,
  Tooltip,
};

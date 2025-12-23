/**
 * AlertCenter - Centralized Notification Panel
 *
 * Displays system alerts, warnings, and errors in a dropdown panel.
 * Shows badge count for unread notifications.
 *
 * Features:
 * - Grouped by severity (error, warning, info)
 * - Mark as read / clear all
 * - Expandable notification details
 * - Click to navigate to source
 */

import React, { useState } from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import {
  Bell,
  AlertCircle,
  AlertTriangle,
  Info,
  CheckCircle,
  X,
  Trash2,
  Check,
} from 'lucide-react';
import { clsx } from 'clsx';
import { useUiStore, selectUnreadNotificationCount } from '../../stores';

// Notification type icons and colors
const notificationTypes = {
  error: {
    icon: AlertCircle,
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
  },
  success: {
    icon: CheckCircle,
    color: 'text-green-400',
    bg: 'bg-green-500/10',
    border: 'border-green-500/30',
  },
  info: {
    icon: Info,
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
  },
};

// Format relative time
function formatRelativeTime(timestamp) {
  const now = Date.now();
  const diff = now - timestamp;

  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return new Date(timestamp).toLocaleDateString();
}

// Single notification item
function NotificationItem({
  notification,
  onMarkRead,
  onRemove,
}) {
  const typeConfig = notificationTypes[notification.type] || notificationTypes.info;
  const Icon = typeConfig.icon;

  return (
    <div
      className={clsx(
        'relative flex gap-3 p-3 rounded-lg border transition-colors',
        notification.read
          ? 'bg-gray-800/30 border-gray-700'
          : clsx(typeConfig.bg, typeConfig.border)
      )}
    >
      {/* Icon */}
      <div className={clsx('flex-shrink-0 mt-0.5', typeConfig.color)}>
        <Icon className="w-5 h-5" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="font-medium text-sm text-white">
            {notification.title}
          </p>
          {!notification.read && (
            <span className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-1.5" />
          )}
        </div>

        {notification.message && (
          <p className="text-sm text-gray-400 mt-1 line-clamp-2">
            {notification.message}
          </p>
        )}

        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-gray-500">
            {formatRelativeTime(notification.timestamp)}
          </span>

          <div className="flex items-center gap-1">
            {!notification.read && (
              <button
                onClick={() => onMarkRead(notification.id)}
                className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-white transition"
                title="Mark as read"
              >
                <Check className="w-3.5 h-3.5" />
              </button>
            )}
            <button
              onClick={() => onRemove(notification.id)}
              className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-red-400 transition"
              title="Remove"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main AlertCenter component
export function AlertCenter({ className = '' }) {
  const [isOpen, setIsOpen] = useState(false);

  const notifications = useUiStore((state) => state.notifications);
  const unreadCount = useUiStore(selectUnreadNotificationCount);
  const markNotificationRead = useUiStore((state) => state.markNotificationRead);
  const markAllNotificationsRead = useUiStore((state) => state.markAllNotificationsRead);
  const removeNotification = useUiStore((state) => state.removeNotification);
  const clearNotifications = useUiStore((state) => state.clearNotifications);

  return (
    <DropdownMenu.Root open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenu.Trigger asChild>
        <button
          className={clsx(
            'relative p-2 rounded-lg transition-colors',
            isOpen ? 'bg-gray-700' : 'hover:bg-gray-800',
            className
          )}
          aria-label="Notifications"
        >
          <Bell className="w-5 h-5 text-gray-400" />

          {/* Badge */}
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 flex items-center justify-center min-w-[18px] h-[18px] px-1 text-xs font-bold text-white bg-red-500 rounded-full">
              {unreadCount > 99 ? '99+' : unreadCount}
            </span>
          )}
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="w-96 max-h-[70vh] overflow-hidden bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-50"
          align="end"
          sideOffset={8}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
            <h3 className="font-semibold text-white">Notifications</h3>
            <div className="flex items-center gap-2">
              {unreadCount > 0 && (
                <button
                  onClick={markAllNotificationsRead}
                  className="text-xs text-blue-400 hover:text-blue-300 transition"
                >
                  Mark all read
                </button>
              )}
              {notifications.length > 0 && (
                <button
                  onClick={clearNotifications}
                  className="p-1.5 rounded hover:bg-gray-800 text-gray-500 hover:text-red-400 transition"
                  title="Clear all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Notifications list */}
          <div className="overflow-y-auto max-h-[calc(70vh-60px)] p-2 space-y-2">
            {notifications.length === 0 ? (
              <div className="py-8 text-center">
                <Bell className="w-8 h-8 text-gray-700 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">No notifications</p>
              </div>
            ) : (
              notifications.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onMarkRead={markNotificationRead}
                  onRemove={removeNotification}
                />
              ))
            )}
          </div>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

export default AlertCenter;

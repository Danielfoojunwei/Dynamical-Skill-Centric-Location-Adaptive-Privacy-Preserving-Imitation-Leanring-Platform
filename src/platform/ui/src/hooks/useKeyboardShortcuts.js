/**
 * Keyboard Shortcuts Hook
 *
 * Provides global keyboard shortcuts for the application.
 * Uses react-hotkeys-hook for cross-platform key handling.
 *
 * Shortcuts:
 * - Ctrl/Cmd + S: Toggle system start/stop
 * - Ctrl/Cmd + R: Toggle recording
 * - Ctrl/Cmd + T: Toggle teleoperation
 * - Ctrl/Cmd + 1-9: Navigate to pages
 * - Escape: Close modals/fullscreen
 * - ?: Show help
 */

import { useHotkeys } from 'react-hotkeys-hook';
import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUiStore, useRobotStore, useSystemStore } from '../stores';

// Shortcut definitions
export const SHORTCUTS = {
  // System control
  TOGGLE_SYSTEM: { key: 'mod+s', description: 'Start/Stop System' },
  EMERGENCY_STOP: { key: 'mod+shift+x', description: 'Emergency Stop' },

  // Recording
  TOGGLE_RECORDING: { key: 'mod+r', description: 'Toggle Recording' },

  // Teleoperation
  TOGGLE_TELEOP: { key: 'mod+t', description: 'Toggle Teleoperation' },

  // Navigation
  NAV_DASHBOARD: { key: 'mod+1', description: 'Go to Dashboard' },
  NAV_PERCEPTION: { key: 'mod+2', description: 'Go to Perception' },
  NAV_SIMULATION: { key: 'mod+3', description: 'Go to Simulation' },
  NAV_DEVICES: { key: 'mod+4', description: 'Go to Devices' },
  NAV_SKILLS: { key: 'mod+5', description: 'Go to Skills' },
  NAV_TRAINING: { key: 'mod+6', description: 'Go to Training' },
  NAV_SAFETY: { key: 'mod+7', description: 'Go to Safety' },
  NAV_SETTINGS: { key: 'mod+0', description: 'Go to Settings' },

  // UI
  TOGGLE_SIDEBAR: { key: 'mod+b', description: 'Toggle Sidebar' },
  TOGGLE_THEME: { key: 'mod+shift+t', description: 'Toggle Theme' },
  CLOSE_MODAL: { key: 'escape', description: 'Close Modal' },
  SHOW_HELP: { key: 'shift+?', description: 'Show Shortcuts Help' },
};

// Navigation routes
const NAV_ROUTES = {
  1: '/',
  2: '/perception',
  3: '/simulation',
  4: '/devices',
  5: '/skills',
  6: '/training',
  7: '/safety',
  0: '/settings',
};

/**
 * Global keyboard shortcuts hook.
 *
 * @param {Object} options - Options
 * @param {Function} options.onSystemToggle - System start/stop callback
 * @param {Function} options.onEmergencyStop - Emergency stop callback
 */
export function useKeyboardShortcuts(options = {}) {
  const { onSystemToggle, onEmergencyStop } = options;

  // Navigation
  let navigate;
  try {
    navigate = useNavigate();
  } catch {
    // Not within router context
    navigate = null;
  }

  // Store actions
  const shortcutsEnabled = useUiStore((state) => state.shortcutsEnabled);
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);
  const toggleTheme = useUiStore((state) => state.toggleTheme);
  const closeModal = useUiStore((state) => state.closeModal);
  const openModal = useUiStore((state) => state.openModal);
  const addToast = useUiStore((state) => state.addToast);

  const teleopEnabled = useRobotStore((state) => state.teleopEnabled);
  const enableTeleop = useRobotStore((state) => state.enableTeleop);
  const disableTeleop = useRobotStore((state) => state.disableTeleop);

  const isRecording = useRobotStore((state) => state.isRecording);
  const startRecording = useRobotStore((state) => state.startRecording);
  const stopRecording = useRobotStore((state) => state.stopRecording);

  const systemStatus = useSystemStore((state) => state.status);

  // Helper to check if shortcuts are enabled
  const shouldHandle = useCallback(() => {
    return shortcutsEnabled;
  }, [shortcutsEnabled]);

  // System toggle
  useHotkeys(
    SHORTCUTS.TOGGLE_SYSTEM.key,
    (e) => {
      e.preventDefault();
      if (!shouldHandle()) return;
      onSystemToggle?.();
      addToast({
        type: 'info',
        message: systemStatus === 'OPERATIONAL' ? 'Stopping system...' : 'Starting system...',
      });
    },
    { enableOnFormTags: false },
    [shouldHandle, onSystemToggle, systemStatus, addToast]
  );

  // Emergency stop
  useHotkeys(
    SHORTCUTS.EMERGENCY_STOP.key,
    (e) => {
      e.preventDefault();
      onEmergencyStop?.();
      addToast({
        type: 'error',
        title: 'Emergency Stop',
        message: 'All robot motion halted',
      });
    },
    { enableOnFormTags: true }, // Always allow emergency stop
    [onEmergencyStop, addToast]
  );

  // Toggle recording
  useHotkeys(
    SHORTCUTS.TOGGLE_RECORDING.key,
    (e) => {
      e.preventDefault();
      if (!shouldHandle()) return;
      if (isRecording) {
        stopRecording();
        addToast({ type: 'success', message: 'Recording stopped' });
      } else {
        startRecording(`rec_${Date.now()}`);
        addToast({ type: 'info', message: 'Recording started' });
      }
    },
    { enableOnFormTags: false },
    [shouldHandle, isRecording, startRecording, stopRecording, addToast]
  );

  // Toggle teleop
  useHotkeys(
    SHORTCUTS.TOGGLE_TELEOP.key,
    (e) => {
      e.preventDefault();
      if (!shouldHandle()) return;
      if (teleopEnabled) {
        disableTeleop();
        addToast({ type: 'info', message: 'Teleoperation disabled' });
      } else {
        enableTeleop('glove');
        addToast({ type: 'success', message: 'Teleoperation enabled' });
      }
    },
    { enableOnFormTags: false },
    [shouldHandle, teleopEnabled, enableTeleop, disableTeleop, addToast]
  );

  // Navigation shortcuts
  Object.entries(NAV_ROUTES).forEach(([key, route]) => {
    useHotkeys(
      `mod+${key}`,
      (e) => {
        e.preventDefault();
        if (!shouldHandle() || !navigate) return;
        navigate(route);
      },
      { enableOnFormTags: false },
      [shouldHandle, navigate]
    );
  });

  // Toggle sidebar
  useHotkeys(
    SHORTCUTS.TOGGLE_SIDEBAR.key,
    (e) => {
      e.preventDefault();
      if (!shouldHandle()) return;
      toggleSidebar();
    },
    { enableOnFormTags: false },
    [shouldHandle, toggleSidebar]
  );

  // Toggle theme
  useHotkeys(
    SHORTCUTS.TOGGLE_THEME.key,
    (e) => {
      e.preventDefault();
      if (!shouldHandle()) return;
      toggleTheme();
    },
    { enableOnFormTags: false },
    [shouldHandle, toggleTheme]
  );

  // Close modal / escape
  useHotkeys(
    SHORTCUTS.CLOSE_MODAL.key,
    (e) => {
      closeModal();
    },
    { enableOnFormTags: true },
    [closeModal]
  );

  // Show help
  useHotkeys(
    SHORTCUTS.SHOW_HELP.key,
    (e) => {
      e.preventDefault();
      openModal('keyboard-shortcuts');
    },
    { enableOnFormTags: false },
    [openModal]
  );
}

/**
 * Get list of all shortcuts for display.
 */
export function getShortcutsList() {
  return Object.entries(SHORTCUTS).map(([id, shortcut]) => ({
    id,
    key: shortcut.key.replace('mod', '⌘/Ctrl'),
    description: shortcut.description,
  }));
}

export default useKeyboardShortcuts;

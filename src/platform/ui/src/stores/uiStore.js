/**
 * UI Store - User Interface State Management
 *
 * Manages UI-specific state including:
 * - Theme (dark/light)
 * - Notifications and toasts
 * - Modal dialogs
 * - Sidebar state
 * - Keyboard shortcuts
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

const initialState = {
  // Theme
  theme: 'dark',

  // Sidebar
  sidebarCollapsed: false,

  // Notifications
  notifications: [],
  maxNotifications: 50,

  // Toasts (temporary notifications)
  toasts: [],

  // Modals
  activeModal: null,
  modalData: null,

  // Loading states
  loadingStates: {},

  // Panel visibility
  panels: {
    metrics: true,
    cameras: true,
    robot3d: true,
    timeline: true,
  },

  // Keyboard shortcuts enabled
  shortcutsEnabled: true,

  // Connection indicator visibility
  showConnectionIndicator: true,
};

let toastIdCounter = 0;
let notificationIdCounter = 0;

export const useUiStore = create(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Theme
        setTheme: (theme) => set({ theme }),
        toggleTheme: () =>
          set((state) => ({
            theme: state.theme === 'dark' ? 'light' : 'dark',
          })),

        // Sidebar
        toggleSidebar: () =>
          set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
        setSidebarCollapsed: (collapsed) =>
          set({ sidebarCollapsed: collapsed }),

        // Toasts
        addToast: (toast) => {
          const id = ++toastIdCounter;
          const newToast = {
            id,
            type: toast.type || 'info', // info, success, warning, error
            title: toast.title,
            message: toast.message,
            duration: toast.duration ?? 5000,
            timestamp: Date.now(),
          };

          set((state) => ({
            toasts: [...state.toasts, newToast],
          }));

          // Auto-remove after duration
          if (newToast.duration > 0) {
            setTimeout(() => {
              get().removeToast(id);
            }, newToast.duration);
          }

          return id;
        },

        removeToast: (id) =>
          set((state) => ({
            toasts: state.toasts.filter((t) => t.id !== id),
          })),

        clearToasts: () => set({ toasts: [] }),

        // Convenience toast methods
        toast: {
          success: (message, title = 'Success') =>
            get().addToast({ type: 'success', title, message }),
          error: (message, title = 'Error') =>
            get().addToast({ type: 'error', title, message, duration: 8000 }),
          warning: (message, title = 'Warning') =>
            get().addToast({ type: 'warning', title, message }),
          info: (message, title = 'Info') =>
            get().addToast({ type: 'info', title, message }),
        },

        // Notifications (persistent)
        addNotification: (notification) => {
          const id = ++notificationIdCounter;
          const newNotification = {
            id,
            type: notification.type || 'info',
            title: notification.title,
            message: notification.message,
            read: false,
            timestamp: Date.now(),
            data: notification.data,
          };

          set((state) => ({
            notifications: [
              newNotification,
              ...state.notifications.slice(0, state.maxNotifications - 1),
            ],
          }));

          return id;
        },

        markNotificationRead: (id) =>
          set((state) => ({
            notifications: state.notifications.map((n) =>
              n.id === id ? { ...n, read: true } : n
            ),
          })),

        markAllNotificationsRead: () =>
          set((state) => ({
            notifications: state.notifications.map((n) => ({ ...n, read: true })),
          })),

        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
          })),

        clearNotifications: () => set({ notifications: [] }),

        // Modals
        openModal: (modalId, data = null) =>
          set({ activeModal: modalId, modalData: data }),

        closeModal: () => set({ activeModal: null, modalData: null }),

        // Loading states
        setLoading: (key, isLoading) =>
          set((state) => ({
            loadingStates: { ...state.loadingStates, [key]: isLoading },
          })),

        isLoading: (key) => get().loadingStates[key] ?? false,

        // Panels
        togglePanel: (panelId) =>
          set((state) => ({
            panels: {
              ...state.panels,
              [panelId]: !state.panels[panelId],
            },
          })),

        setPanelVisibility: (panelId, visible) =>
          set((state) => ({
            panels: { ...state.panels, [panelId]: visible },
          })),

        // Shortcuts
        setShortcutsEnabled: (enabled) => set({ shortcutsEnabled: enabled }),

        // Reset
        reset: () =>
          set({
            ...initialState,
            theme: get().theme, // Preserve theme
          }),
      }),
      {
        name: 'dynamical-ui-store',
        partialize: (state) => ({
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
          panels: state.panels,
          shortcutsEnabled: state.shortcutsEnabled,
        }),
      }
    ),
    { name: 'UiStore' }
  )
);

// Selectors
export const selectUnreadNotificationCount = (state) =>
  state.notifications.filter((n) => !n.read).length;

export const selectRecentNotifications = (limit = 5) => (state) =>
  state.notifications.slice(0, limit);

export const selectIsModalOpen = (modalId) => (state) =>
  state.activeModal === modalId;

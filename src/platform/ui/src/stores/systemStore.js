/**
 * System Store - Global System State Management
 *
 * Manages platform-wide state including:
 * - System status (IDLE, OPERATIONAL, ERROR)
 * - Resource utilization (TFLOPS, memory)
 * - Active components
 * - Connection status
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

const initialState = {
  // System status
  status: 'IDLE', // IDLE, OPERATIONAL, ERROR, STARTING, STOPPING
  isConnected: false,
  lastHeartbeat: null,

  // Resource metrics
  tflopsUsed: 0,
  tflopsTotal: 137.0,
  utilizationPercent: 0,
  memoryUsedGb: 0,
  memoryTotalGb: 32,
  uptimeSeconds: 0,

  // Components
  activeComponents: [],

  // Meta AI models
  metaAiModels: [
    { id: 'dinov3', name: 'DINOv3', tflops: 8.0, status: 'idle', latency: 0 },
    { id: 'sam3', name: 'SAM 3', tflops: 15.0, status: 'idle', latency: 0 },
    { id: 'vjepa2', name: 'V-JEPA 2', tflops: 10.0, status: 'idle', latency: 0 },
  ],

  // Safety
  safetyStatus: 'OK',
  activeHazards: [],

  // Errors
  errors: [],
  warnings: [],
};

export const useSystemStore = create(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Actions
        setStatus: (status) => set({ status }),

        setConnected: (isConnected) =>
          set({
            isConnected,
            lastHeartbeat: isConnected ? Date.now() : get().lastHeartbeat,
          }),

        updateMetrics: (metrics) =>
          set({
            tflopsUsed: metrics.tflops_used ?? get().tflopsUsed,
            tflopsTotal: metrics.tflops_total ?? get().tflopsTotal,
            utilizationPercent: metrics.utilization_percent ?? get().utilizationPercent,
            memoryUsedGb: metrics.memory_used_gb ?? get().memoryUsedGb,
            uptimeSeconds: metrics.uptime_seconds ?? get().uptimeSeconds,
            activeComponents: metrics.active_components ?? get().activeComponents,
            status: metrics.status ?? get().status,
          }),

        updateMetaAiModels: (models) => set({ metaAiModels: models }),

        setSafetyStatus: (status, hazards = []) =>
          set({ safetyStatus: status, activeHazards: hazards }),

        addError: (error) =>
          set((state) => ({
            errors: [
              { id: Date.now(), message: error, timestamp: new Date().toISOString() },
              ...state.errors.slice(0, 99), // Keep last 100
            ],
          })),

        addWarning: (warning) =>
          set((state) => ({
            warnings: [
              { id: Date.now(), message: warning, timestamp: new Date().toISOString() },
              ...state.warnings.slice(0, 99),
            ],
          })),

        clearError: (id) =>
          set((state) => ({
            errors: state.errors.filter((e) => e.id !== id),
          })),

        clearAllErrors: () => set({ errors: [], warnings: [] }),

        reset: () => set(initialState),
      }),
      {
        name: 'dynamical-system-store',
        partialize: (state) => ({
          // Only persist non-volatile settings
          tflopsTotal: state.tflopsTotal,
          memoryTotalGb: state.memoryTotalGb,
        }),
      }
    ),
    { name: 'SystemStore' }
  )
);

// Selectors
export const selectIsOperational = (state) => state.status === 'OPERATIONAL';
export const selectTflopsPercent = (state) =>
  state.tflopsTotal > 0 ? (state.tflopsUsed / state.tflopsTotal) * 100 : 0;
export const selectMemoryPercent = (state) =>
  state.memoryTotalGb > 0 ? (state.memoryUsedGb / state.memoryTotalGb) * 100 : 0;
export const selectActiveMetaAi = (state) =>
  state.metaAiModels.filter((m) => m.status === 'running');
export const selectMetaAiTflops = (state) =>
  state.metaAiModels
    .filter((m) => m.status === 'running')
    .reduce((sum, m) => sum + m.tflops, 0);

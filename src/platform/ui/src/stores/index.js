/**
 * Zustand Stores - Centralized State Management
 *
 * Exports all stores for the Dynamical Edge Platform UI.
 *
 * Usage:
 *   import { useSystemStore, useRobotStore, useUiStore } from './stores';
 *
 *   // In component
 *   const status = useSystemStore((state) => state.status);
 *   const { addToast } = useUiStore();
 */

// System store - platform-wide state
export {
  useSystemStore,
  selectIsOperational,
  selectTflopsPercent,
  selectMemoryPercent,
  selectActiveMetaAi,
  selectMetaAiTflops,
} from './systemStore';

// Robot store - robot state management
export {
  useRobotStore,
  selectActiveRobot,
  selectActiveRobotState,
  selectRobotList,
  selectIsAnyRobotMoving,
} from './robotStore';

// UI store - user interface state
export {
  useUiStore,
  selectUnreadNotificationCount,
  selectRecentNotifications,
  selectIsModalOpen,
} from './uiStore';

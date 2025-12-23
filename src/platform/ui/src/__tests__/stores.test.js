/**
 * Store Tests
 *
 * Tests for Zustand stores:
 * - systemStore
 * - robotStore
 * - uiStore
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import {
  useSystemStore,
  useRobotStore,
  useUiStore,
  selectTflopsPercent,
  selectActiveRobot,
  selectUnreadNotificationCount,
} from '../stores';

describe('SystemStore', () => {
  beforeEach(() => {
    useSystemStore.getState().reset();
  });

  it('should have correct initial state', () => {
    const state = useSystemStore.getState();
    expect(state.status).toBe('IDLE');
    expect(state.isConnected).toBe(false);
    expect(state.tflopsUsed).toBe(0);
    expect(state.tflopsTotal).toBe(137.0);
  });

  it('should update status', () => {
    act(() => {
      useSystemStore.getState().setStatus('OPERATIONAL');
    });
    expect(useSystemStore.getState().status).toBe('OPERATIONAL');
  });

  it('should update connection state', () => {
    act(() => {
      useSystemStore.getState().setConnected(true);
    });
    expect(useSystemStore.getState().isConnected).toBe(true);
    expect(useSystemStore.getState().lastHeartbeat).toBeTruthy();
  });

  it('should update metrics', () => {
    act(() => {
      useSystemStore.getState().updateMetrics({
        tflops_used: 50,
        tflops_total: 137,
        utilization_percent: 36.5,
        memory_used_gb: 16,
        uptime_seconds: 3600,
        status: 'OPERATIONAL',
      });
    });

    const state = useSystemStore.getState();
    expect(state.tflopsUsed).toBe(50);
    expect(state.utilizationPercent).toBe(36.5);
    expect(state.memoryUsedGb).toBe(16);
    expect(state.status).toBe('OPERATIONAL');
  });

  it('should add and clear errors', () => {
    act(() => {
      useSystemStore.getState().addError('Test error 1');
      useSystemStore.getState().addError('Test error 2');
    });

    expect(useSystemStore.getState().errors.length).toBe(2);
    expect(useSystemStore.getState().errors[0].message).toBe('Test error 2');

    act(() => {
      useSystemStore.getState().clearAllErrors();
    });

    expect(useSystemStore.getState().errors.length).toBe(0);
  });

  it('should calculate TFLOPS percent correctly', () => {
    act(() => {
      useSystemStore.getState().updateMetrics({
        tflops_used: 68.5,
        tflops_total: 137.0,
      });
    });

    const percent = selectTflopsPercent(useSystemStore.getState());
    expect(percent).toBe(50);
  });
});

describe('RobotStore', () => {
  beforeEach(() => {
    useRobotStore.getState().reset();
  });

  it('should have correct initial state', () => {
    const state = useRobotStore.getState();
    expect(Object.keys(state.robots).length).toBe(0);
    expect(state.activeRobotId).toBeNull();
    expect(state.teleopEnabled).toBe(false);
    expect(state.isRecording).toBe(false);
  });

  it('should register and unregister robots', () => {
    act(() => {
      useRobotStore.getState().registerRobot('robot_1', {
        name: 'Test Robot',
        type: 'franka',
      });
    });

    expect(useRobotStore.getState().robots['robot_1']).toBeDefined();
    expect(useRobotStore.getState().robots['robot_1'].name).toBe('Test Robot');
    expect(useRobotStore.getState().activeRobotId).toBe('robot_1');

    act(() => {
      useRobotStore.getState().unregisterRobot('robot_1');
    });

    expect(useRobotStore.getState().robots['robot_1']).toBeUndefined();
    expect(useRobotStore.getState().activeRobotId).toBeNull();
  });

  it('should update robot state', () => {
    act(() => {
      useRobotStore.getState().registerRobot('robot_1', {});
      useRobotStore.getState().updateRobotState('robot_1', {
        jointPositions: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        isMoving: true,
      });
    });

    const robot = useRobotStore.getState().robots['robot_1'];
    expect(robot.state.jointPositions).toEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
    expect(robot.state.isMoving).toBe(true);
  });

  it('should handle teleoperation', () => {
    act(() => {
      useRobotStore.getState().enableTeleop('glove');
    });

    expect(useRobotStore.getState().teleopEnabled).toBe(true);
    expect(useRobotStore.getState().teleopSource).toBe('glove');

    act(() => {
      useRobotStore.getState().disableTeleop();
    });

    expect(useRobotStore.getState().teleopEnabled).toBe(false);
    expect(useRobotStore.getState().teleopSource).toBeNull();
  });

  it('should handle recording state', () => {
    act(() => {
      useRobotStore.getState().startRecording('rec_001');
    });

    expect(useRobotStore.getState().isRecording).toBe(true);
    expect(useRobotStore.getState().recordingId).toBe('rec_001');

    act(() => {
      useRobotStore.getState().incrementRecordedFrames();
      useRobotStore.getState().incrementRecordedFrames();
    });

    expect(useRobotStore.getState().recordedFrames).toBe(2);

    act(() => {
      useRobotStore.getState().stopRecording();
    });

    expect(useRobotStore.getState().isRecording).toBe(false);
  });

  it('should select active robot', () => {
    act(() => {
      useRobotStore.getState().registerRobot('robot_1', { name: 'Robot 1' });
      useRobotStore.getState().registerRobot('robot_2', { name: 'Robot 2' });
      useRobotStore.getState().setActiveRobot('robot_2');
    });

    const activeRobot = selectActiveRobot(useRobotStore.getState());
    expect(activeRobot.name).toBe('Robot 2');
  });
});

describe('UiStore', () => {
  beforeEach(() => {
    useUiStore.getState().reset();
  });

  it('should have correct initial state', () => {
    const state = useUiStore.getState();
    expect(state.theme).toBe('dark');
    expect(state.sidebarCollapsed).toBe(false);
    expect(state.notifications.length).toBe(0);
    expect(state.toasts.length).toBe(0);
  });

  it('should toggle theme', () => {
    act(() => {
      useUiStore.getState().toggleTheme();
    });
    expect(useUiStore.getState().theme).toBe('light');

    act(() => {
      useUiStore.getState().toggleTheme();
    });
    expect(useUiStore.getState().theme).toBe('dark');
  });

  it('should toggle sidebar', () => {
    act(() => {
      useUiStore.getState().toggleSidebar();
    });
    expect(useUiStore.getState().sidebarCollapsed).toBe(true);

    act(() => {
      useUiStore.getState().toggleSidebar();
    });
    expect(useUiStore.getState().sidebarCollapsed).toBe(false);
  });

  it('should add and remove toasts', () => {
    let toastId;
    act(() => {
      toastId = useUiStore.getState().addToast({
        type: 'success',
        title: 'Test',
        message: 'Test message',
        duration: 0, // Don't auto-remove
      });
    });

    expect(useUiStore.getState().toasts.length).toBe(1);
    expect(useUiStore.getState().toasts[0].title).toBe('Test');

    act(() => {
      useUiStore.getState().removeToast(toastId);
    });

    expect(useUiStore.getState().toasts.length).toBe(0);
  });

  it('should add and manage notifications', () => {
    act(() => {
      useUiStore.getState().addNotification({
        type: 'warning',
        title: 'Alert',
        message: 'Test notification',
      });
      useUiStore.getState().addNotification({
        type: 'info',
        title: 'Info',
        message: 'Another notification',
      });
    });

    expect(useUiStore.getState().notifications.length).toBe(2);
    expect(selectUnreadNotificationCount(useUiStore.getState())).toBe(2);

    const firstId = useUiStore.getState().notifications[0].id;
    act(() => {
      useUiStore.getState().markNotificationRead(firstId);
    });

    expect(selectUnreadNotificationCount(useUiStore.getState())).toBe(1);

    act(() => {
      useUiStore.getState().markAllNotificationsRead();
    });

    expect(selectUnreadNotificationCount(useUiStore.getState())).toBe(0);
  });

  it('should manage modals', () => {
    act(() => {
      useUiStore.getState().openModal('settings', { tab: 'general' });
    });

    expect(useUiStore.getState().activeModal).toBe('settings');
    expect(useUiStore.getState().modalData).toEqual({ tab: 'general' });

    act(() => {
      useUiStore.getState().closeModal();
    });

    expect(useUiStore.getState().activeModal).toBeNull();
    expect(useUiStore.getState().modalData).toBeNull();
  });

  it('should manage loading states', () => {
    act(() => {
      useUiStore.getState().setLoading('fetchData', true);
    });

    expect(useUiStore.getState().loadingStates['fetchData']).toBe(true);

    act(() => {
      useUiStore.getState().setLoading('fetchData', false);
    });

    expect(useUiStore.getState().loadingStates['fetchData']).toBe(false);
  });
});

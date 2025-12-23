/**
 * Robot Store - Robot State Management
 *
 * Manages robot-specific state including:
 * - Joint positions and velocities
 * - End-effector pose
 * - Gripper state
 * - Control mode (teleop, autonomous, idle)
 * - Connected robots
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

const createDefaultRobotState = () => ({
  jointPositions: new Array(7).fill(0),
  jointVelocities: new Array(7).fill(0),
  jointTorques: new Array(7).fill(0),
  eePose: {
    position: [0, 0, 0],
    orientation: [0, 0, 0, 1], // quaternion
  },
  gripperPosition: 0,
  gripperForce: 0,
  controlMode: 'idle', // idle, teleop, autonomous, recording
  isMoving: false,
  timestamp: 0,
});

const initialState = {
  // Connected robots registry
  robots: {},
  activeRobotId: null,

  // Teleoperation state
  teleopEnabled: false,
  teleopSource: null, // 'glove', 'keyboard', 'spacemouse'

  // Recording state
  isRecording: false,
  recordingId: null,
  recordedFrames: 0,

  // Trajectory preview
  trajectoryPreview: null,

  // Safety limits
  jointLimits: {
    min: [-2.9, -1.8, -2.9, -3.1, -2.9, -0.02, -2.9],
    max: [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9],
  },

  // History for visualization
  poseHistory: [],
  maxHistoryLength: 100,
};

export const useRobotStore = create(
  devtools(
    (set, get) => ({
      ...initialState,

      // Robot registration
      registerRobot: (robotId, config = {}) =>
        set((state) => ({
          robots: {
            ...state.robots,
            [robotId]: {
              id: robotId,
              name: config.name || robotId,
              type: config.type || 'generic_7dof',
              connected: true,
              state: createDefaultRobotState(),
              ...config,
            },
          },
          activeRobotId: state.activeRobotId || robotId,
        })),

      unregisterRobot: (robotId) =>
        set((state) => {
          const { [robotId]: removed, ...remaining } = state.robots;
          return {
            robots: remaining,
            activeRobotId:
              state.activeRobotId === robotId
                ? Object.keys(remaining)[0] || null
                : state.activeRobotId,
          };
        }),

      setActiveRobot: (robotId) => set({ activeRobotId: robotId }),

      // State updates
      updateRobotState: (robotId, newState) =>
        set((state) => {
          const robot = state.robots[robotId];
          if (!robot) return state;

          const updatedState = {
            ...robot.state,
            ...newState,
            timestamp: Date.now(),
          };

          // Update pose history
          let poseHistory = state.poseHistory;
          if (newState.eePose) {
            poseHistory = [
              ...state.poseHistory.slice(-(state.maxHistoryLength - 1)),
              { ...newState.eePose, timestamp: Date.now() },
            ];
          }

          return {
            robots: {
              ...state.robots,
              [robotId]: { ...robot, state: updatedState },
            },
            poseHistory,
          };
        }),

      // Batch update from WebSocket
      batchUpdateState: (data) => {
        const robotId = data.robot_id || get().activeRobotId;
        if (!robotId) return;

        get().updateRobotState(robotId, {
          jointPositions: data.joint_positions || data.q,
          jointVelocities: data.joint_velocities || data.dq,
          jointTorques: data.joint_torques || data.tau,
          eePose: data.ee_pose,
          gripperPosition: data.gripper_position ?? data.gripper,
          isMoving: data.is_moving,
        });
      },

      // Control mode
      setControlMode: (robotId, mode) =>
        set((state) => {
          const robot = state.robots[robotId];
          if (!robot) return state;

          return {
            robots: {
              ...state.robots,
              [robotId]: {
                ...robot,
                state: { ...robot.state, controlMode: mode },
              },
            },
          };
        }),

      // Teleoperation
      enableTeleop: (source = 'glove') =>
        set({ teleopEnabled: true, teleopSource: source }),

      disableTeleop: () =>
        set({ teleopEnabled: false, teleopSource: null }),

      // Recording
      startRecording: (recordingId) =>
        set({
          isRecording: true,
          recordingId,
          recordedFrames: 0,
        }),

      stopRecording: () =>
        set({
          isRecording: false,
          recordingId: null,
        }),

      incrementRecordedFrames: () =>
        set((state) => ({ recordedFrames: state.recordedFrames + 1 })),

      // Trajectory preview
      setTrajectoryPreview: (trajectory) =>
        set({ trajectoryPreview: trajectory }),

      clearTrajectoryPreview: () =>
        set({ trajectoryPreview: null }),

      // Clear history
      clearPoseHistory: () => set({ poseHistory: [] }),

      // Reset
      reset: () => set(initialState),
    }),
    { name: 'RobotStore' }
  )
);

// Selectors
export const selectActiveRobot = (state) =>
  state.activeRobotId ? state.robots[state.activeRobotId] : null;

export const selectActiveRobotState = (state) => {
  const robot = selectActiveRobot(state);
  return robot?.state || createDefaultRobotState();
};

export const selectRobotList = (state) => Object.values(state.robots);

export const selectIsAnyRobotMoving = (state) =>
  Object.values(state.robots).some((r) => r.state.isMoving);

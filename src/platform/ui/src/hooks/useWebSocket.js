/**
 * Enhanced WebSocket Hook for Real-Time Streaming
 *
 * Provides robust WebSocket connectivity with:
 * - Automatic reconnection with exponential backoff
 * - Connection quality monitoring
 * - Integration with Zustand stores
 * - Heartbeat/ping-pong mechanism
 * - Channel-based subscription
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useSystemStore, useRobotStore, useUiStore } from '../stores';

// Configuration
const CONFIG = {
  reconnectBaseDelay: 1000,
  reconnectMaxDelay: 30000,
  reconnectMaxAttempts: 10,
  heartbeatInterval: 5000,
  heartbeatTimeout: 10000,
  messageQueueSize: 100,
};

// Connection states
export const ConnectionState = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',
  DISCONNECTED: 'disconnected',
  FAILED: 'failed',
};

/**
 * Custom hook for WebSocket connection with auto-reconnect and store integration.
 *
 * @param {string} url - WebSocket URL
 * @param {Object} options - Configuration options
 * @returns {Object} WebSocket state and controls
 */
export function useWebSocket(url, options = {}) {
  const {
    channels = ['all'],
    autoConnect = true,
    enableHeartbeat = true,
  } = options;

  // State
  const [connectionState, setConnectionState] = useState(ConnectionState.DISCONNECTED);
  const [lastMessage, setLastMessage] = useState(null);
  const [latency, setLatency] = useState(0);
  const [messageCount, setMessageCount] = useState(0);
  const [error, setError] = useState(null);

  // Refs
  const wsRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef(null);
  const heartbeatInterval = useRef(null);
  const heartbeatTimeout = useRef(null);
  const lastPingTime = useRef(0);
  const messageQueue = useRef([]);

  // Store actions
  const setSystemConnected = useSystemStore((state) => state.setConnected);
  const updateSystemMetrics = useSystemStore((state) => state.updateMetrics);
  const updateMetaAiModels = useSystemStore((state) => state.updateMetaAiModels);
  const setSafetyStatus = useSystemStore((state) => state.setSafetyStatus);
  const addSystemError = useSystemStore((state) => state.addError);

  const registerRobot = useRobotStore((state) => state.registerRobot);
  const batchUpdateRobotState = useRobotStore((state) => state.batchUpdateState);

  const addToast = useUiStore((state) => state.addToast);
  const addNotification = useUiStore((state) => state.addNotification);

  // Calculate reconnect delay with exponential backoff
  const getReconnectDelay = useCallback(() => {
    const delay = CONFIG.reconnectBaseDelay * Math.pow(2, reconnectAttempts.current);
    return Math.min(delay, CONFIG.reconnectMaxDelay);
  }, []);

  // Clear all timers
  const clearTimers = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    if (heartbeatInterval.current) {
      clearInterval(heartbeatInterval.current);
      heartbeatInterval.current = null;
    }
    if (heartbeatTimeout.current) {
      clearTimeout(heartbeatTimeout.current);
      heartbeatTimeout.current = null;
    }
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback((event) => {
    try {
      const message = JSON.parse(event.data);
      setLastMessage(message);
      setMessageCount((prev) => prev + 1);

      // Handle pong response
      if (message.type === 'pong') {
        const latencyMs = Date.now() - lastPingTime.current;
        setLatency(latencyMs);
        if (heartbeatTimeout.current) {
          clearTimeout(heartbeatTimeout.current);
          heartbeatTimeout.current = null;
        }
        return;
      }

      // Route message to appropriate store
      const channel = message.channel || message.type;
      const data = message.data || message;

      switch (channel) {
        case 'system_stats':
        case 'metrics':
          updateSystemMetrics(data);
          break;

        case 'meta_ai_models':
          updateMetaAiModels(data);
          break;

        case 'robot_state':
          batchUpdateRobotState(data);
          break;

        case 'robot_connected':
          registerRobot(data.robot_id, data);
          addToast({
            type: 'success',
            title: 'Robot Connected',
            message: `${data.name || data.robot_id} is now online`,
          });
          break;

        case 'safety':
          setSafetyStatus(data.status, data.hazards);
          if (data.status !== 'OK') {
            addNotification({
              type: 'warning',
              title: 'Safety Alert',
              message: data.message || 'Safety condition detected',
              data: data,
            });
          }
          break;

        case 'error':
          addSystemError(data.message || 'Unknown error');
          addNotification({
            type: 'error',
            title: 'System Error',
            message: data.message,
            data: data,
          });
          break;

        case 'connected':
          console.log('[WS] Connection confirmed:', data);
          break;

        case 'task':
        case 'federated_learning':
        case 'camera':
          // These are handled by specific components via lastMessage
          break;

        default:
          // Unknown channel - add to queue for other handlers
          messageQueue.current.push(message);
          if (messageQueue.current.length > CONFIG.messageQueueSize) {
            messageQueue.current.shift();
          }
      }
    } catch (err) {
      console.error('[WS] Failed to parse message:', err);
    }
  }, [
    updateSystemMetrics,
    updateMetaAiModels,
    batchUpdateRobotState,
    registerRobot,
    setSafetyStatus,
    addSystemError,
    addToast,
    addNotification,
  ]);

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (!enableHeartbeat) return;

    heartbeatInterval.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        lastPingTime.current = Date.now();
        wsRef.current.send(JSON.stringify({ type: 'ping' }));

        // Set timeout for pong response
        heartbeatTimeout.current = setTimeout(() => {
          console.warn('[WS] Heartbeat timeout - connection may be stale');
          // Optionally force reconnect
        }, CONFIG.heartbeatTimeout);
      }
    }, CONFIG.heartbeatInterval);
  }, [enableHeartbeat]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    clearTimers();
    setConnectionState(ConnectionState.CONNECTING);
    setError(null);

    const channelParam = channels.join(',');
    const wsUrl = `${url}?channels=${channelParam}`;

    console.log(`[WS] Connecting to ${wsUrl}`);

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('[WS] Connected');
        setConnectionState(ConnectionState.CONNECTED);
        setSystemConnected(true);
        reconnectAttempts.current = 0;
        startHeartbeat();

        // Subscribe to channels
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: channels,
        }));
      };

      ws.onclose = (event) => {
        console.log('[WS] Closed:', event.code, event.reason);
        setSystemConnected(false);
        clearTimers();

        // Attempt reconnection
        if (reconnectAttempts.current < CONFIG.reconnectMaxAttempts) {
          setConnectionState(ConnectionState.RECONNECTING);
          const delay = getReconnectDelay();
          console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1})`);

          reconnectTimeout.current = setTimeout(() => {
            reconnectAttempts.current += 1;
            connect();
          }, delay);
        } else {
          setConnectionState(ConnectionState.FAILED);
          setError('Maximum reconnection attempts exceeded');
          addNotification({
            type: 'error',
            title: 'Connection Failed',
            message: 'Unable to connect to the server. Please check your connection.',
          });
        }
      };

      ws.onerror = (event) => {
        console.error('[WS] Error:', event);
        setError('WebSocket connection error');
      };

      ws.onmessage = handleMessage;

      wsRef.current = ws;
    } catch (err) {
      console.error('[WS] Failed to create WebSocket:', err);
      setError(err.message);
      setConnectionState(ConnectionState.FAILED);
    }
  }, [
    url,
    channels,
    clearTimers,
    getReconnectDelay,
    handleMessage,
    setSystemConnected,
    startHeartbeat,
    addNotification,
  ]);

  // Disconnect
  const disconnect = useCallback(() => {
    console.log('[WS] Disconnecting');
    clearTimers();
    reconnectAttempts.current = CONFIG.reconnectMaxAttempts; // Prevent reconnection

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }

    setConnectionState(ConnectionState.DISCONNECTED);
    setSystemConnected(false);
  }, [clearTimers, setSystemConnected]);

  // Send message
  const sendMessage = useCallback((message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    console.warn('[WS] Cannot send - not connected');
    return false;
  }, []);

  // Send command
  const sendCommand = useCallback((action, data = {}) => {
    return sendMessage({ type: 'command', action, ...data });
  }, [sendMessage]);

  // Send teleop command
  const sendTeleop = useCallback((teleopData) => {
    return sendMessage({ type: 'teleop', data: teleopData });
  }, [sendMessage]);

  // Subscribe to additional channels
  const subscribe = useCallback((newChannels) => {
    return sendMessage({ type: 'subscribe', channels: newChannels });
  }, [sendMessage]);

  // Unsubscribe from channels
  const unsubscribe = useCallback((removeChannels) => {
    return sendMessage({ type: 'unsubscribe', channels: removeChannels });
  }, [sendMessage]);

  // Get queued messages
  const getQueuedMessages = useCallback((filter) => {
    if (!filter) return [...messageQueue.current];
    return messageQueue.current.filter(filter);
  }, []);

  // Clear queued messages
  const clearQueue = useCallback(() => {
    messageQueue.current = [];
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect]); // Only run on mount/unmount

  // Return state and controls
  return {
    // State
    connectionState,
    isConnected: connectionState === ConnectionState.CONNECTED,
    isReconnecting: connectionState === ConnectionState.RECONNECTING,
    lastMessage,
    latency,
    messageCount,
    error,

    // Controls
    connect,
    disconnect,
    sendMessage,
    sendCommand,
    sendTeleop,
    subscribe,
    unsubscribe,

    // Queue access
    getQueuedMessages,
    clearQueue,
  };
}

export default useWebSocket;

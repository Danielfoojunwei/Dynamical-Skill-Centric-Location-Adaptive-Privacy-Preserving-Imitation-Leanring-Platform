/**
 * WebSocket Hook for Real-Time Simulation Streaming
 *
 * Provides a React hook for connecting to the simulation WebSocket
 * and receiving real-time updates.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

const WS_RECONNECT_DELAY = 2000;
const WS_MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Custom hook for WebSocket connection to simulation server.
 *
 * @param {string} url - WebSocket URL (e.g., ws://localhost:8000/ws/simulation)
 * @param {string[]} channels - Channels to subscribe to
 * @returns {Object} WebSocket state and controls
 */
export function useWebSocket(url, channels = ['all']) {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [robotState, setRobotState] = useState(null);
    const [taskState, setTaskState] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [flState, setFlState] = useState(null);
    const [safetyState, setSafetyState] = useState(null);
    const [cameraFrames, setCameraFrames] = useState({});
    const [error, setError] = useState(null);

    const wsRef = useRef(null);
    const reconnectAttempts = useRef(0);
    const reconnectTimeout = useRef(null);

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        const channelParam = channels.join(',');
        const wsUrl = `${url}?channels=${channelParam}`;

        console.log(`Connecting to WebSocket: ${wsUrl}`);
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WebSocket connected');
            setIsConnected(true);
            setError(null);
            reconnectAttempts.current = 0;
        };

        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            setIsConnected(false);
            wsRef.current = null;

            // Auto-reconnect
            if (reconnectAttempts.current < WS_MAX_RECONNECT_ATTEMPTS) {
                reconnectTimeout.current = setTimeout(() => {
                    reconnectAttempts.current += 1;
                    console.log(`Reconnecting... (attempt ${reconnectAttempts.current})`);
                    connect();
                }, WS_RECONNECT_DELAY);
            }
        };

        ws.onerror = (event) => {
            console.error('WebSocket error:', event);
            setError('WebSocket connection error');
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                setLastMessage(message);

                // Route message to appropriate state
                const channel = message.channel || message.type;
                const data = message.data;

                switch (channel) {
                    case 'robot_state':
                        setRobotState(data);
                        break;
                    case 'task':
                        setTaskState(data);
                        break;
                    case 'metrics':
                        setMetrics(data);
                        break;
                    case 'federated_learning':
                        setFlState(data);
                        break;
                    case 'safety':
                        setSafetyState(data);
                        break;
                    case 'camera':
                        setCameraFrames(prev => ({
                            ...prev,
                            [data.camera_id]: data,
                        }));
                        break;
                    case 'connected':
                    case 'started':
                    case 'stopped':
                    case 'reset':
                        // Control messages
                        break;
                    default:
                        // Unknown channel
                        break;
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        wsRef.current = ws;
    }, [url, channels]);

    // Disconnect from WebSocket
    const disconnect = useCallback(() => {
        if (reconnectTimeout.current) {
            clearTimeout(reconnectTimeout.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    // Send message
    const sendMessage = useCallback((message) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
            return true;
        }
        return false;
    }, []);

    // Subscribe to channels
    const subscribe = useCallback((newChannels) => {
        sendMessage({ type: 'subscribe', channels: newChannels });
    }, [sendMessage]);

    // Unsubscribe from channels
    const unsubscribe = useCallback((removeChannels) => {
        sendMessage({ type: 'unsubscribe', channels: removeChannels });
    }, [sendMessage]);

    // Send command
    const sendCommand = useCallback((action, data = {}) => {
        sendMessage({ type: 'command', action, ...data });
    }, [sendMessage]);

    // Send teleop command
    const sendTeleop = useCallback((data) => {
        sendMessage({ type: 'teleop', data });
    }, [sendMessage]);

    // Connect on mount
    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        isConnected,
        lastMessage,
        robotState,
        taskState,
        metrics,
        flState,
        safetyState,
        cameraFrames,
        error,
        connect,
        disconnect,
        sendMessage,
        subscribe,
        unsubscribe,
        sendCommand,
        sendTeleop,
    };
}

export default useWebSocket;

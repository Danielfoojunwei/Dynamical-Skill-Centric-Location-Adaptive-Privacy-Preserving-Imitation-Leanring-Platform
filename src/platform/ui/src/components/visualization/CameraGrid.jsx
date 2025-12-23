/**
 * CameraGrid - Multi-Camera View Component
 *
 * Displays multiple camera feeds in a configurable grid layout.
 * Supports overlays for pose estimation, segmentation, and depth.
 *
 * Features:
 * - Real-time camera frame updates via WebSocket
 * - Overlay toggle (pose skeleton, segmentation masks, depth)
 * - Full-screen individual camera view
 * - Recording indicator
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Camera,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Layers,
  User,
  Box,
  RefreshCw,
} from 'lucide-react';
import { clsx } from 'clsx';
import { useRobotStore } from '../../stores';

// Overlay types
const OVERLAY_TYPES = {
  none: { id: 'none', label: 'None', icon: EyeOff },
  pose: { id: 'pose', label: 'Pose', icon: User },
  segmentation: { id: 'segmentation', label: 'Segmentation', icon: Layers },
  depth: { id: 'depth', label: 'Depth', icon: Box },
};

// Single camera view component
function CameraView({
  cameraId,
  name,
  frame,
  overlay = 'none',
  onOverlayChange,
  onFullscreen,
  isRecording = false,
  isFullscreen = false,
}) {
  const canvasRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Draw frame to canvas
  useEffect(() => {
    if (!frame?.data || !canvasRef.current) {
      setIsLoading(true);
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    try {
      // Handle base64 image data
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Draw overlay if enabled
        if (overlay !== 'none' && frame.overlay) {
          drawOverlay(ctx, overlay, frame.overlay);
        }

        setIsLoading(false);
        setError(null);
      };
      img.onerror = () => {
        setError('Failed to load frame');
        setIsLoading(false);
      };

      // Set image source (base64 or URL)
      if (frame.data.startsWith('data:')) {
        img.src = frame.data;
      } else {
        img.src = `data:image/jpeg;base64,${frame.data}`;
      }
    } catch (err) {
      setError(err.message);
      setIsLoading(false);
    }
  }, [frame, overlay]);

  // Draw overlay based on type
  const drawOverlay = (ctx, overlayType, overlayData) => {
    switch (overlayType) {
      case 'pose':
        drawPoseOverlay(ctx, overlayData);
        break;
      case 'segmentation':
        drawSegmentationOverlay(ctx, overlayData);
        break;
      case 'depth':
        drawDepthOverlay(ctx, overlayData);
        break;
    }
  };

  // Draw pose skeleton
  const drawPoseOverlay = (ctx, poseData) => {
    if (!poseData?.keypoints) return;

    const { keypoints, connections } = poseData;

    // Draw connections
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    connections?.forEach(([i, j]) => {
      const p1 = keypoints[i];
      const p2 = keypoints[j];
      if (p1 && p2 && p1.confidence > 0.3 && p2.confidence > 0.3) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    });

    // Draw keypoints
    keypoints?.forEach((kp) => {
      if (kp.confidence > 0.3) {
        ctx.fillStyle = '#3b82f6';
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  };

  // Draw segmentation mask
  const drawSegmentationOverlay = (ctx, segData) => {
    if (!segData?.mask) return;

    // Create semi-transparent mask overlay
    ctx.globalAlpha = 0.4;
    ctx.fillStyle = '#8b5cf6';
    // In real implementation, draw the actual mask data
    ctx.globalAlpha = 1.0;
  };

  // Draw depth colormap
  const drawDepthOverlay = (ctx, depthData) => {
    if (!depthData?.data) return;
    // In real implementation, apply depth colormap
  };

  return (
    <div
      className={clsx(
        'relative bg-gray-900 rounded-lg overflow-hidden border border-gray-700',
        isFullscreen && 'fixed inset-4 z-50'
      )}
    >
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 py-2 bg-gradient-to-b from-gray-900/90 to-transparent z-10">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-blue-400" />
          <span className="text-sm font-medium text-white">{name || cameraId}</span>
          {isRecording && (
            <span className="flex items-center gap-1 text-xs text-red-400">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              REC
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          {/* Overlay selector */}
          <select
            value={overlay}
            onChange={(e) => onOverlayChange?.(e.target.value)}
            className="text-xs bg-gray-800/80 border border-gray-600 rounded px-2 py-1 text-gray-300"
          >
            {Object.values(OVERLAY_TYPES).map((type) => (
              <option key={type.id} value={type.id}>
                {type.label}
              </option>
            ))}
          </select>

          {/* Fullscreen toggle */}
          <button
            onClick={() => onFullscreen?.(!isFullscreen)}
            className="p-1.5 rounded hover:bg-gray-700/50 text-gray-400 hover:text-white transition"
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full object-contain"
        style={{ minHeight: isFullscreen ? '80vh' : '200px' }}
      />

      {/* Loading state */}
      {isLoading && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="text-center">
            <Camera className="w-8 h-8 text-gray-600 mx-auto mb-2" />
            <p className="text-sm text-gray-400">{error}</p>
          </div>
        </div>
      )}

      {/* No frame placeholder */}
      {!frame && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <Camera className="w-12 h-12 text-gray-700 mx-auto mb-2" />
            <p className="text-sm text-gray-500">No camera feed</p>
          </div>
        </div>
      )}

      {/* Timestamp */}
      {frame?.timestamp && (
        <div className="absolute bottom-2 right-2 text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded">
          {new Date(frame.timestamp).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

// Main grid component
export function CameraGrid({
  cameras = [],
  frames = {},
  layout = 'auto',
  className = '',
}) {
  const [overlays, setOverlays] = useState({});
  const [fullscreenCamera, setFullscreenCamera] = useState(null);
  const isRecording = useRobotStore((state) => state.isRecording);

  // Default cameras if none provided
  const cameraList =
    cameras.length > 0
      ? cameras
      : [
          { id: 'front', name: 'Front Camera' },
          { id: 'wrist', name: 'Wrist Camera' },
          { id: 'side', name: 'Side Camera' },
        ];

  // Determine grid columns based on camera count
  const getGridCols = () => {
    const count = cameraList.length;
    if (layout !== 'auto') return layout;
    if (count <= 1) return 'grid-cols-1';
    if (count <= 2) return 'grid-cols-2';
    if (count <= 4) return 'grid-cols-2';
    return 'grid-cols-3';
  };

  const handleOverlayChange = (cameraId, overlay) => {
    setOverlays((prev) => ({ ...prev, [cameraId]: overlay }));
  };

  // Fullscreen view
  if (fullscreenCamera) {
    const camera = cameraList.find((c) => c.id === fullscreenCamera);
    return (
      <CameraView
        cameraId={fullscreenCamera}
        name={camera?.name}
        frame={frames[fullscreenCamera]}
        overlay={overlays[fullscreenCamera] || 'none'}
        onOverlayChange={(o) => handleOverlayChange(fullscreenCamera, o)}
        onFullscreen={() => setFullscreenCamera(null)}
        isRecording={isRecording}
        isFullscreen
      />
    );
  }

  return (
    <div className={clsx('grid gap-4', getGridCols(), className)}>
      {cameraList.map((camera) => (
        <CameraView
          key={camera.id}
          cameraId={camera.id}
          name={camera.name}
          frame={frames[camera.id]}
          overlay={overlays[camera.id] || 'none'}
          onOverlayChange={(o) => handleOverlayChange(camera.id, o)}
          onFullscreen={() => setFullscreenCamera(camera.id)}
          isRecording={isRecording}
        />
      ))}
    </div>
  );
}

export default CameraGrid;

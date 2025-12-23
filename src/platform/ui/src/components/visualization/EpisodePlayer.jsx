/**
 * EpisodePlayer - Demonstration Replay Component
 *
 * Enables scrubbing through recorded demonstrations with synchronized:
 * - Camera views
 * - Robot joint positions
 * - Actions taken
 * - Skill activations
 *
 * Features:
 * - Play/pause/step controls
 * - Playback speed adjustment
 * - Frame-by-frame navigation
 * - Timeline with markers for skill boundaries
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import * as Slider from '@radix-ui/react-slider';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  Repeat,
  Download,
  Settings,
  Clock,
} from 'lucide-react';
import { clsx } from 'clsx';

// Playback speeds
const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4];

// Format time as MM:SS.mmm
function formatTime(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const millis = Math.floor((ms % 1000) / 10);
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${millis.toString().padStart(2, '0')}`;
}

// Timeline marker component
function TimelineMarker({ position, label, color = '#3b82f6', onClick }) {
  return (
    <div
      className="absolute top-0 bottom-0 cursor-pointer group"
      style={{ left: `${position}%` }}
      onClick={onClick}
    >
      <div
        className="w-0.5 h-full"
        style={{ backgroundColor: color }}
      />
      <div className="absolute -top-6 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
        <div
          className="text-xs px-2 py-1 rounded whitespace-nowrap"
          style={{ backgroundColor: color }}
        >
          {label}
        </div>
      </div>
    </div>
  );
}

// Frame preview thumbnail
function FramePreview({ frame, timestamp }) {
  return (
    <div className="relative">
      <img
        src={frame}
        alt="Frame preview"
        className="w-full h-auto rounded"
      />
      <div className="absolute bottom-1 right-1 text-xs bg-black/70 px-1 rounded">
        {formatTime(timestamp)}
      </div>
    </div>
  );
}

// Main EpisodePlayer component
export function EpisodePlayer({
  episode = null,
  onFrameChange,
  onClose,
  className = '',
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isLooping, setIsLooping] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const animationRef = useRef(null);
  const lastTimeRef = useRef(0);

  // Episode data
  const totalFrames = episode?.frames?.length || 0;
  const totalDuration = episode?.duration_ms || (totalFrames * 50); // Assume 20Hz if not specified
  const currentTime = (currentFrame / totalFrames) * totalDuration;
  const fps = totalFrames > 0 ? (totalFrames / (totalDuration / 1000)) : 0;

  // Current frame data
  const frameData = episode?.frames?.[currentFrame];

  // Skill markers
  const skillMarkers = episode?.skill_boundaries || [];

  // Animation loop
  useEffect(() => {
    if (!isPlaying || totalFrames === 0) return;

    const frameDuration = (1000 / fps) / playbackSpeed;

    const animate = (timestamp) => {
      if (timestamp - lastTimeRef.current >= frameDuration) {
        setCurrentFrame((prev) => {
          const next = prev + 1;
          if (next >= totalFrames) {
            if (isLooping) {
              return 0;
            }
            setIsPlaying(false);
            return prev;
          }
          return next;
        });
        lastTimeRef.current = timestamp;
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, totalFrames, fps, isLooping]);

  // Notify parent of frame changes
  useEffect(() => {
    onFrameChange?.(currentFrame, frameData);
  }, [currentFrame, frameData, onFrameChange]);

  // Controls
  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const stepFrame = useCallback((delta) => {
    setCurrentFrame((prev) => {
      const next = prev + delta;
      return Math.max(0, Math.min(totalFrames - 1, next));
    });
  }, [totalFrames]);

  const seekTo = useCallback((frame) => {
    setCurrentFrame(Math.max(0, Math.min(totalFrames - 1, frame)));
  }, [totalFrames]);

  const seekToPercent = useCallback((percent) => {
    seekTo(Math.floor((percent / 100) * totalFrames));
  }, [totalFrames, seekTo]);

  const cycleSpeed = useCallback(() => {
    setPlaybackSpeed((prev) => {
      const idx = SPEEDS.indexOf(prev);
      return SPEEDS[(idx + 1) % SPEEDS.length];
    });
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT') return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          togglePlay();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          stepFrame(e.shiftKey ? -10 : -1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          stepFrame(e.shiftKey ? 10 : 1);
          break;
        case 'Home':
          e.preventDefault();
          seekTo(0);
          break;
        case 'End':
          e.preventDefault();
          seekTo(totalFrames - 1);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay, stepFrame, seekTo, totalFrames]);

  // No episode placeholder
  if (!episode) {
    return (
      <div className={clsx('bg-gray-900 rounded-lg p-8 text-center', className)}>
        <Play className="w-12 h-12 text-gray-700 mx-auto mb-4" />
        <p className="text-gray-400">No episode loaded</p>
        <p className="text-sm text-gray-500 mt-2">
          Select an episode from the training manager to replay
        </p>
      </div>
    );
  }

  return (
    <div className={clsx('bg-gray-900 rounded-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h3 className="font-medium text-white">{episode.name || 'Episode'}</h3>
          <p className="text-xs text-gray-500">
            {totalFrames} frames • {formatTime(totalDuration)} • {fps.toFixed(1)} FPS
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded hover:bg-gray-800 text-gray-400"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Frame display area */}
      <div className="relative aspect-video bg-black">
        {frameData?.camera_frame && (
          <img
            src={`data:image/jpeg;base64,${frameData.camera_frame}`}
            alt={`Frame ${currentFrame}`}
            className="w-full h-full object-contain"
          />
        )}

        {/* Frame info overlay */}
        <div className="absolute top-2 left-2 bg-black/70 px-2 py-1 rounded text-xs text-white">
          Frame {currentFrame + 1} / {totalFrames}
        </div>

        {/* Action info */}
        {frameData?.action && (
          <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-1 rounded text-xs">
            <span className="text-gray-400">Action: </span>
            <span className="text-green-400">
              [{frameData.action.map((a) => a.toFixed(2)).join(', ')}]
            </span>
          </div>
        )}

        {/* Active skill */}
        {frameData?.skill && (
          <div className="absolute top-2 right-2 bg-blue-600/90 px-2 py-1 rounded text-xs text-white">
            🎯 {frameData.skill}
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="px-4 py-3">
        <div className="relative h-8">
          {/* Skill markers */}
          {skillMarkers.map((marker, idx) => (
            <TimelineMarker
              key={idx}
              position={(marker.frame / totalFrames) * 100}
              label={marker.skill}
              color={marker.color || '#8b5cf6'}
              onClick={() => seekTo(marker.frame)}
            />
          ))}

          {/* Slider */}
          <Slider.Root
            value={[currentFrame]}
            onValueChange={([v]) => seekTo(v)}
            max={totalFrames - 1}
            step={1}
            className="relative flex items-center w-full h-full select-none touch-none"
          >
            <Slider.Track className="relative grow h-1 bg-gray-700 rounded-full">
              <Slider.Range className="absolute h-full bg-blue-500 rounded-full" />
            </Slider.Track>
            <Slider.Thumb className="block w-4 h-4 bg-white rounded-full shadow-lg hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue-500" />
          </Slider.Root>
        </div>

        {/* Time display */}
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(totalDuration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-2 px-4 pb-4">
        {/* Skip to start */}
        <button
          onClick={() => seekTo(0)}
          className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition"
          title="Go to start (Home)"
        >
          <SkipBack className="w-5 h-5" />
        </button>

        {/* Step back */}
        <button
          onClick={() => stepFrame(-1)}
          className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition"
          title="Previous frame (←)"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>

        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="p-3 rounded-full bg-blue-600 hover:bg-blue-700 text-white transition"
          title="Play/Pause (Space)"
        >
          {isPlaying ? (
            <Pause className="w-6 h-6" />
          ) : (
            <Play className="w-6 h-6 ml-0.5" />
          )}
        </button>

        {/* Step forward */}
        <button
          onClick={() => stepFrame(1)}
          className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition"
          title="Next frame (→)"
        >
          <ChevronRight className="w-5 h-5" />
        </button>

        {/* Skip to end */}
        <button
          onClick={() => seekTo(totalFrames - 1)}
          className="p-2 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition"
          title="Go to end (End)"
        >
          <SkipForward className="w-5 h-5" />
        </button>

        {/* Divider */}
        <div className="w-px h-6 bg-gray-700 mx-2" />

        {/* Speed */}
        <button
          onClick={cycleSpeed}
          className="px-3 py-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition text-sm font-mono"
          title="Playback speed"
        >
          {playbackSpeed}x
        </button>

        {/* Loop */}
        <button
          onClick={() => setIsLooping(!isLooping)}
          className={clsx(
            'p-2 rounded transition',
            isLooping
              ? 'bg-blue-600/20 text-blue-400'
              : 'hover:bg-gray-800 text-gray-400 hover:text-white'
          )}
          title="Toggle loop"
        >
          <Repeat className="w-5 h-5" />
        </button>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="px-4 pb-3 text-center">
        <p className="text-xs text-gray-600">
          Space: Play/Pause • ← →: Step • Shift+←/→: Jump 10 frames
        </p>
      </div>
    </div>
  );
}

export default EpisodePlayer;

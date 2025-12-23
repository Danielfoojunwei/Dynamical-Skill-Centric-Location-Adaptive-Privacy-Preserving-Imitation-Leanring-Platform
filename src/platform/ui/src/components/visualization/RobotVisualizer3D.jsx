/**
 * RobotVisualizer3D - Interactive 3D Robot Visualization
 *
 * Renders a 3D robot arm with real-time joint positions.
 * Uses React Three Fiber for WebGL rendering.
 *
 * Features:
 * - Real-time joint position updates
 * - End-effector trajectory trail
 * - Safety zone visualization
 * - Interactive camera controls
 */

import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line, Text, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { useRobotStore, selectActiveRobotState } from '../../stores';

// Robot arm link component
function RobotLink({ length, radius, rotation, position, color }) {
  return (
    <group position={position} rotation={rotation}>
      {/* Joint sphere */}
      <mesh>
        <sphereGeometry args={[radius * 1.5, 16, 16]} />
        <meshStandardMaterial color="#4b5563" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Link cylinder */}
      <mesh position={[0, length / 2, 0]}>
        <cylinderGeometry args={[radius, radius, length, 16]} />
        <meshStandardMaterial color={color} metalness={0.6} roughness={0.3} />
      </mesh>
    </group>
  );
}

// 7-DOF Robot Arm visualization
function RobotArm({ jointPositions = [], color = '#3b82f6' }) {
  const groupRef = useRef();

  // Link lengths (scaled for visualization)
  const linkLengths = [0.3, 0.4, 0.4, 0.3, 0.2, 0.15, 0.1];
  const linkRadius = 0.05;

  // Convert joint angles to link rotations
  const joints = jointPositions.length >= 7
    ? jointPositions
    : [0, 0, 0, 0, 0, 0, 0];

  return (
    <group ref={groupRef}>
      {/* Base */}
      <mesh position={[0, 0.05, 0]}>
        <cylinderGeometry args={[0.15, 0.15, 0.1, 32]} />
        <meshStandardMaterial color="#1f2937" metalness={0.9} roughness={0.1} />
      </mesh>

      {/* Link 1 - Base rotation */}
      <group rotation={[0, joints[0], 0]}>
        <RobotLink
          length={linkLengths[0]}
          radius={linkRadius}
          position={[0, 0.1, 0]}
          rotation={[0, 0, 0]}
          color={color}
        />

        {/* Link 2 - Shoulder */}
        <group position={[0, 0.1 + linkLengths[0], 0]} rotation={[joints[1], 0, 0]}>
          <RobotLink
            length={linkLengths[1]}
            radius={linkRadius}
            position={[0, 0, 0]}
            rotation={[0, 0, 0]}
            color={color}
          />

          {/* Link 3 - Elbow */}
          <group position={[0, linkLengths[1], 0]} rotation={[joints[2], 0, 0]}>
            <RobotLink
              length={linkLengths[2]}
              radius={linkRadius}
              position={[0, 0, 0]}
              rotation={[0, 0, 0]}
              color={color}
            />

            {/* Link 4 - Wrist 1 */}
            <group position={[0, linkLengths[2], 0]} rotation={[0, joints[3], 0]}>
              <RobotLink
                length={linkLengths[3]}
                radius={linkRadius * 0.8}
                position={[0, 0, 0]}
                rotation={[0, 0, 0]}
                color="#6366f1"
              />

              {/* Link 5 - Wrist 2 */}
              <group position={[0, linkLengths[3], 0]} rotation={[joints[4], 0, 0]}>
                <RobotLink
                  length={linkLengths[4]}
                  radius={linkRadius * 0.6}
                  position={[0, 0, 0]}
                  rotation={[0, 0, 0]}
                  color="#6366f1"
                />

                {/* Link 6 - Wrist 3 */}
                <group position={[0, linkLengths[4], 0]} rotation={[0, joints[5], 0]}>
                  <RobotLink
                    length={linkLengths[5]}
                    radius={linkRadius * 0.5}
                    position={[0, 0, 0]}
                    rotation={[0, 0, 0]}
                    color="#8b5cf6"
                  />

                  {/* End Effector */}
                  <group position={[0, linkLengths[5], 0]} rotation={[joints[6], 0, 0]}>
                    <mesh position={[0, linkLengths[6] / 2, 0]}>
                      <boxGeometry args={[0.08, linkLengths[6], 0.08]} />
                      <meshStandardMaterial color="#22c55e" metalness={0.7} roughness={0.2} />
                    </mesh>
                    {/* Gripper indicator */}
                    <mesh position={[0, linkLengths[6], 0]}>
                      <sphereGeometry args={[0.03, 16, 16]} />
                      <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
                    </mesh>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

// Trajectory trail visualization
function TrajectoryTrail({ points, color = '#22c55e' }) {
  if (!points || points.length < 2) return null;

  const linePoints = points.map((p) => new THREE.Vector3(
    p.position?.[0] || 0,
    p.position?.[1] || 0,
    p.position?.[2] || 0
  ));

  return (
    <Line
      points={linePoints}
      color={color}
      lineWidth={2}
      opacity={0.7}
      transparent
    />
  );
}

// Safety zone visualization
function SafetyZone({ radius = 1.5, height = 2 }) {
  return (
    <mesh position={[0, height / 2, 0]}>
      <cylinderGeometry args={[radius, radius, height, 32, 1, true]} />
      <meshBasicMaterial
        color="#ef4444"
        transparent
        opacity={0.1}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// Ground plane
function Ground() {
  return (
    <Grid
      infiniteGrid
      cellSize={0.5}
      cellThickness={0.5}
      sectionSize={1}
      sectionThickness={1}
      sectionColor="#3b82f6"
      cellColor="#1f2937"
      fadeDistance={10}
      fadeStrength={1}
    />
  );
}

// Main 3D scene
function Scene({ showSafetyZone = true, showTrajectory = true }) {
  const robotState = useRobotStore(selectActiveRobotState);
  const poseHistory = useRobotStore((state) => state.poseHistory);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} castShadow />
      <directionalLight position={[-5, 5, -5]} intensity={0.3} />

      {/* Environment */}
      <Environment preset="warehouse" background={false} />

      {/* Robot */}
      <RobotArm jointPositions={robotState.jointPositions} />

      {/* Trajectory */}
      {showTrajectory && poseHistory.length > 1 && (
        <TrajectoryTrail points={poseHistory} />
      )}

      {/* Safety Zone */}
      {showSafetyZone && <SafetyZone />}

      {/* Ground */}
      <Ground />

      {/* Camera Controls */}
      <OrbitControls
        makeDefault
        minPolarAngle={0}
        maxPolarAngle={Math.PI / 2}
        minDistance={1}
        maxDistance={5}
      />
    </>
  );
}

// Main component
export function RobotVisualizer3D({
  className = '',
  showSafetyZone = true,
  showTrajectory = true,
  showControls = true,
}) {
  return (
    <div className={`relative ${className}`} style={{ minHeight: '400px' }}>
      <Canvas
        camera={{ position: [2, 2, 2], fov: 50 }}
        shadows
        gl={{ antialias: true }}
      >
        <Scene
          showSafetyZone={showSafetyZone}
          showTrajectory={showTrajectory}
        />
      </Canvas>

      {/* Overlay controls */}
      {showControls && (
        <div className="absolute top-4 left-4 text-xs text-gray-400 bg-gray-900/80 px-3 py-2 rounded">
          <div>🖱️ Drag to rotate</div>
          <div>⚲ Scroll to zoom</div>
        </div>
      )}
    </div>
  );
}

export default RobotVisualizer3D;

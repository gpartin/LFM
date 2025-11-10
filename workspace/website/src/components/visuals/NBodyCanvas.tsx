/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

'use client';

import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { NBodyOrbitSimulation } from '@/physics/forces/n-body-orbit';

interface NBodyCanvasProps {
  simulation: React.MutableRefObject<NBodyOrbitSimulation | null>;
  isRunning: boolean;
  showParticles: boolean;
  showTrails: boolean;
  showBackground?: boolean;
}

interface TrailData {
  points: THREE.Vector3[];
  line: THREE.Line | null;
  material: THREE.LineBasicMaterial | null;
}

function Particles({ simulation, showParticles, showTrails }: { 
  simulation: React.MutableRefObject<NBodyOrbitSimulation | null>; 
  showParticles: boolean; 
  showTrails: boolean; 
}) {
  const particleRefs = useRef<THREE.Mesh[]>([]);
  const trailsRef = useRef<TrailData[]>([]);
  const trailGroupRef = useRef<THREE.Group>(null);
  const maxPoints = 500;
  const [initialPositions, setInitialPositions] = React.useState<Array<[number, number, number]>>([]);
  
  const colors = ['#00ffff', '#ff00ff', '#ffff00', '#00ff00', '#ff8800']; // Cyan, Magenta, Yellow, Green, Orange

  // Capture initial positions immediately
  useEffect(() => {
    const sim = simulation.current;
    if (sim) {
      const state = sim.getState();
      setInitialPositions(state.particles.map(p => [...p.position] as [number, number, number]));
    }
  }, [simulation]);

  // Initialize trails
  useEffect(() => {
    if (!trailGroupRef.current) return;
    
    const sim = simulation.current;
    if (!sim) return;
    
    const state = sim.getState();
    const numBodies = state.particles.length;
    
    // Create trail for each body
    trailsRef.current = Array.from({ length: numBodies }, (_, i) => {
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.LineBasicMaterial({ 
        color: colors[i % colors.length],
        transparent: true,
        opacity: 0.6,
        linewidth: 2,
      });
      const line = new THREE.Line(geometry, material);
      trailGroupRef.current!.add(line);
      return { points: [], line, material };
    });
    
    return () => {
      trailsRef.current.forEach(trail => {
        if (trail.line) {
          trailGroupRef.current?.remove(trail.line);
          trail.line.geometry.dispose();
          trail.material?.dispose();
        }
      });
    };
  }, [simulation, colors]);

  // Update particle positions and trails
  useFrame(() => {
    const sim = simulation.current;
    if (!sim) return;

    const state = sim.getState();
    
    state.particles.forEach((particle, i) => {
      // Update particle position
      if (particleRefs.current[i]) {
        particleRefs.current[i].position.set(
          particle.position[0],
          particle.position[1],
          particle.position[2]
        );
      }
      
      // Update trail
      if (showTrails && trailsRef.current[i]) {
        const trail = trailsRef.current[i];
        const pos = new THREE.Vector3(
          particle.position[0],
          particle.position[1],
          particle.position[2]
        );
        
        trail.points.push(pos);
        if (trail.points.length > maxPoints) {
          trail.points.shift();
        }
        
        if (trail.line) {
          trail.line.geometry.setFromPoints(trail.points);
          trail.line.geometry.attributes.position.needsUpdate = true;
        }
      }
    });
  });

  const sim = simulation.current;
  
  // Use current state if available, fall back to initial positions
  const positions = sim ? sim.getState().particles.map(p => p.position) : initialPositions;

  return (
    <group>
      <group ref={trailGroupRef} visible={showTrails} />
      {showParticles && positions.map((position, i) => (
        <mesh
          key={i}
          ref={(el) => {
            if (el) particleRefs.current[i] = el;
          }}
          position={[position[0], position[1], position[2]]}
        >
          <sphereGeometry args={[0.15, 32, 32]} />
          <meshStandardMaterial 
            color={colors[i % colors.length]}
            emissive={colors[i % colors.length]}
            emissiveIntensity={0.5}
          />
        </mesh>
      ))}
    </group>
  );
}

function Scene({ simulation, isRunning, showParticles, showTrails, showBackground = true }: NBodyCanvasProps) {
  return (
    <>
      <color attach="background" args={['#0a0a1a']} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      
      {/* Coordinate system helper */}
      <gridHelper args={[20, 20, '#333366', '#1a1a2e']} />
      
      <Particles 
        simulation={simulation} 
        showParticles={showParticles} 
        showTrails={showTrails} 
      />
      
      {showBackground && <Stars radius={100} depth={50} count={5000} factor={4} fade speed={1} />}
      
      <OrbitControls 
        enableDamping
        dampingFactor={0.05}
        minDistance={2}
        maxDistance={50}
      />
    </>
  );
}

export default function NBodyCanvas({ simulation, isRunning, showParticles, showTrails, showBackground = true }: NBodyCanvasProps) {
  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [0, 8, 12], fov: 60 }}>
        <Scene 
          simulation={simulation} 
          isRunning={isRunning}
          showParticles={showParticles}
          showTrails={showTrails}
          showBackground={showBackground}
        />
      </Canvas>
    </div>
  );
}

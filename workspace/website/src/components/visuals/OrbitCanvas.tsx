/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*-
 * OrbitCanvas - 3D visualization for Binary Orbit Experiment
 *
 * Uses react-three-fiber + drei to render two particles orbiting with trails.
 * This is an AUTHENTIC LFM visualization: positions come from BinaryOrbitSimulation
 * which evolves via the Klein-Gordon lattice on WebGPU.
 */

'use client';

import React, { useRef, useEffect, useCallback, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stats, Effects, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { BinaryOrbitSimulation } from '@/physics/forces/binary-orbit';
import { EffectComposer, Bloom } from '@react-three/postprocessing';

export interface OrbitCanvasProps {
  simulation: React.MutableRefObject<BinaryOrbitSimulation | null>;
  isRunning: boolean;
  showParticles: boolean;
  showTrails: boolean;
  showChi?: boolean;
  showLattice?: boolean;
  showVectors?: boolean;
  showWell?: boolean;
  showDomes?: boolean;
  showIsoShells?: boolean;
  /** Decorative background (starfield + sun). Defaults off for perf. */
  showBackground?: boolean;
  chiStrength: number;
  /** Show black-hole analogue reference rings (RS, 1.5RS, 3RS) */
  showBHRings?: boolean;
  /** Sigma parameter for chi field concentration (needed for BH horizon RS calculation) */
  sigma?: number;
  /** Enable tidal stretch visualization */
  tidalStretch?: boolean;
  /** Cohesion/self-gravity factor for tidal threshold */
  selfGravityFactor?: number;
  /** Primary particle color (for stellar evolution visuals) */
  primaryColor?: string;
  /** Primary particle scale multiplier (for stellar evolution size changes) */
  primaryScale?: number;
  /** Enable light-ray visualization (for gravitational lensing) */
  showLightRays?: boolean;
  /** Light ray config: count, color, speed, spread */
  rayConfig?: { 
    count?: number; color?: string; speed?: number; spread?: number;
    rows?: number; cols?: number; emitterOffset?: number; emitterWidth?: number; headSize?: number; pure?: boolean; debug?: boolean
  };
  /** Lensing-only: show background source-plane sampler */
  showLensingBackground?: boolean;
  /** Hide the secondary particle mesh (useful for lensing) */
  hideSecondaryParticle?: boolean;
}

interface TrailData {
  points: THREE.Vector3[];
  line: THREE.Line | null;
  material: THREE.LineBasicMaterial | null;
  /** Reusable buffer for positions (eliminates per-frame allocations) */
  positionBuffer: Float32Array | null;
  /** Dirty flag to avoid unnecessary geometry updates */
  isDirty: boolean;
}

function Particles({ simulation, showParticles, showTrails, horizonRS, separation, bhMode = false, tidalEnabled = false, tidalAxis, tidalFactor = 0, primaryColor = '#4A90E2', primaryScale = 1.0, hideSecondaryParticle = false }: { 
  simulation: React.MutableRefObject<BinaryOrbitSimulation | null>; 
  showParticles: boolean; 
  showTrails: boolean;
  horizonRS?: number;
  separation?: number;
  bhMode?: boolean;
  tidalEnabled?: boolean;
  tidalAxis?: [number, number, number];
  tidalFactor?: number;
  primaryColor?: string;
  primaryScale?: number;
  hideSecondaryParticle?: boolean;
}) {
  const p1Ref = useRef<THREE.Mesh>(null);
  const p2Ref = useRef<THREE.Mesh>(null);
  const p2GroupRef = useRef<THREE.Group>(null);

  const maxPoints = 800;
  const trail1Ref = useRef<TrailData>({ points: [], line: null, material: null, positionBuffer: null, isDirty: false });
  const trail2Ref = useRef<TrailData>({ points: [], line: null, material: null, positionBuffer: null, isDirty: false });
  const trailGroupRef = useRef<THREE.Group>(null);

  // Initialize trail lines with reusable buffers
  useEffect(() => {
    if (!trailGroupRef.current) return;
      const createTrail = (color: string): TrailData => {
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.LineBasicMaterial({ 
        color, 
        transparent: true, 
        opacity: 0.8,  // Increased from 0.5 for better visibility
        linewidth: 2,
        depthTest: true,
        depthWrite: false  // Prevent z-fighting
      });
      const line = new THREE.Line(geometry, material);
      trailGroupRef.current!.add(line);
      // Preallocate buffer for max trail length (eliminates per-frame allocations)
      const positionBuffer = new Float32Array(maxPoints * 3);
      return { points: [], line, material, positionBuffer, isDirty: false };
    };
    trail1Ref.current = createTrail('#4A90E2'); // Earth blue
    trail2Ref.current = createTrail('#9CA3AF'); // Moon gray
  }, []);

  useFrame((_state, _delta) => {
    const sim = simulation.current;
    if (!sim) return;
    const s = sim.getState();

  // Calculate fade factor for horizon crossing effect
  const fade = (horizonRS !== undefined && separation !== undefined && separation <= horizonRS) ? 0.1 : 1.0;

    // Determine heavier for BH mode decisions
    const p1Heavy = s.particle1.mass >= s.particle2.mass;
    const pHeavy = p1Heavy ? s.particle1 : s.particle2;
    const pLight = p1Heavy ? s.particle2 : s.particle1;

    // Update particle positions and fade; in bhMode, hide the heavier body's mesh (visualized by BH sphere)
    if (p1Ref.current) {
      p1Ref.current.visible = !(bhMode && p1Heavy);
      p1Ref.current.position.set(s.particle1.position[0], s.particle1.position[1], s.particle1.position[2]);
      const mat = p1Ref.current.material as THREE.MeshStandardMaterial;
      if (mat && (mat as any).emissiveIntensity !== undefined) {
        mat.emissiveIntensity = 1.2 * fade;
      }
    }
    if (p2Ref.current) {
      const p2Visible = hideSecondaryParticle ? false : !(bhMode && !p1Heavy);
      p2Ref.current.visible = p2Visible;
      // Keep moon mesh at local origin; group carries world transform
      p2Ref.current.position.set(0, 0, 0);
      const mat = p2Ref.current.material as THREE.MeshStandardMaterial;
      if (mat && (mat as any).emissiveIntensity !== undefined) {
        mat.emissiveIntensity = 0.8 * fade;
      }
      // Position and orient the wrapper group at the physics position
      if (p2GroupRef.current) {
        p2GroupRef.current.position.set(
          s.particle2.position[0],
          s.particle2.position[1],
          s.particle2.position[2]
        );
        p2GroupRef.current.visible = p2Visible;
        if (tidalEnabled) {
          const alpha = tidalFactor;
          const sLong = 1 + 2 * alpha;
          const sTan = Math.max(0.4, 1 - alpha);
          p2GroupRef.current.scale.set(sTan, sTan, sLong);
          if (tidalAxis) {
            const dir = new THREE.Vector3(tidalAxis[0], tidalAxis[1], tidalAxis[2]).normalize();
            p2GroupRef.current.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
          }
        } else {
          p2GroupRef.current.scale.set(1, 1, 1);
          p2GroupRef.current.quaternion.identity();
        }
      }
    }

    // Trails (optimized with ring buffer - no per-frame allocations)
    if (showTrails) {
      // Apply fade to trail opacity
      if (trail1Ref.current.material) {
        trail1Ref.current.material.opacity = 0.8 * fade;
      }
      if (trail2Ref.current.material) {
        trail2Ref.current.material.opacity = 0.8 * fade;
      }

      const addPoint = (trail: TrailData, pos: THREE.Vector3) => {
        if (!trail.line) return; // Guard against uninitialized trail
        
        trail.points.push(pos.clone());
        if (trail.points.length > maxPoints) trail.points.shift();
        
        // Only proceed if we have points to draw
        if (trail.points.length === 0) return;
        
        // Create a properly sized buffer for current points only
        const pointsBuffer = new Float32Array(trail.points.length * 3);
        trail.points.forEach((p, i) => {
          pointsBuffer[i * 3] = p.x;
          pointsBuffer[i * 3 + 1] = p.y;
          pointsBuffer[i * 3 + 2] = p.z;
        });
        
        // Update geometry attribute
        const geometry = trail.line.geometry;
        geometry.setAttribute('position', new THREE.BufferAttribute(pointsBuffer, 3));
        geometry.attributes.position.needsUpdate = true;
        geometry.setDrawRange(0, trail.points.length);
        geometry.computeBoundingSphere();
      };
  // Use physics positions to avoid discrepancies when the moon is wrapped in a transform group
  const pos1 = new THREE.Vector3(s.particle1.position[0], s.particle1.position[1], s.particle1.position[2]);
  const pos2 = new THREE.Vector3(s.particle2.position[0], s.particle2.position[1], s.particle2.position[2]);
  if (p1Ref.current && p1Ref.current.visible) addPoint(trail1Ref.current, pos1);
  if (p2Ref.current && p2Ref.current.visible && !hideSecondaryParticle) addPoint(trail2Ref.current, pos2);
    } else {
      // Clear trails if toggled off
      trail1Ref.current.points = [];
      trail2Ref.current.points = [];
      trail1Ref.current.isDirty = false;
      trail2Ref.current.isDirty = false;
      if (trail1Ref.current.line) trail1Ref.current.line.geometry.setDrawRange(0, 0);
      if (trail2Ref.current.line) trail2Ref.current.line.geometry.setDrawRange(0, 0);
    }
  });

  return (
    <group>
      <group ref={trailGroupRef} />
      {showParticles && (
        <>
          {/* Primary particle (e.g., Earth or Star): configurable color and size */}
          <mesh ref={p1Ref}>
            <sphereGeometry args={[0.28 * primaryScale, 32, 32]} />
            <meshStandardMaterial emissive={primaryColor} color={primaryColor} emissiveIntensity={1.2} />
          </mesh>
          {/* Secondary particle (e.g., Moon or Test Particle): smaller, gray (wrapped in group for tidal stretch) */}
          <group ref={p2GroupRef}>
            <mesh ref={p2Ref}>
              <sphereGeometry args={[0.15, 32, 32]} />
              <meshStandardMaterial emissive={'#9CA3AF'} color={'#9CA3AF'} emissiveIntensity={0.8} />
            </mesh>
          </group>
        </>
      )}
    </group>
  );
}

function Scene({ simulation, showParticles, showTrails, showChi = false, showLattice = false, showVectors = true, showWell = true, showDomes = false, showIsoShells = false, showBackground = false, isRunning, chiStrength, showBHRings = false, sigma = 1.0, tidalStretch = false, selfGravityFactor = 1.0, primaryColor = '#4A90E2', primaryScale = 1.0, showLightRays = false, rayConfig, showLensingBackground = false, hideSecondaryParticle = false }: OrbitCanvasProps) {
  // Calculate horizon RS and separation for fade effect
  const [horizonRS, setHorizonRS] = useState<number | undefined>(undefined);
  const [separation, setSeparation] = useState<number | undefined>(undefined);
  const [tidalAxis, setTidalAxis] = useState<[number, number, number] | undefined>(undefined);
  const [tidalFactor, setTidalFactor] = useState<number>(0);

  useFrame(() => {
    const sim = simulation.current;
    if (!sim) return;
    
    const s = sim.getState();
    const m1 = s.particle1.mass;
    const m2 = s.particle2.mass;
    const heavierMass = Math.max(m1, m2);
    
    // RS = (sigma × sqrt(mass)) / sqrt(2)
    const rs = (sigma * Math.sqrt(heavierMass)) / Math.sqrt(2);
    setHorizonRS(rs);
    
    // Calculate separation
    const dx = s.particle2.position[0] - s.particle1.position[0];
    const dy = s.particle2.position[1] - s.particle1.position[1];
    const dz = s.particle2.position[2] - s.particle1.position[2];
    const sep = Math.sqrt(dx * dx + dy * dy + dz * dz);
  setSeparation(sep);
    // Optional: compute tidal stretch factor when enabled
    if (tidalStretch) {
      try {
        const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
        const secondary = s.particle1.mass >= s.particle2.mass ? s.particle2 : s.particle1;
        const uVec = new THREE.Vector3(
          secondary.position[0] - primary.position[0],
          secondary.position[1] - primary.position[1],
          secondary.position[2] - primary.position[2]
        );
        const sepLen = uVec.length();
        if (sepLen > 1e-6) {
          uVec.normalize();
          const h = Math.min(0.1, Math.max(0.02, sepLen * 0.02));
          const pPlus: [number, number, number] = [
            secondary.position[0] + uVec.x * h,
            secondary.position[1] + uVec.y * h,
            secondary.position[2] + uVec.z * h,
          ];
          const pMinus: [number, number, number] = [
            secondary.position[0] - uVec.x * h,
            secondary.position[1] - uVec.y * h,
            secondary.position[2] - uVec.z * h,
          ];
          const gPlus = sim.analyticChiGradientAt(pPlus);
          const gMinus = sim.analyticChiGradientAt(pMinus);
          const gPlusDot = gPlus[0]*uVec.x + gPlus[1]*uVec.y + gPlus[2]*uVec.z;
          const gMinusDot = gMinus[0]*uVec.x + gMinus[1]*uVec.y + gMinus[2]*uVec.z;
          const dgrdr = (gPlusDot - gMinusDot) / (2*h);
          const moonRadius = 0.15; // matches mesh
          const aTide = Math.abs(dgrdr) * moonRadius;
          const gSelf = Math.max(1e-6, selfGravityFactor * secondary.mass / (moonRadius * moonRadius));
          const stress = aTide / gSelf;
          // Boost tidal visibly near the horizon analogue
          const rsBoost = (sigma && sepLen > 0)
            ? ( (sigma * Math.sqrt(Math.max(m1, m2))) / Math.SQRT2 ) / sepLen
            : 0;
          const boost = 1 + 1.75 * Math.max(0, rsBoost) * Math.max(0, rsBoost); // ~ (rs/r)^2 scaling
          const stressBoosted = stress * boost;
          // Map to visual alpha with aggressive threshold so stretch shows dramatically
          const alpha = Math.max(0, Math.min(4.5, (stressBoosted - 0.3) * 1.5));
          setTidalFactor(alpha);
          setTidalAxis([uVec.x, uVec.y, uVec.z]);
        } else {
          setTidalFactor(0);
          setTidalAxis(undefined);
        }
      } catch {
        // ignore sampling errors
      }
    } else {
      if (tidalFactor !== 0) setTidalFactor(0);
      if (tidalAxis) setTidalAxis(undefined);
    }
  });

  // Soft ambient
  return (
    <>
      {/* Always apply a dark background color; optional decorative elements below */}
      <color attach="background" args={['#0a0a1a']} />
  {showBackground && <Stars radius={100} depth={50} count={5000} factor={4} fade speed={1} />}
  {/* Experiment-only background lensing sampler (source-plane distortion) */}
  {showLensingBackground && <BackgroundSourcePlane simulation={simulation} isRunning={isRunning} />}
      <ambientLight intensity={0.3} />
      <pointLight position={[8, 8, 8]} intensity={1.2} color={'#ffffff'} />
      <pointLight position={[-8, -6, -5]} intensity={0.8} color={'#4da4ff'} />
      {/* Lattice wireframe cube */}
      {showLattice && (
        <group>
          {/* Dynamically scale lattice box to simulation domain */}
          {(() => {
            const sim = simulation.current;
            if (!sim) return null;
            const { width } = sim.latticeWorldExtent();
            return (
              <mesh>
                <boxGeometry args={[width, width, width]} />
                <meshBasicMaterial color={'#2a355e'} wireframe transparent opacity={0.35} />
              </mesh>
            );
          })()}
        </group>
      )}

  {/* Chi field point cloud (sparse) */}
  {showChi && <ChiPointCloud simulation={simulation} />}

  {/* Force vectors (gradient arrows) are disabled on lensing pages when light rays are shown */}
  {showVectors && !showLightRays && <ForceVectors simulation={simulation} />}

  {/* Gravity well heightfield (chi-based) */}
  {showWell && <GravityWell simulation={simulation} />}

  {/* Light rays (gravitational lensing visualization) */}
  {showLightRays && <LightRays simulation={simulation} isRunning={isRunning} chiStrength={chiStrength} sigma={sigma} rayConfig={rayConfig} />}

  {/* Gaussian energy domes */}
  {showDomes && <FieldDomes simulation={simulation} />}

  {/* Iso-shells (point-based thresholds)
      On black-hole pages, we use showBHRings to render the analogue horizon rings.
      To avoid confusing double-visuals (shells around the planet), suppress IsoShells
      when showBHRings is active. Other pages (showBHRings=false) still get IsoShells. */}
  {showIsoShells && !showBHRings && <IsoShells simulation={simulation} />}

  {/* Black-hole analogue visuals */}
  {showBHRings && (
    <>
      <BlackHoleSphere simulation={simulation} />
      <HorizonRings simulation={simulation} sigma={sigma} />
    </>
  )}

  <Particles 
    simulation={simulation} 
    showParticles={showParticles} 
    showTrails={showTrails}
    horizonRS={horizonRS}
    separation={separation}
    bhMode={showBHRings}
    tidalEnabled={tidalStretch}
    tidalAxis={tidalAxis}
    tidalFactor={tidalFactor}
    primaryColor={primaryColor}
    primaryScale={primaryScale}
    hideSecondaryParticle={hideSecondaryParticle}
  />
     <OrbitControls enablePan={false} enableDamping dampingFactor={0.08} maxDistance={60} minDistance={2} />
      <Effects disableGamma>
        <EffectComposer>
          <Bloom intensity={1.2} luminanceThreshold={0.25} luminanceSmoothing={0.9} mipmapBlur />
        </EffectComposer>
      </Effects>
      <Stats />
    </>
  );
}

function ChiPointCloud({ simulation }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null> }) {
  const pointsRef = React.useRef<THREE.Points>(null);
  const skipRef = React.useRef<number>(4); // downsample factor
  const frameCounterRef = React.useRef<number>(0);
  const lastUpdateRef = React.useRef<number>(0);
  const targetIntervalMs = 500; // update ~2 Hz

  const resample = useCallback(async () => {
    const sim = simulation.current;
    if (!sim || !pointsRef.current) return;
    const { samples } = await sim.sampleChiField(skipRef.current);
    const positions = new Float32Array(samples.length * 3);
    const colors = new Float32Array(samples.length * 3);
    let minChi = Infinity, maxChi = -Infinity;
    for (const s of samples) { minChi = Math.min(minChi, s.chi); maxChi = Math.max(maxChi, s.chi); }
    const range = Math.max(1e-6, maxChi - minChi);
    samples.forEach((s, i) => {
      positions[i * 3] = s.x;
      positions[i * 3 + 1] = s.y;
      positions[i * 3 + 2] = s.z;
      const t = (s.chi - minChi) / range;
      // Gradient: deep blue → cyan
      const r = 0.0;
      const g = 0.55 + 0.45 * t;
      const b = 0.75 + 0.25 * t;
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    });
    const geo = pointsRef.current.geometry as THREE.BufferGeometry;
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeBoundingSphere();
  }, [simulation]);

  // Initial sample
  useEffect(() => { resample(); }, [resample]);

  // Periodic resampling tied to animation frames
  useFrame((_state, _delta) => {
    frameCounterRef.current++;
    const now = performance.now();
    if (now - lastUpdateRef.current >= targetIntervalMs) {
      lastUpdateRef.current = now;
      // Fire and forget; results apply next promise resolution
      resample();
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry />
      <pointsMaterial size={0.06} sizeAttenuation vertexColors transparent opacity={0.8} />
    </points>
  );
}

function ForceVectors({ simulation }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null> }) {
  const linesRef = React.useRef<THREE.LineSegments>(null);
  const conesGroupRef = React.useRef<THREE.Group>(null);
  const maxArrows = 72; // 3 rings × 12 × 2
  const conePoolRef = React.useRef<THREE.Mesh[]>([]);

  // Preallocate cone pool once for performance (avoid per-frame allocations & GC churn)
  useEffect(() => {
    if (!conesGroupRef.current || conePoolRef.current.length > 0) return;
    const pool: THREE.Mesh[] = [];
    for (let i = 0; i < maxArrows; i++) {
      const geom = new THREE.ConeGeometry(0.04, 0.12, 6);
      const mat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9 });
      const cone = new THREE.Mesh(geom, mat);
      cone.visible = false;
      conesGroupRef.current.add(cone);
      pool.push(cone);
    }
    conePoolRef.current = pool;
  }, []);

  useFrame(() => {
    const sim = simulation.current;
    if (!sim || !linesRef.current || !conesGroupRef.current) return;
    const s = sim.getState();
    const positions = new Float32Array(maxArrows * 2 * 3);
    const colors = new Float32Array(maxArrows * 2 * 3);
    let cursor = 0;

    // Hide all cones initially
    for (const cone of conePoolRef.current) cone.visible = false;

    const earthBlue = new THREE.Color('#4A90E2');
    const moonGray = new THREE.Color('#9CA3AF');
    const dimBlue = new THREE.Color('#1a3a5a');
    const dimGray = new THREE.Color('#4a4a4a');

    const addArrowsAround = (center: [number, number, number], radii: number[], color: THREE.Color, baseColor: THREE.Color) => {
      for (const radius of radii) {
        const count = 12;
        for (let i = 0; i < count; i++) {
          if (cursor >= maxArrows) return;
          const theta = (i / count) * Math.PI * 2;
          const phi = Math.PI / 12;
          const pos: [number, number, number] = [
            center[0] + radius * Math.cos(theta) * Math.cos(phi),
            center[1] + radius * Math.sin(theta) * Math.cos(phi),
            center[2] + radius * Math.sin(phi),
          ];
          const g = sim.analyticChiGradientAt(pos);
          const gradMag = Math.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
          const minLen = 0.15;
          const maxLen = 0.7;
          const len = Math.min(maxLen, Math.max(minLen, gradMag * 2.0));
          const dir = new THREE.Vector3(g[0], g[1], g[2]).normalize().multiplyScalar(len);
          const from = new THREE.Vector3(pos[0], pos[1], pos[2]);
          const to = from.clone().add(dir);
          const intensity = Math.min(1.0, gradMag * 4.0);
          const finalColor = baseColor.clone().lerp(color, intensity);
          positions[cursor * 6 + 0] = from.x;
          positions[cursor * 6 + 1] = from.y;
          positions[cursor * 6 + 2] = from.z;
          positions[cursor * 6 + 3] = to.x;
          positions[cursor * 6 + 4] = to.y;
          positions[cursor * 6 + 5] = to.z;
          for (let k = 0; k < 2; k++) {
            colors[cursor * 6 + k * 3 + 0] = finalColor.r;
            colors[cursor * 6 + k * 3 + 1] = finalColor.g;
            colors[cursor * 6 + k * 3 + 2] = finalColor.b;
          }
          // Reuse cone from pool
          const cone = conePoolRef.current[cursor];
          (cone.material as THREE.MeshBasicMaterial).color.copy(finalColor);
          cone.position.copy(to);
          cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
          cone.visible = true;
          cursor++;
        }
      }
    };

    addArrowsAround(s.particle1.position, [0.5, 0.9, 1.4], earthBlue, dimBlue);
    addArrowsAround(s.particle2.position, [0.5, 0.9, 1.4], moonGray, dimGray);

    const geo = linesRef.current.geometry as THREE.BufferGeometry;
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setDrawRange(0, cursor * 2);
    geo.computeBoundingSphere();
  });

  return (
    <>
      <lineSegments ref={linesRef}>
        <bufferGeometry />
        <lineBasicMaterial vertexColors transparent opacity={0.85} linewidth={1.5} />
      </lineSegments>
      <group ref={conesGroupRef} />
    </>
  );
}

export const OrbitCanvas: React.FC<OrbitCanvasProps> = ({ simulation, isRunning, showParticles, showTrails, showChi = false, showLattice = false, showVectors = true, showWell = true, showDomes = false, showIsoShells = false, showBackground = false, chiStrength, showBHRings = false, sigma = 1.0, tidalStretch = false, selfGravityFactor = 1.0, showLightRays = false, rayConfig, showLensingBackground = false, hideSecondaryParticle = false }) => {
  // Resize handling via R3F automatically; we can still limit pixel ratio for perf
  return (
    <Canvas
      frameloop="always"
        camera={{ position: [0, 15, 30], fov: 50 }}
      gl={{ antialias: true, powerPreference: 'high-performance', alpha: true }}
      dpr={[1, 1.75]}
      style={{ background: 'transparent' }}
    >
      <Scene 
        simulation={simulation} 
        showParticles={showParticles} 
        showTrails={showTrails} 
        showChi={showChi} 
        showLattice={showLattice} 
        showVectors={showVectors} 
        showWell={showWell} 
        showDomes={showDomes} 
        showIsoShells={showIsoShells} 
        showBackground={showBackground} 
        chiStrength={chiStrength} 
        isRunning={isRunning}
        // Forward BH-specific props from OrbitCanvas to Scene
        showBHRings={showBHRings}
        sigma={sigma}
        tidalStretch={tidalStretch}
        selfGravityFactor={selfGravityFactor}
        showLightRays={showLightRays}
        rayConfig={rayConfig}
        showLensingBackground={showLensingBackground}
        hideSecondaryParticle={hideSecondaryParticle}
      />
    </Canvas>
  );
};

/**
 * LightRays - visualize a bundle of near-light-speed rays bending under chi-field gradients.
 * Integrates simple ray dynamics using analyticChiGradientAt; maintains fixed-speed rays
 * and draws glowing trails. Designed for lensing demos; lightweight and decoupled from particle integrator.
 */
function LightRays({ simulation, isRunning, chiStrength, sigma = 1.0, rayConfig = {} }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null>; isRunning: boolean; chiStrength: number; sigma?: number; rayConfig?: { count?: number; color?: string; speed?: number; spread?: number; rows?: number; cols?: number; emitterOffset?: number; emitterWidth?: number; headSize?: number; pure?: boolean; debug?: boolean } }) {
  const groupRef = React.useRef<THREE.Group>(null);
  const raysRef = React.useRef<Array<{ pos: THREE.Vector3; vel: THREE.Vector3; line: THREE.Line; head: THREE.Mesh; maxPts: number; trail: Float32Array; len: number; gradLine?: THREE.Line; velLine?: THREE.Line }>>([]);
  const initializedRef = React.useRef<boolean>(false);
  const color = rayConfig.color || '#ffeecc';
  const debugRef = React.useRef<THREE.Group | null>(null);
  const prevTimeRef = React.useRef<number>(-1);
  const rows = Math.max(6, Math.min(128, (rayConfig.rows ?? rayConfig.count ?? 36)));
  const cols = Math.max(1, Math.min(64, (rayConfig.cols ?? 6)));
  const speed = Math.max(0.4, Math.min(2.0, rayConfig.speed ?? 1.0));
  const spreadFactor = Math.max(0.2, Math.min(2.0, rayConfig.spread ?? 0.8));
  const baseBendGain = Math.max(0.08, Math.min(1.4, 0.28 * chiStrength / Math.max(0.25, sigma))); // slightly stronger default for clearer bend
  const maxTrailPts = 240;
  const emitterRef = React.useRef<THREE.Mesh | null>(null);
  const headSize = Math.max(0.03, Math.min(0.2, rayConfig.headSize ?? 0.08));

  // Initialize rays when group mounts and simulation is ready
  useEffect(() => {
    const g = groupRef.current;
    const sim = simulation.current;
    if (!g || !sim || initializedRef.current) return;
    initializedRef.current = true;
  const { width, half } = sim.latticeWorldExtent();
  const leftFrac = Math.max(0.02, Math.min(0.2, rayConfig.emitterOffset ?? 0.06));
  const startX = -half + leftFrac * width; // start further left for longer arc
    const z = 0;
  // Base vertical half-span and ensure it's at least several times the body radius so the sheet is clearly wider than the lens
  const spreadRaw = spreadFactor * 0.6 * half; // base concentration near lens
  const bodyRadiusApprox = 0.28 * (sigma ?? 1.0); // matches visual mesh scale factor
  const preferredMin = 4.0 * bodyRadiusApprox; // "much wider" than body (~4×)
  const spread = Math.min(0.95 * half, Math.max(spreadRaw, preferredMin));
  const emitterWidth = Math.max(0.02, Math.min(0.4, rayConfig.emitterWidth ?? 0.10)) * width;
      const createRay = (x: number, y: number, hero: boolean = false) => {
      const geom = new THREE.BufferGeometry();
    const mat = new THREE.LineBasicMaterial({ color: hero ? '#aaf3ff' : color, transparent: true, opacity: hero ? 1.0 : 0.98, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
  const line = new THREE.Line(geom, mat);
      g.add(line);
      const trail = new Float32Array(maxTrailPts * 3);
      // Bright moving head to indicate direction/source
        const headGeom = new THREE.SphereGeometry(headSize, 16, 16);
  const headMat = new THREE.MeshStandardMaterial({ color: '#fff5dd', emissive: '#fff5dd', emissiveIntensity: 2.0, depthTest: false });
      const head = new THREE.Mesh(headGeom, headMat);
      g.add(head);
        const pos = new THREE.Vector3(x, y, z);
      // Pure mode: straight +x launch; otherwise include a small steer toward center for readability
      let vel = new THREE.Vector3(1, 0, 0);
      if (!rayConfig.pure) {
        const steer = Math.max(-0.45, Math.min(0.45, (-y / Math.max(1e-6, spread * 1.2)) * 0.35));
        vel = new THREE.Vector3(1, steer, 0);
      }
      vel.normalize().multiplyScalar(speed);
      // seed initial two points
      trail[0] = pos.x; trail[1] = pos.y; trail[2] = pos.z;
      trail[3] = pos.x + 0.01; trail[4] = pos.y; trail[5] = pos.z;
      geom.setAttribute('position', new THREE.BufferAttribute(trail, 3));
      geom.setDrawRange(0, 2);
      ;(geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      head.position.set(pos.x, pos.y, pos.z);
      // Optional debug vector overlays (gradient and velocity)
      let gradLine: THREE.Line | undefined;
      let velLine: THREE.Line | undefined;
      if (rayConfig.debug) {
        const mkLine = (hex: string) => {
          const g2 = new THREE.BufferGeometry();
          g2.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
          const m2 = new THREE.LineBasicMaterial({
            color: hex,
            transparent: true,
            opacity: 1.0,
            depthWrite: false,
            depthTest: false,
            blending: THREE.AdditiveBlending,
          });
          const l2 = new THREE.Line(g2, m2);
          l2.renderOrder = 10; // draw on top of points and meshes
          return l2;
        };
        gradLine = mkLine('#3cff7a');
        velLine = mkLine('#4da4ff');
        g.add(gradLine);
        g.add(velLine);
      }
      raysRef.current.push({ pos, vel, line, head, maxPts: maxTrailPts, trail, len: 2, gradLine, velLine });
    };
      // Build grid of columns × rows across the emitter width
      for (let ci = 0; ci < cols; ci++) {
        const u = cols === 1 ? 0.0 : ci / (cols - 1);
        const x0 = startX + u * emitterWidth;
        for (let ri = 0; ri < rows; ri++) {
          const v = rows === 1 ? 0.0 : (ri / (rows - 1)) * 2 - 1; // [-1,1]
          const y0 = v * spread;
          const hero = (ci === Math.floor(cols / 2) && ri === Math.floor(rows * 0.6));
          createRay(x0, y0, hero);
        }
      }

    // Add one hero ray near a grazing impact parameter for dramatic bending
    try {
      const s = sim.getState();
      const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
      const rsAnalogue = (sigma * Math.sqrt(Math.max(1e-6, primary.mass))) / Math.SQRT2;
      const heroY = Math.max(-spread * 0.9, Math.min(spread * 0.9, 1.1 * rsAnalogue));
      createRay(startX + 0.5 * emitterWidth, heroY, true);
    } catch {}

    // Add a subtle vertical emitter panel only when debugging is enabled
    if (rayConfig.debug) {
      const emitterGeom = new THREE.PlaneGeometry(
        Math.max(0.02 * width, emitterWidth * 1.02),
        Math.min(1.9 * half, spread * 2.2)
      );
      const emitterMat = new THREE.MeshBasicMaterial({
        color: '#ffffff', transparent: true, opacity: 0.06,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
      });
      const emitter = new THREE.Mesh(emitterGeom, emitterMat);
      emitter.rotation.y = Math.PI; // 180° turn of emitter panel
      emitter.position.set(startX + emitterWidth * 0.5 - 0.01 * width, 0, 0);
      g.add(emitter);
      emitterRef.current = emitter;
    }

    // Debug helpers: emitter bounds rectangle and center marker
    if (rayConfig.debug) {
      const dbg = new THREE.Group();
      const rect = new THREE.BufferGeometry();
      const verts = new Float32Array([
        startX, -spread, 0,
        startX + emitterWidth, -spread, 0,
        startX + emitterWidth, spread, 0,
        startX, spread, 0,
        startX, -spread, 0,
      ]);
      rect.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      const rectLine = new THREE.Line(rect, new THREE.LineBasicMaterial({ color: '#ffaa00', transparent: true, opacity: 0.8 }));
      dbg.add(rectLine);
      // Lens center marker and RS circle
      try {
        const s = sim.getState();
        const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
        const center = new THREE.Vector3(primary.position[0], primary.position[1], primary.position[2]);
        const centerMesh = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), new THREE.MeshBasicMaterial({ color: '#ff4477' }));
        centerMesh.position.copy(center);
        dbg.add(centerMesh);
        const rs = (sigma * Math.sqrt(Math.max(1e-6, primary.mass))) / Math.SQRT2;
        const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.01, 8, 64), new THREE.MeshBasicMaterial({ color: '#ff4477' }));
        ring.position.copy(center);
        ring.scale.set(rs, rs, rs);
        dbg.add(ring);
      } catch {}
      g.add(dbg);
      debugRef.current = dbg;
    }
  }, [simulation, rows, cols, spreadFactor, speed, color, maxTrailPts, headSize, rayConfig.emitterOffset, rayConfig.emitterWidth]);

  // React to debug toggle after initialization: create or hide lines on demand
  useEffect(() => {
    const g = groupRef.current;
    if (!g || raysRef.current.length === 0) return;
    const want = !!rayConfig.debug;
    const sim = simulation.current;
    for (const r of raysRef.current) {
      const ensureLine = (existing: THREE.Line | undefined, colorHex: string) => {
        if (existing) return existing;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
        const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 1.0, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
        const line = new THREE.Line(geo, mat);
        line.renderOrder = 10;
        g.add(line);
        return line;
      };
      if (want) {
        r.gradLine = ensureLine(r.gradLine, '#3cff7a');
        r.velLine = ensureLine(r.velLine, '#4da4ff');
        if (r.gradLine) r.gradLine.visible = true;
        if (r.velLine) r.velLine.visible = true;
      } else {
        if (r.gradLine) r.gradLine.visible = false;
        if (r.velLine) r.velLine.visible = false;
      }
    }
    // Lazily create or hide emitter panel based on debug toggle
    if (want) {
      if (!emitterRef.current && sim) {
        const { width, half } = sim.latticeWorldExtent();
        const leftFrac = Math.max(0.02, Math.min(0.2, rayConfig.emitterOffset ?? 0.06));
        const startX = -half + leftFrac * width;
        const emitterWidth = Math.max(0.02, Math.min(0.4, rayConfig.emitterWidth ?? 0.10)) * width;
        const spreadRaw = spreadFactor * 0.6 * half;
        const bodyRadiusApprox = 0.28 * (sigma ?? 1.0);
        const preferredMin = 4.0 * bodyRadiusApprox;
        const spread = Math.min(0.95 * half, Math.max(spreadRaw, preferredMin));
        const emitterGeom = new THREE.PlaneGeometry(
          Math.max(0.02 * width, emitterWidth * 1.02),
          Math.min(1.9 * half, spread * 2.2)
        );
        const emitterMat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.06, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false });
        const emitter = new THREE.Mesh(emitterGeom, emitterMat);
        emitter.rotation.y = Math.PI;
        emitter.position.set(startX + emitterWidth * 0.5 - 0.01 * width, 0, 0);
        g.add(emitter);
        emitterRef.current = emitter;
      } else if (emitterRef.current) {
        emitterRef.current.visible = true;
      }
    } else if (emitterRef.current) {
      emitterRef.current.visible = false;
    }
  }, [rayConfig.debug]);

  // Integrate rays each frame using analytic gradient; renormalize to fixed speed
  useFrame((_state, _delta) => {
    if (!isRunning) return; // freeze rays when paused
    const sim = simulation.current;
    const g = groupRef.current;
    if (!sim || !g || raysRef.current.length === 0) return;
    // Detect reset (time jumped backwards); re-seed rays
    const t = sim.getState?.().time ?? 0;
    if (prevTimeRef.current >= 0 && t < prevTimeRef.current - 1e-6) {
      const { width, half } = sim.latticeWorldExtent();
      const leftFrac = Math.max(0.02, Math.min(0.2, rayConfig.emitterOffset ?? 0.06));
      const startX = -half + leftFrac * width;
      const emitterWidth = Math.max(0.02, Math.min(0.4, rayConfig.emitterWidth ?? 0.10)) * width;
      const spreadRaw = spreadFactor * 0.6 * half;
      const bodyRadiusApprox = 0.28 * (sigma ?? 1.0);
      const preferredMin = 4.0 * bodyRadiusApprox;
      const spread = Math.min(0.95 * half, Math.max(spreadRaw, preferredMin));
      const z = 0;
      for (const r of raysRef.current) {
        const newY = (Math.random() * 2 - 1) * spread;
        const newX = startX + Math.random() * emitterWidth;
        r.pos.set(newX, newY, z);
        const steer = Math.max(-0.45, Math.min(0.45, (-newY / Math.max(1e-6, half * 0.72 * spreadFactor)) * 0.35));
        r.vel.set(1, rayConfig.pure ? 0 : steer, 0).normalize().multiplyScalar(speed);
        r.len = 2;
        r.trail[0] = r.pos.x; r.trail[1] = r.pos.y; r.trail[2] = z;
        r.trail[3] = r.pos.x + 0.01; r.trail[4] = r.pos.y; r.trail[5] = z;
        r.head.position.set(r.pos.x, r.pos.y, z);
        const geo = r.line.geometry as THREE.BufferGeometry;
        geo.setAttribute('position', new THREE.BufferAttribute(r.trail, 3));
        geo.computeBoundingSphere();
      }
    }
    prevTimeRef.current = t;
    const dtSim = sim.getDt?.() ?? 0.003;
    const dt = Math.min(0.02, dtSim * 4); // small substep for smooth curves
  const { width, half } = sim.latticeWorldExtent();
  const leftFrac = Math.max(0.02, Math.min(0.2, rayConfig.emitterOffset ?? 0.06));
  const startX = -half + leftFrac * width; // match initialization start
  const emitterWidth = Math.max(0.02, Math.min(0.4, rayConfig.emitterWidth ?? 0.10)) * width;
    const endX = half - 0.05 * width;
    const z = 0;
  const spreadRawFrame = spreadFactor * 0.6 * half;
  const bodyRadiusApproxF = 0.28 * (sigma ?? 1.0);
  const preferredMinF = 4.0 * bodyRadiusApproxF;
  const spread = Math.min(0.95 * half, Math.max(spreadRawFrame, preferredMinF));
    for (const r of raysRef.current) {
      // Determine lens center and analogue RS to avoid unphysical absorption
      const s = sim.getState();
      const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
      const cx = primary.position[0];
      const cy = primary.position[1];
      const dx = r.pos.x - cx;
      const dy = r.pos.y - cy;
      const rr = Math.hypot(dx, dy) || 1e-6;
      const rsAnalogue = (sigma * Math.sqrt(Math.max(1e-6, primary.mass))) / Math.SQRT2;
      const coreRadius = 1.15 * rsAnalogue; // soft core to steer rays around

      // Sample gradient at current position
      const grad = sim.analyticChiGradientAt([r.pos.x, r.pos.y, z]);
      // Attenuate near core to prevent absorption: gain ∝ r/coreRadius (clamped)
      const atten = Math.max(0.2, Math.min(1.0, rr / Math.max(1e-6, coreRadius)));
      const bendGain = baseBendGain * atten;
      if (rayConfig.pure) {
        // Pure LFM mapping: force ∝ -∇χ
        r.vel.x += -bendGain * grad[0] * dt;
        r.vel.y += -bendGain * grad[1] * dt;
      } else {
        // Robust attraction (user-facing demo): select sign that bends toward center and add gentle guidance
        const toCenterX = cx - r.pos.x;
        const toCenterY = cy - r.pos.y;
        const gradDotToCenter = grad[0] * toCenterX + grad[1] * toCenterY;
        const gradSign = gradDotToCenter >= 0 ? 1 : -1;
        r.vel.x += bendGain * gradSign * grad[0] * dt;
        r.vel.y += bendGain * gradSign * grad[1] * dt;
        const inv = 1 / (Math.hypot(toCenterX, toCenterY) || 1);
        const guide = 0.12;
        r.vel.x += guide * toCenterX * inv * dt;
        r.vel.y += guide * toCenterY * inv * dt;
      }
      r.vel.z = 0;
      // In pure mode, do not clamp or artificially project. In demo mode, keep gentle stability guards.
      if (!rayConfig.pure) {
        if (rr < coreRadius) {
          const nx = dx / rr;
          const ny = dy / rr;
          const vDotIn = r.vel.x * nx + r.vel.y * ny;
          if (vDotIn < 0) {
            r.vel.x -= vDotIn * nx;
            r.vel.y -= vDotIn * ny;
          }
          r.pos.x += nx * 0.005;
          r.pos.y += ny * 0.005;
        }
        if (r.vel.x < 0) {
          r.vel.x = Math.abs(r.vel.x) * 0.6;
        }
      }
      // Renormalize to fixed speed
      const vmag = Math.hypot(r.vel.x, r.vel.y) || 1;
      r.vel.multiplyScalar(speed / vmag);
      // Advance position
      r.pos.x += r.vel.x * dt;
      r.pos.y += r.vel.y * dt;
      // Append to trail (shift left if full)
      if (r.len < r.maxPts) {
        const i3 = r.len * 3;
        r.trail[i3 + 0] = r.pos.x;
        r.trail[i3 + 1] = r.pos.y;
        r.trail[i3 + 2] = z;
        r.len += 1;
      } else {
        // shift by 1 point (memmove)
        r.trail.copyWithin(0, 3);
        const i3 = (r.maxPts - 1) * 3;
        r.trail[i3 + 0] = r.pos.x;
        r.trail[i3 + 1] = r.pos.y;
        r.trail[i3 + 2] = z;
      }
      const geo = r.line.geometry as THREE.BufferGeometry;
      const attr = geo.getAttribute('position') as THREE.BufferAttribute;
      if (!attr || attr.array !== r.trail) {
        geo.setAttribute('position', new THREE.BufferAttribute(r.trail, 3));
      }
      ;(geo.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
      geo.setDrawRange(0, r.len);
      geo.computeBoundingSphere();
      // Move head
      r.head.position.set(r.pos.x, r.pos.y, z);
      // Update debug vectors (gradient and velocity) at head
      if (rayConfig.debug) {
        const pos0x = r.pos.x;
        const pos0y = r.pos.y;
        const gvec = sim.analyticChiGradientAt([pos0x, pos0y, z]);
        const gscale = 0.35; // visual scale only (slightly larger for visibility)
        if (r.gradLine) {
          const arr = (r.gradLine.geometry as THREE.BufferGeometry).getAttribute('position') as THREE.BufferAttribute;
          arr.setXYZ(0, pos0x, pos0y, z);
          arr.setXYZ(1, pos0x + gscale * gvec[0], pos0y + gscale * gvec[1], z);
          arr.needsUpdate = true;
        }
        if (r.velLine) {
          const arr2 = (r.velLine.geometry as THREE.BufferGeometry).getAttribute('position') as THREE.BufferAttribute;
          arr2.setXYZ(0, pos0x, pos0y, z);
          arr2.setXYZ(1, pos0x + 0.4 * r.vel.x, pos0y + 0.4 * r.vel.y, z);
          arr2.needsUpdate = true;
        }
      }
      // Recycle ray if it leaves bounds (right, top/bottom, or too far left)
      const leftBound = -half + Math.max(0.02, leftFrac * 0.8) * width;
      const out = r.pos.x > endX || r.pos.x < leftBound || Math.abs(r.pos.y) > half * 0.98;
      if (out) {
        const newY = (Math.random() * 2 - 1) * spread;
        const newX = startX + Math.random() * emitterWidth;
        r.pos.set(newX, newY, z);
        const steer = Math.max(-0.45, Math.min(0.45, (-newY / Math.max(1e-6, half * 0.72 * spreadFactor)) * 0.35));
        r.vel.set(1, steer, 0).normalize().multiplyScalar(speed);
        // reset trail
        r.len = 2;
        r.trail[0] = r.pos.x; r.trail[1] = r.pos.y; r.trail[2] = z;
        r.trail[3] = r.pos.x + 0.01; r.trail[4] = r.pos.y; r.trail[5] = z;
        r.head.position.set(r.pos.x, r.pos.y, z);
      }
    }
  });

  return <group ref={groupRef} />;
}

function BackgroundSourcePlane({ simulation, isRunning }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null>; isRunning: boolean }) {
  const pointsRef = React.useRef<THREE.Points>(null);
  const basePositionsRef = React.useRef<Float32Array | null>(null);
  const rows = 28 as number;
  const cols = 56 as number;
  const deflectStrength = 0.18; // visual-only scale for deflection toward lens

  // Initialize positions in a grid spanning the domain (slightly larger for edge arcs)
  useEffect(() => {
    const sim = simulation.current;
    const pts = pointsRef.current;
    if (!sim || !pts) return;
    const { width, half } = sim.latticeWorldExtent();
    const w = half * 1.6;
    const h = half * 1.2;
    const startX = 0.25 * half; // place source-plane to the right
    const positions: number[] = [];
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const u = cols === 1 ? 0 : i / (cols - 1);
        const v = rows === 1 ? 0 : j / (rows - 1);
        const x = startX + (u - 0.5) * w;
        const y = (v - 0.5) * h;
        positions.push(x, y, 0);
      }
    }
    const base = new Float32Array(positions);
    basePositionsRef.current = base;
    const geo = pts.geometry as THREE.BufferGeometry;
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(base), 3));
    geo.computeBoundingSphere();
  }, [simulation]);

  useFrame(() => {
    if (!isRunning) return; // freeze background sampler when paused
    const sim = simulation.current;
    const pts = pointsRef.current;
    const base = basePositionsRef.current;
    if (!sim || !pts || !base) return;
    const { half } = sim.latticeWorldExtent();
    const geo = pts.geometry as THREE.BufferGeometry;
    const attr = geo.getAttribute('position') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let idx = 0; idx < base.length; idx += 3) {
      const bx = base[idx];
      const by = base[idx + 1];
      const g = sim.analyticChiGradientAt([bx, by, 0]);
      const dx = -deflectStrength * g[0];
      const dy = -deflectStrength * g[1];
      // Clamp very near the core to keep the sampler stable
      const mag = Math.hypot(dx, dy);
      const maxD = 0.25 * half;
      const scale = mag > maxD ? maxD / (mag + 1e-9) : 1.0;
      arr[idx] = bx + dx * scale;
      arr[idx + 1] = by + dy * scale;
      // z remains 0
    }
    attr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry />
      <pointsMaterial
        size={0.02}
        sizeAttenuation
        color={0x66ccff}
        opacity={0.85}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

function GravityWell({ simulation }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null> }) {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const N = 64; // grid resolution
  // world units width/height; adapt to lattice size (slightly inset for aesthetics)
  const [size, setSize] = React.useState<number>(6.0);
  const zOffset = -0.05; // push below particle plane
  const scale = 0.3; // displacement scale
  const lastUpdateRef = React.useRef<number>(0);
  const updateEveryMs = 33; // ~30 FPS for this layer

  // Initialize geometry once
  useEffect(() => {
    if (!meshRef.current) return;
    // Determine initial size from simulation (fallback to default)
    const simLocal = simulation.current as any;
    const width = simLocal?.latticeWorldExtent?.().width ?? 12.0;
    setSize(Math.max(2.0, 0.9 * width));
    // Orientation
    meshRef.current.rotation.x = -Math.PI / 2; // XY plane
    meshRef.current.position.z = 0; // centered
  }, []);

  // (Re)build geometry with color attribute whenever size changes
  useEffect(() => {
    if (!meshRef.current) return;
    // Dispose previous geometry if exists
    if (meshRef.current.geometry) {
      meshRef.current.geometry.dispose();
    }
    const geom = new THREE.PlaneGeometry(size, size, N, N);
    const colors = new Float32Array((geom.attributes.position as THREE.BufferAttribute).count * 3);
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    meshRef.current.geometry = geom;
  }, [size]);

  useFrame(() => {
    const sim = simulation.current as any;
    const mesh = meshRef.current;
    if (!sim || !mesh) return;
    const now = performance.now();
    if (now - lastUpdateRef.current < updateEveryMs) return;
    lastUpdateRef.current = now;

    const simLocal = simulation.current as any;
    const width = simLocal?.latticeWorldExtent?.().width ?? size;
    const targetSize = Math.max(2.0, 0.9 * width);
    if (Math.abs(targetSize - size) > 1e-3) {
      setSize(targetSize);
    }
    const geom = mesh.geometry as THREE.PlaneGeometry;
    const posAttr = geom.attributes.position as THREE.BufferAttribute;
    const colAttr = geom.attributes.color as THREE.BufferAttribute;

    // Determine chi range first pass, then write in second pass for stable colors
    let minChi = Infinity, maxChi = -Infinity;
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const y = posAttr.getY(i);
      const chi = sim.analyticChiAt([x, y, 0]);
      if (chi < minChi) minChi = chi;
      if (chi > maxChi) maxChi = chi;
    }
    const baseline = sim.chiBaseline();
    const span = Math.max(1e-6, maxChi - minChi);
    // Ensure color attribute exists (dynamic rebuild safety)
    if (!geom.getAttribute('color')) {
      const newColors = new Float32Array(posAttr.count * 3);
      geom.setAttribute('color', new THREE.BufferAttribute(newColors, 3));
    }
    const colAttrSafe = geom.getAttribute('color') as THREE.BufferAttribute;
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const y = posAttr.getY(i);
      const chi = sim.analyticChiAt([x, y, 0]);
      const depth = -scale * (chi - baseline);
      const z = zOffset + Math.max(-1.0, Math.min(0.5, depth));
      posAttr.setZ(i, z);
      const t = (chi - minChi) / span;
      const r = 0.05 + 0.10 * t;
      const g = 0.35 + 0.55 * t;
      const b = 0.45 + 0.45 * t;
      colAttrSafe.setXYZ(i, r, g, b);
    }
    posAttr.needsUpdate = true;
    colAttrSafe.needsUpdate = true;
    geom.computeVertexNormals();
  });

  return (
    <mesh ref={meshRef} position={[0, 0, -0.001]}>
      <planeGeometry key={size} args={[size, size, N, N]} />
      <meshStandardMaterial vertexColors transparent opacity={0.7} side={THREE.DoubleSide} />
    </mesh>
  );
}

function FieldDomes({ simulation }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null> }) {
  const groupRef = React.useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    const sim = simulation.current;
    const g = groupRef.current;
    if (!sim || !g) return;
    const t = clock.getElapsedTime();
    const s = sim.getState();
    const masses = [s.particle1, s.particle2];
    // Ensure we have two child meshes (create if missing)
    while (g.children.length < 2) {
      const geom = new THREE.SphereGeometry(1, 32, 32);
      const mat = new THREE.ShaderMaterial({
        transparent: true,
        uniforms: {
          uTime: { value: 0 },
          uColor: { value: new THREE.Color('#4A90E2') }, // Earth blue default
          uMass: { value: 1.0 },
        },
        vertexShader: `
          varying vec3 vPos;
          varying float vDist;
          void main() {
            vPos = position;
            vDist = length(position);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform float uTime;
          uniform vec3 uColor;
          uniform float uMass;
          varying vec3 vPos;
          varying float vDist;
          
          void main() {
            float r = vDist;
            
            // Dark at center, lighter toward edges (inverse of typical)
            float brightness = smoothstep(0.0, 1.0, r);
            
            // Subtle color shift over time (very slow, gentle)
            float shift = sin(uTime * 0.3) * 0.1 + 0.5;
            vec3 colorShifted = mix(uColor * 0.6, uColor * 1.2, shift);
            
            // Apply brightness gradient
            vec3 finalColor = colorShifted * mix(0.3, 1.0, brightness);
            
            // Alpha falloff (still want edges to fade)
            float alpha = exp(-r * r * 0.8) * (0.4 + brightness * 0.3);
            
            // Fresnel rim for depth
            float fres = pow(1.0 - abs(dot(normalize(vPos), vec3(0.0, 0.0, 1.0))), 2.0);
            alpha = clamp(alpha + fres * 0.25, 0.0, 1.0);
            
            gl_FragColor = vec4(finalColor, alpha);
          }
        `,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(geom, mat);
      g.add(mesh);
    }
    masses.forEach((m, idx) => {
      const mesh = g.children[idx] as THREE.Mesh;
      const mat = mesh.material as THREE.ShaderMaterial;
      mat.uniforms.uTime.value = t;
      mat.uniforms.uMass.value = m.mass;
      // Color per body: Earth blue, Moon gray
      mat.uniforms.uColor.value.set(idx === 0 ? '#4A90E2' : '#9CA3AF');
      // Static size based on mass (no pulsation)
      const baseR = 0.9 * Math.sqrt(m.mass);
      mesh.scale.setScalar(baseR);
      mesh.position.set(m.position[0], m.position[1], m.position[2]);
    });
  });
  return <group ref={groupRef} />;
}

function IsoShells({ simulation }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null> }) {
  const pointsRef = React.useRef<THREE.Points>(null);
  const lastUpdateRef = React.useRef<number>(0);
  const updateMs = 120; // ~8 FPS refresh
  const shellRes = 24; // lat/long resolution per shell
  const thresholds = [0.4, 0.7]; // relative above baseline
  useFrame(() => {
    const sim = simulation.current;
    if (!sim || !pointsRef.current) return;
    const now = performance.now();
    if (now - lastUpdateRef.current < updateMs) return;
    lastUpdateRef.current = now;
    const baseline = sim.chiBaseline();
    const posArray: number[] = [];
    const colArray: number[] = [];
    const state = sim.getState();
    const centers = [state.particle1.position, state.particle2.position];
    centers.forEach((center, ci) => {
      thresholds.forEach((thr, ti) => {
        const radius = 1.1 * Math.sqrt((ci === 0 ? state.particle1.mass : state.particle2.mass)) * (0.7 + 0.3 * ti);
        for (let iy = 0; iy < shellRes; iy++) {
          const v = iy / (shellRes - 1);
          const theta = v * Math.PI; // polar
          for (let ix = 0; ix < shellRes; ix++) {
            const u = ix / (shellRes - 1);
            const phi = u * Math.PI * 2.0; // azimuth
            const x = center[0] + radius * Math.sin(theta) * Math.cos(phi);
            const y = center[1] + radius * Math.sin(theta) * Math.sin(phi);
            const z = center[2] + radius * Math.cos(theta);
            const chi = sim.analyticChiAt([x, y, z]);
            const rel = chi - baseline;
            if (rel < thr) continue; // skip below threshold
            posArray.push(x, y, z);
            // Color ramp per shell & mass
            const base = ci === 0 ? new THREE.Color('#00d9ff') : new THREE.Color('#ff6b35');
            const c = base.clone().lerp(new THREE.Color('#ffffff'), 0.15 * ti);
            colArray.push(c.r, c.g, c.b);
          }
        }
      });
    });
    const geo = pointsRef.current.geometry as THREE.BufferGeometry;
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(posArray), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colArray), 3));
    geo.computeBoundingSphere();
  });
  return (
    <points ref={pointsRef}>
      <bufferGeometry />
      <pointsMaterial size={0.05} vertexColors transparent opacity={0.85} blending={THREE.AdditiveBlending} />
    </points>
  );
}

function BlackHoleSphere({ simulation }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null> }) {
  const meshRef = React.useRef<THREE.Mesh>(null);

  useFrame(() => {
    const sim = simulation.current as any;
    const mesh = meshRef.current;
    if (!sim || !mesh) return;
    const s = sim.getState();
    // Position at the heavier body (black hole)
    const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
    mesh.position.set(primary.position[0], primary.position[1], primary.position[2]);
  });

  return (
    <mesh ref={meshRef}>
        <sphereGeometry args={[2.0, 32, 32]} />
      <meshStandardMaterial 
        color="#0a0a1a" 
        emissive="#000000" 
        emissiveIntensity={0}
        roughness={0.2}
        metalness={0.8}
      />
    </mesh>
  );
}

function HorizonRings({ simulation, sigma = 1.0 }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null>; sigma?: number }) {
  const ringsRef = React.useRef<THREE.Group>(null);

  useEffect(() => {
    const g = ringsRef.current;
    if (!g) return;
    // Ensure three child torus meshes exist (unit radius, scaled per frame)
    const colors = ['#ff375f', '#ffb020', '#00e0ff'];
    while (g.children.length < 3) {
      const geom = new THREE.TorusGeometry(1, 0.12, 24, 128);
      const mat = new THREE.MeshBasicMaterial({ color: colors[g.children.length], transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthTest: false });
      const mesh = new THREE.Mesh(geom, mat);
      // Keep default orientation (XY plane, axis along Z)
      mesh.rotation.set(0, 0, 0);
      g.add(mesh);
    }
  }, []);

  useFrame(() => {
    const sim = simulation.current as any;
    const g = ringsRef.current;
    if (!sim || !g) return;
    const s = sim.getState();
    const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
    const m = primary.mass;
    const center = primary.position as [number, number, number];
    const rs = (sigma * Math.sqrt(Math.max(1e-6, m))) / Math.SQRT2; // RS_analogue
    const radii = [rs, 1.5 * rs, 3.0 * rs];

    for (let idx = 0; idx < 3; idx++) {
      const mesh = g.children[idx] as THREE.Mesh;
      const r = radii[idx];
      mesh.position.set(center[0], center[1], center[2]);
      mesh.scale.set(r, r, r); // scale unit torus to radius r
    }
  });

  return <group ref={ringsRef} />;
}
export default OrbitCanvas;

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

import React, { useRef, useEffect, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stats, Effects } from '@react-three/drei';
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

function Particles({ simulation, showParticles, showTrails }: { simulation: React.MutableRefObject<BinaryOrbitSimulation | null>; showParticles: boolean; showTrails: boolean; }) {
  const p1Ref = useRef<THREE.Mesh>(null);
  const p2Ref = useRef<THREE.Mesh>(null);

  const maxPoints = 800;
  const trail1Ref = useRef<TrailData>({ points: [], line: null, material: null, positionBuffer: null, isDirty: false });
  const trail2Ref = useRef<TrailData>({ points: [], line: null, material: null, positionBuffer: null, isDirty: false });
  const trailGroupRef = useRef<THREE.Group>(null);

  // Initialize trail lines with reusable buffers
  useEffect(() => {
    if (!trailGroupRef.current) return;
    const createTrail = (color: string): TrailData => {
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 });
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

    // Update particle positions
    if (p1Ref.current) {
      p1Ref.current.position.set(s.particle1.position[0], s.particle1.position[1], s.particle1.position[2]);
    }
    if (p2Ref.current) {
      p2Ref.current.position.set(s.particle2.position[0], s.particle2.position[1], s.particle2.position[2]);
    }

    // Trails (optimized with ring buffer - no per-frame allocations)
    if (showTrails) {
      const addPoint = (trail: TrailData, pos: THREE.Vector3) => {
        trail.points.push(pos.clone());
        if (trail.points.length > maxPoints) trail.points.shift();
        
        // Reuse preallocated buffer instead of creating new Float32Array
        const buffer = trail.positionBuffer!;
        trail.points.forEach((p, i) => {
          buffer[i * 3] = p.x;
          buffer[i * 3 + 1] = p.y;
          buffer[i * 3 + 2] = p.z;
        });
        
        // Only update geometry if buffer changed (dirty tracking)
        const geometry = trail.line!.geometry;
        const existingAttr = geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
        
        // Initialize or update attribute
        if (!existingAttr || existingAttr.array !== buffer) {
          geometry.setAttribute('position', new THREE.BufferAttribute(buffer, 3));
          trail.isDirty = true;
        } else if (trail.isDirty) {
          existingAttr.needsUpdate = true;
          trail.isDirty = false;
        }
        
        geometry.setDrawRange(0, trail.points.length);
        geometry.computeBoundingSphere();
      };
      if (p1Ref.current) addPoint(trail1Ref.current, p1Ref.current.position);
      if (p2Ref.current) addPoint(trail2Ref.current, p2Ref.current.position);
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
          {/* Earth: larger, blue */}
          <mesh ref={p1Ref}>
            <sphereGeometry args={[0.28, 32, 32]} />
            <meshStandardMaterial emissive={'#4A90E2'} color={'#4A90E2'} emissiveIntensity={1.2} />
          </mesh>
          {/* Moon: smaller, gray */}
          <mesh ref={p2Ref}>
            <sphereGeometry args={[0.15, 32, 32]} />
            <meshStandardMaterial emissive={'#9CA3AF'} color={'#9CA3AF'} emissiveIntensity={0.8} />
          </mesh>
        </>
      )}
    </group>
  );
}

function Starfield() {
  const starsRef = React.useRef<THREE.Points>(null);
  
  React.useEffect(() => {
    if (!starsRef.current) return;
    
    const starCount = 2000;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);
    
    // Generate stars in a large sphere around the scene
    for (let i = 0; i < starCount; i++) {
      // Random position in spherical volume
      const radius = 50 + Math.random() * 100;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
      
      // Slight color variation
      const tint = Math.random();
      if (tint < 0.1) {
        colors[i * 3] = 0.7; colors[i * 3 + 1] = 0.8; colors[i * 3 + 2] = 1.0; // blue-ish
      } else if (tint > 0.9) {
        colors[i * 3] = 1.0; colors[i * 3 + 1] = 0.9; colors[i * 3 + 2] = 0.8; // warm
      } else {
        colors[i * 3] = 0.9; colors[i * 3 + 1] = 0.95; colors[i * 3 + 2] = 1.0; // white
      }
      
      sizes[i] = 0.5 + Math.random() * 1.5;
    }
    
    const geometry = starsRef.current.geometry;
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  }, []);
  
  return (
    <points ref={starsRef}>
      <bufferGeometry />
      <pointsMaterial
        size={2}
        sizeAttenuation={true}
        vertexColors={true}
        transparent={true}
        opacity={0.8}
        depthWrite={false}
      />
    </points>
  );
}

function DecorativeSun() {
  return (
    <mesh position={[-25, 15, -40]}>
      <sphereGeometry args={[3, 32, 32]} />
      <meshBasicMaterial color="#FFEB9C" />
      {/* Soft glow */}
      <pointLight color="#FFD280" intensity={0.5} distance={20} decay={2} />
    </mesh>
  );
}

function Scene({ simulation, showParticles, showTrails, showChi = false, showLattice = false, showVectors = true, showWell = true, showDomes = false, showIsoShells = false, showBackground = false, isRunning, chiStrength }: OrbitCanvasProps) {
  // Soft ambient
  return (
    <>
      {/* Always apply a dark background color; optional decorative elements below */}
      <color attach="background" args={[0.039, 0.055, 0.152]} /> {/* #0a0e27 */}
      {showBackground && (
        <>
          <Starfield />
          <DecorativeSun />
        </>
      )}
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

  {/* Force vectors (gradient arrows) */}
  {showVectors && <ForceVectors simulation={simulation} />}

  {/* Gravity well heightfield (chi-based) */}
  {showWell && <GravityWell simulation={simulation} />}

  {/* Gaussian energy domes */}
  {showDomes && <FieldDomes simulation={simulation} />}

  {/* Iso-shells (point-based thresholds) */}
  {showIsoShells && <IsoShells simulation={simulation} />}

  <Particles simulation={simulation} showParticles={showParticles} showTrails={showTrails} />
      <OrbitControls enablePan={false} enableDamping dampingFactor={0.08} maxDistance={25} minDistance={2} />
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

export const OrbitCanvas: React.FC<OrbitCanvasProps> = ({ simulation, isRunning, showParticles, showTrails, showChi = false, showLattice = false, showVectors = true, showWell = true, showDomes = false, showIsoShells = false, showBackground = false, chiStrength }) => {
  // Resize handling via R3F automatically; we can still limit pixel ratio for perf
  return (
    <Canvas
      camera={{ position: [0, 4, 8], fov: 45 }}
      gl={{ antialias: true, powerPreference: 'high-performance', alpha: true }}
      dpr={[1, 1.75]}
      style={{ background: 'transparent' }}
    >
      <Scene simulation={simulation} showParticles={showParticles} showTrails={showTrails} showChi={showChi} showLattice={showLattice} showVectors={showVectors} showWell={showWell} showDomes={showDomes} showIsoShells={showIsoShells} showBackground={showBackground} chiStrength={chiStrength} isRunning={isRunning} />
    </Canvas>
  );
};

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
export default OrbitCanvas;

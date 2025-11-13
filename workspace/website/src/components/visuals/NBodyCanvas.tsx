/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

'use client';

import React, { useRef, useEffect, useCallback } from 'react';
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
  // Advanced visualization layers (align with StandardVisualizationOptions)
  showChi?: boolean;
  showLattice?: boolean;
  showVectors?: boolean;
  showWell?: boolean;
  showDomes?: boolean;
  showIsoShells?: boolean;
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

function LatticeBox({ simulation }: { simulation: React.MutableRefObject<NBodyOrbitSimulation | null> }) {
  const boxRef = React.useRef<THREE.Mesh>(null);
  const [size, setSize] = React.useState<number>(6.0);
  useEffect(() => {
    const sim = simulation.current;
    if (!sim || !boxRef.current) return;
    const ext = sim.latticeWorldExtent();
    setSize(ext.width);
  }, [simulation]);
  return (
    <mesh ref={boxRef}>
      <boxGeometry args={[size, size, size]} />
      <meshBasicMaterial color={'#2a355e'} wireframe transparent opacity={0.35} />
    </mesh>
  );
}

function ChiPointCloud({ simulation }: { simulation: React.MutableRefObject<NBodyOrbitSimulation | null> }) {
  const pointsRef = React.useRef<THREE.Points>(null);
  const lastUpdateRef = React.useRef<number>(0);
  const updateEveryMs = 500; // ~2 Hz

  const resample = useCallback(async () => {
    const sim = simulation.current;
    if (!sim || !pointsRef.current) return;
    const lat = sim.getLattice() as any;
    if (!lat?.readChiField) return;
    const chi = await lat.readChiField();
    const { N, dx, half } = (() => {
      const ext = sim.latticeWorldExtent();
      return { N: ext.N, dx: ext.dx, half: ext.half };
    })();
    const skip = Math.max(2, Math.floor(N / 16));
    const samples: number[] = [];
    const posArray: number[] = [];
    for (let iz = 0; iz < N; iz += skip) {
      for (let iy = 0; iy < N; iy += skip) {
        for (let ix = 0; ix < N; ix += skip) {
          const idx = iz * N * N + iy * N + ix;
          const x = (ix - N / 2) * dx;
          const y = (iy - N / 2) * dx;
          const z = (iz - N / 2) * dx;
          posArray.push(x, y, z);
          samples.push(chi[idx]);
        }
      }
    }
    let minChi = Infinity, maxChi = -Infinity;
    for (const v of samples) { if (v < minChi) minChi = v; if (v > maxChi) maxChi = v; }
    const range = Math.max(1e-6, maxChi - minChi);
    const positions = new Float32Array(posArray);
    const colors = new Float32Array(samples.length * 3);
    for (let i = 0; i < samples.length; i++) {
      const t = (samples[i] - minChi) / range;
      const r = 0.0;
      const g = 0.55 + 0.45 * t;
      const b = 0.75 + 0.25 * t;
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    }
    const geoObj = pointsRef.current?.geometry as THREE.BufferGeometry | undefined;
    if (!geoObj) return;
    geoObj.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geoObj.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geoObj.computeBoundingSphere();
  }, [simulation]);

  useEffect(() => { resample(); }, [resample]);
  useFrame(() => {
    const now = performance.now();
    if (now - lastUpdateRef.current >= updateEveryMs) {
      lastUpdateRef.current = now;
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

function ForceVectors({ simulation }: { simulation: React.MutableRefObject<NBodyOrbitSimulation | null> }) {
  const linesRef = React.useRef<THREE.LineSegments>(null);
  const conesGroupRef = React.useRef<THREE.Group>(null);
  const maxArrows = 96;
  const conePoolRef = React.useRef<THREE.Mesh[]>([]);

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
    // Ensure cone pool is initialized before using
    if (conePoolRef.current.length === 0) return;
    const s = sim.getState();
    const positions = new Float32Array(maxArrows * 2 * 3);
    const colors = new Float32Array(maxArrows * 2 * 3);
    let cursor = 0;

    for (const cone of conePoolRef.current) cone.visible = false;

    const dimBase = new THREE.Color('#335577');
    const bright = new THREE.Color('#77ccff');

    const addRings = (center: [number, number, number]) => {
      const radii = [0.5, 0.9, 1.4];
      for (const radius of radii) {
        const count = 16;
        for (let i = 0; i < count; i++) {
          if (cursor >= maxArrows) return;
          const theta = (i / count) * Math.PI * 2;
          const phi = Math.PI / 12;
          const pos: [number, number, number] = [
            center[0] + radius * Math.cos(theta) * Math.cos(phi),
            center[1] + radius * Math.sin(theta) * Math.cos(phi),
            center[2] + radius * Math.sin(phi),
          ];
          // Use synchronous analytic gradient (no await needed)
          const g = sim.analyticChiGradientAt(pos);
          const grad = new THREE.Vector3(g[0], g[1], g[2]);
          const gradMag = grad.length();
          const len = Math.min(0.7, Math.max(0.15, gradMag * 2.0));
          const dir = grad.normalize().multiplyScalar(len);
          const from = new THREE.Vector3(pos[0], pos[1], pos[2]);
          const to = from.clone().add(dir);
          const intensity = Math.min(1.0, gradMag * 4.0);
          const finalColor = dimBase.clone().lerp(bright, intensity);
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
          const cone = conePoolRef.current[cursor];
          if (cone) {
            (cone.material as THREE.MeshBasicMaterial).color.copy(finalColor);
            cone.position.copy(to);
            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
            cone.visible = true;
          } else {
            // Pool not large enough or not ready; bail out safely
            return;
          }
          cursor++;
        }
      }
    };

    // Around each particle (synchronous now)
    const centers = s.particles.map(p => p.position);
    for (const c of centers) {
      addRings(c as any);
    }

    // Update geometry (synchronous, no promise needed)
    const lines = linesRef.current;
    if (!lines || !lines.geometry) return;
    const geo = lines.geometry as THREE.BufferGeometry;
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

function GravityWell({ simulation }: { simulation: React.MutableRefObject<NBodyOrbitSimulation | null> }) {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const N = 64;
  const [size, setSize] = React.useState<number>(6.0);
  const zOffset = -0.05;
  const scale = 0.3;
  const lastUpdateRef = React.useRef<number>(0);
  const updateEveryMs = 66; // ~15 FPS

  useEffect(() => {
    if (!meshRef.current) return;
    const sim = simulation.current;
    const width = sim?.latticeWorldExtent().width ?? 12.0;
    setSize(Math.max(2.0, 0.9 * width));
    meshRef.current.rotation.x = -Math.PI / 2;
    meshRef.current.position.z = 0;
  }, []);

  useEffect(() => {
    if (!meshRef.current) return;
    if (meshRef.current.geometry) meshRef.current.geometry.dispose();
    const geom = new THREE.PlaneGeometry(size, size, N, N);
    const colors = new Float32Array((geom.attributes.position as THREE.BufferAttribute).count * 3);
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    meshRef.current.geometry = geom;
  }, [size]);

  useFrame(async () => {
    const sim = simulation.current;
    const mesh = meshRef.current;
    if (!sim || !mesh) return;
    const now = performance.now();
    if (now - lastUpdateRef.current < updateEveryMs) return;
    lastUpdateRef.current = now;
    const ext = sim.latticeWorldExtent();
    const target = Math.max(2.0, 0.9 * ext.width);
    if (Math.abs(target - size) > 1e-3) setSize(target);
  const geom = mesh.geometry as THREE.PlaneGeometry | undefined;
  if (!geom) return;
    const posAttr = geom.attributes.position as THREE.BufferAttribute;
    const colAttr = geom.attributes.color as THREE.BufferAttribute;

    // Read chi field once and slice near z≈0
    const lat = sim.getLattice() as any;
    const chi = lat?.readChiField ? await lat.readChiField() : null;
    const Nlat = ext.N;
    const half = Nlat / 2;
    const dx = ext.dx;
    let minChi = Infinity, maxChi = -Infinity;
    // First pass: compute chi at grid points
    const chiAt: number[] = [];
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const y = posAttr.getY(i);
      const ix = Math.round(x / dx + half);
      const iy = Math.round(y / dx + half);
      const iz = Math.round(half);
      const idx = ((iz % Nlat + Nlat) % Nlat) * Nlat * Nlat + ((iy % Nlat + Nlat) % Nlat) * Nlat + ((ix % Nlat + Nlat) % Nlat);
      const v = chi ? chi[idx] : 0;
      chiAt.push(v);
      if (v < minChi) minChi = v; if (v > maxChi) maxChi = v;
    }
    const span = Math.max(1e-6, maxChi - minChi);
    for (let i = 0; i < posAttr.count; i++) {
      const v = chiAt[i];
      const depth = -scale * (v - minChi);
      const z = zOffset + Math.max(-1.0, Math.min(0.5, depth));
      posAttr.setZ(i, z);
      const t = (v - minChi) / span;
      const r = 0.05 + 0.10 * t;
      const g = 0.35 + 0.55 * t;
      const b = 0.45 + 0.45 * t;
      colAttr.setXYZ(i, r, g, b);
    }
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geom.computeVertexNormals();
  });

  return (
    <mesh ref={meshRef} position={[0, 0, -0.001]}>
      <planeGeometry key={size} args={[size, size, N, N]} />
      <meshStandardMaterial vertexColors transparent opacity={0.7} side={THREE.DoubleSide} />
    </mesh>
  );
}

function FieldDomes({ simulation }: { simulation: React.MutableRefObject<NBodyOrbitSimulation | null> }) {
  const groupRef = React.useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    const sim = simulation.current;
    const g = groupRef.current;
    if (!sim || !g) return;
    const t = clock.getElapsedTime();
    const s = sim.getState();
    const masses = s.particles;
    while (g.children.length < masses.length) {
      const geom = new THREE.SphereGeometry(1, 32, 32);
      const mat = new THREE.ShaderMaterial({
        transparent: true,
        uniforms: {
          uTime: { value: 0 },
          uColor: { value: new THREE.Color('#4A90E2') },
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
            float brightness = smoothstep(0.0, 1.0, r);
            float shift = sin(uTime * 0.3) * 0.1 + 0.5;
            vec3 colorShifted = mix(uColor * 0.6, uColor * 1.2, shift);
            vec3 finalColor = colorShifted * mix(0.3, 1.0, brightness);
            float alpha = exp(-r * r * 0.8) * (0.4 + brightness * 0.3);
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
      const color = ['#4A90E2', '#9CA3AF', '#ff6b35', '#00d9ff'][idx % 4];
      mat.uniforms.uColor.value.set(color);
      const baseR = 0.9 * Math.sqrt(m.mass);
      mesh.scale.setScalar(baseR);
      mesh.position.set(m.position[0], m.position[1], m.position[2]);
    });
  });
  return <group ref={groupRef} />;
}

function IsoShells({ simulation }: { simulation: React.MutableRefObject<NBodyOrbitSimulation | null> }) {
  const pointsRef = React.useRef<THREE.Points>(null);
  const lastUpdateRef = React.useRef<number>(0);
  const updateMs = 150;
  const shellRes = 20;
  const thresholds = [0.4, 0.7];
  useFrame(() => {
    const sim = simulation.current;
    if (!sim || !pointsRef.current) return;
    const now = performance.now();
    if (now - lastUpdateRef.current < updateMs) return;
    lastUpdateRef.current = now;
    const posArray: number[] = [];
    const colArray: number[] = [];
    const s = sim.getState();
    s.particles.forEach((p, ci) => {
      thresholds.forEach((thr, ti) => {
        const radius = 1.1 * Math.sqrt(p.mass) * (0.7 + 0.3 * ti);
        for (let iy = 0; iy < shellRes; iy++) {
          const v = iy / (shellRes - 1);
          const theta = v * Math.PI;
          for (let ix = 0; ix < shellRes; ix++) {
            const u = ix / (shellRes - 1);
            const phi = u * Math.PI * 2.0;
            const x = p.position[0] + radius * Math.sin(theta) * Math.cos(phi);
            const y = p.position[1] + radius * Math.sin(theta) * Math.sin(phi);
            const z = p.position[2] + radius * Math.cos(theta);
            // Visual thresholding by radius only (approximate, placeholder for chi-based iso)
            posArray.push(x, y, z);
            const base = ci === 0 ? new THREE.Color('#00d9ff') : ci === 1 ? new THREE.Color('#ff6b35') : new THREE.Color('#c084fc');
            const c = base.clone().lerp(new THREE.Color('#ffffff'), 0.15 * ti);
            colArray.push(c.r, c.g, c.b);
          }
        }
      });
    });
    const geo = pointsRef.current?.geometry as THREE.BufferGeometry | undefined;
    if (!geo) return;
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

function Scene({ simulation, isRunning, showParticles, showTrails, showBackground = true, showChi = false, showLattice = false, showVectors = false, showWell = false, showDomes = false, showIsoShells = false }: NBodyCanvasProps) {
  return (
    <>
      <color attach="background" args={['#0a0a1a']} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      
      {/* Coordinate system helper */}
      <gridHelper args={[20, 20, '#333366', '#1a1a2e']} />
      
      {showLattice && <LatticeBox simulation={simulation} />}
      {showChi && <ChiPointCloud simulation={simulation} />}
      {showVectors && <ForceVectors simulation={simulation} />}
      {showWell && <GravityWell simulation={simulation} />}
      {showDomes && <FieldDomes simulation={simulation} />}
      {showIsoShells && <IsoShells simulation={simulation} />}

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
        maxDistance={25}
      />
    </>
  );
}

export default function NBodyCanvas({ simulation, isRunning, showParticles, showTrails, showBackground = true, showChi = false, showLattice = false, showVectors = false, showWell = false, showDomes = false, showIsoShells = false }: NBodyCanvasProps) {
  return (
    <div className="w-full h-full">
      <Canvas frameloop="always" camera={{ position: [0, 4, 8], fov: 45 }}>
        <Scene 
          simulation={simulation} 
          isRunning={isRunning}
          showParticles={showParticles}
          showTrails={showTrails}
          showBackground={showBackground}
          showChi={showChi}
          showLattice={showLattice}
          showVectors={showVectors}
          showWell={showWell}
          showDomes={showDomes}
          showIsoShells={showIsoShells}
        />
      </Canvas>
    </div>
  );
}

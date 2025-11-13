'use client';

/**
 * BinaryOrbitCanvas — interactive binary orbit preview (browser-only)
 * - Symplectic (leapfrog) 2-body integrator
 * - No WebGPU dependency; lightweight Three.js scene
 * - Preview only: official validation uses Python/CuPy harness
 */

import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { Mesh } from 'three';
import type { ExperimentDefinition } from '@/data/experiments';

interface BinaryOrbitCanvasProps {
  experiment: ExperimentDefinition;
  isRunning?: boolean;
  onMetrics?: (m: { energy?: number; energyDriftPct?: number; time?: number }) => void;
}

type Vec3 = [number, number, number];

function add(a: Vec3, b: Vec3): Vec3 { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function sub(a: Vec3, b: Vec3): Vec3 { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function scale(a: Vec3, s: number): Vec3 { return [a[0]*s, a[1]*s, a[2]*s]; }

interface Body { m: number; x: Vec3; v: Vec3; }

function useBinaryLeapfrog(isRunning: boolean, onMetrics?: (m: { energy?: number; energyDriftPct?: number; time?: number }) => void) {
  const bodiesRef = useRef<Body[]>([
    { m: 1.0, x: [-1.0, 0, 0], v: [0, 0.6, 0] },
    { m: 1.0, x: [ 1.0, 0, 0], v: [0,-0.6, 0] },
  ]);
  const dtRef = useRef(0.01);
  const GRef = useRef(1.0);
  const tRef = useRef(0);
  const E0Ref = useRef<number | null>(null);

  // Symplectic leapfrog step (kick-drift-kick)
  const step = () => {
    const [a, b] = bodiesRef.current;
    const r = sub(b.x, a.x);
    const r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2] + 1e-6; // softening
    const invR3 = 1.0 / Math.pow(r2, 1.5);
    const G = GRef.current;
    const dt = dtRef.current;
    const f = scale(r, G * a.m * b.m * invR3);

    // Kick (half)
    a.v = add(a.v, scale(f, +0.5 * dt / a.m));
    b.v = add(b.v, scale(f, -0.5 * dt / b.m));
    // Drift
    a.x = add(a.x, scale(a.v, dt));
    b.x = add(b.x, scale(b.v, dt));
    // Recompute force at new positions
    const r2b = sub(b.x, a.x);
    const r2b2 = r2b[0]*r2b[0]+r2b[1]*r2b[1]+r2b[2]*r2b[2] + 1e-6;
    const invR3b = 1.0 / Math.pow(r2b2, 1.5);
    const fb = scale(r2b, G * a.m * b.m * invR3b);
    // Kick (half)
    a.v = add(a.v, scale(fb, +0.5 * dt / a.m));
    b.v = add(b.v, scale(fb, -0.5 * dt / b.m));
  };

  useFrame(() => {
    if (isRunning) {
      step();
      tRef.current += dtRef.current;
      // compute energy and drift intermittently
      if (Math.random() < 0.2) {
        const [a, b] = bodiesRef.current;
        const v2a = a.v[0]*a.v[0]+a.v[1]*a.v[1]+a.v[2]*a.v[2];
        const v2b = b.v[0]*b.v[0]+b.v[1]*b.v[1]+b.v[2]*b.v[2];
        const KE = 0.5 * (a.m * v2a + b.m * v2b);
        const rx = b.x[0]-a.x[0];
        const ry = b.x[1]-a.x[1];
        const rz = b.x[2]-a.x[2];
        const r = Math.sqrt(rx*rx+ry*ry+rz*rz + 1e-6);
        const PE = - GRef.current * a.m * b.m / r;
        const Etot = KE + PE;
        if (E0Ref.current === null) E0Ref.current = Etot;
        const driftPct = ((Etot - (E0Ref.current as number)) / (E0Ref.current as number)) * 100;
        onMetrics?.({ energy: Etot, energyDriftPct: driftPct, time: tRef.current });
      }
    }
  });

  return bodiesRef;
}

function Bodies({ bodiesRef }: { bodiesRef: React.MutableRefObject<Body[]> }) {
  const colorA = '#60a5fa';
  const colorB = '#f472b6';
  const aRef = useRef<Mesh>(null);
  const bRef = useRef<Mesh>(null);

  useFrame(() => {
    const [a, b] = bodiesRef.current;
    if (aRef.current) aRef.current.position.set(a.x[0], a.x[1], a.x[2]);
    if (bRef.current) bRef.current.position.set(b.x[0], b.x[1], b.x[2]);
  });

  const [a, b] = bodiesRef.current;
  return (
    <>
      <mesh ref={aRef} position={[a.x[0], a.x[1], a.x[2]]}>
        <sphereGeometry args={[0.15, 32, 32]} />
        <meshStandardMaterial color={colorA} emissive={colorA} emissiveIntensity={0.5} />
      </mesh>
      <mesh ref={bRef} position={[b.x[0], b.x[1], b.x[2]]}>
        <sphereGeometry args={[0.15, 32, 32]} />
        <meshStandardMaterial color={colorB} emissive={colorB} emissiveIntensity={0.5} />
      </mesh>
    </>
  );
}

export default function BinaryOrbitCanvas({ experiment, isRunning = false, onMetrics }: BinaryOrbitCanvasProps) {
  // Hook maintains bodies and integrates when isRunning true
  const bodiesRef = useBinaryLeapfrog(isRunning, onMetrics);

  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ position: [0, 2.5, 4.5], fov: 55 }}>
        <color attach="background" args={[ '#020617' ]} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[4, 6, 4]} intensity={0.8} />
        <gridHelper args={[10, 10, '#1f2937', '#1f2937']} />
        <axesHelper args={[2]} />
        <Bodies bodiesRef={bodiesRef} />
        <OrbitControls enablePan={false} enableZoom={true} />
      </Canvas>
      <div className="pointer-events-none absolute top-2 left-2 text-[11px] text-slate-300 bg-slate-900/60 border border-slate-700 rounded px-2 py-1">
        Preview — gravity orbit (leapfrog). Official validation uses Python/CuPy (GPU).
      </div>
    </div>
  );
}

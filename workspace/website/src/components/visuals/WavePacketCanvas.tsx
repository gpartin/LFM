/*
 * WavePacketCanvas — real-time wave packet simulation (preview) using LFM-style Verlet integration
 * Notes:
 * - This is a browser-based preview for interactivity. Official validation uses Python/CuPy (GPU).
 * - Discrete 1D implementation with periodic boundaries, chi(x) support, and Verlet time-stepping.
 */

import React, { useRef, useEffect, useState } from 'react';
import type { ExperimentDefinition } from '@/data/experiments';

interface Props {
  experiment: ExperimentDefinition;
  isRunning?: boolean;
  onMetrics?: (m: { energy?: number; energyDriftPct?: number; time?: number }) => void;
}

export default function WavePacketCanvas({ experiment, isRunning = false, onMetrics }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const stepsPerFrameRef = useRef<number>(3);
  const [overlayText, setOverlayText] = useState<string>('');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Parameters (fallbacks chosen for stability and visibility)
    const N = Math.max(64, Math.min(512, experiment.initialConditions.latticeSize || 128));
    const dx = experiment.initialConditions.dx || 0.02;
    const dtInput = experiment.initialConditions.dt || 2e-4;
    const chiInit = experiment.initialConditions.chi;

    // Courant stability clamp for 1D c=1: dt <= dx
    const dt = Math.min(dtInput, dx * 0.9);
    const invDx2 = 1.0 / (dx * dx);
    const dt2 = dt * dt;

    // Chi field (1D slice). If array [a,b], create linear gradient; if number, constant; else zero
    const chi = new Float64Array(N);
    if (Array.isArray(chiInit) && chiInit.length >= 2) {
      const a = Number(chiInit[0]) || 0;
      const b = Number(chiInit[1]) || 0;
      for (let i = 0; i < N; i++) chi[i] = a + (b - a) * (i / (N - 1));
    } else if (typeof chiInit === 'number') {
      chi.fill(chiInit);
    } else {
      chi.fill(0);
    }
    const chi2 = new Float64Array(N);
    for (let i = 0; i < N; i++) chi2[i] = chi[i] * chi[i];

    // Field arrays (Double precision for better parity with Python)
    let E = new Float64Array(N);
    let Eprev = new Float64Array(N);
    let Enext = new Float64Array(N);

    // Initialize with a Gaussian packet centered left, small k to move right
    const amp = 1.0;
    const width = (experiment.initialConditions as any)?.wavePacket?.width || 6.0;
    const k = (experiment.initialConditions as any)?.wavePacket?.k?.[0] || 0.4; // radians per unit length
    const x0Index = Math.floor(N * 0.3);
    for (let i = 0; i < N; i++) {
      const x = (i - x0Index) * dx;
      const gauss = Math.exp(-0.5 * (x * x) / (width * width));
      E[i] = amp * gauss * Math.cos(k * x);
    }
    // Zero initial velocity → Eprev = E (leapfrog start)
    Eprev.set(E);

    // Energy helpers
    const energy = (E_curr: Float64Array, E_prev: Float64Array): number => {
      // Approximate kinetic via central velocity, gradient via central difference
      let sum = 0;
      const inv2dx = 1.0 / (2 * dx);
      for (let i = 0; i < N; i++) {
        const im1 = (i + N - 1) % N;
        const ip1 = (i + 1) % N;
        const v = (E_curr[i] - E_prev[i]) / dt;
        const grad = (E_curr[ip1] - E_curr[im1]) * inv2dx;
        const pot = chi2[i] * E_curr[i] * E_curr[i];
        sum += 0.5 * (v * v + grad * grad + pot);
      }
      return sum * dx; // integrate over length
    };

    const tRef = { t: 0 } as { t: number };
    const E0 = energy(E, Eprev) || 1e-16;

    // Single time step (Verlet)
    const step = () => {
      // Periodic neighbors: i-1 and i+1
      const last = N - 1;
      // Handle i = 0 separately for performance (unrolled edges)
      let im1 = last, ip1 = 1, i = 0;
      let lap = (E[im1] - 2 * E[i] + E[ip1]) * invDx2;
      Enext[i] = 2 * E[i] - Eprev[i] + dt2 * (lap - chi2[i] * E[i]);
      // Main interior
      for (i = 1; i < last; i++) {
        lap = (E[i - 1] - 2 * E[i] + E[i + 1]) * invDx2;
        Enext[i] = 2 * E[i] - Eprev[i] + dt2 * (lap - chi2[i] * E[i]);
      }
      // i = last
      im1 = last - 1; ip1 = 0; i = last;
      lap = (E[im1] - 2 * E[i] + E[ip1]) * invDx2;
      Enext[i] = 2 * E[i] - Eprev[i] + dt2 * (lap - chi2[i] * E[i]);

      // Rotate buffers (no allocations)
      const tmp = Eprev; Eprev = E; E = Enext; Enext = tmp;
    };

    // Render current field
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Axis transform: x in [0,N], y in [-1,1]
      ctx.save();
      ctx.translate(0, canvas.height / 2);
      ctx.scale(canvas.width / N, -(canvas.height / 2.2));

      // Baseline
      ctx.strokeStyle = 'rgba(148,163,184,0.35)';
      ctx.lineWidth = 1 / (canvas.width / N);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(N, 0);
      ctx.stroke();

      // Field as continuous polyline
      ctx.beginPath();
      ctx.moveTo(0, E[0]);
      for (let i = 1; i < N; i++) ctx.lineTo(i, E[i]);
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 2 / (canvas.width / N);
      ctx.stroke();

      // Optional: Chi background gradient (visualize medium)
      ctx.globalAlpha = 0.12;
      for (let i = 0; i < N; i += Math.max(1, Math.floor(N / 128))) {
        const v = Math.min(1, Math.max(0, chi[i]));
        ctx.fillStyle = `hsl(${240 - v * 180}, 70%, ${30 + v * 40}%)`;
        ctx.fillRect(i, -1.0, 1, 2.0);
      }
      ctx.globalAlpha = 1.0;

      ctx.restore();

      // Overlay: preview disclaimer
      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace';
      ctx.fillText('Preview visualization — official validation uses Python/CuPy (GPU)', 10, 18);

      // Overlay: live metrics (energy drift)
      if (overlayText) {
        ctx.fillStyle = '#cbd5e1';
        ctx.fillText(overlayText, 10, 36);
      }
    };

    // Static render when paused
    const drawStatic = () => {
      render();
    };

    // Animation loop
    const animate = () => {
      const steps = stepsPerFrameRef.current;
      for (let s = 0; s < steps; s++) step();
      tRef.t += steps * dt;

      // Update metrics occasionally (every ~6 frames to reduce cost)
      if (Math.random() < 0.2) {
        const Et = energy(E, Eprev);
        const driftPct = ((Et - E0) / E0) * 100;
        const text = `Energy drift: ${driftPct.toFixed(3)}%`;
        setOverlayText(text);
        onMetrics?.({ energy: Et, energyDriftPct: driftPct, time: tRef.t });
      }
      render();
      animationRef.current = requestAnimationFrame(animate);
    };

    if (isRunning) {
      animate();
    } else {
      drawStatic();
    }

    return () => {
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    };
  }, [experiment, isRunning, onMetrics]);

  return (
    <div className="w-full h-full flex items-center justify-center bg-slate-950 rounded">
      <canvas
        ref={canvasRef}
        width={600}
        height={260}
        className="w-full h-auto max-h-full object-contain"
        aria-label="Wave packet simulation canvas (preview)"
      />
    </div>
  );
}

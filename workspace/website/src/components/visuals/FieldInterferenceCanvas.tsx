/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

'use client';

import { useEffect, useRef, useState } from 'react';

interface DetectionEvent {
  y: number;
  timestamp: number;
}

interface Props {
  simulation: React.MutableRefObject<any>;
  isRunning: boolean;
  showGrid?: boolean;
  showWave?: boolean;
  showBarrier?: boolean;
  showMetricsOverlay?: boolean;
  fringeSpacing?: string;
  visibility?: string;
  slitIntensityRatio?: string;
  updateInterval?: number;
}

export default function FieldInterferenceCanvas({
  simulation,
  isRunning,
  showGrid = true,
  showWave = true,
  showBarrier = true,
  showMetricsOverlay = true,
  fringeSpacing = '—',
  visibility = '—',
  slitIntensityRatio = '—',
  updateInterval = 3,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameCounterRef = useRef(0);
  const [showMode, setShowMode] = useState<'wave' | 'particle'>('particle');
  const [detections, setDetections] = useState<DetectionEvent[]>([]);
  const detectionAccumulatorRef = useRef<number[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    let animationId: number;

    const render = () => {
      const sim = simulation.current;
      if (!sim || !ctx) return;

      frameCounterRef.current = (frameCounterRef.current + 1) % updateInterval;
      if (frameCounterRef.current !== 0) {
        if (isRunning) animationId = requestAnimationFrame(render);
        return;
      }

      const width = rect.width;
      const height = rect.height;
      ctx.fillStyle = '#0a0a1a';
      ctx.fillRect(0, 0, width, height);

      // Optional grid
      if (showGrid) {
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 8; i++) {
          const y = (i / 8) * height;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }
      }

      // Get field data
      if (showWave && sim.lattice) {
        const lattice = sim.lattice;
        const N = sim.params?.latticeSize ?? 128;
        
        let field: Float32Array | null = null;
        if (typeof lattice.getField === 'function') {
          field = lattice.getField();
        }
        
        if (!field) {
          ctx.fillStyle = '#6b7280';
          ctx.font = '14px monospace';
          ctx.textAlign = 'center';
          ctx.fillText('Reading field data…', width / 2, height / 2);
          if (isRunning) animationId = requestAnimationFrame(render);
          return;
        }

        const z = Math.floor(N / 2);
        const screenX = Math.floor(N * 0.85);

        if (showMode === 'wave') {
          // Show propagating wave function
          const cellWidth = width / N;
          const cellHeight = height / N;

          let maxVal = 0;
          for (let i = 0; i < field.length; i++) {
            const val = Math.abs(field[i]);
            if (val > maxVal) maxVal = val;
          }
          if (maxVal === 0) maxVal = 1;

          for (let iy = 0; iy < N; iy++) {
            for (let ix = 0; ix < N; ix++) {
              const idx = z * N * N + iy * N + ix;
              const val = Math.abs(field[idx]) / maxVal;
              if (val > 0.05) {
                const alpha = Math.min(0.7, val * 0.9);
                ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
                ctx.fillRect(ix * cellWidth, iy * cellHeight, cellWidth + 1, cellHeight + 1);
              }
            }
          }
        } else {
          // PARTICLE DETECTION MODE - Show quantum measurement collapse!
          
          // Initialize accumulator
          if (detectionAccumulatorRef.current.length !== N) {
            detectionAccumulatorRef.current = new Array(N).fill(0);
          }

          // Calculate probability distribution at detection screen
          const probDist: number[] = new Array(N).fill(0);
          let totalProb = 0;
          for (let iy = 0; iy < N; iy++) {
            const idx = z * N * N + iy * N + screenX;
            const prob = field[idx] * field[idx];
            probDist[iy] = prob;
            totalProb += prob;
          }

          // Generate detection events (quantum collapse!)
          if (totalProb > 0.01 && isRunning) {
            const numDetections = Math.random() < 0.4 ? (Math.random() < 0.6 ? 1 : 2) : 0;
            const newDetections: DetectionEvent[] = [];
            
            for (let d = 0; d < numDetections; d++) {
              const r = Math.random() * totalProb;
              let cumulative = 0;
              for (let iy = 0; iy < N; iy++) {
                cumulative += probDist[iy];
                if (cumulative >= r) {
                  detectionAccumulatorRef.current[iy]++;
                  newDetections.push({
                    y: iy / N,
                    timestamp: Date.now()
                  });
                  break;
                }
              }
            }
            
            if (newDetections.length > 0) {
              setDetections(prev => [...prev, ...newDetections].slice(-300));
            }
          }

          // Draw faint wave function (shows what's about to collapse)
          const cellWidth = width / N;
          const cellHeight = height / N;
          let maxVal = 0;
          for (let i = 0; i < field.length; i++) {
            const val = Math.abs(field[i]);
            if (val > maxVal) maxVal = val;
          }
          if (maxVal > 0) {
            for (let iy = 0; iy < N; iy++) {
              for (let ix = 0; ix < N; ix++) {
                const idx = z * N * N + iy * N + ix;
                const val = Math.abs(field[idx]) / maxVal;
                if (val > 0.1) {
                  const alpha = Math.min(0.15, val * 0.2);
                  ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
                  ctx.fillRect(ix * cellWidth, iy * cellHeight, cellWidth + 1, cellHeight + 1);
                }
              }
            }
          }

          // Draw detection screen line
          const screenPx = (screenX / N) * width;
          ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)';
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(screenPx, 0);
          ctx.lineTo(screenPx, height);
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw accumulated histogram (THE INTERFERENCE PATTERN EMERGES!)
          const maxDetections = Math.max(...detectionAccumulatorRef.current, 1);
          const barWidth = width * 0.12;
          ctx.fillStyle = 'rgba(168, 85, 247, 0.7)';
          ctx.strokeStyle = 'rgba(168, 85, 247, 1)';
          ctx.lineWidth = 1;
          
          for (let iy = 0; iy < N; iy++) {
            const count = detectionAccumulatorRef.current[iy];
            if (count > 0) {
              const barHeight = (count / maxDetections) * barWidth;
              const y = (iy / N) * height;
              const cellH = height / N;
              ctx.fillRect(width - barHeight, y, barHeight, cellH);
              ctx.strokeRect(width - barHeight, y, barHeight, cellH);
            }
          }

          // Draw recent detection flashes (the "click" of particle detection)
          const now = Date.now();
          detections.forEach(det => {
            const age = now - det.timestamp;
            if (age < 800) {
              const fade = 1 - (age / 800);
              const size = 2 + fade * 3;
              ctx.globalAlpha = fade * 0.9;
              ctx.fillStyle = '#fff';
              ctx.beginPath();
              ctx.arc(screenPx, det.y * height, size, 0, Math.PI * 2);
              ctx.fill();
              ctx.globalAlpha = 1.0;
            }
          });

          // Show detection count
          const totalDetections = detectionAccumulatorRef.current.reduce((a, b) => a + b, 0);
          ctx.fillStyle = '#a78bfa';
          ctx.font = 'bold 16px monospace';
          ctx.textAlign = 'left';
          ctx.fillText(`⚛️ ${totalDetections} detections`, 10, height - 15);
        }
      }

      // Draw aperture marker
      if (showBarrier && typeof sim.params?.apertureX === 'number' && sim.slitCentersY) {
        const N = sim.params?.latticeSize ?? 128;
        const toY = (idx: number) => (idx / (N - 1)) * height;
        const slitW = sim.params?.slitWidth ?? 6;
        const [s1, s2] = sim.slitCentersY as [number, number];
        const y1a = toY(Math.max(0, s1 - Math.floor(slitW / 2)));
        const y1b = toY(Math.min(N - 1, s1 + Math.floor(slitW / 2)));
        const y2a = toY(Math.max(0, s2 - Math.floor(slitW / 2)));
        const y2b = toY(Math.min(N - 1, s2 + Math.floor(slitW / 2)));
        
        const barX = 8;
        const barW = 10;
        ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.fillRect(barX, 0, barW, height);
        ctx.clearRect(barX, y1a, barW, y1b - y1a);
        ctx.clearRect(barX, y2a, barW, y2b - y2a);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.strokeRect(barX, 0, barW, height);
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#ef4444';
        ctx.textAlign = 'left';
        ctx.fillText('Slits', barX + barW + 6, 14);
      }

      if (isRunning) animationId = requestAnimationFrame(render);
    };

    render();
    return () => { if (animationId) cancelAnimationFrame(animationId); };
  }, [simulation, isRunning, showGrid, showWave, showBarrier, showMetricsOverlay, fringeSpacing, visibility, slitIntensityRatio, updateInterval, showMode, detections]);

  return (
    <div className="relative w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full" style={{ background: '#0a0a1a' }} />
      <div className="absolute top-2 left-2 flex gap-2 z-10">
        <button
          onClick={() => setShowMode('wave')}
          className={`px-3 py-1.5 rounded text-sm font-semibold transition shadow-lg ${
            showMode === 'wave' 
              ? 'bg-blue-500 text-white shadow-blue-500/50' 
              : 'bg-gray-800/80 text-gray-300 hover:bg-gray-700'
          }`}
        >
          🌊 Wave
        </button>
        <button
          onClick={() => setShowMode('particle')}
          className={`px-3 py-1.5 rounded text-sm font-semibold transition shadow-lg ${
            showMode === 'particle' 
              ? 'bg-purple-500 text-white shadow-purple-500/50' 
              : 'bg-gray-800/80 text-gray-300 hover:bg-gray-700'
          }`}
        >
          ⚛️ Measurement
        </button>
        {showMode === 'particle' && (
          <button
            onClick={() => {
              const sim = simulation.current;
              const N = sim?.params?.latticeSize ?? 128;
              detectionAccumulatorRef.current = new Array(N).fill(0);
              setDetections([]);
            }}
            className="px-3 py-1.5 rounded text-sm font-semibold bg-red-600 text-white hover:bg-red-700 transition shadow-lg shadow-red-600/50"
          >
            🔄 Clear
          </button>
        )}
      </div>
    </div>
  );
}

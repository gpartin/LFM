/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * FieldSliceCanvas - 1D quantum field amplitude visualization
 * 
 * Renders wave packet amplitude along x-axis for quantum tunneling.
 * Updates at controlled cadence to avoid heavy GPU readbacks.
 */

'use client';

import { useEffect, useRef } from 'react';

interface FieldSliceCanvasProps {
  simulation: React.MutableRefObject<any>;
  isRunning: boolean;
  showGrid?: boolean;
  showWave?: boolean;
  showBarrier?: boolean;
  showTransmissionOverlay?: boolean;
  transmissionValue?: string;
  reflectionValue?: string;
  conservationValue?: string;
  updateInterval?: number; // frames between redraws (default 3)
}

export default function FieldSliceCanvas({
  simulation,
  isRunning,
  showGrid = true,
  showWave = true,
  showBarrier = true,
  showTransmissionOverlay = false,
  transmissionValue = '—',
  reflectionValue = '—',
  conservationValue = '—',
  updateInterval = 3,
}: FieldSliceCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameCounterRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    let animationId: number;

    const render = async () => {
      const sim = simulation.current;
      if (!sim || !ctx) return;

      // Throttle readbacks
      frameCounterRef.current = (frameCounterRef.current + 1) % updateInterval;
      if (frameCounterRef.current !== 0) {
        if (isRunning) animationId = requestAnimationFrame(render);
        return;
      }

      const width = rect.width;
      const height = rect.height;

      // Clear canvas
      ctx.fillStyle = '#0a0a1a';
      ctx.fillRect(0, 0, width, height);

      // Try to get field data (CPU path or cached GPU)
      let field: Float32Array | null = null;
      try {
        if (sim.lattice) {
          if (typeof sim.lattice.getField === 'function') {
            field = sim.lattice.getField();
          } else if (typeof sim.lattice.readEnergyField === 'function') {
            // GPU readback (async, so we'll skip if not instant)
            // For now, only update when available; keeps frame rate smooth
            field = await Promise.race([
              sim.lattice.readEnergyField(),
              new Promise<null>((resolve) => setTimeout(() => resolve(null), 1)),
            ]);
          }
        }
      } catch (e) {
        console.warn('[FieldSliceCanvas] Error reading field:', e);
      }

      if (!field) {
        // Draw "loading" placeholder
        ctx.fillStyle = '#6b7280';
        ctx.font = '14px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Reading field...', width / 2, height / 2);
        if (isRunning) animationId = requestAnimationFrame(render);
        return;
      }

      // Extract lattice size and projection
      const N = Math.round(Math.cbrt(field.length));
      const projection = new Float32Array(N);

      // Project to 1D: sum |E|² over y,z for each x
      let ptr = 0;
      for (let iz = 0; iz < N; iz++) {
        for (let iy = 0; iy < N; iy++) {
          for (let ix = 0; ix < N; ix++) {
            const v = field[ptr++];
            projection[ix] += v * v;
          }
        }
      }

      // Normalize projection for display
      let maxVal = 0;
      for (let i = 0; i < N; i++) {
        if (projection[i] > maxVal) maxVal = projection[i];
      }
      if (maxVal === 0) maxVal = 1; // avoid division by zero

      // Draw grid if enabled
      if (showGrid) {
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 1;
        // Vertical grid lines
        for (let i = 0; i <= 10; i++) {
          const x = (i / 10) * width;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }
        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
          const y = (i / 5) * height;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }
      }

      if (showWave) {
        // Draw wave packet as filled area (more intuitive than line)
        ctx.fillStyle = 'rgba(139, 92, 246, 0.3)'; // purple with transparency
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, height);

        for (let ix = 0; ix < N; ix++) {
          const x = (ix / (N - 1)) * width;
          const normalized = projection[ix] / maxVal;
          const y = height - normalized * height * 0.8; // leave 20% margin at top
          ctx.lineTo(x, y);
        }
        
        ctx.lineTo(width, height);
        ctx.closePath();
        ctx.fill();
        
        // Draw outline on top
        ctx.beginPath();
        for (let ix = 0; ix < N; ix++) {
          const x = (ix / (N - 1)) * width;
          const normalized = projection[ix] / maxVal;
          const y = height - normalized * height * 0.8;
          
          if (ix === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

  // Draw barrier region - make it prominent
  if (showBarrier && sim.barrierCenterIndex !== undefined && sim.params?.barrierWidth) {
        const barrierStart = Math.max(0, sim.barrierCenterIndex - sim.params.barrierWidth / 2);
        const barrierEnd = Math.min(N, sim.barrierCenterIndex + sim.params.barrierWidth / 2);
        
        const xStart = (barrierStart / N) * width;
        const xEnd = (barrierEnd / N) * width;
        
        // Draw barrier as vertical gradient (wall-like)
        const gradient = ctx.createLinearGradient(xStart, 0, xEnd, 0);
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.3)');
        gradient.addColorStop(0.5, 'rgba(239, 68, 68, 0.5)');
        gradient.addColorStop(1, 'rgba(239, 68, 68, 0.3)');
        ctx.fillStyle = gradient;
        ctx.fillRect(xStart, 0, xEnd - xStart, height);
        
        // Draw barrier edges (walls)
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xStart, 0);
        ctx.lineTo(xStart, height);
        ctx.moveTo(xEnd, 0);
        ctx.lineTo(xEnd, height);
        ctx.stroke();
        
        // Barrier label with icon
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('⚡ BARRIER ⚡', (xStart + xEnd) / 2, 25);
        ctx.font = '10px sans-serif';
        ctx.fillText('(chi field potential)', (xStart + xEnd) / 2, 38);
      }

      // Draw helpful labels
      ctx.fillStyle = '#9ca3af';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Wave Packet Amplitude', 10, 20);
      
      // Draw direction arrow
      ctx.fillStyle = '#60a5fa';
      ctx.font = '14px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('→ Packet traveling right →', 10, height - 10);
      
      // Add legend for what they're seeing
      ctx.font = '11px sans-serif';
      ctx.fillStyle = '#a78bfa';
      ctx.textAlign = 'right';
      ctx.fillText('Purple = Quantum wave', width - 10, height - 30);
      ctx.fillStyle = '#ef4444';
      ctx.fillText('Red = Barrier', width - 10, height - 15);

      // Transmission/Reflection overlay (optional)
      if (showTransmissionOverlay) {
        const panelWidth = 160;
        const panelHeight = 70;
        const x = width - panelWidth - 12;
        const y = 12;
        ctx.fillStyle = 'rgba(17,24,39,0.85)';
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, y, panelWidth, panelHeight, 6);
        ctx.fill();
        ctx.stroke();
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#e0e7ff';
        ctx.fillText('Quantum Metrics', x + 10, y + 18);
        ctx.font = '11px monospace';
        ctx.fillStyle = '#93c5fd';
        ctx.fillText(`T: ${transmissionValue}`, x + 10, y + 35);
        ctx.fillStyle = '#fca5a5';
        ctx.fillText(`R: ${reflectionValue}`, x + 10, y + 49);
        ctx.fillStyle = '#fde68a';
        ctx.fillText(`T+R: ${conservationValue}`, x + 10, y + 63);
      }

      if (isRunning) animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [simulation, isRunning, showGrid, showWave, showBarrier, showTransmissionOverlay, updateInterval]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ background: '#0a0a1a' }}
    />
  );
}

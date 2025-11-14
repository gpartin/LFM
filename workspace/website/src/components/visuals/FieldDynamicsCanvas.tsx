/*
 * FieldDynamicsCanvas — field dynamics simulation for chi evolution and EM coupling
 */

import React, { useRef, useEffect } from 'react';
import type { ExperimentDefinition } from '@/data/experiments';

interface Props {
  experiment: ExperimentDefinition;
  isRunning?: boolean;
  onMetrics?: (m: { energy?: number; energyDriftPct?: number; time?: number }) => void; // unused here
  speedFactor?: number;
  resetCounter?: number;
}

export default function FieldDynamicsCanvas({ experiment, isRunning, speedFactor = 1.0, resetCounter }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const timeRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Safety check for experiment data
    if (!experiment || !experiment.initialConditions) {
      console.warn('FieldDynamicsCanvas: Missing experiment data');
      return;
    }

    const N = experiment.initialConditions.latticeSize || 64;
    const chiRange = experiment.initialConditions.chi || [0, 0.3];
    const chiMin = Array.isArray(chiRange) ? chiRange[0] : 0;
    const chiMax = Array.isArray(chiRange) ? chiRange[1] : 0.3;
    
    // Animation loop
    const animate = () => {
      if (!isRunning) {
        // Draw static state when paused
        drawFrame(ctx, canvas.width, canvas.height, N, chiMin, chiMax, timeRef.current);
        return;
      }
      
      timeRef.current += 0.05 * speedFactor; // Advance time scaled by speedFactor
      drawFrame(ctx, canvas.width, canvas.height, N, chiMin, chiMax, timeRef.current);
      animationRef.current = requestAnimationFrame(animate);
    };

    // Start/stop animation based on isRunning
    if (isRunning) {
      animate();
    } else {
      // Draw initial frame when not running
      drawFrame(ctx, canvas.width, canvas.height, N, chiMin, chiMax, timeRef.current);
    }

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [experiment, isRunning, speedFactor, resetCounter]);

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={480}
      className="w-full h-full rounded bg-slate-950 border border-slate-800"
      aria-label="Field dynamics simulation canvas"
    />
  );
}

// Draw a single frame showing chi-field evolution
function drawFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  N: number,
  chiMin: number,
  chiMax: number,
  time: number
) {
  ctx.clearRect(0, 0, width, height);
  const cellSize = width / N;
  
  // Simulate chi-field gradient with time evolution
  // For COUP-01: gradient from chiMin to chiMax with wave propagation
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = (i - N / 2) / (N / 4);
      const y = (j - N / 2) / (N / 4);
      const r = Math.sqrt(x * x + y * y);
      
      // Chi gradient: varies from chiMin to chiMax
      const chiGradient = chiMin + (chiMax - chiMin) * (i / N);
      
      // Add wave pattern that propagates
      const wave = 0.1 * Math.sin(r * 2 - time) * Math.cos(x - time * 0.5);
      
      // Combined field value
      const val = Math.max(0, Math.min(1, (chiGradient + wave) / (chiMax * 1.2)));
      
      // Color mapping: blue (low chi) to red (high chi)
      const hue = 240 - (val * 180); // 240 = blue, 60 = red
      const saturation = 70;
      const lightness = 30 + val * 40;
      
      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
    }
  }
  
  // Draw grid overlay for lattice visualization
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= N; i += 4) {
    const x = i * cellSize;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    
    const y = i * cellSize;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

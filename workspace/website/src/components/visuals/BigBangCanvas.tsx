/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import React, { useEffect, useRef } from 'react';

export interface BigBangParams {
  centralEnergy: number;    // Initial energy at singularity
  explosionSpeed: number;   // Radial expansion velocity
  concentration: number;    // Initial energy spread (smaller = tighter)
  viscosity: number;        // Energy damping/dissipation
  showEnergy: boolean;      // Render energy density heatmap
  showVelocity: boolean;    // Render velocity field arrows
}

interface Props {
  isRunning: boolean;
  params: BigBangParams;
  onReset?: () => void;
  className?: string;
  gridSize?: number;          // Lattice size (square grid)
  edgeAbsorption?: number;    // 0..0.2, boundary damping strength
  edgeWidth?: number;         // boundary sponge thickness (cells)
}

/**
 * BigBangCanvas — Radial energy explosion from central singularity.
 *
 * Physics:
 * - E(r,t=0) = E₀ exp(-r²/σ²) at center
 * - ∂E/∂t = -∇·(vE) + diffusion
 * - v initialized radially outward with speed
 * - Energy spreads, diffuses, and conserves (approximately)
 */
export default function BigBangCanvas({ isRunning, params, onReset, className, gridSize, edgeAbsorption, edgeWidth }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const tRef = useRef<number>(0);
  const ERef = useRef<Float32Array | null>(null);
  const vxRef = useRef<Float32Array | null>(null);
  const vyRef = useRef<Float32Array | null>(null);
  const tmpERef = useRef<Float32Array | null>(null);

  const NX = Math.max(32, Math.floor(gridSize ?? 128));
  const NY = NX;
  const dx = 1.0;

  // Initialize energy and velocity fields
  const initializeFields = () => {
    if (!ERef.current || !vxRef.current || !vyRef.current) return;
    
    const E0 = params.centralEnergy;
    const sigma = params.concentration;
    const v0 = params.explosionSpeed;
    
    const cx = NX / 2;
    const cy = NY / 2;
    
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const idx = j * NX + i;
        const rx = (i - cx) * dx;
        const ry = (j - cy) * dx;
        const r = Math.sqrt(rx * rx + ry * ry);
        
        // Central energy Gaussian
        ERef.current[idx] = E0 * Math.exp(-(r * r) / (sigma * sigma));
        
        // Radial velocity (normalized)
        const rNorm = Math.max(r, 0.1);
        vxRef.current[idx] = v0 * (rx / rNorm);
        vyRef.current[idx] = v0 * (ry / rNorm);
      }
    }
    
    tRef.current = 0;
  };

  // Allocate arrays (reallocate if grid size changes)
  useEffect(() => {
    ERef.current = new Float32Array(NX * NY);
    vxRef.current = new Float32Array(NX * NY);
    vyRef.current = new Float32Array(NX * NY);
    tmpERef.current = new Float32Array(NX * NY);
    initializeFields();
    return () => {
      ERef.current = null;
      vxRef.current = null;
      vyRef.current = null;
      tmpERef.current = null;
    };
  }, [NX, NY]);

  // Reinitialize on parameter change
  useEffect(() => {
    initializeFields();
  }, [params.centralEnergy, params.explosionSpeed, params.concentration, NX, NY]);

  // Note: Do not invoke onReset from inside the canvas on param changes.
  // The parent controls reset; calling it here can create feedback loops.

  // Rendering and physics loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

  const image = ctx.createImageData(NX, NY);
  // Use an offscreen canvas for scaling to avoid self-draw artifacts
  const offscreen = document.createElement('canvas');
  offscreen.width = NX;
  offscreen.height = NY;
  const offCtx = offscreen.getContext('2d');

    const step = () => {
      if (!canvas || !ctx || !ERef.current || !vxRef.current || !vyRef.current || !tmpERef.current) return;

      const dt = 0.016;
      tRef.current += dt;

      const E = ERef.current;
      const vx = vxRef.current;
      const vy = vyRef.current;
      const tmpE = tmpERef.current;
      const visc = params.viscosity * 0.5;

      // Advection (conservative form): ∂E/∂t = -∇·(vE)
      for (let j = 1; j < NY - 1; j++) {
        for (let i = 1; i < NX - 1; i++) {
          const idx = j * NX + i;
          const E0 = E[idx];
          const vx0 = vx[idx];
          const vy0 = vy[idx];
          // Face fluxes (cell-centered velocity approximation)
          const fluxX = (vx0 * E0 - vx[idx - 1] * E[idx - 1]) / dx;
          const fluxY = (vy0 * E0 - vy[idx - NX] * E[idx - NX]) / dx;
          
          // Diffusion for stability
          const lap = (E[idx - 1] + E[idx + 1] + E[idx - NX] + E[idx + NX] - 4 * E0) / (dx * dx);

          tmpE[idx] = E0 - dt * (fluxX + fluxY) + visc * dt * lap;
          tmpE[idx] = Math.max(0, tmpE[idx]); // energy non-negative
        }
      }
      
      // Copy back
      for (let j = 1; j < NY - 1; j++) {
        for (let i = 1; i < NX - 1; i++) {
          const idx = j * NX + i;
          E[idx] = tmpE[idx];
        }
      }

      // Boundary sponge (absorbing edges) to avoid full-domain saturation
      const spongeWidth = Math.max(1, Math.floor(edgeWidth ?? 8));
      const absorb = Math.max(0, Math.min(0.5, edgeAbsorption ?? 0));
      if (absorb > 0) {
        for (let j = 0; j < NY; j++) {
          for (let i = 0; i < NX; i++) {
            const idx = j * NX + i;
            const d = Math.min(i, j, NX - 1 - i, NY - 1 - j);
            if (d < spongeWidth) {
              const w = (spongeWidth - d) / spongeWidth; // 0..1
              const damp = 1 - absorb * w;
              E[idx] *= damp;
            }
          }
        }
      }

      // Render energy heatmap
      if (params.showEnergy) {
        // Dynamic range (min/max) with gamma for contrast
        let Emax = -Infinity;
        let Emin = Infinity;
        for (let idx = 0; idx < NX * NY; idx++) {
          const val = E[idx];
          if (val > Emax) Emax = val;
          if (val < Emin) Emin = val;
        }
        const range = Math.max(1e-9, Emax - Emin);
        const gamma = 0.6; // enhance subtle structure
        
        for (let j = 0; j < NY; j++) {
          for (let i = 0; i < NX; i++) {
            const idx = j * NX + i;
            const norm = Math.min(1, Math.max(0, (E[idx] - Emin) / range));
            const val = Math.pow(norm, gamma);
            
            // Hot colormap: black → red → orange → yellow → white
            const p = idx * 4;
            if (val < 0.25) {
              const t = val / 0.25;
              image.data[p + 0] = Math.floor(255 * t);
              image.data[p + 1] = 0;
              image.data[p + 2] = 0;
            } else if (val < 0.5) {
              const t = (val - 0.25) / 0.25;
              image.data[p + 0] = 255;
              image.data[p + 1] = Math.floor(165 * t);
              image.data[p + 2] = 0;
            } else if (val < 0.75) {
              const t = (val - 0.5) / 0.25;
              image.data[p + 0] = 255;
              image.data[p + 1] = 165 + Math.floor(90 * t);
              image.data[p + 2] = Math.floor(255 * t * 0.3);
            } else {
              const t = (val - 0.75) / 0.25;
              image.data[p + 0] = 255;
              image.data[p + 1] = 255;
              image.data[p + 2] = 76 + Math.floor(179 * t);
            }
            image.data[p + 3] = 255;
          }
        }
        
        if (offCtx) {
          offCtx.putImageData(image, 0, 0);
          ctx.imageSmoothingEnabled = false;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
        } else {
          // Fallback to direct draw if offscreen context missing
          ctx.putImageData(image, 0, 0);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(canvas, 0, 0, NX, NY, 0, 0, canvas.width, canvas.height);
        }
      } else {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      // Velocity field overlay (sparse arrows)
      if (params.showVelocity) {
        ctx.save();
        const step = 8;
        const arrowScale = 15;
        ctx.strokeStyle = '#4da4ff';
        ctx.fillStyle = '#4da4ff';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.8;
        
        for (let j = step; j < NY - step; j += step) {
          for (let i = step; i < NX - step; i += step) {
            const idx = j * NX + i;
            const vx0 = vx[idx];
            const vy0 = vy[idx];
            const vmag = Math.sqrt(vx0 * vx0 + vy0 * vy0);
            if (vmag < 0.01) continue;
            
            const x0 = (i / NX) * canvas.width;
            const y0 = (j / NY) * canvas.height;
            const x1 = x0 + (vx0 / vmag) * arrowScale;
            const y1 = y0 + (vy0 / vmag) * arrowScale;
            
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            
            // Arrowhead
            const angle = Math.atan2(y1 - y0, x1 - x0);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1 - 5 * Math.cos(angle - Math.PI / 6), y1 - 5 * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(x1 - 5 * Math.cos(angle + Math.PI / 6), y1 - 5 * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fill();
          }
        }
        ctx.restore();
      }
    };

    const loop = () => {
      if (isRunning) step();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [isRunning, params]);

  // Resize canvas to container size
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width));
      canvas.height = Math.max(240, Math.floor(rect.height));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  return (
    <canvas ref={canvasRef} className={className ?? 'w-full h-[600px] rounded-lg bg-space-dark border border-space-border'} />
  );
}

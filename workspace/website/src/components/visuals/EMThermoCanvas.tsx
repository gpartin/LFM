/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import React, { useEffect, useRef } from 'react';

export interface EMThermoParams {
  amplitude: number;      // EM wave amplitude
  frequency: number;      // cycles per domain width
  absorption: number;     // EM→thermal coupling coefficient
  diffusivity: number;    // heat diffusion coefficient
  showEM: boolean;        // render EM intensity overlay
  showHeat: boolean;      // render temperature heatmap
}

interface Props {
  isRunning: boolean;
  params: EMThermoParams;
  className?: string;
}

/**
 * EMThermoCanvas — Minimal 2D EM→Thermal coupling demo.
 *
 * - EM field: E(x, y, t) = A sin(2π f (x - ct)) · exp(-(y/σ)^2)
 * - Power density ~ E^2; deposit into temperature T with coefficient `absorption`.
 * - T diffuses each step using a 5-point Laplacian.
 */
export default function EMThermoCanvas({ isRunning, params, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const tRef = useRef<number>(0);
  const TRef = useRef<Float32Array | null>(null);
  const tmpRef = useRef<Float32Array | null>(null);

  const NX = 160;
  const NY = 96;

  // Initialize temperature grid once
  useEffect(() => {
    TRef.current = new Float32Array(NX * NY);
    tmpRef.current = new Float32Array(NX * NY);
    return () => {
      TRef.current = null;
      tmpRef.current = null;
    };
  }, []);

  // Reset on parameter change that implies a new scenario
  useEffect(() => {
    if (!TRef.current) return;
    TRef.current.fill(0);
    tRef.current = 0;
  }, [params.absorption, params.diffusivity, params.frequency]);

  // Rendering / simulation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(NX, NY);

    const step = () => {
      if (!canvas || !ctx || !TRef.current || !tmpRef.current) return;

      // Advance time
      const dt = 0.016; // ~60 FPS
      tRef.current += dt;
      const t = tRef.current;

      const A = params.amplitude;
      const f = params.frequency; // cycles per width
      const alpha = params.absorption * 0.5; // visual scaling
      const k = 2 * Math.PI * f;
      const c = 0.6; // phase speed (visual)
      const sigma = 0.4; // beam vertical width (fraction)

      // Deposit EM -> heat and render
      for (let j = 0; j < NY; j++) {
        const v = (j / (NY - 1)) * 2 - 1; // -1..1
        const gauss = Math.exp(-(v * v) / (sigma * sigma));
        for (let i = 0; i < NX; i++) {
          const u = i / (NX - 1);
          const E = A * Math.sin(k * (u - c * t)) * gauss;
          const I = E * E; // intensity ~ E^2
          const idx = j * NX + i;
          TRef.current[idx] += alpha * I * dt;

          // Heatmap color (viridis-ish simple ramp)
          if (params.showHeat) {
            const val = Math.min(1.0, TRef.current[idx]);
            const r = Math.floor(255 * Math.min(1, Math.max(0, -1.5 * (val - 0.9) + 1)));
            const g = Math.floor(255 * Math.min(1, val));
            const b = Math.floor(255 * Math.min(1, 1 - 0.8 * val));
            const p = idx * 4;
            image.data[p + 0] = r;
            image.data[p + 1] = g;
            image.data[p + 2] = b;
            image.data[p + 3] = 255;
          }
        }
      }

      // Diffusion step: T_new = T + D * dt * Laplacian(T)
      if (params.diffusivity > 0) {
        const Ddt = params.diffusivity * dt;
        for (let j = 1; j < NY - 1; j++) {
          for (let i = 1; i < NX - 1; i++) {
            const idx = j * NX + i;
            const T0 = TRef.current[idx];
            const lap =
              TRef.current[idx - 1] +
              TRef.current[idx + 1] +
              TRef.current[idx - NX] +
              TRef.current[idx + NX] -
              4 * T0;
            tmpRef.current[idx] = T0 + Ddt * lap;
          }
        }
        // copy back (keep borders)
        for (let j = 1; j < NY - 1; j++) {
          for (let i = 1; i < NX - 1; i++) {
            const idx = j * NX + i;
            TRef.current[idx] = tmpRef.current[idx];
          }
        }
      }

      // Draw heatmap
      if (params.showHeat) {
        // Nearest-neighbor scale to canvas size
        ctx.putImageData(image, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(canvas, 0, 0, NX, NY, 0, 0, canvas.width, canvas.height);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      // EM overlay (lines)
      if (params.showEM) {
        ctx.save();
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = '#66ccff';
        ctx.lineWidth = 1.5;
        for (let j = 12; j < NY - 12; j += 8) {
          ctx.beginPath();
          for (let i = 0; i < NX; i++) {
            const u = i / (NX - 1);
            const v = (j / (NY - 1)) * 2 - 1;
            const gauss = Math.exp(-(v * v) / (sigma * sigma));
            const E = A * Math.sin(k * (u - c * t)) * gauss;
            const x = (i / (NX - 1)) * canvas.width;
            const y = ((j + E * 10) / (NY - 1)) * canvas.height;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
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

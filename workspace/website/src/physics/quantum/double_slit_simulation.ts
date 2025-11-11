/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * DoubleSlitSimulation — 2‑slit interference on LFM lattice (Phase 1)
 *
 * Minimal, backend-agnostic implementation following QuantumTunnelingSimulation
 * patterns. Uses CPU by default; will utilize WebGPU lattice if provided.
 *
 * Physics sketch:
 *  - Plane-like wave (Gaussian in x, uniform in y/z) launched from left
 *  - Aperture plane at x = apertureX with two rectangular slits (in y),
 *    implemented as a thin χ barrier except where slits open (χ = 0)
 *  - Observation “screen” at x = screenX; compute |E|^2 vs y there
 *  - Metrics: fringe spacing Δy, visibility V = (Imax−Imin)/(Imax+Imin),
 *    slitIntensityRatio measured at the aperture plane per opening
 */

import { LFMLatticeCPU, ParticleState } from '../core/lattice-cpu';
import { LFMLatticeWebGPU } from '../core/lattice-webgpu';

export interface DoubleSlitParams {
  latticeSize: number;
  dx: number;
  dt: number;
  k0: number;               // wave number along +x
  sigma: number;            // Gaussian width along x
  slitSeparation: number;   // center-to-center in grid cells (y)
  slitWidth: number;        // opening width in grid cells (y)
  barrierThickness: number; // thickness in x (cells)
  apertureX: number;        // x-index of aperture plane start
  screenX: number;          // x-index of observation plane
  batchSteps: number;       // steps per animation tick
}

export interface DoubleSlitMetrics {
  energy: string;
  drift: string;
  fringeSpacing: string;       // grid units (Δy)
  visibility: string;          // 0..1 in %
  slitIntensityRatio: string;  // upper/lower ratio
  sampleRate?: string;
}

export class DoubleSlitSimulation {
  private params: DoubleSlitParams;
  private lattice: LFMLatticeCPU | LFMLatticeWebGPU | null = null;
  private initialized = false;
  private device: GPUDevice | null = null;
  private useGPU = false;

  // Cached projections and metrics cadence
  private metricsFrameCounter = 0;
  private metricsUpdateInterval = 3;
  private lastScreenIntensityY: Float32Array | null = null;

  // Geometry caches
  private slitCentersY: [number, number] = [0, 0];

  constructor(params: Partial<DoubleSlitParams> = {}, options?: { device?: GPUDevice }) {
    const N = params.latticeSize ?? 64;
    const apertureX = params.apertureX ?? Math.floor(N * 0.45);
    this.params = {
      latticeSize: N,
      dx: params.dx ?? 0.1,
      dt: params.dt ?? 0.001,
      k0: params.k0 ?? 6.0,
      sigma: params.sigma ?? 8.0,
      slitSeparation: params.slitSeparation ?? Math.floor(N * 0.22),
      slitWidth: params.slitWidth ?? Math.max(3, Math.floor(N * 0.06)),
      barrierThickness: params.barrierThickness ?? 2,
      apertureX,
      screenX: params.screenX ?? Math.min(N - 3, apertureX + Math.floor(N * 0.35)),
      batchSteps: params.batchSteps ?? 6,
    };
    if (options?.device) {
      this.device = options.device;
      this.useGPU = true;
    }
  }

  async initialize(): Promise<void> {
    const N = this.params.latticeSize;
    // Backend selection
    if (this.useGPU && this.device) {
      this.lattice = new LFMLatticeWebGPU(this.device, {
        size: N,
        dx: this.params.dx,
        dt: this.params.dt,
        c: 1.0,
        chiStrength: 0.0,
        sigma: 2.0,
      });
    } else {
      this.lattice = new LFMLatticeCPU({
        size: N,
        dx: this.params.dx,
        dt: this.params.dt,
        c: 1.0,
        chiStrength: 0.0,
        sigma: 2.0,
      });
    }
    await (this.lattice as any).initialize();

    // Build aperture barrier with two slits at x in [apertureX, apertureX+thickness)
    this.applyDoubleSlitBarrier();

    // Seed a right-traveling packet that’s Gaussian in x and uniform in y/z
    const x0 = Math.floor(N * 0.20);
    const sigma2 = this.params.sigma * this.params.sigma;
    const field = new Float32Array(N * N * N);
    for (let ix = 0; ix < N; ix++) {
      const xRel = ix - x0;
      const env = Math.exp(-(xRel * xRel) / (2 * sigma2));
      const phase = Math.cos(this.params.k0 * xRel * this.params.dx);
      const value = env * phase; // uniform over y/z to mimic plane wave
      for (let iz = 0; iz < N; iz++) {
        for (let iy = 0; iy < N; iy++) {
          field[this.idx(ix, iy, iz, N)] = value;
        }
      }
    }
    this.lattice.seedInitialField(field, true);

    // Cache slit centers for metrics drawing
    const yCenter = Math.floor(N / 2);
    const halfSep = Math.floor(this.params.slitSeparation / 2);
    this.slitCentersY = [yCenter - halfSep, yCenter + halfSep];

    this.initialized = true;
  }

  async stepBatch(): Promise<void> {
    if (!this.initialized || !this.lattice) return;
    await (this.lattice as any).stepMany(this.params.batchSteps);
    if (this.lattice instanceof LFMLatticeCPU) {
      this.lattice.computeEnergy();
    } else {
      await (this.lattice as LFMLatticeWebGPU).calculateEnergy();
    }
  }

  updateParams(p: Partial<DoubleSlitParams>) {
    this.params = { ...this.params, ...p };
  }

  /** Projection used by canvas: |E|^2 vs y at screenX (collapsed over z) */
  getScreenIntensityProjection(): Float32Array | null {
    if (!this.initialized || !this.lattice) return null;
    // Only compute on throttled cadence to keep UI smooth
    this.metricsFrameCounter = (this.metricsFrameCounter + 1) % this.metricsUpdateInterval;
    if (this.metricsFrameCounter !== 0 && this.lastScreenIntensityY) {
      return this.lastScreenIntensityY;
    }

    const N = this.params.latticeSize;
    const x = Math.min(Math.max(this.params.screenX, 0), N - 1);
    const out = new Float32Array(N);

    if (this.lattice instanceof LFMLatticeCPU) {
      const f = this.lattice.getField();
      for (let iy = 0; iy < N; iy++) {
        let s = 0;
        for (let iz = 0; iz < N; iz++) {
          const v = f[this.idx(x, iy, iz, N)];
          s += v * v;
        }
        out[iy] = s;
      }
    } else {
      // WebGPU async readback could be added; for now, keep last if not available
      // This branch can be enhanced later with a dedicated compute kernel
      if (this.lastScreenIntensityY) return this.lastScreenIntensityY;
      return null;
    }

    this.lastScreenIntensityY = out;
    return out;
  }

  getMetrics(): DoubleSlitMetrics {
    if (!this.initialized || !this.lattice) {
      return { energy: '—', drift: '—', fringeSpacing: '—', visibility: '—', slitIntensityRatio: '—' };
    }

    const energyMetrics = this.lattice instanceof LFMLatticeCPU
      ? this.lattice.getEnergyMetrics()
      : { current: (this.lattice as any).totalEnergy ?? 0, drift: (this.lattice as LFMLatticeWebGPU).getEnergyDrift(), history: [] };

    const proj = this.getScreenIntensityProjection();
    let spacing = '—';
    let visibility = '—';
    if (proj && proj.length > 8) {
      const peaks = detectPeaks(proj);
      if (peaks.length >= 2) {
        // Use first two central peaks for spacing
        const center = Math.floor(proj.length / 2);
        peaks.sort((a, b) => Math.abs(a - center) - Math.abs(b - center));
        const p1 = peaks[0], p2 = peaks[1];
        spacing = this.formatNumber(Math.abs(p2 - p1));
      }
      // Simple central visibility using local max/min around center window
      const center = Math.floor(proj.length / 2);
      const window = Math.max(4, Math.floor(proj.length * 0.05));
      let Imax = 0, Imin = Number.POSITIVE_INFINITY;
      for (let i = center - window; i <= center + window; i++) {
        const ii = Math.min(Math.max(i, 0), proj.length - 1);
        const v = proj[ii];
        if (v > Imax) Imax = v;
        if (v < Imin) Imin = v;
      }
      const V = (Imax - Imin) / Math.max(Imax + Imin, 1e-12);
      visibility = this.formatPercent(V);
    }

    // Slit intensity ratio at aperture plane
    const ratio = this.estimateSlitIntensityRatio();

    return {
      energy: this.formatNumber(energyMetrics.current),
      drift: this.formatPercent(energyMetrics.drift),
      fringeSpacing: spacing,
      visibility,
      slitIntensityRatio: ratio,
      sampleRate: `1/${this.metricsUpdateInterval} frame cadence`,
    };
  }

  async reset(): Promise<void> {
    if (!this.lattice) return;
    this.initialized = false;
    this.lastScreenIntensityY = null;
    await this.initialize();
  }

  // ----------------------------- internals ------------------------------
  private idx(x: number, y: number, z: number, N: number): number {
    return z * N * N + y * N + x;
  }

  private formatPercent(v: number): string {
    if (!isFinite(v)) return '—';
    return (v * 100).toFixed(2) + '%';
  }

  private formatNumber(v: number): string {
    if (!isFinite(v)) return '—';
    if (Math.abs(v) < 1e-6) return v.toExponential(2);
    return v.toFixed(3);
  }

  private applyDoubleSlitBarrier() {
    if (!this.lattice) return;
    const N = this.params.latticeSize;
    const x0 = this.params.apertureX;
    const thickness = this.params.barrierThickness;
    const slitW = this.params.slitWidth;
    const halfSep = Math.floor(this.params.slitSeparation / 2);
    const yCenter = Math.floor(N / 2);
    const slit1c = yCenter - halfSep;
    const slit2c = yCenter + halfSep;

    // Save for overlays
    this.slitCentersY = [slit1c, slit2c];

    if (this.lattice instanceof LFMLatticeCPU) {
      const particles: ParticleState[] = [];
      this.lattice.initializeChiField(particles); // clears chi
      const chi = this.lattice.getChiField();
      const barrierChi = 2.0; // moderate barrier (opaque except slits)
      for (let dx = 0; dx < thickness; dx++) {
        const xx = Math.min(x0 + dx, N - 1);
        for (let iy = 0; iy < N; iy++) {
          const open1 = Math.abs(iy - slit1c) <= Math.floor(slitW / 2);
          const open2 = Math.abs(iy - slit2c) <= Math.floor(slitW / 2);
          const isOpen = open1 || open2;
          for (let iz = 0; iz < N; iz++) {
            const id = this.idx(xx, iy, iz, N);
            chi[id] = isOpen ? 0.0 : barrierChi;
          }
        }
      }
    } else if (this.lattice instanceof LFMLatticeWebGPU) {
      const chi = new Float32Array(N * N * N);
      const barrierChi = 2.0;
      for (let dx = 0; dx < thickness; dx++) {
        const xx = Math.min(x0 + dx, N - 1);
        for (let iy = 0; iy < N; iy++) {
          const open1 = Math.abs(iy - slit1c) <= Math.floor(slitW / 2);
          const open2 = Math.abs(iy - slit2c) <= Math.floor(slitW / 2);
          const isOpen = open1 || open2;
          for (let iz = 0; iz < N; iz++) {
            chi[this.idx(xx, iy, iz, N)] = isOpen ? 0.0 : barrierChi;
          }
        }
      }
      this.lattice.setChiField(chi);
    }
  }

  private estimateSlitIntensityRatio(): string {
    // Integrate |E|^2 in small windows at the aperture plane per slit
    if (!this.lattice) return '—';
    const N = this.params.latticeSize;
    const x = Math.min(Math.max(this.params.apertureX, 0), N - 1);
    const halfW = Math.max(1, Math.floor(this.params.slitWidth / 2));
    const [y1, y2] = this.slitCentersY;

    let i1 = 0, i2 = 0;
    if (this.lattice instanceof LFMLatticeCPU) {
      const f = this.lattice.getField();
      for (let dy = -halfW; dy <= halfW; dy++) {
        const yy1 = Math.min(Math.max(y1 + dy, 0), N - 1);
        const yy2 = Math.min(Math.max(y2 + dy, 0), N - 1);
        for (let iz = 0; iz < N; iz++) {
          const v1 = f[this.idx(x, yy1, iz, N)];
          const v2 = f[this.idx(x, yy2, iz, N)];
          i1 += v1 * v1;
          i2 += v2 * v2;
        }
      }
    } else {
      return '—';
    }
    const ratio = i2 > 0 ? i1 / i2 : Infinity;
    if (!isFinite(ratio)) return '—';
    return ratio.toFixed(2) + ':1';
  }
}

// -------------------------- helpers ---------------------------

function detectPeaks(data: Float32Array, minSep = 2): number[] {
  const peaks: number[] = [];
  for (let i = 1; i < data.length - 1; i++) {
    if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minSep) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * QuantumTunnelingSimulation (Phase 2 – Initial CPU Implementation)
 *
 * Implements a minimal quantum tunneling scenario using the existing CPU lattice
 * backend (LFMLatticeCPU) to remain backend‑agnostic while we iterate. When the
 * browser has WebGPU, the UI presents an "advanced" mode, but physics metrics
 * are still computed from this CPU lattice until a full GPU init helper is added.
 *
 * Physics Model:
 *   - Start with a 1D Gaussian wave packet embedded in a 3D lattice (variation only in x)
 *   - Packet: E(x) = exp(-(x-x0)^2/(2*sigma^2)) * cos(k0 (x - x0))
 *   - Barrier: chi field augmented by barrierHeight for |x - xBarrier| < barrierWidth/2
 *   - Transmission (T): Fraction of |E|^2 mass to the RIGHT of barrier center
 *   - Reflection  (R): Fraction of |E|^2 mass to the LEFT of packet origin once packet interacts
 *   - Conservation Check: T + R (reported as percentage; target ~100%)
 *
 * Design Notes:
 *   - Uses small lattice (size configurable, default 64) for WebGPU users and 32 for CPU fallback
 *   - For now, always CPU lattice; upgrade path: add GPU seeding + shared field access API
 *   - Metrics updated every stepBatch invocation (batched for performance)
 */

import { LFMLatticeCPU, ParticleState } from '../core/lattice-cpu';
import { LFMLatticeWebGPU } from '../core/lattice-webgpu';

export interface QuantumTunnelingParams {
  latticeSize: number;      // Cubic lattice size (N)
  dx: number;               // Spatial step
  dt: number;               // Time step
  k0: number;               // Packet central wavenumber
  sigma: number;            // Packet spatial width
  barrierHeight: number;    // Added chi strength in barrier region
  barrierWidth: number;     // Width of barrier region in lattice units (grid cells)
  batchSteps: number;       // Steps per animation tick
}

export interface QuantumTunnelingMetrics {
  energy: string;
  drift: string;
  transmission: string;
  reflection: string;
  conservation: string; // T+R
  sampleRate?: string;  // Optional metrics update cadence label
}

export class QuantumTunnelingSimulation {
  private params: QuantumTunnelingParams;
  private lattice: LFMLatticeCPU | LFMLatticeWebGPU | null = null;
  private initialized = false;
  private totalNormInitial = 1; // Normalization factor of initial |E|^2
  private packetCenterIndex = 0;
  private barrierCenterIndex = 0;
  private device: GPUDevice | null = null;
  private useGPU: boolean = false;
  private projectionScratch: Float32Array | null = null; // reuse buffer for x-projection
  private metricsFrameCounter = 0;
  private metricsUpdateInterval = 3; // update metrics every N calls to getMetrics for smoother perf
  private lastLeftSum: number = 0;
  private lastRightSum: number = 0;

  constructor(params: Partial<QuantumTunnelingParams> = {}, options?: { device?: GPUDevice }) {
    // Sensible defaults
    this.params = {
      latticeSize: params.latticeSize ?? 64,
      dx: params.dx ?? 0.1,
      dt: params.dt ?? 0.001,
      k0: params.k0 ?? 4.0,
      sigma: params.sigma ?? 6.0,
      barrierHeight: params.barrierHeight ?? 1.5,
      barrierWidth: params.barrierWidth ?? 6,
      batchSteps: params.batchSteps ?? 6,
    };
    if (options?.device) {
      this.device = options.device;
      this.useGPU = true;
    }
  }

  /** Initialize lattice and seed wave packet + barrier chi field */
  async initialize(): Promise<void> {
    if (this.useGPU && this.device) {
      this.lattice = new LFMLatticeWebGPU(this.device, {
        size: this.params.latticeSize,
        dx: this.params.dx,
        dt: this.params.dt,
        c: 1.0,
        chiStrength: 0.0,
        sigma: 2.0,
      });
    } else {
      this.lattice = new LFMLatticeCPU({
        size: this.params.latticeSize,
        dx: this.params.dx,
        dt: this.params.dt,
        c: 1.0,
        chiStrength: 0.0, // baseline chi; barrier applied separately
        sigma: 2.0,
      });
    }
    await (this.lattice as any).initialize();

    const N = this.params.latticeSize;
    this.packetCenterIndex = Math.floor(N * 0.25); // start left
    this.barrierCenterIndex = Math.floor(N * 0.55); // barrier mid-right

    // Build chi field with barrier (Gaussian + flat barrier region)
    // Build chi field with barrier
    if (this.lattice instanceof LFMLatticeCPU) {
      const particles: ParticleState[] = [];
      this.lattice.initializeChiField(particles); // clears chi
      const chiField = this.lattice.getChiField();
      for (let ix = 0; ix < N; ix++) {
        const inBarrier = Math.abs(ix - this.barrierCenterIndex) <= this.params.barrierWidth / 2;
        if (inBarrier) {
          for (let iz = 0; iz < N; iz++) {
            for (let iy = 0; iy < N; iy++) {
              const idx = this.idx(ix, iy, iz, N);
              chiField[idx] = this.params.barrierHeight;
            }
          }
        }
      }
    } else if (this.lattice instanceof LFMLatticeWebGPU) {
      const N3 = N * N * N;
      const chiField = new Float32Array(N3);
      for (let ix = 0; ix < N; ix++) {
        const inBarrier = Math.abs(ix - this.barrierCenterIndex) <= this.params.barrierWidth / 2;
        if (inBarrier) {
          for (let iz = 0; iz < N; iz++) {
            for (let iy = 0; iy < N; iy++) {
              const idx = this.idx(ix, iy, iz, N);
              chiField[idx] = this.params.barrierHeight;
            }
          }
        }
      }
      this.lattice.setChiField(chiField);
    }

    // Seed wave packet into fieldCurrent/fieldPrevious identical (zero initial time derivative)
    // Seed wave packet into fieldCurrent (and previous for zero dE/dt)
    const N3 = N * N * N;
    const initial = new Float32Array(N3);
    const sigma2 = this.params.sigma * this.params.sigma;
    for (let ix = 0; ix < N; ix++) {
      const xRel = ix - this.packetCenterIndex;
      const envelope = Math.exp(-(xRel * xRel) / (2 * sigma2));
      const phase = Math.cos(this.params.k0 * xRel * this.params.dx);
      const value = envelope * phase;
      for (let iz = 0; iz < N; iz++) {
        for (let iy = 0; iy < N; iy++) {
          const idx = this.idx(ix, iy, iz, N);
          initial[idx] = value;
        }
      }
    }
    if (this.lattice instanceof LFMLatticeCPU) {
      this.lattice.seedInitialField(initial, true);
    } else if (this.lattice instanceof LFMLatticeWebGPU) {
      this.lattice.seedInitialField(initial, true);
    }

  // Compute initial norm (|E|^2) for transmission/reflection normalization
  let norm = 0;
  for (let i = 0; i < initial.length; i++) norm += initial[i] * initial[i];
  this.totalNormInitial = norm !== 0 ? norm : 1;

    this.initialized = true;
  }

  /** Advance simulation by configured batch steps */
  async stepBatch(): Promise<void> {
    if (!this.initialized || !this.lattice) return;
    await (this.lattice as any).stepMany(this.params.batchSteps);
    // Update energy metrics internally (CPU) or compute lazily on demand for GPU
    if (this.lattice instanceof LFMLatticeCPU) {
      this.lattice.computeEnergy();
    } else {
      await (this.lattice as LFMLatticeWebGPU).calculateEnergy();
    }
  }

  /** Update parameter subset (live tweaks) */
  updateParams(p: Partial<QuantumTunnelingParams>): void {
    this.params = { ...this.params, ...p };
    // For simplicity, regeneration on major changes could be added later
  }

  /** Compute T, R, Conservation from current field */
  getMetrics(): QuantumTunnelingMetrics {
    if (!this.initialized || !this.lattice) {
      return { energy: '—', drift: '—', transmission: '—', reflection: '—', conservation: '—' };
    }
    const N = this.params.latticeSize;

    // Throttle heavy operations (projection + optional GPU readback)
    this.metricsFrameCounter = (this.metricsFrameCounter + 1) % this.metricsUpdateInterval;
    let leftSum = 0, rightSum = 0;
    if (this.metricsFrameCounter === 0) {
      // Acquire field
      if (this.lattice instanceof LFMLatticeCPU) {
        const field = this.lattice.getField();
        this.ensureProjectionBuffer(N);
        const proj = this.projectionScratch!;
        proj.fill(0);
        // Collapse over y,z to get 1D x-projection of |E|^2
        let ptr = 0;
        for (let iz = 0; iz < N; iz++) {
          for (let iy = 0; iy < N; iy++) {
            for (let ix = 0; ix < N; ix++) {
              const v = field[ptr];
              proj[ix] += v * v;
              ptr++;
            }
          }
        }
        for (let ix = 0; ix < N; ix++) {
          const amp2 = proj[ix];
            if (ix >= this.barrierCenterIndex) rightSum += amp2; else leftSum += amp2;
        }
      } else if (this.lattice instanceof LFMLatticeWebGPU) {
        // GPU path: readback full field (future optimization: dedicated reduction kernel)
        // NOTE: readEnergyField is async; for now we skip GPU T/R update if not yet available
        // and keep last values (would need refactor to async metrics flow for full correctness).
        // Placeholder: skip recompute to avoid blocking main thread.
        // Could implement a promise-based update hook later.
        // Fall through leaving leftSum/rightSum = 0 (will produce previous ratio retention externally if stored)
      }
      // Store last computed totals
      this.lastLeftSum = leftSum;
      this.lastRightSum = rightSum;
    } else {
      // Reuse last computed sums
      leftSum = this.lastLeftSum;
      rightSum = this.lastRightSum;
    }

    const T = rightSum / this.totalNormInitial;
    const R = leftSum / this.totalNormInitial;
    const conservation = T + R;
    const energyMetrics = this.lattice instanceof LFMLatticeCPU
      ? this.lattice.getEnergyMetrics()
      : { current: (this.lattice as LFMLatticeWebGPU as any).totalEnergy ?? 0, drift: (this.lattice as LFMLatticeWebGPU).getEnergyDrift(), history: [] };
    return {
      energy: this.formatNumber(energyMetrics.current),
      drift: this.formatPercent(energyMetrics.drift),
      transmission: this.formatPercent(T),
      reflection: this.formatPercent(R),
      conservation: this.formatPercent(conservation),
      sampleRate: `1/${this.metricsUpdateInterval} frame cadence`
    };
  }

  /** Reset simulation to initial packet */
  async reset(): Promise<void> {
    if (!this.lattice) return;
    this.initialized = false;
    // Reinitialize entirely to support both CPU/GPU paths cleanly
    await this.initialize();
  }

  /** Utility indexer */
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
    return v.toFixed(4);
  }

  private ensureProjectionBuffer(N: number) {
    if (!this.projectionScratch || this.projectionScratch.length !== N) {
      this.projectionScratch = new Float32Array(N);
    }
  }
}

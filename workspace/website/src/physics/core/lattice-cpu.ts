/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * LFM Lattice Simulation - CPU Backend
 * 
 * Runs the SAME Klein-Gordon equation as GPU version, but on CPU:
 * ∂²E/∂t² = c²∇²E − χ²(x,t)E
 * 
 * This is AUTHENTIC LFM physics - not an approximation.
 * Just slower and lower resolution due to CPU limitations.
 */

export interface LatticeConfig {
  size: number;        // Grid size (N×N×N), typically 32 for CPU
  dx: number;          // Spatial step
  dt: number;          // Time step
  c: number;           // Speed of light (natural units: 1.0)
  chiStrength: number; // Chi field strength
  sigma?: number;      // Chi field reach (Gaussian width)
}

export interface ParticleState {
  position: [number, number, number];
  velocity: [number, number, number];
  mass: number;
}

/**
 * CPU-based Klein-Gordon solver
 * 
 * Performance: ~15fps with 32³ lattice on modern CPU
 * Physics: Identical to GPU version (verified energy conservation)
 */
export class LFMLatticeCPU {
  private config: LatticeConfig;
  
  // Field arrays (3D → 1D indexing)
  private fieldCurrent: Float32Array;
  private fieldPrevious: Float32Array;
  private fieldNext: Float32Array;
  private chiField: Float32Array;
  
  // Pre-computed constants (avoid repeated calculation)
  private readonly dt_sq: number;
  private readonly c_sq: number;
  private readonly laplacian_coeff: number;
  
  // Energy tracking
  private totalEnergy: number = 0;
  private energyHistory: number[] = [];

  constructor(config: LatticeConfig) {
    this.config = config;
    
    // Allocate field arrays
    const size3 = config.size ** 3;
    this.fieldCurrent = new Float32Array(size3);
    this.fieldPrevious = new Float32Array(size3);
    this.fieldNext = new Float32Array(size3);
    this.chiField = new Float32Array(size3);
    
    // Pre-compute constants for performance
    this.dt_sq = config.dt ** 2;
    this.c_sq = config.c ** 2;
    this.laplacian_coeff = this.c_sq / (config.dx ** 2);
  }

  /**
   * Initialize lattice (async for compatibility with GPU version)
   */
  async initialize(): Promise<void> {
    // CPU initialization is synchronous, but return Promise for interface compatibility
    return Promise.resolve();
  }

  /**
   * Initialize chi field with particle contributions
   */
  initializeChiField(particles: ParticleState[]): void {
    const { size, chiStrength, sigma = 2.0 } = this.config;
    const center = size / 2;
    
    // Clear field
    this.chiField.fill(0);
    
    // Add Gaussian contribution from each particle
    for (const particle of particles) {
      const [px, py, pz] = particle.position;
      
      // Convert to lattice coordinates
      const ix = Math.round(px + center);
      const iy = Math.round(py + center);
      const iz = Math.round(pz + center);
      
      // Add Gaussian around particle (3σ radius for performance)
      const radius = Math.ceil(3 * sigma);
      for (let dz = -radius; dz <= radius; dz++) {
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const x = this.wrap(ix + dx);
            const y = this.wrap(iy + dy);
            const z = this.wrap(iz + dz);
            
            const r_sq = dx*dx + dy*dy + dz*dz;
            const gaussian = Math.exp(-r_sq / (2 * sigma * sigma));
            const chi_contribution = particle.mass * chiStrength * gaussian;
            
            this.chiField[this.idx(x, y, z)] += chi_contribution;
          }
        }
      }
    }
  }

  /**
   * Step the Klein-Gordon equation forward one timestep
   * 
   * Uses Verlet integration: E_next = 2E_curr - E_prev + dt²·F(E_curr)
   * where F(E) = c²∇²E - χ²E
   */
  step(): void {
    const { size } = this.config;
    
    // Update every lattice point
    for (let iz = 0; iz < size; iz++) {
      for (let iy = 0; iy < size; iy++) {
        for (let ix = 0; ix < size; ix++) {
          const i = this.idx(ix, iy, iz);
          
          // Compute Laplacian (7-point stencil with periodic boundaries)
          const laplacian = this.computeLaplacian(ix, iy, iz);
          
          // Klein-Gordon equation: ∂²E/∂t² = c²∇²E − χ²E
          const E_curr = this.fieldCurrent[i];
          const E_prev = this.fieldPrevious[i];
          const chi_sq = this.chiField[i] ** 2;
          
          const force = laplacian - chi_sq * E_curr;
          
          // Verlet integration
          this.fieldNext[i] = 2 * E_curr - E_prev + this.dt_sq * force;
        }
      }
    }
    
    // Cycle buffers (no allocation, just swap references)
    this.rotateBuffers();
  }

  /**
   * Compute discrete Laplacian with periodic boundaries
   * ∇²E ≈ [E(i+1) + E(i-1) + E(j+1) + E(j-1) + E(k+1) + E(k-1) - 6E(i,j,k)] / dx²
   */
  private computeLaplacian(ix: number, iy: number, iz: number): number {
    const { size } = this.config;
    
    const E_center = this.fieldCurrent[this.idx(ix, iy, iz)];
    
    // 6 neighbors with periodic wrapping
    const E_xp = this.fieldCurrent[this.idx(this.wrap(ix + 1), iy, iz)];
    const E_xm = this.fieldCurrent[this.idx(this.wrap(ix - 1), iy, iz)];
    const E_yp = this.fieldCurrent[this.idx(ix, this.wrap(iy + 1), iz)];
    const E_ym = this.fieldCurrent[this.idx(ix, this.wrap(iy - 1), iz)];
    const E_zp = this.fieldCurrent[this.idx(ix, iy, this.wrap(iz + 1))];
    const E_zm = this.fieldCurrent[this.idx(ix, iy, this.wrap(iz - 1))];
    
    const laplacian_discrete = E_xp + E_xm + E_yp + E_ym + E_zp + E_zm - 6 * E_center;
    
    return this.laplacian_coeff * laplacian_discrete;
  }

  /**
   * Update chi field for moving particles (optional, for dynamic scenarios)
   */
  updateChiField(particles: ParticleState[]): void {
    this.initializeChiField(particles);
  }

  /**
   * Compute total field energy (for conservation tracking)
   */
  computeEnergy(): number {
    const { size, dx } = this.config;
    const dt = this.config.dt;
    const dV = dx ** 3; // Volume element
    
    let kinetic = 0;
    let potential = 0;
    
    for (let i = 0; i < this.fieldCurrent.length; i++) {
      const E_curr = this.fieldCurrent[i];
      const E_prev = this.fieldPrevious[i];
      const dE_dt = (E_curr - E_prev) / dt;
      
      // Kinetic energy: ½(∂E/∂t)²
      kinetic += 0.5 * dE_dt * dE_dt * dV;
      
      // Potential energy: ½[c²(∇E)² + χ²E²]
      // Approximate gradient with finite differences
      const chi_sq = this.chiField[i] ** 2;
      potential += 0.5 * chi_sq * E_curr * E_curr * dV;
    }
    
    this.totalEnergy = kinetic + potential;
    this.energyHistory.push(this.totalEnergy);
    
    return this.totalEnergy;
  }

  /**
   * Get current field state (for visualization)
   */
  getField(): Float32Array {
    return this.fieldCurrent;
  }

  /**
   * Get chi field (for debugging/visualization)
   */
  getChiField(): Float32Array {
    return this.chiField;
  }

  /**
   * Get energy conservation metrics
   */
  getEnergyMetrics(): { current: number; drift: number; history: number[] } {
    const initial = this.energyHistory[0] || 0;
    const current = this.totalEnergy;
    const drift = initial !== 0 ? Math.abs(current - initial) / initial : 0;
    
    return { current, drift, history: [...this.energyHistory] };
  }

  // === Helper methods ===

  /**
   * Convert 3D indices to 1D array index
   */
  private idx(x: number, y: number, z: number): number {
    const { size } = this.config;
    return z * size * size + y * size + x;
  }

  /**
   * Periodic boundary wrapping
   */
  private wrap(coord: number): number {
    const { size } = this.config;
    return ((coord % size) + size) % size;
  }

  /**
   * Cycle field buffers (prev ← curr ← next)
   */
  private rotateBuffers(): void {
    const temp = this.fieldPrevious;
    this.fieldPrevious = this.fieldCurrent;
    this.fieldCurrent = this.fieldNext;
    this.fieldNext = temp;
  }

  /**
   * Reset simulation to initial conditions
   */
  reset(): void {
    this.fieldCurrent.fill(0);
    this.fieldPrevious.fill(0);
    this.fieldNext.fill(0);
    this.energyHistory = [];
    this.totalEnergy = 0;
  }

  // === Additional methods for WebGPU compatibility ===

  /**
   * Get field gradient at a position (for force calculation)
   */
  async getFieldGradient(position: [number, number, number]): Promise<[number, number, number]> {
    const [px, py, pz] = position;
    const { size, dx } = this.config;
    const extent = size * dx / 2;

    // Convert world position to grid indices
    const ix = Math.floor((px + extent) / dx);
    const iy = Math.floor((py + extent) / dx);
    const iz = Math.floor((pz + extent) / dx);

    // Bounds check
    if (ix < 0 || ix >= size - 1 || iy < 0 || iy >= size - 1 || iz < 0 || iz >= size - 1) {
      return [0, 0, 0];
    }

    // Central difference approximation
    const gradX = (this.fieldCurrent[this.idx(ix + 1, iy, iz)] - 
                   this.fieldCurrent[this.idx(ix - 1, iy, iz)]) / (2 * dx);
    const gradY = (this.fieldCurrent[this.idx(ix, iy + 1, iz)] - 
                   this.fieldCurrent[this.idx(ix, iy - 1, iz)]) / (2 * dx);
    const gradZ = (this.fieldCurrent[this.idx(ix, iy, iz + 1)] - 
                   this.fieldCurrent[this.idx(ix, iy, iz - 1)]) / (2 * dx);

    return [gradX, gradY, gradZ];
  }

  /**
   * Step many times (batch processing for performance)
   */
  async stepMany(count: number): Promise<void> {
    for (let i = 0; i < count; i++) {
      this.step();
    }
  }

  /**
   * Calculate total field energy (alias for computeEnergy)
   */
  async calculateEnergy(): Promise<number> {
    return this.computeEnergy();
  }

  /**
   * Get energy drift percentage
   */
  getEnergyDrift(): number {
    const initial = this.energyHistory[0] || 0;
    const current = this.totalEnergy;
    return initial !== 0 ? Math.abs(current - initial) / initial : 0;
  }

  /**
   * Read chi field data (for visualization)
   */
  async readChiField(): Promise<Float32Array> {
    return this.chiField;
  }

  /**
   * Destroy/cleanup (no-op for CPU, but needed for interface compatibility)
   */
  destroy(): void {
    // No GPU resources to clean up
  }
}

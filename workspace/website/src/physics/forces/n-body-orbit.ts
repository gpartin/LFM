/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * N-Body Orbit Simulation
 * 
 * Generic N-body system where gravity emerges from chi field gradients.
 * Extends the binary orbit concept to handle 3+ bodies for chaotic dynamics.
 */

import { LFMLatticeWebGPU, LatticeConfig, ParticleState } from '../core/lattice-webgpu';
import { LFMLatticeCPU } from '../core/lattice-cpu';
import { PHYSICS_DEFAULTS } from '@/lib/constants';

export interface NBodyConfig {
  bodies: BodyConfig[];      // Array of body configurations
  chiStrength: number;       // Chi field coupling strength
  latticeSize: number;       // Grid size (32, 64, etc.)
  dt?: number;               // Optional timestep override
  sigma?: number;            // Optional Gaussian width for chi field
}

export interface BodyConfig {
  mass: number;
  position: [number, number, number];
  velocity: [number, number, number];
}

export interface NBodyState {
  particles: ParticleState[];
  time: number;
  energy: number;
  angularMomentum: number;
}

export class NBodyOrbitSimulation {
  private lattice: LFMLatticeWebGPU | LFMLatticeCPU;
  private config: NBodyConfig;
  private state: NBodyState;
  private latticeConfig: LatticeConfig;
  private stepCount: number = 0;
  private initialEnergy: number = 0;

  constructor(device: GPUDevice | null, config: NBodyConfig, useCPU: boolean = false) {
    this.latticeConfig = {
      size: config.latticeSize,
      dx: 0.1,
      dt: config.dt ?? 0.003,
      c: 1.0,
      chiStrength: config.chiStrength,
      sigma: config.sigma ?? 2.0,
    };

    if (useCPU || !device) {
      this.lattice = new LFMLatticeCPU(this.latticeConfig);
    } else {
      this.lattice = new LFMLatticeWebGPU(device!, this.latticeConfig);
    }
    
    this.config = config;
    this.state = this.initializeState();
  }

  private initializeState(): NBodyState {
    return {
      particles: this.config.bodies.map(body => ({
        position: [...body.position] as [number, number, number],
        velocity: [...body.velocity] as [number, number, number],
        mass: body.mass,
      })),
      time: 0,
      energy: 0,
      angularMomentum: 0,
    };
  }

  async initialize(): Promise<void> {
    await this.lattice.initialize();
    await this.lattice.updateChiField(this.state.particles);
    this.initialEnergy = this.calculateTotalEnergy();
  }

  /**
   * Calculate total energy (kinetic + gravitational potential)
   */
  private calculateTotalEnergy(): number {
    let kineticEnergy = 0;
    
    // Kinetic energy: sum of (1/2) * m * v^2
    for (const p of this.state.particles) {
      const vx = p.velocity[0];
      const vy = p.velocity[1];
      const vz = p.velocity[2];
      const v2 = vx * vx + vy * vy + vz * vz;
      kineticEnergy += 0.5 * p.mass * v2;
    }
    
    // Gravitational potential energy: -G * sum(mi*mj/rij) for all pairs
    let potentialEnergy = 0;
    const G = this.config.chiStrength; // Effective gravitational constant
    
    for (let i = 0; i < this.state.particles.length; i++) {
      for (let j = i + 1; j < this.state.particles.length; j++) {
        const pi = this.state.particles[i];
        const pj = this.state.particles[j];
        const dx = pi.position[0] - pj.position[0];
        const dy = pi.position[1] - pj.position[1];
        const dz = pi.position[2] - pj.position[2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r > 1e-8) {
          potentialEnergy -= G * pi.mass * pj.mass / r;
        }
      }
    }
    
    return kineticEnergy + potentialEnergy;
  }

  /**
   * Calculate total angular momentum (sum of r × p for all bodies)
   */
  private calculateAngularMomentum(): number {
    let Lz = 0;
    
    for (const p of this.state.particles) {
      const [x, y, z] = p.position;
      const [vx, vy, vz] = p.velocity;
      // L = r × p, we only track z-component for 2D orbits
      Lz += p.mass * (x * vy - y * vx);
    }
    
    return Lz;
  }

  /**
   * Step simulation forward by N frames
   */
  async stepBatch(frames: number): Promise<void> {
    for (let i = 0; i < frames; i++) {
      await this.step();
    }
  }

  /**
   * Single simulation step
   */
  private async step(): Promise<void> {
    const dt = this.latticeConfig.dt;
    
    // Update chi field with current particle positions
    await this.lattice.updateChiField(this.state.particles);
    
    // Get forces from chi field gradients for each particle
    const forces = await Promise.all(
      this.state.particles.map(p => this.lattice.getFieldGradient(p.position))
    );
    
    // Update velocities and positions using Verlet integration
    for (let i = 0; i < this.state.particles.length; i++) {
      const p = this.state.particles[i];
      const [fx, fy, fz] = forces[i];
      
      // a = F / m (but F already includes mass from chi field, so we scale)
      const ax = -fx * this.config.chiStrength;
      const ay = -fy * this.config.chiStrength;
      const az = -fz * this.config.chiStrength;
      
      // Velocity Verlet: v(t+dt) = v(t) + a*dt
      p.velocity[0] += ax * dt;
      p.velocity[1] += ay * dt;
      p.velocity[2] += az * dt;
      
      // Update position: x(t+dt) = x(t) + v*dt
      p.position[0] += p.velocity[0] * dt;
      p.position[1] += p.velocity[1] * dt;
      p.position[2] += p.velocity[2] * dt;
      
      // Wrap to lattice boundaries (periodic)
      const halfWidth = (this.config.latticeSize * this.latticeConfig.dx) / 2;
      p.position[0] = this.wrap(p.position[0], halfWidth);
      p.position[1] = this.wrap(p.position[1], halfWidth);
      p.position[2] = this.wrap(p.position[2], halfWidth);
    }
    
    // Update state
    this.state.time += dt;
    this.state.energy = this.calculateTotalEnergy();
    this.state.angularMomentum = this.calculateAngularMomentum();
    this.stepCount++;
  }

  private wrap(x: number, halfWidth: number): number {
    const width = 2 * halfWidth;
    if (x > halfWidth) return x - width;
    if (x < -halfWidth) return x + width;
    return x;
  }

  /**
   * Get current simulation state
   */
  getState(): NBodyState {
    return { ...this.state };
  }

  /**
   * Get energy drift since initialization
   */
  getEnergyDrift(): number {
    if (this.initialEnergy === 0) return 0;
    return (this.state.energy - this.initialEnergy) / Math.abs(this.initialEnergy);
  }

  /**
   * Reset simulation to initial conditions
   */
  reset(): void {
    this.state = this.initializeState();
    this.stepCount = 0;
  }

  /**
   * Cleanup GPU resources
   */
  destroy(): void {
    this.lattice.destroy();
  }

  /**
   * Update chi field without stepping (for parameter changes)
   */
  async refreshChiField(): Promise<void> {
    await this.lattice.updateChiField(this.state.particles);
  }

  /**
   * Get lattice for visualization
   */
  getLattice(): LFMLatticeWebGPU | LFMLatticeCPU {
    return this.lattice;
  }

  /**
   * Get lattice world extent for visualization (compatibility with BinaryOrbitSimulation)
   */
  latticeWorldExtent(): { N: number; dx: number; width: number; half: number } {
    const N = this.config.latticeSize;
    const dx = this.latticeConfig.dx;
    const width = N * dx;
    const half = width / 2;
    return { N, dx, width, half };
  }

  /**
   * Analytic chi field gradient at position (for visualization)
   * Returns gradient of chi field: ∇χ = -2 * Σ(m_i * exp(-r_i²/σ²) * (x - x_i) / σ²)
   */
  analyticChiGradientAt(pos: [number, number, number]): [number, number, number] {
    const sigma = this.latticeConfig.sigma ?? 2.0;
    const invSigma2 = 1.0 / (sigma * sigma);
    let gx = 0, gy = 0, gz = 0;
    for (const p of this.state.particles) {
      const dx = pos[0] - p.position[0];
      const dy = pos[1] - p.position[1];
      const dz = pos[2] - p.position[2];
      const r2 = dx*dx + dy*dy + dz*dz;
      const w = p.mass * Math.exp(-r2 * invSigma2);
      const coeff = w * (-2 * invSigma2);
      gx += coeff * dx;
      gy += coeff * dy;
      gz += coeff * dz;
    }
    return [gx, gy, gz];
  }

  /**
   * Analytic chi field value at position (baseline + particle Gaussians, for visualization)
   */
  analyticChiAt(pos: [number, number, number]): number {
    const sigma = this.latticeConfig.sigma ?? 2.0;
    const invSigma2 = 1.0 / (sigma * sigma);
    let chi = this.config.chiStrength; // baseline
    for (const p of this.state.particles) {
      const dx = pos[0] - p.position[0];
      const dy = pos[1] - p.position[1];
      const dz = pos[2] - p.position[2];
      const r2 = dx*dx + dy*dy + dz*dz;
      chi += p.mass * Math.exp(-r2 * invSigma2);
    }
    return chi;
  }

  /**
   * Baseline chi field value (for visualization compatibility)
   */
  chiBaseline(): number {
    return this.config.chiStrength;
  }
}

/**
 * Factory function: Create classic three-body problem with figure-8 orbit
 */
export function createFigure8ThreeBody(
  device: GPUDevice | null,
  chiStrength: number = 0.3,
  latticeSize: number = 64,
  useCPU: boolean = false
): NBodyOrbitSimulation {
  // Classic figure-8 solution (Chenciner & Montgomery, 2000)
  // Scaled for our lattice
  const config: NBodyConfig = {
    bodies: [
      {
        mass: 1.0,
        position: [0.970, 0.243, 0],
        velocity: [-0.466, -0.433, 0],
      },
      {
        mass: 1.0,
        position: [-0.970, -0.243, 0],
        velocity: [-0.466, -0.433, 0],
      },
      {
        mass: 1.0,
        position: [0, 0, 0],
        velocity: [0.932, 0.866, 0],
      },
    ],
    chiStrength,
    latticeSize,
    sigma: 1.5,
    dt: 0.002,
  };
  
  return new NBodyOrbitSimulation(device, config, useCPU);
}

/**
 * Factory function: Create Lagrange triangle (L4/L5 points)
 */
export function createLagrangeTriangle(
  device: GPUDevice | null,
  chiStrength: number = 0.3,
  latticeSize: number = 64,
  useCPU: boolean = false
): NBodyOrbitSimulation {
  const R = 3.0; // Triangle side length
  const omega = Math.sqrt(chiStrength / R); // Angular velocity
  
  const config: NBodyConfig = {
    bodies: [
      {
        mass: 1.0,
        position: [R, 0, 0],
        velocity: [0, omega * R, 0],
      },
      {
        mass: 1.0,
        position: [-R / 2, R * Math.sqrt(3) / 2, 0],
        velocity: [-omega * R * Math.sqrt(3) / 2, -omega * R / 2, 0],
      },
      {
        mass: 1.0,
        position: [-R / 2, -R * Math.sqrt(3) / 2, 0],
        velocity: [omega * R * Math.sqrt(3) / 2, -omega * R / 2, 0],
      },
    ],
    chiStrength,
    latticeSize,
    sigma: 1.5,
    dt: 0.002,
  };
  
  return new NBodyOrbitSimulation(device, config, useCPU);
}

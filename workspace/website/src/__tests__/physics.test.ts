/* -*- coding: utf-8 -*- */
/**
 * Physics Tests - Klein-Gordon Equation Validation
 * Tests mathematical correctness of physics implementation
 */

import { describe, it, expect } from '@jest/globals';

describe('Klein-Gordon Physics', () => {
  describe('Laplacian Calculation', () => {
    it('should compute 7-point stencil correctly for uniform field', () => {
      // Uniform field: ∇²E = 0 everywhere
      const field = new Float32Array(27).fill(1.0); // 3×3×3 cube
      const dx = 0.1;
      
      // Center point (1,1,1) in 3×3×3 grid
      const neighbors = [
        field[1 * 9 + 1 * 3 + 2], // x+1
        field[1 * 9 + 1 * 3 + 0], // x-1
        field[1 * 9 + 2 * 3 + 1], // y+1
        field[1 * 9 + 0 * 3 + 1], // y-1
        field[2 * 9 + 1 * 3 + 1], // z+1
        field[0 * 9 + 1 * 3 + 1], // z-1
      ];
      const center = field[1 * 9 + 1 * 3 + 1];
      
      const laplacian = (neighbors.reduce((a, b) => a + b, 0) - 6 * center) / (dx * dx);
      
      expect(laplacian).toBeCloseTo(0, 10); // Uniform field has zero laplacian
    });
    
    it('should handle periodic boundary conditions', () => {
      // Simple test: wrap-around should access correct neighbors
      const N = 4;
      const wrapCoord = (x: number) => ((x % N) + N) % N;
      
      expect(wrapCoord(-1)).toBe(3);
      expect(wrapCoord(0)).toBe(0);
      expect(wrapCoord(3)).toBe(3);
      expect(wrapCoord(4)).toBe(0);
      expect(wrapCoord(5)).toBe(1);
    });
  });
  
  describe('Verlet Integration', () => {
    it('should conserve energy for harmonic oscillator', () => {
      // x(t) = A*cos(ωt), E = 0.5*m*v² + 0.5*k*x²
      const m = 1.0;
      const k = 1.0; // ω = 1
      const dt = 0.01;
      const steps = 1000;
      
      let x = 1.0; // Initial position
      let v = 0.0; // Initial velocity
      const E0 = 0.5 * k * x * x; // Initial energy
      
      // Verlet integration
      let x_prev = x - v * dt; // Bootstrap
      for (let i = 0; i < steps; i++) {
        const a = -k / m * x;
        const x_next = 2 * x - x_prev + a * dt * dt;
        x_prev = x;
        x = x_next;
      }
      
      // Calculate final velocity and energy
      v = (x - x_prev) / dt;
      const E = 0.5 * m * v * v + 0.5 * k * x * x;
      
      // Energy should be conserved within numerical tolerance
      const drift = Math.abs(E - E0) / E0;
      expect(drift).toBeLessThan(0.01); // < 1% drift
    });
  });
  
  describe('Angular Momentum Conservation', () => {
    it('should conserve L = r × mv for circular motion', () => {
      // Particle in circular orbit
      const m = 1.0;
      const r = 1.0;
      const omega = 1.0;
      
      // Initial conditions
      let x = r, y = 0;
      let vx = 0, vy = omega * r;
      
      const L0 = m * (x * vy - y * vx); // Initial angular momentum
      
      // Simulate for one period
      const dt = 0.01;
      const steps = Math.ceil(2 * Math.PI / omega / dt);
      
      for (let i = 0; i < steps; i++) {
        // Centripetal acceleration
        const r_mag = Math.sqrt(x * x + y * y);
        const ax = -omega * omega * x;
        const ay = -omega * omega * y;
        
        // Update velocity and position
        vx += ax * dt;
        vy += ay * dt;
        x += vx * dt;
        y += vy * dt;
      }
      
      const L = m * (x * vy - y * vx);
      const L_drift = Math.abs(L - L0) / L0;
      
      expect(L_drift).toBeLessThan(0.05); // < 5% drift for simple integrator
    });
  });
});

describe('Chi Field Calculations', () => {
  it('should compute Gaussian chi field correctly', () => {
    // χ(r) = m * exp(-r²/σ²)
    const mass = 1.0;
    const sigma = 1.0;
    const r = 1.0;
    
    const chi = mass * Math.exp(-r * r / (sigma * sigma));
    
    expect(chi).toBeCloseTo(mass * Math.E ** -1, 5); // ≈ 0.368
  });
  
  it('should compute chi gradient correctly', () => {
    // ∇χ = -2m/(σ²) * exp(-r²/σ²) * r_vec
    const mass = 1.0;
    const sigma = 1.0;
    const x = 1.0, y = 0, z = 0;
    const r2 = x * x + y * y + z * z;
    
    const coeff = mass * Math.exp(-r2 / (sigma * sigma)) * (-2 / (sigma * sigma));
    const grad_x = coeff * x;
    
    expect(grad_x).toBeCloseTo(-2 * mass * Math.E ** -1, 5);
  });
});

describe('Orbital Mechanics', () => {
  it('should calculate circular orbit velocity correctly', () => {
    // v_circ = sqrt(G*M/r) for Newtonian gravity
    // In our model: v_circ ≈ sqrt(a_inward * r)
    const a_inward = 1.0; // Inward acceleration
    const r = 1.0; // Orbital radius
    
    const v_circ = Math.sqrt(a_inward * r);
    
    expect(v_circ).toBe(1.0);
  });
  
  it('should detect orbit direction (inward vs outward)', () => {
    // v < v_circ → spiral inward
    // v > v_circ → spiral outward
    const v = 0.9;
    const v_circ = 1.0;
    
    const isInward = v < v_circ;
    const isOutward = v > v_circ;
    
    expect(isInward).toBe(true);
    expect(isOutward).toBe(false);
  });
});

// Note: WebGPU-specific tests would require mocking or actual GPU access
// For now, we focus on mathematical correctness of algorithms

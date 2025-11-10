/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * Unified Lattice Interface
 * 
 * Abstracts between GPU (WebGPU) and CPU implementations
 * Both run the SAME Klein-Gordon equation, just different execution targets
 */

import { LFMLatticeWebGPU, LatticeConfig, ParticleState } from './lattice-webgpu';
import { LFMLatticeCPU } from './lattice-cpu';
import { detectBackend, type PhysicsBackend } from './backend-detector';

/**
 * Factory: Create appropriate lattice backend based on device capabilities
 */
export async function createLattice(preferredBackend?: PhysicsBackend): Promise<{
  lattice: LFMLatticeWebGPU | LFMLatticeCPU;
  backend: PhysicsBackend;
  config: LatticeConfig;
}> {
  // Detect capabilities
  const capabilities = await detectBackend();
  const backend = preferredBackend || capabilities.backend;
  
  // Adjust config based on backend
  const config: LatticeConfig = {
    size: getOptimalLatticeSize(backend),
    dx: 0.1,
    dt: 0.002,
    c: 1.0,
    chiStrength: 0.25,
    sigma: 2.0,
  };
  
  console.log(`[LFM Lattice] Using ${backend} backend with ${config.size}³ lattice`);
  
  if (backend === 'webgpu') {
    // GPU path
    if (!navigator.gpu) {
      console.warn('[LFM] WebGPU requested but not available, falling back to CPU');
      const cpuLattice = new LFMLatticeCPU(config);
      await cpuLattice.initialize();
      return { lattice: cpuLattice, backend: 'cpu', config };
    }
    
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('Failed to get GPU adapter');
      }
      const device = await adapter.requestDevice();
      
      const gpuLattice = new LFMLatticeWebGPU(device, config);
      await gpuLattice.initialize();
      return { lattice: gpuLattice, backend: 'webgpu', config };
    } catch (error) {
      console.warn('[LFM] WebGPU initialization failed, falling back to CPU:', error);
      const cpuLattice = new LFMLatticeCPU(config);
      await cpuLattice.initialize();
      return { lattice: cpuLattice, backend: 'cpu', config };
    }
    
  } else {
    // CPU fallback (WebGL or CPU)
    // For now, WebGL also uses CPU implementation
    // (Future: could implement WebGL compute shaders)
    const cpuLattice = new LFMLatticeCPU(config);
    await cpuLattice.initialize();
    return { lattice: cpuLattice, backend, config };
  }
}

/**
 * Get optimal lattice size based on backend capabilities
 */
function getOptimalLatticeSize(backend: PhysicsBackend): number {
  switch (backend) {
    case 'webgpu':
      return 64; // 64³ = 262K points, runs at 60fps on modern GPU
    case 'webgl':
      return 32; // 32³ = 32K points, reasonable for WebGL fragment shaders
    case 'cpu':
      return 32; // 32³ = 32K points, ~15fps on modern CPU
  }
}

/**
 * Get performance warning message for backend
 */
export function getBackendWarning(backend: PhysicsBackend): string | null {
  switch (backend) {
    case 'webgpu':
      return null; // No warning, optimal
    case 'webgl':
      return 'Using WebGL fallback. Physics is authentic but resolution is reduced. Consider upgrading your browser for WebGPU support.';
    case 'cpu':
      return 'Using CPU fallback. Physics is authentic but performance is limited. Enable GPU acceleration in your browser or upgrade for better experience.';
  }
}

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * Backend Capability Detection
 * 
 * Detects available compute backends and selects the best one:
 * 1. WebGPU (preferred) - Full LFM lattice simulation
 * 2. WebGL2 - Approximate simulation (fallback)
 * 3. CPU - Simplified Newtonian (last resort)
 */

export type PhysicsBackend = 'webgpu' | 'webgl' | 'cpu';

export interface BackendCapabilities {
  backend: PhysicsBackend;
  maxLatticeSize: number;
  features: {
    realLFM: boolean;           // True only for WebGPU
    chiFieldVisualization: boolean;
    latticeVisualization: boolean;
    energyConservation: boolean;
  };
  performance: {
    estimatedFPS: number;
    computeUnits: number;
  };
}

/**
 * Detect and select the best available physics backend
 */
export async function detectBackend(): Promise<BackendCapabilities> {
  // Try WebGPU first
  if (await hasWebGPU()) {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    
    if (device) {
      return {
        backend: 'webgpu',
        maxLatticeSize: 512, // 512³ lattice (plenty of room for orbits)
        features: {
          realLFM: true,
          chiFieldVisualization: true,
          latticeVisualization: true,
          energyConservation: true,
        },
        performance: {
          estimatedFPS: 60,
          computeUnits: adapter?.limits.maxComputeWorkgroupsPerDimension || 256,
        },
      };
    }
  }

  // Fallback to WebGL2
  if (hasWebGL2()) {
    return {
      backend: 'webgl',
      maxLatticeSize: 32, // Smaller lattice
      features: {
        realLFM: false, // ⚠️ Approximate only
        chiFieldVisualization: true,
        latticeVisualization: false,
        energyConservation: false,
      },
      performance: {
        estimatedFPS: 30,
        computeUnits: 1,
      },
    };
  }

  // Last resort: CPU fallback
  return {
    backend: 'cpu',
    maxLatticeSize: 16, // Very small
    features: {
      realLFM: false, // ⚠️ Newtonian approximation
      chiFieldVisualization: false,
      latticeVisualization: false,
      energyConservation: false,
    },
    performance: {
      estimatedFPS: 10,
      computeUnits: 1,
    },
  };
}

async function hasWebGPU(): Promise<boolean> {
  if (!navigator.gpu) {
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch (e) {
    console.warn('WebGPU detection failed:', e);
    return false;
  }
}

function hasWebGL2(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    return gl !== null;
  } catch (e) {
    return false;
  }
}

/**
 * Get a human-readable description of the backend
 */
export function getBackendDescription(backend: PhysicsBackend): string {
  switch (backend) {
    case 'webgpu':
      return 'WebGPU - Authentic LFM Simulation';
    case 'webgl':
      return 'WebGL2 - Approximate Physics (Not Real LFM)';
    case 'cpu':
      return 'CPU Fallback - Simplified Newtonian (Not Real LFM)';
  }
}

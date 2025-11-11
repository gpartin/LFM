/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Simulation Profile Decision
 *
 * Centralizes the decision of which UI complexity and simulation dimensionality
 * to use based on detected backend capabilities and experiment profile.
 *
 * Policy:
 * - If backend === 'webgpu' → advanced UI + 3D lattice (full experience)
 * - Else (webgl/cpu) → simplified UI + 1D (or thin-slab) lattice to preserve interactivity
 */

import type { BackendCapabilities, PhysicsBackend } from './backend-detector';

export type UIStrata = 'advanced' | 'simple';
export type Dimensionality = '3d' | '1d';

export interface SimulationProfile {
  ui: UIStrata;
  dim: Dimensionality;
}

export function decideSimulationProfile(
  backend: PhysicsBackend,
  capabilities: BackendCapabilities | null,
  experimentProfile: 'classical' | 'quantum' = 'classical'
): SimulationProfile {
  // For both classical and quantum, prefer full 3D when WebGPU is available
  if (backend === 'webgpu') {
    return { ui: 'advanced', dim: '3d' };
  }

  // Fallbacks: keep UI simple and switch to 1D (or thin slab emulation)
  // Quantum defaults accept lattice overlays even in simple mode
  return { ui: 'simple', dim: '1d' };
}

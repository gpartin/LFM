/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

/**
 * Shared types for physics simulation control across all canvas types.
 * 
 * These interfaces enable external control of physics simulations,
 * particularly for step-by-step debugging in RESEARCH mode experiments.
 */

/**
 * Generic simulation state that can hold state for any simulation type.
 * Different canvases use different subsets of these fields.
 */
export interface SimulationState {
  currentStep: number;
  
  // Wave packet state (WavePacketCanvas)
  E?: Float32Array;
  E_prev?: Float32Array;
  initialEnergy?: number;
  
  // N-body state (NBodyCanvas)
  energyDrift?: number;
  orbitalPeriod?: number;
  
  // Field dynamics state (FieldDynamicsCanvas)
  chiGradient?: number;
  
  // Extensible for future simulation types
  [key: string]: any;
}

/**
 * Control interface exposed by all canvas components.
 * Allows parent components to:
 * - Step through physics one frame at a time
 * - Capture simulation state for history/undo
 * - Restore previous states
 */
export interface SimulationControls {
  step: () => void;
  getState: () => SimulationState;
  setState: (state: SimulationState) => void;
}

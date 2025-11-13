/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Consolidated simulation state management using useReducer
 * Replaces 20+ individual useState calls to prevent excessive re-renders
 */

import { useReducer, Dispatch } from 'react';
import type { PhysicsBackend, BackendCapabilities } from '@/physics/core/backend-detector';

export interface SimulationParams {
  massRatio: number;
  orbitalDistance: number;
  chiStrength: number;
  sigma: number;
  dt: number;
  latticeSize: number;
  simSpeed: number;
  startPreset: number;
  startAngleDeg: number;
  velocityScale: number;
}

export interface SimulationMetrics {
  energy: string;
  drift: string;
  angularMomentum: string;
  orbitalPeriod: string;
  fps: string;
  separation: string;
  vRatio: string;
  effectiveSpeed: string;
  // LFM-derived GR-like metrics
  chiLocal?: string;
  gradMag?: string;
  tidalRad?: string;
  energyKE?: string;
  energyField?: string;
  energyRate?: string;
  clockRate?: string;
  tidalStress?: string; // Tidal force vs self-gravity ratio
  disruptionStatus?: string; // "Safe" | "Stressed" | "Disrupted"
  // Stellar evolution metrics
  evolutionPhase?: string; // "main-sequence" | "red-giant" | "white-dwarf"
  stellarRadius?: string;
  stellarTemperature?: string;
  coreDensity?: string;
  surfaceGravity?: string;
  luminosity?: string;
  // Gravitational lensing metrics
  lensingScenario?: string; // "solar-eclipse" | "galaxy-cluster" | "black-hole"
  deflectionAngle?: string;
  impactParameter?: string;
  einsteinRadius?: string;
  lightDelay?: string;
}

export interface SimulationUI {
  showParticles: boolean;
  showTrails: boolean;
  showChi: boolean;
  showLattice: boolean;
  showVectors: boolean;
  showWell: boolean;
  showDomes: boolean;
  showIsoShells: boolean;
  showBackground: boolean;
  fastForward: boolean;
  // Quantum-specific toggles
  showWave?: boolean;
  showBarrier?: boolean;
  showPhase?: boolean;
  showEnergyDensity?: boolean;
  showTransmissionPlot?: boolean;
}

export interface SimulationState {
  backend: PhysicsBackend;
  capabilities: BackendCapabilities | null;
  isLoading: boolean;
  isRunning: boolean;
  params: SimulationParams;
  metrics: SimulationMetrics;
  ui: SimulationUI;
  resetTrigger: number; // Increment to force simulation recreation
}

export type SimulationAction =
  | { type: 'SET_BACKEND'; payload: { backend: PhysicsBackend; capabilities: BackendCapabilities | null } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_RUNNING'; payload: boolean }
  | { type: 'UPDATE_PARAM'; payload: { key: keyof SimulationParams; value: number } }
  | { type: 'UPDATE_PARAMS'; payload: Partial<SimulationParams> }
  | { type: 'UPDATE_METRICS'; payload: Partial<SimulationMetrics> }
  | { type: 'UPDATE_UI'; payload: { key: keyof SimulationUI; value: boolean } }
  | { type: 'RESET_METRICS' }
  | { type: 'APPLY_PRESET'; payload: { preset: number } };

const initialMetrics: SimulationMetrics = {
  energy: '—',
  drift: '—',
  angularMomentum: '—',
  orbitalPeriod: '—',
  fps: '—',
  separation: '—',
  vRatio: '—',
  effectiveSpeed: '25',
};

const initialParams: SimulationParams = {
  massRatio: 16.0,  // matches blackHoleMass to moonMass ratio in page init
  orbitalDistance: 3.0,
  chiStrength: 0.25,
  sigma: 1.0,
  dt: 0.001,
  latticeSize: 64,
  simSpeed: 500.0,  // Black hole experiment needs higher speed to see action
  startPreset: 1,
  startAngleDeg: 0,
  velocityScale: 1.0,
};

const initialUI: SimulationUI = {
  showParticles: true,
  showTrails: true,
  showChi: false,
  showLattice: false,  // Default OFF - Simulation Grid should be hidden
  showVectors: true,
  showWell: true,
  showDomes: false,
  showIsoShells: false,
  showBackground: true,  // Default ON - Stars & Background should be visible
  fastForward: false,
  // Quantum defaults (used by quantum-profile pages)
  showWave: true,
  showBarrier: true,
  showPhase: false,
  showEnergyDensity: false,
  showTransmissionPlot: true,
};

export const initialSimulationState: SimulationState = {
  backend: 'cpu',
  capabilities: null,
  isLoading: true,
  isRunning: false,
  params: initialParams,
  metrics: initialMetrics,
  ui: initialUI,
  resetTrigger: 0,
};

export function simulationReducer(state: SimulationState, action: SimulationAction): SimulationState {
  switch (action.type) {
    case 'SET_BACKEND':
      return {
        ...state,
        backend: action.payload.backend,
        capabilities: action.payload.capabilities,
        isLoading: false,
      };
    
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_RUNNING':
      return { ...state, isRunning: action.payload };
    
    case 'UPDATE_PARAM':
      return {
        ...state,
        params: {
          ...state.params,
          [action.payload.key]: action.payload.value,
        },
      };
    
    case 'UPDATE_PARAMS':
      return {
        ...state,
        params: {
          ...state.params,
          ...action.payload,
        },
      };
    
    case 'UPDATE_METRICS':
      return {
        ...state,
        metrics: {
          ...state.metrics,
          ...action.payload,
        },
      };
    
    case 'UPDATE_UI':
      return {
        ...state,
        ui: {
          ...state.ui,
          [action.payload.key]: action.payload.value,
        },
      };
    
    case 'RESET_METRICS':
      return {
        ...state,
        metrics: initialMetrics,
      };
    
    case 'APPLY_PRESET':
      const preset = action.payload.preset;
      let velocityScale = 1.0;
      let startAngleDeg = 0;
      
      switch (preset) {
        case 1: // Near-circular
          velocityScale = 1.0;
          startAngleDeg = 0;
          break;
        case 2: // Slightly low speed (falls inward)
          velocityScale = 0.94;
          startAngleDeg = 45;
          break;
        case 3: // Slightly high speed (drifts outward)
          velocityScale = 1.06;
          startAngleDeg = 90;
          break;
      }
      
      return {
        ...state,
        params: {
          ...state.params,
          startPreset: preset,
          velocityScale,
          startAngleDeg,
          massRatio: 81.3,
          orbitalDistance: 3.0,
          chiStrength: 0.25,
          sigma: 2.0,
          dt: 0.001,
          latticeSize: 64,
          simSpeed: 25.0,  // Match new default speed
        },
        metrics: initialMetrics,
        resetTrigger: state.resetTrigger + 1, // Force simulation recreation
      };
    
    default:
      return state;
  }
}

export function useSimulationState(): [SimulationState, Dispatch<SimulationAction>] {
  return useReducer(simulationReducer, initialSimulationState);
}

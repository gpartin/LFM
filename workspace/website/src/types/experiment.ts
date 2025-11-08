/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Core Experiment Type System
 * 
 * Defines the standard interface for all physics experiments in the LFM website.
 * This enables extensibility, searchability, and consistent UI/UX across all experiments.
 */

import { ReactNode } from 'react';

/**
 * Experiment category for organization and filtering
 */
export type ExperimentCategory = 
  | 'orbital-mechanics'
  | 'gravity'
  | 'electromagnetic'
  | 'quantization'
  | 'relativistic'
  | 'energy-conservation'
  | 'thermodynamics'
  | 'coupling'
  | 'cosmology'
  | 'advanced';

/**
 * Difficulty level for user guidance
 */
export type ExperimentDifficulty = 'beginner' | 'intermediate' | 'advanced' | 'research';

/**
 * Backend requirements for the experiment
 */
export interface BackendRequirements {
  /** Minimum backend needed (webgpu preferred, webgl fallback, cpu last resort) */
  minBackend: 'webgpu' | 'webgl' | 'cpu';
  /** GPU features required (compute shaders, specific texture formats, etc.) */
  requiredFeatures?: string[];
  /** Estimated VRAM requirements in MB */
  estimatedVRAM?: number;
}

/**
 * Educational content and scientific context
 */
export interface ExperimentEducation {
  /** What physical phenomenon does this demonstrate? */
  whatYouSee: string;
  /** Scientific principles involved */
  principles: string[];
  /** Real-world applications or relevance */
  realWorld?: string;
  /** Links to related research papers, documentation, or evidence */
  references?: {
    title: string;
    url: string;
    type: 'paper' | 'documentation' | 'evidence' | 'external';
  }[];
  /** Common questions and answers */
  faq?: { question: string; answer: string; }[];
}

/**
 * Experiment metadata for discovery, search, and display
 */
export interface ExperimentMetadata {
  /** Unique identifier (kebab-case, used in URLs) */
  id: string;
  /** Display title */
  title: string;
  /** Short description (1-2 sentences for cards) */
  shortDescription: string;
  /** Full description (shown on experiment page) */
  fullDescription: string;
  /** Category for grouping */
  category: ExperimentCategory;
  /** Tags for search and filtering */
  tags: string[];
  /** Difficulty level */
  difficulty: ExperimentDifficulty;
  /** Author(s) */
  authors?: string[];
  /** Version number (for tracking updates) */
  version: string;
  /** Date created (ISO 8601) */
  created: string;
  /** Date last updated (ISO 8601) */
  updated: string;
  /** Is this experiment featured on the home page? */
  featured?: boolean;
  /** Backend requirements */
  backend: BackendRequirements;
  /** Educational content */
  education: ExperimentEducation;
  /** Preview image/thumbnail path */
  thumbnail?: string;
  /** Estimated runtime (seconds, for user expectations) */
  estimatedRuntime?: number;
}

/**
 * Parameter definition for experiment controls
 */
export interface ExperimentParameter<T = number> {
  /** Parameter key */
  key: string;
  /** Display label */
  label: string;
  /** Description/tooltip */
  description: string;
  /** Parameter type */
  type: 'number' | 'boolean' | 'select' | 'color';
  /** Default value */
  defaultValue: T;
  /** Min value (for number type) */
  min?: number;
  /** Max value (for number type) */
  max?: number;
  /** Step size (for number type) */
  step?: number;
  /** Unit label (e.g., "×", "m/s", "J") */
  unit?: string;
  /** Options (for select type) */
  options?: { label: string; value: T; }[];
  /** Can this parameter be changed while simulation is running? */
  liveUpdate?: boolean;
}

/**
 * Experiment configuration schema
 */
export interface ExperimentConfig {
  /** All tunable parameters */
  parameters: ExperimentParameter[];
  /** Preset configurations (quick starting points) */
  presets?: {
    id: string;
    label: string;
    description: string;
    values: Record<string, any>;
  }[];
  /** Default view settings */
  defaultViews?: {
    showParticles?: boolean;
    showTrails?: boolean;
    showField?: boolean;
    showGrid?: boolean;
    showVectors?: boolean;
    [key: string]: boolean | undefined;
  };
}

/**
 * Real-time metrics/telemetry from simulation
 */
export interface ExperimentMetrics {
  /** Display label */
  label: string;
  /** Current value (formatted string) */
  value: string;
  /** Status indicator (for color coding) */
  status?: 'good' | 'warning' | 'error' | 'neutral';
  /** Unit label */
  unit?: string;
  /** Tooltip explanation */
  tooltip?: string;
}

/**
 * Experiment result data (for export, analysis, or evidence)
 */
export interface ExperimentResults {
  /** Timestamp when results were generated */
  timestamp: string;
  /** Parameter values used */
  parameters: Record<string, any>;
  /** Final metrics */
  metrics: Record<string, any>;
  /** Any plots or visualizations (as data URLs or paths) */
  plots?: { label: string; data: string; }[];
  /** Raw data arrays (for further analysis) */
  rawData?: Record<string, number[] | Float32Array | Float64Array>;
  /** Notes or observations */
  notes?: string;
}

/**
 * Main Experiment interface - all experiments must implement this
 */
export interface Experiment {
  /** Experiment metadata */
  metadata: ExperimentMetadata;
  
  /** Configuration schema */
  config: ExperimentConfig;
  
  /** 
   * Initialize experiment (load assets, set up GPU resources, etc.)
   * Called once when experiment is first loaded
   */
  initialize: () => Promise<void>;
  
  /**
   * Clean up resources (GPU buffers, event listeners, etc.)
   * Called when experiment is unmounted or user navigates away
   */
  cleanup: () => Promise<void>;
  
  /**
   * Reset experiment to initial state
   */
  reset: () => Promise<void>;
  
  /**
   * Start/resume simulation
   */
  start: () => void;
  
  /**
   * Pause simulation
   */
  pause: () => void;
  
  /**
   * Step simulation by N frames (for debugging/analysis)
   */
  step?: (frames: number) => Promise<void>;
  
  /**
   * Update parameters on the fly
   */
  updateParameters: (params: Record<string, any>) => void;
  
  /**
   * Get current metrics/telemetry
   */
  getMetrics: () => ExperimentMetrics[];
  
  /**
   * Get current results (for export or display)
   */
  getResults: () => ExperimentResults;
  
  /**
   * Export results as JSON
   */
  exportResults: (format?: 'json' | 'csv') => Promise<string>;
  
  /**
   * React component to render experiment visualization
   * Receives current state as props
   */
  RenderComponent: React.ComponentType<{
    isRunning: boolean;
    parameters: Record<string, any>;
    views: Record<string, boolean>;
  }>;
}

/**
 * Experiment factory function type
 * Creates a new experiment instance with given device and config
 */
export type ExperimentFactory = (
  device: GPUDevice,
  initialConfig?: Partial<Record<string, any>>
) => Experiment;

/**
 * Experiment registry entry
 * Used by the registry to lazy-load experiments
 */
export interface ExperimentRegistryEntry {
  /** Experiment metadata (for search/display without loading full module) */
  metadata: ExperimentMetadata;
  
  /** Lazy loader for experiment module */
  loader: () => Promise<{ default: ExperimentFactory }>;
}

/* -*- coding: utf-8 -*- */
/**
 * Configuration constants for physics simulations
 * Extracted from inline magic numbers for maintainability
 */

// Binary Orbit Simulation
export const ORBIT_CONSTANTS = {
  /** Lattice field updates per micro-batch (balance between accuracy and perf) */
  LATTICE_UPDATE_INTERVAL: 100,
  
  /** Maximum physics steps per frame to prevent UI freezing */
  MAX_PHYSICS_STEPS_PER_FRAME: 2000,
  
  /** Frame time budget in milliseconds (target 60fps = 16.7ms, use 14ms for safety) */
  FRAME_TIME_BUDGET_MS: 14,
  
  /** Diagnostic logging interval (steps) */
  DIAGNOSTIC_SAMPLE_INTERVAL: 10,
  
  /** Energy calculation throttle interval (steps) */
  ENERGY_CALC_INTERVAL: 20,
} as const;

// 3D Visualization
export const VISUAL_CONSTANTS = {
  /** Maximum trail points to render per particle */
  TRAIL_MAX_POINTS: 800,
  
  /** Trail opacity */
  TRAIL_OPACITY: 0.5,
  
  /** Earth particle radius (scene units) */
  EARTH_RADIUS: 0.28,
  
  /** Moon particle radius (scene units) */
  MOON_RADIUS: 0.15,
  
  /** Starfield star count for background */
  STARFIELD_COUNT: 2000,
  
  /** FPS diagnostic logging interval (frames) */
  FPS_LOG_INTERVAL: 60,
} as const;

// GPU Configuration
export const GPU_CONSTANTS = {
  /** Default workgroup size for compute shader */
  COMPUTE_WORKGROUP_SIZE: 4,
  
  /** Float32 byte size */
  FLOAT32_SIZE: 4,
} as const;

// Physics Defaults
export const PHYSICS_DEFAULTS = {
  /** Earth-Moon mass ratio */
  EARTH_MOON_MASS_RATIO: 81.3,
  
  /** Default orbital separation (scene units) */
  DEFAULT_SEPARATION: 3.0,
  
  /** Default chi coupling strength */
  DEFAULT_CHI_STRENGTH: 0.25,
  
  /** Default Gaussian width (field reach) */
  DEFAULT_SIGMA: 2.0,
  
  /** Default time step for stability */
  DEFAULT_DT: 0.001,
  
  /** Default lattice size (power of 2) */
  DEFAULT_LATTICE_SIZE: 64,
  
  /** Default playback speed multiplier */
  DEFAULT_SIM_SPEED: 50.0,
  
  /**
   * Gaussian safety margin (multiples of sigma)
   * 
   * Ensures particle Gaussian chi-field contributions don't clip at lattice boundaries.
   * - 3σ encompasses 99.7% of Gaussian mass (3-sigma rule from statistics)
   * - Empirical validation: 2σ shows visible truncation artifacts, 4σ wastes lattice space
   * - Used in orbit initialization to prevent particles escaping visible domain
   */
  GAUSSIAN_SAFETY_MARGIN_SIGMAS: 3.0,
  
  /**
   * Velocity refinement gain for orbit initialization
   * 
   * Controls how aggressively initial tangential velocity is adjusted toward
   * lattice-sampled circular orbit estimate. Lower = more conservative.
   * - Range: 0.1-0.5 (empirically tested)
   * - 0.25 chosen to balance stability vs. convergence rate
   * - Values >0.5 can overshoot and cause oscillations
   */
  VELOCITY_REFINEMENT_GAIN: 0.25,
} as const;

// Adaptive Performance
export const PERF_CONSTANTS = {
  /** Base chunk size for physics batching */
  BASE_CHUNK_SIZE: 250,
  
  /** Chunk sizes per frame time (ms) */
  CHUNK_SIZE_FAST: 500,      // < 8ms
  CHUNK_SIZE_GOOD: 400,      // < 12ms
  CHUNK_SIZE_OK: 300,        // < 16ms
  CHUNK_SIZE_SLOW: 180,      // >= 16ms
  
  /** Min/max chunk sizes */
  MIN_CHUNK_SIZE: 80,
  MAX_CHUNK_SIZE: 600,
  
  /** Speed change threshold for immediate update vs gradual ramp */
  SPEED_LARGE_JUMP_THRESHOLD: 0.5,
  
  /** Max delta for gradual speed ramp per frame */
  SPEED_RAMP_MAX_DELTA: 25,
  
  /** Speed ramp rate (% of current speed) */
  SPEED_RAMP_RATE: 0.35,
} as const;

// Preset Configurations
export const PRESETS = {
  /** Preset 1: Near-circular orbit */
  CIRCULAR: {
    velocityScale: 1.0,
    startAngleDeg: 0,
  },
  
  /** Preset 2: Low speed (spiral inward) */
  INWARD: {
    velocityScale: 0.94,
    startAngleDeg: 45,
  },
  
  /** Preset 3: High speed (spiral outward) */
  OUTWARD: {
    velocityScale: 1.06,
    startAngleDeg: 90,
  },
} as const;

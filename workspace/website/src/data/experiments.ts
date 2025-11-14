/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

/**
 * Experiment Registry - Single Source of Truth for All Experiments
 * 
 * This file defines all experiments available in the LFM website.
 * 
 * Types:
 * - SHOWCASE: Interactive demonstrations for public (Earth-Moon, Three-Body, etc.)
 * - RESEARCH: Test harness experiments (REL-01, GRAV-12, ENER-03, etc.)
 * 
 * Data flows:
 * 1. Test harness configs (workspace/config/) define physics parameters
 * 2. Discoveries.json (workspace/docs/discoveries/) provides metadata
 * 3. This registry unifies both for website navigation
 */

export type ExperimentType = 'SHOWCASE' | 'RESEARCH';
export type SimulationType = 'binary-orbit' | 'n-body' | 'wave-packet' | 'field-dynamics';
export type TierName = 'Relativistic' | 'Gravity' | 'Energy' | 'Quantization' | 'Electromagnetic' | 'Coupling' | 'Thermodynamics';

export interface ExperimentDefinition {
  // Identity
  id: string;                          // URL slug: "earth-moon" or "REL-01"
  testId?: string;                     // Test harness ID: "REL-01" (for RESEARCH type)
  displayName: string;                 // "Earth-Moon Orbit" or "REL-01: Isotropy Test"
  type: ExperimentType;
  
  // Classification
  tier?: number;                       // 1-7 for research experiments
  tierName?: TierName;                 // "Relativistic", "Gravity", etc.
  category: string;                    // "Orbital Mechanics" or "Lorentz Invariance"
  
  // Presentation
  tagline: string;                     // Short description for cards
  description: string;                 // Full explanation paragraph
  icon?: string;                       // Emoji or icon identifier
  difficulty?: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  featured?: boolean;                  // Show on homepage
  
  // Physics Configuration
  simulation: SimulationType;
  backend: 'webgpu' | 'cpu' | 'both';
  
  // Initial Conditions (maps to simulation params)
  initialConditions: {
    latticeSize: number;               // Grid points (64, 128, 256, 512)
    dt: number;                        // Time step
    dx: number;                        // Space step
    steps: number;                     // Simulation steps
    chi: number | [number, number];   // Constant chi or gradient [min, max]
    
    // Type-specific fields
    particles?: Array<{                // For orbital simulations
      mass: number;
      position: [number, number, number];
      velocity: [number, number, number];
    }>;
    
    wavePacket?: {                     // For wave propagation
      amplitude: number;
      width: number;
      k: [number, number, number];     // Wave vector
    };
    
    fieldConfig?: {                    // For field dynamics
      initialProfile: 'gaussian' | 'uniform' | 'random';
      chiDynamics?: boolean;           // Enable dynamic chi evolution
    };
  };
  
  // Validation Criteria (for research experiments)
  validation?: {
    energyDrift: number;               // Max allowed energy drift
    phaseError?: number;               // Max phase velocity error
    anisotropy?: number;               // Max anisotropy
    customMetrics?: Record<string, number>;
  };
  
  // Visualization Configuration
  visualizationPreset?: string;        // Preset name from visualization-presets.ts
  visualization: {
    showParticles: boolean;
    showTrails: boolean;
    showChi: boolean;
    showLattice: boolean;
    showVectors: boolean;
    showWell: boolean;
    showDomes: boolean;
    showIsoShells: boolean;
    showBackground: boolean;
  };
  
  // Documentation Links
  links: {
    testHarnessConfig?: string;        // Path to config JSON
    results?: string;                  // Path to results directory
    discovery?: string;                // Discovery JSON entry
    documentation?: string;            // Additional docs
  };
  
  // Status
  status: 'production' | 'beta' | 'development' | 'planned';
  implementationNotes?: string;        // Technical notes

  // Optional per-experiment notice for context (e.g., why skipped)
  specialNotice?: {
    heading?: string;
    body: string;
    severity?: 'info' | 'warning' | 'critical';
  };

  // UI Metadata (auto-derived for RESEARCH via generation script; manual for SHOWCASE)
  ui?: {
    panels: string[];                  // Ordered list of logical panels (e.g., ['field','wave','particles','metrics','energy'])
    controls: {
      play: boolean;                   // Show play/pause button
      reset: boolean;                  // Show reset control
      speedSlider?: boolean;           // Show speed adjustment slider
      chiToggle?: boolean;             // Allow toggling chi field overlay
    };
    metrics?: {
      showEnergyDrift?: boolean;       // Display preview energy drift metric
      showPrimaryMetric?: boolean;     // Display primary metric preview if approximable in browser
      showTransmission?: boolean;      // Wave packet transmission (tunneling/double slit)
      showReflection?: boolean;        // Wave packet reflection
      showOrbitElements?: boolean;     // Orbital mechanics panel
    };
  };
}

/**
 * SHOWCASE EXPERIMENTS - Interactive Demonstrations
 */
const SHOWCASE_EXPERIMENTS: ExperimentDefinition[] = [
  {
    id: 'binary-orbit',
    displayName: 'Earth-Moon Orbit',
    type: 'SHOWCASE',
    category: 'Orbital Mechanics',
    tagline: 'Watch Earth and Moon orbit from emergent gravity',
    description: 'Demonstrates how orbital mechanics emerges naturally from the LFM field equation. No Newtonian force law needed—gravity arises from χ-field gradients.',
    icon: '🌍',
    difficulty: 'beginner',
    featured: true,
    simulation: 'binary-orbit',
    backend: 'both',
    initialConditions: {
      latticeSize: 64,
      dt: 0.005,
      dx: 0.1,
      steps: 10000,
      chi: 0.25,
      particles: [
        { mass: 81.3, position: [0, 0, 0], velocity: [0, 0, 0] },
        { mass: 1.0, position: [5, 0, 0], velocity: [0, 0.2, 0] }
      ]
    },
    visualization: {
      showParticles: true,
      showTrails: true,
      showChi: true,
      showLattice: false,
      showVectors: true,
      showWell: true,
      showDomes: false,
      showIsoShells: false,
      showBackground: true
    },
    links: {
      discovery: 'Tier 2 - Gravitational',
      documentation: 'docs/text/LFM_Master.txt'
    },
    status: 'production'
  },
  
  {
    id: 'three-body',
    displayName: 'Three-Body Problem',
    type: 'SHOWCASE',
    category: 'Chaotic Dynamics',
    tagline: 'Famous figure-8 orbit in emergent spacetime',
    description: 'The classic three-body problem with the Chenciner-Montgomery figure-8 solution. Demonstrates stable periodic orbits in chaotic system.',
    icon: '🔺',
    difficulty: 'intermediate',
    featured: true,
    simulation: 'n-body',
    backend: 'webgpu',
    initialConditions: {
      latticeSize: 64,
      dt: 0.002,
      dx: 0.05,
      steps: 15000,
      chi: 0.25,
      particles: [
        { mass: 1.0, position: [-0.97000436, 0.24308753, 0], velocity: [0.466203685, 0.43236573, 0] },
        { mass: 1.0, position: [0, 0, 0], velocity: [-0.93240737, -0.86473146, 0] },
        { mass: 1.0, position: [0.97000436, -0.24308753, 0], velocity: [0.466203685, 0.43236573, 0] }
      ]
    },
    visualization: {
      showParticles: true,
      showTrails: true,
      showChi: false,
      showLattice: false,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: true
    },
    links: {
      documentation: 'https://arxiv.org/abs/math/0011268'
    },
    status: 'production'
  },
  
  {
    id: 'black-hole',
    displayName: 'Black Hole',
    type: 'SHOWCASE',
    category: 'Extreme Gravity',
    tagline: 'Event horizon formation from concentrated χ-field',
    description: 'Demonstrates black hole physics as emergent phenomenon. Extreme χ-concentration creates event horizon where escape velocity exceeds c.',
    icon: '⚫',
    difficulty: 'advanced',
    featured: true,
    simulation: 'binary-orbit',
    backend: 'webgpu',
    initialConditions: {
      latticeSize: 64,
      dt: 0.005,
      dx: 0.1,
      steps: 10000,
      chi: 0.8,
      particles: [
        { mass: 10000, position: [0, 0, 0], velocity: [0, 0, 0] },
        { mass: 1.0, position: [3, 0, 0], velocity: [0, 0.5, 0] }
      ]
    },
    visualization: {
      showParticles: true,
      showTrails: true,
      showChi: true,
      showLattice: false,
      showVectors: true,
      showWell: true,
      showDomes: true,
      showIsoShells: true,
      showBackground: true
    },
    links: {
      discovery: 'Tier 2 - Gravitational'
    },
    status: 'production'
  },
  
  {
    id: 'gravitational-lensing',
    displayName: 'Gravitational Lensing',
    type: 'SHOWCASE',
    category: 'Astrophysics',
    tagline: 'Light bending around massive objects (Einstein 1919)',
    description: 'Shows how light rays bend when passing near massive objects due to χ field gradients. Demonstrates Einstein\'s 1919 eclipse observation, galaxy cluster lensing, and black hole photon spheres—all from pure LFM field dynamics.',
    icon: '💫',
    difficulty: 'intermediate',
    featured: true,
    simulation: 'binary-orbit',
    backend: 'webgpu',
    initialConditions: {
      latticeSize: 64,
      dt: 0.0001,
      dx: 0.1,
      steps: 10000,
      chi: 0.5,
      particles: [
        { mass: 2000, position: [0, 0, 0], velocity: [0, 0, 0] },
        { mass: 0.01, position: [3, 0, 0], velocity: [0, 0.9, 0] }
      ]
    },
    visualization: {
      showParticles: true,
      showTrails: true,
      showChi: false,
      showLattice: false,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: true
    },
    links: {},
    status: 'production'
  },
  
  {
    id: 'big-bang',
    displayName: 'Big Bang',
    type: 'SHOWCASE',
    category: 'Cosmology',
    tagline: 'Cosmological expansion from primordial singularity',
    description: 'Models big bang expansion as explosive χ-field release from point singularity. Field energy spreads across lattice creating expansion.',
    icon: '💥',
    difficulty: 'expert',
    featured: false,
    simulation: 'binary-orbit',
    backend: 'webgpu',
    initialConditions: {
      latticeSize: 64,
      dt: 0.01,
      dx: 0.2,
      steps: 5000,
      chi: 0.8,
      particles: [
        { mass: 10000, position: [0, 0, 0], velocity: [0, 0, 0] },
        { mass: 1.0, position: [0.5, 0, 0], velocity: [0, 2.0, 0] }
      ]
    },
    visualization: {
      showParticles: true,
      showTrails: true,
      showChi: true,
      showLattice: false,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: true
    },
    links: {},
    status: 'production'
  },
  
  {
    id: 'quantum-tunneling',
    displayName: 'Quantum Tunneling',
    type: 'SHOWCASE',
    category: 'Quantum Mechanics',
    tagline: 'Wave packet tunneling through chi barrier',
    description: 'Demonstrates quantum tunneling as wave packet interaction with finite potential barrier. Shows transmission (T), reflection (R), and conservation (T+R≈1) with real-time field visualization.',
    icon: '⚛️',
    difficulty: 'advanced',
    featured: true,
    simulation: 'wave-packet',
    backend: 'both',
    initialConditions: {
      latticeSize: 64,
      dt: 0.0002,
      dx: 0.02,
      steps: 5000,
      chi: 0.0,
      wavePacket: {
        amplitude: 1.0,
        width: 3.0,
        k: [5.0, 0, 0]
      }
    },
    validation: {
      energyDrift: 1e-4
    },
    visualization: {
      showParticles: false,
      showTrails: false,
      showChi: true,
      showLattice: true,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: false
    },
    links: {
      discovery: 'Tier 4 - Quantization'
    },
    status: 'production'
  }
  ,
  {
    id: 'double-slit',
    displayName: 'Quantum Double Slit',
    type: 'SHOWCASE',
    category: 'Quantum Mechanics',
    tagline: 'Deterministic interference pattern from two apertures',
    description: 'A plane-like wave passes through two narrow slits and forms a classic interference pattern on the screen. Live metrics show fringe spacing, visibility, and slit intensity ratio.',
    icon: '🧪',
    difficulty: 'beginner',
    featured: true,
    simulation: 'wave-packet',
    backend: 'both',
    initialConditions: {
      latticeSize: 64,
      dt: 0.001,
      dx: 0.1,
      steps: 6000,
      chi: 0.0,
      wavePacket: {
        amplitude: 1.0,
        width: 8.0,
        k: [6.0, 0, 0]
      }
    },
    validation: {
      energyDrift: 1e-4
    },
    visualization: {
      showParticles: false,
      showTrails: false,
      showChi: true,
      showLattice: true,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: false
    },
    links: {
      discovery: 'Tier 4 - Quantization'
    },
    status: 'production'
  }
  ,
  {
    id: 'em-thermo-coupling',
    displayName: 'EM → Thermal Coupling',
    type: 'SHOWCASE',
    category: 'Electromagnetic + Thermodynamics',
    tagline: 'Light heats a medium: E² power deposits into temperature field',
    description: 'A minimal crossover demo. A Gaussian EM wave deposits energy (∝ E²) into a temperature field that diffuses over time. Adjust amplitude, frequency, absorption, and diffusivity to see heating patterns and spread.',
    icon: '⚡',
    difficulty: 'beginner',
    featured: false,
    simulation: 'field-dynamics',
    backend: 'cpu',
    initialConditions: {
      latticeSize: 64,
      dt: 0.01,
      dx: 0.1,
      steps: 4000,
      chi: 0.0,
      fieldConfig: { initialProfile: 'gaussian', chiDynamics: false }
    },
    visualization: {
      showParticles: false,
      showTrails: false,
      showChi: false,
      showLattice: false,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: false
    },
    links: {},
    status: 'beta',
    implementationNotes: 'Canvas-based EMThermo demo. Consider adding Poynting vectors and energy conservation metrics.'
  }
];

/**
 * RESEARCH EXPERIMENTS - Test Harness Validation
 * 
 * These map 1:1 with test harness experiments.
 * Each entry references the corresponding config file.
 */
const RESEARCH_EXPERIMENTS: ExperimentDefinition[] = [
  // Tier 1: Relativistic (16 tests)
  {
    id: 'REL-01',
    testId: 'REL-01',
    displayName: 'REL-01: Isotropy — Coarse Grid',
    type: 'RESEARCH',
    tier: 1,
    tierName: 'Relativistic',
    category: 'Lorentz Invariance',
    tagline: 'Validates wave propagation isotropy on coarse grid',
    description: 'Tests that wave propagation speed is identical in all directions (x, y, z) on coarse spatial grid. Validates Lorentz symmetry emergence from discrete lattice.',
    difficulty: 'intermediate',
    simulation: 'wave-packet',
    backend: 'both',
    initialConditions: {
      latticeSize: 512,
      dt: 0.0025,
      dx: 0.008,
      steps: 6000,
      chi: 0.0,
      wavePacket: {
        amplitude: 1.0,
        width: 2.0,
        k: [0.05, 0, 0]
      }
    },
    validation: {
      energyDrift: 1e-6,
      anisotropy: 0.01
    },
    visualization: {
      showParticles: false,
      showTrails: false,
      showChi: false,
      showLattice: true,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: false
    },
    links: {
      testHarnessConfig: 'workspace/config/config_tier1_relativistic.json',
      results: 'workspace/results/Relativistic/REL-01/',
      discovery: 'Tier 1 - Relativistic'
    },
    status: 'production'
  },
  
];

// ============================================================================
// AUTO-GENERATED RESEARCH EXPERIMENTS
// ============================================================================

/**
 * ⚠️ AUTO-GENERATED from test harness configs
 * Source: scripts/generate-research-experiments.js
 * To regenerate: npm run generate:experiments
 * 
 * Contains 105 research experiments from all 7 tiers
 */
import { RESEARCH_EXPERIMENTS as GENERATED_RESEARCH_EXPERIMENTS } from './research-experiments-generated';

// Merge manual showcase example with generated experiments
// Apply overrides for specific experiments (non-generated fields, e.g., special notices)
const RESEARCH_OVERRIDES: Record<string, Partial<ExperimentDefinition>> = {
  'GRAV-09': {
    specialNotice: {
      heading: 'Why this test is skipped on the website',
      severity: 'warning',
      body: `GRAV-09 probes a property that only has a well-defined meaning under a continuum assumption.
The quantity it evaluates presumes an infinitely divisible spacetime; on a discrete lattice with finite Δx, the metric is not invariant and does not converge in the intended sense.
Because LFM models physics on a discrete lattice (by construction), presenting GRAV-09 here would imply a continuum-only result that is outside this model’s scope.
Skipping it is deliberate and scientifically significant: it draws a clear boundary between continuum assumptions and discrete-lattice predictions.
For context and discussion, see analysis/grav09_discrete_grid_limitation.md.`
    }
  }
};

const ALL_RESEARCH_EXPERIMENTS: ExperimentDefinition[] = [
  // Keep the manual REL-01 as a showcase template
  RESEARCH_EXPERIMENTS[0],
  // Add all generated experiments (includes REL-01 again, but with simpler metadata)
  ...GENERATED_RESEARCH_EXPERIMENTS
].map((exp) => ({
  ...exp,
  ...(RESEARCH_OVERRIDES[exp.id] || {})
}));

/**
 * Get all experiments (showcase + research)
 */
export function getAllExperiments(): ExperimentDefinition[] {
  return [...SHOWCASE_EXPERIMENTS, ...ALL_RESEARCH_EXPERIMENTS];
}

/**
 * Get showcase experiments only
 */
export function getShowcaseExperiments(): ExperimentDefinition[] {
  return SHOWCASE_EXPERIMENTS;
}

/**
 * Get research experiments by tier
 */
export function getResearchExperimentsByTier(tier: number): ExperimentDefinition[] {
  return ALL_RESEARCH_EXPERIMENTS.filter(exp => exp.tier === tier);
}

/**
 * Get single experiment by ID
 */
export function getExperimentById(id: string): ExperimentDefinition | undefined {
  return getAllExperiments().find(exp => exp.id === id);
}

/**
 * Get featured experiments
 */
export function getFeaturedExperiments(): ExperimentDefinition[] {
  return SHOWCASE_EXPERIMENTS.filter(exp => exp.featured);
}

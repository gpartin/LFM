/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * 
 * Visualization Preset System
 * ==========================
 * 
 * Defines which visualization options are available for each experiment type.
 * 
 * Design Principles:
 * - **Modular**: Each preset is self-contained and reusable
 * - **Scalable**: Easy to add new presets or options
 * - **Performant**: Static config, zero runtime overhead
 * - **Maintainable**: Centralized definitions, single source of truth
 * - **Intuitive**: Presets named after physics concepts, not implementation details
 * 
 * Architecture:
 * - Presets define WHICH options are available
 * - Defaults define WHICH options are initially enabled
 * - Experiment metadata maps to preset ID
 * - UI components render available options only
 */

// ============================================================================
// Core Types
// ============================================================================

/**
 * Visualization option identifier
 * 
 * Naming convention:
 * - Keep names concise but descriptive
 * - Match user-facing labels where possible
 * - Avoid technical jargon in option names
 */
export type VisualizationOption = 
  | 'bodies'          // Physical objects (Earth, Moon, particles)
  | 'trails'          // Trajectory paths / orbital history
  | 'chi-field-2d'    // 2D heatmap of chi field magnitude
  | 'lattice-grid'    // Simulation grid overlay
  | 'force-vectors'   // Vector field arrows
  | 'gravity-well-3d' // 3D surface plot of potential
  | 'field-bubbles'   // 3D volumetric field magnitude
  | 'field-shells'    // Isosurface shells
  | 'starfield';      // Background celestial objects

/**
 * User-facing labels for each option
 * 
 * Guidelines:
 * - Use plain language, avoid jargon
 * - Be specific about what's shown
 * - Keep under 25 characters for UI space
 */
export const VISUALIZATION_LABELS: Record<VisualizationOption, string> = {
  'bodies': 'Earth & Moon',
  'trails': 'Orbital Paths',
  'chi-field-2d': 'Gravity Field (2D)',
  'lattice-grid': 'Simulation Grid',
  'force-vectors': 'Force Arrows',
  'gravity-well-3d': 'Gravity Well (Surface)',
  'field-bubbles': 'Field Bubbles (3D)',
  'field-shells': 'Field Shells',
  'starfield': 'Stars & Sun'
};

/**
 * Tooltip descriptions for each option
 * 
 * Explains what the visualization shows and why it's useful
 */
export const VISUALIZATION_TOOLTIPS: Record<VisualizationOption, string> = {
  'bodies': 'Show physical objects (Earth, Moon, particles). Toggle off to see just the fields.',
  'trails': 'Show trajectory history. Useful for seeing orbital shapes and stability over time.',
  'chi-field-2d': '2D color map of chi field strength. Darker = stronger field. Shows how "heavy" spacetime is.',
  'lattice-grid': 'Simulation grid overlay. Shows the discrete lattice structure of spacetime.',
  'force-vectors': 'Arrows showing field gradient direction. Points where objects will accelerate.',
  'gravity-well-3d': '3D surface plot of gravitational potential. See the "dip" objects create in spacetime.',
  'field-bubbles': '3D volumetric rendering of field magnitude. See field structure in all dimensions.',
  'field-shells': 'Isosurface shells at constant field strength. Like topographic contours but in 3D.',
  'starfield': 'Background stars and sun for context. Purely cosmetic, not part of simulation.'
};

/**
 * Maps legacy field names (from ExperimentDefinition) to new option IDs
 * 
 * Maintains backward compatibility with existing generated data
 */
export const LEGACY_FIELD_MAPPING: Record<string, VisualizationOption> = {
  'showParticles': 'bodies',
  'showTrails': 'trails',
  'showChi': 'chi-field-2d',
  'showLattice': 'lattice-grid',
  'showVectors': 'force-vectors',
  'showWell': 'gravity-well-3d',
  'showDomes': 'field-bubbles',
  'showIsoShells': 'field-shells',
  'showBackground': 'starfield'
};

/**
 * Reverse mapping: new option IDs to legacy field names
 * 
 * Used when interfacing with existing simulation components
 */
export const OPTION_TO_LEGACY_FIELD: Record<VisualizationOption, string> = {
  'bodies': 'showParticles',
  'trails': 'showTrails',
  'chi-field-2d': 'showChi',
  'lattice-grid': 'showLattice',
  'force-vectors': 'showVectors',
  'gravity-well-3d': 'showWell',
  'field-bubbles': 'showDomes',
  'field-shells': 'showIsoShells',
  'starfield': 'showBackground'
};

// ============================================================================
// Preset Definitions
// ============================================================================

/**
 * Visualization preset configuration
 * 
 * Each preset defines:
 * - Which options are available (UI shows only these)
 * - Which options are enabled by default
 * - Metadata for documentation
 */
export interface VisualizationPreset {
  /** Unique identifier */
  id: string;
  
  /** Display name for preset selector (if we add UI for switching presets) */
  name: string;
  
  /** Short description of what this preset is for */
  description: string;
  
  /** Which visualization options are available in this preset */
  availableOptions: VisualizationOption[];
  
  /** Which options are enabled by default (must be subset of availableOptions) */
  defaultEnabled: VisualizationOption[];
  
  /** Recommended use cases (for documentation) */
  useCases: string[];
}

/**
 * All available presets
 * 
 * Design notes:
 * - Start with 5 core presets covering main experiment types
 * - Can add more presets as needed without breaking existing experiments
 * - Presets are ordered from most feature-rich to most minimal
 */
export const VISUALIZATION_PRESETS: Record<string, VisualizationPreset> = {
  /**
   * ORBITAL SHOWCASE
   * ================
   * Full visualization suite for orbital mechanics demonstrations.
   * Used by: Earth-Moon, Three-Body, Black Hole, Stellar Collapse, Big Bang
   */
  'orbital-showcase': {
    id: 'orbital-showcase',
    name: 'Orbital Mechanics (Full Suite)',
    description: 'Complete visualization for orbital dynamics with all 3D field representations',
    availableOptions: [
      'bodies',
      'trails',
      'chi-field-2d',
      'lattice-grid',
      'force-vectors',
      'gravity-well-3d',
      'field-bubbles',
      'field-shells',
      'starfield'
    ],
    defaultEnabled: [
      'bodies',
      'trails',
      'force-vectors',
      'gravity-well-3d',
      'starfield'
    ],
    useCases: [
      'Earth-Moon orbit demonstrations',
      'N-body problem visualizations',
      'Black hole dynamics',
      'Gravitational showcase experiments'
    ]
  },

  /**
   * ORBITAL MINIMAL
   * ===============
   * Simplified orbital visualization without heavy 3D field rendering.
   * Used by: Gravitational research tests (GRAV-*)
   */
  'orbital-minimal': {
    id: 'orbital-minimal',
    name: 'Orbital Mechanics (Minimal)',
    description: 'Essential orbital visualization without expensive 3D field rendering',
    availableOptions: [
      'bodies',
      'trails',
      'chi-field-2d',
      'lattice-grid',
      'force-vectors',
      'starfield'
    ],
    defaultEnabled: [
      'bodies',
      'trails',
      'chi-field-2d',
      'lattice-grid'
    ],
    useCases: [
      'Gravitational redshift tests',
      'Orbit stability analysis',
      'Time dilation measurements',
      'Research-grade orbital simulations'
    ]
  },

  /**
   * WAVE DYNAMICS
   * =============
   * Wave propagation visualization with field emphasis.
   * Used by: Relativistic tests (REL-*), Wave packet experiments
   */
  'wave-dynamics': {
    id: 'wave-dynamics',
    name: 'Wave Propagation',
    description: 'Field-focused visualization for wave dynamics and propagation',
    availableOptions: [
      'chi-field-2d',
      'lattice-grid',
      'field-bubbles',
      'field-shells'
    ],
    defaultEnabled: [
      'chi-field-2d',
      'lattice-grid',
      'field-bubbles'
    ],
    useCases: [
      'Isotropy tests (REL-01 through REL-05)',
      'Causality tests (REL-06 through REL-08)',
      'Wave packet propagation',
      'Phase velocity measurements'
    ]
  },

  /**
   * FIELD ONLY
   * ==========
   * Pure field visualization without particles or orbits.
   * Used by: EM tests (EM-*), Coupling tests (COUP-*), Quantization (QUAN-*)
   */
  'field-only': {
    id: 'field-only',
    name: 'Field Visualization',
    description: 'Field magnitude and structure without orbital mechanics',
    availableOptions: [
      'chi-field-2d',
      'lattice-grid',
      'force-vectors',
      'field-bubbles',
      'field-shells'
    ],
    defaultEnabled: [
      'chi-field-2d',
      'lattice-grid',
      'field-bubbles'
    ],
    useCases: [
      'Electromagnetic field configurations',
      'Coupling strength analysis',
      'Quantization and bound states',
      'Field structure studies'
    ]
  },

  /**
   * MINIMAL
   * =======
   * Bare minimum visualization for performance or simplicity.
   * Used by: Energy conservation tests (ENER-*), Thermodynamics (THERM-*)
   */
  'minimal': {
    id: 'minimal',
    name: 'Minimal Visualization',
    description: 'Essential visualization only - maximum performance',
    availableOptions: [
      'chi-field-2d',
      'lattice-grid'
    ],
    defaultEnabled: [
      'lattice-grid'
    ],
    useCases: [
      'Energy conservation validation',
      'Performance-critical tests',
      'Numerical accuracy studies',
      'High-resolution simulations'
    ]
  }
};

// ============================================================================
// Inference Logic
// ============================================================================

/**
 * Infer visualization preset from experiment characteristics
 * 
 * This function is called by generate_website_experiments.js during build
 * to automatically assign presets to all research experiments.
 * 
 * Logic hierarchy (checked in order):
 * 1. Explicit test characteristics (description keywords)
 * 2. Simulation type
 * 3. Tier number
 * 4. Fallback to sensible default
 * 
 * @param tierNum - Test tier (1-7)
 * @param testId - Test identifier (e.g., "REL-01", "GRAV-12")
 * @param description - Test description from config
 * @param simulation - Simulation type from experiment definition
 * @returns Preset ID
 */
export function inferVisualizationPreset(
  tierNum: number,
  testId: string,
  description: string,
  simulation: string
): string {
  const desc = description.toLowerCase();
  
  // Explicit orbital mechanics tests
  if (desc.includes('orbit') || desc.includes('gravitational') || desc.includes('binary')) {
    return 'orbital-showcase';
  }
  
  // Wave propagation tests
  if (desc.includes('wave') || desc.includes('packet') || desc.includes('propagation') ||
      desc.includes('isotropy') || desc.includes('causality') || desc.includes('phase velocity')) {
    return 'wave-dynamics';
  }
  
  // Field-only tests
  if ((desc.includes('field') || desc.includes('electromagnetic') || desc.includes('coupling')) &&
      !desc.includes('orbit')) {
    return 'field-only';
  }
  
  // Energy/thermodynamics tests (minimal for performance)
  if (desc.includes('energy') || desc.includes('conservation') || desc.includes('thermodynamic')) {
    return 'minimal';
  }
  
  // Fallback by simulation type
  if (simulation === 'binary-orbit' || simulation === 'n-body') {
    return 'orbital-minimal'; // Research tests use minimal variant by default
  }
  
  if (simulation === 'wave-packet') {
    return 'wave-dynamics';
  }
  
  if (simulation === 'field-dynamics') {
    return 'field-only';
  }
  
  // Fallback by tier
  switch (tierNum) {
    case 1: // Relativistic - wave propagation
      return 'wave-dynamics';
    
    case 2: // Gravity - orbital mechanics (but minimal for research)
      return 'orbital-minimal';
    
    case 3: // Energy - minimal visualization
      return 'minimal';
    
    case 4: // Quantization - field structure
      return 'field-only';
    
    case 5: // Electromagnetic - field visualization
      return 'field-only';
    
    case 6: // Coupling - field interactions
      return 'field-only';
    
    case 7: // Thermodynamics - minimal
      return 'minimal';
    
    default:
      // Ultimate fallback - wave dynamics is safest middle ground
      return 'wave-dynamics';
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get preset configuration by ID
 */
export function getPreset(presetId: string): VisualizationPreset | undefined {
  return VISUALIZATION_PRESETS[presetId];
}

/**
 * Get all available preset IDs
 */
export function getPresetIds(): string[] {
  return Object.keys(VISUALIZATION_PRESETS);
}

/**
 * Validate that a preset's default options are all available
 * 
 * Called during build to catch configuration errors
 */
export function validatePreset(preset: VisualizationPreset): boolean {
  return preset.defaultEnabled.every(opt => preset.availableOptions.includes(opt));
}

/**
 * Validate all presets
 * 
 * Returns array of validation errors (empty if all valid)
 */
export function validateAllPresets(): string[] {
  const errors: string[] = [];
  
  for (const [id, preset] of Object.entries(VISUALIZATION_PRESETS)) {
    if (!validatePreset(preset)) {
      errors.push(`Preset '${id}': defaultEnabled contains options not in availableOptions`);
    }
    
    if (preset.defaultEnabled.length === 0) {
      errors.push(`Preset '${id}': must have at least one default enabled option`);
    }
    
    if (preset.availableOptions.length === 0) {
      errors.push(`Preset '${id}': must have at least one available option`);
    }
  }
  
  return errors;
}

/**
 * Convert legacy visualization config to preset-based options
 * 
 * Used when migrating existing experiment definitions
 */
export function legacyConfigToOptions(legacyViz: Record<string, boolean>): {
  availableOptions: VisualizationOption[];
  enabledOptions: VisualizationOption[];
} {
  const availableOptions: VisualizationOption[] = [];
  const enabledOptions: VisualizationOption[] = [];
  
  for (const [legacyField, optionId] of Object.entries(LEGACY_FIELD_MAPPING)) {
    if (legacyField in legacyViz) {
      availableOptions.push(optionId);
      if (legacyViz[legacyField]) {
        enabledOptions.push(optionId);
      }
    }
  }
  
  return { availableOptions, enabledOptions };
}

/**
 * Convert preset-based options to legacy field config
 * 
 * Used when interfacing with existing simulation components
 */
export function optionsToLegacyConfig(
  availableOptions: VisualizationOption[],
  enabledOptions: VisualizationOption[]
): Record<string, boolean> {
  const legacyConfig: Record<string, boolean> = {};
  
  // Set all available options (enabled or disabled)
  for (const option of availableOptions) {
    const legacyField = OPTION_TO_LEGACY_FIELD[option];
    if (legacyField) {
      legacyConfig[legacyField] = enabledOptions.includes(option);
    }
  }
  
  return legacyConfig;
}

// ============================================================================
// Build-Time Validation
// ============================================================================

/**
 * Run all validations and throw if any fail
 * 
 * Called by generate_website_experiments.js to catch errors at build time
 */
export function runBuildTimeValidation(): void {
  const errors = validateAllPresets();
  
  if (errors.length > 0) {
    throw new Error(
      `Visualization preset validation failed:\n${errors.map(e => `  - ${e}`).join('\n')}`
    );
  }
  
  console.log('✓ Visualization presets validated successfully');
  console.log(`  ${getPresetIds().length} presets defined`);
  console.log(`  ${Object.keys(VISUALIZATION_LABELS).length} visualization options available`);
}

// Run validation when imported (catches errors at build time)
if (process.env.NODE_ENV !== 'production') {
  try {
    runBuildTimeValidation();
  } catch (error) {
    console.error(error);
    // Don't throw in development to avoid breaking hot reload
  }
}

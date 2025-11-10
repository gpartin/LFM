#!/usr/bin/env node
/**
 * © 2025 Emergent Physics Lab. All rights reserved.
 * 
 * Generate website experiment metadata from test harness configs
 * =============================================================
 * 
 * This is SOURCE TOOLING - lives in workspace/tools/, not website/
 * 
 * Purpose: Single source of truth sync
 * - Reads test harness configs (workspace/config/*.json)
 * - Generates TypeScript metadata for website
 * - Ensures website always reflects validated test parameters
 * 
 * Usage:
 *   node workspace/tools/generate_website_experiments.js
 *   
 * Called by:
 *   - Pre-commit validation (ensures sync before commit)
 *   - Website prebuild (regenerates before deploy)
 *   - Manual: npm run generate:experiments (from website/)
 * 
 * Reads:
 *   - workspace/config/config_tier*.json (canonical test definitions)
 *   - workspace/docs/discoveries/discoveries.json (metadata)
 * 
 * Writes:
 *   - workspace/website/src/data/research-experiments-generated.ts
 */

const fs = require('fs');
const path = require('path');

// Paths relative to workspace root
const SCRIPT_DIR = __dirname;  // workspace/tools
const WORKSPACE_ROOT = path.join(SCRIPT_DIR, '..');
const CONFIG_DIR = path.join(WORKSPACE_ROOT, 'config');
const DISCOVERIES_PATH = path.join(WORKSPACE_ROOT, 'docs', 'discoveries', 'discoveries.json');
const OUTPUT_PATH = path.join(WORKSPACE_ROOT, 'website', 'src', 'data', 'research-experiments-generated.ts');

// Tier definitions (from workspace/src/harness/lfm_tiers.py)
const TIERS = [
  { tier: 1, name: 'Relativistic', prefix: 'REL', config: 'config_tier1_relativistic.json', expected: 16 },
  { tier: 2, name: 'Gravity', prefix: 'GRAV', config: 'config_tier2_gravityanalogue.json', expected: 26 },
  { tier: 3, name: 'Energy', prefix: 'ENER', config: 'config_tier3_energy.json', expected: 11 },
  { tier: 4, name: 'Quantization', prefix: 'QUAN', config: 'config_tier4_quantization.json', expected: 9 },
  { tier: 5, name: 'Electromagnetic', prefix: 'EM', config: 'config_tier5_electromagnetic.json', expected: 21 },
  { tier: 6, name: 'Coupling', prefix: 'COUP', config: 'config_tier6_coupling.json', expected: 12 },
  { tier: 7, name: 'Thermodynamics', prefix: 'THERM', config: 'config_tier7_thermodynamics.json', expected: 5 },
];

/**
 * Read JSON file with UTF-8 encoding
 */
function readJSON(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch (e) {
    console.warn(`Warning: Could not read ${filePath}: ${e.message}`);
    return null;
  }
}

/**
 * Load discoveries metadata
 */
function loadDiscoveries() {
  return readJSON(DISCOVERIES_PATH) || [];
}

/**
 * Infer simulation type from test characteristics
 */
function inferSimulationType(tierNum, testId, description) {
  const desc = description.toLowerCase();
  
  if (desc.includes('orbit') || desc.includes('gravitational')) return 'binary-orbit';
  if (desc.includes('wave') || desc.includes('packet')) return 'wave-packet';
  if (desc.includes('field') || desc.includes('chi')) return 'field-dynamics';
  
  // Default by tier
  if (tierNum <= 2) return 'wave-packet';  // Relativistic, Gravity
  if (tierNum === 3) return 'wave-packet';  // Energy conservation
  if (tierNum === 4) return 'field-dynamics'; // Quantization
  if (tierNum >= 5) return 'field-dynamics'; // EM, Coupling, Thermo
  
  return 'wave-packet';
}

/**
 * Generate experiment entry from test config
 * 
 * CRITICAL: This must exactly mirror test harness parameters
 * Any mismatch will be caught by validate_website_sync.py
 */
function generateExperimentEntry(tierInfo, test, baseParams) {
  // Handle both "test_id" (Tier 1-3) and "id" (Tier 4-7) field names
  const testId = test.test_id || test.id || 'UNKNOWN';
  const description = test.description || test.name || 'No description';
  const simType = inferSimulationType(tierInfo.tier, testId, description);
  
  // Validate required fields
  if (!testId || testId === 'UNKNOWN') {
    console.warn(`Warning: Test missing test_id/id in tier ${tierInfo.tier}:`, JSON.stringify(test));
    return null; // Skip invalid entries
  }
  
  const entry = {
    id: testId,
    testId: testId,
    displayName: `${testId}: ${description}`,
    type: 'RESEARCH',
    tier: tierInfo.tier,
    tierName: tierInfo.name,
    category: tierInfo.name,
    tagline: description,
    description: `Research validation test for ${tierInfo.name.toLowerCase()} tier. ${description}`,
    difficulty: 'intermediate',
    simulation: simType,
    backend: 'both',
    initialConditions: {
      latticeSize: baseParams.grid_points || baseParams.N || 256,
      dt: baseParams.dt || baseParams.time_step || 0.001,
      dx: baseParams.dx || baseParams.space_step || 0.01,
      steps: test.steps || baseParams.steps || 5000,
      chi: test.chi_const !== undefined ? test.chi_const : (baseParams.chi || 0.0),
    },
    validation: {
      energyDrift: baseParams.tolerances?.energy_drift || 1e-6,
    },
    visualization: {
      showParticles: false,
      showTrails: false,
      showChi: simType === 'field-dynamics',
      showLattice: true,
      showVectors: false,
      showWell: false,
      showDomes: false,
      showIsoShells: false,
      showBackground: false,
    },
    links: {
      testHarnessConfig: `workspace/config/${tierInfo.config}`,
      results: `workspace/results/${tierInfo.name}/${testId}/`,
      discovery: `Tier ${tierInfo.tier} - ${tierInfo.name}`,
    },
    status: test.skip ? 'development' : 'production',
  };
  
  // Add validation tolerances if present
  if (test.tolerances) {
    entry.validation = { ...entry.validation, ...test.tolerances };
  }
  
  return entry;
}

/**
 * Process single tier config file
 */
function processTierConfig(tierInfo) {
  const configPath = path.join(CONFIG_DIR, tierInfo.config);
  const config = readJSON(configPath);
  
  if (!config) {
    console.warn(`Skipping tier ${tierInfo.tier} - config not found`);
    return [];
  }
  
  const baseParams = {
    ...config.parameters,
    tolerances: config.tolerances,
    grid_points: config.parameters?.grid_points,
    N: config.parameters?.N,
  };
  
  // Handle both "variants" and "tests" schema
  const tests = config.variants || config.tests || [];
  
  // Map and filter out null entries (invalid tests)
  return tests.map(test => generateExperimentEntry(tierInfo, test, baseParams))
              .filter(entry => entry !== null);
}

/**
 * Generate TypeScript file content
 */
function generateTypeScriptFile(allExperiments) {
  const header = `/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * ⚠️ AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
 * 
 * Generated from test harness configs: ${new Date().toISOString()}
 * Source: workspace/tools/generate_website_experiments.js
 * 
 * This file is regenerated:
 * - Before every website build (prebuild hook)
 * - By pre-commit validation (ensures sync)
 * 
 * Do NOT edit this file directly. Edit test harness configs instead:
 * - workspace/config/config_tier*.json
 */

import { ExperimentDefinition } from './experiments';

export const RESEARCH_EXPERIMENTS: ExperimentDefinition[] = [
`;

  const entries = allExperiments.map(exp => 
    '  ' + JSON.stringify(exp, null, 2).replace(/\n/g, '\n  ')
  ).join(',\n');

  const footer = `
];

// Summary (logged at import time for debugging)
if (typeof window === 'undefined') {
  console.log(\`Loaded \${RESEARCH_EXPERIMENTS.length} research experiments:\`);
  const byTier = RESEARCH_EXPERIMENTS.reduce((acc, exp) => {
    const tier = exp.tier || 0;
    acc[tier] = (acc[tier] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);
  Object.entries(byTier).forEach(([tier, count]) => {
    console.log(\`  Tier \${tier}: \${count} tests\`);
  });
}
`;

  return header + entries + footer;
}

/**
 * Main execution
 */
function main() {
  console.log('🔬 Generating website experiment metadata from test harness configs...\n');
  console.log(`   Source: ${CONFIG_DIR}`);
  console.log(`   Output: ${OUTPUT_PATH}\n`);
  
  const discoveries = loadDiscoveries();
  console.log(`📋 Loaded ${discoveries.length} discoveries\n`);
  
  let allExperiments = [];
  let totalExpected = 0;
  
  for (const tierInfo of TIERS) {
    console.log(`Processing Tier ${tierInfo.tier} (${tierInfo.name})...`);
    const experiments = processTierConfig(tierInfo);
    allExperiments = allExperiments.concat(experiments);
    totalExpected += tierInfo.expected;
    console.log(`  Generated ${experiments.length} / ${tierInfo.expected} expected tests\n`);
  }
  
  // Generate TypeScript file
  const tsContent = generateTypeScriptFile(allExperiments);
  
  // Ensure output directory exists
  const outputDir = path.dirname(OUTPUT_PATH);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  fs.writeFileSync(OUTPUT_PATH, tsContent, 'utf-8');
  
  console.log(`✅ Generated ${allExperiments.length} research experiments`);
  console.log(`   Expected: ${totalExpected} total tests`);
  console.log(`   Output: ${OUTPUT_PATH}\n`);
  
  // Summary by tier
  console.log('Summary by tier:');
  const byTier = {};
  allExperiments.forEach(exp => {
    byTier[exp.tier] = (byTier[exp.tier] || 0) + 1;
  });
  Object.entries(byTier).forEach(([tier, count]) => {
    const tierInfo = TIERS.find(t => t.tier === parseInt(tier));
    console.log(`  Tier ${tier} (${tierInfo?.name}): ${count} / ${tierInfo?.expected} tests`);
  });
  
  return 0;
}

// Run if executed directly
if (require.main === module) {
  try {
    process.exit(main());
  } catch (error) {
    console.error('❌ Error generating experiments:', error);
    process.exit(1);
  }
}

module.exports = { main, generateExperimentEntry, processTierConfig, TIERS };

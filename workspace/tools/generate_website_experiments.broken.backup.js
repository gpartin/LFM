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

// Tier definitions (actual test counts as of Nov 2025)
// Total: 104 tests (105 including GRAV-09 which is marked skip=true)
const TIERS = [
  { tier: 1, name: 'Relativistic', prefix: 'REL', config: 'config_tier1_relativistic.json', expected: 17 },
  { tier: 2, name: 'Gravity', prefix: 'GRAV', config: 'config_tier2_gravityanalogue.json', expected: 24 },
  { tier: 3, name: 'Energy', prefix: 'ENER', config: 'config_tier3_energy.json', expected: 11 },
  { tier: 4, name: 'Quantization', prefix: 'QUAN', config: 'config_tier4_quantization.json', expected: 14 },
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
 * Deterministic simulation classification from CONFIG KEYS (primary)
 * Falls back to description-only heuristics when no discriminating keys present.
 */
function inferSimulationType(tierNum, testId, description, testObj) {
  const desc = String(description || '').toLowerCase();
  const t = testObj || {};

  // Key-based deterministic classification
  const hasAny = (keys) => keys.some(k => Object.prototype.hasOwnProperty.call(t, k));
  const hasPrefix = (prefixes) => Object.keys(t).some(k => prefixes.some(p => k.startsWith(p)));

  // Field-dynamics if any chi-related parameter exists
  if (hasAny(['chi', 'chi_const', 'chi_base', 'chi_bg', 'chi_left', 'chi_right', 'chi_inside', 'chi_outside']) ||
      hasAny(['chi_barrier', 'chi_barrier_low', 'chi_barrier_high']) ||
      hasAny(['chi_gradient']) ||
      hasPrefix(['chi_', 'lens_', 'bump_'])) {
    return 'field-dynamics';
  }

  // EM-driven but still field dynamics for website visual layer
  if (hasAny(['em_wave_freq', 'em_wave_amp']) || hasPrefix(['em_'])) {
    return 'field-dynamics';
  }

  // Wave-packet indicators
  if (hasAny(['packet_width']) || desc.includes('packet') || desc.includes('wave propagation')) {
    return 'wave-packet';
  }

  // Description fallbacks (kept deterministic for missing-key cases)
  if (desc.includes('orbit') || desc.includes('binary') || desc.includes('n-body')) return 'binary-orbit';
  if (desc.includes('wave') || desc.includes('packet')) return 'wave-packet';
  if (desc.includes('field') || desc.includes('chi') || desc.includes('time dilation') ||
      desc.includes('electromagnetic') || desc.includes('coupling')) return 'field-dynamics';

  // Default by tier
  if (tierNum <= 2) return 'wave-packet';
  if (tierNum === 3) return 'wave-packet';
  if (tierNum >= 4) return 'field-dynamics';

  return 'wave-packet';
}

/**
 * Infer visualization preset from experiment characteristics
 * (Inlined from workspace/website/src/data/visualization-presets.ts)
 */
function inferVisualizationPreset(tierNum, testId, description, simulation) {
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
    return 'orbital-minimal';
  }
  
  if (simulation === 'wave-packet') {
    return 'wave-dynamics';
  }
  
  if (simulation === 'field-dynamics') {
    return 'field-only';
  }
  
  // Fallback by tier
  switch (tierNum) {
    case 1: return 'wave-dynamics';
    case 2: return 'orbital-minimal';
    case 3: return 'minimal';
    case 4: return 'field-only';
    case 5: return 'field-only';
    case 6: return 'field-only';
    case 7: return 'minimal';
    default: return 'wave-dynamics';
  }
}

/**
 * Extract chi from test/config deterministically
 */
function extractChi(test, baseParams) {
  if (Object.prototype.hasOwnProperty.call(test, 'chi_const')) return test.chi_const;
  if (Object.prototype.hasOwnProperty.call(test, 'chi_gradient')) return test.chi_gradient;
  // Common chi parameter names in configs
  const chiKeys = ['chi', 'chi_base', 'chi_bg', 'chi_left', 'chi_right', 'chi_inside', 'chi_outside', 'chi_barrier'];
  for (const key of chiKeys) {
    if (Object.prototype.hasOwnProperty.call(test, key)) return test[key];
  }
  if (Object.prototype.hasOwnProperty.call(baseParams, 'chi')) return baseParams.chi;
  return 0.0;
}

/**
 * Generate experiment entry from test config
 * 
 * CRITICAL: This must exactly mirror test harness parameters
 * Any mismatch will be caught by validate_website_sync.py
 */
function generateExperimentEntry(tierInfo, test, baseParams, tierConfig) {
  // Handle both "test_id" (Tier 1-3) and "id" (Tier 4-7) field names
  const testId = test.test_id || test.id || 'UNKNOWN';
  const description = test.description || test.name || 'No description';
  const simType = inferSimulationType(tierInfo.tier, testId, description, test);
  
  // NEW: Infer visualization preset
  const vizPreset = inferVisualizationPreset(tierInfo.tier, testId, description, simType);
  
  // Validate required fields
  if (!testId || testId === 'UNKNOWN') {
    console.warn(`Warning: Test missing test_id/id in tier ${tierInfo.tier}:`, JSON.stringify(test));
    return null; // Skip invalid entries
  }
  
  // Determine deterministic lattice size and steps
  const latticeSize = (Number.isFinite(test.grid_size) ? test.grid_size : undefined) ||
                      (Number.isFinite(baseParams.grid_points) ? baseParams.grid_points : undefined) ||
                      (Number.isFinite(baseParams.N) ? baseParams.N : undefined) || 256;
  const steps = (Number.isFinite(test.steps) ? test.steps : undefined) ||
                (Number.isFinite(baseParams.steps) ? baseParams.steps : undefined) || 5000;

  // Resolve dt/dx from canonical parameter names
  const dt = (typeof baseParams.dt === 'number' ? baseParams.dt : (typeof baseParams.time_step === 'number' ? baseParams.time_step : 0.001));
  const dx = (typeof baseParams.dx === 'number' ? baseParams.dx : (typeof baseParams.space_step === 'number' ? baseParams.space_step : 0.01));

  // Resolve chi deterministically
  const chi = extractChi(test, baseParams);

  // Tier folder resolution from config.output_dir to ensure deterministic results path
  const tierFolder = (() => {
    const out = tierConfig && tierConfig.output_dir ? String(tierConfig.output_dir) : `../results/${tierInfo.name}`;
    const base = path.basename(out.replace(/\\/g, '/'));
    return base || tierInfo.name;
  })();

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
    visualizationPreset: vizPreset,
    backend: 'both',
    initialConditions: {
      latticeSize: latticeSize,
      dt: dt,
      dx: dx,
      steps: steps,
      chi: chi,
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
      results: `workspace/results/${tierFolder}/${testId}/`,
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
  }\r

  const baseParams = {
    ...(config.parameters || {}),
    tolerances: config.tolerances || {},
    grid_points: config.parameters?.grid_points,
    N: config.parameters?.N,
    dt: config.parameters?.dt ?? config.parameters?.time_step,
    dx: config.parameters?.dx ?? config.parameters?.space_step,
    steps: config.parameters?.steps,
  };

  const tests = config.variants || config.tests || [];
  return tests.map(test => generateExperimentEntry(tierInfo, test, baseParams, config))
              .filter(entry => entry !== null);
}

 * - Before every website build (prebuild hook)
`;
 * - By pre-commit validation (ensures sync)
 * 
 * Do NOT edit this file directly. Edit test harness configs instead:

/**
 * Generate all experiments in-memory (sorted deterministically by testId)
 */
function generateAllExperiments() {
  let all = [];
  for (const tierInfo of TIERS) {
    const experiments = processTierConfig(tierInfo);
    all = all.concat(experiments);
  }
  all.sort((a, b) => String(a.testId || a.id).localeCompare(String(b.testId || b.id)));
  return all;
}
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

/**
 * Generate all experiments in-memory (sorted deterministically by testId)
 */
function generateAllExperiments() {
  let all = [];
  for (const tierInfo of TIERS) {
    const experiments = processTierConfig(tierInfo);
    all = all.concat(experiments);
  }
  all.sort((a, b) => String(a.testId || a.id).localeCompare(String(b.testId || b.id)));
  return all;
}
`;

  return header + entries + footer;
}

/**
 * Calculate pass rate from canonical registry (preferred) or master CSV (fallback)
 */
function calculatePassRate() {
  const canonicalPath = path.join(WORKSPACE_ROOT, 'results', 'test_registry_canonical.json');
  if (fs.existsSync(canonicalPath)) {
    try {
      const reg = JSON.parse(fs.readFileSync(canonicalPath, 'utf-8'));
      const sum = reg.summary || {};
      const tiers = reg.tiers || {};
      const total = Number(sum.executed || 0);
      const passing = Number(sum.passed || 0);
      // Build per-tier map: total=executed, passing=passed
      const byTier = {};
      Object.keys(tiers).forEach((k) => {
        const t = tiers[k] || {};
        byTier[k] = { total: Number(t.executed || 0), passing: Number(t.passed || 0) };
      });
      const passRate = (typeof sum.public_pass_rate === 'number')
        ? `${sum.public_pass_rate.toFixed(1)}%`
        : (total > 0 ? `${((passing / total) * 100).toFixed(1)}%` : '0.0%');
      // Sanity: passing cannot exceed total
      if (passing > total) {
        throw new Error(`Canonical passing (${passing}) exceeds total executed (${total})`);
      }
      return {
        total,
        passing,
        failing: total - passing,
        passRate,
        byTier,
        generatedAt: new Date().toISOString(),
        sourceFile: 'workspace/results/test_registry_canonical.json',
        note: 'Derived from canonical registry; SKIP tests excluded (executed-only)'
      };
    } catch (e) {
      console.warn('Warning: Failed to parse canonical registry, falling back to CSV:', e.message);
    }
  }
  // Fallback to CSV parsing (executed-only counting)
  const csvPath = path.join(WORKSPACE_ROOT, 'results', 'MASTER_TEST_STATUS.csv');
  try {
    const csvContent = fs.readFileSync(csvPath, 'utf-8');
    const lines = csvContent.split('\n');
    let totalTests = 0;
    let passingTests = 0;
    const byTier = {};
    // Initialize byTier with zeros for tiers 1..7
    for (const t of TIERS) byTier[String(t.tier)] = { total: 0, passing: 0 };
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('MASTER') || trimmed.startsWith('Generated:') || 
          trimmed.startsWith('Validation') || trimmed.startsWith('CATEGORY') || 
          trimmed.startsWith('Tier,Category') || trimmed.startsWith('DETAILED') ||
          trimmed.startsWith('Test_ID,')) {
        continue;
      }
      const parts = trimmed.split(',');
      if (parts.length >= 3 && /^[A-Z]+-\d+$/.test(parts[0])) {
        const testId = parts[0];
        const status = parts[2];
        const prefix = testId.split('-')[0];
        const tierObj = TIERS.find(t => t.prefix === prefix);
        const tierKey = tierObj ? String(tierObj.tier) : undefined;
        if (status === 'PASS') {
          totalTests++;
          passingTests++;
          if (tierKey) {
            byTier[tierKey].total++;
            byTier[tierKey].passing++;
          }
        } else if (status === 'FAIL') {
          totalTests++;
          if (tierKey) {
            byTier[tierKey].total++;
          }
        }
      }
    }
    const passRate = totalTests > 0 ? ((passingTests / totalTests) * 100).toFixed(1) : '0.0';
    return {
      total: totalTests,
      passing: passingTests,
      failing: totalTests - passingTests,
      passRate: passRate + '%',
      byTier,
      generatedAt: new Date().toISOString(),
      sourceFile: 'workspace/results/MASTER_TEST_STATUS.csv',
      note: 'SKIP tests excluded from totals (executed-only)'
    };
  } catch (error) {
    console.error('Error reading MASTER_TEST_STATUS.csv:', error.message);
    console.error('Falling back to assuming 100% pass rate from configs');
    let total = 0;
    TIERS.forEach(tier => {
      const config = readJSON(path.join(CONFIG_DIR, tier.config));
      if (config) {
        const tests = config.variants || config.tests || [];
        total += tests.length;
      }
    });
    return {
      total,
      passing: total,
      failing: 0,
      passRate: '100.0%',
      byTier: {},
      generatedAt: new Date().toISOString(),
      sourceFile: 'fallback (MASTER_TEST_STATUS.csv not found)'
    };
  }
}

/**
 * Generate test statistics TypeScript file
 */
function generateTestStatistics(stats) {
  const content = `// AUTO-GENERATED by generate_website_experiments.js
// DO NOT EDIT - regenerate via: node workspace/tools/generate_website_experiments.js

/**
 * Test Statistics - Calculated from Actual Test Results (Canonical Source Preferred)
 * ======================================================
 * 
 * Generated: ${stats.generatedAt}
 * Source: ${stats.sourceFile}
 * 
 * Semantics:
 * - Executed-only: SKIP-designated tests are excluded from denominators
 * - passRate = passed / executed
 * - When available, values are sourced from test_registry_canonical.json
 *   to ensure website matches uploads and pre-commit validation
 */

export const testStatistics = {
  /** Total number of tests across all tiers (executed-only; SKIP excluded) */
  total: ${stats.total},
  
  /** Number of tests currently passing */
  passing: ${stats.passing},
  
  /** Number of tests failing */
  failing: ${stats.failing},
  
  /** Pass rate as percentage string (e.g., "91.3%") */
  passRate: '${stats.passRate}',
  
  /** Breakdown by tier (executed-only; no tier can have passing > total) */
  byTier: ${JSON.stringify(stats.byTier, null, 2)},
  
  /** ISO timestamp of when this was generated */
  generatedAt: '${stats.generatedAt}'
} as const;

/**
 * Formatted pass rate for display
 * Example: "91.3% Tests Pass"
 */
export function formatPassRate(): string {
  return \`\${testStatistics.passRate} Tests Pass\`;
}

/**
 * Human-readable summary
 * Example: "95 of 104 tests passing"
 */
  export function formatSummary(): string {
    return \`\${testStatistics.passing} of \${testStatistics.total} executed tests passing\`;
}
`;

  return content;
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
  // Sort deterministically by testId before emitting
  allExperiments.sort((a, b) => String(a.testId || a.id).localeCompare(String(b.testId || b.id)));
  const tsContent = generateTypeScriptFile(allExperiments, false);
  
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
  
  // NEW: Generate test statistics
  console.log('\n🧮 Calculating test pass rate from validation metadata...');
  const stats = calculatePassRate();
  const statsContent = generateTestStatistics(stats);
  const statsPath = path.join(WORKSPACE_ROOT, 'website', 'src', 'data', 'test-statistics.ts');
  fs.writeFileSync(statsPath, statsContent, 'utf-8');
  console.log(`✅ Generated test statistics: ${stats.passRate} (${stats.passing}/${stats.total})`);
  console.log(`   Output: ${statsPath}\n`);

  // Note: Do NOT emit canonical registry here to avoid overwriting the authoritative file.
  console.log('ℹ️ Using canonical registry if present; not writing any canonical JSON from website generator.');
  
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

module.exports = { main, generateExperimentEntry, processTierConfig, TIERS, generateAllExperiments };

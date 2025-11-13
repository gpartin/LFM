#!/usr/bin/env node
/**
 * Deterministic experiment sync API for tests
 * Produces research experiment definitions from configs without writing files.
 */
const fs = require('fs');
const path = require('path');

const SCRIPT_DIR = __dirname;  // workspace/tools
const WORKSPACE_ROOT = path.join(SCRIPT_DIR, '..');
const CONFIG_DIR = path.join(WORKSPACE_ROOT, 'config');

const TIERS = [
  { tier: 1, name: 'Relativistic', prefix: 'REL', config: 'config_tier1_relativistic.json' },
  { tier: 2, name: 'Gravity', prefix: 'GRAV', config: 'config_tier2_gravityanalogue.json' },
  { tier: 3, name: 'Energy', prefix: 'ENER', config: 'config_tier3_energy.json' },
  { tier: 4, name: 'Quantization', prefix: 'QUAN', config: 'config_tier4_quantization.json' },
  { tier: 5, name: 'Electromagnetic', prefix: 'EM', config: 'config_tier5_electromagnetic.json' },
  { tier: 6, name: 'Coupling', prefix: 'COUP', config: 'config_tier6_coupling.json' },
  { tier: 7, name: 'Thermodynamics', prefix: 'THERM', config: 'config_tier7_thermodynamics.json' },
];

function readJSON(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

function inferSimulation(test, tierInfo, description) {
  const t = test || {};
  const desc = String(description || '').toLowerCase();
  const hasAny = (keys) => keys.some(k => Object.prototype.hasOwnProperty.call(t, k));
  const hasPrefix = (prefixes) => Object.keys(t).some(k => prefixes.some(p => k.startsWith(p)));
  if (hasAny(['chi', 'chi_const', 'chi_base', 'chi_bg', 'chi_left', 'chi_right', 'chi_inside', 'chi_outside']) ||
      hasAny(['chi_barrier', 'chi_barrier_low', 'chi_barrier_high']) ||
      hasAny(['chi_gradient']) || hasPrefix(['chi_', 'lens_', 'bump_', 'em_'])) return 'field-dynamics';
  if (hasAny(['packet_width']) || desc.includes('packet')) return 'wave-packet';
  if (desc.includes('orbit') || desc.includes('binary') || desc.includes('n-body')) return 'binary-orbit';
  if (tierInfo.tier <= 3) return 'wave-packet';
  return 'field-dynamics';
}

function extractChi(test, baseParams) {
  if (Object.prototype.hasOwnProperty.call(test, 'chi_const')) return test.chi_const;
  if (Object.prototype.hasOwnProperty.call(test, 'chi_gradient')) return test.chi_gradient;
  const chiKeys = ['chi', 'chi_base', 'chi_bg', 'chi_left', 'chi_right', 'chi_inside', 'chi_outside', 'chi_barrier'];
  for (const key of chiKeys) {
    if (Object.prototype.hasOwnProperty.call(test, key)) return test[key];
  }
  if (Object.prototype.hasOwnProperty.call(baseParams, 'chi')) return baseParams.chi;
  return 0.0;
}

function generateExperimentEntry(tierInfo, test, baseParams, tierConfig) {
  const testId = test.test_id || test.id || 'UNKNOWN';
  const description = test.description || test.name || '';
  const simType = inferSimulation(test, tierInfo, description);

  const latticeSize = (Number.isFinite(test.grid_size) ? test.grid_size : undefined) ||
                      (Number.isFinite(baseParams.grid_points) ? baseParams.grid_points : undefined) ||
                      (Number.isFinite(baseParams.N) ? baseParams.N : undefined) || 256;
  const steps = (Number.isFinite(test.steps) ? test.steps : undefined) ||
                (Number.isFinite(baseParams.steps) ? baseParams.steps : undefined) || 5000;
  const dt = (typeof baseParams.dt === 'number' ? baseParams.dt : (typeof baseParams.time_step === 'number' ? baseParams.time_step : 0.001));
  const dx = (typeof baseParams.dx === 'number' ? baseParams.dx : (typeof baseParams.space_step === 'number' ? baseParams.space_step : 0.01));
  const chi = extractChi(test, baseParams);
  const tierFolder = (() => {
    const out = tierConfig && tierConfig.output_dir ? String(tierConfig.output_dir) : `../results/${tierInfo.name}`;
    const base = out.replace(/\\/g, '/').split('/').pop();
    return base || tierInfo.name;
  })();

  return {
    id: testId,
    testId,
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
    initialConditions: { latticeSize, dt, dx, steps, chi },
    validation: { energyDrift: baseParams.tolerances?.energy_drift || 1e-6 },
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
}

function processTierConfig(tierInfo) {
  const configPath = path.join(CONFIG_DIR, tierInfo.config);
  const config = readJSON(configPath);
  if (!config) return [];
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

function generateAllExperiments() {
  let all = [];
  for (const tierInfo of TIERS) {
    all = all.concat(processTierConfig(tierInfo));
  }
  all.sort((a, b) => String(a.testId || a.id).localeCompare(String(b.testId || b.id)));
  return all;
}

module.exports = { generateAllExperiments, TIERS };

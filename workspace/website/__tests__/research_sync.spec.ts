/* -*- coding: utf-8 -*- */
/**
 * @jest-environment node
 */
/**
 * Deterministic Research Experiment Generation Tests
 * Ensures website experiments mirror the test harness configs exactly.
 */

import * as fs from 'fs';
import * as path from 'path';
import crypto from 'crypto';

// Import generator tooling from workspace/tools
// eslint-disable-next-line @typescript-eslint/no-var-requires
const gen = require('../../tools/experiment_sync_api.js');

function loadJSON(p: string) {
  return JSON.parse(fs.readFileSync(p, 'utf-8'));
}

function stableHash(obj: unknown): string {
  const json = JSON.stringify(obj);
  return crypto.createHash('sha256').update(json).digest('hex');
}

describe('Research experiments generation is deterministic and config-driven', () => {
  const WORKSPACE_ROOT = path.join(__dirname, '..', '..');
  const CONFIG_DIR = path.join(WORKSPACE_ROOT, 'config');

  test('COUP-01 parameters match config_tier6_coupling.json', () => {
    const experiments = gen.generateAllExperiments();
    const coup01 = experiments.find((e: any) => e.testId === 'COUP-01');
    expect(coup01).toBeTruthy();

    const cfg = loadJSON(path.join(CONFIG_DIR, 'config_tier6_coupling.json'));
    const test = (cfg.tests as any[]).find(t => t.test_id === 'COUP-01');
    expect(test).toBeTruthy();

    // Base params from config.parameters
    const params = cfg.parameters || {};

    // Deterministic mappings
    expect(coup01.initialConditions.dt).toBeCloseTo(params.time_step ?? params.dt, 12);
    expect(coup01.initialConditions.dx).toBeCloseTo(params.space_step ?? params.dx, 12);
    expect(coup01.initialConditions.latticeSize).toBe(test.grid_size ?? params.N ?? params.grid_points);
    expect(coup01.initialConditions.steps).toBe(test.steps ?? params.steps);

    // Chi mapping: COUP-01 uses chi_gradient [min, max]
    expect(Array.isArray(coup01.initialConditions.chi)).toBe(true);
    expect(coup01.initialConditions.chi[0]).toBe(test.chi_gradient[0]);
    expect(coup01.initialConditions.chi[1]).toBe(test.chi_gradient[1]);

    // Results path must point to Coupling tier folder
    expect(coup01.links.results).toContain('/results/Coupling/COUP-01/');
    expect(coup01.links.testHarnessConfig.endsWith('config_tier6_coupling.json')).toBe(true);
  });

  test('Full experiment set is stable across runs (order and content)', () => {
    const a = gen.generateAllExperiments();
    const b = gen.generateAllExperiments();
    expect(a.length).toBeGreaterThan(0);
    expect(a).toEqual(b);

    // Stable hash of stringified content
    const hashA = stableHash(a);
    const hashB = stableHash(b);
    expect(hashA).toEqual(hashB);
  });
});

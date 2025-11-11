/* -*- coding: utf-8 -*- */
/**
 * Regression: Website testStatistics must match canonical registry (executed-only semantics)
 * - Source of truth: workspace/results/test_registry_canonical.json
 * - Guards against drift between uploads and website data generation
 */

import fs from 'fs';
import path from 'path';
import { testStatistics } from '@/data/test-statistics';

type CanonicalTier = {
  executed: number;
  passed: number;
  failed: number;
  skipped: number;
  pass_rate: number;
  tests: Array<{ id: string; status: string; skip_exempt: boolean }>;
};

type Canonical = {
  summary: {
    executed: number;
    passed: number;
    failed: number;
    public_pass_rate: number;
  };
  tiers: Record<string, CanonicalTier>;
};

function loadCanonical(): Canonical {
  const canonicalPath = path.resolve(__dirname, '../../../results/test_registry_canonical.json');
  const raw = fs.readFileSync(canonicalPath, { encoding: 'utf-8' });
  return JSON.parse(raw);
}

function toPct(passed: number, executed: number): string {
  if (executed === 0) return '0.0%';
  return `${((passed / executed) * 100).toFixed(1)}%`;
}

describe('Canonical statistics consistency', () => {
  const canonical = loadCanonical();

  it('matches totals and pass rate (executed-only)', () => {
    expect(testStatistics.total).toBe(canonical.summary.executed);
    expect(testStatistics.passing).toBe(canonical.summary.passed);
    expect(testStatistics.failing).toBe(canonical.summary.failed);
    expect(testStatistics.passRate).toBe(toPct(canonical.summary.passed, canonical.summary.executed));
  });

  it('matches per-tier executed and passed counts', () => {
    const tiers = Object.keys(canonical.tiers);
    for (const tier of tiers) {
      // Ensure the website exposes this tier
      expect(testStatistics.byTier).toHaveProperty(tier);
      const wTier = (testStatistics.byTier as any)[tier];
      const cTier = canonical.tiers[tier];

      expect(wTier.total).toBe(cTier.executed);
      expect(wTier.passing).toBe(cTier.passed);
      expect(wTier.passing).toBeLessThanOrEqual(wTier.total);
    }
  });
});

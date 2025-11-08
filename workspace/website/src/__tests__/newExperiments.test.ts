/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Basic smoke tests for new experiments
 */

import { describe, it, expect } from '@jest/globals';

describe('New Experiments', () => {
  describe('Three-Body', () => {
    it('module exports default factory', async () => {
      const module = await import('@/experiments/orbital-mechanics/three-body');
      expect(module.default).toBeDefined();
      expect(typeof module.default).toBe('function');
    });
  });

  describe('Big Bang', () => {
    it('module exports default factory', async () => {
      const module = await import('@/experiments/cosmology/big-bang');
      expect(module.default).toBeDefined();
      expect(typeof module.default).toBe('function');
    });
  });

  describe('Stellar Collapse', () => {
    it('module exports default factory', async () => {
      const module = await import('@/experiments/gravity/stellar-collapse');
      expect(module.default).toBeDefined();
      expect(typeof module.default).toBe('function');
    });
  });
});

describe('Experiment Registry', () => {
  it('includes all 5 experiments', async () => {
    const { getAllExperiments } = await import('@/lib/experimentRegistry');
    const experiments = getAllExperiments();
    expect(experiments).toHaveLength(5);
    
    const ids = experiments.map(e => e.metadata.id);
    expect(ids).toContain('binary-orbit');
    expect(ids).toContain('black-hole');
    expect(ids).toContain('three-body');
    expect(ids).toContain('big-bang');
    expect(ids).toContain('stellar-collapse');
  });

  it('includes cosmology category', async () => {
    const { getAllCategories } = await import('@/lib/experimentRegistry');
    const categories = getAllCategories();
    expect(categories).toContain('cosmology');
  });

  it('big-bang is in cosmology category', async () => {
    const { getExperimentsByCategory } = await import('@/lib/experimentRegistry');
    const cosmologyExps = getExperimentsByCategory('cosmology');
    expect(cosmologyExps).toHaveLength(1);
    expect(cosmologyExps[0].metadata.id).toBe('big-bang');
  });
});

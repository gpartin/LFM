/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Unit tests for Experiment Registry
 * 
 * Tests search, filter, and registry functions
 */

import {
  getAllExperiments,
  getExperimentById,
  getExperimentsByCategory,
  getFeaturedExperiments,
  searchExperiments,
  filterExperiments,
  getAllCategories,
  getAllTags,
  getCategoryStats,
} from '@/lib/experimentRegistry';

describe('Experiment Registry', () => {
  describe('getAllExperiments', () => {
    it('should return an array of experiments', () => {
      const experiments = getAllExperiments();
      expect(Array.isArray(experiments)).toBe(true);
      expect(experiments.length).toBeGreaterThan(0);
    });

    it('should return experiments with required metadata fields', () => {
      const experiments = getAllExperiments();
      experiments.forEach((exp) => {
        expect(exp.metadata).toBeDefined();
        expect(exp.metadata.id).toBeDefined();
        expect(exp.metadata.title).toBeDefined();
        expect(exp.metadata.category).toBeDefined();
        expect(exp.metadata.difficulty).toBeDefined();
        expect(exp.loader).toBeDefined();
        expect(typeof exp.loader).toBe('function');
      });
    });
  });

  describe('getExperimentById', () => {
    it('should return the correct experiment by ID', () => {
      const exp = getExperimentById('binary-orbit');
      expect(exp).toBeDefined();
      expect(exp?.metadata.id).toBe('binary-orbit');
      expect(exp?.metadata.title).toBe('Earth-Moon Orbit');
    });

    it('should return undefined for non-existent ID', () => {
      const exp = getExperimentById('non-existent-experiment');
      expect(exp).toBeUndefined();
    });
  });

  describe('getExperimentsByCategory', () => {
    it('should return experiments in the specified category', () => {
      const orbitalExps = getExperimentsByCategory('orbital-mechanics');
      expect(Array.isArray(orbitalExps)).toBe(true);
      orbitalExps.forEach((exp) => {
        expect(exp.metadata.category).toBe('orbital-mechanics');
      });
    });

    it('should return gravity experiments', () => {
      const gravityExps = getExperimentsByCategory('gravity');
      expect(Array.isArray(gravityExps)).toBe(true);
      gravityExps.forEach((exp) => {
        expect(exp.metadata.category).toBe('gravity');
      });
    });

    it('should return empty array for category with no experiments', () => {
      const quantumExps = getExperimentsByCategory('quantization');
      expect(Array.isArray(quantumExps)).toBe(true);
      // May be empty if no quantization experiments yet
    });
  });

  describe('getFeaturedExperiments', () => {
    it('should return only featured experiments', () => {
      const featured = getFeaturedExperiments();
      expect(Array.isArray(featured)).toBe(true);
      featured.forEach((exp) => {
        expect(exp.metadata.featured).toBe(true);
      });
    });

    it('should include binary-orbit experiment', () => {
      const featured = getFeaturedExperiments();
      const binaryOrbit = featured.find((exp) => exp.metadata.id === 'binary-orbit');
      expect(binaryOrbit).toBeDefined();
    });
  });

  describe('searchExperiments', () => {
    it('should find experiments by title', () => {
      const results = searchExperiments('orbit');
      expect(results.length).toBeGreaterThan(0);
      const titles = results.map((exp) => exp.metadata.title.toLowerCase());
      expect(titles.some((title) => title.includes('orbit'))).toBe(true);
    });

    it('should find experiments by tag', () => {
      const results = searchExperiments('gravity');
      expect(results.length).toBeGreaterThan(0);
      const tags = results.flatMap((exp) => exp.metadata.tags);
      expect(tags.some((tag) => tag.includes('gravity'))).toBe(true);
    });

    it('should be case-insensitive', () => {
      const lowerResults = searchExperiments('black hole');
      const upperResults = searchExperiments('BLACK HOLE');
      expect(lowerResults.length).toBe(upperResults.length);
    });

    it('should return empty array for no matches', () => {
      const results = searchExperiments('zzz-nonexistent-experiment-xyz');
      expect(results).toEqual([]);
    });
  });

  describe('filterExperiments', () => {
    it('should filter by category', () => {
      const results = filterExperiments({ category: 'gravity' });
      results.forEach((exp) => {
        expect(exp.metadata.category).toBe('gravity');
      });
    });

    it('should filter by difficulty', () => {
      const results = filterExperiments({ difficulty: 'beginner' });
      results.forEach((exp) => {
        expect(exp.metadata.difficulty).toBe('beginner');
      });
    });

    it('should filter by tags', () => {
      const results = filterExperiments({ tags: ['webgpu'] });
      results.forEach((exp) => {
        expect(exp.metadata.tags).toContain('webgpu');
      });
    });

    it('should apply multiple filters simultaneously', () => {
      const results = filterExperiments({
        category: 'orbital-mechanics',
        difficulty: 'beginner',
      });
      results.forEach((exp) => {
        expect(exp.metadata.category).toBe('orbital-mechanics');
        expect(exp.metadata.difficulty).toBe('beginner');
      });
    });

    it('should return all experiments when no filters applied', () => {
      const results = filterExperiments({});
      const allExperiments = getAllExperiments();
      expect(results.length).toBe(allExperiments.length);
    });
  });

  describe('getAllCategories', () => {
    it('should return an array of unique categories', () => {
      const categories = getAllCategories();
      expect(Array.isArray(categories)).toBe(true);
      const uniqueCategories = new Set(categories);
      expect(uniqueCategories.size).toBe(categories.length);
    });

    it('should include orbital-mechanics and gravity', () => {
      const categories = getAllCategories();
      expect(categories).toContain('orbital-mechanics');
      expect(categories).toContain('gravity');
    });
  });

  describe('getAllTags', () => {
    it('should return an array of unique tags', () => {
      const tags = getAllTags();
      expect(Array.isArray(tags)).toBe(true);
      const uniqueTags = new Set(tags);
      expect(uniqueTags.size).toBe(tags.length);
    });

    it('should include common tags', () => {
      const tags = getAllTags();
      expect(tags.length).toBeGreaterThan(0);
      // Tags should be sorted
      const sortedTags = [...tags].sort();
      expect(tags).toEqual(sortedTags);
    });
  });

  describe('getCategoryStats', () => {
    it('should return experiment count per category', () => {
      const stats = getCategoryStats();
      expect(typeof stats).toBe('object');
      
      // Should have counts for existing categories
      const categories = getAllCategories();
      categories.forEach((cat) => {
        expect(stats[cat]).toBeGreaterThanOrEqual(0);
      });
    });

    it('should have correct counts', () => {
      const stats = getCategoryStats();
      const allExperiments = getAllExperiments();
      
      // Sum of all category counts should equal total experiments
      const totalCount = Object.values(stats).reduce((sum, count) => sum + count, 0);
      expect(totalCount).toBe(allExperiments.length);
    });
  });

  describe('Experiment metadata validation', () => {
    it('should have valid ISO 8601 dates', () => {
      const experiments = getAllExperiments();
      experiments.forEach((exp) => {
        expect(exp.metadata.created).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
        expect(exp.metadata.updated).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
      });
    });

    it('should have valid version numbers', () => {
      const experiments = getAllExperiments();
      experiments.forEach((exp) => {
        expect(exp.metadata.version).toMatch(/^\d+\.\d+\.\d+$/);
      });
    });

    it('should have valid backend requirements', () => {
      const experiments = getAllExperiments();
      experiments.forEach((exp) => {
        expect(['webgpu', 'webgl', 'cpu']).toContain(exp.metadata.backend.minBackend);
        if (exp.metadata.backend.estimatedVRAM) {
          expect(exp.metadata.backend.estimatedVRAM).toBeGreaterThan(0);
        }
      });
    });

    it('should have educational content', () => {
      const experiments = getAllExperiments();
      experiments.forEach((exp) => {
        expect(exp.metadata.education.whatYouSee).toBeDefined();
        expect(exp.metadata.education.whatYouSee.length).toBeGreaterThan(0);
        expect(Array.isArray(exp.metadata.education.principles)).toBe(true);
        expect(exp.metadata.education.principles.length).toBeGreaterThan(0);
      });
    });
  });
});

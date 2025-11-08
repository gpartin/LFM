/* -*- coding: utf-8 -*- */
/**
 * Experiment Registry
 * 
 * Central registry for all physics experiments.
 * Enables dynamic loading, search, categorization, and discovery.
 * 
 * To add a new experiment:
 * 1. Create experiment implementation in src/experiments/{category}/{experiment-id}
 * 2. Add entry to EXPERIMENTS array below
 * 3. Experiment will automatically appear in search/browse UI
 */

import type { ExperimentRegistryEntry, ExperimentCategory } from '@/types/experiment';

/**
 * All registered experiments
 * Metadata is always loaded, full module is lazy-loaded on demand
 */
export const EXPERIMENTS: ExperimentRegistryEntry[] = [
  {
    metadata: {
      id: 'binary-orbit',
      title: 'Earth-Moon Orbit',
      shortDescription: 'Watch Earth and Moon orbit due to emergent gravity from chi field gradients',
      fullDescription: 'Real Klein-Gordon physics running on your GPU—not Newtonian mechanics. Gravity emerges naturally from wave-like field equations.',
      category: 'orbital-mechanics',
      tags: ['gravity', 'orbit', 'emergent-gravity', 'earth-moon', 'chi-field', 'webgpu'],
      difficulty: 'beginner',
      version: '1.0.0',
      created: '2025-11-01T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: true,
      backend: {
        minBackend: 'webgpu',
        requiredFeatures: ['compute'],
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'Earth and Moon orbiting each other, with gravity emerging from field gradients instead of Newton\'s laws.',
        principles: [
          'Klein-Gordon wave equation: ∂²E/∂t² = c²∇²E − χ²(x,t)E',
          'Emergent gravity from field gradients',
          'Energy conservation in lattice field medium',
          'Angular momentum conservation',
        ],
        realWorld: 'While this is exploratory physics, understanding emergent phenomena could inform quantum gravity research.',
        references: [
          {
            title: 'LFM Master Document',
            url: '/docs/LFM_Master.txt',
            type: 'documentation',
          },
          {
            title: 'Binary Orbit Test Results',
            url: '/evidence/binary-orbit',
            type: 'evidence',
          },
        ],
      },
      thumbnail: '/thumbnails/binary-orbit.png',
      estimatedRuntime: 60,
    },
    loader: () => import('@/experiments/orbital-mechanics/binary-orbit'),
  },
  
  {
    metadata: {
      id: 'black-hole',
      title: 'Black Hole Orbit',
      shortDescription: 'A tiny black hole warps spacetime—watch a moon spiral in its gravitational grip',
      fullDescription: 'Simulate extreme gravity with a massive black hole at the center. Observe time dilation, event horizons, and orbital decay in emergent spacetime.',
      category: 'gravity',
      tags: ['black-hole', 'extreme-gravity', 'event-horizon', 'time-dilation', 'chi-field', 'webgpu'],
      difficulty: 'intermediate',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: true,
      backend: {
        minBackend: 'webgpu',
        requiredFeatures: ['compute'],
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'A moon orbiting a black hole, experiencing extreme spacetime curvature. Watch orbital precession, time dilation effects, and gravitational radiation.',
        principles: [
          'Extreme chi field gradients mimicking black hole gravity',
          'Orbital mechanics under strong field conditions',
          'Energy loss via gravitational radiation (emergent)',
          'Event horizon behavior in lattice field medium',
        ],
        realWorld: 'Black holes are real astrophysical objects. This simulation explores emergent analogues using LFM physics.',
        references: [
          {
            title: 'Gravity Test Results (Tier 2)',
            url: '/evidence/gravity',
            type: 'evidence',
          },
        ],
      },
      thumbnail: '/thumbnails/black-hole.png',
      estimatedRuntime: 90,
    },
    loader: () => import('@/experiments/gravity/black-hole'),
  },
  
  {
    metadata: {
      id: 'three-body',
      title: 'Three-Body Problem',
      shortDescription: 'Watch chaotic dynamics emerge from three gravitating bodies',
      fullDescription: 'The three-body problem is famously unsolvable analytically. Watch as tiny changes in initial conditions lead to wildly different orbital patterns.',
      category: 'orbital-mechanics',
      tags: ['chaos', 'n-body', 'gravity', 'emergent'],
      difficulty: 'advanced',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'Three bodies interacting chaotically through emergent gravity',
        principles: [
          'No general analytical solution exists',
          'Demonstrates chaos theory',
          'Emergent multi-body gravity from overlapping chi fields',
        ],
        realWorld: 'Models triple star systems and complex satellite orbits',
        references: [],
      },
      estimatedRuntime: 60,
    },
    loader: () => import('@/experiments/orbital-mechanics/three-body'),
  },
  
  {
    metadata: {
      id: 'big-bang',
      title: 'Big Bang',
      shortDescription: 'Watch energy explode outward in all directions from a single point',
      fullDescription: 'Pure wave propagation from the LFM equation. Energy starts concentrated at the center and expands spherically at the speed of light.',
      category: 'cosmology',
      tags: ['waves', 'expansion', 'cosmology', 'big-bang'],
      difficulty: 'intermediate',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'Spherical energy wave expanding from the center',
        principles: [
          'Wave equation: ∂²E/∂t² = c²∇²E',
          'Spherical symmetry',
          'Energy conservation during expansion',
        ],
        realWorld: 'Simplified model of early universe expansion',
        references: [],
      },
      estimatedRuntime: 30,
    },
    loader: () => import('@/experiments/cosmology/big-bang'),
  },
  
  {
    metadata: {
      id: 'stellar-collapse',
      title: 'Stellar Collapse',
      shortDescription: 'Watch a massive star collapse under its own gravity toward a black hole',
      fullDescription: 'Demonstrates gravitational collapse through time-evolving chi fields. The star begins stable, then catastrophically collapses as gravity overwhelms internal pressure.',
      category: 'gravity',
      tags: ['black-hole', 'collapse', 'gravity', 'stellar-evolution', 'supernova'],
      difficulty: 'research',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'A massive object collapsing into an increasingly dense state',
        principles: [
          'Gravitational collapse dynamics',
          'Time-evolving chi field intensification',
          'Approach to event horizon (singularity avoided for stability)',
        ],
        realWorld: 'Models supernova core collapse and black hole formation',
        references: [],
      },
      estimatedRuntime: 60,
    },
    loader: () => import('@/experiments/gravity/stellar-collapse'),
  },
];

/**
 * Get all experiments
 */
export function getAllExperiments(): ExperimentRegistryEntry[] {
  return EXPERIMENTS;
}

/**
 * Get experiment by ID
 */
export function getExperimentById(id: string): ExperimentRegistryEntry | undefined {
  return EXPERIMENTS.find((exp) => exp.metadata.id === id);
}

/**
 * Get experiments by category
 */
export function getExperimentsByCategory(category: ExperimentCategory): ExperimentRegistryEntry[] {
  return EXPERIMENTS.filter((exp) => exp.metadata.category === category);
}

/**
 * Get featured experiments
 */
export function getFeaturedExperiments(): ExperimentRegistryEntry[] {
  return EXPERIMENTS.filter((exp) => exp.metadata.featured === true);
}

/**
 * Search experiments by query
 * Searches in title, description, tags
 */
export function searchExperiments(query: string): ExperimentRegistryEntry[] {
  const lowerQuery = query.toLowerCase();
  return EXPERIMENTS.filter((exp) => {
    const { title, shortDescription, fullDescription, tags } = exp.metadata;
    return (
      title.toLowerCase().includes(lowerQuery) ||
      shortDescription.toLowerCase().includes(lowerQuery) ||
      fullDescription.toLowerCase().includes(lowerQuery) ||
      tags.some((tag) => tag.toLowerCase().includes(lowerQuery))
    );
  });
}

/**
 * Filter experiments by multiple criteria
 */
export function filterExperiments(filters: {
  category?: ExperimentCategory;
  difficulty?: string;
  tags?: string[];
  backend?: string;
}): ExperimentRegistryEntry[] {
  return EXPERIMENTS.filter((exp) => {
    if (filters.category && exp.metadata.category !== filters.category) {
      return false;
    }
    if (filters.difficulty && exp.metadata.difficulty !== filters.difficulty) {
      return false;
    }
    if (filters.tags && !filters.tags.every((tag) => exp.metadata.tags.includes(tag))) {
      return false;
    }
    if (filters.backend && exp.metadata.backend.minBackend !== filters.backend) {
      return false;
    }
    return true;
  });
}

/**
 * Get all unique categories
 */
export function getAllCategories(): ExperimentCategory[] {
  const categories = new Set<ExperimentCategory>();
  EXPERIMENTS.forEach((exp) => categories.add(exp.metadata.category));
  return Array.from(categories).sort();
}

/**
 * Get all unique tags
 */
export function getAllTags(): string[] {
  const tags = new Set<string>();
  EXPERIMENTS.forEach((exp) => exp.metadata.tags.forEach((tag) => tags.add(tag)));
  return Array.from(tags).sort();
}

/**
 * Get experiment count by category
 */
export function getCategoryStats(): Record<ExperimentCategory, number> {
  const stats: Partial<Record<ExperimentCategory, number>> = {};
  EXPERIMENTS.forEach((exp) => {
    const cat = exp.metadata.category;
    stats[cat] = (stats[cat] || 0) + 1;
  });
  return stats as Record<ExperimentCategory, number>;
}

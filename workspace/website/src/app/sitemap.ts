/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

import { MetadataRoute } from 'next';
import { getShowcaseExperiments } from '@/data/experiments';
import { RESEARCH_EXPERIMENTS } from '@/data/research-experiments-generated';

/**
 * Dynamic sitemap generation for search engine optimization.
 * Automatically includes all showcase and research experiment pages.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://emergentphysicslab.com';
  const now = new Date();
  
  const SHOWCASE_EXPERIMENTS = getShowcaseExperiments();

  // Static pages
  const staticPages = [
    {
      url: baseUrl,
      lastModified: now,
      changeFrequency: 'weekly' as const,
      priority: 1.0,
    },
    {
      url: `${baseUrl}/experiments`,
      lastModified: now,
      changeFrequency: 'weekly' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/about`,
      lastModified: now,
      changeFrequency: 'monthly' as const,
      priority: 0.7,
    },
    {
      url: `${baseUrl}/research`,
      lastModified: now,
      changeFrequency: 'weekly' as const,
      priority: 0.9,
    },
  ];

  // Showcase experiment pages
  const showcasePages = SHOWCASE_EXPERIMENTS.map((exp) => ({
    url: `${baseUrl}/experiments/${exp.id}`,
    lastModified: now,
    changeFrequency: 'monthly' as const,
    priority: 0.8,
  }));

  // Research experiment pages (all 105 tests)
  const researchPages = RESEARCH_EXPERIMENTS.map((exp) => ({
    url: `${baseUrl}/experiments/${exp.id}`,
    lastModified: now,
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }));

  return [...staticPages, ...showcasePages, ...researchPages];
}

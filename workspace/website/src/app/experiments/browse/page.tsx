/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import { 
  getAllExperiments, 
  searchExperiments, 
  filterExperiments, 
  getAllCategories, 
  getAllTags,
  getCategoryStats,
} from '@/lib/experimentRegistry';
import type { ExperimentCategory } from '@/types/experiment';

export default function ExperimentsBrowsePage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<ExperimentCategory | 'all'>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  const allExperiments = getAllExperiments();
  const allCategories = getAllCategories();
  const allTags = getAllTags();
  const categoryStats = getCategoryStats();

  // Filter experiments based on search and filters
  const filteredExperiments = useMemo(() => {
    let results = allExperiments;

    // Apply search query
    if (searchQuery.trim()) {
      results = searchExperiments(searchQuery);
    }

    // Apply filters
    const filters: any = {};
    if (selectedCategory !== 'all') {
      filters.category = selectedCategory;
    }
    if (selectedDifficulty !== 'all') {
      filters.difficulty = selectedDifficulty;
    }
    if (selectedTags.size > 0) {
      filters.tags = Array.from(selectedTags);
    }

    if (Object.keys(filters).length > 0) {
      results = filterExperiments(filters);
    }

    return results;
  }, [searchQuery, selectedCategory, selectedDifficulty, selectedTags, allExperiments]);

  const toggleTag = (tag: string) => {
    const newTags = new Set(selectedTags);
    if (newTags.has(tag)) {
      newTags.delete(tag);
    } else {
      newTags.add(tag);
    }
    setSelectedTags(newTags);
  };

  const clearFilters = () => {
    setSearchQuery('');
    setSelectedCategory('all');
    setSelectedDifficulty('all');
    setSelectedTags(new Set());
  };

  const difficultyColors = {
    beginner: 'text-green-400 border-green-500',
    intermediate: 'text-yellow-400 border-yellow-500',
    advanced: 'text-orange-400 border-orange-500',
    research: 'text-red-400 border-red-500',
  };

  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />
      
      <main className="flex-1 pt-20">
        <div className="container mx-auto px-4 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-accent-chi mb-2">Explore Physics Experiments</h1>
            <p className="text-text-secondary">
              Browse our collection of interactive physics simulations. Each experiment demonstrates emergent phenomena from lattice field medium.
            </p>
          </div>

          {/* Search Bar */}
          <div className="mb-8">
            <div className="relative">
              <input
                type="search"
                placeholder="Search experiments by name, description, or tags..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-6 py-4 bg-space-border/30 border-2 border-accent-chi/30 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:border-accent-chi transition-colors"
                aria-label="Search experiments"
              />
              <svg className="absolute right-4 top-1/2 -translate-y-1/2 w-6 h-6 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>

          {/* Filters */}
          <div className="mb-8 panel">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-accent-chi">Filters</h2>
              <button
                onClick={clearFilters}
                className="text-sm text-accent-particle hover:text-accent-chi transition-colors"
                aria-label="Clear all filters"
              >
                Clear All
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Category Filter */}
              <div>
                <label className="block text-sm font-semibold text-text-primary mb-2">Category</label>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value as any)}
                  className="w-full px-4 py-2 bg-space-border/30 border-2 border-accent-chi/30 rounded-lg text-text-primary focus:outline-none focus:border-accent-chi transition-colors"
                  aria-label="Filter by category"
                >
                  <option value="all">All Categories ({allExperiments.length})</option>
                  {allCategories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({categoryStats[cat] || 0})
                    </option>
                  ))}
                </select>
              </div>

              {/* Difficulty Filter */}
              <div>
                <label className="block text-sm font-semibold text-text-primary mb-2">Difficulty</label>
                <select
                  value={selectedDifficulty}
                  onChange={(e) => setSelectedDifficulty(e.target.value)}
                  className="w-full px-4 py-2 bg-space-border/30 border-2 border-accent-chi/30 rounded-lg text-text-primary focus:outline-none focus:border-accent-chi transition-colors"
                  aria-label="Filter by difficulty"
                >
                  <option value="all">All Levels</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="research">Research</option>
                </select>
              </div>

              {/* Tags Filter */}
              <div>
                <label className="block text-sm font-semibold text-text-primary mb-2">Tags</label>
                <div className="flex flex-wrap gap-2">
                  {allTags.slice(0, 6).map((tag) => (
                    <button
                      key={tag}
                      onClick={() => toggleTag(tag)}
                      className={`px-3 py-1 text-xs rounded-full border-2 transition-colors ${
                        selectedTags.has(tag)
                          ? 'bg-accent-chi/20 border-accent-chi text-accent-chi'
                          : 'bg-space-border/30 border-accent-chi/30 text-text-secondary hover:border-accent-chi'
                      }`}
                      aria-pressed={selectedTags.has(tag)}
                      aria-label={`Filter by ${tag}`}
                    >
                      {tag}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Results Count */}
          <div className="mb-6 text-text-secondary">
            Showing <span className="font-semibold text-accent-chi">{filteredExperiments.length}</span> experiment{filteredExperiments.length !== 1 ? 's' : ''}
          </div>

          {/* Experiments Grid */}
          {filteredExperiments.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredExperiments.map((exp) => (
                <Link
                  key={exp.metadata.id}
                  href={`/experiments/${exp.metadata.id}`}
                  className="panel hover:border-accent-chi/60 transition-all transform hover:scale-[1.02] hover:shadow-lg hover:shadow-accent-chi/20"
                  aria-label={`View ${exp.metadata.title} experiment`}
                >
                  {/* Thumbnail */}
                  <div className="w-full h-48 bg-space-border/50 rounded-lg mb-4 flex items-center justify-center text-4xl">
                    {exp.metadata.category === 'gravity' ? '⚫' : 
                     exp.metadata.category === 'orbital-mechanics' ? '🌍' :
                     exp.metadata.category === 'electromagnetic' ? '⚡' :
                     exp.metadata.category === 'quantization' ? '🔬' : '⚛️'}
                  </div>

                  {/* Content */}
                  <div>
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-xl font-bold text-accent-chi">{exp.metadata.title}</h3>
                      {exp.metadata.featured && (
                        <span className="text-yellow-400 text-xl" title="Featured experiment">⭐</span>
                      )}
                    </div>

                    <p className="text-sm text-text-secondary mb-4 line-clamp-2">
                      {exp.metadata.shortDescription}
                    </p>

                    {/* Metadata */}
                    <div className="flex items-center gap-4 mb-4">
                      <span className={`text-xs px-2 py-1 rounded border ${difficultyColors[exp.metadata.difficulty as keyof typeof difficultyColors]}`}>
                        {exp.metadata.difficulty}
                      </span>
                      <span className="text-xs text-text-secondary">
                        {exp.metadata.backend.minBackend.toUpperCase()}
                      </span>
                      {exp.metadata.estimatedRuntime && (
                        <span className="text-xs text-text-secondary">
                          ~{exp.metadata.estimatedRuntime}s
                        </span>
                      )}
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-2">
                      {exp.metadata.tags.slice(0, 3).map((tag) => (
                        <span key={tag} className="text-xs px-2 py-1 bg-accent-chi/10 text-accent-chi rounded">
                          {tag}
                        </span>
                      ))}
                      {exp.metadata.tags.length > 3 && (
                        <span className="text-xs text-text-secondary">+{exp.metadata.tags.length - 3}</span>
                      )}
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <div className="text-6xl mb-4">🔍</div>
              <h3 className="text-2xl font-bold text-accent-chi mb-2">No experiments found</h3>
              <p className="text-text-secondary mb-6">
                Try adjusting your search or filters
              </p>
              <button
                onClick={clearFilters}
                className="px-6 py-3 bg-accent-chi hover:bg-accent-chi/80 text-space-dark font-semibold rounded-lg transition-colors"
              >
                Clear Filters
              </button>
            </div>
          )}

          {/* Info Panel */}
          <div className="mt-12 panel bg-accent-chi/10 border-accent-chi/30">
            <h2 className="text-2xl font-bold text-accent-chi mb-4">About These Experiments</h2>
            <div className="prose prose-invert max-w-none text-text-secondary">
              <p className="mb-4">
                Each experiment in this collection demonstrates a different aspect of emergent physics from the 
                <strong className="text-accent-chi"> Lattice Field Medium (LFM)</strong> framework. These are:
              </p>
              <ul className="list-disc list-inside space-y-2 mb-4">
                <li><strong>GPU-accelerated</strong> — Real physics simulations running on your graphics card</li>
                <li><strong>Interactive</strong> — Adjust parameters and see results in real-time</li>
                <li><strong>Educational</strong> — Learn about emergent phenomena and field theory</li>
                <li><strong>Open Science</strong> — All source code and research data available</li>
              </ul>
              <p className="text-sm text-yellow-400">
                ⚠️ <strong>Scientific Disclosure:</strong> These are exploratory simulations, not validated physics models. 
                <Link href="/about" className="text-accent-chi hover:underline ml-1">Learn more about our approach →</Link>
              </p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

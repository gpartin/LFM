'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useMemo } from 'react';
import { formatPassRate } from '@/data/test-statistics';
import { getShowcaseExperiments, type ExperimentDefinition } from '@/data/experiments';

export default function Header() {
  const [experimentsOpen, setExperimentsOpen] = useState(false);
  // In test environments, usePathname() can be null; default to empty string to avoid errors
  const pathname = usePathname() ?? '';
  
  // Separate featured experiments for top-level display
  const { featuredExperiments, otherExperiments } = useMemo(() => {
    const experiments = getShowcaseExperiments();
    const featured = experiments.filter(exp => exp.featured);
    const other = experiments.filter(exp => !exp.featured);
    
    return { featuredExperiments: featured, otherExperiments: other };
  }, []);
  
  // Determine "from" parameter for About link based on current page
  const getFromParam = () => {
    if (pathname === '/') return 'Home';
    
    // Try to match against registered experiments
    const experiments = getShowcaseExperiments();
    const currentExperiment = experiments.find(exp => 
      pathname.includes(`/experiments/${exp.id}`)
    );
    if (currentExperiment) return currentExperiment.displayName;
    
    if (pathname.includes('/experiments/browse')) return 'Browse Experiments';
    if (pathname.includes('/experiments')) return 'Experiments';
    return 'Home';
  };
  
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-space-dark/95 backdrop-blur-sm border-b border-space-border">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-accent-chi to-accent-particle rounded-lg flex items-center justify-center">
              <span className="text-2xl">⚛️</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-accent-chi">Emergent Physics Lab</h1>
              <p className="text-xs text-text-secondary">Where forces emerge from the lattice</p>
            </div>
          </Link>

          {/* Research Links */}
          <nav className="hidden md:flex items-center space-x-4">
            {/* Experiments Dropdown */}
            <div 
              className="relative"
              onMouseEnter={() => setExperimentsOpen(true)}
              onMouseLeave={() => setExperimentsOpen(false)}
            >
              <button className="px-4 py-2 text-text-secondary hover:text-accent-chi transition-colors flex items-center space-x-1">
                <span>Experiments</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {experimentsOpen && (
                <div className="absolute top-full left-0 pt-2 w-80">
                  <div className="bg-space-panel border border-space-border rounded-lg shadow-xl overflow-hidden max-h-[80vh] overflow-y-auto">
                    {/* Featured Experiments - Always Visible */}
                    <div className="px-4 py-3 bg-gradient-to-r from-accent-chi/10 to-accent-particle/10 border-b border-space-border">
                      <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
                        ✨ Featured
                      </div>
                      <div className="space-y-1">
                        {featuredExperiments.slice(0, 4).map(exp => (
                          <Link
                            key={exp.id}
                            href={`/experiments/${exp.id}`}
                            className="block px-3 py-2 rounded hover:bg-space-dark/50 transition-colors group"
                          >
                            <div className="flex items-center space-x-2">
                              <span className="text-lg">{exp.icon}</span>
                              <div className="flex-1">
                                <div className="text-sm font-medium text-text-primary group-hover:text-accent-chi transition-colors">
                                  {exp.displayName}
                                </div>
                                <div className="text-xs text-text-muted line-clamp-1">
                                  {exp.tagline}
                                </div>
                              </div>
                            </div>
                          </Link>
                        ))}
                      </div>
                    </div>
                    
                    {/* Browse All - Prominent */}
                    <Link
                      href="/experiments/browse"
                      className="block px-4 py-3 hover:bg-space-dark transition-colors border-b border-space-border bg-accent-chi/5"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold text-accent-chi">Browse All Experiments</div>
                          <div className="text-xs text-text-muted">
                            {featuredExperiments.length + otherExperiments.length} total experiments
                          </div>
                        </div>
                        <svg className="w-5 h-5 text-accent-chi" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </Link>
                    
                    {/* Other Experiments - Collapsed by default */}
                    {otherExperiments.length > 0 && (
                      <div className="py-2">
                        <div className="px-4 py-2 text-xs font-semibold text-text-muted uppercase tracking-wider">
                          More Experiments
                        </div>
                        {otherExperiments.map(exp => (
                          <Link
                            key={exp.id}
                            href={`/experiments/${exp.id}`}
                            className="block px-4 py-2 hover:bg-space-dark transition-colors text-text-primary hover:text-accent-chi"
                          >
                            <div className="flex items-center space-x-2">
                              <span>{exp.icon}</span>
                              <span className="text-sm">{exp.displayName}</span>
                            </div>
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            <ExternalLink
              href="https://osf.io/6agn8"
              label="OSF Project"
              badge="10.17605/OSF.IO/6AGN8"
              icon="🔬"
            />
            <ExternalLink
              href="https://zenodo.org/records/17536484"
              label="Zenodo"
              badge="10.5281/zenodo.17536484"
              icon="📚"
            />
            <ExternalLink
              href="https://github.com/gpartin/LFM"
              label="GitHub"
              badge={formatPassRate()}
              icon="💻"
            />
            {/* Implications link intentionally removed from public nav */}
            <Link
              href={`/about?from=${encodeURIComponent(getFromParam())}`}
              className="px-4 py-2 text-text-secondary hover:text-accent-chi transition-colors"
            >
              About
            </Link>
          </nav>

          {/* Mobile menu button */}
          <button className="md:hidden p-2 text-text-secondary hover:text-accent-chi">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}

function ExternalLink({ href, label, badge, icon }: { href: string; label: string; badge: string; icon: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex flex-col items-start px-4 py-2 rounded-lg hover:bg-space-panel transition-all"
    >
      <div className="flex items-center space-x-2">
        <span>{icon}</span>
        <span className="text-sm font-semibold text-text-primary group-hover:text-accent-chi transition-colors">
          {label}
        </span>
      </div>
      <span className="text-xs text-text-muted font-mono">{badge}</span>
    </a>
  );
}

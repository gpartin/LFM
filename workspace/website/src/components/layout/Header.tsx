'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import Link from 'next/link';
import { useState } from 'react';

export default function Header() {
  const [experimentsOpen, setExperimentsOpen] = useState(false);
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
                <div className="absolute top-full left-0 pt-2 w-64">
                  <div className="bg-space-panel border border-space-border rounded-lg shadow-xl overflow-hidden">
                    <Link
                      href="/experiments/browse"
                      className="block px-4 py-3 hover:bg-space-dark transition-colors border-b border-space-border"
                    >
                      <div className="font-semibold text-accent-chi">Browse All Experiments</div>
                      <div className="text-xs text-text-muted">Search and filter by category</div>
                    </Link>
                    
                    <div className="py-2">
                      <div className="px-4 py-2 text-xs font-semibold text-text-muted uppercase tracking-wider">
                        Orbital Mechanics
                      </div>
                      <Link
                        href="/experiments/binary-orbit"
                        className="block px-4 py-2 hover:bg-space-dark transition-colors text-text-primary hover:text-accent-chi"
                      >
                        <div className="flex items-center space-x-2">
                          <span>🌍</span>
                          <span>Binary Orbit</span>
                        </div>
                        <div className="text-xs text-text-muted ml-6">Earth-Moon simulation</div>
                      </Link>
                    </div>
                    
                    <div className="py-2">
                      <Link
                        href="/experiments/three-body"
                        className="block px-4 py-2 hover:bg-space-dark transition-colors text-text-primary hover:text-accent-chi"
                      >
                        <div className="flex items-center space-x-2">
                          <span>🔺</span>
                          <span>Three-Body Problem</span>
                        </div>
                        <div className="text-xs text-text-muted ml-6">Chaotic N-body dynamics</div>
                      </Link>
                    </div>
                    
                    <div className="py-2 border-t border-space-border">
                      <div className="px-4 py-2 text-xs font-semibold text-text-muted uppercase tracking-wider">
                        Gravity
                      </div>
                      <Link
                        href="/experiments/black-hole"
                        className="block px-4 py-2 hover:bg-space-dark transition-colors text-text-primary hover:text-accent-chi"
                      >
                        <div className="flex items-center space-x-2">
                          <span>⚫</span>
                          <span>Black Hole</span>
                        </div>
                        <div className="text-xs text-text-muted ml-6">Extreme gravity simulation</div>
                      </Link>
                      <Link
                        href="/experiments/stellar-collapse"
                        className="block px-4 py-2 hover:bg-space-dark transition-colors text-text-primary hover:text-accent-chi"
                      >
                        <div className="flex items-center space-x-2">
                          <span>💫</span>
                          <span>Stellar Collapse</span>
                        </div>
                        <div className="text-xs text-text-muted ml-6">Star collapsing to black hole</div>
                      </Link>
                    </div>
                    
                    <div className="py-2 border-t border-space-border">
                      <div className="px-4 py-2 text-xs font-semibold text-text-muted uppercase tracking-wider">
                        Cosmology
                      </div>
                      <Link
                        href="/experiments/big-bang"
                        className="block px-4 py-2 hover:bg-space-dark transition-colors text-text-primary hover:text-accent-chi"
                      >
                        <div className="flex items-center space-x-2">
                          <span>💥</span>
                          <span>Big Bang</span>
                        </div>
                        <div className="text-xs text-text-muted ml-6">Energy explosion from a point</div>
                      </Link>
                    </div>
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
              badge="91.4% Tests Pass"
              icon="💻"
            />
            <Link
              href="/about"
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

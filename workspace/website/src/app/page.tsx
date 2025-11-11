/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import Link from 'next/link';
import { testStatistics, formatSummary } from '@/data/test-statistics';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 pt-20">
        {/* Hero Section */}
        <section className="relative overflow-hidden py-20">
          {/* Background gradient */}
          <div className="absolute inset-0 bg-gradient-to-b from-accent-chi/5 to-transparent" />
          
          <div className="container mx-auto px-4 relative z-10">
            <div className="text-center max-w-4xl mx-auto">
              <h1 className="text-5xl md:text-7xl font-bold mb-6">
                <span className="text-accent-chi glow-text">Fundamental Forces</span>
                <br />
                <span className="text-text-primary">Emerging from</span>
                <br />
                <span className="text-accent-particle">a Single Equation</span>
              </h1>
              
              <p className="text-xl text-text-secondary mb-8 leading-relaxed">
                Watch gravity, relativity, and quantum phenomena emerge from the{' '}
                <span className="text-accent-chi font-semibold">Lattice Field Medium</span> —
                a modified Klein-Gordon equation running on each point of a 3D lattice.
              </p>

              {/* Equation Display */}
              <div className="bg-space-panel border-2 border-accent-chi/30 rounded-lg p-6 mb-10 inline-block">
                <div className="font-mono text-2xl md:text-3xl text-accent-chi mb-2">
                  ∂²E/∂t² = c²∇²E − χ²(x,t)E
                </div>
                <div className="text-sm text-text-muted">
                  One equation. All fundamental forces.
                </div>
              </div>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
                <Link href="/experiments/binary-orbit" className="button-primary text-lg">
                  🪐 Launch Orbital Experiment
                </Link>
                <Link href="/about?from=Home" className="button-secondary text-lg">
                  📚 Read the Science
                </Link>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
                <StatCard 
                  value={testStatistics.passRate} 
                  label="Tests Passing" 
                  description={formatSummary()}
                />
                <StatCard 
                  value="<0.01%" 
                  label="Energy Drift" 
                  description="Conservation verified"
                />
                <StatCard 
                  value="7 Domains" 
                  label="Emerge" 
                  description="Gravity, EM, relativity, quantum, thermodynamics"
                />
              </div>
            </div>
          </div>
        </section>

        {/* What You Can Explore */}
        <section className="py-20 bg-space-panel/50">
          <div className="container mx-auto px-4">
            <h2 className="text-4xl font-bold text-center mb-12 text-accent-chi">
              Interactive Experiments
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
              <ExperimentCard
                title="Earth-Moon Orbit"
                icon="🌍"
                description="Watch two masses orbit due to emergent gravity from chi field gradients. Real-time parameter control."
                href="/experiments/binary-orbit"
                status="available"
              />
              <ExperimentCard
                title="Wave Propagation"
                icon="〰️"
                description="Observe electromagnetic waves emerging from lattice vibrations. Adjustable frequency and amplitude."
                href="/experiments/wave-propagation"
                status="coming-soon"
              />
              <ExperimentCard
                title="Gravitational Lensing"
                icon="🌟"
                description="Light bending around massive objects through chi field distortions. Pure emergence."
                href="/experiments/lensing"
                status="coming-soon"
              />
              <ExperimentCard
                title="Relativistic Effects"
                icon="⚡"
                description="Lorentz contraction and time dilation arising from lattice structure. No special relativity needed."
                href="/experiments/relativity"
                status="coming-soon"
              />
              <ExperimentCard
                title="Quantum Tunneling"
                icon="🌊"
                description="Particles passing through barriers via wave mechanics on the lattice. Classical → Quantum."
                href="/experiments/tunneling"
                status="coming-soon"
              />
              <ExperimentCard
                title="Chi Field Viewer"
                icon="🔬"
                description="Visualize the scalar field that gives rise to all forces. Interactive 3D exploration."
                href="/experiments/chi-field"
                status="coming-soon"
              />
            </div>
          </div>
        </section>

        {/* Key Features */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <h2 className="text-4xl font-bold text-center mb-12 text-accent-chi">
              Why This Matters
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
              <FeatureCard
                icon="✓"
                title="Authentic Physics"
                description="These aren't visualizations of equations — they're real simulations running Klein-Gordon on GPU. Every result is calculated from first principles."
              />
              <FeatureCard
                icon="🔍"
                title="Fully Open Source"
                description="All code is public on GitHub. Inspect the physics engine, verify the math, run your own tests. Complete scientific transparency."
              />
              <FeatureCard
                icon="⚡"
                title="Real-Time Emergence"
                description="Watch fundamental forces emerge in real-time as you adjust parameters. This is not approximate — it's the actual LFM framework."
              />
              <FeatureCard
                icon="🎯"
                title="Energy Conservation"
                description="Every simulation tracks energy drift. See for yourself that conservation laws hold to <0.01% — as documented in our OSF/Zenodo records."
              />
            </div>
          </div>
        </section>

        {/* Research Links */}
        <section className="py-20 bg-space-panel/30">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-6 text-accent-chi">
              Publicly Disclosed Research
            </h2>
            <p className="text-lg text-text-secondary mb-8 max-w-3xl mx-auto">
              This website demonstrates the Lattice Field Medium framework, an open research project with
              public disclosures on OSF and Zenodo. It has not yet undergone formal peer review.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="https://osf.io/6agn8"
                target="_blank"
                rel="noopener noreferrer"
                className="button-primary"
              >
                🔬 View OSF Project
              </a>
              <a
                href="https://zenodo.org/records/17536484"
                target="_blank"
                rel="noopener noreferrer"
                className="button-secondary"
              >
                📚 Zenodo Archive
              </a>
              <a
                href="https://github.com/gpartin/LFM"
                target="_blank"
                rel="noopener noreferrer"
                className="button-secondary"
              >
                💻 Source Code
              </a>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}

function StatCard({ value, label, description }: { value: string; label: string; description: string }) {
  return (
    <div className="panel text-center">
      <div className="text-3xl font-bold text-accent-chi mb-2">{value}</div>
      <div className="text-lg font-semibold text-text-primary mb-1">{label}</div>
      <div className="text-sm text-text-muted">{description}</div>
    </div>
  );
}

function ExperimentCard({ 
  title, 
  icon, 
  description, 
  href, 
  status 
}: { 
  title: string; 
  icon: string; 
  description: string; 
  href: string; 
  status: 'available' | 'coming-soon';
}) {
  const isAvailable = status === 'available';
  
  return (
    <Link
      href={isAvailable ? href : '#'}
      className={`
        panel group transition-all duration-300
        ${isAvailable ? 'hover:border-accent-chi cursor-pointer hover:glow-box' : 'opacity-60 cursor-not-allowed'}
      `}
    >
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-bold text-text-primary mb-2 group-hover:text-accent-chi transition-colors">
        {title}
      </h3>
      <p className="text-sm text-text-secondary leading-relaxed mb-4">
        {description}
      </p>
      {isAvailable ? (
        <div className="text-accent-chi text-sm font-semibold">
          Launch Experiment →
        </div>
      ) : (
        <div className="text-text-muted text-sm">
          Coming Soon
        </div>
      )}
    </Link>
  );
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="panel">
      <div className="flex items-start space-x-4">
        <div className="text-3xl text-accent-glow">{icon}</div>
        <div>
          <h3 className="text-xl font-bold text-text-primary mb-2">{title}</h3>
          <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
        </div>
      </div>
    </div>
  );
}

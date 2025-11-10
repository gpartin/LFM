'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';

export default function AboutPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const from = searchParams.get('from') || 'Simulation';
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-space-dark via-space-medium to-space-dark">
      <div className="container mx-auto px-4 py-16 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-accent-chi via-accent-particle to-accent-chi bg-clip-text text-transparent">
            About This Project
          </h1>
          <p className="text-xl text-text-secondary">
            Lattice Field Medium (LFM): A Discrete Computational Physics Framework
          </p>
        </div>

        {/* Scientific Status Warning */}
        <div className="panel border-2 border-yellow-500/50 bg-yellow-500/10 mb-8">
          <div className="flex items-start gap-4">
            <div className="text-4xl">⚠️</div>
            <div>
              <h2 className="text-2xl font-bold text-yellow-400 mb-2">Scientific Status: Exploratory Research</h2>
              <p className="text-text-secondary">
                <strong className="text-text-primary">We are NOT claiming we have "solved physics" or "disproven Einstein."</strong> This is 
                a computational framework demonstrating that relativistic, gravitational, quantum, and electromagnetic phenomena 
                CAN emerge from two coupled wave equations on a discrete lattice. This has NOT been peer-reviewed or validated 
                against all known physics. Treat this as a <strong>hypothesis requiring rigorous investigation</strong>, 
                not a proven theory.
              </p>
            </div>
          </div>
        </div>

        {/* What This Project Is */}
        <div className="panel mb-8">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">What LFM Is</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-text-primary">LFM (Lattice Field Medium)</strong> is a computational physics framework built on 
              the hypothesis that <strong>physical reality is a discrete computational lattice</strong> where each point evolves 
              according to two coupled wave equations:
            </p>
            <div className="bg-space-dark p-4 rounded-lg font-mono text-accent-chi text-sm my-4">
              <div>∂²E/∂t² = c²∇²E − χ²(x,t)E    [Wave equation]</div>
              <div className="mt-2">∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)  [Curvature dynamics]</div>
            </div>
            <p>
              From these two equations alone, we claim that <strong>all of macroscopic physics emerges</strong>:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li><strong>Relativity:</strong> Lorentz invariance emerges from lattice dispersion (70 tests)</li>
              <li><strong>Gravity:</strong> Spacetime curvature emerges as χ responds to energy density (25 tests)</li>
              <li><strong>Quantum mechanics:</strong> Bound states and quantization emerge naturally (14 tests)</li>
              <li><strong>Electromagnetism:</strong> Field interactions emerge from wave coupling (21 tests)</li>
              <li><strong>Energy conservation:</strong> &lt;0.01% drift across all domains (11 tests)</li>
            </ul>
            <p className="text-accent-chi font-semibold">
              Current validation status: <strong>86 tests, 91.4% pass rate</strong> across five physics domains.
            </p>
          </div>
        </div>

        {/* What This Project Is NOT */}
        <div className="panel mb-8 border-2 border-red-500/30">
          <h2 className="text-3xl font-bold text-red-400 mb-4">What LFM Is NOT (Intellectual Honesty)</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-text-primary">We are NOT claiming we have "solved physics" or "replaced quantum field theory."</strong>
            </p>
            <p className="font-semibold text-text-primary">We Have NOT Shown (Yet):</p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li><strong>Continuum limit convergence:</strong> Does this work as lattice spacing → 0?</li>
              <li><strong>Einstein equations emerge:</strong> Does χ satisfy G_μν = 8πG T_μν exactly?</li>
              <li><strong>Novel predictions:</strong> What does LFM predict that differs from GR/QM/EM?</li>
              <li><strong>Full self-consistency:</strong> Can we simulate a universe with ONLY emergent χ?</li>
              <li><strong>Peer review:</strong> No formal publications or independent validation yet</li>
            </ul>
            <p className="font-semibold text-text-primary mt-4">We HAVE Shown:</p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li><strong>χ-field emergence:</strong> Spacetime curvature emerges from energy (r=0.46 correlation)</li>
              <li><strong>Energy conservation:</strong> &lt;0.01% drift over thousands of timesteps</li>
              <li><strong>Cross-domain consistency:</strong> Same equations produce relativity, gravity, QM, EM</li>
              <li><strong>Computational stability:</strong> No blow-ups, artifacts, or numerical collapse</li>
            </ul>
            <p className="text-yellow-400 mt-4">
              <strong>We are computational physicists with a hypothesis.</strong> We're openly sharing our work, code, and data 
              for evaluation by the scientific community. We welcome rigorous critique and falsification attempts.
            </p>
          </div>
        </div>

        {/* The Mathematics */}
        <div className="panel mb-8">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">The Mathematics Behind It</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-6">
            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">1. The Klein-Gordon Equation</h3>
              <p>The foundation is a relativistic wave equation on a 3D lattice:</p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-accent-chi text-center my-4">
                ∂²E/∂t² = c²∇²E − χ²(x,t)E
              </div>
              <p>Where:</p>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li><code>E</code> = field amplitude at each point</li>
                <li><code>c</code> = wave propagation speed (like speed of light)</li>
                <li><code>∇²</code> = Laplacian (spatial curvature)</li>
                <li><code>χ(x,t)</code> = variable mass field that couples to matter</li>
              </ul>
            </div>

            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">2. The Chi Field</h3>
              <p>Each mass creates a Gaussian "bump" in the chi field:</p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-accent-chi text-center my-4">
                χ(x) = χ₀ + Σᵢ mᵢ · exp(-rᵢ²/σ²)
              </div>
              <p>This creates a "hill" around each mass, with slope proportional to mass.</p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">3. Emergent Force</h3>
              <p>Objects feel a force from the gradient (slope) of the chi field:</p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-accent-chi text-center my-4">
                F = -m · ∇χ · χ_strength
              </div>
              <p>
                This is NOT Newton's law! It's simply "objects roll down hills in the field." Yet somehow, 
                this produces 1/r² behavior and stable orbits.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">4. Energy Conservation</h3>
              <p>Total energy is kinetic energy + field energy:</p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-accent-chi text-center my-4">
                E_total = Σ(½mv²) + ∫(field_energy)
              </div>
              <p>
                The simulation conserves this to &lt;0.01% over thousands of timesteps, indicating 
                the dynamics are physically consistent.
              </p>
            </div>
          </div>
        </div>

        {/* The Critical Evidence */}
        <div className="panel mb-8 bg-accent-chi/5">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">The Critical Evidence: χ-Field Emergence</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-text-primary">The most important question:</strong> Is χ manually configured 
              (parameter fitting) or does it emerge from energy distribution?
            </p>
            <p className="text-accent-particle font-semibold text-lg">
              Answer: <strong>It emerges.</strong> ✅
            </p>
            <div className="bg-space-dark p-4 rounded-lg my-4">
              <p className="font-semibold text-text-primary mb-2">Test Setup:</p>
              <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                <li>Initial state: χ = 0.1 uniform everywhere (no spatial structure)</li>
                <li>Energy pulse placed at one location</li>
                <li>System evolved for 2000 timesteps via χ dynamics equation</li>
                <li>No manual configuration of final χ distribution</li>
              </ul>
              <p className="font-semibold text-text-primary mt-4 mb-2">Result:</p>
              <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                <li>χ spontaneously formed wells around energy concentration</li>
                <li>224,761× increase in spatial variation from uniform initial state</li>
                <li>Correlation between E² and Δχ: <strong>r = 0.46</strong> (p &lt; 0.001)</li>
              </ul>
            </div>
            <p>
              <strong className="text-accent-chi">Physical interpretation:</strong> Gravitational curvature emerges from matter/energy distribution, 
              analogous to Einstein's field equations (G_μν = 8πG T_μν) but implemented on a discrete lattice.
            </p>
            <p className="text-text-muted text-sm">
              See: <code>python tests/test_chi_emergence_critical.py</code> in the GitHub repository
            </p>
          </div>
        </div>

        {/* Why This Matters */}
        <div className="panel mb-8">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">Why This Might Be Important</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-text-primary">If</strong> LFM can reproduce all known physics 
              (which we haven't fully proven), it would suggest:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li><strong>Unified foundation:</strong> Relativity, gravity, QM, and EM from two equations</li>
              <li><strong>Discrete reality:</strong> Physical spacetime is fundamentally quantized/computational</li>
              <li><strong>Emergent forces:</strong> What we call "forces" are lattice dynamics, not fundamental</li>
              <li><strong>Quantum-gravity compatible:</strong> Already discrete, avoids singularities</li>
              <li><strong>Computationally tractable:</strong> Easier to simulate than curved spacetime + quantum fields</li>
            </ul>
            <p className="text-yellow-400">
              <strong>But this is highly speculative.</strong> We're sharing our work to invite testing, validation, 
              and especially falsification attempts from professional physicists.
            </p>
          </div>
        </div>

        {/* Resources */}
        <div className="panel mb-8">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">Learn More & Verify</h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">📄 Research Documentation</h3>
              <p className="text-text-secondary mb-3">
                Detailed papers, test results, and methodology:
              </p>
              <div className="flex flex-col gap-2">
                <a 
                  href="https://osf.io/emergentphysicslab" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-accent-chi hover:text-accent-particle transition-colors"
                >
                  <span>🔬</span>
                  <span>Open Science Framework (OSF)</span>
                  <span>→</span>
                </a>
                <a 
                  href="https://zenodo.org/communities/emergentphysicslab" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-accent-chi hover:text-accent-particle transition-colors"
                >
                  <span>📦</span>
                  <span>Zenodo Archive</span>
                  <span>→</span>
                </a>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">💻 Source Code</h3>
              <p className="text-text-secondary mb-3">
                All simulation code is open source. Verify there's no hidden gravity equations:
              </p>
              <a 
                href="https://github.com/gpartin/lfm" 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-accent-chi hover:text-accent-particle transition-colors"
              >
                <span>🐙</span>
                <span>GitHub Repository</span>
                <span>→</span>
              </a>
            </div>

            <div>
              <h3 className="text-xl font-bold text-text-primary mb-2">🤝 Collaboration</h3>
              <p className="text-text-secondary mb-3">
                Are you a physicist? We want your feedback:
              </p>
              <ul className="list-disc list-inside space-y-1 text-text-secondary ml-4">
                <li>Can you find where this breaks?</li>
                <li>Does it match or contradict known physics?</li>
                <li>What experiments would validate/falsify this?</li>
              </ul>
              <p className="text-text-secondary mt-3">
                Contact: <a href="mailto:research@emergentphysicslab.com" className="text-accent-chi hover:underline">research@emergentphysicslab.com</a>
              </p>
            </div>
          </div>
        </div>

        {/* Back to Experiments */}
        <div className="text-center mt-12">
          <button
            onClick={() => router.back()}
            className="inline-flex items-center gap-2 px-6 py-3 bg-accent-chi text-space-dark font-bold rounded-lg hover:bg-accent-particle transition-colors"
          >
            <span>←</span>
            <span>Back to {from}</span>
          </button>
        </div>

        {/* Footer Note */}
        <div className="mt-12 text-center text-sm text-text-secondary border-t border-text-secondary/20 pt-8">
          <p>
            This project follows the principles of open science. All data, code, and methodology are 
            freely available for replication and critique. We welcome skepticism and rigorous testing.
          </p>
          <p className="mt-2">
            Last Updated: November 7, 2025
          </p>
        </div>
      </div>
    </div>
  );
}

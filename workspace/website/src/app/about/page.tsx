'use client';

import Link from 'next/link';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-space-dark via-space-medium to-space-dark">
      <div className="container mx-auto px-4 py-16 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-accent-chi via-accent-particle to-accent-chi bg-clip-text text-transparent">
            About This Project
          </h1>
          <p className="text-xl text-text-secondary">
            Exploring Emergent Gravity Through Computational Physics
          </p>
        </div>

        {/* Scientific Status Warning */}
        <div className="panel border-2 border-yellow-500/50 bg-yellow-500/10 mb-8">
          <div className="flex items-start gap-4">
            <div className="text-4xl">⚠️</div>
            <div>
              <h2 className="text-2xl font-bold text-yellow-400 mb-2">Scientific Status: Exploratory Research</h2>
              <p className="text-text-secondary">
                <strong className="text-text-primary">We are NOT claiming this is how gravity actually works in nature.</strong> This is 
                a computational demonstration that gravity-like orbital mechanics CAN emerge from wave field dynamics 
                without explicitly programming Newton's law of gravity. This has NOT been peer-reviewed or validated 
                against comprehensive astronomical data. Treat this as a <strong>hypothesis requiring investigation</strong>, 
                not a proven theory.
              </p>
            </div>
          </div>
        </div>

        {/* What This Project Is */}
        <div className="panel mb-8">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">What This Project Is</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              This website demonstrates a computational simulation where <strong className="text-text-primary">orbital mechanics 
              emerge naturally</strong> from a wave equation (the Klein-Gordon equation) coupled with a variable mass field (chi field), 
              without programming Newton's law of gravity (<code>F = GMm/r²</code>) or Einstein's field equations.
            </p>
            <p>
              When you watch the Earth-Moon simulation:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li>The Moon <strong>really is orbiting</strong> the Earth in the simulation</li>
              <li>Energy is <strong>conserved to &lt;0.01%</strong> (a key physics requirement)</li>
              <li>The code contains <strong>no gravitational force equations</strong></li>
              <li>Instead, objects respond to <strong>gradients (slopes) in a field</strong></li>
            </ul>
            <p>
              This raises a fascinating question: <em>If orbital mechanics can emerge from simpler field dynamics, 
              could this be how gravity actually works in nature?</em>
            </p>
          </div>
        </div>

        {/* What This Project Is NOT */}
        <div className="panel mb-8 border-2 border-red-500/30">
          <h2 className="text-3xl font-bold text-red-400 mb-4">What This Project Is NOT</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-text-primary">This is NOT a claim that we have "solved gravity" or "disproven Einstein."</strong>
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li>We have NOT tested this against all gravitational phenomena (black holes, gravitational waves, cosmology, etc.)</li>
              <li>We have NOT proven this is fundamentally different from General Relativity (it might be mathematically equivalent)</li>
              <li>We have NOT published peer-reviewed papers validating these results</li>
              <li>We have NOT made predictions that differ from and have been confirmed against Einstein's theory</li>
            </ul>
            <p className="text-yellow-400">
              <strong>We are programmers and computational researchers, not theoretical physicists.</strong> We've built 
              something interesting and are openly sharing it for evaluation by the scientific community.
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

        {/* Why This Matters */}
        <div className="panel mb-8">
          <h2 className="text-3xl font-bold text-accent-chi mb-4">Why This Might Be Important</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-text-primary">If</strong> this approach can reproduce all gravitational phenomena 
              (which we haven't proven), it would have profound implications:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li><strong>Simpler foundation:</strong> Gravity emerges from field dynamics rather than being a fundamental force</li>
              <li><strong>Quantum compatible:</strong> The lattice is already discrete/quantized, avoiding singularities</li>
              <li><strong>Computationally tractable:</strong> Easier to simulate than curved spacetime</li>
              <li><strong>Conceptually intuitive:</strong> "Rolling down hills" vs. "warping spacetime"</li>
            </ul>
            <p className="text-yellow-400">
              <strong>But again: this is speculative.</strong> We're sharing our work to invite testing, validation, 
              and critique from people with more physics expertise than we have.
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
          <Link 
            href="/experiments/binary-orbit"
            className="inline-flex items-center gap-2 px-6 py-3 bg-accent-chi text-space-dark font-bold rounded-lg hover:bg-accent-particle transition-colors"
          >
            <span>←</span>
            <span>Try the Earth-Moon Simulation</span>
          </Link>
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

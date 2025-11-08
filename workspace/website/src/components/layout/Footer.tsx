'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

export default function Footer() {
  return (
    <footer className="border-t border-space-border bg-space-panel mt-20">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-lg font-bold text-accent-chi mb-3">About This Project</h3>
            <p className="text-sm text-text-secondary leading-relaxed">
              Interactive demonstrations of the Lattice Field Medium framework, showing how fundamental
              forces emerge from a single modified Klein-Gordon equation.
            </p>
          </div>

          {/* Research Links */}
          <div>
            <h3 className="text-lg font-bold text-accent-chi mb-3">Research</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="https://osf.io/6agn8"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-text-secondary hover:text-accent-chi transition-colors flex items-center space-x-2"
                >
                  <span>🔬</span>
                  <span>OSF Project (DOI: 10.17605/OSF.IO/6AGN8)</span>
                </a>
              </li>
              <li>
                <a
                  href="https://zenodo.org/records/17536484"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-text-secondary hover:text-accent-chi transition-colors flex items-center space-x-2"
                >
                  <span>📚</span>
                  <span>Zenodo Archive (DOI: 10.5281/zenodo.17536484)</span>
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/gpartin/LFM"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-text-secondary hover:text-accent-chi transition-colors flex items-center space-x-2"
                >
                  <span>💻</span>
                  <span>GitHub Repository (All Source Code)</span>
                </a>
              </li>
            </ul>
          </div>

          {/* License */}
          <div>
            <h3 className="text-lg font-bold text-accent-chi mb-3">License & Attribution</h3>
            <div className="text-sm text-text-secondary space-y-2">
              <p>
                <strong>Research Content:</strong><br />
                CC BY-NC-ND 4.0
              </p>
              <p>
                <strong>Website Code:</strong><br />
                MIT License
              </p>
              <p className="pt-2">
                <strong>Author:</strong> Greg Partin
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-6 border-t border-space-border text-center text-sm text-text-muted">
          <p>
            © {new Date().getFullYear()} Emergent Physics Lab. All physics simulations run authentic LFM code.
          </p>
          <p className="mt-2">
            This website demonstrates real scientific research. All claims are backed by reproducible tests.
          </p>
        </div>
      </div>
    </footer>
  );
}

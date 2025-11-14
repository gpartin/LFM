/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Research experiments browser - test harness validation suite
 */

'use client';

import React, { useState } from 'react';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import { getAllExperiments, getResearchExperimentsByTier } from '@/data/experiments';

export default function ResearchPage() {
  const [selectedTier, setSelectedTier] = useState<number | null>(null);
  
  const allExperiments = getAllExperiments();
  const researchExperiments = allExperiments.filter(exp => exp.type === 'RESEARCH');
  
  // Group by tier
  const experimentsByTier: Record<number, typeof researchExperiments> = {};
  researchExperiments.forEach(exp => {
    if (exp.tier) {
      if (!experimentsByTier[exp.tier]) {
        experimentsByTier[exp.tier] = [];
      }
      experimentsByTier[exp.tier].push(exp);
    }
  });
  
  const tierNames: Record<number, string> = {
    1: 'Relativistic',
    2: 'Gravity',
    3: 'Energy',
    4: 'Quantization',
    5: 'Electromagnetic',
    6: 'Coupling',
    7: 'Thermodynamics',
  };
  
  // Sort experiments alphabetically by test ID
  const selectedExperiments = (selectedTier 
    ? getResearchExperimentsByTier(selectedTier) 
    : researchExperiments
  ).sort((a, b) => a.id.localeCompare(b.id));
  
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 pt-20">
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white p-4 sm:p-6 md:p-8">
          <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold mb-2">Research Experiments</h1>
          <p className="text-slate-300 text-base sm:text-lg">
            {researchExperiments.length} research experiments from the test harness
          </p>
          {/* Sorting text intentionally omitted per product guidance */}
        </div>
        
        {/* Validation System Coming Soon Notice */}
        <div className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 border border-blue-500/30 rounded-lg p-4 sm:p-6 mb-6 sm:mb-8">
          <div className="flex flex-col sm:flex-row items-start gap-3 sm:gap-4">
            <div className="text-3xl sm:text-4xl">🔬</div>
            <div className="flex-1">
              <h3 className="text-lg sm:text-xl font-bold text-blue-300 mb-2">
                Cryptographic Validation System - Coming Soon
              </h3>
              <p className="text-sm sm:text-base text-gray-300 mb-3">
                These experiments currently demonstrate the physics qualitatively using WebGPU. 
                Soon, you'll be able to run the <strong>exact same Python code</strong> as our test harness 
                and receive cryptographically signed validation certificates.
              </p>
              <ul className="text-sm text-gray-400 space-y-1 mb-4">
                <li>✓ Run identical code via WebAssembly (Pyodide)</li>
                <li>✓ Cryptographic signatures with RFC 3161 timestamps</li>
                <li>✓ Your validations help establish scientific consensus</li>
                <li>✓ Tamper-proof validation registry</li>
              </ul>
              <a 
                href="https://github.com/gpartin/LFM/blob/main/analysis/validation_system_architecture.md"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-blue-400 hover:text-blue-300 underline"
              >
                Learn about the validation system →
              </a>
            </div>
          </div>
        </div>
        
        {/* Tier Selector */}
        <div className="mb-6 sm:mb-8 flex gap-2 flex-wrap">
          <button
            onClick={() => setSelectedTier(null)}
            className={`px-3 sm:px-4 py-2 text-sm sm:text-base rounded-lg font-semibold transition-all ${
              selectedTier === null 
                ? 'bg-indigo-600 text-white' 
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            All Tiers
          </button>
          {Object.entries(tierNames).map(([tier, name]) => (
            <button
              key={tier}
              onClick={() => setSelectedTier(parseInt(tier))}
              className={`px-3 sm:px-4 py-2 text-sm sm:text-base rounded-lg font-semibold transition-all ${
                selectedTier === parseInt(tier) 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <span className="hidden sm:inline">Tier {tier}: {name} </span>
              <span className="sm:hidden">T{tier} </span>
              ({experimentsByTier[parseInt(tier)]?.length || 0})
            </button>
          ))}
        </div>
        
        {/* Experiments Grid */}
        <div className="grid gap-4">
          {selectedExperiments.map(exp => (
            <a
              key={exp.id}
              href={`/research/${encodeURIComponent(exp.id)}`}
              className="block bg-slate-800/50 border border-slate-700 rounded-lg p-4 sm:p-6 hover:bg-slate-800 transition-all"
            >
              <div className="flex flex-col sm:flex-row items-start justify-between gap-3">
                <div className="flex-1 w-full sm:w-auto">
                  <div className="flex flex-wrap items-center gap-2 sm:gap-3 mb-2">
                    <h3 className="text-lg sm:text-xl font-bold text-white">{exp.testId}</h3>
                    <span className="px-2 py-1 bg-indigo-600/30 text-indigo-300 text-xs sm:text-sm rounded">
                      Tier {exp.tier}
                    </span>
                    <span className="px-2 py-1 bg-slate-700 text-slate-300 text-xs sm:text-sm rounded">
                      {exp.simulation}
                    </span>
                  </div>
                  <p className="text-sm sm:text-base text-slate-300 mb-2">{exp.tagline}</p>
                  <div className="text-xs sm:text-sm text-slate-400 space-y-1">
                    <div>Lattice: {exp.initialConditions.latticeSize}³, dt: {exp.initialConditions.dt}, dx: {exp.initialConditions.dx}</div>
                    <div>Steps: {exp.initialConditions.steps}, χ: {exp.initialConditions.chi}</div>
                  </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <span className={`px-3 py-1 rounded text-sm font-semibold ${
                    exp.status === 'production' ? 'bg-green-600/30 text-green-300' :
                    exp.status === 'beta' ? 'bg-yellow-600/30 text-yellow-300' :
                    exp.status === 'development' ? 'bg-orange-600/30 text-orange-300' :
                    'bg-slate-600/30 text-slate-300'
                  }`}>
                    {exp.status}
                  </span>
                </div>
              </div>
            </a>
          ))}
          </div>
        </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

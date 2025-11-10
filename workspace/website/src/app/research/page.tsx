/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Research experiments browser - test harness validation suite
 */

'use client';

import React, { useState } from 'react';
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
  
  const selectedExperiments = selectedTier 
    ? getResearchExperimentsByTier(selectedTier) 
    : researchExperiments;
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Research Experiments</h1>
          <p className="text-slate-300 text-lg">
            {researchExperiments.length} validation tests from test harness
          </p>
        </div>
        
        {/* Tier Selector */}
        <div className="mb-8 flex gap-2 flex-wrap">
          <button
            onClick={() => setSelectedTier(null)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
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
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                selectedTier === parseInt(tier) 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Tier {tier}: {name} ({experimentsByTier[parseInt(tier)]?.length || 0})
            </button>
          ))}
        </div>
        
        {/* Experiments Grid */}
        <div className="grid gap-4">
          {selectedExperiments.map(exp => (
            <div
              key={exp.id}
              className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 hover:bg-slate-800 transition-all"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-xl font-bold text-white">{exp.testId}</h3>
                    <span className="px-2 py-1 bg-indigo-600/30 text-indigo-300 text-sm rounded">
                      Tier {exp.tier}
                    </span>
                    <span className="px-2 py-1 bg-slate-700 text-slate-300 text-sm rounded">
                      {exp.simulation}
                    </span>
                  </div>
                  <p className="text-slate-300 mb-2">{exp.tagline}</p>
                  <div className="text-sm text-slate-400">
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
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

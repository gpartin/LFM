/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Validation Dashboard — batch validation status for all 105 experiments
 */

'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { getAllExperiments } from '@/data/experiments';
import type { ExperimentDefinition } from '@/data/experiments';

interface ValidationStatus {
  experimentId: string;
  status: 'pass' | 'fail' | 'unknown';
  timestamp?: string;
  hash?: string;
}

export default function DashboardPage() {
  const [statuses, setStatuses] = useState<Map<string, ValidationStatus>>(new Map());
  const experiments = getAllExperiments().filter(exp => exp.type === 'RESEARCH');
  
  useEffect(() => {
    // Load certification files from local storage or API
    // For now, initialize with 'unknown' status
    const initialStatuses = new Map<string, ValidationStatus>();
    experiments.forEach(exp => {
      initialStatuses.set(exp.id, {
        experimentId: exp.id,
        status: 'unknown',
      });
    });
    setStatuses(initialStatuses);
  }, []);
  
  // Group by tier
  const byTier: Record<number, ExperimentDefinition[]> = {};
  experiments.forEach(exp => {
    if (exp.tier) {
      if (!byTier[exp.tier]) byTier[exp.tier] = [];
      byTier[exp.tier].push(exp);
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
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Validation Dashboard</h1>
          <p className="text-slate-300 text-lg">
            Batch validation status for all {experiments.length} research experiments
          </p>
        </div>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-slate-900/60 rounded-lg border border-slate-700 p-4">
            <div className="text-2xl font-bold text-green-400">
              {Array.from(statuses.values()).filter(s => s.status === 'pass').length}
            </div>
            <div className="text-sm text-slate-400">Passed</div>
          </div>
          <div className="bg-slate-900/60 rounded-lg border border-slate-700 p-4">
            <div className="text-2xl font-bold text-red-400">
              {Array.from(statuses.values()).filter(s => s.status === 'fail').length}
            </div>
            <div className="text-sm text-slate-400">Failed</div>
          </div>
          <div className="bg-slate-900/60 rounded-lg border border-slate-700 p-4">
            <div className="text-2xl font-bold text-slate-400">
              {Array.from(statuses.values()).filter(s => s.status === 'unknown').length}
            </div>
            <div className="text-sm text-slate-400">Not Run</div>
          </div>
          <div className="bg-slate-900/60 rounded-lg border border-slate-700 p-4">
            <div className="text-2xl font-bold text-indigo-400">{experiments.length}</div>
            <div className="text-sm text-slate-400">Total Tests</div>
          </div>
        </div>
        
        {/* Tier-by-Tier Grid */}
        {Object.entries(byTier).map(([tier, exps]) => (
          <div key={tier} className="mb-8">
            <h2 className="text-2xl font-bold mb-4">
              Tier {tier}: {tierNames[parseInt(tier)]} ({exps.length} tests)
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {exps.map(exp => {
                const status = statuses.get(exp.id);
                const color = 
                  status?.status === 'pass' ? 'border-green-600 bg-green-900/20' :
                  status?.status === 'fail' ? 'border-red-600 bg-red-900/20' :
                  'border-slate-600 bg-slate-800/40';
                return (
                  <Link
                    key={exp.id}
                    href={`/research/${exp.id}`}
                    className={`p-3 rounded-lg border ${color} hover:bg-slate-700/40 transition-all`}
                  >
                    <div className="font-mono text-sm font-semibold">{exp.testId || exp.id}</div>
                    <div className="text-xs text-slate-400 mt-1">{status?.status || 'unknown'}</div>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { ExperimentDefinition } from '@/data/experiments';
import { CheckCircle2, XCircle, AlertCircle, Loader2 } from 'lucide-react';

interface ValidationPanelProps {
  experiment: ExperimentDefinition;
  status: 'idle' | 'running' | 'pass' | 'fail';
  results: any;
  onValidate: () => void;
}

export default function ValidationPanel({
  experiment,
  status,
  results,
  onValidate
}: ValidationPanelProps) {
  
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white">Validation</h3>
        
        {/* Status Badge */}
        {status === 'pass' && (
          <div className="flex items-center gap-1 text-green-400">
            <CheckCircle2 className="w-4 h-4" />
            <span className="text-sm font-semibold">PASS</span>
          </div>
        )}
        {status === 'fail' && (
          <div className="flex items-center gap-1 text-red-400">
            <XCircle className="w-4 h-4" />
            <span className="text-sm font-semibold">FAIL</span>
          </div>
        )}
        {status === 'running' && (
          <div className="flex items-center gap-1 text-yellow-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm font-semibold">RUNNING</span>
          </div>
        )}
      </div>
      
      <p className="text-sm text-text-secondary mb-4">
        Run validation to compare web simulation against test harness results.
      </p>
      
      <button
        onClick={onValidate}
        disabled={status === 'running'}
        className="w-full px-4 py-2 bg-accent-chi hover:bg-accent-chi/80 
                   disabled:bg-slate-700 disabled:text-text-muted disabled:cursor-not-allowed
                   rounded-lg text-white font-semibold transition-colors"
      >
        {status === 'running' ? 'Running Validation...' : 'Run Validation'}
      </button>
      
      {/* Validation Results */}
      {results && (
        <div className="mt-4 pt-4 border-t border-slate-700">
          <h4 className="text-sm font-semibold text-text-secondary mb-3">Results</h4>
          
          {/* Validation Details from backend */}
          {results.validationDetails && results.validationDetails.length > 0 && (
            <div className="mb-3 space-y-1">
              {results.validationDetails.map((detail: string, idx: number) => {
                const isPass = detail.includes('✓');
                return (
                  <div key={idx} className={`text-xs p-2 rounded ${
                    isPass ? 'bg-green-500/10 text-green-300' : 'bg-red-500/10 text-red-300'
                  }`}>
                    {detail}
                  </div>
                );
              })}
            </div>
          )}
          
          {/* UI Metrics vs Python Metrics Comparison */}
          {results.uiMetrics && (
            <div className="space-y-3">
              <div className="text-xs font-semibold text-text-secondary">UI Simulation</div>
              <div className="space-y-2">
                {Object.entries(results.uiMetrics).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-text-muted font-mono">{key}</span>
                    <span className="text-white font-mono">
                      {typeof value === 'number' ? (value as number).toExponential(3) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {results.pythonMetrics && Object.keys(results.pythonMetrics).length > 0 && (
            <div className="space-y-3 mt-3 pt-3 border-t border-slate-700">
              <div className="text-xs font-semibold text-text-secondary">Python Harness</div>
              <div className="space-y-2">
                {Object.entries(results.pythonMetrics).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-text-muted font-mono">{key}</span>
                    <span className="text-white font-mono">
                      {typeof value === 'number' ? (value as number).toExponential(3) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {results.duration && (
            <div className="mt-3 text-xs text-text-muted">
              Duration: {results.duration.toFixed(2)}s
            </div>
          )}
          
          {results.notes && (
            <div className="mt-3 p-2 bg-slate-700/50 rounded text-xs text-text-secondary">
              {results.notes}
            </div>
          )}
        </div>
      )}
      
      {/* Info Notice */}
      <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded flex gap-2">
        <AlertCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-blue-300">
          Validation runs the test harness simulation and writes diagnostic results to 
          <code className="mx-1 px-1 bg-slate-900 rounded">
            workspace/results/website_validation/{experiment.testId || experiment.id}/
          </code>
        </p>
      </div>
    </div>
  );
}

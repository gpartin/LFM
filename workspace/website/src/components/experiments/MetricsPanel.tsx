/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { ExperimentDefinition } from '@/data/experiments';

interface MetricsPanelProps {
  experiment: ExperimentDefinition;
  metrics: Record<string, number | string>;
  currentStep: number;
}

export default function MetricsPanel({
  experiment,
  metrics,
  currentStep
}: MetricsPanelProps) {
  
  // Format metric value for display
  const formatValue = (value: number | string): string => {
    if (typeof value === 'number') {
      if (Math.abs(value) < 0.001 || Math.abs(value) > 1000) {
        return value.toExponential(3);
      }
      return value.toFixed(4);
    }
    return String(value);
  };
  
  // Check if metric exceeds validation threshold
  const isOutOfBounds = (key: string, value: number): boolean => {
    if (!experiment.validation || typeof value !== 'number') return false;
    
    const validation = experiment.validation as Record<string, any>;
    const threshold = validation[key];
    if (typeof threshold === 'number') {
      return Math.abs(value) > threshold;
    }
    return false;
  };
  
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
      <h3 className="text-lg font-bold text-white mb-4">Metrics</h3>
      
      {Object.keys(metrics).length === 0 ? (
        <p className="text-sm text-text-muted italic">
          No metrics available. Start simulation to see live data.
        </p>
      ) : (
        <div className="space-y-3">
          {Object.entries(metrics).map(([key, value]) => {
            const outOfBounds = isOutOfBounds(key, value as number);
            
            return (
              <div 
                key={key} 
                className={`p-2 rounded ${outOfBounds ? 'bg-red-500/10 border border-red-500/30' : 'bg-slate-700/30'}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-secondary">{key}</span>
                  {outOfBounds && (
                    <span className="text-xs text-red-400">⚠️ Out of bounds</span>
                  )}
                </div>
                <div className="text-xl font-mono text-white mt-1">
                  {formatValue(value)}
                </div>
                
                {/* Show threshold if available */}
                {experiment.validation && (experiment.validation as Record<string, any>)[key] && (
                  <div className="text-xs text-text-muted mt-1">
                    Threshold: {typeof (experiment.validation as Record<string, any>)[key] === 'number' 
                      ? `< ${(experiment.validation as Record<string, any>)[key]}`
                      : (experiment.validation as Record<string, any>)[key]}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      
      {/* Validation Thresholds Summary */}
      {experiment.validation && (
        <div className="mt-6 pt-4 border-t border-slate-700">
          <h4 className="text-sm font-semibold text-text-secondary mb-3">Validation Criteria</h4>
          <div className="space-y-2 text-xs">
            {Object.entries(experiment.validation).map(([key, threshold]) => (
              <div key={key} className="flex items-center justify-between text-text-muted">
                <span className="font-mono">{key}</span>
                <span className="font-mono">
                  {typeof threshold === 'number' ? `< ${threshold}` : String(threshold)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

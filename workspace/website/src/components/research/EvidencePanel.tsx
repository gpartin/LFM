/**
 * EvidencePanel component for research experiments.
 * Displays metadata about validation results, certification, and reproducibility.
 * 
 * Shows:
 * - Latest validation timestamp
 * - Energy conservation results
 * - Primary physics metric
 * - Certification hash
 * - Reproducibility score
 */

'use client';

import React from 'react';
import { ExperimentDefinition } from '@/data/experiments';

interface EvidencePanelProps {
  experiment: ExperimentDefinition;
  /** Latest validation summary (if available) */
  validationSummary?: Record<string, any>;
  /** Number of times this experiment has been validated */
  validationCount?: number;
}

export default function EvidencePanel({ experiment, validationSummary, validationCount = 0 }: EvidencePanelProps) {
  const hasValidation = !!validationSummary;
  
  // Extract key metrics from validation summary
  const energyDrift = validationSummary?.energy_drift || validationSummary?.energyDrift;
  const primaryMetric = validationSummary?.primary_metric || validationSummary?.primaryMetric;
  const passed = validationSummary?.passed;
  const timestamp = validationSummary?.timestamp || validationSummary?.certification_timestamp;
  const hash = validationSummary?.hash || validationSummary?.certification_hash;
  
  // Compute reproducibility score based on available metadata
  const reproducibilityScore = computeReproducibilityScore(experiment, validationSummary);
  
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-6">
      <h3 className="text-lg font-semibold text-white mb-4">🔬 Evidence & Validation</h3>
      {/* Quick links to config/results paths for traceability */}
      <div className="mb-4 grid grid-cols-1 gap-2 text-xs text-white/70">
        {experiment.links?.testHarnessConfig && (
          <div>
            <span className="text-white/50 mr-2">Config:</span>
            <code className="px-1 py-0.5 bg-white/10 rounded break-all" title="Test harness config path">
              {experiment.links.testHarnessConfig}
            </code>
          </div>
        )}
        {experiment.links?.results && (
          <div>
            <span className="text-white/50 mr-2">Results dir:</span>
            <code className="px-1 py-0.5 bg-white/10 rounded break-all" title="Results directory path">
              {experiment.links.results}
            </code>
          </div>
        )}
      </div>
      
      {/* Validation count - always show */}
      <div className="mb-4 pb-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-white/70">Unique Validators</span>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-purple-300">{validationCount}</span>
            <span className="text-sm text-white/50">{validationCount === 1 ? 'validator' : 'validators'}</span>
          </div>
        </div>
        <p className="text-xs text-white/40">
          Each validator has a unique browser fingerprint. Multiple validations from the same 
          validator are tracked but counted as one to prevent spam.
        </p>
      </div>
      
      {hasValidation ? (
        <div className="space-y-4">
          {/* Validation status */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className={`w-3 h-3 rounded-full ${passed ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm font-medium text-white/70">
                Validation Status
              </span>
            </div>
            <div className="text-base text-white">
              {passed ? '✓ Passed' : '✗ Failed'}
            </div>
          </div>
          
          {/* Energy conservation */}
          {energyDrift !== undefined && (
            <div>
              <div className="text-sm font-medium text-white/70 mb-1">Energy Conservation</div>
              <div className="text-base text-white">
                {typeof energyDrift === 'number' ? energyDrift.toExponential(2) : energyDrift}
                <span className="text-sm text-white/50 ml-2">
                  (threshold: {experiment.validation?.energyDrift ?? '1e-4'})
                </span>
              </div>
            </div>
          )}
          
          {/* Primary metric */}
          {primaryMetric !== undefined && (
            <div>
              <div className="text-sm font-medium text-white/70 mb-1">Primary Metric</div>
              <div className="text-base text-white">
                {typeof primaryMetric === 'number' ? primaryMetric.toFixed(4) : primaryMetric}
              </div>
            </div>
          )}
          
          {/* Certification */}
          {hash && (
            <div>
              <div className="text-sm font-medium text-white/70 mb-1">Certification Hash</div>
              <div className="text-xs font-mono text-white/80 break-all">
                {hash}
              </div>
            </div>
          )}
          
          {/* Timestamp */}
          {timestamp && (
            <div>
              <div className="text-sm font-medium text-white/70 mb-1">Validation Time</div>
              <div className="text-sm text-white">
                {new Date(timestamp).toLocaleString()}
              </div>
            </div>
          )}
          
          {/* Reproducibility score */}
          <div className="pt-4 border-t border-white/10">
            <div className="text-sm font-medium text-white/70 mb-2">Reproducibility Score</div>
            <div className="flex items-center gap-3">
              <div className="flex-1 bg-white/10 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-full rounded-full transition-all"
                  style={{ width: `${reproducibilityScore}%` }}
                />
              </div>
              <span className="text-sm text-white font-medium">{reproducibilityScore}%</span>
            </div>
            <div className="text-xs text-white/50 mt-2">
              Based on: config availability, results traceability, documentation completeness,
              certification presence
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="text-white/50 mb-4">No validation data available</div>
          <div className="text-xs text-white/30">
            Run validation to generate evidence and certification
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Compute reproducibility score (0-100) based on available metadata.
 * 
 * Criteria:
 * - Config links: +25 points
 * - Results links: +25 points
 * - Documentation links: +20 points
 * - Certification hash: +20 points
 * - Validation criteria defined: +10 points
 */
function computeReproducibilityScore(
  experiment: ExperimentDefinition,
  validationSummary?: Record<string, any>
): number {
  let score = 0;
  
  // Config availability
  if (experiment.links?.testHarnessConfig) {
    score += 25;
  }
  
  // Results availability
  if (experiment.links?.results) {
    score += 25;
  }
  
  // Documentation availability
  if (experiment.links?.discovery || experiment.links?.documentation) {
    score += 20;
  }
  
  // Certification presence
  const hash = validationSummary?.hash || validationSummary?.certification_hash;
  if (hash) {
    score += 20;
  }
  
  // Validation criteria defined
  // Validation criteria defined (approx via energyDrift presence)
  if (typeof experiment.validation?.energyDrift === 'number') score += 10;
  
  return score;
}

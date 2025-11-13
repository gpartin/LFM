"use client";

/*
 * ResearchExperimentPanel
 * Shared UI for running and certifying research experiments 1:1 with test harness
 */

import React, { useCallback, useMemo, useState } from 'react';
import SimulationDispatcher from './SimulationDispatcher';
import AutoParameterPanel from './AutoParameterPanel';
import DocumentationPanel from './DocumentationPanel';
import EvidencePanel from './EvidencePanel';
import type { ExperimentDefinition } from '@/data/experiments';
import { getValidatorFingerprint, getEnvironmentMetadata } from '@/lib/fingerprint';

type Props = {
  experiment: ExperimentDefinition;
};

type ValidationResponse = {
  ok: boolean;
  message?: string;
  details?: any;
};

export default function ResearchExperimentPanel({ experiment }: Props) {
  const [isValidating, setIsValidating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [result, setResult] = useState<ValidationResponse | null>(null);
  const [validationCount, setValidationCount] = useState<number>(0);
  const [liveMetrics, setLiveMetrics] = useState<{ energy?: number; energyDriftPct?: number; time?: number } | null>(null);

  // Animation state (separate from validation)
  const isAnimating = isValidating || isPlaying;
  
  // Fetch validation count for this experiment
  React.useEffect(() => {
    fetch('/api/certifications')
      .then(res => res.json())
      .then(data => {
        const stats = data.counts?.[experiment.id];
        if (stats) {
          // Use uniqueValidators as the primary count
          setValidationCount(stats.uniqueValidators || stats.totalValidations || 0);
        } else {
          setValidationCount(0);
        }
      })
      .catch(err => console.error('Failed to load validation count:', err));
  }, [experiment.id]);

  // Attestation display - standardized extraction
  const attestation = useMemo(() => {
    if (!result || !result.details) return null;
    
    // Try to extract from certification file or summary
    const certPath = result.details.certificationPath;
    const summary = result.details.summary;
    
    if (!summary) return null;
    
    // Standardized field extraction
    const hash = summary.certification_hash || summary.hash || summary.sha256;
    const timestamp = summary.certification_timestamp || summary.timestamp || summary.time;
    
    if (hash && timestamp) {
      return { path: certPath, hash, timestamp };
    }
    
    return null;
  }, [result]);

  const info = useMemo(() => ({
    lattice: `${experiment.initialConditions.latticeSize}³`,
    dt: experiment.initialConditions.dt,
    dx: experiment.initialConditions.dx,
    steps: experiment.initialConditions.steps,
    chi: experiment.initialConditions.chi,
  }), [experiment]);

  const runValidation = useCallback(async () => {
    if (!experiment.id) return;
    setIsPlaying(false); // Override play state
    setIsValidating(true);
    setResult(null);
    try {
      console.log(`[Validation] Starting validation for ${experiment.id}`);
      
      // Get validator fingerprint and environment
      const validatorFingerprint = await getValidatorFingerprint();
      const environmentMeta = getEnvironmentMetadata();
      
      // Check if user already validated this experiment
      const checkRes = await fetch(`/api/experiments/${encodeURIComponent(experiment.id)}/check-validation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ validatorFingerprint }),
      });
      const checkData = await checkRes.json();
      
      if (checkData.alreadyValidated) {
        console.log(`[Validation] Already validated by this validator`);
        setResult({
          ok: true,
          message: 'You have already validated this experiment. Your validation has been recorded.',
          details: {
            summary: checkData.certification.validation,
            certificationPath: checkData.certificationPath,
            alreadyValidated: true,
          },
        });
        setIsValidating(false);
        return;
      }
      
      const res = await fetch(`/api/experiments/${encodeURIComponent(experiment.id)}/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          validatorFingerprint,
          environment: {
            ...environmentMeta,
            // Python environment will be added by backend
            latticeSize: experiment.initialConditions.latticeSize,
            dt: experiment.initialConditions.dt,
            dx: experiment.initialConditions.dx,
          },
        }),
      });
      console.log(`[Validation] Response status: ${res.status}`);
      const json = await res.json();
      console.log(`[Validation] Response:`, json);
      setResult(json as ValidationResponse);
      
      // Refetch validation count after successful validation
      if (json.ok) {
        fetch('/api/certifications')
          .then(res => res.json())
          .then(data => {
            const stats = data.counts?.[experiment.id];
            if (stats) {
              setValidationCount(stats.uniqueValidators || stats.totalValidations || 0);
            }
          })
          .catch(err => console.error('Failed to reload validation count:', err));
      }
    } catch (err: any) {
      console.error('[Validation] Error:', err);
      setResult({ ok: false, message: `Validation failed: ${err?.message || String(err)}` });
    } finally {
      setIsValidating(false);
    }
  }, [experiment.id, experiment.initialConditions]);

  const togglePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-6 items-start">
      {/* Left: Elegant multi-canvas layout - FIXED HEIGHT */}
      <div className="space-y-4 sticky top-6">
        <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-4 h-[560px]">
          {/* Primary canvas (large) — injected by SimulationDispatcher */}
          <div className="relative bg-slate-900/60 rounded-lg border border-slate-700 overflow-hidden md:row-span-2">
            <div className="absolute top-0 left-0 right-0 px-3 py-2 text-xs font-semibold tracking-wide text-indigo-200 bg-indigo-950/50 border-b border-slate-700 z-10">
              Primary Simulation — {experiment.simulation}
            </div>
            <div className="absolute inset-0 pt-8">
              <SimulationDispatcher 
                experiment={experiment} 
                isRunning={isAnimating} 
                key={`sim-${experiment.id}`}
                onMetrics={(m) => setLiveMetrics(m)}
              />
            </div>
            
            {/* Play/Pause control overlay */}
            <div className="absolute bottom-4 left-4 z-20">
              <button
                onClick={togglePlayPause}
                disabled={isValidating}
                className={`
                  px-4 py-2 rounded-lg font-medium text-sm shadow-lg transition-all
                  ${isValidating 
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed' 
                    : isPlaying
                      ? 'bg-red-500 hover:bg-red-600 text-white'
                      : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                  }
                `}
                title={isValidating ? 'Validation running...' : isPlaying ? 'Pause simulation' : 'Play simulation'}
              >
                {isValidating ? '⏳ Validating...' : isPlaying ? '⏸ Pause' : '▶️ Play'}
              </button>
            </div>
          </div>

          {/* Auxiliary canvas (top-right) */}
          <div className="relative bg-slate-900/60 rounded-lg border border-slate-700 overflow-hidden">
            <div className="absolute top-0 left-0 right-0 px-3 py-2 text-xs font-semibold tracking-wide text-slate-200 bg-slate-950/50 border-b border-slate-700 z-10">
              Auxiliary View — χ / Grid
            </div>
            <div className="absolute inset-0 pt-8 flex items-center justify-center text-slate-400 text-xs">
              Secondary visualization (e.g., chi field, lattice)
            </div>
          </div>

          {/* Diagnostics canvas (bottom-right) */}
          <div className="relative bg-slate-900/60 rounded-lg border border-slate-700 overflow-hidden">
            <div className="absolute top-0 left-0 right-0 px-3 py-2 text-xs font-semibold tracking-wide text-slate-200 bg-slate-950/50 border-b border-slate-700 z-10">
              Diagnostics — Spectrum / Energy
            </div>
            <div className="absolute inset-0 pt-8 flex items-center justify-center text-slate-300 text-xs">
              {liveMetrics ? (
                <div className="space-y-1 text-center">
                  {typeof liveMetrics.energyDriftPct === 'number' && (
                    <div>Energy drift (preview): <span className="font-mono">{liveMetrics.energyDriftPct.toFixed(3)}%</span></div>
                  )}
                  {typeof liveMetrics.energy === 'number' && (
                    <div>Total energy (preview): <span className="font-mono">{liveMetrics.energy.toExponential(3)}</span></div>
                  )}
                  {typeof liveMetrics.time === 'number' && (
                    <div>Time: <span className="font-mono">{liveMetrics.time.toFixed(3)} s</span></div>
                  )}
                  <div className="text-[10px] text-slate-500 mt-1">
                    Preview-only metrics. Official validation uses Python/CuPy results.
                  </div>
                </div>
              ) : (
                <div className="text-slate-500">Spectrum, energy timeline, or FFT diagnostics</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Right: Validation + Controls + Documentation - SCROLLABLE */}
      <div className="space-y-4 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2">
        {/* Validation - MOVED TO TOP */}
        <div className="bg-slate-900/60 rounded-lg border border-slate-700 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold">Run Validation</h3>
            <span className="text-xs text-slate-400">GPU-accelerated</span>
          </div>
          <p className="text-slate-300 text-sm mb-3">
            Validate this test against the official LFM physics harness. Results include a cryptographic 
            certification with SHA-256 hash and ISO-8601 timestamp for tamper-evident validation.
          </p>
          <div className="flex gap-2">
            <button
              className={`w-full px-4 py-3 rounded-lg font-semibold transition-colors ${
                isValidating 
                  ? 'bg-slate-700 text-slate-300 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'
              }`}
              onClick={runValidation}
              disabled={isValidating}
            >
              {isValidating ? '⏳ Running Validation...' : '▶️ Run Validation'}
            </button>
          </div>

          {result && (
            <div className={`mt-4 rounded border p-3 ${result.ok ? 'border-green-700 bg-green-900/20' : 'border-red-700 bg-red-900/20'}`}>
              <div className="text-sm">
                {result.details?.alreadyValidated ? (
                  <div>
                    <div className="font-semibold mb-2">✅ Already Validated</div>
                    <p className="text-slate-300 mb-2">
                      You have already validated this experiment. Your validation has been recorded 
                      and cannot be duplicated. Thank you for contributing to scientific verification!
                    </p>
                  </div>
                ) : (
                  <div className="font-semibold mb-2">{result.ok ? '✅ Validation PASSED' : '❌ Validation FAILED'}</div>
                )}
                {result.details?.summary && (
                  <div className="space-y-1 text-slate-200">
                    <div>Energy Drift: <span className="font-mono text-xs">{result.details.summary.energy_drift?.toExponential(2)}</span></div>
                    {result.details.summary.primary_metric !== undefined && (
                      <div>Primary Metric: <span className="font-mono text-xs">{result.details.summary.primary_metric?.toExponential(2)}</span> ({result.details.summary.metric_name})</div>
                    )}
                    {result.details.summary.runtime_sec && (
                      <div>Runtime: <span className="font-mono text-xs">{result.details.summary.runtime_sec.toFixed(2)}s</span></div>
                    )}
                    {result.details.summary.notes && (
                      <div className="text-xs text-slate-300 mt-1">{result.details.summary.notes}</div>
                    )}
                  </div>
                )}
                {attestation && (
                  <div className="mt-3 p-2 rounded bg-slate-950/70 border border-indigo-700">
                    <div className="font-semibold text-indigo-300 mb-1">🔒 Certification</div>
                    <div className="text-xs text-slate-300 mb-1">Hash: <span className="font-mono break-all">{attestation.hash.substring(0, 16)}...</span></div>
                    <div className="text-xs text-slate-300 mb-1">Timestamp: <span className="font-mono">{new Date(attestation.timestamp).toLocaleString()}</span></div>
                    {result.details?.certificationPath && (
                      <div className="text-xs text-slate-400">ID: <span className="font-mono">{result.details.certificationPath}</span></div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Parameters - Auto-generated */}
        <div className="bg-slate-900/60 rounded-lg border border-slate-700 p-4">
          <h3 className="text-lg font-semibold mb-3">Experiment Parameters</h3>
          <AutoParameterPanel 
            experiment={experiment} 
            onParameterChange={(key, value) => {
              console.log(`Parameter ${key} changed to ${value}`);
              // TODO: update local state and trigger rerun
            }}
          />
        </div>

        {/* Documentation Panel */}
        <DocumentationPanel experiment={experiment} />

        {/* Evidence Panel */}
        <EvidencePanel 
          experiment={experiment} 
          validationSummary={result?.details?.summary}
          validationCount={validationCount}
        />
      </div>
    </div>
  );
}

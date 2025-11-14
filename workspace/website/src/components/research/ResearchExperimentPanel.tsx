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
  status?: string; // optional status codes like 'disabled'
};

export default function ResearchExperimentPanel({ experiment }: Props) {
  const [isValidating, setIsValidating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [result, setResult] = useState<ValidationResponse | null>(null);
  const [validationCount, setValidationCount] = useState<number>(0);
  const [liveMetrics, setLiveMetrics] = useState<{ energy?: number; energyDriftPct?: number; time?: number } | null>(null);
  const [speedFactor, setSpeedFactor] = useState<number>(1.0);
  const [resetCounter, setResetCounter] = useState<number>(0);

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

  // Derive UI metadata (manual SHOWCASE or generated RESEARCH)
  const ui = experiment.ui || { panels: ['metrics','energy'], controls: { play: true, reset: true }, metrics: { showEnergyDrift: true } };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-6 items-start">
      {/* Left: Elegant multi-canvas layout - FIXED HEIGHT */}
      <div className="space-y-4 sticky top-6">
        <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-4 h-[560px]">
          {/* Primary canvas (large) — injected by SimulationDispatcher */}
          <div className="relative bg-slate-900/60 rounded-lg border border-slate-700 overflow-hidden md:row-span-2">
            <div className="absolute top-0 left-0 right-0 px-3 py-2 text-xs font-semibold tracking-wide text-indigo-200 bg-indigo-950/50 border-b border-slate-700 z-10 flex items-center justify-between">
              <span>Primary Simulation — {experiment.simulation}</span>
              {ui.controls.speedSlider && (
                <div className="flex items-center gap-2 text-[10px]">
                  <label htmlFor="speedRange" className="text-slate-300">Speed</label>
                  <input
                    id="speedRange"
                    type="range"
                    min={0.25}
                    max={2.0}
                    step={0.25}
                    value={speedFactor}
                    className="accent-indigo-400"
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setSpeedFactor(v);
                      console.log(`Speed factor set to ${v}`);
                    }}
                  />
                </div>
              )}
            </div>
            <div className="absolute inset-0 pt-8">
              <SimulationDispatcher 
                experiment={experiment} 
                isRunning={isAnimating} 
                key={`sim-${experiment.id}-${resetCounter}`}
                onMetrics={(m) => setLiveMetrics(m)}
                speedFactor={speedFactor}
                resetCounter={resetCounter}
              />
            </div>
            
            {/* Play/Pause control overlay */}
            {ui.controls.play && (
              <div className="absolute bottom-4 left-4 z-20 flex gap-2">
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
                {ui.controls.reset && (
                  <button
                    onClick={() => {
                      console.log('Reset simulation requested');
                      setIsPlaying(false);
                      setLiveMetrics(null);
                      setResetCounter(c => c + 1);
                    }}
                    className="px-4 py-2 rounded-lg font-medium text-sm shadow-lg bg-slate-600 hover:bg-slate-500 text-white"
                    title="Reset simulation"
                  >
                    🔄 Reset
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Auxiliary canvas (top-right) */}
          <div className="relative bg-slate-900/60 rounded-lg border border-slate-700 overflow-hidden">
            <div className="absolute top-0 left-0 right-0 px-3 py-2 text-xs font-semibold tracking-wide text-slate-200 bg-slate-950/50 border-b border-slate-700 z-10">
              {ui.panels.includes('wave') && !ui.panels.includes('field') ? 'Wave Diagnostics' : 'Auxiliary View'}
            </div>
            <div className="absolute inset-0 pt-8 flex items-center justify-center text-slate-400 text-xs">
              {ui.panels.includes('field') && ui.controls.chiToggle && (
                <div className="text-center space-y-1">
                  <div>χ Field Overlay Enabled</div>
                  {Array.isArray(experiment.initialConditions.chi) && (
                    <div className="text-[10px] text-slate-500">Gradient: {experiment.initialConditions.chi[0]} → {experiment.initialConditions.chi[1]}</div>
                  )}
                </div>
              )}
              {!ui.panels.includes('field') && ui.panels.includes('wave') && (
                <div className="text-center space-y-1">
                  <div>Wave Packet Preview</div>
                  {experiment.initialConditions.wavePacket && (
                    <div className="text-[10px] text-slate-500">k = {experiment.initialConditions.wavePacket.k.join(', ')}</div>
                  )}
                </div>
              )}
              {!ui.panels.includes('field') && !ui.panels.includes('wave') && 'Secondary visualization (e.g., chi field, lattice)'}
            </div>
          </div>

          {/* Diagnostics canvas (bottom-right) */}
          <div className="relative bg-slate-900/60 rounded-lg border border-slate-700 overflow-hidden">
            <div className="absolute top-0 left-0 right-0 px-3 py-2 text-xs font-semibold tracking-wide text-slate-200 bg-slate-950/50 border-b border-slate-700 z-10">
              Diagnostics — {ui.panels.includes('particles') ? 'Orbit / Energy' : 'Spectrum / Energy'}
            </div>
            <div className="absolute inset-0 pt-8 flex items-center justify-center text-slate-300 text-xs">
              {liveMetrics ? (
                <div className="space-y-1 text-center">
                  {ui.metrics?.showEnergyDrift && typeof liveMetrics.energyDriftPct === 'number' && (
                    <div>Energy drift (preview): <span className="font-mono">{liveMetrics.energyDriftPct.toFixed(3)}%</span></div>
                  )}
                  {typeof liveMetrics.energy === 'number' && (
                    <div>Total energy (preview): <span className="font-mono">{liveMetrics.energy.toExponential(3)}</span></div>
                  )}
                  {ui.metrics?.showOrbitElements && (
                    <div className="text-[10px] text-slate-500">Orbit elements preview unavailable in static mode</div>
                  )}
                  {ui.metrics?.showTransmission && (
                    <div className="text-[10px] text-slate-500">Transmission/Reflection computed during play (placeholder)</div>
                  )}
                  {typeof liveMetrics.time === 'number' && (
                    <div>Time: <span className="font-mono">{liveMetrics.time.toFixed(3)} s</span></div>
                  )}
                  <div className="text-[10px] text-slate-500 mt-1">
                    Preview-only metrics. Official validation uses Python/CuPy results.
                  </div>
                </div>
              ) : (
                <div className="text-slate-500">{ui.panels.includes('particles') ? 'Orbit metrics / energy timeline' : 'Spectrum, energy timeline, or FFT diagnostics'}</div>
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
            <h3 className="text-lg font-semibold">Validation (Coming Soon)</h3>
            <span className="text-xs text-slate-400">Static mode</span>
          </div>
          <div className="rounded-md border border-yellow-700 bg-yellow-900/20 p-3 mb-3">
            <div className="text-sm font-semibold text-yellow-300 mb-1">Live Harness Execution Disabled</div>
            <p className="text-xs text-yellow-100 leading-relaxed">
              This deployment does not run the GPU physics harness directly. A future update will enable
              reproducible live validation runs with cryptographic certification. For now, published results
              are static certified artifacts produced on an internal GPU system. Reproduce locally using the
              repository, config, and commands below.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              className="w-full px-4 py-3 rounded-lg font-semibold bg-slate-700 text-slate-400 cursor-not-allowed"
              disabled
              title="Live validation is not available in this deployment"
            >
              🚫 Validation Unavailable
            </button>
          </div>

          {result && (
            <div
              className={`mt-4 rounded border p-3 ${
                result.ok
                  ? 'border-green-700 bg-green-900/20'
                  : result.status === 'disabled'
                    ? 'border-yellow-700 bg-yellow-900/20'
                    : 'border-red-700 bg-red-900/20'
              }`}
            >
              <div className="text-sm">
                {result.status === 'disabled' ? (
                  <div>
                    <div className="font-semibold mb-2">⚠️ Validation Disabled</div>
                    <p className="text-slate-300 mb-2">
                      Live harness execution is disabled in this deployment. Displaying static evidence
                      {result.details?.summary?.static_mode && ' (precomputed summary)'} when available.
                    </p>
                    <p className="text-xs text-slate-400">
                      Reason: {result.details?.disabledReason || 'Not provided'}
                    </p>
                  </div>
                ) : result.details?.alreadyValidated ? (
                  <div>
                    <div className="font-semibold mb-2">✅ Already Validated</div>
                    <p className="text-slate-300 mb-2">
                      You have already validated this experiment. Your validation has been recorded
                      and cannot be duplicated. Thank you for contributing to scientific verification!
                    </p>
                  </div>
                ) : (
                  <div className="font-semibold mb-2">
                    {result.ok ? '✅ Validation PASSED' : '❌ Validation FAILED'}
                  </div>
                )}
                {result.details?.summary && (
                  <div className="space-y-1 text-slate-200">
                    <div>
                      Energy Drift:{' '}
                      <span className="font-mono text-xs">
                        {result.details.summary.energy_drift?.toExponential(2)}
                      </span>
                      {result.details.summary.execution_disabled && (
                        <span className="ml-2 text-[10px] text-yellow-400">(static)</span>
                      )}
                    </div>
                    {result.details.summary.primary_metric !== undefined && (
                      <div>
                        Primary Metric:{' '}
                        <span className="font-mono text-xs">
                          {result.details.summary.primary_metric?.toExponential(2)}
                        </span>{' '}
                        ({result.details.summary.metric_name})
                      </div>
                    )}
                    {result.details.summary.runtime_sec && (
                      <div>
                        Runtime:{' '}
                        <span className="font-mono text-xs">
                          {result.details.summary.runtime_sec.toFixed(2)}s
                        </span>
                      </div>
                    )}
                    {result.details.summary.notes && (
                      <div className="text-xs text-slate-300 mt-1">
                        {result.details.summary.notes}
                      </div>
                    )}
                  </div>
                )}
                {attestation && result.status !== 'disabled' && (
                  <div className="mt-3 p-2 rounded bg-slate-950/70 border border-indigo-700">
                    <div className="font-semibold text-indigo-300 mb-1">🔒 Certification</div>
                    <div className="text-xs text-slate-300 mb-1">
                      Hash:{' '}
                      <span className="font-mono break-all">
                        {attestation.hash.substring(0, 16)}...
                      </span>
                    </div>
                    <div className="text-xs text-slate-300 mb-1">
                      Timestamp:{' '}
                      <span className="font-mono">
                        {new Date(attestation.timestamp).toLocaleString()}
                      </span>
                    </div>
                    {result.details?.certificationPath && (
                      <div className="text-xs text-slate-400">
                        ID:{' '}
                        <span className="font-mono">{result.details.certificationPath}</span>
                      </div>
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

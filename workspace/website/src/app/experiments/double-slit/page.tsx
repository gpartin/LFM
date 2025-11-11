/*
 * Double Slit — Showcase Experiment (QUAN)
 *
 * Deterministic wave interference from two apertures. Shows screen intensity
 * and live metrics: fringe spacing, visibility, slit intensity ratio.
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import QuantumVisualizationOptions from '@/components/experiment/QuantumVisualizationOptions';
import StandardMetricsPanel from '@/components/experiment/StandardMetricsPanel';
import ParameterSlider from '@/components/ui/ParameterSlider';
import FieldInterferenceCanvas from '@/components/visuals/FieldInterferenceCanvas';
import { detectBackend } from '@/physics/core/backend-detector';
import { decideSimulationProfile } from '@/physics/core/simulation-profile';
import { useSimulationState } from '@/hooks/useSimulationState';
import { DoubleSlitSimulation } from '@/physics/quantum/double_slit_simulation';

export default function DoubleSlitPage() {
  const [state, dispatch] = useSimulationState();
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  const [uiMode, setUiMode] = useState<'advanced' | 'simple'>('advanced');

  const simRef = useRef<any>(null);
  const lastMetricsRef = useRef<{ fringeSpacing?: string; visibility?: string; slitIntensityRatio?: string } | null>(null);
  const applyTimerRef = useRef<number | null>(null);

  // Local parameter state (applied via reset debounce)
  const [slitSeparation, setSlitSeparation] = useState<number>(14);
  const [slitWidth, setSlitWidth] = useState<number>(6);
  const [k0, setK0] = useState<number>(6.0);
  const [sigma, setSigma] = useState<number>(8.0);

  const stopSimulation = useCallback(() => {
    dispatch({ type: 'SET_RUNNING', payload: false });
  }, [dispatch]);

  // Backend detection → UI profile
  // NOTE: Double Slit requires CPU backend for now (field readback not yet implemented for WebGPU)
  useEffect(() => {
    detectBackend().then((caps) => {
      const effective = 'cpu'; // Force CPU until WebGPU field readback is implemented
      dispatch({ type: 'SET_BACKEND', payload: { backend: effective, capabilities: caps } });
      setBackend(effective);
      const prof = decideSimulationProfile(effective, caps, 'quantum');
      setUiMode(prof.ui);
    });
  }, [dispatch]);

  // Initialize simulation
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        let device: GPUDevice | null = null;
        if (backend === 'webgpu') {
          const adapter = await navigator.gpu?.requestAdapter();
          device = await adapter?.requestDevice() || null;
        }

        const N = backend === 'webgpu' ? 64 : 32;
        const sim = new DoubleSlitSimulation({
          latticeSize: N,
          dx: 0.1,
          dt: 0.001,
          k0,
          sigma,
          slitSeparation: slitSeparation,
          slitWidth: slitWidth,
          barrierThickness: 2,
          apertureX: Math.floor(N * 0.45),
          screenX: Math.min(N - 3, Math.floor(N * 0.80)),
          batchSteps: state.params.simSpeed > 0 ? Math.max(1, Math.floor(state.params.simSpeed / 5)) : 6,
        }, backend === 'webgpu' && device ? { device } : undefined);
        await sim.initialize();
        if (cancelled) return;
        simRef.current = sim;
      } catch (e) {
        console.error('[DoubleSlit] init failed:', e);
      }
    }
    init();
    return () => { cancelled = true; stopSimulation(); simRef.current = null; };
  }, [backend, state.resetTrigger, state.params.simSpeed, stopSimulation, slitSeparation, slitWidth, k0, sigma]);

  // Animation loop
  useEffect(() => {
    if (!state.isRunning) {
      console.log('[DoubleSlit] Animation loop: not running');
      return;
    }
    console.log('[DoubleSlit] Animation loop: starting');
    let rafId: number;
    const tick = async () => {
      const sim = simRef.current;
      if (!sim) {
        console.warn('[DoubleSlit] tick: no simulation ref');
        return;
      }
      await sim.stepBatch();
      const m = sim.getMetrics();
      dispatch({ type: 'UPDATE_METRICS', payload: { energy: m.energy, drift: m.drift } });
      lastMetricsRef.current = { fringeSpacing: m.fringeSpacing, visibility: m.visibility, slitIntensityRatio: m.slitIntensityRatio };
      if (state.isRunning) rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => { 
      console.log('[DoubleSlit] Animation loop: cleanup');
      if (rafId) cancelAnimationFrame(rafId); 
    };
  }, [state.isRunning, dispatch]);

  return (
    <ExperimentLayout
      title="🧪 Quantum Double Slit"
      description="Watch a deterministic wave create an interference pattern through two apertures. No mysticism—just wave physics."
      backend={backend}
      experimentId="double-slit"
      visualizationOptions={
        <QuantumVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
          showAdvancedOptions={true}
        />
      }
      footerContent={
        <div className="mt-8 panel">
          <h2 className="text-2xl font-bold text-accent-chi mb-4">What to Look For</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              A right‑moving wave hits an aperture with two openings. Beyond it, a clear interference pattern forms on the screen.
              The central bright fringe sits midway between the slits; surrounding fringes are roughly equally spaced.
            </p>
            <p>
              Live metrics show fringe spacing (Δy), visibility V = (Imax−Imin)/(Imax+Imin), and the relative intensity from the two slits.
            </p>
          </div>
        </div>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <div className="bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]">
            <FieldInterferenceCanvas
              simulation={simRef}
              isRunning={state.isRunning}
              showGrid={true}
              showWave={state.ui.showWave}
              showBarrier={state.ui.showBarrier}
              showMetricsOverlay={true}
              fringeSpacing={lastMetricsRef.current?.fringeSpacing ?? '—'}
              visibility={lastMetricsRef.current?.visibility ?? '—'}
              slitIntensityRatio={lastMetricsRef.current?.slitIntensityRatio ?? '—'}
              updateInterval={3}
            />
          </div>

          <div className="mt-4 flex gap-4">
            <button
              onClick={() => dispatch({ type: 'SET_RUNNING', payload: !state.isRunning })}
              className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
                state.isRunning
                  ? 'bg-yellow-600 hover:bg-yellow-700'
                  : 'bg-accent-chi hover:bg-accent-chi/80'
              }`}
            >
              {state.isRunning ? '⏸ Pause' : '▶ Play'}
            </button>
            <button
              onClick={() => { simRef.current?.reset?.(); dispatch({ type: 'RESET_METRICS' }); lastMetricsRef.current = null; }}
              className="px-6 py-3 bg-space-border hover:bg-space-border/80 rounded-lg font-semibold transition-colors"
            >
              🔄 Reset
            </button>
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className="panel" data-panel="experiment-parameters">
            <h3 className="text-xl font-bold text-purple-400 mb-1">Experiment Parameters</h3>
            <div className="space-y-6" data-section="profile-parameters">
              <ParameterSlider label="Sim Speed" min={1} max={100} step={1} value={state.params.simSpeed} unit="x" onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })} />
              <ParameterSlider label="Slit Separation" min={4} max={28} step={1} value={slitSeparation} unit="cells" onChange={(v) => {
                setSlitSeparation(v);
                if (applyTimerRef.current) window.clearTimeout(applyTimerRef.current);
                applyTimerRef.current = window.setTimeout(() => { lastMetricsRef.current = null; simRef.current?.updateParams?.({ slitSeparation: v }); simRef.current?.reset?.(); dispatch({ type: 'RESET_METRICS' }); }, 300);
              }} />
              <ParameterSlider label="Slit Width" min={2} max={14} step={1} value={slitWidth} unit="cells" onChange={(v) => {
                setSlitWidth(v);
                if (applyTimerRef.current) window.clearTimeout(applyTimerRef.current);
                applyTimerRef.current = window.setTimeout(() => { lastMetricsRef.current = null; simRef.current?.updateParams?.({ slitWidth: v }); simRef.current?.reset?.(); dispatch({ type: 'RESET_METRICS' }); }, 300);
              }} />
              <ParameterSlider label="k0 (wave number)" min={2} max={12} step={0.5} value={k0} unit="1/dx" onChange={(v) => {
                setK0(v);
                if (applyTimerRef.current) window.clearTimeout(applyTimerRef.current);
                applyTimerRef.current = window.setTimeout(() => { lastMetricsRef.current = null; simRef.current?.updateParams?.({ k0: v }); simRef.current?.reset?.(); dispatch({ type: 'RESET_METRICS' }); }, 300);
              }} />
              <ParameterSlider label="Sigma (packet width)" min={3} max={14} step={1} value={sigma} unit="cells" onChange={(v) => {
                setSigma(v);
                if (applyTimerRef.current) window.clearTimeout(applyTimerRef.current);
                applyTimerRef.current = window.setTimeout(() => { lastMetricsRef.current = null; simRef.current?.updateParams?.({ sigma: v }); simRef.current?.reset?.(); dispatch({ type: 'RESET_METRICS' }); }, 300);
              }} />
            </div>
          </div>

          <div className="panel mt-6">
            <h3 className="text-xl font-bold text-purple-400 mb-4">System Metrics</h3>
            <StandardMetricsPanel
              coreMetrics={{ energy: state.metrics.energy, drift: state.metrics.drift, angularMomentum: '—' }}
              additionalMetrics={(function(){
                const m = lastMetricsRef.current;
                return [
                  { label: 'Fringe Spacing (Δy)', value: m?.fringeSpacing ?? '—', status: 'neutral' as const },
                  { label: 'Visibility', value: m?.visibility ?? '—', status: 'neutral' as const },
                  { label: 'Slit Intensity Ratio', value: m?.slitIntensityRatio ?? '—', status: 'neutral' as const },
                ];
              })()}
              title="System Metrics"
              titleColorClass="text-purple-400"
            />
          </div>
        </div>
      </div>
    </ExperimentLayout>
  );
}

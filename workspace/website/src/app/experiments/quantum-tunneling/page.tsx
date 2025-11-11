/*
 * Quantum Tunneling – Showcase Experiment (QUAN)
 * 
 * Wave packet incident on a chi-field barrier. Displays Transmission (T),
 * Reflection (R), and Conservation (T+R) alongside system energy metrics.
 *
 * Phase 1: Scaffold UI (validator-compliant). Physics wiring in Phase 2.
 */

'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import QuantumVisualizationOptions from '@/components/experiment/QuantumVisualizationOptions';
import StandardMetricsPanel from '@/components/experiment/StandardMetricsPanel';
import ParameterSlider from '@/components/ui/ParameterSlider';
import { detectBackend } from '@/physics/core/backend-detector';
import { decideSimulationProfile } from '@/physics/core/simulation-profile';
import SimpleCanvas from '@/components/visuals/SimpleCanvas';
import FieldSliceCanvas from '@/components/visuals/FieldSliceCanvas';
import { QuantumTunnelingSimulation } from '@/physics/quantum/quantum_tunneling_simulation';
import { useSimulationState } from '@/hooks/useSimulationState';

// experimentMeta intentionally omitted for Next.js export constraints; registry holds metadata

export default function QuantumTunnelingPage() {
  // State management
  const [state, dispatch] = useSimulationState();
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  const [uiMode, setUiMode] = useState<'advanced' | 'simple'>('advanced');
  const [dimMode, setDimMode] = useState<'3d' | '1d'>('3d');

  // Refs
  const simRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const isRunningRef = useRef<boolean>(false);
  const deviceRef = useRef<GPUDevice | null>(null);
  // Cache last quantum-specific metrics so panel freezes on pause/reset
  const lastQMetricsRef = useRef<{ transmission?: string; reflection?: string; conservation?: string } | null>(null);
  useEffect(() => { isRunningRef.current = state.isRunning; }, [state.isRunning]);

  // Simulation control
  const stopSimulation = useCallback(() => {
    dispatch({ type: 'SET_RUNNING', payload: false });
    isRunningRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [dispatch]);

  // Backend detection
  useEffect(() => {
    detectBackend().then((caps) => {
      const effectiveBackend = caps.backend === 'webgpu' || caps.backend === 'cpu' ? caps.backend : 'cpu';
      dispatch({ type: 'SET_BACKEND', payload: { backend: effectiveBackend, capabilities: caps } });
      setBackend(effectiveBackend);
      const prof = decideSimulationProfile(effectiveBackend, caps, 'quantum');
      setUiMode(prof.ui);
      setDimMode(prof.dim);
    });
  }, [dispatch]);

  // Simulation initialization (Phase 2 wired physics)
  useEffect(() => {
    let cancelled = false;
    async function initSim() {
      try {
        // Request GPU device if using WebGPU
        let device: GPUDevice | null = null;
        if (backend === 'webgpu') {
          const adapter = await navigator.gpu?.requestAdapter();
          device = await adapter?.requestDevice() || null;
          if (!device) {
            console.warn('[QuantumTunneling] GPU requested but not available');
            return;
          }
          deviceRef.current = device;
        }

        if (cancelled) return;

        // Cleanup previous simulation
        try { simRef.current?.destroy?.(); } catch (e) { console.error('[QuantumTunneling] destroy error:', e); }

        // Initialize tunneling simulation (GPU when available, CPU fallback)
        const sim = new QuantumTunnelingSimulation({
          latticeSize: backend === 'webgpu' ? 64 : 32,
          dx: 0.1,
          dt: 0.001,
          k0: 4.0,
          sigma: 6.0,
          barrierHeight: 1.5,
          barrierWidth: 6,
          batchSteps: state.params.simSpeed > 0 ? Math.max(1, Math.floor(state.params.simSpeed / 5)) : 6,
        }, backend === 'webgpu' && device ? { device } : undefined);
        await sim.initialize();
        if (cancelled) { await sim.reset(); return; }
        simRef.current = sim;
      } catch (e) {
        console.error('Simulation init failed:', e);
      }
    }
    initSim();
    return () => {
      cancelled = true;
      stopSimulation();
      try { simRef.current?.destroy?.(); } catch (e) { console.error('Cleanup error:', e); }
      simRef.current = null;
      deviceRef.current = null;
    };
  }, [backend, state.capabilities, state.resetTrigger, stopSimulation]);

  // Animation loop (steps physics and updates metrics)
  useEffect(() => {
    if (!state.isRunning) return;
    let rafId: number;
    const tick = async () => {
      const sim = simRef.current;
      if (!sim) return;

      await sim.stepBatch();
      const m = sim.getMetrics();
      dispatch({ type: 'UPDATE_METRICS', payload: { energy: m.energy, drift: m.drift, angularMomentum: '—' } });
      // Cache T/R metrics for display; prevents updates when paused
      lastQMetricsRef.current = {
        transmission: m?.transmission,
        reflection: m?.reflection,
        conservation: m?.conservation,
      };
      // Update additional metrics by re-rendering StandardMetricsPanel via state
      // (We pass values directly below.)

      if (state.isRunning) rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => { if (rafId) cancelAnimationFrame(rafId); };
  }, [state.isRunning, dispatch]);

  return (
    <ExperimentLayout
      title="🕳️ Quantum Tunneling"
      description="A localized wave packet impinges on a chi barrier. Partially reflects (R) and tunnels (T)."
      backend={backend}
      experimentId="quantum-tunneling"
      visualizationOptions={
        <QuantumVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
        />
      }
      footerContent={
        <div className="mt-8 panel">
          <h2 className="text-2xl font-bold text-accent-chi mb-4">What You're Seeing</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p>
              <strong className="text-purple-400">The Purple Wave:</strong> A quantum wave packet traveling from left to right. 
              The height shows the wave's amplitude (probability of finding energy at that location).
            </p>
            <p>
              <strong className="text-red-400">The Red Barrier:</strong> A finite chi-field potential wall (the "⚡ BARRIER ⚡" region). 
              This is like a hill the wave must climb over or tunnel through.
            </p>
            <p>
              <strong>What Happens:</strong> Watch as the wave packet hits the barrier. Part of it <em>reflects</em> back (bounces off), 
              part <em>tunnels through</em> (quantum tunneling - passes through despite not having enough energy classically), 
              and the metrics show Transmission (T), Reflection (R), and Conservation (T+R≈1).
            </p>
            <p className="text-sm text-text-muted">
              <strong>Technical note:</strong> The visualization shows wave amplitude |E|² projected along the x-axis. 
              The actual simulation runs in 3D, but we're showing a 1D "slice" for clarity.
            </p>
          </div>
        </div>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <div className="bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]">
            {uiMode === 'advanced' ? (
              <FieldSliceCanvas 
                simulation={simRef}
                isRunning={state.isRunning}
                showGrid={true}
                showWave={state.ui.showWave}
                showBarrier={state.ui.showBarrier}
                showTransmissionOverlay={state.ui.showTransmissionPlot}
                transmissionValue={lastQMetricsRef.current?.transmission ?? '—'}
                reflectionValue={lastQMetricsRef.current?.reflection ?? '—'}
                conservationValue={lastQMetricsRef.current?.conservation ?? '—'}
                updateInterval={3}
              />
            ) : (
              <SimpleCanvas isRunning={state.isRunning} parameters={{ ...state.params, __dim: dimMode }} views={{ showGrid: false, showField: false }} />
            )}
          </div>

          <div className="mt-4 flex gap-4">
            <button
              onClick={() => dispatch({ type: 'SET_RUNNING', payload: !state.isRunning })}
              disabled={backend !== 'webgpu'}
              className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
                backend !== 'webgpu'
                  ? 'bg-space-border cursor-not-allowed opacity-50'
                  : state.isRunning
                  ? 'bg-yellow-600 hover:bg-yellow-700'
                  : 'bg-accent-chi hover:bg-accent-chi/80'
              }`}
            >
              {state.isRunning ? '⏸ Pause' : '▶ Play'}
            </button>
            <button
              onClick={() => {
                simRef.current?.reset?.();
                dispatch({ type: 'RESET_METRICS' });
                // Clear cached quantum metrics so panel shows dashes until resumed
                lastQMetricsRef.current = null;
              }}
              className="px-6 py-3 bg-space-border hover:bg-space-border/80 rounded-lg font-semibold transition-colors"
            >
              🔄 Reset
            </button>
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className="panel" data-panel="experiment-parameters">
            <h3 className="text-xl font-bold text-purple-400 mb-1">Experiment Parameters</h3>
            {uiMode === 'simple' && (
              <div className="text-xs text-text-secondary mb-3" aria-label="Simple Mode">(Simple Mode)</div>
            )}
            <div className="space-y-6" data-section="profile-parameters">
              {/* Phase 2: Add packet k0, sigma; barrier height, width; presets */}
              <ParameterSlider label="Sim Speed" min={1} max={100} step={1} value={state.params.simSpeed} unit="x" onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })} />
            </div>
          </div>

          <div className="panel mt-6">
            <h3 className="text-xl font-bold text-purple-400 mb-4">System Metrics</h3>
            <StandardMetricsPanel
              coreMetrics={{
                energy: state.metrics.energy,
                drift: state.metrics.drift,
                angularMomentum: state.metrics.angularMomentum,
              }}
              additionalMetrics={(function(){
                // Use cached quantum metrics so panel doesn't change while paused
                const m = lastQMetricsRef.current;
                return [
                  { label: 'Transmission (T)', value: m?.transmission ?? '—', status: 'neutral' as const },
                  { label: 'Reflection (R)', value: m?.reflection ?? '—', status: 'neutral' as const },
                  { label: 'Conservation (T+R)', value: m?.conservation ?? '—', status: 'neutral' as const },
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

'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import BackendBadge from '@/components/ui/BackendBadge';
import ScientificDisclosure from '@/components/ui/ScientificDisclosure';
import ParameterSlider from '@/components/ui/ParameterSlider';
import VisualizationOptions from '@/components/ui/VisualizationOptions';
import { detectBackend } from '@/physics/core/backend-detector';
import { BinaryOrbitSimulation, OrbitConfig } from '@/physics/forces/binary-orbit';
import OrbitCanvas from '@/components/visuals/OrbitCanvas';
import { useSimulationState } from '@/hooks/useSimulationState';

export default function BigBangPage() {
  const [state, dispatch] = useSimulationState();
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  
  const deviceRef = useRef<GPUDevice | null>(null);
  const simRef = useRef<BinaryOrbitSimulation | null>(null);
  const rafRef = useRef<number | null>(null);
  const isRunningRef = useRef<boolean>(false);

  useEffect(() => { isRunningRef.current = state.isRunning; }, [state.isRunning]);

  const stopSimulation = useCallback(() => {
    dispatch({ type: 'SET_RUNNING', payload: false });
    isRunningRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [dispatch]);

  // Detect backend
  useEffect(() => {
    detectBackend().then((caps) => {
      const effectiveBackend = caps.backend === 'webgpu' || caps.backend === 'cpu' ? caps.backend : 'cpu';
      dispatch({ 
        type: 'SET_BACKEND', 
        payload: { backend: effectiveBackend, capabilities: caps } 
      });
      setBackend(effectiveBackend);
    });
  }, [dispatch]);

  // Create simulation for big bang (concentrated energy point that explodes outward)
  useEffect(() => {
    if (state.backend !== 'webgpu') {
      stopSimulation();
      return;
    }
    let cancelled = false;
    async function initSim() {
      try {
        const adapter = await navigator.gpu?.requestAdapter();
        const device = await adapter?.requestDevice();
        if (!device || cancelled) return;
        deviceRef.current = device;

        // Big bang: extremely concentrated energy point with very high chi strength
        // Map massRatio to energy (8000-15000 range)
        const energyMass = 2000 + (state.params.massRatio * 1000);
        const testParticle = 1.0; // Test mass to visualize expansion

        const actualLatticeSize = Math.min(state.capabilities?.maxLatticeSize ?? state.params.latticeSize, state.params.latticeSize);

        const config: OrbitConfig = {
          mass1: energyMass,
          mass2: testParticle,
          initialSeparation: state.params.orbitalDistance || 0.5, // Very small initial radius
          chiStrength: state.params.chiStrength || 0.8, // Very high field strength
          latticeSize: actualLatticeSize,
          dt: 0.0002,
          sigma: state.params.sigma || 0.3, // Extremely concentrated point
          startAngleDeg: 0,
          velocityScale: state.params.velocityScale || 2.0, // High initial velocity for explosion
        };

        try {
          simRef.current?.destroy();
        } catch (e) {
          console.error('[BigBang] Error destroying previous simulation:', e);
        }

        const sim = new BinaryOrbitSimulation(device, config);
        await sim.initialize();
        if (cancelled) { sim.destroy(); return; }
        simRef.current = sim;

        const s = sim.getState();
        dispatch({
          type: 'UPDATE_METRICS',
          payload: {
            angularMomentum: s.angularMomentum.toFixed(3),
            energy: s.energy ? s.energy.toFixed(4) + ' J' : '—',
            drift: (sim.getEnergyDrift() * 100).toFixed(4) + '%',
          },
        });
      } catch (e) {
        console.error('Big bang simulation init failed:', e);
      }
    }
    initSim();
    return () => {
      cancelled = true;
      stopSimulation();
      try { simRef.current?.destroy(); } catch (e) {
        console.error('Cleanup error:', e);
      }
      simRef.current = null;
      deviceRef.current = null;
    };
  }, [state.backend, state.params.latticeSize, state.params.massRatio, state.params.orbitalDistance, state.params.chiStrength, state.params.sigma, state.params.velocityScale, state.resetTrigger, state.capabilities, stopSimulation, dispatch]);

  const tick = useCallback(async (t: number) => {
    if (!isRunningRef.current) return;
    const sim = simRef.current;
    if (!sim) return;

    // Simple step execution
    const stepsPerFrame = Math.round(state.params.simSpeed || 100);
    await sim.stepBatch(Math.min(stepsPerFrame, 300));

    // Update metrics
    const s = sim.getState();
    const driftVal = sim.getEnergyDrift();
    const metricsUpdate: Record<string, string> = {};

    if (isFinite(s.angularMomentum)) metricsUpdate.angularMomentum = s.angularMomentum.toFixed(3);
    if (isFinite(driftVal)) metricsUpdate.drift = (driftVal * 100).toFixed(4) + '%';
    if (isFinite(s.energy)) metricsUpdate.energy = s.energy.toFixed(4) + ' J';

    if (Object.keys(metricsUpdate).length > 0) {
      dispatch({ type: 'UPDATE_METRICS', payload: metricsUpdate });
    }

    if (isRunningRef.current) rafRef.current = requestAnimationFrame(tick);
  }, [state.params.simSpeed, dispatch]);

  const startSimulation = useCallback(() => {
    if (state.backend !== 'webgpu') return;
    if (isRunningRef.current) return;
    isRunningRef.current = true;
    dispatch({ type: 'SET_RUNNING', payload: true });
    rafRef.current = requestAnimationFrame(tick);
  }, [state.backend, tick, dispatch]);

  const resetSimulation = useCallback(async () => {
    stopSimulation();
    try {
      simRef.current?.reset();
      await simRef.current?.initialize();
      dispatch({ type: 'RESET_METRICS' });
    } catch (e) {
      console.error('Reset failed:', e);
    }
  }, [stopSimulation, dispatch]);

  if (state.isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-space-dark">
        <div className="text-center">
          <div className="text-4xl mb-4 animate-pulse">💥</div>
          <div className="text-xl text-text-primary">Initializing Big Bang...</div>
          <div className="text-sm text-text-secondary mt-2">Concentrating primordial energy</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />

      <main className="flex-1 pt-20">
        <div className="container mx-auto px-4 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <div className="mb-4">
              <h1 className="text-4xl font-bold text-purple-400 mb-2">💥 Big Bang Simulation</h1>
              <p className="text-text-secondary">
                Watch the universe begin from a single point of concentrated energy. An explosive release of chi field energy 
                creates matter and drives cosmic expansion—all from emergent LFM dynamics.
              </p>
            </div>

            <ScientificDisclosure experimentName="Big Bang" />
          </div>

          {/* Backend Status */}
          <div className="mb-8">
            <BackendBadge backend={state.backend} />
          </div>

          {/* Main Experiment Area */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 3D Canvas */}
            <div className="lg:col-span-2">
              <div className="panel h-[600px] relative overflow-hidden">
                {state.backend === 'webgpu' ? (
                  <OrbitCanvas
                    simulation={simRef}
                    isRunning={state.isRunning}
                    showParticles={state.ui.showParticles}
                    showTrails={state.ui.showTrails}
                    showChi={state.ui.showChi}
                    showLattice={state.ui.showLattice}
                    showVectors={state.ui.showVectors}
                    showDomes={state.ui.showDomes}
                    showIsoShells={state.ui.showIsoShells}
                    showWell={state.ui.showWell}
                    showBackground={state.ui.showBackground}
                    chiStrength={state.params.chiStrength || 0.8}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-6xl mb-4">🖥️</div>
                      <h3 className="text-2xl font-bold text-purple-400 mb-2">WebGPU Required</h3>
                      <p className="text-text-secondary mb-6">
                        Big bang simulations require WebGPU. Upgrade your browser or enable experimental flags.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="mt-4 flex items-center justify-center space-x-4" role="group" aria-label="Simulation controls">
                <button
                  onClick={startSimulation}
                  disabled={state.backend !== 'webgpu' || state.isRunning}
                  aria-label="Start simulation"
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
                    state.backend !== 'webgpu' || state.isRunning
                      ? 'bg-accent-glow/40 text-space-dark/60 cursor-not-allowed'
                      : 'bg-accent-glow hover:bg-accent-glow/80 text-space-dark'
                  }`}
                >
                  ▶ Play
                </button>
                <button
                  onClick={stopSimulation}
                  disabled={!state.isRunning}
                  aria-label="Pause simulation"
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors border-2 ${
                    !state.isRunning
                      ? 'border-purple-500/40 text-purple-400/40 cursor-not-allowed'
                      : 'border-purple-500 text-purple-400 hover:bg-purple-500/10'
                  }`}
                >
                  ⏸ Pause
                </button>
                <button
                  onClick={resetSimulation}
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-indigo-500 hover:bg-indigo-400 text-white"
                  aria-label="Reset simulation"
                >
                  💥 Reset Big Bang
                </button>
              </div>
            </div>

            {/* Control Panel */}
            <div className="space-y-6">
              {/* Parameters */}
              <div className="panel">
                <h3 className="text-lg font-bold text-purple-400 mb-4">Big Bang Parameters</h3>

                <div className="space-y-4">
                  <ParameterSlider
                    label="Initial Energy"
                    value={state.params.massRatio}
                    min={6}
                    max={13}
                    step={0.5}
                    unit="E₀"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'massRatio', value: v } })}
                    tooltip="Primordial energy concentration at the singularity. Higher energy = more explosive expansion and stronger field. (Scales from 8000-15000 internal units)"
                  />
                  <ParameterSlider
                    label="Expansion Radius"
                    value={state.params.orbitalDistance}
                    min={0.2}
                    max={2.0}
                    step={0.1}
                    unit="units"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: v } })}
                    tooltip="Initial singularity size before explosion. Smaller = tighter concentration = more violent expansion."
                  />
                  <ParameterSlider
                    label="Expansion Speed"
                    value={state.params.velocityScale}
                    min={0.5}
                    max={5.0}
                    step={0.5}
                    unit="c"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'velocityScale', value: v } })}
                    tooltip="Initial expansion velocity. Higher = faster cosmic expansion, like inflation in real cosmology."
                  />
                  <ParameterSlider
                    label="Field Strength (χ)"
                    value={state.params.chiStrength || 0.8}
                    min={0.3}
                    max={1.0}
                    step={0.05}
                    unit=""
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'chiStrength', value: v } })}
                    tooltip="Chi field coupling strength. Extreme values create universe-scale dynamics and particle creation analogue."
                  />
                  <ParameterSlider
                    label="Concentration (σ)"
                    value={state.params.sigma || 0.3}
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    unit=""
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'sigma', value: v } })}
                    tooltip="Singularity concentration. Smaller = point-like singularity = extreme gradients mimicking big bang conditions."
                  />
                  <ParameterSlider
                    label="Playback Speed"
                    value={state.params.simSpeed || 100}
                    min={10}
                    max={300}
                    step={10}
                    unit="×"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })}
                    tooltip="Simulation speed - higher = more physics steps per frame to see expansion evolution faster."
                  />
                </div>
              </div>

              <VisualizationOptions
                toggles={[
                  { key: 'showParticles', label: 'Bodies', checked: state.ui.showParticles },
                  { key: 'showTrails', label: 'Orbital Trails', checked: state.ui.showTrails },
                  { key: 'showChi', label: 'Chi Field', checked: state.ui.showChi },
                  { key: 'showLattice', label: 'Simulation Grid', checked: state.ui.showLattice },
                  { key: 'showVectors', label: 'Force Arrows', checked: state.ui.showVectors },
                  { key: 'showWell', label: 'Gravity Well', checked: state.ui.showWell },
                  { key: 'showDomes', label: 'Field Bubbles', checked: state.ui.showDomes },
                  { key: 'showIsoShells', label: 'Field Shells', checked: state.ui.showIsoShells },
                  { key: 'showBackground', label: 'Stars & Background', checked: state.ui.showBackground },
                ]}
                onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
              />

              {/* Metrics */}
              <div className="panel">
                <h3 className="text-lg font-bold text-purple-400 mb-4">Universe Metrics</h3>

                <div className="space-y-3">
                  <MetricDisplay label="Total Energy" value={state.metrics.energy} status="neutral" />
                  <MetricDisplay label="Energy Drift" value={state.metrics.drift} status="neutral" />
                  <MetricDisplay label="Angular Momentum" value={state.metrics.angularMomentum} status="neutral" />
                </div>
              </div>
            </div>
          </div>

          {/* Explanation Panel */}
          <div className="mt-8 panel">
            <h3 className="text-xl font-bold text-purple-400 mb-4">What You're Seeing</h3>
            <div className="prose prose-invert max-w-none text-text-secondary">
              <p className="mb-4">
                This simulation demonstrates <strong>cosmological expansion</strong> from a primordial singularity using LFM physics. 
                An extremely concentrated point of chi field energy explosively expands, creating matter and driving cosmic evolution—all 
                from lattice field dynamics with no external expansion mechanism.
              </p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-purple-400 text-center my-4">
                Point singularity χ²(x,t) → Explosive expansion → Emergent cosmic inflation
              </div>
              <ul className="space-y-2 list-disc list-inside">
                <li><strong>Primordial singularity</strong> — Extremely small σ creates point-like energy concentration</li>
                <li><strong>Explosive expansion</strong> — High field gradients drive rapid outward acceleration</li>
                <li><strong>Energy distribution</strong> — Field energy spreads across lattice, creating matter analogue</li>
                <li><strong>Inflation analogue</strong> — Expansion speed parameter mimics cosmic inflation epoch</li>
                <li><strong>Structure formation</strong> — Field inhomogeneities seed galaxy-like clustering</li>
              </ul>
              <p className="mt-4 text-yellow-400">
                <strong>⚠️ MVP Note:</strong> This is a highly simplified big bang demonstration. Real cosmology requires dark energy, 
                matter/radiation separation, quantum fluctuations, and baryogenesis—all requiring extended LFM equations currently in development. 
                This shows the <em>principle</em> of emergent expansion from field dynamics.
              </p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

function MetricDisplay({ 
  label, 
  value, 
  status 
}: { 
  label: string; 
  value: string; 
  status: 'conserved' | 'good' | 'neutral' | 'warning';
}) {
  const statusColors = {
    conserved: 'text-accent-glow',
    good: 'text-accent-glow',
    neutral: 'text-text-primary',
    warning: 'text-yellow-500',
  };

  return (
    <div className="flex items-center justify-between py-2 border-b border-space-border last:border-b-0">
      <span className="text-sm text-text-secondary">{label}</span>
      <span className={`text-sm font-mono font-semibold ${statusColors[status]}`}>{value}</span>
    </div>
  );
}

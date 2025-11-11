'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import StandardVisualizationOptions from '@/components/experiment/StandardVisualizationOptions';
import StandardMetricsPanel from '@/components/experiment/StandardMetricsPanel';
import ParameterSlider from '@/components/ui/ParameterSlider';
import { detectBackend } from '@/physics/core/backend-detector';
import { BinaryOrbitSimulation, OrbitConfig } from '@/physics/forces/binary-orbit';
import OrbitCanvas from '@/components/visuals/OrbitCanvas';
import { useSimulationState } from '@/hooks/useSimulationState';

export default function StellarCollapsePage() {
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

  // Create simulation for stellar collapse (massive central body with concentrated field)
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

        // Stellar collapse: massive star at center with very small sigma (concentrated)
        // Map massRatio to star mass (1000-10000 range)
        const starMass = 1000 + (state.params.massRatio * 500);
        const testParticle = 1.0; // Small test mass to visualize field

        const actualLatticeSize = Math.min(state.capabilities?.maxLatticeSize ?? state.params.latticeSize, state.params.latticeSize);

        const config: OrbitConfig = {
          mass1: starMass,
          mass2: testParticle,
          initialSeparation: state.params.orbitalDistance,
          chiStrength: state.params.chiStrength || 0.5,
          latticeSize: actualLatticeSize,
          dt: 0.0003,
          sigma: state.params.sigma || 0.5, // Very concentrated field
          startAngleDeg: 0,
          velocityScale: 0.3, // Slow initial velocity
        };

        try {
          simRef.current?.destroy();
        } catch (e) {
          console.error('[StellarCollapse] Error destroying previous simulation:', e);
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
        console.error('Stellar collapse simulation init failed:', e);
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
  }, [state.backend, state.params.latticeSize, state.params.massRatio, state.params.orbitalDistance, state.params.chiStrength, state.params.sigma, state.resetTrigger, state.capabilities, stopSimulation, dispatch]);

  const tick = useCallback(async (t: number) => {
    if (!isRunningRef.current) return;
    const sim = simRef.current;
    if (!sim) return;

    // Simple step execution
    const stepsPerFrame = Math.round(state.params.simSpeed || 50);
    await sim.stepBatch(Math.min(stepsPerFrame, 200));

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
          <div className="text-4xl mb-4 animate-pulse">⭐</div>
          <div className="text-xl text-text-primary">Initializing Stellar Collapse...</div>
          <div className="text-sm text-text-secondary mt-2">Concentrating chi field</div>
        </div>
      </div>
    );
  }

  return (
    <ExperimentLayout
      title="⭐ Stellar Collapse Simulation"
      description="Watch a massive star collapse under its own chi field. As field concentration increases, the star's gravitational pull intensifies—demonstrating emergent gravity from extreme field gradients."
      backend={state.backend}
      experimentId="stellar-collapse"
      visualizationOptions={
        <StandardVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
          showAdvancedOptions={true}
        />
      }
      footerContent={
        <div className="mt-8 panel">
          <h3 className="text-xl font-bold text-purple-400 mb-4">What You're Seeing</h3>
          <div className="prose prose-invert max-w-none text-text-secondary">
            <p className="mb-4">
              This simulation demonstrates <strong>stellar collapse</strong> in the LFM framework. A massive star creates an 
              extremely concentrated chi field—as the field concentration increases, the emergent gravitational pull intensifies.
            </p>
            <div className="bg-space-dark p-4 rounded-lg font-mono text-purple-400 text-center my-4">
              Concentrated χ²(x,t) → Steep field gradients → Emergent gravity collapse
            </div>
            <ul className="space-y-2 list-disc list-inside">
              <li><strong>Field concentration</strong> — Small σ creates extremely steep field gradients near the star</li>
              <li><strong>Emergent event horizon</strong> — Test particles cannot escape beyond critical radius</li>
              <li><strong>Gravitational collapse</strong> — Star's own field creates inward acceleration</li>
              <li><strong>Black hole formation analogue</strong> — Extreme concentration mimics event horizon physics</li>
            </ul>
            <p className="mt-4 text-yellow-400">
              <strong>⚠️ MVP Note:</strong> This is a simplified demonstration using a test particle to visualize the collapsing 
              field. Full self-consistent stellar collapse (with pressure, radiation, matter dynamics) requires extended LFM equations 
              currently in development.
            </p>
          </div>
        </div>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* 3D Canvas */}
            <div className="lg:col-span-3">
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
                    chiStrength={state.params.chiStrength || 0.5}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-6xl mb-4">🖥️</div>
                      <h3 className="text-2xl font-bold text-purple-400 mb-2">WebGPU Required</h3>
                      <p className="text-text-secondary mb-6">
                        Stellar collapse simulations require WebGPU. Upgrade your browser or enable experimental flags.
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
                  ⭐ Reset Collapse
                </button>
              </div>
            </div>

            {/* Control Panel */}
            <div className="space-y-6">
              {/* Parameters */}
              <div className="panel">
                <h3 className="text-lg font-bold text-purple-400 mb-4">Collapse Parameters</h3>

                <div className="space-y-4">
                  <ParameterSlider
                    label="Star Mass"
                    value={state.params.massRatio}
                    min={2}
                    max={18}
                    step={1}
                    unit="M☉"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'massRatio', value: v } })}
                    tooltip="Mass of the collapsing star. Higher mass creates stronger field concentration and more extreme gravity. (Scales from 2000-10000 internal units)"
                  />
                  <ParameterSlider
                    label="Collapse Radius"
                    value={state.params.orbitalDistance}
                    min={2.0}
                    max={8.0}
                    step={0.5}
                    unit="units"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: v } })}
                    tooltip="Starting radius of the star before collapse. Smaller radius = more concentrated initial state."
                  />
                  <ParameterSlider
                    label="Field Strength (χ)"
                    value={state.params.chiStrength || 0.5}
                    min={0.2}
                    max={0.8}
                    step={0.05}
                    unit=""
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'chiStrength', value: v } })}
                    tooltip="Chi field coupling strength. Higher values create stronger emergent gravity and faster collapse."
                  />
                  <ParameterSlider
                    label="Concentration (σ)"
                    value={state.params.sigma || 0.5}
                    min={0.2}
                    max={2.0}
                    step={0.1}
                    unit=""
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'sigma', value: v } })}
                    tooltip="Field concentration parameter. Smaller values = tighter concentration = more extreme gravitational gradients."
                  />
                  <ParameterSlider
                    label="Playback Speed"
                    value={state.params.simSpeed || 50}
                    min={1}
                    max={200}
                    step={1}
                    unit="×"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })}
                    tooltip="Simulation speed - higher = more physics steps per frame to see collapse happen faster."
                  />
                </div>
              </div>

              {/* Metrics */}
              <StandardMetricsPanel
                coreMetrics={{
                  energy: state.metrics.energy,
                  drift: state.metrics.drift,
                  angularMomentum: state.metrics.angularMomentum,
                }}
                title="System Metrics"
                titleColorClass="text-purple-400"
              />
            </div>
          </div>
    </ExperimentLayout>
  );
}

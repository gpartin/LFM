/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import { useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import BackendBadge from '@/components/ui/BackendBadge';
import { detectBackend } from '@/physics/core/backend-detector';
import { BinaryOrbitSimulation, OrbitConfig } from '@/physics/forces/binary-orbit';
import OrbitCanvas from '@/components/visuals/OrbitCanvas';
import { useSimulationState } from '@/hooks/useSimulationState';
import { WebGPUErrorBoundary } from '@/components/ErrorBoundary';

export default function BlackHolePage() {
  // Consolidated state management
  const [state, dispatch] = useSimulationState();
  
  // Refs
  const isRunningRef = useRef<boolean>(false);
  useEffect(() => { isRunningRef.current = state.isRunning; }, [state.isRunning]);

  const effectiveSpeedRef = useRef<number>(50.0);
  const wasRunningBeforeDragRef = useRef<boolean>(false);

  const deviceRef = useRef<GPUDevice | null>(null);
  const simRef = useRef<BinaryOrbitSimulation | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  const canvasContainerRef = useRef<HTMLDivElement | null>(null);

  const stopSimulation = useCallback(() => {
    dispatch({ type: 'SET_RUNNING', payload: false });
    isRunningRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [dispatch]);
  
  // Detect backend on mount
  useEffect(() => {
    detectBackend().then((caps) => {
      dispatch({ 
        type: 'SET_BACKEND', 
        payload: { backend: caps.backend, capabilities: caps } 
      });
    });
  }, [dispatch]);

  // Recreate simulation for black hole setup
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
        
        // Black hole configuration: extremely high mass ratio
        const blackHoleMass = 1000; // 1000× moon mass
        const moonMass = 1.0;
        
        const actualLatticeSize = Math.min(state.capabilities?.maxLatticeSize ?? state.params.latticeSize, state.params.latticeSize);
        
        const config: OrbitConfig = {
          mass1: blackHoleMass,
          mass2: moonMass,
          initialSeparation: state.params.orbitalDistance,
          chiStrength: state.params.chiStrength,
          latticeSize: actualLatticeSize,
          dt: 0.0005, // Smaller timestep for stability
          sigma: 1.0, // Concentrated field for black hole
          startAngleDeg: 0,
          velocityScale: 1.0,
        };
        
        try { 
          simRef.current?.destroy(); 
        } catch (e) {
          console.error('[BlackHole] Error destroying previous simulation:', e);
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
        console.error('Black hole simulation init failed:', e);
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
  }, [state.backend, state.params.latticeSize, state.params.orbitalDistance, state.params.chiStrength, state.params.sigma, state.resetTrigger, state.capabilities, stopSimulation, dispatch]);

  // Map slider value to steps per frame
  const mapSimSpeed = useCallback((raw: number) => {
    const minSteps = 10;
    const maxSteps = 400;
    const stepsPerFrame = minSteps + ((raw - 1) / 1999) * (maxSteps - minSteps);
    return Math.round(stepsPerFrame);
  }, []);

  const tick = useCallback(async (t: number) => {
    if (!isRunningRef.current) return;
    const sim = simRef.current;
    if (!sim) return;

    // Frame timing / FPS
    if (lastFrameTimeRef.current) {
      const dtMs = t - lastFrameTimeRef.current;
      const fpsCalc = 1000 / (dtMs || 16.7);
      if (isFinite(fpsCalc)) {
        dispatch({ type: 'UPDATE_METRICS', payload: { fps: fpsCalc.toFixed(0) } });
      }
    }
    lastFrameTimeRef.current = t;

    const MAX_STEPS_PER_FRAME = 2000;

    // Speed smoothing
    const target = state.params.simSpeed;
    const current = effectiveSpeedRef.current;
    if (target !== current) {
      const largeJump = Math.abs(target - current) > current * 0.5 || target <= 100;
      if (largeJump) {
        effectiveSpeedRef.current = target;
      } else {
        const maxDelta = Math.max(25, Math.round(current * 0.35));
        const delta = Math.sign(target - current) * Math.min(Math.abs(target - current), maxDelta);
        effectiveSpeedRef.current = current + delta;
      }
      dispatch({ 
        type: 'UPDATE_METRICS', 
        payload: { effectiveSpeed: Math.round(effectiveSpeedRef.current).toString() } 
      });
    }

    const effSpeed = Math.round(effectiveSpeedRef.current);
    const stepsPerFrame = mapSimSpeed(effSpeed);
    
    const ffActive = stepsPerFrame > 100;
    if (ffActive !== state.ui.fastForward) {
      dispatch({ type: 'UPDATE_UI', payload: { key: 'fastForward', value: ffActive } });
    }

    const plannedSteps = Math.max(1, Math.min(MAX_STEPS_PER_FRAME, stepsPerFrame));
    let remaining = plannedSteps;
    const frameStart = performance.now();
    const microBatchSize = Math.min(100, Math.ceil(plannedSteps / 4));
    
    while (remaining > 0 && isRunningRef.current) {
      const chunk = Math.min(remaining, microBatchSize);
      await sim.stepBatch(chunk);
      remaining -= chunk;
      if (performance.now() - frameStart > 14) break;
    }

    // Update metrics
    const s = sim.getState();
    const driftVal = sim.getEnergyDrift();
    const metricsUpdate: Record<string, string> = {};
    
    if (isFinite(s.angularMomentum)) metricsUpdate.angularMomentum = s.angularMomentum.toFixed(3);
    if (isFinite(driftVal)) metricsUpdate.drift = (driftVal * 100).toFixed(4) + '%';
    if (isFinite(s.energy)) metricsUpdate.energy = s.energy.toFixed(4) + ' J';
    if (isFinite(s.orbitalPeriod)) metricsUpdate.orbitalPeriod = s.orbitalPeriod.toFixed(2) + ' s';

    // Diagnostics
    try {
      const diag = (sim as any).getDiagnostics?.();
      if (diag) {
        if (isFinite(diag.separation)) metricsUpdate.separation = diag.separation.toFixed(3);
        if (isFinite(diag.vOverVcirc)) metricsUpdate.vRatio = diag.vOverVcirc.toFixed(2) + '×';
      }
    } catch (e) {
      // Diagnostics not available
    }

    if (Object.keys(metricsUpdate).length > 0) {
      dispatch({ type: 'UPDATE_METRICS', payload: metricsUpdate });
    }

    if (isRunningRef.current) rafRef.current = requestAnimationFrame(tick);
  }, [state.params.simSpeed, dispatch, mapSimSpeed, state.ui.fastForward]);

  // Speed snap on pause or large jump
  useEffect(() => {
    if (!state.isRunning) {
      effectiveSpeedRef.current = state.params.simSpeed;
      dispatch({ 
        type: 'UPDATE_METRICS', 
        payload: { effectiveSpeed: Math.round(state.params.simSpeed).toString() } 
      });
      return;
    }
    const current = effectiveSpeedRef.current;
    const target = state.params.simSpeed;
    if (Math.abs(target - current) > current * 0.75) {
      effectiveSpeedRef.current = target;
      dispatch({ 
        type: 'UPDATE_METRICS', 
        payload: { effectiveSpeed: Math.round(target).toString() } 
      });
    }
  }, [state.params.simSpeed, state.isRunning, dispatch]);

  const handleSpeedDragStart = useCallback(() => {
    wasRunningBeforeDragRef.current = isRunningRef.current;
    if (isRunningRef.current) {
      isRunningRef.current = false;
      dispatch({ type: 'SET_RUNNING', payload: false });
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    }
  }, [dispatch]);

  const handleSpeedDragEnd = useCallback(() => {
    effectiveSpeedRef.current = state.params.simSpeed;
    dispatch({ 
      type: 'UPDATE_METRICS', 
      payload: { effectiveSpeed: Math.round(state.params.simSpeed).toString() } 
    });
    if (wasRunningBeforeDragRef.current) {
      isRunningRef.current = true;
      dispatch({ type: 'SET_RUNNING', payload: true });
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [state.params.simSpeed, tick, dispatch]);

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
          <div className="text-4xl mb-4 animate-pulse">⚫</div>
          <div className="text-xl text-text-primary">Initializing Black Hole Physics...</div>
          <div className="text-sm text-text-secondary mt-2">Warping spacetime on your GPU</div>
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
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <h1 className="text-4xl font-bold text-purple-400 mb-2">🌌 Black Hole Orbit Simulation</h1>
                <p className="text-text-secondary">
                  Witness extreme gravity in action. A tiny black hole warps spacetime—watch a moon dance in its gravitational grip.
                </p>
              </div>
              <Link 
                href="/about"
                className="px-4 py-2 bg-yellow-500/20 border-2 border-yellow-500/50 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors whitespace-nowrap text-sm font-semibold"
              >
                ⚠️ Read About This Project
              </Link>
            </div>
            <div className="bg-yellow-500/10 border-l-4 border-yellow-500 p-4 rounded">
              <p className="text-sm text-text-secondary">
                <strong className="text-yellow-400">Scientific Disclosure:</strong> This is an exploratory simulation using emergent gravity from lattice fields. 
                NOT a validated model of actual black holes. <Link href="/about" className="text-accent-chi hover:underline">Learn more →</Link>
              </p>
            </div>
          </div>

          {/* Backend Status */}
          <div className="mb-8">
            <BackendBadge backend={state.backend} />
          </div>

          {/* View Options */}
          <div className="mb-6 panel">
            <h3 className="text-sm font-bold text-purple-400 mb-3">Visualization Options</h3>
            <div className="flex flex-wrap gap-x-6 gap-y-2" role="group" aria-label="Visualization options">
              <ViewToggle label="Black Hole & Moon" checked={state.ui.showParticles} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showParticles', value: v } })} />
              <ViewToggle label="Orbital Paths" checked={state.ui.showTrails} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showTrails', value: v } })} />
              <ViewToggle label="Gravity Field (2D)" checked={state.ui.showChi} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showChi', value: v } })} />
              <ViewToggle label="Simulation Grid" checked={state.ui.showLattice} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showLattice', value: v } })} />
              <ViewToggle label="Force Arrows" checked={state.ui.showVectors} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showVectors', value: v } })} />
              <ViewToggle label="Gravity Well (Surface)" checked={state.ui.showWell} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showWell', value: v } })} />
              <ViewToggle label="Field Bubbles (3D)" checked={state.ui.showDomes} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showDomes', value: v } })} />
              <ViewToggle label="Event Horizon Shell" checked={state.ui.showIsoShells} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showIsoShells', value: v } })} />
              <ViewToggle label="Stars & Background" checked={state.ui.showBackground} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showBackground', value: v } })} />
            </div>
          </div>

          {/* Main Experiment Area */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 3D Canvas */}
            <div className="lg:col-span-2">
              <div ref={canvasContainerRef} className="panel h-[600px] relative overflow-hidden">
                {state.ui.fastForward && (
                  <div className="absolute top-2 right-2 z-10 px-3 py-1 rounded bg-yellow-500/20 border border-yellow-500/50 text-yellow-300 text-xs font-semibold tracking-wide">
                    FAST-FORWARD
                  </div>
                )}
                {state.backend === 'webgpu' ? (
                  <WebGPUErrorBoundary>
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
                      chiStrength={state.params.chiStrength}
                    />
                  </WebGPUErrorBoundary>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-6xl mb-4">🖥️</div>
                      <h3 className="text-2xl font-bold text-purple-400 mb-2">WebGPU Required</h3>
                      <p className="text-text-secondary mb-6">
                        Black hole simulations require WebGPU. Upgrade your browser or enable experimental flags.
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
                  ⚫ Reset Black Hole
                </button>
              </div>
            </div>

            {/* Control Panel */}
            <div className="space-y-6">
              {/* Parameters */}
              <div className="panel">
                <h3 className="text-lg font-bold text-purple-400 mb-4">Black Hole Parameters</h3>

                <div className="space-y-4">
                  <ParameterSlider 
                    label="Orbital Distance" 
                    value={state.params.orbitalDistance} 
                    min={1.5} 
                    max={6.0} 
                    step={0.1} 
                    unit="units" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: v } })} 
                    tooltip="Starting distance from black hole. Closer = stronger gravity, faster orbit."
                  />
                  <ParameterSlider 
                    label="Gravity Strength" 
                    value={state.params.chiStrength} 
                    min={0.1} 
                    max={0.6} 
                    step={0.01} 
                    unit="" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'chiStrength', value: v } })} 
                    tooltip="Field coupling strength - how strong the gravitational pull is."
                  />
                  <ParameterSlider 
                    label="Field Concentration (σ)" 
                    value={state.params.sigma} 
                    min={0.3} 
                    max={2.5} 
                    step={0.1} 
                    unit="" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'sigma', value: v } })} 
                    tooltip="How concentrated the black hole's field is. Smaller = more extreme gradients."
                  />
                  <ParameterSlider 
                    label="Playback Speed" 
                    value={state.params.simSpeed} 
                    min={1} 
                    max={500} 
                    step={1} 
                    unit="×" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })} 
                    onDragStart={handleSpeedDragStart} 
                    onDragEnd={handleSpeedDragEnd} 
                    tooltip="Simulation speed - higher = more physics steps per frame."
                  />
                </div>
              </div>

              {/* Metrics */}
              <div className="panel">
                <h3 className="text-lg font-bold text-purple-400 mb-4">System Metrics</h3>
                
                <div className="space-y-3">
                  <MetricDisplay label="Total Energy" value={state.metrics.energy} status="neutral" />
                  <MetricDisplay label="Energy Drift" value={state.metrics.drift} status="neutral" />
                  <MetricDisplay label="Angular Momentum" value={state.metrics.angularMomentum} status="neutral" />
                  <MetricDisplay label="Distance from Black Hole" value={state.metrics.separation} status="warning" />
                  <MetricDisplay label="Speed Ratio (v/v_circ)" value={state.metrics.vRatio} status="neutral" />
                  <MetricDisplay label="Effective Speed" value={state.metrics.effectiveSpeed + '×'} status="good" />
                  <MetricDisplay label="Frame Rate" value={state.metrics.fps} status="good" />
                </div>
              </div>
            </div>
          </div>

          {/* Explanation Panel */}
          <div className="mt-8 panel">
            <h3 className="text-xl font-bold text-purple-400 mb-4">What You're Seeing</h3>
            <div className="prose prose-invert max-w-none text-text-secondary">
              <p className="mb-4">
                This simulation places a <strong>tiny black hole</strong> at the center of the lattice with a moon in orbit. 
                The black hole is represented by an <strong>extremely concentrated chi field</strong> (small σ, high mass).
              </p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-purple-400 text-center my-4">
                Extreme χ²(x,t) → Strong spacetime curvature
              </div>
              <ul className="space-y-2 list-disc list-inside">
                <li><strong>Emergent event horizon</strong> — Field becomes so steep that escape requires extreme velocity</li>
                <li><strong>Orbital precession</strong> — Moon's orbit may precess due to strong field gradients</li>
                <li><strong>Inspiral dynamics</strong> — Energy loss (numerical or physical) causes moon to spiral inward</li>
                <li><strong>Time dilation analogue</strong> — Timesteps near black hole effectively "slow down" in extreme field</li>
              </ul>
              <p className="mt-4 text-yellow-400">
                <strong>⚠️ Exploratory Physics:</strong> This is NOT a validated model of real black holes. 
                It explores emergent gravity from lattice field medium (LFM) with extreme parameters.
              </p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

// Reusable components
function ParameterSlider({ 
  label, 
  value, 
  min, 
  max, 
  step, 
  unit, 
  onChange,
  onDragStart,
  onDragEnd,
  tooltip,
}: { 
  label: string; 
  value: number; 
  min: number; 
  max: number; 
  step: number; 
  unit: string;
  onChange: (value: number) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  tooltip?: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-semibold text-text-primary" title={tooltip}>{label}</label>
        <span className="text-sm font-mono text-purple-400">{value.toFixed(step < 0.01 ? 4 : step < 0.1 ? 3 : step < 1 ? 2 : 1)} {unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        onMouseDown={onDragStart}
        onTouchStart={onDragStart}
        onMouseUp={onDragEnd}
        onTouchEnd={onDragEnd}
        onBlur={onDragEnd}
        title={tooltip}
        className="w-full h-2 bg-space-border rounded-lg appearance-none cursor-pointer accent-purple-500"
      />
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

function ViewToggle({ 
  label, 
  checked, 
  onChange 
}: { 
  label: string; 
  checked: boolean; 
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center space-x-3 cursor-pointer group">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-5 h-5 rounded border-2 border-purple-500 checked:bg-purple-500 checked:border-purple-500 focus:ring-2 focus:ring-purple-500/50 transition-colors"
        aria-label={label}
        aria-checked={checked}
        role="switch"
      />
      <span className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">
        {label}
      </span>
    </label>
  );
}

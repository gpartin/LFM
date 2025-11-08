'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

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

export default function BinaryOrbitPage() {
  // Consolidated state management (replaces 20+ individual useState calls)
  const [state, dispatch] = useSimulationState();
  
  // Ref mirror to avoid stale closure issues inside RAF callbacks
  const isRunningRef = useRef<boolean>(false);
  useEffect(() => { isRunningRef.current = state.isRunning; }, [state.isRunning]);

  // Smoothed effective speed used for incremental ramp to prevent sudden UI stalls
  const effectiveSpeedRef = useRef<number>(50.0);
  const wasRunningBeforeDragRef = useRef<boolean>(false);

  // Refs for simulation and RAF
  const deviceRef = useRef<GPUDevice | null>(null);
  const simRef = useRef<BinaryOrbitSimulation | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const frameCounterRef = useRef<number>(0);
  const renderEveryRef = useRef<number>(1);
  const renderCounterRef = useRef<number>(0);

  // Container for overlay badges
  const canvasContainerRef = useRef<HTMLDivElement | null>(null);
  // (No 2D draw refs anymore; R3F renders directly from sim state)

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

  // Recreate simulation whenever backend or structural params change
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
        // Recompute masses from ratio (total mass kept constant for scaling stability)
        // massRatio = Earth/Moon ratio, so m1 (Earth) is larger
        const totalMass = 2.0;
        const clampedRatio = Math.max(0.1, Math.min(100, state.params.massRatio));
        const m1 = clampedRatio * (totalMass / (1 + clampedRatio));  // Earth (larger)
        const m2 = totalMass - m1;  // Moon (smaller)
        const actualLatticeSize = Math.min(state.capabilities?.maxLatticeSize ?? state.params.latticeSize, state.params.latticeSize);
        console.log(`[CONFIG] latticeSize state=${state.params.latticeSize}, capabilities.maxLatticeSize=${state.capabilities?.maxLatticeSize}, actual=${actualLatticeSize}`);
        const config: OrbitConfig = {
          mass1: m1,
          mass2: m2,
          initialSeparation: state.params.orbitalDistance,
          chiStrength: state.params.chiStrength,
          latticeSize: actualLatticeSize,
          dt: state.params.dt,
          sigma: state.params.sigma,
          startAngleDeg: state.params.startAngleDeg,
          velocityScale: state.params.velocityScale,
        };
        // Destroy any previous simulation cleanly
        try { 
          simRef.current?.destroy(); 
        } catch (e) {
          console.error('[BinaryOrbit] Error destroying previous simulation:', e);
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
        console.error('Simulation init failed:', e);
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
  }, [state.backend, state.params.latticeSize, state.params.dt, state.params.chiStrength, state.params.sigma, state.params.massRatio, state.params.orbitalDistance, state.params.startAngleDeg, state.params.velocityScale, state.resetTrigger, state.capabilities, stopSimulation, dispatch]);

  // Environment-based diagnostics toggle (no UI exposure)
  // NEXT_PUBLIC_LFM_DIAGNOSTICS=off|basic|full
  useEffect(() => {
    const mode = process.env.NEXT_PUBLIC_LFM_DIAGNOSTICS || 'off';
    const enable = mode === 'basic' || mode === 'full' || mode === 'record' || mode === 'on';
    if (!enable) return;
    const sim = simRef.current as any;
    try {
      sim?.setDiagnosticsEnabled?.(true);
      // Note: No UI is shown; recording happens internally for developer builds only
      // Export must be performed via developer tools or local scripts (not exposed publicly)
      // This keeps production surface clean and secure
      // console.info('[Diagnostics] Enabled by env NEXT_PUBLIC_LFM_DIAGNOSTICS');
    } catch (e) {
      console.error('[BinaryOrbit] Error enabling diagnostics:', e);
    }
  }, []);

  // Earth-Moon preset (scaled units)
  // Conservative: small orbit in small lattice (known to work)
  const applyEarthMoonPreset = useCallback(() => {
    dispatch({ 
      type: 'APPLY_PRESET', 
      payload: { preset: state.params.startPreset } 
    });
  }, [dispatch, state.params.startPreset]);

  // (Removed legacy 2D starfield and draw routines; background handled in 3D scene)

  // Apply parameter changes on the fly
  useEffect(() => {
    if (state.backend !== 'webgpu') return;
    const sim = simRef.current;
    if (!sim) return;

    // Recompute masses from ratio
    // massRatio = Earth/Moon ratio, so m1 (Earth) is larger
    const totalMass = 2.0;
    const clampedRatio = Math.max(0.1, Math.min(100, state.params.massRatio));
    const m1 = clampedRatio * (totalMass / (1 + clampedRatio));  // Earth (larger)
    const m2 = totalMass - m1;  // Moon (smaller)

    sim.updateParameters({
      mass1: m1,
      mass2: m2,
      initialSeparation: state.params.orbitalDistance,
      chiStrength: state.params.chiStrength,
      sigma: state.params.sigma,
    });
    // Refresh chi field because strength/positions might have changed
    sim.refreshChiField();

    dispatch({ type: 'RESET_METRICS' });
  }, [state.backend, state.params.massRatio, state.params.orbitalDistance, state.params.chiStrength, state.params.sigma, dispatch]);

  // Map slider value (1-2000) to steps per frame for controlling emergent spacetime evolution rate
  // This directly controls how much physical time advances per rendered frame
  const mapSimSpeed = useCallback((raw: number) => {
    // Linear mapping: slider 1-2000 → steps per frame 10-400
    // At slider=25 (default): 10 + (25/2000 * 390) ≈ 15 steps/frame (slow, observable)
    // At slider=100: 10 + (100/2000 * 390) ≈ 30 steps/frame (moderate)
    // At slider=500: 10 + (500/2000 * 390) ≈ 108 steps/frame (fast)
    // At slider=2000: 10 + (2000/2000 * 390) = 400 steps/frame (maximum speed)
    const minSteps = 10;
    const maxSteps = 400;
    const stepsPerFrame = minSteps + ((raw - 1) / 1999) * (maxSteps - minSteps);
    return Math.round(stepsPerFrame);
  }, []);

  const tick = useCallback(async (t: number) => {
    // If we've been paused between frames, abort without scheduling a new RAF.
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

    // Speed smoothing only for moderate changes; large jumps apply immediately (handled in simSpeed effect)
    const target = state.params.simSpeed;
    const current = effectiveSpeedRef.current;
    if (target !== current) {
      const largeJump = Math.abs(target - current) > current * 0.5 || target <= 100; // small speeds: immediate
      if (largeJump) {
        effectiveSpeedRef.current = target;
      } else {
        const maxDelta = Math.max(25, Math.round(current * 0.35)); // slightly faster ramp
        const delta = Math.sign(target - current) * Math.min(Math.abs(target - current), maxDelta);
        effectiveSpeedRef.current = current + delta;
      }
      dispatch({ 
        type: 'UPDATE_METRICS', 
        payload: { effectiveSpeed: Math.round(effectiveSpeedRef.current).toString() } 
      });
    }

  // Get steps per frame from slider value - this controls emergent spacetime evolution rate
  const effSpeed = Math.round(effectiveSpeedRef.current);
  const stepsPerFrame = mapSimSpeed(effSpeed);
  
  // Fast-forward indicator (for UI feedback only)
  const ffActive = stepsPerFrame > 100;
    if (ffActive !== state.ui.fastForward) {
      dispatch({ type: 'UPDATE_UI', payload: { key: 'fastForward', value: ffActive } });
    }

  // Use slider-controlled steps per frame (respects emergent spacetime dynamics)
  const plannedSteps = Math.max(1, Math.min(MAX_STEPS_PER_FRAME, stepsPerFrame));
    let remaining = plannedSteps;
    const frameStart = performance.now();
    // Chunk size for micro-batching (breaks large step counts into smaller GPU submissions)
    const microBatchSize = Math.min(100, Math.ceil(plannedSteps / 4));
    while (remaining > 0 && isRunningRef.current) {
      const chunk = Math.min(remaining, microBatchSize);
      await sim.stepBatch(chunk);
      remaining -= chunk;
      // Keep within ~14ms UI budget
      if (performance.now() - frameStart > 14) break;
      // Mid-frame speed change responsiveness
      const newTarget = state.params.simSpeed;
      if (newTarget !== target) {
        effectiveSpeedRef.current = newTarget;
        dispatch({ 
          type: 'UPDATE_METRICS', 
          payload: { effectiveSpeed: Math.round(newTarget).toString() } 
        });
        // Do NOT alter physics progression mid-frame; trajectory must remain invariant.
        // We simply ignore mid-frame replanning now.
      }
    }

    // Update metrics every frame
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
        frameCounterRef.current = (frameCounterRef.current + 1) % 60;
        if (frameCounterRef.current === 0) {
          console.log('[DIAG]', {
            r: Number(diag.separation.toFixed(4)),
            vr: Number(diag.radialVelocity.toFixed(5)),
            vt: Number(diag.tangentialVelocity.toFixed(5)),
            v: Number(diag.speed.toFixed(5)),
            v_over_vcirc: Number(diag.vOverVcirc.toFixed(3)),
            a_req: Number(diag.requiredCentripetalAcc.toExponential(3)),
            a_grav_radial: Number(diag.radialGravityAcc.toExponential(3)),
            grav_over_req: Number(diag.gravityToCentripetal.toFixed(3)),
          });
        }
      }
    } catch (e) {
      console.error('Diagnostics error:', e);
    }

    // Dispatch all metrics at once
    if (Object.keys(metricsUpdate).length > 0) {
      dispatch({ type: 'UPDATE_METRICS', payload: metricsUpdate });
    }

    // Rendering handled by R3F; we just step physics here.

    // Only schedule next frame if still running
    if (isRunningRef.current) rafRef.current = requestAnimationFrame(tick);
  }, [state.params.simSpeed, dispatch, mapSimSpeed, state.ui.fastForward]);

  // Immediate reaction for large speed slider changes & when paused
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
    if (Math.abs(target - current) > current * 0.75) { // very large jump → snap
      effectiveSpeedRef.current = target;
      dispatch({ 
        type: 'UPDATE_METRICS', 
        payload: { effectiveSpeed: Math.round(target).toString() } 
      });
    }
  }, [state.params.simSpeed, state.isRunning, dispatch]);

  // Speed slider drag handlers: pause during drag, resume after
  const handleSpeedDragStart = useCallback(() => {
    wasRunningBeforeDragRef.current = isRunningRef.current;
    if (isRunningRef.current) {
      // Pause immediately
      isRunningRef.current = false;
      dispatch({ type: 'SET_RUNNING', payload: false });
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    }
  }, [dispatch]);

  const handleSpeedDragEnd = useCallback(() => {
    // Snap effective speed to latest slider value
    effectiveSpeedRef.current = state.params.simSpeed;
    dispatch({ 
      type: 'UPDATE_METRICS', 
      payload: { effectiveSpeed: Math.round(state.params.simSpeed).toString() } 
    });
    if (wasRunningBeforeDragRef.current) {
      // Resume
      isRunningRef.current = true;
      dispatch({ type: 'SET_RUNNING', payload: true });
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [state.params.simSpeed, tick, dispatch]);

  // (Removed legacy 2D draw routines)

  const startSimulation = useCallback(() => {
    if (state.backend !== 'webgpu') return; // Only supported for WebGPU for now
    if (isRunningRef.current) return;
    isRunningRef.current = true; // set ref first so first tick sees it
    dispatch({ type: 'SET_RUNNING', payload: true });
    rafRef.current = requestAnimationFrame(tick);
  }, [state.backend, tick, dispatch]);

  // stopSimulation defined earlier to satisfy dependency order

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
          <div className="text-4xl mb-4 animate-pulse">⚛️</div>
          <div className="text-xl text-text-primary">Initializing Physics Engine...</div>
          <div className="text-sm text-text-secondary mt-2">Detecting GPU capabilities</div>
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
                <h1 className="text-4xl font-bold text-accent-chi mb-2">Earth-Moon Orbit Simulation</h1>
                <p className="text-text-secondary">
                  Watch the Earth and Moon orbit due to emergent gravity from chi field gradients.
                  Real Klein-Gordon physics running on your GPU—not Newtonian mechanics.
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
                <strong className="text-yellow-400">Scientific Disclosure:</strong> This is an exploratory simulation. 
                We are NOT claiming this is proven physics. <Link href="/about" className="text-accent-chi hover:underline">Learn more about our approach and limitations →</Link>
              </p>
            </div>
          </div>

          {/* Backend Status */}
          <div className="mb-8">
            <BackendBadge backend={state.backend} />
          </div>

          {/* View Options - Horizontal layout above canvas */}
          <div className="mb-6 panel">
            <h3 className="text-sm font-bold text-accent-chi mb-3">What to Show</h3>
            <div className="flex flex-wrap gap-x-6 gap-y-2" role="group" aria-label="Visualization options">
              <ViewToggle label="Earth & Moon" checked={state.ui.showParticles} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showParticles', value: v } })} />
              <ViewToggle label="Orbital Paths" checked={state.ui.showTrails} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showTrails', value: v } })} />
              <ViewToggle label="Gravity Field (2D)" checked={state.ui.showChi} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showChi', value: v } })} />
              <ViewToggle label="Simulation Grid" checked={state.ui.showLattice} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showLattice', value: v } })} />
              <ViewToggle label="Force Arrows" checked={state.ui.showVectors} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showVectors', value: v } })} />
              <ViewToggle label="Gravity Well (Surface)" checked={state.ui.showWell} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showWell', value: v } })} />
              <ViewToggle label="Field Bubbles (3D)" checked={state.ui.showDomes} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showDomes', value: v } })} />
              <ViewToggle label="Field Shells" checked={state.ui.showIsoShells} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showIsoShells', value: v } })} />
              <ViewToggle label="Stars & Sun" checked={state.ui.showBackground} onChange={(v) => dispatch({ type: 'UPDATE_UI', payload: { key: 'showBackground', value: v } })} />
            </div>
          </div>

          {/* Main Experiment Area */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 3D Canvas (Left side - 2/3 width) */}
            <div className="lg:col-span-2">
              <div ref={canvasContainerRef} className="panel h-[600px] relative overflow-hidden">
                {state.ui.fastForward && (
                  <div className="absolute top-2 right-2 z-10 px-3 py-1 rounded bg-yellow-500/20 border border-yellow-500/50 text-yellow-300 text-xs font-semibold tracking-wide">
                    FAST-FORWARD ×{renderEveryRef.current}
                  </div>
                )}
                {/* 3D Canvas renders full scene; no 2D overlay needed */}
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
                      <h3 className="text-2xl font-bold text-accent-chi mb-2">WebGPU Not Available</h3>
                      <p className="text-text-secondary mb-6">
                        Full 3D visualization requires WebGPU. Upgrade your browser or enable experimental flags.
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
                  aria-disabled={state.backend !== 'webgpu' || state.isRunning}
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
                  aria-disabled={!state.isRunning}
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors border-2 ${
                    !state.isRunning
                      ? 'border-accent-chi/40 text-accent-chi/40 cursor-not-allowed'
                      : 'border-accent-chi text-accent-chi hover:bg-accent-chi/10'
                  }`}
                >
                  ⏸ Pause
                </button>
                <button
                  onClick={async () => { 
                    // Stop simulation first
                    const wasRunning = state.isRunning;
                    stopSimulation();
                    
                    // Apply Earth-Moon preset to state
                    dispatch({ type: 'UPDATE_METRICS', payload: { orbitalPeriod: '—' } }); 
                    applyEarthMoonPreset();
                    
                    // Wait a tick for state to update
                    await new Promise(resolve => setTimeout(resolve, 0));
                    
                    // The useEffect will detect parameter changes and recreate simulation
                    // Just need to restart if it was running
                    if (wasRunning) {
                      // Give useEffect time to recreate simulation
                      setTimeout(() => {
                        if (simRef.current) {
                          startSimulation();
                        }
                      }, 150);
                    }
                  }}
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-indigo-500 hover:bg-indigo-400 text-white"
                  title="Restore Earth-Moon default parameters"
                  aria-label="Reset to default Earth-Moon parameters"
                >🌍🌙 Reset to Defaults</button>
                {/* Rebuild removed: Reset to Defaults covers reinitialization; keeping UI minimal for public release */}
                {/* Starting preset selector */}
                <div className="mt-3 w-full max-w-md">
                  <label htmlFor="preset-slider" className="block text-xs font-semibold text-gray-300 mb-1">Starting Position Preset</label>
                  <div className="flex items-center gap-3">
                    <input 
                      id="preset-slider"
                      type="range" 
                      min={1} 
                      max={3} 
                      step={1} 
                      value={state.params.startPreset} 
                      onChange={(e) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'startPreset', value: parseInt(e.target.value, 10) } })} 
                      className="flex-1"
                      aria-label="Select starting position preset"
                      aria-valuemin={1}
                      aria-valuemax={3}
                      aria-valuenow={state.params.startPreset}
                      aria-valuetext={`Preset ${state.params.startPreset}: ${state.params.startPreset === 1 ? 'near-circular' : state.params.startPreset === 2 ? 'low speed (inward)' : 'high speed (outward)'}`}
                    />
                    <span className="text-xs" aria-live="polite">{state.params.startPreset}</span>
                    <button
                      onClick={async () => { 
                        // Stop simulation first
                        const wasRunning = state.isRunning;
                        stopSimulation();
                        
                        // Apply preset to state
                        dispatch({ type: 'UPDATE_METRICS', payload: { orbitalPeriod: '—' } }); 
                        applyEarthMoonPreset();
                        
                        // Wait a tick for state to update
                        await new Promise(resolve => setTimeout(resolve, 0));
                        
                        // The useEffect will detect parameter changes and recreate simulation
                        // Just need to restart if it was running
                        if (wasRunning) {
                          // Give useEffect time to recreate simulation
                          setTimeout(() => {
                            if (simRef.current) {
                              startSimulation();
                            }
                          }, 150);
                        }
                      }}
                      className="px-3 py-1 rounded text-xs bg-indigo-600 hover:bg-indigo-500 text-white"
                      aria-label="Apply selected preset to simulation"
                    >Apply Preset</button>
                  </div>
                  <p className="text-[11px] text-gray-400 mt-1">1: near-circular • 2: low speed (inward) • 3: high speed (outward)</p>
                </div>
              </div>
            </div>

            {/* Control Panel (Right side - 1/3 width) */}
            <div className="space-y-6">
              {/* Parameters */}
              <div className="panel">
                <h3 className="text-lg font-bold text-accent-chi mb-4">Parameters</h3>

                <div className="space-y-4">
                  <ParameterSlider 
                    label="Earth/Moon Mass Ratio" 
                    value={state.params.massRatio} 
                    min={0.1} 
                    max={10} 
                    step={0.1} 
                    unit="×" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'massRatio', value: v } })} 
                    tooltip="Earth is 81.3× more massive than the Moon. Try different ratios to see Jupiter-moon systems!"
                  />
                  <ParameterSlider 
                    label="Distance Between Bodies" 
                    value={state.params.orbitalDistance} 
                    min={1.0} 
                    max={10} 
                    step={0.1} 
                    unit="units" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: v } })} 
                    tooltip="Initial separation between Earth and Moon. Larger distances = slower, wider orbits."
                  />
                  <ParameterSlider 
                    label="Gravity Strength" 
                    value={state.params.chiStrength} 
                    min={0.05} 
                    max={0.5} 
                    step={0.01} 
                    unit="" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'chiStrength', value: v } })} 
                    tooltip="Chi field coupling strength - determines how strongly the field gradient pulls objects together."
                  />
                  <ParameterSlider 
                    label="Gravity Reach (σ)" 
                    value={state.params.sigma} 
                    min={0.5} 
                    max={4.0} 
                    step={0.1} 
                    unit="" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'sigma', value: v } })} 
                    tooltip="Gaussian width of chi field - how far the gravity 'reaches'. Larger σ = longer-range force."
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
                    tooltip="Controls how fast the emergent spacetime evolves - higher = more physics steps per frame."
                  />
                  <ParameterSlider 
                    label="Timestep (dt)" 
                    value={state.params.dt} 
                    min={0.001} 
                    max={0.005} 
                    step={0.0005} 
                    unit="" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'dt', value: v } })} 
                    tooltip="Physics integration timestep - smaller = more accurate but needs more steps. Affects numerical stability."
                  />
                </div>

                {/* Educational preset readout */}
                <div className="mt-3 text-[11px] text-gray-400 flex flex-wrap gap-x-3 gap-y-1 items-center">
                  <span className="font-mono">Angle: {state.params.startAngleDeg.toFixed(0)}°</span>
                  <span className="font-mono">v scale: {state.params.velocityScale.toFixed(2)}×</span>
                  <span className="opacity-70">Preset {state.params.startPreset}</span>
                </div>

                <div className="mt-4 bg-gray-800/50 rounded p-4">
                  <h4 className="font-semibold text-sm mb-2">Why is a perfect orbit hard?</h4>
                  <p className="text-xs text-gray-300 mb-2">Try presets 2 and 3 and watch the orbit drift inward or outward.</p>
                  <details className="group">
                    <summary className="cursor-pointer text-indigo-300 text-xs hover:text-indigo-200">Check Answer</summary>
                    <div className="mt-2 text-[11px] leading-relaxed text-gray-200">
                      A perfectly circular orbit requires matching centrifugal demand v²/r with inward acceleration.
                      In this emergent model, inward acceleration comes from a discretized Gaussian chi field and a Verlet time integrator.
                      Small deviations (a few %) in tangential speed change energy and angular momentum, causing slow radial drift.
                      Discretization (dx), timestep (dt), and finite Gaussian width (σ) slightly distort the effective force.
                      Preset 2 undershoots v_circ so gravity dominates (spiral inward). Preset 3 overshoots, adding excess kinetic energy (spiral outward).
                      This turns sensitivity into a teaching tool: orbits are naturally a fine balance, and numerical representation makes the balance visibly delicate.
                    </div>
                  </details>
                </div>
              </div>

              {/* Metrics */}
              <div className="panel">
                <h3 className="text-lg font-bold text-accent-chi mb-4">System Stats</h3>
                
                <div className="space-y-3">
                  <MetricDisplay label="Total Energy" value={state.metrics.energy} status={state.metrics.energy === '—' ? 'neutral' : 'conserved'} />
                  <MetricDisplay label="Energy Conservation" value={state.metrics.drift} status={state.metrics.drift === '—' ? 'neutral' : 'good'} />
                  <MetricDisplay label="Spin (Angular Momentum)" value={state.metrics.angularMomentum} status={state.metrics.angularMomentum === '—' ? 'neutral' : 'conserved'} />
                  <MetricDisplay label="Separation (r)" value={state.metrics.separation} status={state.metrics.separation === '—' ? 'neutral' : 'good'} />
                  <MetricDisplay label="Speed / v_circ" value={state.metrics.vRatio} status={state.metrics.vRatio === '—' ? 'neutral' : 'warning'} />
                  <MetricDisplay label="Effective Speed" value={state.metrics.effectiveSpeed + '×'} status={'good'} />
                  <MetricDisplay label="Time for One Orbit" value={state.metrics.orbitalPeriod} status="neutral" />
                  <MetricDisplay label="Frame Rate" value={state.metrics.fps} status={state.metrics.fps === '—' ? 'neutral' : 'good'} />
                </div>
              </div>
            </div>
          </div>

          {/* Explanation Panel */}
          <div className="mt-8 panel">
            <h3 className="text-xl font-bold text-accent-chi mb-4">What You're Seeing</h3>
            <div className="prose prose-invert max-w-none text-text-secondary">
              <p className="mb-4">
                This is <strong className="text-accent-chi">not a traditional gravity simulation</strong>. 
                Instead of using Newton's laws, the Earth-Moon orbit <strong>emerges naturally</strong> from a wave-like field equation:
              </p>
              <div className="bg-space-dark p-4 rounded-lg font-mono text-accent-chi text-center my-4">
                ∂²E/∂t² = c²∇²E − χ²(x,t)E
              </div>
              <p className="mb-4 text-text-primary">
                <strong>Think of it like this:</strong> Imagine space as a pond. Each body creates ripples (the chi field). 
                These ripples push objects together — that's gravity! No mysterious "force at a distance" needed.
              </p>
              <ul className="space-y-2 list-disc list-inside">
                <li><strong>No Newton's laws programmed in</strong> — Gravity naturally emerges from the field</li>
                <li><strong>Like ripples in a pond</strong> — Each body creates a "dip" in the field that pulls objects toward it</li>
                <li><strong>Energy stays constant</strong> — Just like a pendulum, total energy is conserved (watch "Energy Conservation" stay near 0%)</li>
                <li><strong>Real physics</strong> — Earth is 81× more massive than the Moon, so it barely moves while Moon orbits</li>
              </ul>
              <p className="mt-4">
                <strong>Try experimenting:</strong> Increase "Gravity Strength" to make orbits tighter. 
                Increase "Distance Between Bodies" to make orbits wider. Change "Earth/Moon Mass Ratio" to see different scenarios 
                (like Jupiter and its moons!). The <strong>🌍🌙 Reset to Defaults</strong> button brings back Earth-Moon.
              </p>
              <div className="mt-6 p-4 bg-accent-chi/10 border-l-4 border-accent-chi rounded">
                <p className="text-text-primary font-semibold mb-2">
                  Want to understand the full scientific context?
                </p>
                <Link 
                  href="/about" 
                  className="inline-flex items-center gap-2 text-accent-chi hover:text-accent-particle transition-colors font-semibold"
                >
                  <span>Read our full About page</span>
                  <span>→</span>
                </Link>
                <p className="text-sm text-text-secondary mt-2">
                  Learn about the mathematics, limitations, and how to access our research data and source code.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

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
        <span className="text-sm font-mono text-accent-chi">{value.toFixed(1)} {unit}</span>
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
        className="w-full h-2 bg-space-border rounded-lg appearance-none cursor-pointer accent-accent-chi"
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
        className="w-5 h-5 rounded border-2 border-accent-chi checked:bg-accent-chi checked:border-accent-chi focus:ring-2 focus:ring-accent-chi/50 transition-colors"
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

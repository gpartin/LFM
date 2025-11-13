/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import Link from 'next/link';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import StandardVisualizationOptions from '@/components/experiment/StandardVisualizationOptions';
import StandardMetricsPanel from '@/components/experiment/StandardMetricsPanel';
import ParameterSlider from '@/components/ui/ParameterSlider';
import { detectBackend } from '@/physics/core/backend-detector';
import { BinaryOrbitSimulation, OrbitConfig } from '@/physics/forces/binary-orbit';
import UniversalCanvas from '@/components/visuals/UniversalCanvas';
import { useSimulationState } from '@/hooks/useSimulationState';
import { WebGPUErrorBoundary } from '@/components/ErrorBoundary';
import { decideSimulationProfile } from '@/physics/core/simulation-profile';
import SimpleCanvas from '@/components/visuals/SimpleCanvas';

export default function BlackHolePage() {
  // Consolidated state management
  const [state, dispatch] = useSimulationState();
  const [uiMode, setUiMode] = useState<'advanced' | 'simple'>('advanced');
  const [dimMode, setDimMode] = useState<'3d' | '1d'>('3d');
  // Showcase extras
  const [tidalStretch, setTidalStretch] = useState<boolean>(true);
  const [cohesion, setCohesion] = useState<number>(0.4); // Lower default = easier disruption
  
  // Refs
  const isRunningRef = useRef<boolean>(false);
  useEffect(() => { isRunningRef.current = state.isRunning; }, [state.isRunning]);

  const effectiveSpeedRef = useRef<number>(50.0);
  const wasRunningBeforeDragRef = useRef<boolean>(false);

  const deviceRef = useRef<GPUDevice | null>(null);
  const simRef = useRef<BinaryOrbitSimulation | null>(null);
  const firstPlayNudgedRef = useRef<boolean>(false);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const prevEnergyRef = useRef<{ E: number; t: number } | null>(null);
  // If true, start running immediately after next successful init
  const autoStartOnInitRef = useRef<boolean>(false);

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
      const prof = decideSimulationProfile(caps.backend === 'webgpu' || caps.backend === 'cpu' ? caps.backend : 'cpu', caps, 'classical');
      setUiMode(prof.ui);
      setDimMode(prof.dim);
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
        
  // Black hole configuration sized to lattice domain (avoid edge truncation)
  // RS_analogue = (sigma × sqrt(mass)) / sqrt(2)
  // Choose mass so RS stays well inside the domain at default sigma.
  const blackHoleMass = 16; // 16× moon mass → RS ≈ 2.83 when sigma=1.0
        const moonMass = 1.0;
        
        const actualLatticeSize = Math.min(state.capabilities?.maxLatticeSize ?? state.params.latticeSize, state.params.latticeSize);
        
        // Ensure the moon starts outside the analogue horizon (RS)
        const rsAnalogue = (state.params.sigma * Math.sqrt(blackHoleMass)) / Math.SQRT2;
        const minStartSep = rsAnalogue * 1.15; // start a bit beyond RS for clarity
        const chosenStartSep = Math.max(state.params.orbitalDistance, minStartSep);

        const config: OrbitConfig = {
          mass1: blackHoleMass,
          mass2: moonMass,
          initialSeparation: chosenStartSep,
          chiStrength: state.params.chiStrength,
          latticeSize: actualLatticeSize,
          dt: 0.0005, // Smaller timestep for stability
          sigma: state.params.sigma,
          startAngleDeg: 0,
          velocityScale: state.params.velocityScale,
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

        // If we adjusted the separation internally, surface the value in metrics for transparency
        try {
          const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
          const secondary = s.particle1.mass >= s.particle2.mass ? s.particle2 : s.particle1;
          const dx = primary.position[0] - secondary.position[0];
          const dy = primary.position[1] - secondary.position[1];
          const dz = primary.position[2] - secondary.position[2];
          const sep = Math.sqrt(dx*dx + dy*dy + dz*dz);
          dispatch({ type: 'UPDATE_METRICS', payload: { separation: sep.toFixed(3) } });
        } catch {}

        // Auto-start if flagged (e.g., after Approach preset)
        if (autoStartOnInitRef.current) {
          autoStartOnInitRef.current = false;
          isRunningRef.current = true;
          dispatch({ type: 'SET_RUNNING', payload: true });
          rafRef.current = requestAnimationFrame(tick);
        }
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
  }, [state.backend, state.params.latticeSize, state.params.orbitalDistance, state.params.chiStrength, state.params.sigma, state.params.velocityScale, state.resetTrigger, state.capabilities, stopSimulation, dispatch]);

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
    // If simulation isn't ready yet (still initializing), keep the RAF loop alive
    if (!sim) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }

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
      // Apply soft horizon damping if inside analogue RS to avoid fly-through
      try {
        const sNow = sim.getState();
        const primary = sNow.particle1.mass >= sNow.particle2.mass ? sNow.particle1 : sNow.particle2;
        const secondary = sNow.particle1.mass >= sNow.particle2.mass ? sNow.particle2 : sNow.particle1;
        const dxh = secondary.position[0] - primary.position[0];
        const dyh = secondary.position[1] - primary.position[1];
        const dzh = secondary.position[2] - primary.position[2];
        const sepH = Math.hypot(dxh, dyh, dzh);
        const rsH = (state.params.sigma * Math.sqrt(Math.max(1e-6, primary.mass))) / Math.SQRT2;
        if (sepH <= rsH) {
          sim.applyHorizonDamping?.(rsH, 0.85, 0.97);
        }
      } catch {}
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
      // LFM-derived GR-like metrics
      const secondary = s.particle1.mass >= s.particle2.mass ? s.particle2 : s.particle1;
      const chiLocal = sim.analyticChiAt(secondary.position as [number, number, number]);
      const chiBase = sim.chiBaseline();
      const grad = sim.analyticChiGradientAt(secondary.position as [number, number, number]);
      const gradMag = Math.hypot(grad[0], grad[1], grad[2]);
      // Tidal (radial derivative of grad along radial)
      const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
      const rx = secondary.position[0] - primary.position[0];
      const ry = secondary.position[1] - primary.position[1];
      const rz = secondary.position[2] - primary.position[2];
      const r = Math.max(1e-8, Math.hypot(rx, ry, rz));
      const rhat: [number, number, number] = [rx / r, ry / r, rz / r];
      const h = Math.min(0.1, Math.max(0.02, r * 0.02));
      const pPlus: [number, number, number] = [secondary.position[0] + rhat[0] * h, secondary.position[1] + rhat[1] * h, secondary.position[2] + rhat[2] * h];
      const pMinus: [number, number, number] = [secondary.position[0] - rhat[0] * h, secondary.position[1] - rhat[1] * h, secondary.position[2] - rhat[2] * h];
      const gPlus = sim.analyticChiGradientAt(pPlus);
      const gMinus = sim.analyticChiGradientAt(pMinus);
      const gPlusDot = gPlus[0]*rhat[0] + gPlus[1]*rhat[1] + gPlus[2]*rhat[2];
      const gMinusDot = gMinus[0]*rhat[0] + gMinus[1]*rhat[1] + gMinus[2]*rhat[2];
      const tidalRad = (gPlusDot - gMinusDot) / (2 * h);
      // Tidal stress ratio (aTide / gSelf) for disruption tracking
      const moonRadius = 0.15;
      const aTide = Math.abs(tidalRad) * moonRadius;
      const gSelf = Math.max(1e-6, cohesion * secondary.mass / (moonRadius * moonRadius));
      const tidalStress = aTide / gSelf;
      metricsUpdate.tidalStress = tidalStress.toFixed(2);
      // Disruption status based on stress vs cohesion threshold
      if (tidalStress < 0.8) {
        metricsUpdate.disruptionStatus = 'Safe';
      } else if (tidalStress < cohesion * 2) {
        metricsUpdate.disruptionStatus = 'Stressed';
      } else {
        metricsUpdate.disruptionStatus = 'Disrupting!';
      }
      // Energy breakdown and rate (per simulated second)
      const eb = sim.getEnergyBreakdown?.();
      if (eb) {
        metricsUpdate.energyKE = eb.kinetic.toFixed(4) + ' J';
        metricsUpdate.energyField = eb.field.toFixed(4) + ' J';
        const prev = prevEnergyRef.current;
        if (prev && s.time > prev.t) {
          const dEdt = (eb.total - prev.E) / (s.time - prev.t);
          metricsUpdate.energyRate = dEdt.toExponential(2) + ' J/s';
        }
        prevEnergyRef.current = { E: eb.total, t: s.time };
      }
      // Clock rate analogue from chi (scaled, dimensionless)
      const fieldIntensity = chiLocal - chiBase;
      const alpha = 0.5; // visual scaling only
      const clockRate = 1 / Math.max(1e-6, 1 + alpha * Math.max(0, fieldIntensity));
      metricsUpdate.clockRate = clockRate.toFixed(3) + '×';
      metricsUpdate.chiLocal = chiLocal.toFixed(3);
      metricsUpdate.gradMag = gradMag.toFixed(3);
      metricsUpdate.tidalRad = tidalRad.toExponential(2);
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
      if (state.backend !== 'webgpu') {
        console.warn('[BlackHole] Cannot start: backend is', state.backend);
        return;
      }
      if (!simRef.current) {
        console.warn('[BlackHole] Cannot start: simulation not initialized');
        return;
      }
      if (isRunningRef.current) {
        console.warn('[BlackHole] Already running');
        return;
      }
      console.log('[BlackHole] Starting simulation');
    // One-time gentle nudge toward the black hole on the first Play to ensure motion
    try {
      if (!firstPlayNudgedRef.current) {
        simRef.current.applyRadialNudgeTowardPrimary?.(0.05);
        firstPlayNudgedRef.current = true;
      }
    } catch (e) {
      console.warn('[BlackHole] First-play nudge failed:', e);
    }
    isRunningRef.current = true;
    dispatch({ type: 'SET_RUNNING', payload: true });
    rafRef.current = requestAnimationFrame(tick);
  }, [state.backend, tick, dispatch]);

  const resetSimulation = useCallback(async () => {
    stopSimulation();
    try {
      simRef.current?.reset();
      await simRef.current?.initialize();
      // After a full reset, allow the first-play nudge again
      firstPlayNudgedRef.current = false;
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
    <ExperimentLayout
      title="🌌 Black Hole Orbit Simulation"
      description="Witness extreme gravity in action. A tiny black hole warps spacetime—watch a moon dance in its gravitational grip."
      backend={state.backend}
      experimentId="black-hole"
      visualizationOptions={
        <StandardVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
          showAdvancedOptions={true}
          labelOverrides={{
            showParticles: 'Black Hole & Moon',
            showTrails: 'Orbital Paths',
            showIsoShells: 'Event Horizon Shell',
          }}
          additionalControls={
            <div className="flex items-center gap-4">
              <label className="flex items-center space-x-2">
                <input type="checkbox" checked={tidalStretch} onChange={(e) => setTidalStretch(e.target.checked)} className="accent-purple-500" />
                <span className="text-text-primary">Tidal Stretch</span>
              </label>
              <div className="w-56">
                <ParameterSlider 
                  label="Cohesion"
                  value={cohesion}
                  min={0.3}
                  max={3.0}
                  step={0.1}
                  unit="×"
                  onChange={(v) => setCohesion(v)}
                  tooltip="Higher = stronger self-gravity; lowers stretch"
                />
              </div>
            </div>
          }
        />
      }
      footerContent={
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
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* 3D Canvas */}
            <div className="lg:col-span-3">
              <div ref={canvasContainerRef} className="panel h-[600px] relative overflow-hidden">
                {state.ui.fastForward && (
                  <div className="absolute top-2 right-2 z-10 px-3 py-1 rounded bg-yellow-500/20 border border-yellow-500/50 text-yellow-300 text-xs font-semibold tracking-wide">
                    FAST-FORWARD
                  </div>
                )}
                {uiMode === 'advanced' ? (
                  <WebGPUErrorBoundary>
                    <UniversalCanvas
                      kind="orbit"
                      simulation={simRef}
                      isRunning={state.isRunning}
                      ui={state.ui}
                      chiStrength={state.params.chiStrength}
                      sigma={state.params.sigma}
                      showBHRings={state.ui.showIsoShells}
                      tidalStretch={tidalStretch}
                      selfGravityFactor={cohesion}
                    />
                  </WebGPUErrorBoundary>
                ) : (
                  <SimpleCanvas
                    isRunning={state.isRunning}
                    parameters={{ ...state.params, __dim: dimMode }}
                    views={{ showGrid: false, showField: false }}
                  />
                )}
              </div>

              {/* Controls */}
              <div className="mt-4 flex items-center justify-center space-x-4" role="group" aria-label="Simulation controls">
                <button
                  onClick={startSimulation}
                  disabled={uiMode !== 'advanced' || state.isRunning}
                  aria-label="Start simulation"
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
                    uiMode !== 'advanced' || state.isRunning
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
                  ⚫ Reset
                </button>
                <button
                  onClick={() => {
                    stopSimulation();
                    // Set velocity to 65% of circular to force inward spiral
                    dispatch({ type: 'UPDATE_PARAM', payload: { key: 'velocityScale', value: 0.65 } });
                    // Start automatically after re-init
                    autoStartOnInitRef.current = true;
                  }}
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-red-600 hover:bg-red-500 text-white"
                  aria-label="Set approach trajectory"
                >
                  🔥 Approach
                </button>
                <button
                  onClick={() => {
                    stopSimulation();
                    // Stable: start outside 3 RS and use circular velocity
                    try {
                      const mBH = 16; // keep in sync with init
                      const rs = (state.params.sigma * Math.sqrt(mBH)) / Math.SQRT2;
                      const targetR = 3.5 * rs;
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: targetR } });
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'velocityScale', value: 1.0 } });
                      autoStartOnInitRef.current = true;
                    } catch {}
                  }}
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-emerald-600 hover:bg-emerald-500 text-white"
                  aria-label="Stable orbit preset"
                >
                  🟢 Stable
                </button>
                <button
                  onClick={() => {
                    stopSimulation();
                    // Plunge: inside 2 RS with sub-circular speed
                    try {
                      const mBH = 16;
                      const rs = (state.params.sigma * Math.sqrt(mBH)) / Math.SQRT2;
                      const targetR = 1.8 * rs;
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: targetR } });
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'velocityScale', value: 0.5 } });
                      autoStartOnInitRef.current = true;
                    } catch {}
                  }}
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-orange-600 hover:bg-orange-500 text-white"
                  aria-label="Plunge preset"
                >
                  🟠 Plunge
                </button>
              </div>
            </div>

            {/* Control Panel */}
            <div className="space-y-6">
              {/* Unified Experiment Parameters */}
              <div className="panel" data-panel="experiment-parameters">
                <h3 className="text-lg font-bold text-purple-400 mb-4">Experiment Parameters</h3>

                <div className="space-y-4" data-section="profile-parameters">
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
                    min={250} 
                    max={750} 
                    step={10} 
                    unit="×" 
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })} 
                    onDragStart={handleSpeedDragStart} 
                    onDragEnd={handleSpeedDragEnd} 
                    tooltip="Simulation speed - higher = more physics steps per frame."
                  />
                </div>
                <div className="sr-only" aria-hidden="true" data-section="experiment-parameters" />
              </div>

              {/* Metrics */}
              {(() => {
                const sim = simRef.current as any;
                let rsDisplay = '—';
                let crossedDisplay = '—';
                try {
                  if (sim) {
                    const s = sim.getState();
                    const primary = s.particle1.mass >= s.particle2.mass ? s.particle1 : s.particle2;
                    const secondary = s.particle1.mass >= s.particle2.mass ? s.particle2 : s.particle1;
                    const sigma = state.params.sigma; // Get from state, not private config
                    const rs = (sigma * Math.sqrt(Math.max(1e-6, primary.mass))) / Math.SQRT2;
                    rsDisplay = rs.toFixed(2) + ' u';
                    const dx = primary.position[0] - secondary.position[0];
                    const dy = primary.position[1] - secondary.position[1];
                    const dz = primary.position[2] - secondary.position[2];
                    const sep = Math.sqrt(dx*dx + dy*dy + dz*dz);
                    crossedDisplay = sep <= rs ? 'yes' : 'no';
                  }
                } catch {}
                return (
                  <StandardMetricsPanel
                coreMetrics={{
                  energy: state.metrics.energy,
                  drift: state.metrics.drift,
                  angularMomentum: state.metrics.angularMomentum,
                }}
                additionalMetrics={[
                  { label: 'Distance from Black Hole', value: state.metrics.separation, status: 'warning' },
                  { label: 'Analogue Horizon (RS)', value: rsDisplay, status: 'neutral' },
                  { label: 'Crossed Horizon?', value: crossedDisplay, status: (crossedDisplay === 'yes' ? 'warning' : 'neutral') },
                  { label: 'Tidal Stress Ratio (aTide/gSelf)', value: state.metrics.tidalStress ?? '—', status: state.metrics.disruptionStatus === 'Disrupting!' ? 'warning' : (state.metrics.disruptionStatus === 'Stressed' ? 'warning' : 'neutral') },
                  { label: 'Body Integrity', value: state.metrics.disruptionStatus ?? '—', status: state.metrics.disruptionStatus === 'Disrupting!' ? 'warning' : (state.metrics.disruptionStatus === 'Stressed' ? 'warning' : 'good') },
                  { label: 'Orbital Speed vs Circular (v/v_circ)', value: state.metrics.vRatio, status: 'neutral' },
                  { label: 'Field Intensity at Moon (χ local)', value: state.metrics.chiLocal ?? '—', status: 'neutral' },
                  { label: 'Gravitational Pull Strength (|∇χ|)', value: state.metrics.gradMag ?? '—', status: 'neutral' },
                  { label: 'Tidal Stretching Force (∂/∂r (∇χ·r̂))', value: state.metrics.tidalRad ?? '—', status: 'neutral' },
                  { label: 'Kinetic Energy (KE)', value: state.metrics.energyKE ?? '—', status: 'neutral' },
                  { label: 'Field Energy', value: state.metrics.energyField ?? '—', status: 'neutral' },
                  { label: 'Energy Loss Rate (dE/dt)', value: state.metrics.energyRate ?? '—', status: 'neutral' },
                  { label: 'Time Dilation Effect (clock rate)', value: state.metrics.clockRate ?? '—', status: 'neutral' },
                  { label: 'Effective Speed', value: state.metrics.effectiveSpeed + '×', status: 'good' },
                  { label: 'Frame Rate', value: state.metrics.fps, status: 'good' },
                ]}
                title="System Metrics"
                titleColorClass="text-purple-400"
              />
                );
              })()}

            </div>
          </div>
    </ExperimentLayout>
  );
}

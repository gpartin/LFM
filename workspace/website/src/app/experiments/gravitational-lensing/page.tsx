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
import UniversalCanvas from '@/components/visuals/UniversalCanvas';
import { useSimulationState } from '@/hooks/useSimulationState';
import { decideSimulationProfile } from '@/physics/core/simulation-profile';
import SimpleCanvas from '@/components/visuals/SimpleCanvas';

// Gravitational lensing scenario presets
type LensingScenario = 'solar-eclipse' | 'galaxy-cluster' | 'black-hole';

interface ScenarioConfig {
  lensMass: number;       // Mass of lensing object
  sigma: number;          // Field concentration
  chiStrength: number;    // Field coupling strength
  impactParameter: number; // Closest approach distance
  particleSpeed: number;  // Light-speed analogue
  color: string;          // Visual color for lens
  description: string;    // Scenario description
}

const LENSING_SCENARIOS: Record<LensingScenario, ScenarioConfig> = {
  'solar-eclipse': {
    lensMass: 1000,
    sigma: 1.0,
    chiStrength: 0.25,
    impactParameter: 1.5,
    particleSpeed: 0.9, // ~0.9c
    color: '#FDB813', // Solar yellow
    description: '1919 Solar Eclipse - Einstein\'s first GR test (1.75" deflection)'
  },
  'galaxy-cluster': {
    lensMass: 5000,
    sigma: 2.0,
    chiStrength: 0.5,
    impactParameter: 3.0,
    particleSpeed: 0.95,
    color: '#9D4EDD', // Purple galaxy
    description: 'Galaxy cluster strong lensing - Einstein rings and arcs'
  },
  'black-hole': {
    lensMass: 10000,
    sigma: 0.5,
    chiStrength: 0.8,
    impactParameter: 2.0,
    particleSpeed: 0.99,
    color: '#000000', // Black
    description: 'Black hole photon sphere - extreme light bending'
  }
};

// Metrics derived from lensing configuration
function estimateDeflectionAngle(mass: number, sigma: number, impactParameter: number): number {
  // Einstein deflection angle: α ≈ 4GM/(c²b) in GR
  // LFM analogue: proportional to field gradient at closest approach
  // Simplified: α ∝ M/(σ²×b)
  return (mass / (sigma * sigma * impactParameter)) * 0.001; // Scale to arcseconds
}

function estimateEinsteinRadius(mass: number, sigma: number): number {
  // Einstein radius: θ_E ∝ sqrt(M/D_L)
  // LFM: characteristic lensing scale from field parameters
  return Math.sqrt(mass * sigma) * 0.1;
}

function estimateLightTravelTime(impactParameter: number, mass: number): number {
  // Shapiro time delay: Δt ∝ GM/c³ × ln(b)
  // LFM: light slowed in strong fields
  return mass * Math.log(impactParameter + 1) * 0.00001;
}

export default function GravitationalLensingPage() {
  const [state, dispatch] = useSimulationState();
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  const [uiMode, setUiMode] = useState<'advanced' | 'simple'>('advanced');
  const [dimMode, setDimMode] = useState<'3d' | '1d'>('3d');
  const [scenario, setScenario] = useState<LensingScenario>('solar-eclipse');
  const [showLightDebug, setShowLightDebug] = useState<boolean>(false);
  
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
      const prof = decideSimulationProfile(effectiveBackend, caps, 'classical');
      setUiMode(prof.ui);
      setDimMode(prof.dim);
      // Lens-specific default visuals: turn ONLY lensing effects on by default
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showVectors', value: false } });
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showTrails', value: false } });
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showIsoShells', value: false } });
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showDomes', value: false } });
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showChi', value: false } });
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showLattice', value: false } });
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showParticles', value: true } });
      // Default: turn gravity well surface OFF for this experiment on load
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showWell', value: false } });
      // Also keep the decorative starfield/background OFF so only lensing layers are visible
      dispatch({ type: 'UPDATE_UI', payload: { key: 'showBackground', value: false } });
    });
  }, [dispatch]);

  // Create simulation for gravitational lensing
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

        // Get scenario configuration
        const scenarioConfig = LENSING_SCENARIOS[scenario];
        
        // Light ray (photon) - very small mass, very high velocity
        const photonMass = 0.01;
        const actualLatticeSize = Math.min(state.capabilities?.maxLatticeSize ?? state.params.latticeSize, state.params.latticeSize);

        const config: OrbitConfig = {
          mass1: scenarioConfig.lensMass,
          mass2: photonMass,
          initialSeparation: scenarioConfig.impactParameter * 2, // Start far away
          chiStrength: scenarioConfig.chiStrength,
          latticeSize: actualLatticeSize,
          dt: 0.0001, // Small timestep for fast particles
          sigma: scenarioConfig.sigma,
          startAngleDeg: 180, // Coming from the left
          velocityScale: scenarioConfig.particleSpeed, // Near light speed
        };

        try {
          simRef.current?.destroy();
        } catch (e) {
          console.error('[GravitationalLensing] Error destroying previous simulation:', e);
        }

        const sim = new BinaryOrbitSimulation(device, config);
        await sim.initialize();
        if (cancelled) { sim.destroy(); return; }
        simRef.current = sim;

        const s = sim.getState();
        
        // Calculate lensing metrics
        const deflectionAngle = estimateDeflectionAngle(
          scenarioConfig.lensMass, 
          scenarioConfig.sigma, 
          scenarioConfig.impactParameter
        );
        const einsteinRadius = estimateEinsteinRadius(
          scenarioConfig.lensMass, 
          scenarioConfig.sigma
        );
        const travelTime = estimateLightTravelTime(
          scenarioConfig.impactParameter, 
          scenarioConfig.lensMass
        );
        
        dispatch({
          type: 'UPDATE_METRICS',
          payload: {
            angularMomentum: s.angularMomentum.toFixed(3),
            energy: s.energy ? s.energy.toFixed(4) + ' J' : '—',
            drift: (sim.getEnergyDrift() * 100).toFixed(4) + '%',
            lensingScenario: scenario,
            deflectionAngle: deflectionAngle.toFixed(3),
            impactParameter: scenarioConfig.impactParameter.toFixed(2),
            einsteinRadius: einsteinRadius.toFixed(2),
            lightDelay: travelTime.toFixed(6),
          },
        });
      } catch (e) {
        console.error('Gravitational lensing simulation init failed:', e);
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
  }, [state.backend, state.params.latticeSize, state.resetTrigger, state.capabilities, stopSimulation, dispatch, scenario]);

  const tick = useCallback(async (t: number) => {
    if (!isRunningRef.current) return;
    const sim = simRef.current;
    if (!sim) return;

    // Step simulation
    const stepsPerFrame = Math.round(state.params.simSpeed || 50);
    await sim.stepBatch(Math.min(stepsPerFrame, 200));

    // Update metrics
    const s = sim.getState();
    const driftVal = sim.getEnergyDrift();
    
    // Get current scenario config for lensing metrics
    const scenarioConfig = LENSING_SCENARIOS[scenario];
    const deflectionAngle = estimateDeflectionAngle(
      scenarioConfig.lensMass, 
      scenarioConfig.sigma, 
      scenarioConfig.impactParameter
    );
    const einsteinRadius = estimateEinsteinRadius(
      scenarioConfig.lensMass, 
      scenarioConfig.sigma
    );
    const travelTime = estimateLightTravelTime(
      scenarioConfig.impactParameter, 
      scenarioConfig.lensMass
    );
    
    const metricsUpdate: Record<string, string> = {};

    if (isFinite(s.angularMomentum)) metricsUpdate.angularMomentum = s.angularMomentum.toFixed(3);
    if (isFinite(driftVal)) metricsUpdate.drift = (driftVal * 100).toFixed(4) + '%';
    if (isFinite(s.energy)) metricsUpdate.energy = s.energy.toFixed(4) + ' J';
    
    // Lensing metrics
    metricsUpdate.lensingScenario = scenario;
    metricsUpdate.deflectionAngle = deflectionAngle.toFixed(3);
    metricsUpdate.impactParameter = scenarioConfig.impactParameter.toFixed(2);
    metricsUpdate.einsteinRadius = einsteinRadius.toFixed(2);
    metricsUpdate.lightDelay = travelTime.toFixed(6);

    if (Object.keys(metricsUpdate).length > 0) {
      dispatch({ type: 'UPDATE_METRICS', payload: metricsUpdate });
    }

    if (isRunningRef.current) rafRef.current = requestAnimationFrame(tick);
  }, [state.params.simSpeed, dispatch, scenario]);

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
          <div className="text-4xl mb-4 animate-pulse">💫</div>
          <div className="text-xl text-text-primary">Initializing Gravitational Lensing...</div>
          <div className="text-sm text-text-secondary mt-2">Bending light rays</div>
        </div>
      </div>
    );
  }

  return (
    <ExperimentLayout
      title="💫 Gravitational Lensing"
      description="Watch light rays bend around massive objects due to LFM field gradients. Experience Einstein's 1919 eclipse observation, galaxy cluster lensing, and extreme photon sphere bending—all from pure χ field dynamics."
      backend={state.backend}
      experimentId="gravitational-lensing"
      visualizationOptions={
        <StandardVisualizationOptions
          state={state.ui}
          onChange={(key, value) => {
            if (key === 'lightDebug') {
              setShowLightDebug(Boolean(value));
            } else {
              dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } });
            }
          }}
          showAdvancedOptions={true}
          extraToggles={[{
            key: 'lightDebug',
            label: 'Light debug overlay (∇χ, v, emitter bounds)',
            checked: showLightDebug,
          }]}
        />
      }
      footerContent={
        <div className="mt-8 panel">
          <h3 className="text-xl font-bold text-purple-400 mb-4">Einstein's Prediction: Light Bends in Gravity</h3>
          <div className="prose prose-invert max-w-none text-text-secondary">
            <p className="mb-4">
              In 1919, <strong>Arthur Eddington's solar eclipse expedition</strong> confirmed Einstein's boldest prediction: 
              starlight bends when passing near the Sun. This experiment demonstrates that same phenomenon using 
              <strong>pure LFM χ field gradients</strong>—no curved spacetime required.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-6">
              <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                <div className="text-2xl mb-2">☀️</div>
                <h4 className="font-bold text-yellow-400 mb-2">Solar Eclipse (1919)</h4>
                <p className="text-sm">
                  Light from distant stars grazes the Sun during eclipse. Measured deflection: <strong>1.75 arcseconds</strong>. 
                  LFM shows this as χ field gradient deflection.
                </p>
              </div>
              
              <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                <div className="text-2xl mb-2">🌌</div>
                <h4 className="font-bold text-purple-400 mb-2">Galaxy Cluster</h4>
                <p className="text-sm">
                  Strong lensing creates <strong>Einstein rings</strong> and multiple images of background galaxies. 
                  Massive χ field distorts light paths dramatically.
                </p>
              </div>
              
              <div className="bg-red-500/10 p-4 rounded-lg border border-red-500/30">
                <div className="text-2xl mb-2">⚫</div>
                <h4 className="font-bold text-red-400 mb-2">Black Hole</h4>
                <p className="text-sm">
                  Extreme bending near event horizon. Light can orbit at the <strong>photon sphere</strong> (1.5× Schwarzschild radius). 
                  Creates black hole "shadow" seen by Event Horizon Telescope.
                </p>
              </div>
            </div>

            <div className="bg-space-dark p-4 rounded-lg font-mono text-purple-400 text-center my-4">
              χ field gradient → particle trajectory curvature → apparent position shift
            </div>
            
            <h4 className="text-lg font-bold text-purple-400 mt-6 mb-3">How LFM Produces Gravitational Lensing</h4>
            <ul className="space-y-2 list-disc list-inside">
              <li><strong>Field gradient force</strong> — Particles deflect toward steeper χ regions (∇χ acts like gravity)</li>
              <li><strong>Impact parameter</strong> — Closer passes experience stronger deflection (scales as M/b)</li>
              <li><strong>Einstein angle</strong> — Characteristic deflection scale θ_E ∝ √(M/D_L) emerges from field geometry</li>
              <li><strong>Light delay</strong> — Particles slow in dense χ fields (Shapiro time delay analogue)</li>
              <li><strong>Multiple images</strong> — Strong lensing creates multiple light paths converging at observer</li>
            </ul>
            
            <p className="mt-4 text-blue-400">
              <strong>🔬 Pure LFM Physics:</strong> This demonstration uses only χ field gradients and particle 
              trajectories—no spacetime curvature, no metric tensors, no coordinate transforms. The same field 
              dynamics that produce orbital mechanics also produce gravitational lensing.
            </p>
          </div>
        </div>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* 3D Canvas */}
            <div className="lg:col-span-3">
              <div className="panel h-[600px] relative overflow-hidden">
                {uiMode === 'advanced' ? (
                  <UniversalCanvas
                    kind="orbit"
                    simulation={simRef}
                    isRunning={state.isRunning}
                    ui={state.ui}
                    chiStrength={LENSING_SCENARIOS[scenario].chiStrength}
                    sigma={LENSING_SCENARIOS[scenario].sigma}
                    primaryColor={LENSING_SCENARIOS[scenario].color}
                    primaryScale={LENSING_SCENARIOS[scenario].sigma}
                    showBHRings={scenario === 'black-hole'}
                    showLightRays={true}
                    showLensingBackground={true}
                      rayConfig={{
                        // Very dense, very wide wavefront from far left
                        rows: scenario === 'galaxy-cluster' ? 80 : scenario === 'black-hole' ? 64 : 72,
                        cols: scenario === 'galaxy-cluster' ? 36 : scenario === 'black-hole' ? 28 : 30,
                        emitterOffset: 0.035,  // start even further left
                        emitterWidth: 0.35,    // much wider sheet across X
                        headSize: 0.06,        // slightly smaller heads for dense grid
                        color: '#fff1cc',
                        speed: 0.9,
                        spread: scenario === 'solar-eclipse' ? 0.9 : 1.0,
                        pure: true,
                        debug: showLightDebug,
                      }}
                    lightDebug={showLightDebug}
                    hideSecondaryParticle={true}
                  />
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
                  💫 Reset
                </button>
              </div>
              
              {/* Lensing Scenario Presets */}
              <div className="mt-4 panel">
                <h3 className="text-sm font-bold text-purple-400 mb-3">Lensing Scenarios</h3>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => {
                      stopSimulation();
                      setScenario('solar-eclipse');
                    }}
                    className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
                      scenario === 'solar-eclipse'
                        ? 'bg-yellow-500/30 text-yellow-200 border-2 border-yellow-400'
                        : 'bg-space-light/50 text-text-secondary hover:bg-space-light border border-space-light'
                    }`}
                  >
                    ☀️ Solar Eclipse
                  </button>
                  <button
                    onClick={() => {
                      stopSimulation();
                      setScenario('galaxy-cluster');
                    }}
                    className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
                      scenario === 'galaxy-cluster'
                        ? 'bg-purple-500/30 text-purple-200 border-2 border-purple-400'
                        : 'bg-space-light/50 text-text-secondary hover:bg-space-light border border-space-light'
                    }`}
                  >
                    🌌 Galaxy Cluster
                  </button>
                  <button
                    onClick={() => {
                      stopSimulation();
                      setScenario('black-hole');
                    }}
                    className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
                      scenario === 'black-hole'
                        ? 'bg-red-500/30 text-red-200 border-2 border-red-400'
                        : 'bg-space-light/50 text-text-secondary hover:bg-space-light border border-space-light'
                    }`}
                  >
                    ⚫ Black Hole
                  </button>
                </div>
                <p className="text-xs text-text-secondary mt-2">
                  {LENSING_SCENARIOS[scenario].description}
                </p>
              </div>
            </div>

            {/* Control Panel */}
            <div className="space-y-6">
              {/* Experiment Parameters */}
              <div className="panel" data-panel="experiment-parameters">
                <h3 className="text-lg font-bold text-purple-400 mb-4">Lensing Parameters</h3>

                <div className="space-y-4" data-section="profile-parameters">
                  <ParameterSlider
                    label="Impact Parameter"
                    value={state.params.orbitalDistance}
                    min={1.0}
                    max={6.0}
                    step={0.5}
                    unit="Rs"
                    onChange={(v) => {
                      const val = Number(v);
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'orbitalDistance', value: val } });
                      // Map impact parameter to initial separation for the sim
                      try {
                        simRef.current?.updateParameters({ initialSeparation: val * 2 });
                        simRef.current?.refreshChiField();
                      } catch {}
                    }}
                    tooltip="Closest approach distance of light ray to lens. Smaller = stronger bending."
                  />
                  <ParameterSlider
                    label="Lens Mass"
                    value={state.params.massRatio}
                    min={2}
                    max={20}
                    step={1}
                    unit="M☉"
                    onChange={(v) => {
                      const val = Number(v);
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'massRatio', value: val } });
                      // Scale mass around the scenario baseline (10 ≈ baseline)
                      const scenarioConfig = LENSING_SCENARIOS[scenario];
                      const scaledMass = (val / 10) * scenarioConfig.lensMass;
                      try {
                        simRef.current?.updateParameters({ mass1: Math.max(1, scaledMass) });
                        simRef.current?.refreshChiField();
                      } catch {}
                    }}
                    tooltip="Mass of lensing object. Higher mass = stronger field gradients = more deflection."
                  />
                  <ParameterSlider
                    label="Field Strength (χ)"
                    value={state.params.chiStrength || 0.5}
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    unit=""
                    onChange={(v) => {
                      const val = Number(v);
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'chiStrength', value: val } });
                      try {
                        simRef.current?.updateParameters({ chiStrength: val });
                        simRef.current?.refreshChiField();
                      } catch {}
                    }}
                    tooltip="Chi field coupling strength. Determines deflection magnitude."
                  />
                  <ParameterSlider
                    label="Field Concentration (σ)"
                    value={state.params.sigma || 1.0}
                    min={0.3}
                    max={3.0}
                    step={0.1}
                    unit=""
                    onChange={(v) => {
                      const val = Number(v);
                      dispatch({ type: 'UPDATE_PARAM', payload: { key: 'sigma', value: val } });
                      try {
                        simRef.current?.updateParameters({ sigma: val });
                        simRef.current?.refreshChiField();
                      } catch {}
                    }}
                    tooltip="Field concentration. Smaller = more compact lens, larger = diffuse lens."
                  />
                  <ParameterSlider
                    label="Playback Speed"
                    value={state.params.simSpeed || 50}
                    min={10}
                    max={200}
                    step={10}
                    unit="×"
                    onChange={(v) => dispatch({ type: 'UPDATE_PARAM', payload: { key: 'simSpeed', value: v } })}
                    tooltip="Simulation speed - higher = see light path faster."
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
                additionalMetrics={[
                  { 
                    label: 'Current Scenario', 
                    value: state.metrics.lensingScenario || '—', 
                    status: 'neutral' 
                  },
                  { 
                    label: 'Deflection Angle (arcsec)', 
                    value: state.metrics.deflectionAngle || '—', 
                    status: 'neutral' 
                  },
                  { 
                    label: 'Impact Parameter (Rs)', 
                    value: state.metrics.impactParameter || '—', 
                    status: 'neutral' 
                  },
                  { 
                    label: 'Einstein Radius (Rs)', 
                    value: state.metrics.einsteinRadius || '—', 
                    status: 'neutral' 
                  },
                  { 
                    label: 'Light Time Delay (s)', 
                    value: state.metrics.lightDelay || '—', 
                    status: 'neutral' 
                  },
                ]}
                title="Lensing Metrics"
                titleColorClass="text-purple-400"
              />
            </div>
          </div>
    </ExperimentLayout>
  );
}

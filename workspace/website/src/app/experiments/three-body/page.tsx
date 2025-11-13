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
import { NBodyOrbitSimulation, createFigure8ThreeBody } from '@/physics/forces/n-body-orbit';
import UniversalCanvas from '@/components/visuals/UniversalCanvas';
import { useSimulationState } from '@/hooks/useSimulationState';
import { decideSimulationProfile } from '@/physics/core/simulation-profile';
import SimpleCanvas from '@/components/visuals/SimpleCanvas';

export default function ThreeBodyPage() {
  const [state, dispatch] = useSimulationState();
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  const [uiMode, setUiMode] = useState<'advanced' | 'simple'>('advanced');
  const [dimMode, setDimMode] = useState<'3d' | '1d'>('3d');
  // Editable parameters (MVP)
  const [chiStrength, setChiStrength] = useState<number>(0.25);
  const [latticeN, setLatticeN] = useState<number>(64);
  const [simReady, setSimReady] = useState(0); // Force re-render when sim initializes
  
  const deviceRef = useRef<GPUDevice | null>(null);
  const simRef = useRef<NBodyOrbitSimulation | null>(null);
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
    });
  }, [dispatch]);

  // Initialize simulation - three-body preset with editable chiStrength and lattice size
  useEffect(() => {
    let cancelled = false;
    async function initSim() {
      try {
        let device: GPUDevice | null = null;
        if (backend === 'webgpu') {
          const adapter = await navigator.gpu?.requestAdapter();
          device = await adapter?.requestDevice() || null;
          if (!device) {
            console.warn('[ThreeBody] GPU requested but not available');
            return;
          }
          deviceRef.current = device;
        }
        
        if (cancelled) return;
        
        try { 
          simRef.current?.destroy(); 
        } catch (e) {
          console.error('[ThreeBody] Error destroying previous simulation:', e);
        }
        
        // Create classic figure-8 three-body orbit (Chenciner & Montgomery, 2000)
        const capMax = state.capabilities?.maxLatticeSize ?? latticeN;
        const actualLatticeSize = Math.min(capMax, latticeN);
        const sim = createFigure8ThreeBody(device, chiStrength, actualLatticeSize, false);
        await sim.initialize();
        if (cancelled) { sim.destroy(); return; }
        simRef.current = sim;
        setSimReady(prev => prev + 1); // Trigger re-render so NBodyCanvas shows initial state
      } catch (e) {
        console.error('Three-body simulation init failed:', e);
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
  }, [backend, state.capabilities, state.resetTrigger, stopSimulation, chiStrength, latticeN]);

  // Animation loop to step simulation
  useEffect(() => {
    if (!state.isRunning) {
      // Cancel any in-flight RAF when pausing
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }
    
    const tick = async () => {
      const sim = simRef.current;
      if (!sim || !isRunningRef.current) return;
      
      // Step simulation (10 steps per frame for smooth animation)
      await sim.stepBatch(10);
      
      // Update metrics
      const s = sim.getState();
      const driftVal = sim.getEnergyDrift();
      
      dispatch({
        type: 'UPDATE_METRICS',
        payload: {
          energy: s.energy.toFixed(4) + ' J',
          angularMomentum: s.angularMomentum.toFixed(3),
          drift: (driftVal * 100).toFixed(4) + '%',
        },
      });
      
      if (isRunningRef.current) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [state.isRunning, dispatch]);

  return (
    <ExperimentLayout
      title="🔺 Three-Body Problem"
      description="Watch three equal masses perform the famous figure-8 orbit—a choreographed dance discovered in 2000. This chaotic system emerges purely from chi field gradients with no programmed gravitational equations."
      backend={backend}
      experimentId="three-body"
      visualizationOptions={
        <StandardVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
          showAdvancedOptions={true}
        />
      }
      footerContent={
        <div className="mt-8 panel">
          <h2 className="text-2xl font-bold text-accent-chi mb-4">The Famous Figure-8 Orbit</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            <p className="text-sm text-yellow-400 mb-4">
              <strong>Note:</strong> Using the Chenciner-Montgomery figure-8 solution. You can now adjust chi strength and lattice resolution in the right panel; more controls coming soon.
            </p>
            <p>
              This simulation demonstrates the <strong>three-body problem</strong>—one of physics' most famous challenges. 
              Unlike two-body systems (which have analytical solutions), three bodies create chaotic dynamics where tiny 
              changes in initial conditions lead to completely different outcomes.
            </p>
            <p>
              <strong className="text-text-primary">The Figure-8 Solution:</strong> Discovered by Chenciner & Montgomery in 2000, 
              this is a special periodic orbit where three equal masses chase each other in a figure-8 pattern. It's one of the 
              few stable three-body configurations—most are chaotic and unpredictable.
            </p>
            <p>
              <strong className="text-text-primary">In LFM:</strong> No gravitational force equations are programmed. Three masses 
              create overlapping chi fields, and their motion emerges purely from field gradients. The choreographed dance you see 
              arises naturally from the lattice dynamics—gravity is emergent, not fundamental.
            </p>
            <p className="text-yellow-400">
              <strong>Coming Soon:</strong> Interactive controls for mass ratios, different initial configurations (Lagrange points, 
              unstable chaos), and 4+ body systems!
            </p>
          </div>
        </div>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-3">
              <div className="bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]">
                {uiMode === 'advanced' ? (
                  <UniversalCanvas
                    key={simReady}
                    kind="nbody"
                    simulation={simRef}
                    isRunning={state.isRunning}
                    ui={state.ui}
                    chiStrength={chiStrength}
                  />
                ) : (
                  <SimpleCanvas
                    isRunning={state.isRunning}
                    parameters={{ ...state.params, __dim: dimMode }}
                    views={{ showGrid: false, showField: false }}
                  />
                )}
              </div>
              
              <div className="mt-4 flex gap-4">
                <button
                  onClick={() => dispatch({ type: 'SET_RUNNING', payload: !state.isRunning })}
                  disabled={uiMode !== 'advanced'}
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
                    uiMode !== 'advanced'
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
                    simRef.current?.reset();
                    dispatch({ type: 'RESET_METRICS' });
                  }}
                  className="px-6 py-3 bg-space-border hover:bg-space-border/80 rounded-lg font-semibold transition-colors"
                >
                  🔄 Reset
                </button>
              </div>
            </div>

            <div className="lg:col-span-1 space-y-6">
              <div className="panel" data-panel="experiment-parameters">
                <h3 className="text-lg font-bold mb-4">Experiment Parameters</h3>
                <div className="space-y-5" data-section="profile-parameters">
                  <div className="text-sm text-text-secondary grid grid-cols-2 gap-2">
                    <div className="flex justify-between"><span>Number of Bodies:</span><span className="text-accent-chi font-semibold">3</span></div>
                    <div className="flex justify-between"><span>Mass (each):</span><span className="text-text-primary">1.0 M☉</span></div>
                    <div className="flex justify-between"><span>Initial Pattern:</span><span className="text-text-primary">Figure-8</span></div>
                  </div>
                  <ParameterSlider
                    label="Chi Strength"
                    value={chiStrength}
                    min={0.05}
                    max={0.6}
                    step={0.01}
                    unit=""
                    tooltip="Effective gravitational coupling in the emergent model"
                    onDragStart={() => stopSimulation()}
                    onChange={(v) => setChiStrength(Number.isFinite(v) ? v : chiStrength)}
                  />
                  <ParameterSlider
                    label="Lattice Size (N)"
                    value={latticeN}
                    min={32}
                    max={96}
                    step={16}
                    unit=""
                    tooltip="Grid resolution for chi field (higher is slower, but more accurate)"
                    onDragStart={() => stopSimulation()}
                    onChange={(v) => setLatticeN(Math.round(v))}
                  />
                  <div className="text-xs text-text-muted">Changing parameters reinitializes the simulation.</div>
                </div>
              </div>

              <StandardMetricsPanel
                coreMetrics={{
                  energy: state.metrics.energy,
                  drift: state.metrics.drift,
                  angularMomentum: state.metrics.angularMomentum,
                }}
                title="System Metrics"
              />
            </div>
          </div>
    </ExperimentLayout>
  );
}

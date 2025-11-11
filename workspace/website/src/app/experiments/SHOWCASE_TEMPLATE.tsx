/*
 * SHOWCASE EXPERIMENT TEMPLATE (STRICT)
 * 
 * This is the MANDATORY template for ALL showcase experiments.
 * Copy this file and ONLY change the marked sections.
 * DO NOT modify structure, imports, or component usage.
 * 
 * Last Updated: 2025-11-11
 */

'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

// ============================================================================
// REQUIRED IMPORTS (DO NOT MODIFY ORDER)
// ============================================================================
import { useEffect, useRef, useCallback, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import StandardVisualizationOptions from '@/components/experiment/StandardVisualizationOptions';
import StandardMetricsPanel from '@/components/experiment/StandardMetricsPanel';
import ParameterSlider from '@/components/ui/ParameterSlider';
import { detectBackend } from '@/physics/core/backend-detector';
import { useSimulationState } from '@/hooks/useSimulationState';

// ----------------------------------------------------------------------------
// REQUIRED METADATA (keep structure; customize values in your experiment page)
// ----------------------------------------------------------------------------
// profile: 'classical' | 'quantum' (affects validator expectations & defaults)
export const experimentMeta = {
  id: 'your-experiment-id',
  tier: 4, // set appropriately (1..7)
  domain: 'quantization', // e.g., 'relativistic' | 'gravity' | 'quantization'
  profile: 'classical' as 'classical' | 'quantum',
  summary: 'Short one-line summary of the phenomenon.',
  relatedTests: [], // e.g., ['QUAN-01']
} as const;

// ============================================================================
// EXPERIMENT-SPECIFIC IMPORTS (CUSTOMIZE HERE)
// ============================================================================
// import { YourSimulationClass } from '@/physics/forces/your-simulation';
// import YourCanvas from '@/components/visuals/YourCanvas';
// import { WebGPUErrorBoundary } from '@/components/ErrorBoundary'; // If using WebGPU

// ============================================================================
// COMPONENT DEFINITION
// ============================================================================
export default function YourExperimentPage() {
  // --------------------------------------------------------------------------
  // STATE MANAGEMENT (REQUIRED - DO NOT MODIFY)
  // --------------------------------------------------------------------------
  const [state, dispatch] = useSimulationState();
  const [backend, setBackend] = useState<'webgpu' | 'cpu'>('webgpu');
  
  // --------------------------------------------------------------------------
  // REFS (REQUIRED - DO NOT MODIFY)
  // --------------------------------------------------------------------------
  const simRef = useRef<any>(null); // Replace 'any' with your simulation type
  const rafRef = useRef<number | null>(null);
  const isRunningRef = useRef<boolean>(false);
  const deviceRef = useRef<GPUDevice | null>(null); // Only if using WebGPU

  useEffect(() => { isRunningRef.current = state.isRunning; }, [state.isRunning]);

  // --------------------------------------------------------------------------
  // SIMULATION CONTROL (REQUIRED - DO NOT MODIFY)
  // --------------------------------------------------------------------------
  const stopSimulation = useCallback(() => {
    dispatch({ type: 'SET_RUNNING', payload: false });
    isRunningRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [dispatch]);

  // --------------------------------------------------------------------------
  // BACKEND DETECTION (REQUIRED - DO NOT MODIFY)
  // --------------------------------------------------------------------------
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

  // --------------------------------------------------------------------------
  // SIMULATION INITIALIZATION (CUSTOMIZE HERE)
  // --------------------------------------------------------------------------
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
            console.warn('[YourExperiment] GPU requested but not available');
            return;
          }
          deviceRef.current = device;
        }
        
        if (cancelled) return;
        
        // Cleanup previous simulation
        try { 
          simRef.current?.destroy(); 
        } catch (e) {
          console.error('[YourExperiment] Error destroying previous simulation:', e);
        }
        
        // TODO: Initialize your simulation here
        // const sim = new YourSimulation(device, config);
        // await sim.initialize();
        // if (cancelled) { sim.destroy(); return; }
        // simRef.current = sim;
        
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
  }, [backend, state.capabilities, state.resetTrigger, stopSimulation]);

  // --------------------------------------------------------------------------
  // ANIMATION LOOP (CUSTOMIZE HERE)
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (!state.isRunning) return;
    
    let rafId: number;
    const tick = async () => {
      const sim = simRef.current;
      if (!sim) return;
      
      // TODO: Step your simulation
      // await sim.stepBatch(10);
      
      // TODO: Update metrics
      // const s = sim.getState();
      // dispatch({
      //   type: 'UPDATE_METRICS',
      //   payload: {
      //     energy: s.energy.toFixed(4) + ' J',
      //     angularMomentum: s.angularMomentum.toFixed(3),
      //     drift: (sim.getEnergyDrift() * 100).toFixed(4) + '%',
      //   },
      // });
      
      if (state.isRunning) {
        rafId = requestAnimationFrame(tick);
      }
    };
    
    rafId = requestAnimationFrame(tick);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [state.isRunning, dispatch]);

  // --------------------------------------------------------------------------
  // RENDER (CUSTOMIZE CONTENT, NOT STRUCTURE)
  // --------------------------------------------------------------------------
  return (
    <ExperimentLayout
      // CUSTOMIZE: Your experiment title
      title="🔬 Your Experiment Title"
      // CUSTOMIZE: Brief description  
      description="Brief description of your experiment"
      backend={backend}
      // CUSTOMIZE: Must match URL path (e.g., 'binary-orbit')
      experimentId="your-experiment-id"
      visualizationOptions={
        <StandardVisualizationOptions
          state={state.ui}
          onChange={(key, value) => dispatch({ type: 'UPDATE_UI', payload: { key: key as any, value } })}
          // REQUIRED: Always true for consistency across all showcase experiments
          showAdvancedOptions={true}
          // OPTIONAL: Only add labelOverrides if you need custom labels
          labelOverrides={{
            // showParticles: 'Custom Label',
            // showTrails: 'Custom Label',
          }}
        />
      }
      footerContent={
        /* OPTIONAL - Explanation section */
        <div className="mt-8 panel">
          <h2 className="text-2xl font-bold text-accent-chi mb-4">What You're Seeing</h2>
          <div className="prose prose-invert max-w-none text-text-secondary space-y-4">
            {/* YOUR EXPLANATION HERE */}
          </div>
        </div>
      }
    >
      {/* ====================================================================== */}
      {/* MAIN CONTENT AREA (REQUIRED STRUCTURE)                                */}
      {/* ====================================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* ---------------------------------------------------------------------- */}
        {/* LEFT SIDE: Canvas (3/4 width) - REQUIRED STRUCTURE                    */}
        {/* ---------------------------------------------------------------------- */}
        <div className="lg:col-span-3">
          <div className="bg-space-panel rounded-lg overflow-hidden border border-space-border h-[600px]">
            {backend === 'webgpu' ? (
              /* TODO: Your canvas component here */
              <div className="h-full flex items-center justify-center">
                <p className="text-text-secondary">Canvas goes here</p>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="text-center p-8">
                  <div className="text-6xl mb-4">🖥️</div>
                  <h3 className="text-2xl font-bold text-purple-400 mb-2">WebGPU Required</h3>
                  <p className="text-text-secondary mb-4">
                    This simulation requires WebGPU for real-time physics computation.
                  </p>
                  <p className="text-sm text-text-muted">
                    Please use Chrome/Edge 113+ or enable experimental WebGPU flags in your browser.
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* REQUIRED: Play/Pause/Reset buttons */}
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
                simRef.current?.reset();
                dispatch({ type: 'RESET_METRICS' });
              }}
              className="px-6 py-3 bg-space-border hover:bg-space-border/80 rounded-lg font-semibold transition-colors"
            >
              🔄 Reset
            </button>
          </div>
        </div>

        {/* ---------------------------------------------------------------------- */}
        {/* RIGHT SIDE: Controls + Metrics (1/4 width) - REQUIRED STRUCTURE       */}
        {/* ---------------------------------------------------------------------- */}
        <div className="lg:col-span-1 space-y-6">
          {/* OPTIONAL: Configuration panel */}
          <div className="panel">
            <h3 className="text-lg font-bold mb-4">Configuration</h3>
            <div className="space-y-3 text-sm text-text-secondary">
              {/* YOUR CONFIGURATION INFO HERE */}
              <div className="flex justify-between">
                <span>Parameter:</span>
                <span className="text-text-primary">Value</span>
              </div>
            </div>
          </div>

          {/* REQUIRED: Metrics panel - ALWAYS same structure */}
          <StandardMetricsPanel
            coreMetrics={{
              energy: state.metrics.energy,
              drift: state.metrics.drift,
              angularMomentum: state.metrics.angularMomentum,
            }}
            // OPTIONAL: Additional metrics only if experiment-specific
            // additionalMetrics={[
            //   { label: 'Custom Metric', value: 'value', status: 'good' },
            // ]}
            // REQUIRED: Use this exact title for consistency
            title="System Metrics"
          />
        </div>
      </div>
    </ExperimentLayout>
  );
}

// ============================================================================
// CHECKLIST BEFORE COMMITTING
// ============================================================================
/*
 * [ ] All required imports present and in correct order
 * [ ] useSimulationState hook used (defaults: showBackground=true, showLattice=false)
 * [ ] Backend detection implemented
 * [ ] StandardVisualizationOptions with showAdvancedOptions={true}
 * [ ] StandardMetricsPanel with core metrics (energy, drift, angularMomentum)
 * [ ] Grid layout: 4 columns (3 canvas, 1 controls)
 * [ ] Canvas container: bg-space-panel, h-[600px], etc.
 * [ ] Play/Pause/Reset buttons below canvas
 * [ ] WebGPU fallback message implemented
 * [ ] Title uses "System Metrics" (not "System Stats" or other)
 * [ ] experimentId matches URL path
 * [ ] Verify defaults: Stars & Background ON, Simulation Grid OFF
 * [ ] Run: npm run validate:experiments (must pass)
 * [ ] Test in browser: visualization options visible at top
 * [ ] Test in browser: all 9 visualization toggles work
 * [ ] Test in browser: core 3 metrics update correctly
 */

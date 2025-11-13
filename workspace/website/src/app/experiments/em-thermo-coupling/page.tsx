'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */

import { useCallback, useEffect, useRef, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import StandardVisualizationOptions from '@/components/experiment/StandardVisualizationOptions';
import ParameterSlider from '@/components/ui/ParameterSlider';
import EMThermoCanvas, { EMThermoParams } from '@/components/visuals/EMThermoCanvas';

export default function EMThermoCouplingPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [params, setParams] = useState<EMThermoParams>({
    amplitude: 0.9,
    frequency: 2.0,
    absorption: 0.6,
    diffusivity: 0.15,
    showEM: true,
    showHeat: true,
  });

  // Basic metrics
  const [metrics, setMetrics] = useState<Record<string, string>>({
    note: 'EM power heats the medium; heat diffuses over time',
  });

  const start = useCallback(() => setIsRunning(true), []);
  const pause = useCallback(() => setIsRunning(false), []);
  const reset = useCallback(() => {
    // Flip a parameter to trigger reset effect inside canvas
    setParams((p) => ({ ...p, absorption: p.absorption + 0.00001 }));
  }, []);

  // Visualization options (merge experiment-only toggles into main group)
  const [viz, setViz] = useState({ showEM: true, showHeat: true, showBackground: false });

  useEffect(() => {
    setParams((p) => ({ ...p, showEM: viz.showEM, showHeat: viz.showHeat }));
  }, [viz.showEM, viz.showHeat]);

  return (
    <ExperimentLayout
      title="⚡ Heat from Light: EM → Thermal Coupling"
      description="A minimal demonstration of electromagnetic energy converting into heat. A Gaussian EM wave deposits energy (∝ E²) into a temperature field that diffuses over time."
      backend="cpu"
      experimentId="em-thermo-coupling"
      visualizationOptions={
        <StandardVisualizationOptions
          state={{ showParticles: false, showTrails: false, showBackground: viz.showBackground }}
          onChange={(key, value) => {
            if (key === 'showBackground') setViz((v) => ({ ...v, showBackground: value }));
          }}
          showAdvancedOptions={false}
          extraToggles={[
            { key: 'showEM', label: 'Show EM Wave', checked: viz.showEM },
            { key: 'showHeat', label: 'Show Heatmap', checked: viz.showHeat },
          ]}
        />
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <div className="panel h-[600px] relative overflow-hidden">
            <EMThermoCanvas isRunning={isRunning} params={params} className="w-full h-[600px]" />
          </div>

          <div className="mt-4 flex items-center justify-center space-x-4" role="group" aria-label="Simulation controls">
            <button onClick={start} disabled={isRunning} className={`px-6 py-3 rounded-lg font-semibold ${isRunning ? 'bg-accent-glow/40 text-space-dark/60' : 'bg-accent-glow hover:bg-accent-glow/80 text-space-dark'}`}>▶ Play</button>
            <button onClick={pause} disabled={!isRunning} className={`px-6 py-3 rounded-lg font-semibold border-2 ${!isRunning ? 'border-purple-500/40 text-purple-400/40' : 'border-purple-500 text-purple-400 hover:bg-purple-500/10'}`}>⏸ Pause</button>
            <button onClick={reset} className="px-6 py-3 rounded-lg font-semibold bg-indigo-500 hover:bg-indigo-400 text-white">💫 Reset Heat</button>
          </div>

          <div className="mt-4 panel">
            <h3 className="text-sm font-bold text-purple-400 mb-3">Coupling Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <ParameterSlider label="Amplitude" value={params.amplitude} min={0.2} max={1.2} step={0.05} unit="" onChange={(v) => setParams((p) => ({ ...p, amplitude: Number(v) }))} />
              <ParameterSlider label="Frequency (cycles/width)" value={params.frequency} min={0.5} max={5} step={0.25} unit="cy/w" onChange={(v) => setParams((p) => ({ ...p, frequency: Number(v) }))} />
              <ParameterSlider label="Absorption (EM→Heat)" value={params.absorption} min={0} max={1} step={0.05} unit="" onChange={(v) => setParams((p) => ({ ...p, absorption: Number(v) }))} />
              <ParameterSlider label="Diffusivity" value={params.diffusivity} min={0} max={0.5} step={0.01} unit="" onChange={(v) => setParams((p) => ({ ...p, diffusivity: Number(v) }))} />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="panel">
            <h3 className="text-lg font-bold text-purple-400 mb-2">Metrics</h3>
            <ul className="text-sm text-text-secondary space-y-1">
              <li><span className="text-text-primary">Experiment: </span>EM → Thermal Coupling</li>
              <li><span className="text-text-primary">Note: </span>{metrics.note}</li>
              <li><span className="text-text-primary">Controls: </span>A, f, absorption, diffusivity</li>
            </ul>
          </div>
          <div className="panel">
            <h3 className="text-lg font-bold text-purple-400 mb-2">How to read it</h3>
            <p className="text-sm text-text-secondary">
              Blue lines are the EM wave. The heatmap accumulates where intensity is high (∝ E²). Increasing absorption raises temperature faster; diffusion spreads heat.
            </p>
          </div>
        </div>
      </div>
    </ExperimentLayout>
  );
}

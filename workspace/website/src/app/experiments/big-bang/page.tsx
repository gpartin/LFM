'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import { useCallback, useState } from 'react';
import ExperimentLayout from '@/components/experiment/ExperimentLayout';
import StandardVisualizationOptions from '@/components/experiment/StandardVisualizationOptions';
import ParameterSlider from '@/components/ui/ParameterSlider';
import BigBangCanvas, { BigBangParams } from '@/components/visuals/BigBangCanvas';

export default function BigBangPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [params, setParams] = useState<BigBangParams>({
    centralEnergy: 10.0,
    explosionSpeed: 1.6,
    concentration: 8.0,
    viscosity: 0.06,
    showEnergy: true,
    showVelocity: false,
  });

  // Visualization options (experiment-only toggles merged into main group)
  const [viz, setViz] = useState({ showEnergy: true, showVelocity: false, showBackground: false });
  // Lattice and boundary controls
  const [gridSize, setGridSize] = useState<number>(128);
  const [edgeAbsorption, setEdgeAbsorption] = useState<number>(0.03);

  const start = useCallback(() => setIsRunning(true), []);
  const pause = useCallback(() => setIsRunning(false), []);
  const reset = useCallback(() => {
    // Trigger reinitialization by slightly tweaking a parameter
    setParams((p) => ({ ...p, centralEnergy: p.centralEnergy + 0.00001 }));
  }, []);

  return (
    <ExperimentLayout
      title="💥 Big Bang: Primordial Energy Explosion"
      description="Watch the universe begin from a concentrated energy singularity. Pure LFM field dynamics show explosive radial expansion—matter and space emerging from a point, no external mechanism required."
      backend="cpu"
      experimentId="big-bang"
      visualizationOptions={
        <StandardVisualizationOptions
          state={{ showParticles: false, showTrails: false, showBackground: viz.showBackground }}
          onChange={(key, value) => {
            if (key === 'showBackground') setViz((v) => ({ ...v, showBackground: value }));
            else if (key === 'showEnergy') setViz((v) => ({ ...v, showEnergy: value }));
            else if (key === 'showVelocity') setViz((v) => ({ ...v, showVelocity: value }));
          }}
          showAdvancedOptions={false}
          extraToggles={[
            { key: 'showEnergy', label: 'Energy Density Heatmap', checked: viz.showEnergy },
            { key: 'showVelocity', label: 'Velocity Field Arrows', checked: viz.showVelocity },
          ]}
        />
      }
      footerContent={
        <div className="mt-8 panel">
          <h3 className="text-xl font-bold text-purple-400 mb-4">The Big Bang: Energy from a Point</h3>
          <div className="prose prose-invert max-w-none text-text-secondary">
            <p className="mb-4">
              This simulation shows <strong>primordial energy explosion</strong>—a concentrated point releases energy that 
              propagates radially outward. The field evolves naturally: energy spreads, diffuses, and conserves. No external 
              expansion mechanism, no spacetime curvature—just lattice field dynamics from a singularity.
            </p>
            <div className="bg-space-dark p-4 rounded-lg font-mono text-purple-400 text-center my-4">
              E(r=0, t=0) = E₀ → Radial expansion → Energy shell propagation → Universe formation
            </div>
            <ul className="space-y-2 list-disc list-inside">
              <li><strong>Central singularity</strong> — Energy concentrated at lattice center (r=0)</li>
              <li><strong>Explosive release</strong> — Radial velocity field drives outward expansion</li>
              <li><strong>Energy conservation</strong> — Total field energy remains approximately constant (with viscosity)</li>
              <li><strong>Shell formation</strong> — Energy propagates as expanding wave front</li>
              <li><strong>Natural evolution</strong> — LFM dynamics govern spread, no ad-hoc inflation</li>
            </ul>
            <p className="mt-4 text-blue-400">
              <strong>🔬 Pure LFM Big Bang:</strong> This is a minimal demonstration of energy explosion from a point. 
              Real cosmology requires quantum fluctuations, matter/radiation separation, dark energy, and baryogenesis. 
              This shows the <em>principle</em>: concentrated field energy can explode and evolve into structure—all 
              from lattice dynamics with no external driver.
            </p>
          </div>
        </div>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <div className="panel h-[600px] relative overflow-hidden">
            <BigBangCanvas
              isRunning={isRunning}
              params={{ ...params, showEnergy: viz.showEnergy, showVelocity: viz.showVelocity }}
              onReset={reset}
              gridSize={gridSize}
              edgeAbsorption={edgeAbsorption}
              className="w-full h-[600px]"
            />
          </div>

          <div className="mt-4 flex items-center justify-center space-x-4" role="group" aria-label="Simulation controls">
            <button onClick={start} disabled={isRunning} className={`px-6 py-3 rounded-lg font-semibold ${isRunning ? 'bg-accent-glow/40 text-space-dark/60' : 'bg-accent-glow hover:bg-accent-glow/80 text-space-dark'}`}>▶ Play</button>
            <button onClick={pause} disabled={!isRunning} className={`px-6 py-3 rounded-lg font-semibold border-2 ${!isRunning ? 'border-purple-500/40 text-purple-400/40' : 'border-purple-500 text-purple-400 hover:bg-purple-500/10'}`}>⏸ Pause</button>
            <button onClick={reset} className="px-6 py-3 rounded-lg font-semibold bg-indigo-500 hover:bg-indigo-400 text-white">💥 Reset Big Bang</button>
          </div>

          <div className="mt-4 panel">
            <h3 className="text-sm font-bold text-purple-400 mb-3">Singularity Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <ParameterSlider label="Central Energy (E₀)" value={params.centralEnergy} min={5} max={20} step={0.5} unit="" onChange={(v) => setParams((p) => ({ ...p, centralEnergy: Number(v) }))} tooltip="Initial energy at the singularity point. Higher = more explosive expansion." />
              <ParameterSlider label="Explosion Speed" value={params.explosionSpeed} min={0.5} max={3} step={0.1} unit="v" onChange={(v) => setParams((p) => ({ ...p, explosionSpeed: Number(v) }))} tooltip="Radial expansion velocity. Higher = faster universe expansion." />
              <ParameterSlider label="Concentration (σ)" value={params.concentration} min={2} max={15} step={0.5} unit="" onChange={(v) => setParams((p) => ({ ...p, concentration: Number(v) }))} tooltip="Initial energy spread. Smaller = tighter singularity = sharper explosion front." />
              <ParameterSlider label="Viscosity" value={params.viscosity} min={0} max={0.5} step={0.05} unit="" onChange={(v) => setParams((p) => ({ ...p, viscosity: Number(v) }))} tooltip="Energy diffusion/dissipation. Higher = smoother expansion, lower = sharper shocks." />
              <ParameterSlider label="Edge Absorption" value={edgeAbsorption} min={0} max={0.1} step={0.005} unit="" onChange={(v) => setEdgeAbsorption(Number(v))} tooltip="Damps energy near boundaries to avoid full-domain saturation. 0 = closed box, 0.02–0.05 recommended." />
              <ParameterSlider label="Lattice Size (cells)" value={gridSize} min={64} max={256} step={64} unit="px" onChange={(v) => setGridSize(Number(v))} tooltip="Grid resolution for the field simulation. Larger grid = more room before reaching boundaries (more compute)." />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="panel">
            <h3 className="text-lg font-bold text-purple-400 mb-2">Metrics</h3>
            <ul className="text-sm text-text-secondary space-y-1">
              <li><span className="text-text-primary">Experiment: </span>Big Bang Expansion</li>
              <li><span className="text-text-primary">Physics: </span>Radial energy explosion from singularity</li>
              <li><span className="text-text-primary">Controls: </span>E₀, speed, concentration, viscosity</li>
              <li><span className="text-text-primary">Note: </span>Energy conserved approximately with viscosity</li>
            </ul>
          </div>
          <div className="panel">
            <h3 className="text-lg font-bold text-purple-400 mb-2">How to read it</h3>
            <p className="text-sm text-text-secondary">
              The heatmap shows energy density: bright white/yellow = high energy, red/orange = medium, black = void. 
              Blue arrows show velocity field direction. Watch the expanding shell propagate outward from the center singularity. 
              Increase Central Energy for a brighter explosion; increase Explosion Speed for faster expansion.
            </p>
          </div>
        </div>
      </div>
    </ExperimentLayout>
  );
}

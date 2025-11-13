/*
 * AutoParameterPanel — auto-generates parameter controls from experiment.initialConditions
 * Enables live parameter adjustment and rerun of simulation/validation
 */

import React from 'react';
import type { ExperimentDefinition } from '@/data/experiments';

interface Props {
  experiment: ExperimentDefinition;
  onParameterChange?: (key: string, value: any) => void;
}

export default function AutoParameterPanel({ experiment, onParameterChange }: Props) {
  const ic = experiment.initialConditions;
  
  // Extract core parameters
  const coreParams = [
    { key: 'latticeSize', label: 'Lattice Size', value: ic.latticeSize, min: 32, max: 512, step: 32, unit: '³' },
    { key: 'dt', label: 'Time Step (dt)', value: ic.dt, min: 0.0001, max: 0.01, step: 0.0001, unit: '' },
    { key: 'dx', label: 'Space Step (dx)', value: ic.dx, min: 0.001, max: 0.1, step: 0.001, unit: '' },
    { key: 'steps', label: 'Steps', value: ic.steps, min: 1000, max: 20000, step: 1000, unit: '' },
    { key: 'chi', label: 'χ Field', value: Array.isArray(ic.chi) ? ic.chi[0] : ic.chi, min: 0, max: 1, step: 0.01, unit: '' },
  ];
  
  // Type-specific parameters
  const typeParams: any[] = [];
  if (ic.wavePacket) {
    typeParams.push(
      { key: 'wavePacket.amplitude', label: 'Amplitude', value: ic.wavePacket.amplitude, min: 0.1, max: 2, step: 0.1, unit: '' },
      { key: 'wavePacket.width', label: 'Width', value: ic.wavePacket.width, min: 0.5, max: 5, step: 0.5, unit: '' },
      { key: 'wavePacket.k[0]', label: 'Wave Vector k', value: ic.wavePacket.k[0], min: 0.01, max: 10, step: 0.01, unit: '' }
    );
  }
  if (ic.particles && ic.particles.length > 0) {
    typeParams.push(
      { key: 'particles[0].mass', label: 'Primary Mass', value: ic.particles[0].mass, min: 1, max: 10000, step: 1, unit: '' }
    );
    if (ic.particles.length > 1) {
      typeParams.push(
        { key: 'particles[1].mass', label: 'Secondary Mass', value: ic.particles[1].mass, min: 0.1, max: 100, step: 0.1, unit: '' }
      );
    }
  }
  
  const allParams = [...coreParams, ...typeParams];
  
  return (
    <div className="space-y-3">
      {allParams.map(param => (
        <div key={param.key}>
          <label className="text-xs text-slate-400 block mb-1">
            {param.label}: <span className="text-slate-200 font-mono">{param.value}{param.unit}</span>
          </label>
          <input
            type="range"
            min={param.min}
            max={param.max}
            step={param.step}
            value={param.value}
            onChange={(e) => onParameterChange?.(param.key, parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
          />
        </div>
      ))}
    </div>
  );
}

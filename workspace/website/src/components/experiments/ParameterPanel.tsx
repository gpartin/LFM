/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { ExperimentDefinition } from '@/data/experiments';

interface ParameterPanelProps {
  experiment: ExperimentDefinition;
  parameters: any;
  onParameterChange: (param: string, value: any) => void;
  disabled?: boolean;
  mode?: 'SHOWCASE' | 'RESEARCH';  // Controls read-only behavior
}

export default function ParameterPanel({
  experiment,
  parameters,
  onParameterChange,
  disabled = false,
  mode = 'SHOWCASE'  // Default to editable
}: ParameterPanelProps) {
  
  // RESEARCH experiments have read-only parameters (locked to test harness config)
  const isReadOnly = mode === 'RESEARCH';
  
  // Extract editable parameters from initial conditions
  const renderParameter = (key: string, value: any) => {
    // Skip complex nested objects for now (handled by custom parameter sections)
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      return null;
    }
    
    // Number parameters
    if (typeof value === 'number') {
      // Determine sensible min/max/step based on key and value
      let min = 0, max = value * 2, step = value / 100;
      
      if (key === 'dt' || key === 'dx') {
        min = value / 10;
        max = value * 10;
        step = value / 10;
      } else if (key === 'steps') {
        min = 10;
        max = 10000;
        step = 10;
      } else if (key === 'latticeSize') {
        min = 16;
        max = 256;
        step = 16;
      } else if (key === 'chi') {
        min = 0;
        max = 1.0;
        step = 0.01;
      }
      
      return (
        <div key={key} className="space-y-2">
          <label className="text-sm text-text-secondary flex items-center justify-between">
            <span className="font-mono">{key}</span>
            <span className="text-white font-mono">{value.toExponential ? value.toExponential(2) : value}</span>
          </label>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onParameterChange(key, parseFloat(e.target.value))}
            disabled={disabled || isReadOnly}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer
                       disabled:opacity-50 disabled:cursor-not-allowed
                       [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 
                       [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full 
                       [&::-webkit-slider-thumb]:bg-accent-energy [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>
      );
    }
    
    return null;
  };
  
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
      <h3 className="text-lg font-bold text-white mb-4">
        {isReadOnly ? 'Test Configuration (Read-Only)' : 'Parameters'}
      </h3>
      
      {isReadOnly && (
        <p className="text-sm text-indigo-300 mb-4 bg-indigo-400/10 border border-indigo-400/30 rounded p-2">
          🔒 Parameters locked to validated test harness configuration
        </p>
      )}
      
      {!isReadOnly && disabled && (
        <p className="text-sm text-yellow-400 mb-4 bg-yellow-400/10 border border-yellow-400/30 rounded p-2">
          ⚠️ Pause simulation to adjust parameters
        </p>
      )}
      
      <div className="space-y-4">
        {Object.entries(parameters).map(([key, value]) => renderParameter(key, value))}
      </div>
      
      {/* Static parameter display */}
      <div className="mt-6 pt-4 border-t border-slate-700">
        <h4 className="text-sm font-semibold text-text-secondary mb-3">Configuration</h4>
        <div className="space-y-1 text-xs font-mono text-text-muted">
          {parameters.latticeSize && (
            <div>Grid: {parameters.latticeSize}³</div>
          )}
          {parameters.dt && (
            <div>dt: {parameters.dt.toExponential(2)}</div>
          )}
          {parameters.dx && (
            <div>dx: {parameters.dx.toExponential(2)}</div>
          )}
          {parameters.steps && (
            <div>Steps: {parameters.steps.toLocaleString()}</div>
          )}
        </div>
      </div>
    </div>
  );
}

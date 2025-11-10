/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

'use client';

import React from 'react';

interface ViewToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function ViewToggle({ label, checked, onChange }: ViewToggleProps) {
  return (
    <label className="flex items-center space-x-3 cursor-pointer group">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-5 h-5 rounded border-2 border-purple-500 bg-space-dark checked:bg-purple-500 checked:border-purple-400 focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-space-darker transition-all duration-200 cursor-pointer"
      />
      <span className="text-sm text-text-secondary group-hover:text-purple-400 transition-colors duration-200">
        {label}
      </span>
    </label>
  );
}

export interface VisualizationToggle {
  key: string;
  label: string;
  checked: boolean;
}

interface VisualizationOptionsProps {
  toggles: VisualizationToggle[];
  onChange: (key: string, checked: boolean) => void;
}

/**
 * Standardized visualization options panel with toggles for all experiments.
 * 
 * Usage:
 * ```tsx
 * <VisualizationOptions
 *   toggles={[
 *     { key: 'showParticles', label: 'Bodies', checked: state.ui.showParticles },
 *     { key: 'showTrails', label: 'Orbital Trails', checked: state.ui.showTrails },
 *     { key: 'showChi', label: 'Chi Field', checked: state.ui.showChi },
 *     { key: 'showBackground', label: 'Stars & Background', checked: state.ui.showBackground },
 *   ]}
 *   onChange={(key, checked) => dispatch({ type: 'UPDATE_UI', payload: { key, value: checked } })}
 * />
 * ```
 */
export default function VisualizationOptions({ toggles, onChange }: VisualizationOptionsProps) {
  return (
    <div className="panel">
      <h3 className="text-lg font-bold text-purple-400 mb-4">Visualization Options</h3>
      <div className="space-y-3">
        {toggles.map((toggle) => (
          <ViewToggle
            key={toggle.key}
            label={toggle.label}
            checked={toggle.checked}
            onChange={(checked) => onChange(toggle.key, checked)}
          />
        ))}
      </div>
    </div>
  );
}

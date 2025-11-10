/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

interface ParameterSliderProps {
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
}

/**
 * Reusable parameter slider component with tooltip support.
 * Used across all experiment pages for consistent UI/UX.
 */
export default function ParameterSlider({
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
}: ParameterSliderProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <label className="text-sm font-semibold text-text-primary">{label}</label>
          {tooltip && (
            <span 
              className="text-blue-400 hover:text-blue-300 cursor-help transition-colors" 
              title={tooltip}
              aria-label={`Info: ${tooltip}`}
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </span>
          )}
        </div>
        <span className="text-sm font-mono text-accent-chi">
          {value < 0.1 ? value.toFixed(4) : value < 1 ? value.toFixed(2) : value.toFixed(1)} {unit}
        </span>
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

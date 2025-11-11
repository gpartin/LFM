/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */

'use client';

import React from 'react';
import VisualizationOptions from '@/components/ui/VisualizationOptions';

interface QuantumVisualizationState {
  showWave?: boolean;
  showBarrier?: boolean;
  showPhase?: boolean;
  showEnergyDensity?: boolean;
  showTransmissionPlot?: boolean;
  showBackground?: boolean;
}

interface Props {
  state: QuantumVisualizationState;
  onChange: (key: keyof QuantumVisualizationState, value: boolean) => void;
  showAdvancedOptions?: boolean;
}

/**
 * QuantumVisualizationOptions - Minimal, meaningful toggles for quantum experiments
 *
 * Chosen to avoid classical-only concepts (particles, trails, gravity well, etc.).
 */
export default function QuantumVisualizationOptions({ state, onChange, showAdvancedOptions = true }: Props) {
  const toggles = [
    { key: 'showWave', label: 'Quantum Wave', checked: state.showWave ?? true },
    { key: 'showBarrier', label: 'Barrier (χ Field)', checked: state.showBarrier ?? true },
    { key: 'showTransmissionPlot', label: 'T/R Overlay', checked: state.showTransmissionPlot ?? true },
    // Phase / Energy Density forthcoming; hidden until implemented to avoid dead toggles
  ];

  return (
    <div className="flex items-center gap-6 flex-wrap">
      <VisualizationOptions
        toggles={toggles}
        onChange={(key, value) => onChange(key as keyof QuantumVisualizationState, value)}
      />
    </div>
  );
}

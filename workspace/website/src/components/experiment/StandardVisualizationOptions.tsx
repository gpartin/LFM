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

/**
 * StandardVisualizationOptions - Unified visualization controls for all experiments
 * 
 * Provides consistent UI controls across showcase experiments while allowing
 * experiment-specific label customization.
 */

export interface VisualizationState {
  showParticles: boolean;
  showTrails: boolean;
  showChi?: boolean;
  showLattice?: boolean;
  showVectors?: boolean;
  showWell?: boolean;
  showDomes?: boolean;
  showIsoShells?: boolean;
  showBackground: boolean;
}

export interface LabelOverrides {
  showParticles?: string;
  showTrails?: string;
  showChi?: string;
  showLattice?: string;
  showVectors?: string;
  showWell?: string;
  showDomes?: string;
  showIsoShells?: string;
  showBackground?: string;
}

interface Props {
  state: VisualizationState;
  onChange: (key: string, value: boolean) => void;
  /** Experiment-specific label overrides (e.g., "Bodies" vs "Earth & Moon") */
  labelOverrides?: LabelOverrides;
  /** Show advanced visualization options (chi field, lattice, etc.) */
  showAdvancedOptions?: boolean;
  /** Additional custom controls to append */
  additionalControls?: React.ReactNode;
}

const defaultLabels: Required<LabelOverrides> = {
  showParticles: 'Bodies',
  showTrails: 'Orbital Trails',
  showChi: 'Chi Field (2D)',
  showLattice: 'Simulation Grid',
  showVectors: 'Force Arrows',
  showWell: 'Gravity Well (Surface)',
  showDomes: 'Field Bubbles (3D)',
  showIsoShells: 'Field Shells',
  showBackground: 'Stars & Background',
};

export default function StandardVisualizationOptions({
  state,
  onChange,
  labelOverrides = {},
  showAdvancedOptions = true,
  additionalControls,
}: Props) {
  const labels = { ...defaultLabels, ...labelOverrides };

  const basicToggles = [
    { key: 'showParticles', label: labels.showParticles, checked: state.showParticles },
    { key: 'showTrails', label: labels.showTrails, checked: state.showTrails },
    { key: 'showBackground', label: labels.showBackground, checked: state.showBackground },
  ];

  const advancedToggles = showAdvancedOptions ? [
    { key: 'showChi', label: labels.showChi, checked: state.showChi ?? false },
    { key: 'showLattice', label: labels.showLattice, checked: state.showLattice ?? false },
    { key: 'showVectors', label: labels.showVectors, checked: state.showVectors ?? false },
    { key: 'showWell', label: labels.showWell, checked: state.showWell ?? false },
    { key: 'showDomes', label: labels.showDomes, checked: state.showDomes ?? false },
    { key: 'showIsoShells', label: labels.showIsoShells, checked: state.showIsoShells ?? false },
  ] : [];

  const allToggles = [...basicToggles, ...advancedToggles];

  return (
    <div className="flex items-center gap-6 flex-wrap">
      <VisualizationOptions toggles={allToggles} onChange={onChange} />
      {additionalControls}
    </div>
  );
}

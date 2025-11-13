/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import React from 'react';
import type { SimulationUI } from '@/hooks/useSimulationState';
import NBodyCanvas from '@/components/visuals/NBodyCanvas';
import OrbitCanvas from '@/components/visuals/OrbitCanvas';
import FieldSliceCanvas from '@/components/visuals/FieldSliceCanvas';

export type UniversalCanvasKind = 'nbody' | 'orbit' | 'quantum' | 'simple';

interface UniversalCanvasProps {
  kind: UniversalCanvasKind;
  simulation: React.MutableRefObject<any>;
  isRunning: boolean;
  ui: SimulationUI;
  /** Optional chi strength for orbit visuals requiring well/field scaling */
  chiStrength?: number;
  /** Optional: show black-hole analogue reference rings (RS, 1.5RS, 3RS) */
  showBHRings?: boolean;
  /** Optional: sigma parameter for chi field concentration (needed for BH horizon calculation) */
  sigma?: number;
  /** Optional: enable tidal stretch visuals for orbit canvas */
  tidalStretch?: boolean;
  /** Optional: effective self-gravity/cohesion factor used in tidal metric */
  selfGravityFactor?: number;
  /** Optional: primary particle color (for stellar evolution visuals) */
  primaryColor?: string;
  /** Optional: primary particle scale multiplier (for stellar evolution size changes) */
  primaryScale?: number;
  /** Optional: enable light-ray visualization (for lensing) */
  showLightRays?: boolean;
  /** Optional: config for light rays */
  rayConfig?: {
    count?: number;           // rows if grid unspecified
    color?: string;
    speed?: number;
    spread?: number;          // vertical half-span factor (0-2 range typical)
    rows?: number;            // rows in emitter grid
    cols?: number;            // columns in emitter grid
    emitterOffset?: number;   // fraction of width from left (default ~0.06)
    emitterWidth?: number;    // fraction of width for emitter grid span (default ~0.10)
    headSize?: number;        // ray head sphere size
    pure?: boolean;           // pure physics mode: dv = -k∇χ, no guidance/clamps
    debug?: boolean;          // show debugging overlays (vectors, emitter bounds)
  };
  /** Lensing-only: show background source-plane sampler */
  showLensingBackground?: boolean;
  /** Lensing-only: toggle light debug overlay */
  lightDebug?: boolean;
  /** Optional: hide secondary particle mesh (useful for lensing) */
  hideSecondaryParticle?: boolean;
  /** Optional quantum metrics strings for overlay (T/R/T+R) */
  transmissionValue?: string;
  reflectionValue?: string;
  conservationValue?: string;
}

export default function UniversalCanvas({
  kind,
  simulation,
  isRunning,
  ui,
  chiStrength = 0.25,
  showBHRings = false,
  sigma = 1.0,
  tidalStretch = false,
  selfGravityFactor = 1.0,
  primaryColor = '#4A90E2',
  primaryScale = 1.0,
  showLightRays = false,
  rayConfig,
  showLensingBackground,
  lightDebug,
  hideSecondaryParticle = false,
  transmissionValue,
  reflectionValue,
  conservationValue,
}: UniversalCanvasProps) {
  switch (kind) {
    case 'nbody':
      return (
        <NBodyCanvas
          simulation={simulation}
          isRunning={isRunning}
          showParticles={ui.showParticles}
          showTrails={ui.showTrails}
          showBackground={ui.showBackground}
          showChi={ui.showChi}
          showLattice={ui.showLattice}
          showVectors={ui.showVectors}
          showWell={ui.showWell}
          showDomes={ui.showDomes}
          showIsoShells={ui.showIsoShells}
        />
      );
    case 'orbit':
      return (
        <OrbitCanvas
          simulation={simulation}
          isRunning={isRunning}
          showParticles={ui.showParticles}
          showTrails={ui.showTrails}
          showChi={ui.showChi}
          showLattice={ui.showLattice}
          showVectors={ui.showVectors}
          showWell={ui.showWell}
          showDomes={ui.showDomes}
          showIsoShells={ui.showIsoShells}
          showBackground={ui.showBackground}
          chiStrength={chiStrength}
          showBHRings={showBHRings}
          sigma={sigma}
          tidalStretch={tidalStretch}
          selfGravityFactor={selfGravityFactor}
          primaryColor={primaryColor}
          primaryScale={primaryScale}
          showLightRays={showLightRays}
          rayConfig={{ ...rayConfig, debug: lightDebug ?? rayConfig?.debug }}
          showLensingBackground={showLensingBackground}
          hideSecondaryParticle={hideSecondaryParticle}
        />
      );
    case 'quantum':
      return (
        <div className="w-full h-full">
          <FieldSliceCanvas
            simulation={simulation}
            isRunning={isRunning}
            showGrid={ui.showLattice ?? true}
            showWave={ui.showWave ?? true}
            showBarrier={ui.showBarrier ?? true}
            showTransmissionOverlay={ui.showTransmissionPlot ?? false}
            transmissionValue={transmissionValue}
            reflectionValue={reflectionValue}
            conservationValue={conservationValue}
            updateInterval={3}
          />
        </div>
      );
    case 'simple':
    default:
      return (
        <div className="w-full h-[600px] bg-space-dark border border-space-border rounded-lg flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4">🧩</div>
            <h3 className="text-2xl font-bold text-accent-chi mb-2">Visualization Adapter Missing</h3>
            <p className="text-text-secondary">This experiment hasn't registered a visualization yet.</p>
          </div>
        </div>
      );
  }
}

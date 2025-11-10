'use client';

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import { PhysicsBackend } from '@/physics/core/backend-detector';

interface BackendBadgeProps {
  backend: PhysicsBackend;
  fps?: number;
  energyDrift?: number;
}

export default function BackendBadge({ backend, fps, energyDrift }: BackendBadgeProps) {
  const isOptimal = backend === 'webgpu';
  const isRealLFM = true; // ALL backends now run authentic LFM

  return (
    <div className="flex flex-col space-y-2">
      {/* Main status badge */}
      <div
        className={`
          px-4 py-3 rounded-lg border-2 flex items-center justify-between
          ${isOptimal 
            ? 'bg-accent-glow/10 border-accent-glow' 
            : 'bg-amber-500/10 border-amber-500'
          }
        `}
      >
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${isOptimal ? 'bg-accent-glow' : 'bg-amber-500'} animate-pulse-glow`} />
          <div>
            <div className={`font-semibold ${isOptimal ? 'text-accent-glow' : 'text-amber-400'}`}>
              {getBackendLabel(backend)}
            </div>
            <div className="text-xs text-text-muted">{getBackendDescription(backend)}</div>
          </div>
        </div>
        {fps !== undefined && (
          <div className="text-right">
            <div className="text-sm font-mono text-text-primary">{fps.toFixed(0)} FPS</div>
            {energyDrift !== undefined && (
              <div className="text-xs text-text-secondary">Drift: {(energyDrift * 100).toFixed(4)}%</div>
            )}
          </div>
        )}
      </div>

      {/* Info for non-optimal backends */}
      {!isOptimal && (
        <div className="bg-amber-500/10 border-2 border-amber-500/50 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <span className="text-2xl">ℹ️</span>
            <div>
              <h4 className="font-bold text-amber-400 mb-1">Running Authentic LFM at Lower Resolution</h4>
              <p className="text-sm text-text-secondary leading-relaxed">
                You're running the <strong className="text-amber-400">same Klein-Gordon equation</strong> as GPU mode, just on a {backend === 'cpu' ? '32³' : '32³'} lattice instead of 64³.
                Physics is authentic - only resolution and speed are reduced.
              </p>
              <p className="text-xs text-text-muted mt-2">
                For higher resolution and 60fps, use Chrome 113+ or Edge 113+ with a compatible GPU.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Info for optimal backend */}
      {isOptimal && (
        <div className="bg-accent-glow/5 border border-accent-glow/30 rounded-lg p-3">
          <div className="flex items-start space-x-2">
            <span className="text-lg">✓</span>
            <div className="text-xs text-text-secondary leading-relaxed">
              Running authentic Klein-Gordon lattice simulation on GPU. Gravity emerges from chi field gradients.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getBackendLabel(backend: PhysicsBackend): string {
  switch (backend) {
    case 'webgpu':
      return 'GPU (WebGPU) - Optimal';
    case 'webgl':
      return 'GPU (WebGL2) - Authentic LFM';
    case 'cpu':
      return 'CPU (JavaScript) - Authentic LFM';
  }
}

function getBackendDescription(backend: PhysicsBackend): string {
  switch (backend) {
    case 'webgpu':
      return '64³ lattice, Klein-Gordon equation @ 60fps';
    case 'webgl':
      return '32³ lattice, Klein-Gordon equation @ 30fps';
    case 'cpu':
      return '32³ lattice, Klein-Gordon equation @ 15fps';
  }
}

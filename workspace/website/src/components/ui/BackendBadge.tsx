'use client';

import { PhysicsBackend } from '@/physics/core/backend-detector';

interface BackendBadgeProps {
  backend: PhysicsBackend;
  fps?: number;
  energyDrift?: number;
}

export default function BackendBadge({ backend, fps, energyDrift }: BackendBadgeProps) {
  const isRealLFM = backend === 'webgpu';

  return (
    <div className="flex flex-col space-y-2">
      {/* Main status badge */}
      <div
        className={`
          px-4 py-3 rounded-lg border-2 flex items-center justify-between
          ${isRealLFM 
            ? 'bg-accent-glow/10 border-accent-glow' 
            : 'bg-yellow-500/10 border-yellow-500'
          }
        `}
      >
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${isRealLFM ? 'bg-accent-glow' : 'bg-yellow-500'} animate-pulse-glow`} />
          <div>
            <div className={`font-semibold ${isRealLFM ? 'text-accent-glow' : 'text-yellow-500'}`}>
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

      {/* Warning for non-LFM backends */}
      {!isRealLFM && (
        <div className="bg-yellow-500/10 border-2 border-yellow-500 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <span className="text-2xl">⚠️</span>
            <div>
              <h4 className="font-bold text-yellow-500 mb-1">Not Running Authentic LFM</h4>
              <p className="text-sm text-text-secondary leading-relaxed">
                Your browser doesn't support WebGPU, so you're seeing a simplified {backend === 'webgl' ? 'approximation' : 'Newtonian simulation'}.
                This is <strong className="text-yellow-500">not the real physics</strong>.
              </p>
              <p className="text-xs text-text-muted mt-2">
                For the authentic experience, use Chrome 113+ or Edge 113+ with a compatible GPU.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Info for real LFM */}
      {isRealLFM && (
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
      return 'WebGPU - Authentic LFM';
    case 'webgl':
      return 'WebGL2 - Approximate Physics';
    case 'cpu':
      return 'CPU - Simplified Newtonian';
  }
}

function getBackendDescription(backend: PhysicsBackend): string {
  switch (backend) {
    case 'webgpu':
      return '64³ lattice, real Klein-Gordon equation';
    case 'webgl':
      return '32³ lattice approximation (not real LFM)';
    case 'cpu':
      return 'Newtonian approximation (not real LFM)';
  }
}

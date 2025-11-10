/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * Backend Information Banner
 * 
 * Displays current physics backend and provides upgrade recommendations
 */
'use client';

import React from 'react';
import type { PhysicsBackend } from '@/physics/core/backend-detector';
import { getBackendDescription, getBackendRecommendations } from '@/physics/core/backend-detector';

interface BackendBannerProps {
  backend: PhysicsBackend;
  latticeSize: number;
  onDismiss?: () => void;
}

export default function BackendBanner({ backend, latticeSize, onDismiss }: BackendBannerProps) {
  const [dismissed, setDismissed] = React.useState(false);
  
  // Don't show banner for optimal backend (WebGPU)
  if (backend === 'webgpu' || dismissed) {
    return null;
  }
  
  const handleDismiss = () => {
    setDismissed(true);
    onDismiss?.();
  };
  
  const recommendations = getBackendRecommendations(backend);
  const description = getBackendDescription(backend);
  
  return (
    <div className="bg-amber-900/20 border border-amber-500/50 rounded-lg p-4 mb-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl">{backend === 'cpu' ? '🐌' : '⚡'}</span>
            <h3 className="font-semibold text-amber-200">
              {description}
            </h3>
          </div>
          
          <p className="text-sm text-amber-100 mb-3">
            You're running <strong>authentic Klein-Gordon physics</strong> on a <strong>{latticeSize}³ lattice</strong>.
            The equation is identical to the GPU version, just at lower resolution/speed.
          </p>
          
          {recommendations.length > 0 && (
            <ul className="text-sm text-amber-100/90 space-y-1 mb-3">
              {recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          )}
          
          <div className="text-xs text-amber-200/70">
            <strong>Physics integrity maintained:</strong> Energy conservation, relativistic correctness, and emergent gravity all work identically. Only resolution and frame rate are affected.
          </div>
        </div>
        
        <button
          onClick={handleDismiss}
          className="text-amber-200 hover:text-amber-100 transition-colors p-1"
          aria-label="Dismiss"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}

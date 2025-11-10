/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/**
 * Backend Selector Component
 * 
 * Allows GPU users to manually switch to CPU mode (for comparison/testing)
 * CPU users cannot switch up to GPU (hardware limitation)
 */
'use client';

import React from 'react';
import type { PhysicsBackend } from '@/physics/core/backend-detector';

interface BackendSelectorProps {
  currentBackend: PhysicsBackend;
  availableBackends: PhysicsBackend[];
  onBackendChange: (backend: PhysicsBackend) => void;
  disabled?: boolean;
}

export default function BackendSelector({
  currentBackend,
  availableBackends,
  onBackendChange,
  disabled = false,
}: BackendSelectorProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  
  const getBackendLabel = (backend: PhysicsBackend): string => {
    switch (backend) {
      case 'webgpu': return 'GPU (WebGPU)';
      case 'webgl': return 'GPU (WebGL2)';
      case 'cpu': return 'CPU (JavaScript)';
    }
  };
  
  const getBackendIcon = (backend: PhysicsBackend): string => {
    switch (backend) {
      case 'webgpu': return '🚀';
      case 'webgl': return '⚡';
      case 'cpu': return '🐌';
    }
  };
  
  const getBackendPerformance = (backend: PhysicsBackend): string => {
    switch (backend) {
      case 'webgpu': return '64³ @ 60fps';
      case 'webgl': return '64³ @ 30fps';
      case 'cpu': return '32³ @ 15fps';
    }
  };
  
  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className="flex items-center gap-2 px-3 py-2 bg-space-panel border border-space-border rounded-lg hover:border-accent-chi/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        aria-label="Select physics backend"
      >
        <span className="text-lg">{getBackendIcon(currentBackend)}</span>
        <div className="text-left">
          <div className="text-sm font-medium text-text-primary">
            {getBackendLabel(currentBackend)}
          </div>
          <div className="text-xs text-text-muted">
            {getBackendPerformance(currentBackend)}
          </div>
        </div>
        <svg 
          className={`w-4 h-4 text-text-secondary transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setIsOpen(false)}
          />
          
          {/* Dropdown */}
          <div className="absolute top-full mt-2 left-0 z-20 min-w-[280px] bg-space-panel border border-space-border rounded-lg shadow-xl overflow-hidden">
            <div className="p-2 border-b border-space-border bg-space-dark/50">
              <p className="text-xs text-text-muted">
                <strong className="text-accent-chi">All backends run authentic Klein-Gordon physics.</strong>
                {' '}Only resolution and speed differ.
              </p>
            </div>
            
            <div className="py-1">
              {availableBackends.map((backend) => {
                const isAvailable = availableBackends.includes(backend);
                const isCurrent = backend === currentBackend;
                const canSwitch = isAvailable && !isCurrent;
                
                // Prevent switching TO a better backend (hardware limitation)
                const isUpgrade = 
                  (currentBackend === 'cpu' && backend !== 'cpu') ||
                  (currentBackend === 'webgl' && backend === 'webgpu');
                
                const isDisabled = !isAvailable || isCurrent || isUpgrade;
                
                return (
                  <button
                    key={backend}
                    onClick={() => {
                      if (canSwitch && !isUpgrade) {
                        onBackendChange(backend);
                        setIsOpen(false);
                      }
                    }}
                    disabled={isDisabled}
                    className={`
                      w-full px-4 py-3 text-left flex items-center gap-3
                      transition-colors
                      ${isCurrent ? 'bg-accent-chi/10' : ''}
                      ${canSwitch && !isUpgrade ? 'hover:bg-space-border/50 cursor-pointer' : ''}
                      ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}
                    `}
                  >
                    <span className="text-2xl">{getBackendIcon(backend)}</span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-text-primary">
                          {getBackendLabel(backend)}
                        </span>
                        {isCurrent && (
                          <span className="text-xs px-2 py-0.5 bg-accent-chi/20 text-accent-chi rounded">
                            Active
                          </span>
                        )}
                        {isUpgrade && !isCurrent && (
                          <span className="text-xs px-2 py-0.5 bg-red-500/20 text-red-400 rounded">
                            Hardware Required
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-text-muted mt-0.5">
                        {getBackendPerformance(backend)}
                      </div>
                      {isUpgrade && !isCurrent && (
                        <div className="text-xs text-red-400 mt-1">
                          Your device doesn't support this backend
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
            
            <div className="p-2 border-t border-space-border bg-space-dark/50">
              <p className="text-xs text-text-muted">
                💡 <strong>Tip:</strong> GPU users can switch down to CPU to see performance difference. Physics accuracy is identical.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

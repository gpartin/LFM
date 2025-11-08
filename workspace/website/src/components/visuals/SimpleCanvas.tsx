/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Simple Experiment Canvas - MVP visualization
 * 
 * Placeholder for experiments that don't yet have full visualization
 */
'use client';

import React from 'react';

interface SimpleCanvasProps {
  isRunning: boolean;
  parameters: Record<string, any>;
  views: Record<string, boolean>;
}

export default function SimpleCanvas({ isRunning, parameters, views }: SimpleCanvasProps) {
  return (
    <div className="w-full h-[600px] bg-space-dark border border-space-border rounded-lg flex items-center justify-center">
      <div className="text-center">
        <div className="text-6xl mb-4">🚧</div>
        <h3 className="text-2xl font-bold text-accent-chi mb-2">Visualization Coming Soon</h3>
        <p className="text-text-secondary mb-4">Physics simulation is running in the background</p>
        <div className="text-sm text-text-muted">
          {isRunning ? (
            <span className="text-green-400">● Simulation Running</span>
          ) : (
            <span className="text-gray-400">○ Simulation Paused</span>
          )}
        </div>
      </div>
    </div>
  );
}

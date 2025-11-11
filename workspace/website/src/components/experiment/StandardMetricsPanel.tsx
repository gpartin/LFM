/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */

'use client';

import React from 'react';

/**
 * StandardMetricsPanel - Unified metrics display for all experiments
 * 
 * Shows core physics metrics (energy, drift, angular momentum) with optional
 * experiment-specific additional metrics.
 */

export interface MetricConfig {
  label: string;
  value: string;
  status: 'conserved' | 'good' | 'neutral' | 'warning';
}

export interface CoreMetricsState {
  energy: string;
  drift: string;
  angularMomentum: string;
}

interface Props {
  /** Core physics metrics (always displayed) */
  coreMetrics: CoreMetricsState;
  /** Experiment-specific additional metrics */
  additionalMetrics?: MetricConfig[];
  /** Panel title (defaults to "System Metrics") */
  title?: string;
  /** Title color class (defaults to text-accent-chi) */
  titleColorClass?: string;
}

function MetricDisplay({ 
  label, 
  value, 
  status 
}: { 
  label: string; 
  value: string; 
  status: 'conserved' | 'good' | 'neutral' | 'warning';
}) {
  const statusColors = {
    conserved: 'text-accent-glow',
    good: 'text-accent-glow',
    neutral: 'text-text-primary',
    warning: 'text-yellow-500',
  };

  return (
    <div className="flex items-center justify-between py-2 border-b border-space-border last:border-b-0">
      <span className="text-sm text-text-secondary">{label}</span>
      <span className={`text-sm font-mono font-semibold ${statusColors[status]}`}>{value}</span>
    </div>
  );
}

export default function StandardMetricsPanel({
  coreMetrics,
  additionalMetrics = [],
  title = 'System Metrics',
  titleColorClass = 'text-accent-chi',
}: Props) {
  // Determine status dynamically based on value
  const getEnergyStatus = (value: string): 'conserved' | 'neutral' => 
    value === '—' ? 'neutral' : 'conserved';
  
  const getDriftStatus = (value: string): 'good' | 'neutral' => 
    value === '—' ? 'neutral' : 'good';

  return (
    <div className="panel">
      <h3 className={`text-lg font-bold ${titleColorClass} mb-4`}>{title}</h3>
      
      <div className="space-y-3">
        {/* Core metrics (always shown) */}
        <MetricDisplay 
          label="Total Energy" 
          value={coreMetrics.energy} 
          status={getEnergyStatus(coreMetrics.energy)} 
        />
        <MetricDisplay 
          label="Energy Drift" 
          value={coreMetrics.drift} 
          status={getDriftStatus(coreMetrics.drift)} 
        />
        <MetricDisplay 
          label="Angular Momentum" 
          value={coreMetrics.angularMomentum} 
          status={getEnergyStatus(coreMetrics.angularMomentum)} 
        />
        
        {/* Experiment-specific additional metrics */}
        {additionalMetrics.map((metric, idx) => (
          <MetricDisplay 
            key={idx}
            label={metric.label}
            value={metric.value}
            status={metric.status}
          />
        ))}
      </div>
    </div>
  );
}

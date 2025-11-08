/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import type { Experiment } from '@/types/experiment';

export default function StellarCollapsePage() {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function init() {
      try {
        const adapter = await navigator.gpu?.requestAdapter();
        const gpuDevice = await adapter?.requestDevice();
        
        if (!gpuDevice) {
          setError('WebGPU not supported');
          return;
        }
        
        const module = await import('@/experiments/gravity/stellar-collapse');
        const exp = module.default(gpuDevice);
        await exp.initialize();
        setExperiment(exp);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize');
      }
    }
    
    init();
    
    return () => {
      if (experiment) {
        experiment.cleanup();
      }
    };
  }, []);

  const handleStart = () => {
    if (experiment) {
      experiment.start();
      setIsRunning(true);
    }
  };

  const handlePause = () => {
    if (experiment) {
      experiment.pause();
      setIsRunning(false);
    }
  };

  const handleReset = async () => {
    if (experiment) {
      await experiment.reset();
      setIsRunning(false);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen bg-space-dark text-text-primary">
        <Header />
        <main className="container mx-auto px-4 pt-24">
          <div className="bg-red-900/20 border border-red-500 rounded-lg p-8 text-center">
            <h2 className="text-2xl font-bold text-red-400 mb-4">Error</h2>
            <p>{error}</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="min-h-screen bg-space-dark text-text-primary">
        <Header />
        <main className="container mx-auto px-4 pt-24">
          <div className="text-center py-20">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-accent-chi mx-auto"></div>
            <p className="mt-4 text-text-secondary">Loading experiment...</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  const metrics = experiment.getMetrics();
  const RenderComponent = experiment.RenderComponent;

  return (
    <div className="min-h-screen bg-space-dark text-text-primary">
      <Header />
      <main className="container mx-auto px-4 pt-24 pb-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-accent-chi mb-2">{experiment.metadata.title}</h1>
          <p className="text-text-secondary text-lg">{experiment.metadata.fullDescription}</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3">
            <div className="bg-space-panel rounded-lg p-4 border border-space-border">
              <RenderComponent isRunning={isRunning} parameters={{}} views={{}} />
            </div>
            
            <div className="mt-4 flex gap-4">
              {!isRunning ? (
                <button onClick={handleStart} className="px-6 py-3 bg-accent-chi hover:bg-accent-chi/80 rounded-lg font-semibold transition-colors">
                  💫 Start Collapse
                </button>
              ) : (
                <button onClick={handlePause} className="px-6 py-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-semibold transition-colors">
                  Pause
                </button>
              )}
              <button onClick={handleReset} className="px-6 py-3 bg-space-border hover:bg-space-border/80 rounded-lg font-semibold transition-colors">
                Reset
              </button>
            </div>
          </div>

          <div className="lg:col-span-1">
            <div className="bg-space-panel rounded-lg p-4 border border-space-border">
              <h3 className="text-lg font-semibold text-accent-chi mb-4">Metrics</h3>
              <div className="space-y-3">
                {metrics.map((metric, idx) => (
                  <div key={idx}>
                    <div className="text-sm text-text-muted">{metric.label}</div>
                    <div className="text-xl font-mono">{metric.value} {metric.unit}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

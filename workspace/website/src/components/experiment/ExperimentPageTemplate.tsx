/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * ExperimentPageTemplate
 * 
 * Reusable template for experiment pages. Eliminates ~400 lines of boilerplate per experiment.
 * Handles device initialization, simulation lifecycle, standard controls, metrics display.
 */

'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import Link from 'next/link';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import BackendBadge from '@/components/ui/BackendBadge';
import KeyboardShortcuts from '@/components/ui/KeyboardShortcuts';
import { detectBackend, type BackendCapabilities, type PhysicsBackend } from '@/physics/core/backend-detector';
import { getExperimentById } from '@/lib/experimentRegistry';
import type { Experiment, ExperimentMetrics as ExperimentMetricsType } from '@/types/experiment';
import { WebGPUErrorBoundary } from '@/components/ErrorBoundary';
import { ExperimentSkeleton } from '@/components/ui/LoadingSkeleton';

export interface ExperimentPageTemplateProps {
  /** Experiment ID from registry */
  experimentId: string;
  /** Optional custom content to show below explanation panel */
  customContent?: React.ReactNode;
}

export default function ExperimentPageTemplate({ 
  experimentId,
  customContent,
}: ExperimentPageTemplateProps) {
  const [backend, setBackend] = useState<PhysicsBackend>('cpu');
  const [capabilities, setCapabilities] = useState<BackendCapabilities | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [metrics, setMetrics] = useState<ExperimentMetricsType[]>([]);
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [views, setViews] = useState<Record<string, boolean>>({});
  
  const deviceRef = useRef<GPUDevice | null>(null);
  const rafRef = useRef<number | null>(null);
  const isRunningRef = useRef<boolean>(false);
  
  // Get experiment metadata from registry
  const registryEntry = getExperimentById(experimentId);
  
  if (!registryEntry) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-space-dark">
        <div className="text-center">
          <div className="text-6xl mb-4">❌</div>
          <h3 className="text-2xl font-bold text-accent-chi mb-2">Experiment Not Found</h3>
          <p className="text-text-secondary mb-6">
            Could not find experiment with ID: {experimentId}
          </p>
          <Link href="/experiments/browse" className="btn-primary">
            Browse Experiments
          </Link>
        </div>
      </div>
    );
  }
  
  const metadata = registryEntry.metadata;
  
  // Detect backend on mount
  useEffect(() => {
    detectBackend().then((caps) => {
      setBackend(caps.backend);
      setCapabilities(caps);
    });
  }, []);
  
  // Initialize experiment when backend is ready
    if (!registryEntry) {
      setError(`Experiment '${experimentId}' not found in registry`);
      setIsLoading(false);
      return;
    }
    
  useEffect(() => {
    if (backend !== 'webgpu') {
      setIsLoading(false);
      return;
    }
    
    let cancelled = false;
    
    async function initExperiment() {
      try {
        setError(null);
        
        // Request WebGPU device
        const adapter = await navigator.gpu?.requestAdapter();
        const device = await adapter?.requestDevice();
        
        if (!device || cancelled) return;
        
        deviceRef.current = device;
        
        // Load experiment module
  const module = await registryEntry!.loader();
        const factory = module.default;
        
        // Create experiment instance
        const exp = factory(device);
        await exp.initialize();
        
        if (cancelled) {
          await exp.cleanup();
          return;
        }
        
        setExperiment(exp);
        
        // Set initial parameters and views
        const initialParams: Record<string, any> = {};
        exp.config.parameters.forEach(param => {
          initialParams[param.key] = param.defaultValue;
        });
        setParameters(initialParams);
        const defaultViews: Record<string, boolean> = {};
        if (exp.config.defaultViews) {
          Object.entries(exp.config.defaultViews).forEach(([key, val]) => {
            if (val !== undefined) defaultViews[key] = val;
          });
        }
        setViews(defaultViews);
        
        // Get initial metrics
        setMetrics(exp.getMetrics());
        setIsLoading(false);
        
      } catch (err) {
        console.error('Experiment initialization failed:', err);
        setError(err instanceof Error ? err.message : String(err));
        setIsLoading(false);
      }
    }
    
    initExperiment();
    
    return () => {
      cancelled = true;
      stopSimulation();
      if (experiment) {
        experiment.cleanup();
      }
      deviceRef.current = null;
    };
  }, [backend, experimentId]);
  
  // Animation loop
  const tick = useCallback(async () => {
    if (!isRunningRef.current || !experiment) return;
    
    // Step simulation
    if (experiment.step) {
      await experiment.step(1);
    }
    
    // Update metrics
    setMetrics(experiment.getMetrics());
    
    // Schedule next frame
    if (isRunningRef.current) {
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [experiment]);
  
  const startSimulation = useCallback(() => {
    if (backend !== 'webgpu' || !experiment) return;
    if (isRunningRef.current) return;
    
    isRunningRef.current = true;
    setIsRunning(true);
    experiment.start();
    rafRef.current = requestAnimationFrame(tick);
  }, [backend, experiment, tick]);
  
  const stopSimulation = useCallback(() => {
    isRunningRef.current = false;
    setIsRunning(false);
    
    if (experiment) {
      experiment.pause();
    }
    
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [experiment]);
  
  const resetSimulation = useCallback(async () => {
    stopSimulation();
    if (experiment) {
      await experiment.reset();
      setMetrics(experiment.getMetrics());
    }
  }, [experiment, stopSimulation]);
  
  const handleParameterChange = useCallback((key: string, value: any) => {
    setParameters(prev => ({ ...prev, [key]: value }));
    if (experiment) {
      experiment.updateParameters({ [key]: value });
    }
  }, [experiment]);
  
  const handleViewChange = useCallback((key: string, value: boolean) => {
    setViews(prev => ({ ...prev, [key]: value }));
  }, []);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      // Space or K - Toggle play/pause
      if ((e.key === ' ' || e.key === 'k' || e.key === 'K') && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        if (isRunningRef.current) {
          stopSimulation();
        } else {
          startSimulation();
        }
      }
      
      // R - Reset
      if ((e.key === 'r' || e.key === 'R') && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        resetSimulation();
      }
      
      // Escape - Pause
      if (e.key === 'Escape') {
        e.preventDefault();
        stopSimulation();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [startSimulation, stopSimulation, resetSimulation]);
  
  // Loading state
  if (isLoading) {
    return <ExperimentSkeleton stage="initializing" experimentTitle={metadata.title} />;
  }
  
  // Error state
  if (error) {
    return (
      <div className="min-h-screen flex flex-col bg-space-dark">
        <Header />
        <main className="flex-1 flex items-center justify-center pt-20">
          <div className="text-center">
            <div className="text-6xl mb-4">⚠️</div>
            <h3 className="text-2xl font-bold text-red-500 mb-2">Initialization Failed</h3>
            <p className="text-text-secondary mb-6 max-w-md">{error}</p>
            <button onClick={() => window.location.reload()} className="btn-primary">
              Retry
            </button>
          </div>
        </main>
        <Footer />
      </div>
    );
  }
  
  // Render experiment
  const RenderComponent = experiment?.RenderComponent;
  
  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />
      
      <main className="flex-1 pt-20">
        <div className="container mx-auto px-4 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <h1 className="text-4xl font-bold text-accent-chi mb-2">{metadata.title}</h1>
                <p className="text-text-secondary">{metadata.fullDescription}</p>
              </div>
              <Link 
                href={`/about?from=${experimentId}`}
                className="px-4 py-2 bg-yellow-500/20 border-2 border-yellow-500/50 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors whitespace-nowrap text-sm font-semibold"
              >
                ⚠️ Read About This Project
              </Link>
            </div>
            <div className="bg-yellow-500/10 border-l-4 border-yellow-500 p-4 rounded">
              <p className="text-sm text-text-secondary">
                <strong className="text-yellow-400">Scientific Disclosure:</strong> This is an exploratory simulation. 
                We are NOT claiming this is proven physics. <Link href={`/about?from=${experimentId}`} className="text-accent-chi hover:underline">Learn more about our approach and limitations →</Link>
              </p>
            </div>
          </div>

          {/* Backend Status */}
          <div className="mb-8">
            <BackendBadge backend={backend} />
          </div>

          {/* Main Experiment Area */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Visualization Canvas (Left side - 2/3 width) */}
            <div className="lg:col-span-2">
              <div className="panel h-[600px] relative overflow-hidden">
                {backend === 'webgpu' && RenderComponent ? (
                  <WebGPUErrorBoundary>
                    <RenderComponent
                      isRunning={isRunning}
                      parameters={parameters}
                      views={views}
                    />
                  </WebGPUErrorBoundary>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-6xl mb-4">🖥️</div>
                      <h3 className="text-2xl font-bold text-accent-chi mb-2">WebGPU Not Available</h3>
                      <p className="text-text-secondary mb-6">
                        This experiment requires WebGPU. Upgrade your browser or enable experimental flags.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="mt-4 flex items-center justify-center space-x-4">
                <button
                  onClick={startSimulation}
                  disabled={backend !== 'webgpu' || isRunning}
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
                    backend !== 'webgpu' || isRunning
                      ? 'bg-accent-glow/40 text-space-dark/60 cursor-not-allowed'
                      : 'bg-accent-glow hover:bg-accent-glow/80 text-space-dark'
                  }`}
                >
                  ▶ Play
                </button>
                <button
                  onClick={stopSimulation}
                  disabled={!isRunning}
                  className={`px-6 py-3 rounded-lg font-semibold transition-colors border-2 ${
                    !isRunning
                      ? 'border-accent-chi/40 text-accent-chi/40 cursor-not-allowed'
                      : 'border-accent-chi text-accent-chi hover:bg-accent-chi/10'
                  }`}
                >
                  ⏸ Pause
                </button>
                <button
                  onClick={resetSimulation}
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-indigo-500 hover:bg-indigo-400 text-white"
                >
                  🔄 Reset
                </button>
              </div>
            </div>

            {/* Control Panel (Right side - 1/3 width) */}
            <div className="space-y-6">
              {/* Parameters */}
              {experiment && experiment.config.parameters.length > 0 && (
                <div className="panel">
                  <h3 className="text-lg font-bold text-accent-chi mb-4">Parameters</h3>
                  <div className="space-y-4">
                    {experiment.config.parameters.map((param) => (
                      <ParameterControl
                        key={param.key}
                        param={param}
                        value={parameters[param.key]}
                        onChange={(value) => handleParameterChange(param.key, value)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Metrics */}
              {metrics.length > 0 && (
                <div className="panel">
                  <h3 className="text-lg font-bold text-accent-chi mb-4">System Stats</h3>
                  <div className="space-y-3">
                    {metrics.map((metric, idx) => (
                      <MetricDisplay key={idx} metric={metric} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Explanation Panel */}
          <div className="mt-8 panel">
            <h3 className="text-xl font-bold text-accent-chi mb-4">What You're Seeing</h3>
            <div className="prose prose-invert max-w-none text-text-secondary">
              <p className="mb-4">{metadata.education.whatYouSee}</p>
              
              {metadata.education.principles.length > 0 && (
                <>
                  <h4 className="text-lg font-semibold text-text-primary mb-2">Key Principles:</h4>
                  <ul className="space-y-2 list-disc list-inside mb-4">
                    {metadata.education.principles.map((principle, idx) => (
                      <li key={idx}>{principle}</li>
                    ))}
                  </ul>
                </>
              )}
              
              {metadata.education.realWorld && (
                <p className="mt-4">
                  <strong className="text-text-primary">Real-World Context:</strong> {metadata.education.realWorld}
                </p>
              )}
              
              <div className="mt-6 p-4 bg-accent-chi/10 border-l-4 border-accent-chi rounded">
                <p className="text-text-primary font-semibold mb-2">
                  Want to understand the full scientific context?
                </p>
                <Link 
                  href={`/about?from=${experimentId}`}
                  className="inline-flex items-center gap-2 text-accent-chi hover:text-accent-particle transition-colors font-semibold"
                >
                  <span>Read our full About page</span>
                  <span>→</span>
                </Link>
                <p className="text-sm text-text-secondary mt-2">
                  Learn about the mathematics, limitations, and how to access our research data and source code.
                </p>
              </div>
            </div>
          </div>
          
          {/* Custom content slot */}
          {customContent}
        </div>
      </main>

      <Footer />
      
      {/* Keyboard Shortcuts Overlay */}
      <KeyboardShortcuts 
        shortcuts={[
          { key: 'Space', description: 'Play / Pause simulation' },
          { key: 'K', description: 'Play / Pause simulation' },
          { key: 'R', description: 'Reset simulation' },
          { key: 'Escape', displayKey: 'Esc', description: 'Pause simulation' },
          { key: '?', description: 'Show / Hide this help', displayKey: 'Shift + /' },
        ]}
      />
    </div>
  );
}

/**
 * Parameter control component (number slider, boolean checkbox, etc.)
 */
function ParameterControl({ 
  param, 
  value, 
  onChange 
}: { 
  param: any; 
  value: any; 
  onChange: (value: any) => void;
}) {
  if (param.type === 'number') {
    return (
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold text-text-primary" title={param.description}>
            {param.label}
          </label>
          <span className="text-sm font-mono text-accent-chi">
            {typeof value === 'number' 
              ? `${value.toFixed(param.step < 0.01 ? 4 : param.step < 0.1 ? 3 : param.step < 1 ? 2 : 1)} ${param.unit || ''}`
              : value}
          </span>
        </div>
        <input
          type="range"
          min={param.min}
          max={param.max}
          step={param.step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          title={param.description}
          className="w-full h-2 bg-space-border rounded-lg appearance-none cursor-pointer accent-accent-chi"
        />
        {param.description && (
          <p className="text-xs text-text-secondary mt-1">{param.description}</p>
        )}
      </div>
    );
  }
  
  if (param.type === 'boolean') {
    return (
      <label className="flex items-center space-x-3 cursor-pointer">
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(e.target.checked)}
          className="w-5 h-5 rounded border-2 border-accent-chi checked:bg-accent-chi"
        />
        <span className="text-sm text-text-primary">{param.label}</span>
      </label>
    );
  }
  
  return null;
}

/**
 * Metric display component
 */
function MetricDisplay({ metric }: { metric: ExperimentMetricsType }) {
  const statusColors = {
    good: 'text-accent-glow',
    warning: 'text-yellow-500',
    error: 'text-red-500',
    neutral: 'text-text-primary',
  };
  
  return (
    <div className="flex items-center justify-between py-2 border-b border-space-border last:border-b-0">
      <span className="text-sm text-text-secondary" title={metric.tooltip}>
        {metric.label}
      </span>
      <span className={`text-sm font-mono font-semibold ${statusColors[metric.status || 'neutral']}`}>
        {metric.value} {metric.unit || ''}
      </span>
    </div>
  );
}

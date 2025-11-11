/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { useState, useCallback, useRef } from 'react';
import { ExperimentDefinition } from '@/data/experiments';
import SimulationDispatcher from './SimulationDispatcher';
import { SimulationControls } from './canvases/types';
import ControlPanel from './ControlPanel';
import ParameterPanel from './ParameterPanel';
import MetricsPanel from './MetricsPanel';
import ValidationPanel from './ValidationPanel';
import VisualizationOptions from '@/components/ui/VisualizationOptions';
import Link from 'next/link';

interface ExperimentTemplateProps {
  experiment: ExperimentDefinition;
}

export default function ExperimentTemplate({ experiment }: ExperimentTemplateProps) {
  // Determine mode from experiment type
  const mode = experiment.type || 'SHOWCASE';
  
  // Simulation state
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [speed, setSpeed] = useState(1.0);
  
  // Parameters (initialized from experiment config)
  const [parameters, setParameters] = useState(experiment.initialConditions);
  
  // Simulation history for step-by-step navigation (RESEARCH mode)
  const [simulationHistory, setSimulationHistory] = useState<any[]>([]);
  
  // Reference to simulation controls (for step forward/back)
  const simulationRef = useRef<SimulationControls | null>(null);
  
  // Auto-pause when reaching max steps
  const maxSteps = parameters.steps || 6000;
  if (isRunning && currentStep >= maxSteps) {
    setIsRunning(false);
  }
  
  // Metrics (updated during simulation)
  const [metrics, setMetrics] = useState<Record<string, number | string>>({});
  
  // Visualization toggles (initialized from experiment config)
  const [visualizationToggles, setVisualizationToggles] = useState(() => {
    const viz = experiment.visualization || {};
    return {
      showParticles: viz.showParticles ?? true,
      showTrails: viz.showTrails ?? false,
      showChi: viz.showChi ?? false,
      showLattice: viz.showLattice ?? true,
      showVectors: viz.showVectors ?? false,
      showWell: viz.showWell ?? false,
      showDomes: viz.showDomes ?? false,
      showIsoShells: viz.showIsoShells ?? false,
      showBackground: viz.showBackground ?? true,
    };
  });
  
  // Validation state
  const [validationStatus, setValidationStatus] = useState<'idle' | 'running' | 'pass' | 'fail'>('idle');
  const [validationResults, setValidationResults] = useState<any>(null);
  
  // Control handlers
  const handlePlay = useCallback(() => {
    setIsRunning(true);
  }, []);
  
  const handlePause = useCallback(() => {
    setIsRunning(false);
  }, []);
  
  const handleReset = useCallback(() => {
    setIsRunning(false);
    setCurrentStep(0);
    setParameters(experiment.initialConditions);
    setMetrics({});
    setValidationStatus('idle');
    setValidationResults(null);
    setSimulationHistory([]);  // Clear history
  }, [experiment.initialConditions]);
  
  // Step forward handler (RESEARCH mode)
  const handleStepForward = useCallback(() => {
    if (currentStep >= maxSteps || !simulationRef.current) return;
    
    // 1. Capture current state BEFORE stepping
    const currentState = simulationRef.current.getState();
    
    // 2. Store in history buffer (circular, keep last 100)
    setSimulationHistory(prev => {
      const newHistory = [...prev, currentState];
      if (newHistory.length > 100) {
        newHistory.shift(); // Remove oldest
      }
      return newHistory;
    });
    
    // 3. Execute one physics step
    simulationRef.current.step();
    
    // Note: currentStep is updated by the simulation's onStepUpdate callback
  }, [currentStep, maxSteps]);
  
  // Step backward handler (RESEARCH mode)
  const handleStepBackward = useCallback(() => {
    if (currentStep === 0 || simulationHistory.length === 0 || !simulationRef.current) return;
    
    // 1. Pop last state from history
    const prevState = simulationHistory[simulationHistory.length - 1];
    setSimulationHistory(prev => prev.slice(0, -1));
    
    // 2. Restore simulation state
    simulationRef.current.setState(prevState);
    
    // Note: currentStep is updated by setState's onStepUpdate callback
  }, [currentStep, simulationHistory]);
  
  const handleParameterChange = useCallback((param: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }));
  }, []);
  
  const handleVisualizationToggle = useCallback((key: string, value: boolean) => {
    setVisualizationToggles(prev => ({
      ...prev,
      [key]: value
    }));
  }, []);
  
  const handleMetricsUpdate = useCallback((newMetrics: Record<string, number | string>) => {
    setMetrics(prev => ({
      ...prev,
      ...newMetrics
    }));
  }, []);
  
  const handleValidate = useCallback(async () => {
    setValidationStatus('running');
    try {
      const response = await fetch(`/api/experiments/${experiment.id}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          parameters,
          metrics  // Send current UI metrics for validation
        })
      });
      const result = await response.json();
      setValidationResults(result);
      setValidationStatus(result.status === 'PASS' ? 'pass' : 'fail');
    } catch (error) {
      console.error('Validation failed:', error);
      setValidationStatus('fail');
    }
  }, [experiment.id, parameters, metrics]);
  
  // Status badge color
  const statusColor = {
    production: 'bg-green-600/30 text-green-300',
    beta: 'bg-yellow-600/30 text-yellow-300',
    development: 'bg-orange-600/30 text-orange-300',
    planned: 'bg-slate-600/30 text-slate-300',
  }[experiment.status];
  
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header Section */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <Link href={mode === 'RESEARCH' ? '/research' : '/experiments'} className="text-accent-chi hover:underline">
            ← {mode === 'RESEARCH' ? 'Research' : 'Experiments'}
          </Link>
          <span className="text-text-muted">/</span>
          <span className="text-text-secondary">{experiment.testId || experiment.id}</span>
          {mode && (
            <span className={`px-2 py-1 rounded text-xs font-bold ${
              mode === 'RESEARCH' 
                ? 'bg-indigo-600/30 text-indigo-300' 
                : 'bg-green-600/30 text-green-300'
            }`}>
              {mode}
            </span>
          )}
        </div>
        
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <h1 className="text-4xl font-bold text-white mb-2">{experiment.displayName}</h1>
            <p className="text-xl text-text-secondary mb-4">{experiment.tagline}</p>
          </div>
          <div className="flex flex-col gap-2 items-end">
            <span className={`px-3 py-1 rounded text-sm font-semibold ${statusColor}`}>
              {experiment.status}
            </span>
            {experiment.tier && (
              <span className="px-3 py-1 bg-indigo-600/30 text-indigo-300 rounded text-sm">
                Tier {experiment.tier}: {experiment.tierName}
              </span>
            )}
            <span className="px-3 py-1 bg-slate-700 text-slate-300 rounded text-sm">
              {experiment.simulation}
            </span>
          </div>
        </div>
        
        <p className="text-text-secondary leading-relaxed max-w-4xl">
          {experiment.description}
        </p>
      </div>
      
      {/* GRAV-09 Skip Warning Banner */}
      {(experiment.testId === 'GRAV-09' || experiment.id === 'GRAV-09') && (
        <div className="bg-yellow-900/30 border-2 border-yellow-600 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 text-yellow-500 text-3xl">⚠️</div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-yellow-400 mb-2">Test Design Limitation (Skipped in Validation)</h3>
              <p className="text-yellow-200/90 mb-3 leading-relaxed">
                This test is <strong>skipped</strong> in the official validation suite due to a fundamental discretization artifact in Klein-Gordon field theory on finite grids.
              </p>
              <div className="space-y-2 text-yellow-100/80 text-sm leading-relaxed">
                <p><strong>Scientific Context:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Continuous Klein-Gordon allows bound states with ω≈χ (k→0 limit)</li>
                  <li>Discrete grid with dx=0.5 requires k_min=2π/L ≈ 2.26 (36% of k_max)</li>
                  <li>Any localized initial condition couples to grid modes where k² dominates χ² by factor ~100</li>
                  <li>Both wells measure ω≈2.25 (dispersion relation), not χA=0.30, χB=0.14</li>
                  <li>Makes ratio validation impossible (measured 1.03 vs expected 2.14)</li>
                </ul>
                <p className="mt-3"><strong>Why This Matters:</strong></p>
                <p className="ml-4">
                  This demonstrates an important lesson about discrete vs continuous field theory that physicists must account for when designing numerical tests. 
                  The discrete grid introduces unavoidable k-content; tests must be designed recognizing ω²=k²+χ², not assuming ω≈χ.
                </p>
                <p className="mt-3"><strong>Alternative Validation:</strong></p>
                <p className="ml-4">
                  Tests GRAV-07, GRAV-08, GRAV-13, and GRAV-18 successfully validate χ-dependent effects without this discretization artifact.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Simulation Canvas */}
        <div className="lg:col-span-2">
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg overflow-hidden">
            <SimulationDispatcher
              experiment={experiment}
              isRunning={isRunning}
              parameters={parameters}
              visualizationToggles={visualizationToggles}
              onMetricsUpdate={handleMetricsUpdate}
              onStepUpdate={setCurrentStep}
              simulationRef={simulationRef}
            />
          </div>
          
          {/* Control Panel */}
          <div className="mt-4">
            <ControlPanel
              mode={mode}
              isRunning={isRunning}
              speed={speed}
              currentStep={currentStep}
              totalSteps={parameters.steps}
              onPlay={handlePlay}
              onPause={handlePause}
              onReset={handleReset}
              onSpeedChange={setSpeed}
              onStepForward={mode === 'RESEARCH' ? handleStepForward : undefined}
              onStepBackward={mode === 'RESEARCH' ? handleStepBackward : undefined}
            />
          </div>
        </div>
        
        {/* Right Column: Parameters & Metrics */}
        <div className="space-y-6">
          {/* Visualization Options */}
          <VisualizationOptions
            toggles={[
              { key: 'showParticles', label: 'Bodies / Particles', checked: visualizationToggles.showParticles },
              { key: 'showTrails', label: 'Trails / Trajectories', checked: visualizationToggles.showTrails },
              { key: 'showChi', label: 'Chi Field (Heatmap)', checked: visualizationToggles.showChi },
              { key: 'showLattice', label: 'Lattice Grid', checked: visualizationToggles.showLattice },
              { key: 'showVectors', label: 'Force Vectors', checked: visualizationToggles.showVectors },
              { key: 'showBackground', label: 'Stars & Background', checked: visualizationToggles.showBackground },
            ].filter(toggle => {
              // Only show toggles relevant to this simulation type
              if (experiment.simulation === 'wave-packet') {
                return ['showLattice', 'showChi', 'showVectors', 'showBackground'].includes(toggle.key);
              } else if (experiment.simulation === 'field-dynamics') {
                return ['showChi', 'showLattice', 'showVectors', 'showBackground'].includes(toggle.key);
              } else if (experiment.simulation === 'n-body' || experiment.simulation === 'binary-orbit') {
                return ['showParticles', 'showTrails', 'showChi', 'showBackground'].includes(toggle.key);
              }
              return true; // Show all by default
            })}
            onChange={handleVisualizationToggle}
          />
          
          {/* Parameter Panel */}
          <ParameterPanel
            mode={mode}
            experiment={experiment}
            parameters={parameters}
            onParameterChange={handleParameterChange}
            disabled={isRunning}
          />
          
          {/* Metrics Panel */}
          <MetricsPanel
            experiment={experiment}
            metrics={metrics}
            currentStep={currentStep}
          />
          
          {/* Validation Panel */}
          {experiment.validation && (
            <div className="space-y-4">
              {/* Coming Soon Notice */}
              <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <div className="text-2xl">🔬</div>
                  <div className="flex-1">
                    <h4 className="font-bold text-blue-300 mb-1">Validation Coming Soon</h4>
                    <p className="text-sm text-gray-400">
                      Cryptographic validation with exact Python code reproduction will be available in the next update.
                    </p>
                  </div>
                </div>
              </div>
              
              <ValidationPanel
                experiment={experiment}
                status={validationStatus}
                results={validationResults}
                onValidate={handleValidate}
              />
            </div>
          )}
          
          {/* Documentation Links */}
          {Object.keys(experiment.links).length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <h3 className="text-lg font-bold text-white mb-3">Documentation</h3>
              <div className="space-y-2">
                {experiment.links.testHarnessConfig && (
                  <a
                    href={experiment.links.testHarnessConfig}
                    className="block text-accent-chi hover:underline text-sm"
                  >
                    → Test Config JSON
                  </a>
                )}
                {experiment.links.results && (
                  <a
                    href={experiment.links.results}
                    className="block text-accent-chi hover:underline text-sm"
                  >
                    → Test Results
                  </a>
                )}
                {experiment.links.discovery && (
                  <a
                    href={experiment.links.discovery}
                    className="block text-accent-chi hover:underline text-sm"
                  >
                    → Discovery Entry
                  </a>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

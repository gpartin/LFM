/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { useEffect, useRef } from 'react';
import { ExperimentDefinition } from '@/data/experiments';
import { SimulationControls, SimulationState } from './types';

interface NBodyCanvasProps {
  experiment: ExperimentDefinition;
  isRunning: boolean;
  parameters: any;
  visualizationToggles: Record<string, boolean>;
  onMetricsUpdate: (metrics: Record<string, number | string>) => void;
  onStepUpdate: (step: number) => void;
  simulationRef?: React.MutableRefObject<SimulationControls | null>;
}

/**
 * Canvas component for n-body simulations (Tier 2 Gravity tests with orbital mechanics).
 * Renders particles orbiting in chi field with emergent gravitational effects.
 * 
 * TODO: Implement WebGL simulation
 * - Initialize particles from parameters.particles
 * - Compute chi field gradients
 * - Apply emergent forces (gradient coupling)
 * - Render orbital trajectories
 * - Track orbital stability, period, eccentricity
 * - Update metrics: energyDrift, orbital_period, eccentricity
 */
export default function NBodyCanvas({
  experiment,
  isRunning,
  parameters,
  visualizationToggles,
  onMetricsUpdate,
  onStepUpdate,
  simulationRef
}: NBodyCanvasProps) {
  const animationRef = useRef<number | null>(null);
  const stepRef = useRef(0);
  const energyDriftRef = useRef(0);
  const orbitalPeriodRef = useRef(100);
  
  // Expose simulation controls to parent
  useEffect(() => {
    if (!simulationRef) return;
    
    simulationRef.current = {
      step: () => {
        executePhysicsStep();
      },
      getState: () => ({
        currentStep: stepRef.current,
        energyDrift: energyDriftRef.current,
        orbitalPeriod: orbitalPeriodRef.current
      }),
      setState: (state: SimulationState) => {
        stepRef.current = state.currentStep;
        energyDriftRef.current = state.energyDrift ?? energyDriftRef.current;
        orbitalPeriodRef.current = state.orbitalPeriod ?? orbitalPeriodRef.current;
        onStepUpdate(stepRef.current);
        onMetricsUpdate({
          energyDrift: energyDriftRef.current,
          orbital_period: orbitalPeriodRef.current,
          step: stepRef.current
        });
      }
    };
  }, [simulationRef, onMetricsUpdate, onStepUpdate]);
  
  /**
   * Execute one physics timestep (stub implementation)
   * TODO: Replace with actual n-body physics simulation
   */
  const executePhysicsStep = () => {
    stepRef.current += 1;
    energyDriftRef.current = 1e-4 * Math.random();
    orbitalPeriodRef.current = 100 + 5 * Math.random();
    
    onStepUpdate(stepRef.current);
    onMetricsUpdate({
      energyDrift: energyDriftRef.current,
      orbital_period: orbitalPeriodRef.current,
      step: stepRef.current
    });
  };
  
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }
    
    const animate = () => {
      executePhysicsStep();
      
      if (stepRef.current < parameters.steps) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };
    
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRunning, parameters.steps, onMetricsUpdate, onStepUpdate]);
  
  return (
    <div className="w-full h-[600px] bg-slate-900 relative">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center text-text-muted">
          <div className="text-xl font-semibold mb-2">N-Body Simulation</div>
          <div className="text-sm mb-4">{experiment.testId || experiment.id}</div>
          <div className="text-xs">
            WebGL implementation coming soon
          </div>
          <div className="mt-4 text-xs font-mono">
            Step: {stepRef.current} / {parameters.steps}
          </div>
        </div>
      </div>
    </div>
  );
}

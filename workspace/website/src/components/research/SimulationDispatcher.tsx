/*
 * SimulationDispatcher — injects the correct simulation canvas for each experiment type
 * Registry-based architecture for maintainability and extensibility
 */

import React from 'react';
import type { ExperimentDefinition, SimulationType } from '@/data/experiments';

import WavePacketCanvas from '@/components/visuals/WavePacketCanvas';
import FieldDynamicsCanvas from '@/components/visuals/FieldDynamicsCanvas';
import BinaryOrbitCanvas from '@/components/visuals/BinaryOrbitCanvas';

interface LiveMetrics {
  energy?: number;
  energyDriftPct?: number;
  time?: number;
}

interface Props {
  experiment: ExperimentDefinition;
  isRunning?: boolean;
  onMetrics?: (m: LiveMetrics) => void;
  speedFactor?: number;       // From UI speed slider (1.0 default)
  resetCounter?: number;      // Increment to trigger reset
}

// Registry pattern: map simulation type to canvas component
type CanvasCommonProps = { experiment: ExperimentDefinition; isRunning?: boolean; onMetrics?: (m: LiveMetrics) => void; speedFactor?: number; resetCounter?: number };
const CANVAS_REGISTRY: Record<SimulationType, React.ComponentType<CanvasCommonProps>> = {
  'wave-packet': WavePacketCanvas as React.ComponentType<CanvasCommonProps>,
  'n-body': BinaryOrbitCanvas as React.ComponentType<CanvasCommonProps>,
  'binary-orbit': BinaryOrbitCanvas as React.ComponentType<CanvasCommonProps>,
  'field-dynamics': FieldDynamicsCanvas as React.ComponentType<CanvasCommonProps>,
};

export default function SimulationDispatcher({ experiment, isRunning, onMetrics, speedFactor = 1.0, resetCounter = 0 }: Props) {
  // Safety check
  if (!experiment) {
    return (
      <div className="h-full w-full flex items-center justify-center text-slate-400">
        <div className="text-center">
          <div className="text-2xl mb-2">⚠️</div>
          <div>Experiment data not loaded</div>
        </div>
      </div>
    );
  }

  const CanvasComponent = CANVAS_REGISTRY[experiment.simulation];
  
  if (!CanvasComponent) {
    return (
      <div className="h-full w-full flex items-center justify-center text-slate-400">
        Unknown simulation type: {experiment.simulation}
      </div>
    );
  }
  
  return (
    <div className="h-full w-full flex items-center justify-center">
      <CanvasComponent 
        experiment={experiment} 
        isRunning={isRunning} 
        onMetrics={onMetrics} 
        speedFactor={speedFactor} 
        resetCounter={resetCounter}
      />
    </div>
  );
}

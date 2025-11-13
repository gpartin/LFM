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
}

// Registry pattern: map simulation type to canvas component
const CANVAS_REGISTRY: Record<SimulationType, React.ComponentType<{ experiment: ExperimentDefinition; isRunning?: boolean; onMetrics?: (m: LiveMetrics) => void }>> = {
  'wave-packet': WavePacketCanvas,
  'n-body': BinaryOrbitCanvas,
  'binary-orbit': BinaryOrbitCanvas,
  'field-dynamics': FieldDynamicsCanvas,
};

export default function SimulationDispatcher({ experiment, isRunning, onMetrics }: Props) {
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
      <CanvasComponent experiment={experiment} isRunning={isRunning} onMetrics={onMetrics} />
    </div>
  );
}

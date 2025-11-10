/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { ExperimentDefinition } from '@/data/experiments';
import WavePacketCanvas from './canvases/WavePacketCanvas';
import FieldDynamicsCanvas from './canvases/FieldDynamicsCanvas';
import NBodyCanvas from './canvases/NBodyCanvas';

interface SimulationDispatcherProps {
  experiment: ExperimentDefinition;
  isRunning: boolean;
  parameters: any;
  visualizationToggles: Record<string, boolean>;
  onMetricsUpdate: (metrics: Record<string, number | string>) => void;
  onStepUpdate: (step: number) => void;
}

export default function SimulationDispatcher({
  experiment,
  isRunning,
  parameters,
  visualizationToggles,
  onMetricsUpdate,
  onStepUpdate
}: SimulationDispatcherProps) {
  
  // Map simulation type to appropriate canvas component
  switch (experiment.simulation) {
    case 'wave-packet':
      return (
        <WavePacketCanvas
          experiment={experiment}
          isRunning={isRunning}
          parameters={parameters}
          visualizationToggles={visualizationToggles}
          onMetricsUpdate={onMetricsUpdate}
          onStepUpdate={onStepUpdate}
        />
      );
      
    case 'field-dynamics':
      return (
        <FieldDynamicsCanvas
          experiment={experiment}
          isRunning={isRunning}
          parameters={parameters}
          visualizationToggles={visualizationToggles}
          onMetricsUpdate={onMetricsUpdate}
          onStepUpdate={onStepUpdate}
        />
      );
      
    case 'n-body':
    case 'binary-orbit':
      return (
        <NBodyCanvas
          experiment={experiment}
          isRunning={isRunning}
          parameters={parameters}
          visualizationToggles={visualizationToggles}
          onMetricsUpdate={onMetricsUpdate}
          onStepUpdate={onStepUpdate}
        />
      );
      
    default:
      return (
        <div className="h-[600px] flex items-center justify-center text-text-muted">
          <div className="text-center">
            <p className="text-xl mb-2">Simulation type not yet implemented</p>
            <p className="text-sm">{experiment.simulation}</p>
          </div>
        </div>
      );
  }
}

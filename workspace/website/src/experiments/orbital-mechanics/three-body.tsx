/* -*- coding: utf-8 -*- */
/**
 * Three-Body Problem - MVP
 * 
 * Demonstrates chaotic N-body dynamics using emergent gravity.
 * MVP: Currently uses binary orbit as base (full 3-body coming soon)
 */

import type { 
  Experiment, 
  ExperimentFactory,
  ExperimentMetadata,
  ExperimentConfig,
  ExperimentMetrics,
  ExperimentResults,
  ExperimentParameter,
} from '@/types/experiment';
import { BinaryOrbitSimulation, OrbitConfig } from '@/physics/forces/binary-orbit';
import dynamic from 'next/dynamic';

const SimpleCanvas = dynamic(() => import('@/components/visuals/SimpleCanvas'), { ssr: false });

const PARAMETERS: ExperimentParameter[] = [
  {
    key: 'mass1',
    label: 'Body 1 Mass',
    description: 'Mass of first body',
    type: 'number',
    defaultValue: 1.0,
    min: 0.5,
    max: 3.0,
    step: 0.1,
    unit: 'M',
    liveUpdate: true,
  },
  {
    key: 'mass2',
    label: 'Body 2 Mass',
    description: 'Mass of second body',
    type: 'number',
    defaultValue: 1.0,
    min: 0.5,
    max: 3.0,
    step: 0.1,
    unit: 'M',
    liveUpdate: true,
  },
  {
    key: 'separation',
    label: 'Initial Separation',
    description: 'Starting distance between bodies',
    type: 'number',
    defaultValue: 3.0,
    min: 2.0,
    max: 8.0,
    step: 0.5,
    unit: 'units',
    liveUpdate: false,
  },
  {
    key: 'chiStrength',
    label: 'Chi Strength',
    description: 'Coupling strength',
    type: 'number',
    defaultValue: 0.3,
    min: 0.1,
    max: 0.6,
    step: 0.05,
    unit: '',
    liveUpdate: true,
  },
];

class ThreeBodyExperiment implements Experiment {
  private device: GPUDevice;
  private simulation: BinaryOrbitSimulation | null = null;
  private parameters: Record<string, any>;
  
  metadata: ExperimentMetadata;
  config: ExperimentConfig;
  RenderComponent: React.ComponentType<any>;
  
  constructor(device: GPUDevice, initialParams?: Partial<Record<string, any>>) {
    this.device = device;
    
    this.parameters = {};
    PARAMETERS.forEach(param => {
      this.parameters[param.key] = initialParams?.[param.key] ?? param.defaultValue;
    });
    
    this.metadata = {
      id: 'three-body',
      title: 'Three-Body Problem',
      shortDescription: 'Watch chaotic dynamics emerge from three gravitating bodies',
      fullDescription: 'The three-body problem demonstrates chaos - tiny changes lead to wildly different outcomes.',
      category: 'orbital-mechanics',
      tags: ['chaos', 'gravity', 'n-body'],
      difficulty: 'advanced',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'Three bodies interacting chaotically',
        principles: ['No analytical solution', 'Chaos theory', 'Emergent gravity'],
        references: [],
      },
      estimatedRuntime: 60,
    };
    
    this.config = {
      parameters: PARAMETERS,
      defaultViews: {
        showParticles: true,
        showTrails: true,
      },
    };
    
    this.RenderComponent = SimpleCanvas as any;
  }
  
  async initialize(): Promise<void> {
    const config: OrbitConfig = {
      mass1: this.parameters.mass1,
      mass2: this.parameters.mass2,
      initialSeparation: this.parameters.separation,
      chiStrength: this.parameters.chiStrength,
      latticeSize: 64,
      dt: 0.002,
      sigma: 1.5,
    };
    
    this.simulation = new BinaryOrbitSimulation(this.device, config);
    await this.simulation.initialize();
  }
  
  async cleanup(): Promise<void> {
    if (this.simulation) {
      this.simulation.destroy();
      this.simulation = null;
    }
  }
  
  async reset(): Promise<void> {
    if (this.simulation) {
      this.simulation.reset();
      await this.simulation.initialize();
    }
  }
  
  start(): void {}
  pause(): void {}
  
  async step(frames: number = 1): Promise<void> {
    if (this.simulation) {
      await this.simulation.stepBatch(frames);
    }
  }
  
  updateParameters(params: Record<string, any>): void {
    Object.assign(this.parameters, params);
    if (this.simulation) {
      this.simulation.updateParameters({
        mass1: this.parameters.mass1,
        mass2: this.parameters.mass2,
        chiStrength: this.parameters.chiStrength,
      });
      this.simulation.refreshChiField();
    }
  }
  
  getMetrics(): ExperimentMetrics[] {
    if (!this.simulation) return [];
    const state = this.simulation.getState();
    return [
      { label: 'Energy', value: state.energy.toFixed(4), unit: 'J', status: 'good' },
      { label: 'Angular Momentum', value: state.angularMomentum.toFixed(3), unit: '', status: 'good' },
    ];
  }
  
  getResults(): ExperimentResults {
    const state = this.simulation?.getState();
    return {
      timestamp: new Date().toISOString(),
      parameters: this.parameters,
      metrics: { energy: state?.energy },
    };
  }
  
  async exportResults(format: 'json' | 'csv' = 'json'): Promise<string> {
    return JSON.stringify(this.getResults(), null, 2);
  }
}

const createThreeBodyExperiment: ExperimentFactory = (device, initialConfig) => {
  return new ThreeBodyExperiment(device, initialConfig);
};

export default createThreeBodyExperiment;

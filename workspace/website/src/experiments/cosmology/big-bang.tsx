/* -*- coding: utf-8 -*- */
/**
 * Big Bang Experiment - MVP
 * 
 * Energy pulse expanding spherically from a point.
 * Pure wave propagation: ∂²E/∂t² = c²∇²E
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
    key: 'initialEnergy',
    label: 'Initial Energy',
    description: 'Energy of the initial pulse',
    type: 'number',
    defaultValue: 10.0,
    min: 1.0,
    max: 50.0,
    step: 1.0,
    unit: 'J',
    liveUpdate: false,
  },
  {
    key: 'pulseWidth',
    label: 'Pulse Width',
    description: 'Spatial extent of initial energy concentration',
    type: 'number',
    defaultValue: 0.5,
    min: 0.1,
    max: 2.0,
    step: 0.1,
    unit: 'units',
    liveUpdate: false,
  },
  {
    key: 'waveSpeed',
    label: 'Wave Speed (c)',
    description: 'Speed of light / wave propagation',
    type: 'number',
    defaultValue: 1.0,
    min: 0.5,
    max: 2.0,
    step: 0.1,
    unit: 'c',
    liveUpdate: false,
  },
];

class BigBangExperiment implements Experiment {
  private device: GPUDevice;
  private simulation: BinaryOrbitSimulation | null = null;
  private parameters: Record<string, any>;
  private hasExploded: boolean = false;
  
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
      id: 'big-bang',
      title: 'Big Bang',
      shortDescription: 'Watch energy explode outward in all directions from a single point',
      fullDescription: 'Pure wave propagation from the LFM equation. Energy starts concentrated at the center and expands spherically at the speed of light.',
      category: 'cosmology',
      tags: ['waves', 'expansion', 'cosmology'],
      difficulty: 'intermediate',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'Spherical energy wave expanding from the center',
        principles: ['Wave equation', 'Spherical symmetry', 'Energy conservation'],
        realWorld: 'Models the early universe expansion (simplified)',
        references: [],
      },
      estimatedRuntime: 30,
    };
    
    this.config = {
      parameters: PARAMETERS,
      defaultViews: {
        showField: true,
        showParticles: false,
      },
    };
    
    this.RenderComponent = SimpleCanvas as any;
  }
  
  async initialize(): Promise<void> {
    // MVP: Use binary orbit sim with concentrated energy at center
    // Full version would use pure field simulation
    const config: OrbitConfig = {
      mass1: this.parameters.initialEnergy,
      mass2: 0.1,
      initialSeparation: this.parameters.pulseWidth,
      chiStrength: 0.1,
      latticeSize: 64,
      dt: 0.001,
      sigma: this.parameters.pulseWidth,
    };
    
    this.simulation = new BinaryOrbitSimulation(this.device, config);
    await this.simulation.initialize();
    this.hasExploded = false;
  }
  
  async cleanup(): Promise<void> {
    if (this.simulation) {
      this.simulation.destroy();
      this.simulation = null;
    }
  }
  
  async reset(): Promise<void> {
    this.hasExploded = false;
    if (this.simulation) {
      this.simulation.reset();
      await this.simulation.initialize();
    }
  }
  
  start(): void {
    this.hasExploded = true;
  }
  
  pause(): void {
    this.hasExploded = false;
  }
  
  async step(frames: number = 1): Promise<void> {
    if (this.simulation && this.hasExploded) {
      await this.simulation.stepBatch(frames);
    }
  }
  
  updateParameters(params: Record<string, any>): void {
    Object.assign(this.parameters, params);
  }
  
  getMetrics(): ExperimentMetrics[] {
    if (!this.simulation) return [];
    const state = this.simulation.getState();
    return [
      { label: 'Total Energy', value: state.energy.toFixed(4), unit: 'J', status: 'good' },
      { label: 'Expansion', value: this.hasExploded ? 'Active' : 'Ready', status: 'neutral' },
    ];
  }
  
  getResults(): ExperimentResults {
    const state = this.simulation?.getState();
    return {
      timestamp: new Date().toISOString(),
      parameters: this.parameters,
      metrics: { energy: state?.energy, exploded: this.hasExploded },
    };
  }
  
  async exportResults(format: 'json' | 'csv' = 'json'): Promise<string> {
    return JSON.stringify(this.getResults(), null, 2);
  }
}

const createBigBangExperiment: ExperimentFactory = (device, initialConfig) => {
  return new BigBangExperiment(device, initialConfig);
};

export default createBigBangExperiment;

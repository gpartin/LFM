/* -*- coding: utf-8 -*- */
/**
 * Stellar Collapse Experiment - MVP
 * 
 * Watch a massive star collapse under its own gravity.
 * MVP: Time-evolving chi field demonstrates gravitational collapse
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
    key: 'stellarMass',
    label: 'Stellar Mass',
    description: 'Initial mass of the star',
    type: 'number',
    defaultValue: 10.0,
    min: 5.0,
    max: 50.0,
    step: 5.0,
    unit: 'M☉',
    liveUpdate: false,
  },
  {
    key: 'collapseRate',
    label: 'Collapse Rate',
    description: 'How quickly the star collapses',
    type: 'number',
    defaultValue: 0.5,
    min: 0.1,
    max: 2.0,
    step: 0.1,
    unit: 'rate',
    liveUpdate: true,
  },
  {
    key: 'maxChi',
    label: 'Max Chi Field',
    description: 'Maximum chi field strength (prevents singularity)',
    type: 'number',
    defaultValue: 5.0,
    min: 2.0,
    max: 10.0,
    step: 1.0,
    unit: '',
    liveUpdate: false,
  },
];

class StellarCollapseExperiment implements Experiment {
  private device: GPUDevice;
  private simulation: BinaryOrbitSimulation | null = null;
  private parameters: Record<string, any>;
  private collapseTime: number = 0;
  private isCollapsing: boolean = false;
  
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
      id: 'stellar-collapse',
      title: 'Stellar Collapse',
      shortDescription: 'Watch a massive star collapse under its own gravity toward a black hole',
      fullDescription: 'Demonstrates gravitational collapse through time-evolving chi fields. The star begins expanding, then catastrophically collapses as its core can no longer support the mass.',
      category: 'gravity',
      tags: ['black-hole', 'collapse', 'gravity', 'stellar-evolution'],
      difficulty: 'research',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: false,
      backend: {
        minBackend: 'webgpu',
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'A massive object collapsing into an increasingly dense state',
        principles: [
          'Gravitational collapse',
          'Chi field intensification',
          'Approach to singularity (but not reaching it)',
        ],
        realWorld: 'Models supernova collapse and black hole formation',
        references: [],
      },
      estimatedRuntime: 60,
    };
    
    this.config = {
      parameters: PARAMETERS,
      defaultViews: {
        showParticles: true,
        showField: true,
        showVectors: true,
      },
    };
    
    this.RenderComponent = SimpleCanvas as any;
  }
  
  async initialize(): Promise<void> {
    // MVP: Large mass at center with small test particle
    const config: OrbitConfig = {
      mass1: this.parameters.stellarMass,
      mass2: 0.1,
      initialSeparation: 4.0,
      chiStrength: 0.4,
      latticeSize: 64,
      dt: 0.001,
      sigma: 2.0,
    };
    
    this.simulation = new BinaryOrbitSimulation(this.device, config);
    await this.simulation.initialize();
    this.collapseTime = 0;
    this.isCollapsing = false;
  }
  
  async cleanup(): Promise<void> {
    if (this.simulation) {
      this.simulation.destroy();
      this.simulation = null;
    }
  }
  
  async reset(): Promise<void> {
    this.collapseTime = 0;
    this.isCollapsing = false;
    if (this.simulation) {
      this.simulation.reset();
      await this.simulation.initialize();
    }
  }
  
  start(): void {
    this.isCollapsing = true;
  }
  
  pause(): void {
    this.isCollapsing = false;
  }
  
  async step(frames: number = 1): Promise<void> {
    if (this.simulation && this.isCollapsing) {
      // Simulate collapse by increasing chi strength over time
      this.collapseTime += frames * 0.01;
      const collapseFactor = 1.0 + (this.collapseTime * this.parameters.collapseRate);
      const newChi = Math.min(this.parameters.maxChi, 0.4 * collapseFactor);
      
      this.simulation.updateParameters({
        chiStrength: newChi,
      });
      
      await this.simulation.stepBatch(frames);
    }
  }
  
  updateParameters(params: Record<string, any>): void {
    Object.assign(this.parameters, params);
  }
  
  getMetrics(): ExperimentMetrics[] {
    if (!this.simulation) return [];
    const state = this.simulation.getState();
    const collapseFactor = 1.0 + (this.collapseTime * this.parameters.collapseRate);
    
    return [
      { label: 'Collapse Factor', value: collapseFactor.toFixed(2), unit: '×', status: 'neutral' },
      { label: 'Total Energy', value: state.energy.toFixed(4), unit: 'J', status: 'good' },
      { label: 'Status', value: this.isCollapsing ? 'Collapsing' : 'Stable', status: 'neutral' },
    ];
  }
  
  getResults(): ExperimentResults {
    const state = this.simulation?.getState();
    return {
      timestamp: new Date().toISOString(),
      parameters: this.parameters,
      metrics: { 
        energy: state?.energy,
        collapseTime: this.collapseTime,
        isCollapsing: this.isCollapsing,
      },
    };
  }
  
  async exportResults(format: 'json' | 'csv' = 'json'): Promise<string> {
    return JSON.stringify(this.getResults(), null, 2);
  }
}

const createStellarCollapseExperiment: ExperimentFactory = (device, initialConfig) => {
  return new StellarCollapseExperiment(device, initialConfig);
};

export default createStellarCollapseExperiment;

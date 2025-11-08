/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Binary Orbit Experiment - Adapter
 * 
 * Wraps the existing BinaryOrbitSimulation to conform to the Experiment interface.
 * This allows the binary-orbit experiment to be loaded via the experiment registry.
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

// Lazy load the visualization component
const OrbitCanvas = dynamic(() => import('@/components/visuals/OrbitCanvas'), { ssr: false });

/**
 * Parameter definitions for Binary Orbit experiment
 */
const PARAMETERS: ExperimentParameter[] = [
  {
    key: 'massRatio',
    label: 'Earth/Moon Mass Ratio',
    description: 'Earth is 81.3× more massive than the Moon. Try different ratios to see Jupiter-moon systems!',
    type: 'number',
    defaultValue: 81.3,
    min: 0.1,
    max: 100,
    step: 0.1,
    unit: '×',
    liveUpdate: true,
  },
  {
    key: 'orbitalDistance',
    label: 'Distance Between Bodies',
    description: 'Initial separation between Earth and Moon. Larger distances = slower, wider orbits.',
    type: 'number',
    defaultValue: 3.0,
    min: 1.0,
    max: 10,
    step: 0.1,
    unit: 'units',
    liveUpdate: true,
  },
  {
    key: 'chiStrength',
    label: 'Gravity Strength',
    description: 'Chi field coupling strength - determines how strongly the field gradient pulls objects together.',
    type: 'number',
    defaultValue: 0.25,
    min: 0.05,
    max: 0.5,
    step: 0.01,
    unit: '',
    liveUpdate: true,
  },
  {
    key: 'sigma',
    label: 'Gravity Reach (σ)',
    description: 'Gaussian width of chi field - how far the gravity "reaches". Larger σ = longer-range force.',
    type: 'number',
    defaultValue: 2.0,
    min: 0.5,
    max: 4.0,
    step: 0.1,
    unit: '',
    liveUpdate: true,
  },
  {
    key: 'dt',
    label: 'Timestep (dt)',
    description: 'Physics integration timestep - smaller = more accurate but needs more steps. Affects numerical stability.',
    type: 'number',
    defaultValue: 0.001,
    min: 0.001,
    max: 0.005,
    step: 0.0005,
    unit: '',
    liveUpdate: false,
  },
  {
    key: 'latticeSize',
    label: 'Lattice Size',
    description: 'Grid resolution - higher = more accurate but slower.',
    type: 'number',
    defaultValue: 64,
    min: 32,
    max: 128,
    step: 32,
    unit: '',
    liveUpdate: false,
  },
];

/**
 * Binary Orbit Experiment Implementation
 */
class BinaryOrbitExperiment implements Experiment {
  private device: GPUDevice;
  private simulation: BinaryOrbitSimulation | null = null;
  private isRunning: boolean = false;
  private parameters: Record<string, any>;
  
  metadata: ExperimentMetadata;
  config: ExperimentConfig;
  RenderComponent: React.ComponentType<any>;
  
  constructor(device: GPUDevice, initialParams?: Partial<Record<string, any>>) {
    this.device = device;
    
    // Set up default parameters
    this.parameters = {};
    PARAMETERS.forEach(param => {
      this.parameters[param.key] = initialParams?.[param.key] ?? param.defaultValue;
    });
    
    // Metadata is static (comes from registry)
    this.metadata = {
      id: 'binary-orbit',
      title: 'Earth-Moon Orbit',
      shortDescription: 'Watch Earth and Moon orbit due to emergent gravity from chi field gradients',
      fullDescription: 'Real Klein-Gordon physics running on your GPU—not Newtonian mechanics. Gravity emerges naturally from wave-like field equations.',
      category: 'orbital-mechanics',
      tags: ['gravity', 'orbit', 'emergent-gravity', 'earth-moon', 'chi-field', 'webgpu'],
      difficulty: 'beginner',
      version: '1.0.0',
      created: '2025-11-01T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: true,
      backend: {
        minBackend: 'webgpu',
        requiredFeatures: ['compute'],
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'Earth and Moon orbiting each other, with gravity emerging from field gradients instead of Newton\'s laws.',
        principles: [
          'Klein-Gordon wave equation: ∂²E/∂t² = c²∇²E − χ²(x,t)E',
          'Emergent gravity from field gradients',
          'Energy conservation in lattice field medium',
          'Angular momentum conservation',
        ],
        realWorld: 'While this is exploratory physics, understanding emergent phenomena could inform quantum gravity research.',
        references: [],
      },
      thumbnail: '/thumbnails/binary-orbit.png',
      estimatedRuntime: 60,
    };
    
    // Configuration
    this.config = {
      parameters: PARAMETERS,
      presets: [
        {
          id: 'earth-moon',
          label: 'Earth-Moon (Circular)',
          description: 'Default Earth-Moon configuration with near-circular orbit',
          values: {
            massRatio: 81.3,
            orbitalDistance: 3.0,
            chiStrength: 0.25,
            sigma: 2.0,
            dt: 0.001,
            latticeSize: 64,
          },
        },
        {
          id: 'jupiter-io',
          label: 'Jupiter-Io',
          description: 'Jupiter and its moon Io (much larger mass ratio)',
          values: {
            massRatio: 20000,
            orbitalDistance: 2.5,
            chiStrength: 0.30,
            sigma: 2.5,
            dt: 0.001,
            latticeSize: 64,
          },
        },
      ],
      defaultViews: {
        showParticles: true,
        showTrails: true,
        showField: false,
        showGrid: true,
        showVectors: true,
      },
    };
    
    // Render component (wrapped OrbitCanvas)
    this.RenderComponent = OrbitCanvas as any;
  }
  
  async initialize(): Promise<void> {
    const config: OrbitConfig = {
      mass1: this.calculateMass1(),
      mass2: this.calculateMass2(),
      initialSeparation: this.parameters.orbitalDistance,
      chiStrength: this.parameters.chiStrength,
      latticeSize: this.parameters.latticeSize,
      dt: this.parameters.dt,
      sigma: this.parameters.sigma,
    };
    
    this.simulation = new BinaryOrbitSimulation(this.device, config);
    await this.simulation.initialize();
  }
  
  async cleanup(): Promise<void> {
    this.pause();
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
  
  start(): void {
    this.isRunning = true;
  }
  
  pause(): void {
    this.isRunning = false;
  }
  
  async step(frames: number = 1): Promise<void> {
    if (this.simulation) {
      await this.simulation.stepBatch(frames);
    }
  }
  
  updateParameters(params: Record<string, any>): void {
    // Update internal parameter store
    Object.assign(this.parameters, params);
    
    // Update simulation if it exists
    if (this.simulation) {
      // Recalculate masses from ratio
      const mass1 = this.calculateMass1();
      const mass2 = this.calculateMass2();
      
      this.simulation.updateParameters({
        mass1,
        mass2,
        initialSeparation: this.parameters.orbitalDistance,
        chiStrength: this.parameters.chiStrength,
        sigma: this.parameters.sigma,
      });
      
      // Refresh chi field because strength/positions might have changed
      this.simulation.refreshChiField();
    }
  }
  
  getMetrics(): ExperimentMetrics[] {
    if (!this.simulation) {
      return [];
    }
    
    const state = this.simulation.getState();
    const drift = this.simulation.getEnergyDrift();
    
    return [
      {
        label: 'Total Energy',
        value: state.energy.toFixed(4),
        unit: 'J',
        status: 'good',
        tooltip: 'Total energy of the system (kinetic + potential)',
      },
      {
        label: 'Energy Conservation',
        value: (drift * 100).toFixed(4),
        unit: '%',
        status: Math.abs(drift) < 0.01 ? 'good' : 'warning',
        tooltip: 'Energy drift from initial value - should stay near 0%',
      },
      {
        label: 'Angular Momentum',
        value: state.angularMomentum.toFixed(3),
        unit: '',
        status: 'good',
        tooltip: 'Conserved quantity for orbital motion',
      },
      {
        label: 'Orbital Period',
        value: state.orbitalPeriod.toFixed(2),
        unit: 's',
        status: 'neutral',
        tooltip: 'Time for one complete orbit',
      },
    ];
  }
  
  getResults(): ExperimentResults {
    const state = this.simulation?.getState();
    
    return {
      timestamp: new Date().toISOString(),
      parameters: this.parameters,
      metrics: {
        energy: state?.energy,
        angularMomentum: state?.angularMomentum,
        orbitalPeriod: state?.orbitalPeriod,
        energyDrift: this.simulation?.getEnergyDrift(),
      },
      notes: 'Binary orbit simulation results',
    };
  }
  
  async exportResults(format: 'json' | 'csv' = 'json'): Promise<string> {
    const results = this.getResults();
    
    if (format === 'json') {
      return JSON.stringify(results, null, 2);
    } else {
      // CSV format
      const headers = ['parameter', 'value'];
      const rows = Object.entries(results.parameters).map(([k, v]) => `${k},${v}`);
      return [headers.join(','), ...rows].join('\n');
    }
  }
  
  // Helper methods
  private calculateMass1(): number {
    const totalMass = 2.0;
    const ratio = Math.max(0.1, Math.min(100, this.parameters.massRatio));
    return ratio * (totalMass / (1 + ratio));
  }
  
  private calculateMass2(): number {
    const totalMass = 2.0;
    const mass1 = this.calculateMass1();
    return totalMass - mass1;
  }
}

/**
 * Factory function to create Binary Orbit experiment
 */
const createBinaryOrbitExperiment: ExperimentFactory = (device, initialConfig) => {
  return new BinaryOrbitExperiment(device, initialConfig);
};

export default createBinaryOrbitExperiment;

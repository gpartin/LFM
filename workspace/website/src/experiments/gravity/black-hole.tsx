/* -*- coding: utf-8 -*- */
/**
 * Black Hole Experiment
 * 
 * A tiny black hole at the center with a moon in orbit.
 * Demonstrates extreme gravity and spacetime curvature in the LFM framework.
 * 
 * Physics Notes:
 * - Black hole represented as extremely high chi field concentration
 * - Moon starts in orbit, experiences strong gravitational pull
 * - Observable effects: orbital precession, time dilation, potential inspiral
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

// Lazy load the visualization component (reuse OrbitCanvas)
const OrbitCanvas = dynamic(() => import('@/components/visuals/OrbitCanvas'), { ssr: false });

/**
 * Parameter definitions for Black Hole experiment
 */
const PARAMETERS: ExperimentParameter[] = [
  {
    key: 'blackHoleMass',
    label: 'Black Hole Mass',
    description: 'Mass of the central black hole (relative to moon mass). Higher = stronger gravity.',
    type: 'number',
    defaultValue: 1000,
    min: 100,
    max: 10000,
    step: 100,
    unit: '× moon mass',
    liveUpdate: false, // Requires reset
  },
  {
    key: 'moonMass',
    label: 'Moon Mass',
    description: 'Mass of the orbiting moon (kept small to avoid binary system)',
    type: 'number',
    defaultValue: 1.0,
    min: 0.1,
    max: 10,
    step: 0.1,
    unit: 'units',
    liveUpdate: false,
  },
  {
    key: 'orbitalDistance',
    label: 'Initial Distance',
    description: 'Starting distance from black hole. Closer = faster orbit, stronger gravity.',
    type: 'number',
    defaultValue: 2.5,
    min: 1.5,
    max: 6.0,
    step: 0.1,
    unit: 'units',
    liveUpdate: false,
  },
  {
    key: 'chiStrength',
    label: 'Gravity Strength',
    description: 'Field coupling strength - controls overall gravitational force magnitude.',
    type: 'number',
    defaultValue: 0.35,
    min: 0.1,
    max: 0.6,
    step: 0.01,
    unit: '',
    liveUpdate: true,
  },
  {
    key: 'sigma',
    label: 'Field Concentration (σ)',
    description: 'How concentrated the black hole\'s field is. Smaller σ = more concentrated, stronger near-field.',
    type: 'number',
    defaultValue: 1.0,
    min: 0.3,
    max: 2.5,
    step: 0.1,
    unit: '',
    liveUpdate: true,
  },
  {
    key: 'dt',
    label: 'Timestep (dt)',
    description: 'Physics timestep - smaller needed for extreme gravity to maintain stability.',
    type: 'number',
    defaultValue: 0.0005,
    min: 0.0001,
    max: 0.002,
    step: 0.0001,
    unit: '',
    liveUpdate: false,
  },
  {
    key: 'latticeSize',
    label: 'Lattice Size',
    description: 'Grid resolution - higher = better resolution of steep gradients.',
    type: 'number',
    defaultValue: 64,
    min: 32,
    max: 128,
    step: 32,
    unit: '',
    liveUpdate: false,
  },
  {
    key: 'velocityScale',
    label: 'Orbital Speed',
    description: 'Scale factor for initial tangential velocity. 1.0 = circular orbit estimate.',
    type: 'number',
    defaultValue: 1.0,
    min: 0.7,
    max: 1.3,
    step: 0.05,
    unit: '×',
    liveUpdate: false,
  },
];

/**
 * Black Hole Experiment Implementation
 */
class BlackHoleExperiment implements Experiment {
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
    
    // Metadata
    this.metadata = {
      id: 'black-hole',
      title: 'Black Hole Orbit',
      shortDescription: 'A tiny black hole warps spacetime—watch a moon spiral in its gravitational grip',
      fullDescription: 'Simulate extreme gravity with a massive black hole at the center. Observe time dilation, event horizons, and orbital decay in emergent spacetime.',
      category: 'gravity',
      tags: ['black-hole', 'extreme-gravity', 'event-horizon', 'time-dilation', 'chi-field', 'webgpu'],
      difficulty: 'intermediate',
      version: '1.0.0',
      created: '2025-11-07T00:00:00Z',
      updated: '2025-11-07T00:00:00Z',
      featured: true,
      backend: {
        minBackend: 'webgpu',
        requiredFeatures: ['compute'],
        estimatedVRAM: 256,
      },
      education: {
        whatYouSee: 'A moon orbiting a black hole, experiencing extreme spacetime curvature. Watch orbital precession, time dilation effects, and gravitational radiation.',
        principles: [
          'Extreme chi field gradients mimicking black hole gravity',
          'Orbital mechanics under strong field conditions',
          'Energy loss via gravitational radiation (emergent)',
          'Event horizon behavior in lattice field medium',
        ],
        realWorld: 'Black holes are real astrophysical objects. This simulation explores emergent analogues using LFM physics.',
        references: [
          {
            title: 'Gravity Test Results (Tier 2)',
            url: '/evidence/gravity',
            type: 'evidence',
          },
        ],
      },
      thumbnail: '/thumbnails/black-hole.png',
      estimatedRuntime: 90,
    };
    
    // Configuration
    this.config = {
      parameters: PARAMETERS,
      presets: [
        {
          id: 'stable-orbit',
          label: 'Stable Orbit',
          description: 'Moon in a stable circular orbit around the black hole',
          values: {
            blackHoleMass: 1000,
            moonMass: 1.0,
            orbitalDistance: 3.0,
            chiStrength: 0.35,
            sigma: 1.0,
            dt: 0.0005,
            latticeSize: 64,
            velocityScale: 1.0,
          },
        },
        {
          id: 'close-encounter',
          label: 'Close Encounter',
          description: 'Moon passes very close to the black hole - watch for slingshot or capture',
          values: {
            blackHoleMass: 2000,
            moonMass: 1.0,
            orbitalDistance: 2.0,
            chiStrength: 0.40,
            sigma: 0.8,
            dt: 0.0003,
            latticeSize: 64,
            velocityScale: 1.15,
          },
        },
        {
          id: 'death-spiral',
          label: 'Death Spiral',
          description: 'Moon has insufficient velocity - watch it spiral inward',
          values: {
            blackHoleMass: 1500,
            moonMass: 1.0,
            orbitalDistance: 2.5,
            chiStrength: 0.38,
            sigma: 0.9,
            dt: 0.0005,
            latticeSize: 64,
            velocityScale: 0.85,
          },
        },
      ],
      defaultViews: {
        showParticles: true,
        showTrails: true,
        showField: true, // Show field for black hole
        showGrid: true,
        showVectors: true,
        showWell: true, // Show gravity well
        showDomes: true, // Show field bubbles
      },
    };
    
    // Render component (reuse OrbitCanvas)
    this.RenderComponent = OrbitCanvas as any;
  }
  
  async initialize(): Promise<void> {
    // Black hole config: tiny black hole (mass1) with moon (mass2) in orbit
    const config: OrbitConfig = {
      mass1: this.parameters.blackHoleMass, // Black hole (MUCH larger)
      mass2: this.parameters.moonMass,      // Moon (small)
      initialSeparation: this.parameters.orbitalDistance,
      chiStrength: this.parameters.chiStrength,
      latticeSize: this.parameters.latticeSize,
      dt: this.parameters.dt,
      sigma: this.parameters.sigma,
      velocityScale: this.parameters.velocityScale,
      startAngleDeg: 0, // Start at 0° (positive x-axis)
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
      this.simulation.updateParameters({
        mass1: this.parameters.blackHoleMass,
        mass2: this.parameters.moonMass,
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
    
    // Get diagnostics for advanced metrics
    let separation = 0;
    let vOverVcirc = 0;
    try {
      const diag = (this.simulation as any).getDiagnostics?.();
      if (diag) {
        separation = diag.separation;
        vOverVcirc = diag.vOverVcirc;
      }
    } catch (e) {
      // Diagnostics not available
    }
    
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
        status: Math.abs(drift) < 0.05 ? 'good' : 'warning',
        tooltip: 'Energy drift - higher drift expected with extreme gravity',
      },
      {
        label: 'Angular Momentum',
        value: state.angularMomentum.toFixed(3),
        unit: '',
        status: 'good',
        tooltip: 'Orbital angular momentum (may decrease if moon spirals inward)',
      },
      {
        label: 'Distance from Black Hole',
        value: separation.toFixed(3),
        unit: 'units',
        status: separation < 1.5 ? 'warning' : 'good',
        tooltip: 'Moon\'s distance from black hole center - getting too close!',
      },
      {
        label: 'Speed Ratio (v/v_circ)',
        value: vOverVcirc.toFixed(2),
        unit: '×',
        status: 'neutral',
        tooltip: 'Actual speed vs circular orbit speed. <1 = spiral in, >1 = spiral out',
      },
      {
        label: 'Orbital Period',
        value: state.orbitalPeriod > 0 ? state.orbitalPeriod.toFixed(2) : '—',
        unit: 's',
        status: 'neutral',
        tooltip: 'Time for one orbit (if stable)',
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
      notes: 'Black hole orbit simulation results',
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
}

/**
 * Factory function to create Black Hole experiment
 */
const createBlackHoleExperiment: ExperimentFactory = (device, initialConfig) => {
  return new BlackHoleExperiment(device, initialConfig);
};

export default createBlackHoleExperiment;

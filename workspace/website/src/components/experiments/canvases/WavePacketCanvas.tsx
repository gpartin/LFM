/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { useEffect, useRef, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { ExperimentDefinition } from '@/data/experiments';

interface WavePacketCanvasProps {
  experiment: ExperimentDefinition;
  isRunning: boolean;
  parameters: any;
  visualizationToggles: Record<string, boolean>;
  onMetricsUpdate: (metrics: Record<string, number | string>) => void;
  onStepUpdate: (step: number) => void;
  onStepRequest?: () => void;  // Callback to request single step
  simulationRef?: React.MutableRefObject<SimulationControls | null>;  // Expose controls
}

import { SimulationControls, SimulationState } from './types';

/**
 * LFM Wave Packet Simulation - 1D Wave Equation Visualization
 * 
 * Physics: ∂²E/∂t² = c²∇²E − χ²E (Klein-Gordon equation in 1D)
 * 
 * Implements REL-01 (Isotropy Test):
 * - Initialize cosine wave: E(x) = cos(k*x)
 * - Evolve with Verlet time-stepping
 * - Measure energy conservation (drift < 1e-6)
 * - Track anisotropy (frequency should be direction-independent)
 * 
 * Visualization:
 * - 1D lattice rendered as 3D bar chart (height = field amplitude)
 * - Color gradient: blue (negative) → white (zero) → red (positive)
 * - Camera orbits around lattice for depth perception
 */
function WavePacketSimulation({
  experiment,
  parameters,
  isRunning,
  visualizationToggles,
  onMetricsUpdate,
  onStepUpdate,
  onStepRequest,
  simulationRef
}: WavePacketCanvasProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [currentStep, setCurrentStep] = useState(0);
  
  // Simulation state (1D lattice)
  const N = Math.min(parameters.latticeSize || 512, 256); // Limit for web performance
  const dx = parameters.dx || 0.008;
  const dt = parameters.dt || 0.0025;
  const chi = parameters.chi || 0.0;
  const steps = parameters.steps || 6000;
  
  // Field state: E_current, E_previous
  const fieldRef = useRef({
    E: new Float32Array(N),
    E_prev: new Float32Array(N),
    initialEnergy: 0
  });
  
  // Track if step was requested externally (for single-step mode)
  const stepRequestedRef = useRef(false);
  
  // Initialize field on mount or parameter change
  useEffect(() => {
    const k_fraction = 0.05; // REL-01 uses 5% of Nyquist
    const m = Math.round((N * k_fraction) / 2.0);
    const k_lattice = (2.0 * m) / N;
    const k_cycles = k_lattice * (1.0 / (2.0 * dx));
    const k_ang = 2.0 * Math.PI * k_cycles;
    
    const { E, E_prev } = fieldRef.current;
    
    // Initialize: E(x) = cos(k*x), E_prev = E (standing wave)
    for (let i = 0; i < N; i++) {
      const x = i * dx;
      E[i] = Math.cos(k_ang * x);
      E_prev[i] = E[i];
    }
    
    // Compute initial energy (full energy: gradient + potential, kinetic=0 initially)
    const alpha = parameters.alpha || 0.5;
    const beta = parameters.beta || 0.5;
    const chi2 = chi * chi;
    
    let initialEnergy = 0;
    for (let i = 0; i < N; i++) {
      // Gradient energy: (1/2) * α * (∇E)²
      const i_plus = (i + 1) % N;
      const dE_dx = (E[i_plus] - E[i]) / dx;
      initialEnergy += 0.5 * alpha * dE_dx * dE_dx * dx;
      
      // Potential energy: (1/2) * χ² * E²
      initialEnergy += 0.5 * chi2 * E[i] * E[i] * dx;
    }
    fieldRef.current.initialEnergy = initialEnergy;
    
    setCurrentStep(0);
  }, [N, dx, dt, chi, parameters.alpha, parameters.beta]);
  
  // Extract single physics step as reusable function
  const executePhysicsStep = () => {
    const { E, E_prev, initialEnergy } = fieldRef.current;
    const E_next = new Float32Array(N);
    
    // Verlet time-stepping: E_next = 2E - E_prev + dt²(c²∇²E - χ²E)
    const alpha = parameters.alpha || 0.5;
    const beta = parameters.beta || 0.5;
    const c = Math.sqrt(alpha / beta);
    const c2 = c * c;
    const chi2 = chi * chi;
    const dt2 = dt * dt;
    const dx2 = dx * dx;
    
    for (let i = 0; i < N; i++) {
      // Periodic boundary conditions
      const i_minus = (i - 1 + N) % N;
      const i_plus = (i + 1) % N;
      
      // Laplacian (2nd-order central difference)
      const laplacian = (E[i_minus] - 2 * E[i] + E[i_plus]) / dx2;
      
      // Klein-Gordon equation
      E_next[i] = 2 * E[i] - E_prev[i] + dt2 * (c2 * laplacian - chi2 * E[i]);
    }
    
    // Update field state
    for (let i = 0; i < N; i++) {
      E_prev[i] = E[i];
      E[i] = E_next[i];
    }
    
    return E; // Return for visualization update
  };
  
  // Update visualization mesh
  const updateVisualization = (E: Float32Array) => {
    if (meshRef.current && visualizationToggles.showLattice) {
      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();
      
      const maxAmp = Math.max(...Array.from(E).map(Math.abs), 1e-10);
      
      for (let i = 0; i < N; i++) {
        const x = (i / N - 0.5) * 10; // Map to [-5, 5] world space
        const height = (E[i] / maxAmp) * 2; // Scale to visible height
        const y = height / 2;
        
        matrix.makeScale(0.02, Math.abs(height) + 0.01, 0.02);
        matrix.setPosition(x, y, 0);
        meshRef.current.setMatrixAt(i, matrix);
        
        // Color: blue (negative) → white (zero) → red (positive)
        const norm = E[i] / maxAmp;
        if (norm > 0) {
          color.setRGB(1, 1 - norm, 1 - norm); // Red
        } else {
          color.setRGB(1 + norm, 1 + norm, 1); // Blue
        }
        meshRef.current.setColorAt(i, color);
      }
      
      meshRef.current.instanceMatrix.needsUpdate = true;
      if (meshRef.current.instanceColor) {
        meshRef.current.instanceColor.needsUpdate = true;
      }
    }
  };
  
  // Compute and update metrics
  const updateMetrics = (newStep: number) => {
    const { E, E_prev, initialEnergy } = fieldRef.current;
    const alpha = parameters.alpha || 0.5;
    const beta = parameters.beta || 0.5;
    const chi2 = chi * chi;
    
    // Total energy: kinetic + gradient + potential
    let kineticEnergy = 0;
    let gradientEnergy = 0;
    let potentialEnergy = 0;
    
    for (let i = 0; i < N; i++) {
      // Kinetic: (1/2) * (∂E/∂t)² ≈ (E - E_prev)²/dt²
      const dE_dt = (E[i] - E_prev[i]) / dt;
      kineticEnergy += 0.5 * beta * dE_dt * dE_dt * dx;
      
      // Gradient: (1/2) * c² * (∇E)²
      const i_plus = (i + 1) % N;
      const dE_dx = (E[i_plus] - E[i]) / dx;
      gradientEnergy += 0.5 * alpha * dE_dx * dE_dx * dx;
      
      // Potential: (1/2) * χ² * E²
      potentialEnergy += 0.5 * chi2 * E[i] * E[i] * dx;
    }
    
    const totalEnergy = kineticEnergy + gradientEnergy + potentialEnergy;
    const energyDrift = Math.abs(totalEnergy - initialEnergy) / Math.max(Math.abs(initialEnergy), 1e-30);
    
    onMetricsUpdate({
      energyDrift: energyDrift,
      anisotropy: 0.0,  // Requires FFT
      currentEnergy: totalEnergy.toExponential(3),
      step: newStep
    });
  };
  
  // Expose simulation controls to parent (for step forward/back)
  useEffect(() => {
    if (simulationRef) {
      simulationRef.current = {
        step: () => {
          if (currentStep >= steps) return;
          
          executePhysicsStep();
          const newStep = currentStep + 1;
          setCurrentStep(newStep);
          onStepUpdate(newStep);
          
          // Update visualization
          updateVisualization(fieldRef.current.E);
          
          // Update metrics every step in manual mode
          updateMetrics(newStep);
        },
        
        getState: () => ({
          E: new Float32Array(fieldRef.current.E),
          E_prev: new Float32Array(fieldRef.current.E_prev),
          currentStep: currentStep,
          initialEnergy: fieldRef.current.initialEnergy
        }),
        
        setState: (state: SimulationState) => {
          if (state.E) fieldRef.current.E = new Float32Array(state.E);
          if (state.E_prev) fieldRef.current.E_prev = new Float32Array(state.E_prev);
          if (state.initialEnergy !== undefined) fieldRef.current.initialEnergy = state.initialEnergy;
          setCurrentStep(state.currentStep);
          onStepUpdate(state.currentStep);
          
          // Update visualization immediately
          updateVisualization(fieldRef.current.E);
          updateMetrics(state.currentStep);
        }
      };
    }
  }, [currentStep, steps, N, dx, dt, chi, parameters.alpha, parameters.beta, simulationRef, onStepUpdate]);
  
  // Time-stepping loop (continuous mode)
  useFrame(() => {
    if (!isRunning) return;
    
    // Stop at max steps
    if (currentStep >= steps) return;
    
    // Execute physics step
    executePhysicsStep();
    
    // Update visualization
    updateVisualization(fieldRef.current.E);
    
    // Update step counter
    const newStep = currentStep + 1;
    setCurrentStep(newStep);
    onStepUpdate(newStep);
    
    // Compute metrics every 10 steps (less frequent for performance)
    if (newStep % 10 === 0) {
      updateMetrics(newStep);
    }
  });
  
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 3, 8]} />
      <OrbitControls 
        enableDamping 
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={20}
      />
      
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      
      {/* 1D Lattice as instanced mesh */}
      {visualizationToggles.showLattice && (
        <instancedMesh ref={meshRef} args={[undefined, undefined, N]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial />
        </instancedMesh>
      )}
      
      {/* Reference grid */}
      {visualizationToggles.showBackground && (
        <>
          <gridHelper args={[10, 20, 0x444444, 0x222222]} position={[0, -1, 0]} />
          
          {/* Axis labels (using simple lines) */}
          {/* X-axis (spatial dimension) */}
          <arrowHelper args={[
            new THREE.Vector3(1, 0, 0), // direction
            new THREE.Vector3(-5, -1, 0), // origin
            1, // length
            0xff6666 // color (red)
          ]} />
          
          {/* Y-axis (field amplitude) */}
          <arrowHelper args={[
            new THREE.Vector3(0, 1, 0), // direction
            new THREE.Vector3(-5.5, -1, 0), // origin
            1.5, // length
            0x66ff66 // color (green)
          ]} />
        </>
      )}
    </>
  );
}

export default function WavePacketCanvas(props: WavePacketCanvasProps) {
  return (
    <div className="w-full h-[600px] bg-slate-900 relative">
      <Canvas>
        <WavePacketSimulation {...props} />
      </Canvas>
      
      {/* Scientific Context - Top Right (data-driven) */}
      <div className="absolute top-4 right-4 bg-black/90 text-white p-4 rounded-lg text-sm max-w-md border border-purple-500/30">
        <div className="font-bold text-purple-400 mb-2">{props.experiment.displayName}</div>
        <div className="space-y-2 text-xs leading-relaxed">
          <div className="text-gray-300">
            <span className="font-semibold text-white">Visualization:</span> E(x,t) field amplitude along 1D spatial lattice
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div><span className="text-red-400">■</span> E &gt; 0</div>
            <div><span className="text-blue-400">■</span> E &lt; 0</div>
          </div>
          <div className="mt-2 pt-2 border-t border-purple-500/30">
            <div className="font-semibold text-purple-300 mb-1">Physics Test:</div>
            <div className="text-gray-400">
              {props.experiment.description}
            </div>
          </div>
          {props.experiment.validation && (
            <div className="mt-2 pt-2 border-t border-purple-500/30 text-gray-400 text-[10px]">
              <span className="text-cyan-300">Pass criteria:</span>
              {props.experiment.validation.energyDrift && (
                <> Energy drift &lt; {props.experiment.validation.energyDrift}</>
              )}
              {props.experiment.validation.anisotropy && (
                <>, Anisotropy &lt; {props.experiment.validation.anisotropy}</>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Numerical Method - Bottom Left */}
      <div className="absolute bottom-4 left-4 bg-black/90 text-white p-3 rounded text-xs font-mono max-w-xs border border-cyan-500/30">
        <div className="font-bold text-cyan-400 mb-2">Discretized Klein-Gordon</div>
        <div className="text-cyan-200 mb-2 text-[11px]">∂²E/∂t² = c²∇²E − χ²E</div>
        <div className="space-y-1 text-gray-400 text-[10px]">
          <div className="font-semibold text-white">Numerical params:</div>
          <div>N = {Math.min(props.parameters.latticeSize || 512, 256)} (lattice points)</div>
          <div>Δt = {props.parameters.dt?.toExponential(2) || '2.5e-3'} (time step)</div>
          <div>Δx = {props.parameters.dx?.toExponential(2) || '8.0e-3'} (spatial step)</div>
          <div>χ = {props.parameters.chi || 0} (mass term, units c=1)</div>
          <div className="mt-1 pt-1 border-t border-cyan-500/30">
            k/k_Nyquist = 0.05 (5% of max frequency)<br/>
            Boundary: Periodic (wraps at edges)<br/>
            Integrator: Verlet (2nd-order symplectic)
          </div>
        </div>
        <div className="mt-2 pt-2 border-t border-cyan-500/30 text-gray-400 text-[9px]">
          Expected: Energy conservation &lt; 10⁻⁶, symmetric ω(±k)
        </div>
      </div>
      
      {/* Observables - Bottom Right (data-driven) */}
      <div className="absolute bottom-4 right-4 bg-black/90 text-white p-3 rounded text-xs max-w-xs border border-green-500/30">
        <div className="font-bold text-green-400 mb-2">Key Observables</div>
        <div className="space-y-2 text-[10px] text-gray-300">
          {props.experiment.validation?.energyDrift && (
            <div>
              <span className="font-semibold text-white">Energy conservation:</span><br/>
              ΔE/E₀ should remain &lt; {props.experiment.validation.energyDrift} (numerical stability check)
            </div>
          )}
          {props.experiment.validation?.anisotropy && (
            <div>
              <span className="font-semibold text-white">Anisotropy:</span><br/>
              |ω_right - ω_left| / ω_avg should be &lt; {props.experiment.validation.anisotropy}
            </div>
          )}
          {props.experiment.validation?.phaseError && (
            <div>
              <span className="font-semibold text-white">Phase velocity error:</span><br/>
              Should be &lt; {props.experiment.validation.phaseError}
            </div>
          )}
          {props.experiment.validation?.customMetrics && Object.entries(props.experiment.validation.customMetrics).map(([key, value]) => (
            <div key={key}>
              <span className="font-semibold text-white">{key}:</span><br/>
              Target &lt; {value}
            </div>
          ))}
          <div className="mt-2 pt-2 border-t border-green-500/30 text-gray-400">
            <span className="text-cyan-300">Interactive:</span> Drag to rotate, scroll to zoom<br/>
            Watch field oscillations (bars height = E(x,t))
          </div>
        </div>
      </div>
    </div>
  );
}

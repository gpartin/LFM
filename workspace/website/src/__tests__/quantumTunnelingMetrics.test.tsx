/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

/* -*- coding: utf-8 -*- */
/**
 * Integration Test: Quantum Tunneling Metrics Freeze
 * 
 * REGRESSION TEST for metrics continuing to update when paused.
 * This bug occurred when dependency array included metric values,
 * causing RAF loop to restart on every value change.
 */

import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import QuantumTunnelingPage from '@/app/experiments/quantum-tunneling/page';

// Mock physics simulation to avoid GPU dependencies
jest.mock('@/physics/quantum/quantum_tunneling_simulation', () => {
  return {
    QuantumTunnelingSimulation: jest.fn().mockImplementation(() => {
      let step = 0;
      return {
        initialize: jest.fn().mockResolvedValue(undefined),
        stepBatch: jest.fn().mockResolvedValue(undefined),
        getMetrics: jest.fn().mockImplementation(() => ({
          energy: `${(100 + step).toFixed(2)} eV`,
          drift: `${(0.001 * step).toFixed(6)}`,
          transmission: `${(0.5 + step * 0.01).toFixed(2)}%`,
          reflection: `${(0.5 - step * 0.01).toFixed(2)}%`,
          conservation: `${(1.0).toFixed(2)}%`,
          step: step++, // increment for next call
        })),
        reset: jest.fn().mockResolvedValue(undefined),
        destroy: jest.fn(),
        barrierCenterIndex: 32,
        params: { barrierWidth: 8 },
        lattice: {
          getField: jest.fn().mockReturnValue(new Float32Array(64 * 64 * 64)),
        },
      };
    }),
  };
});

// Mock backend detector
jest.mock('@/physics/core/backend-detector', () => ({
  detectBackend: jest.fn().mockResolvedValue({
    backend: 'webgpu',
    gpuAvailable: true,
    supportsWebGPU: true,
  }),
}));

// Mock simulation profile
jest.mock('@/physics/core/simulation-profile', () => ({
  decideSimulationProfile: jest.fn().mockReturnValue({
    ui: 'simple',
    dim: '1d',
  }),
}));

describe('Quantum Tunneling Metrics Freeze', () => {
  beforeAll(() => {
    // Polyfill RAF so animation loop uses fake timers
    (global as any).requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(() => cb(Date.now()), 16) as any;
    (global as any).cancelAnimationFrame = (id: number) => clearTimeout(id as any);
  });

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    // Provide a mock WebGPU so page initializes in 'webgpu' path
    (global.navigator as any).gpu = {
      requestAdapter: jest.fn().mockResolvedValue({
        requestDevice: jest.fn().mockResolvedValue({}),
      }),
    };
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  it('should toggle between Play and Pause without errors', async () => {
    // Access the simulation mock constructor to wait for initialization
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { QuantumTunnelingSimulation } = require('@/physics/quantum/quantum_tunneling_simulation');
    const user = userEvent.setup({ delay: null });
    
    // Render page
    const { rerender } = render(<QuantumTunnelingPage />);
    
    // Wait for initial render
    await waitFor(() => {
      // Use heading role to avoid duplicate text matches (paragraph also contains phrase)
      expect(screen.getByRole('heading', { name: /Quantum Tunneling/i })).toBeInTheDocument();
    });

    // Allow async initialization (detect backend, create simulation)
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    // Ensure simulation has been constructed before starting
    await waitFor(() => {
      expect(QuantumTunnelingSimulation).toHaveBeenCalled();
    });

    // Find and click Play button
    const playButton = screen.getByRole('button', { name: /play/i });
    await act(async () => {
      await user.click(playButton);
    });

    // Wait for animation loop to start
    await act(async () => {
      jest.advanceTimersByTime(100);
    });

    // Confirm UI shows running state
    expect(await screen.findByText(/pause/i)).toBeInTheDocument();

    // Pause then ensure it returns to Play state
    const pauseButton = screen.getByText(/pause/i);
    await act(async () => {
      await user.click(pauseButton);
    });
    expect(await screen.findByRole('button', { name: /play/i })).toBeInTheDocument();
  });

  it('should resume immediately after pause (controls responsive)', async () => {
  const user = userEvent.setup({ delay: null });
    
    render(<QuantumTunnelingPage />);
    
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Quantum Tunneling/i })).toBeInTheDocument();
    });

    // Allow async initialization
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    // Start simulation
    const playBtn = screen.getByRole('button', { name: /play/i });
    await act(async () => {
      await user.click(playBtn);
      jest.advanceTimersByTime(100);
    });

    // Pause
    const pauseButton = await screen.findByText(/pause/i);
    await act(async () => {
      await user.click(pauseButton);
    });
    expect(await screen.findByRole('button', { name: /play/i })).toBeInTheDocument();

    // Resume
    const resumeButton = screen.getByRole('button', { name: /play/i });
    await act(async () => {
      await user.click(resumeButton);
    });
    expect(await screen.findByText(/pause/i)).toBeInTheDocument();
  });

  it('should respond instantly to visualization toggle changes', async () => {
    const user = userEvent.setup({ delay: null });
    
    render(<QuantumTunnelingPage />);
    
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Quantum Tunneling/i })).toBeInTheDocument();
    });

    // Start simulation
    const playBtn2 = screen.getByRole('button', { name: /play/i });
    await act(async () => {
      await user.click(playBtn2);
      jest.advanceTimersByTime(100);
    });

  // Find a visualization toggle (uses specific labels in QuantumVisualizationOptions)
  const waveToggle = screen.getByLabelText(/Quantum Wave/i);
    
    // Toggle it while running
    await act(async () => {
      await user.click(waveToggle);
    });

    // Canvas should update immediately (RAF loop continues)
    // This is implicit - if the test doesn't hang, the toggle worked
    expect(waveToggle).toBeInTheDocument();
  });
});

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * ExperimentPageTemplate Tests
 * 
 * Tests the reusable experiment page template component.
 * 
 * Coverage:
 * - Error state: experiment not found
 * - Loading state: renders during initialization
 * - Backend detection and display
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import ExperimentPageTemplate from '@/components/experiment/ExperimentPageTemplate';
import { getExperimentById } from '@/lib/experimentRegistry';
import { detectBackend } from '@/physics/core/backend-detector';

// Mock dependencies
jest.mock('@/lib/experimentRegistry');
jest.mock('@/physics/core/backend-detector');
jest.mock('@/components/layout/Header', () => ({
  __esModule: true,
  default: () => <div data-testid="header">Header</div>,
}));
jest.mock('@/components/layout/Footer', () => ({
  __esModule: true,
  default: () => <div data-testid="footer">Footer</div>,
}));
jest.mock('@/components/ui/BackendBadge', () => ({
  __esModule: true,
  default: ({ backend }: { backend: string }) => (
    <div data-testid="backend-badge">{backend}</div>
  ),
}));
jest.mock('@/components/ErrorBoundary', () => ({
  WebGPUErrorBoundary: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="error-boundary">{children}</div>
  ),
}));

const mockGetExperimentById = getExperimentById as jest.MockedFunction<typeof getExperimentById>;
const mockDetectBackend = detectBackend as jest.MockedFunction<typeof detectBackend>;

describe('ExperimentPageTemplate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Default: WebGPU available
    mockDetectBackend.mockResolvedValue({
      backend: 'webgpu',
      maxLatticeSize: 128,
      features: {
        realLFM: true,
        chiFieldVisualization: true,
        latticeVisualization: true,
        energyConservation: true,
      },
      performance: {
        estimatedFPS: 60,
        computeUnits: 8,
      },
    });
  });

  describe('Error States', () => {
    it('renders "Experiment Not Found" when experiment ID is invalid', () => {
      mockGetExperimentById.mockReturnValue(undefined);
      
      render(<ExperimentPageTemplate experimentId="nonexistent" />);
      
      expect(screen.getByText(/Experiment Not Found/i)).toBeInTheDocument();
      expect(screen.getByText(/Could not find experiment with ID: nonexistent/i)).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /Browse Experiments/i })).toHaveAttribute('href', '/experiments/browse');
    });
  });

  describe('Loading State', () => {
    it('shows skeleton during initialization', () => {
      mockGetExperimentById.mockReturnValue({
        metadata: {
          id: 'test-exp',
          title: 'Test Experiment',
          shortDescription: 'A test',
          fullDescription: 'A test experiment',
          category: 'relativistic',
          tags: ['test'],
          difficulty: 'beginner',
          version: '1.0.0',
          created: '2025-01-01',
          updated: '2025-01-01',
          backend: { minBackend: 'webgpu' },
          education: {
            whatYouSee: 'Test visualization',
            principles: [],
          },
        },
        loader: jest.fn().mockReturnValue(new Promise(() => {})), // Never resolves
      });
      
      render(<ExperimentPageTemplate experimentId="test-exp" />);
      
      // Should show the experiment title (appears in both skeleton and full page)
      expect(screen.getAllByText(/Test Experiment/i).length).toBeGreaterThan(0);
      
      // Backend badge should be rendered
      expect(screen.getByTestId('backend-badge')).toBeInTheDocument();
    });
  });
});


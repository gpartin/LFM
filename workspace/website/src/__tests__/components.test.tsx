/* -*- coding: utf-8 -*- */
/**
 * Component Tests - React Components
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WebGPUErrorBoundary } from '@/components/ErrorBoundary';

describe('WebGPUErrorBoundary', () => {
  it('should render children when no error', () => {
    render(
      <WebGPUErrorBoundary>
        <div>Test Content</div>
      </WebGPUErrorBoundary>
    );
    
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });
  
  it('should render error UI when child throws', () => {
    // Suppress console.error for this test
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    const ThrowError = () => {
      throw new Error('Test error');
    };
    
    render(
      <WebGPUErrorBoundary>
        <ThrowError />
      </WebGPUErrorBoundary>
    );
    
    expect(screen.getByText(/GPU Simulation Error/i)).toBeInTheDocument();
    expect(screen.getByText(/Reload Simulation/i)).toBeInTheDocument();
    
    spy.mockRestore();
  });
});

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Component Tests - React Components and UI Elements
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
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

// Mock UI Components for testing
function ParameterSlider({ 
  label, 
  value, 
  min, 
  max, 
  step, 
  unit, 
  onChange,
  tooltip,
}: { 
  label: string; 
  value: number; 
  min: number; 
  max: number; 
  step: number; 
  unit: string;
  onChange: (value: number) => void;
  tooltip?: string;
}) {
  return (
    <div data-testid="parameter-slider">
      <div>
        <label title={tooltip}>{label}</label>
        <span data-testid="slider-value">{value.toFixed(2)} {unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        title={tooltip}
        aria-label={label}
      />
    </div>
  );
}

function ViewToggle({ 
  label, 
  checked, 
  onChange 
}: { 
  label: string; 
  checked: boolean; 
  onChange: (checked: boolean) => void;
}) {
  return (
    <label data-testid="view-toggle">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        aria-label={label}
        aria-checked={checked}
        role="switch"
      />
      <span>{label}</span>
    </label>
  );
}

describe('UI Components', () => {
  describe('ParameterSlider', () => {
    it('should render with label and value', () => {
      render(
        <ParameterSlider
          label="Test Parameter"
          value={5.5}
          min={0}
          max={10}
          step={0.1}
          unit="×"
          onChange={() => {}}
        />
      );

      expect(screen.getByText('Test Parameter')).toBeInTheDocument();
      expect(screen.getByTestId('slider-value')).toHaveTextContent('5.50 ×');
    });

    it('should call onChange when slider is moved', () => {
      const handleChange = jest.fn();
      render(
        <ParameterSlider
          label="Test Parameter"
          value={5}
          min={0}
          max={10}
          step={1}
          unit="×"
          onChange={handleChange}
        />
      );

      const slider = screen.getByRole('slider', { name: 'Test Parameter' });
      fireEvent.change(slider, { target: { value: '7' } });

      expect(handleChange).toHaveBeenCalledWith(7);
    });

    it('should have proper ARIA labels', () => {
      render(
        <ParameterSlider
          label="Gravity Strength"
          value={0.5}
          min={0}
          max={1}
          step={0.01}
          unit=""
          onChange={() => {}}
        />
      );

      const slider = screen.getByRole('slider', { name: 'Gravity Strength' });
      expect(slider).toBeInTheDocument();
    });
  });

  describe('ViewToggle', () => {
    it('should render with label', () => {
      render(
        <ViewToggle
          label="Show Particles"
          checked={true}
          onChange={() => {}}
        />
      );

      expect(screen.getByText('Show Particles')).toBeInTheDocument();
    });

    it('should reflect checked state', () => {
      const { rerender } = render(
        <ViewToggle label="Show Particles" checked={true} onChange={() => {}} />
      );

      const checkbox = screen.getByRole('switch', { name: 'Show Particles' }) as HTMLInputElement;
      expect(checkbox.checked).toBe(true);

      rerender(
        <ViewToggle label="Show Particles" checked={false} onChange={() => {}} />
      );
      expect(checkbox.checked).toBe(false);
    });

    it('should call onChange when toggled', () => {
      const handleChange = jest.fn();
      render(
        <ViewToggle
          label="Show Particles"
          checked={false}
          onChange={handleChange}
        />
      );

      const checkbox = screen.getByRole('switch', { name: 'Show Particles' });
      fireEvent.click(checkbox);

      expect(handleChange).toHaveBeenCalledWith(true);
    });

    it('should have proper ARIA attributes', () => {
      render(
        <ViewToggle label="Show Particles" checked={true} onChange={() => {}} />
      );

      const checkbox = screen.getByRole('switch', { name: 'Show Particles' });
      expect(checkbox).toHaveAttribute('aria-label', 'Show Particles');
      expect(checkbox).toHaveAttribute('aria-checked', 'true');
    });
  });
});

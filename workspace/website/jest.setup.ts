/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Jest Setup File
 * Runs before each test suite
 */

import '@testing-library/jest-dom';

// Mock WebGPU (not available in jsdom)
global.navigator = global.navigator || {};
(global.navigator as any).gpu = undefined;

// Mock window.matchMedia (used by some components)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
} as any;

// Mock/override Canvas 2D context (jsdom throws not implemented); unconditional to ensure stability
const __canvasNoop = () => {};
(HTMLCanvasElement.prototype.getContext as any) = function getContext(this: HTMLCanvasElement, type: string) {
  if (type === '2d') {
    const self = this;
    return {
      canvas: self,
      fillRect: __canvasNoop,
      clearRect: __canvasNoop,
      getImageData: () => ({ data: new Uint8ClampedArray(0) }),
      putImageData: __canvasNoop,
      createImageData: () => ({ data: new Uint8ClampedArray(0) }),
      setTransform: __canvasNoop,
      drawImage: __canvasNoop,
      save: __canvasNoop,
      restore: __canvasNoop,
      beginPath: __canvasNoop,
      closePath: __canvasNoop,
      stroke: __canvasNoop,
      fill: __canvasNoop,
      clip: __canvasNoop,
      moveTo: __canvasNoop,
      lineTo: __canvasNoop,
      arc: __canvasNoop,
      quadraticCurveTo: __canvasNoop,
      bezierCurveTo: __canvasNoop,
      fillText: __canvasNoop,
      strokeText: __canvasNoop,
      measureText: (text: string) => ({ width: text.length * 8 }),
      createLinearGradient: () => ({ addColorStop: __canvasNoop }),
      createPattern: () => null,
      createRadialGradient: () => ({ addColorStop: __canvasNoop }),
      translate: __canvasNoop,
      scale: __canvasNoop,
      rotate: __canvasNoop,
      setLineDash: __canvasNoop,
      lineWidth: 1,
      lineJoin: 'miter' as const,
      lineCap: 'butt' as const,
      font: '12px monospace',
      textAlign: 'start' as const,
      textBaseline: 'alphabetic' as const,
      globalAlpha: 1,
      globalCompositeOperation: 'source-over' as const,
      strokeStyle: '#000',
      fillStyle: '#000',
    } as unknown as CanvasRenderingContext2D;
  }
  return null;
};

// Suppress console errors in tests (unless debugging)
if (process.env.DEBUG !== 'true') {
  global.console = {
    ...console,
    error: jest.fn(),
    warn: jest.fn(),
  };
}

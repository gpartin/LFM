/* -*- coding: utf-8 -*- */
/**
 * React Error Boundary for WebGPU operations
 * Catches GPU initialization failures and provides graceful degradation
 */

'use client';

import React, { Component, ReactNode } from 'react';
import Link from 'next/link';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class WebGPUErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[WebGPUErrorBoundary] Caught error:', error);
    console.error('[WebGPUErrorBoundary] Component stack:', errorInfo.componentStack);
    this.setState({ error, errorInfo });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-space-dark px-4">
          <div className="max-w-2xl w-full bg-space-panel rounded-lg border border-space-border p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="text-5xl">⚠️</div>
              <div className="flex-1">
                <h1 className="text-2xl font-bold text-accent-chi mb-2">
                  GPU Simulation Error
                </h1>
                <p className="text-text-secondary">
                  The physics simulation encountered an error and couldn't continue.
                </p>
              </div>
            </div>

            {this.state.error && (
              <div className="mb-6 p-4 bg-space-dark rounded border border-red-500/30">
                <h2 className="text-sm font-semibold text-red-400 mb-2">Error Details:</h2>
                <code className="text-xs text-text-secondary font-mono break-all">
                  {this.state.error.toString()}
                </code>
              </div>
            )}

            <div className="space-y-4 mb-6">
              <h2 className="text-lg font-semibold text-text-primary">Common Causes:</h2>
              <ul className="space-y-2 text-sm text-text-secondary list-disc list-inside">
                <li>GPU device lost (driver crash, tab suspended, or power saving mode)</li>
                <li>WebGPU not available in your browser</li>
                <li>Insufficient GPU memory for simulation</li>
                <li>Browser tab was inactive for too long</li>
              </ul>
            </div>

            <div className="space-y-4 mb-6">
              <h2 className="text-lg font-semibold text-text-primary">Recommended Actions:</h2>
              <ul className="space-y-2 text-sm text-text-secondary list-disc list-inside">
                <li>Try reloading the page (button below)</li>
                <li>Update your browser to the latest version</li>
                <li>Close other GPU-intensive tabs</li>
                <li>Check if WebGPU is enabled in your browser settings</li>
              </ul>
            </div>

            <div className="flex gap-4">
              <button
                onClick={this.handleReset}
                className="px-6 py-3 bg-accent-chi hover:bg-accent-chi/80 text-space-dark rounded-lg font-semibold transition-colors"
              >
                🔄 Reload Simulation
              </button>
              <Link
                href="/"
                className="px-6 py-3 bg-space-border hover:bg-space-border/80 text-text-primary rounded-lg font-semibold transition-colors"
              >
                ← Back to Home
              </Link>
            </div>

            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <details className="mt-6 p-4 bg-space-dark rounded border border-space-border">
                <summary className="text-sm font-semibold text-text-primary cursor-pointer mb-2">
                  Component Stack (Dev Only)
                </summary>
                <pre className="text-xs text-text-secondary overflow-auto">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

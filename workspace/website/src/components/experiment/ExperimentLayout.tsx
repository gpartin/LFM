/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * ExperimentLayout
 * 
 * Shared layout component for ALL experiment pages (showcase + research).
 * Handles consistent structure: title, visualization options, disclosure/backend at bottom.
 * Each experiment page provides: simulation canvas, parameters, and controls.
 */

'use client';

import { ReactNode } from 'react';
import Link from 'next/link';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import BackendBadge from '@/components/ui/BackendBadge';
import type { PhysicsBackend } from '@/physics/core/backend-detector';

export interface ExperimentLayoutProps {
  /** Experiment title */
  title: string;
  /** Brief description shown under title */
  description: string;
  /** Current backend (cpu/webgpu) */
  backend: PhysicsBackend;
  /** Experiment ID for About page link */
  experimentId: string;
  /** Visualization options (checkboxes at top) */
  visualizationOptions?: ReactNode;
  /** Main simulation canvas/content */
  children: ReactNode;
  /** Optional footer content above disclosure */
  footerContent?: ReactNode;
}

export default function ExperimentLayout({
  title,
  description,
  backend,
  experimentId,
  visualizationOptions,
  children,
  footerContent,
}: ExperimentLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />
      
      <main className="flex-1 pt-20">
        <div className="container mx-auto px-4 py-8">
          {/* Page Header */}
          <div className="mb-6">
            <h1 className="text-4xl font-bold text-accent-chi mb-2">{title}</h1>
            <p className="text-text-secondary">{description}</p>
          </div>

          {/* Visualization Options Bar (Horizontal at top) */}
          {visualizationOptions && (
            <div className="mb-6">
              {visualizationOptions}
            </div>
          )}

          {/* Main Experiment Content (provided by each page) */}
          {children}

          {/* Footer Content (optional - explanations, notes, etc.) */}
          {footerContent}

          {/* Backend Status & Scientific Disclosure (Bottom) */}
          <div className="mt-8 space-y-4">
            <BackendBadge backend={backend} />
            
            <div className="bg-yellow-500/10 border-l-4 border-yellow-500 p-4 rounded">
              <div className="flex items-start justify-between gap-4">
                <p className="text-sm text-text-secondary flex-1">
                  <strong className="text-yellow-400">Scientific Disclosure:</strong> This is an exploratory simulation. 
                  We are NOT claiming this is proven physics. <Link href={`/about?from=${experimentId}`} className="text-accent-chi hover:underline">Learn more about our approach and limitations →</Link>
                </p>
                <Link 
                  href={`/about?from=${experimentId}`}
                  className="px-4 py-2 bg-yellow-500/20 border-2 border-yellow-500/50 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors whitespace-nowrap text-sm font-semibold shrink-0"
                >
                  ⚠️ Read About This Project
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

/**
 * Reusable visualization checkbox component
 */
export function VisualizationCheckbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center space-x-2 cursor-pointer group">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 rounded border-space-border bg-space-dark checked:bg-accent-chi checked:border-accent-chi focus:ring-2 focus:ring-accent-chi/50 cursor-pointer"
      />
      <span className="text-text-primary group-hover:text-accent-chi transition-colors text-sm">
        {label}
      </span>
    </label>
  );
}

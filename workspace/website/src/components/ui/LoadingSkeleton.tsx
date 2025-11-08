/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
/**
 * Loading Skeleton Components
 * 
 * Provides structured loading states with skeleton UI to prevent layout shift
 * and improve perceived performance during async operations.
 */

'use client';

import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';

export type LoadingStage = 'detecting' | 'loading' | 'initializing';

export interface ExperimentSkeletonProps {
  stage: LoadingStage;
  experimentTitle?: string;
}

/**
 * Loading spinner component
 */
export function LoadingSpinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizeClasses = {
    sm: 'h-8 w-8 border-2',
    md: 'h-16 w-16 border-4',
    lg: 'h-24 w-24 border-4',
  };
  
  return (
    <div className={`animate-spin rounded-full border-accent-chi border-t-transparent ${sizeClasses[size]}`} />
  );
}

/**
 * Experiment page skeleton with loading indicator
 */
export function ExperimentSkeleton({ stage, experimentTitle }: ExperimentSkeletonProps) {
  const stageMessages = {
    detecting: '🔍 Detecting GPU capabilities...',
    loading: '📦 Loading physics engine...',
    initializing: '⚛️ Initializing simulation...',
  };
  
  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />
      
      <main className="flex-1 pt-20">
        <div className="container mx-auto px-4 py-8">
          {/* Skeleton header */}
          <div className="mb-8">
            {experimentTitle ? (
              <h1 className="text-4xl font-bold text-accent-chi/50 mb-2 animate-pulse">
                {experimentTitle}
              </h1>
            ) : (
              <div className="h-12 w-96 bg-space-border animate-pulse rounded mb-2" />
            )}
            <div className="h-6 w-full max-w-2xl bg-space-border animate-pulse rounded" />
          </div>
          
          {/* Skeleton backend badge */}
          <div className="mb-8">
            <div className="h-8 w-48 bg-space-border animate-pulse rounded" />
          </div>
          
          {/* Main content area */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Skeleton canvas (2/3 width) */}
            <div className="lg:col-span-2">
              <div className="panel h-[600px] flex items-center justify-center">
                <div className="text-center">
                  <LoadingSpinner size="lg" />
                  <div className="mt-6 text-text-secondary text-lg">
                    {stageMessages[stage]}
                  </div>
                  <div className="mt-2 text-text-muted text-sm">
                    This may take a few seconds...
                  </div>
                </div>
              </div>
              
              {/* Skeleton controls */}
              <div className="mt-4 flex items-center justify-center space-x-4">
                <div className="h-12 w-32 bg-space-border animate-pulse rounded-lg" />
                <div className="h-12 w-32 bg-space-border animate-pulse rounded-lg" />
                <div className="h-12 w-32 bg-space-border animate-pulse rounded-lg" />
              </div>
            </div>
            
            {/* Skeleton control panel (1/3 width) */}
            <div className="space-y-6">
              {/* Skeleton parameters */}
              <div className="panel">
                <div className="h-7 w-32 bg-space-border animate-pulse rounded mb-4" />
                <div className="space-y-4">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="h-5 w-40 bg-space-border animate-pulse rounded" />
                        <div className="h-5 w-16 bg-space-border animate-pulse rounded" />
                      </div>
                      <div className="h-2 w-full bg-space-border animate-pulse rounded" />
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Skeleton metrics */}
              <div className="panel">
                <div className="h-7 w-32 bg-space-border animate-pulse rounded mb-4" />
                <div className="space-y-3">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="flex items-center justify-between py-2 border-b border-space-border">
                      <div className="h-5 w-32 bg-space-border animate-pulse rounded" />
                      <div className="h-6 w-20 bg-space-border animate-pulse rounded" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          {/* Skeleton explanation panel */}
          <div className="mt-8 panel">
            <div className="h-8 w-48 bg-space-border animate-pulse rounded mb-4" />
            <div className="space-y-3">
              <div className="h-4 w-full bg-space-border animate-pulse rounded" />
              <div className="h-4 w-full bg-space-border animate-pulse rounded" />
              <div className="h-4 w-3/4 bg-space-border animate-pulse rounded" />
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

/**
 * Simple loading indicator for smaller components
 */
export function LoadingIndicator({ message }: { message?: string }) {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="text-center">
        <LoadingSpinner />
        {message && (
          <div className="mt-4 text-text-secondary">{message}</div>
        )}
      </div>
    </div>
  );
}

/**
 * Inline skeleton for lists
 */
export function ListSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-4">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="panel p-4">
          <div className="flex items-center gap-4">
            <div className="h-16 w-16 bg-space-border animate-pulse rounded" />
            <div className="flex-1 space-y-2">
              <div className="h-6 w-3/4 bg-space-border animate-pulse rounded" />
              <div className="h-4 w-full bg-space-border animate-pulse rounded" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

interface ControlPanelProps {
  isRunning: boolean;
  speed: number;
  currentStep: number;
  totalSteps: number;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  mode?: 'SHOWCASE' | 'RESEARCH';  // Controls which buttons appear
  onStepForward?: () => void;  // RESEARCH mode: advance 1 step
  onStepBackward?: () => void;  // RESEARCH mode: revert 1 step
}

/**
 * Standardized control panel matching existing showcase experiments.
 * Uses Unicode symbols (▶ ⏸) consistent with binary-orbit, three-body, etc.
 * 
 * Modes:
 * - SHOWCASE: Play/Pause/Reset with speed control (editable experiments)
 * - RESEARCH: Play/Pause/Reset with Step Forward/Back (locked test configs)
 */
export default function ControlPanel({
  isRunning,
  speed,
  currentStep,
  totalSteps,
  onPlay,
  onPause,
  onReset,
  onSpeedChange,
  mode = 'SHOWCASE',
  onStepForward,
  onStepBackward
}: ControlPanelProps) {
  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;
  
  return (
    <div className="mt-4 space-y-4">
      {/* Play/Pause/Reset Controls */}
      <div className="flex items-center justify-center space-x-4" role="group" aria-label="Simulation controls">
        <button
          onClick={onPlay}
          disabled={isRunning}
          aria-label="Start simulation"
          className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
            isRunning
              ? 'bg-accent-glow/40 text-space-dark/60 cursor-not-allowed'
              : 'bg-accent-glow hover:bg-accent-glow/80 text-space-dark'
          }`}
        >
          ▶ Play
        </button>
        <button
          onClick={onPause}
          disabled={!isRunning}
          aria-label="Pause simulation"
          className={`px-6 py-3 rounded-lg font-semibold transition-colors border-2 ${
            !isRunning
              ? 'border-accent-chi/40 text-accent-chi/40 cursor-not-allowed'
              : 'border-accent-chi text-accent-chi hover:bg-accent-chi/10'
          }`}
        >
          ⏸ Pause
        </button>
        <button
          onClick={onReset}
          className="px-6 py-3 rounded-lg font-semibold transition-colors bg-indigo-500 hover:bg-indigo-400 text-white"
          aria-label="Reset simulation"
        >
          🔄 Reset
        </button>
        
        {/* Step Controls (RESEARCH mode only) */}
        {mode === 'RESEARCH' && onStepBackward && onStepForward && (
          <>
            <button
              onClick={onStepBackward}
              disabled={isRunning || currentStep === 0}
              aria-label="Step backward one frame"
              className={`px-6 py-3 rounded-lg font-semibold transition-colors border-2 ${
                isRunning || currentStep === 0
                  ? 'border-slate-600 text-slate-600 cursor-not-allowed'
                  : 'border-purple-500 text-purple-400 hover:bg-purple-500/10'
              }`}
            >
              ⏮ Step Back
            </button>
            <button
              onClick={onStepForward}
              disabled={isRunning || currentStep >= totalSteps}
              aria-label="Step forward one frame"
              className={`px-6 py-3 rounded-lg font-semibold transition-colors border-2 ${
                isRunning || currentStep >= totalSteps
                  ? 'border-slate-600 text-slate-600 cursor-not-allowed'
                  : 'border-purple-500 text-purple-400 hover:bg-purple-500/10'
              }`}
            >
              Step Forward ⏭
            </button>
          </>
        )}
      </div>
      
      {/* Step Counter */}
      <div className="text-center text-sm text-text-secondary">
        Step: <span className="text-white font-mono">{currentStep}</span> / {totalSteps.toLocaleString()}
      </div>
      
      {/* Progress Bar */}
      <div className="w-full max-w-2xl mx-auto">
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div 
            className="h-full bg-accent-chi transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  );
}

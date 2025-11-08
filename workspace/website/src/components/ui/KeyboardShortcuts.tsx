/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

// -*- coding: utf-8 -*-
'use client';

import { useEffect, useState } from 'react';

interface KeyboardShortcut {
  key: string;
  description: string;
  displayKey?: string;
}

interface KeyboardShortcutsProps {
  shortcuts: KeyboardShortcut[];
}

/**
 * Keyboard shortcuts overlay component
 * 
 * Displays a modal showing available keyboard shortcuts.
 * Triggered by pressing '?' key.
 */
export default function KeyboardShortcuts({ shortcuts }: KeyboardShortcutsProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Toggle with '?' key (Shift + /)
      if (e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        setIsVisible(prev => !prev);
      }
      
      // Close with Escape when visible
      if (e.key === 'Escape' && isVisible) {
        e.preventDefault();
        setIsVisible(false);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div 
      className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={() => setIsVisible(false)}
    >
      <div 
        className="bg-space-dark border-2 border-accent-chi/50 rounded-lg shadow-2xl max-w-md w-full p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-accent-chi">⌨️ Keyboard Shortcuts</h2>
          <button
            onClick={() => setIsVisible(false)}
            className="text-text-secondary hover:text-text-primary transition-colors text-2xl leading-none"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="space-y-3">
          {shortcuts.map((shortcut, index) => (
            <div 
              key={index} 
              className="flex items-center justify-between py-2 border-b border-text-secondary/20 last:border-0"
            >
              <span className="text-text-secondary">{shortcut.description}</span>
              <kbd className="px-3 py-1 bg-gray-800 border border-gray-600 rounded text-text-primary font-mono text-sm shadow-sm">
                {shortcut.displayKey || shortcut.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-6 pt-4 border-t border-text-secondary/20 text-center">
          <p className="text-sm text-text-secondary">
            Press <kbd className="px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs">?</kbd> anytime to toggle this help
          </p>
        </div>
      </div>
    </div>
  );
}

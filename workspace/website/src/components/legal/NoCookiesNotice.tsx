/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function NoCookiesNotice() {
  const [dismissed, setDismissed] = useState(true);

  useEffect(() => {
    try {
      const v = window.localStorage.getItem('lfm_no_cookies_notice');
      if (v !== 'dismissed') {
        setDismissed(false);
      }
    } catch {
      // ignore storage errors
      setDismissed(false);
    }
  }, []);

  const dismiss = () => {
    setDismissed(true);
    try { window.localStorage.setItem('lfm_no_cookies_notice', 'dismissed'); } catch {}
  };

  if (dismissed) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50">
      <div className="mx-auto max-w-5xl m-4 rounded-lg border border-space-border bg-space-panel/95 backdrop-blur p-4 flex items-center justify-between gap-4">
        <p className="text-sm text-text-secondary">
          We use <strong>no cookies</strong> and no trackers. Learn more in our{' '}
          <Link href="/legal/privacy" className="text-accent-chi hover:underline">Privacy Policy</Link>
          {' '}and <Link href="/legal/cookies" className="text-accent-chi hover:underline">Cookie Policy</Link>.
        </p>
        <button 
          onClick={dismiss}
          className="px-3 py-1.5 text-sm font-semibold bg-accent-chi text-space-dark rounded hover:bg-accent-particle transition-colors"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import Link from 'next/link';

export default function CookiesPage() {
  return (
    <div className="min-h-screen bg-space-dark text-text-primary">
      <div className="container mx-auto px-4 py-16 max-w-3xl">
        <h1 className="text-3xl font-bold text-accent-chi mb-4">Cookie Policy</h1>
        <p className="text-text-secondary mb-8">Last updated: November 8, 2025</p>

        <section className="space-y-6 text-text-secondary leading-7">
          <p>
            We do not use cookies. We do not use tracking pixels, fingerprinting, or advertising cookies.
            If in the future we introduce strictly necessary client-side preferences (e.g., a theme preference
            stored in <code>localStorage</code>), it will be for functionality only and not for tracking.
          </p>

          <p>
            If this policy changes, we will update this page and the Privacy Policy and indicate the date above.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">Contact</h2>
          <p>
            Questions about this policy? Contact: {' '}
            <a href="mailto:research@emergentphysicslab.com" className="text-accent-chi hover:underline">research@emergentphysicslab.com</a>
          </p>

          <p className="mt-8">
            See also: <Link className="text-accent-chi hover:underline" href="/legal/privacy">Privacy Policy</Link> • <Link className="text-accent-chi hover:underline" href="/legal/terms">Terms of Service</Link>
          </p>

          <div className="mt-12 text-center">
            <Link href="/" className="inline-flex items-center gap-2 px-5 py-2.5 rounded bg-accent-chi text-space-dark font-semibold hover:bg-accent-particle transition-colors">
              <span>←</span>
              <span>Back to Home</span>
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
}

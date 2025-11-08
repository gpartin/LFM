/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import Link from 'next/link';

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-space-dark text-text-primary">
      <div className="container mx-auto px-4 py-16 max-w-3xl">
        <h1 className="text-3xl font-bold text-accent-chi mb-4">Privacy Policy</h1>
        <p className="text-text-secondary mb-8">Last updated: November 8, 2025</p>

        <section className="space-y-6 text-text-secondary leading-7">
          <h2 className="text-xl font-semibold text-text-primary">Summary</h2>
          <p>
            We do not collect personal information, do not set cookies, and do not use analytics or advertising trackers.
            Any data generated in your browser (e.g., simulation parameters) remains client-side unless you explicitly export or share it.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">Cookies</h2>
          <p>
            We use <strong>no cookies</strong>. If a future feature requires strictly necessary client-side storage
            (e.g., a user preference in <code>localStorage</code>), it will not be used for tracking and will be disclosed here.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">Third-Party Services</h2>
          <p>
            We do not embed third-party analytics or advertising. If we link to external resources, those sites have their own policies.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">Contact</h2>
          <p>
            Questions about this policy? Contact us at <a className="text-accent-chi hover:underline" href="mailto:research@emergentphysicslab.com">research@emergentphysicslab.com</a>.
          </p>

          <p className="mt-8">
            See also: <Link className="text-accent-chi hover:underline" href="/legal/cookies">Cookie Policy</Link> • <Link className="text-accent-chi hover:underline" href="/legal/terms">Terms of Service</Link>
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

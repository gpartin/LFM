/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import Link from 'next/link';

export default function DisclaimerPage() {
  return (
    <div className="min-h-screen bg-space-dark text-text-primary">
      <div className="container mx-auto px-4 py-16 max-w-3xl">
        <h1 className="text-3xl font-bold text-accent-chi mb-4">Research & Scientific Disclaimer</h1>
        <p className="text-text-secondary mb-8">Last updated: November 8, 2025</p>

        <section className="space-y-6 text-text-secondary leading-7">
          <p>
            This project is an exploratory research effort. The simulations, models, and results are provided for
            scientific discussion and hypothesis generation. They have not undergone formal peer review and
            should not be interpreted as established physical law or definitive claims.
          </p>

          <p>
            The code and materials are provided under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
            (CC BY-NC-ND 4.0) license. Non-commercial use with attribution is permitted; no distribution of modified versions and no
            commercial use without a separate written agreement. No warranty is provided regarding correctness, performance, or
            suitability for any purpose.
          </p>

          <p>
            Always validate results independently and consult appropriate domain experts before relying on any outputs.
          </p>

          <p className="mt-8">
            See also: <Link className="text-accent-chi hover:underline" href="/legal/terms">Terms of Service</Link> • <Link className="text-accent-chi hover:underline" href="/legal/privacy">Privacy Policy</Link>
          </p>

          <h2 className="text-xl font-semibold text-text-primary mt-12">Contact</h2>
          <p>
            Questions about this disclaimer or requests for clarification: {' '}
            <a href="mailto:research@emergentphysicslab.com" className="text-accent-chi hover:underline">research@emergentphysicslab.com</a>
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

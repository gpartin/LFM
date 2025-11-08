/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

/* -*- coding: utf-8 -*- */
'use client';

import Link from 'next/link';

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-space-dark text-text-primary">
      <div className="container mx-auto px-4 py-16 max-w-3xl">
        <h1 className="text-3xl font-bold text-accent-chi mb-4">Terms of Service</h1>
        <p className="text-text-secondary mb-8">Last updated: November 8, 2025</p>

        <section className="space-y-6 text-text-secondary leading-7">
          <p>
            These Terms govern your access to and use of the Emergent Physics Lab website and
            the open research materials we publish (the "Service"). By accessing or using the Service,
            you agree to these Terms.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">1. Research Use Only</h2>
          <p>
            The Service is provided for non-commercial research, education, and scientific collaboration.
            No commercial use of our ideas, methods, processes, code, or outputs is permitted without a
            separate written license from the project owner.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">2. No Professional Advice</h2>
          <p>
            Content is exploratory and provided "as is" for scientific discussion. It does not constitute
            professional advice (including scientific, engineering, medical, or investment advice).
          </p>

          <h2 className="text-xl font-semibold text-text-primary">3. Intellectual Property & License</h2>
          <p>
            Unless otherwise noted, the source code and research materials are licensed under
            the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
            (CC BY-NC-ND 4.0) license. Non-commercial usage with attribution is permitted; no
            distribution of modified versions and no commercial use without a separate written
            agreement. See the LICENSE file for full terms.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">4. Third-Party Components</h2>
          <p>
            We may include third-party libraries subject to their own licenses. You are responsible
            for complying with any third-party terms.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">5. No Warranty</h2>
          <p>
            THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED.
            We make no guarantee of accuracy, fitness for a particular purpose, or non-infringement.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">6. Limitation of Liability</h2>
          <p>
            TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE SHALL NOT BE LIABLE FOR ANY INDIRECT,
            INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR ANY LOSS OF PROFITS OR
            REVENUES, WHETHER INCURRED DIRECTLY OR INDIRECTLY.
          </p>

          <h2 className="text-xl font-semibold text-text-primary">7. Changes</h2>
          <p>
            We may update these Terms. Material changes will be indicated by updating the date above.
          </p>

          <p className="mt-8">
            See also: <Link className="text-accent-chi hover:underline" href="/legal/privacy">Privacy Policy</Link> • <Link className="text-accent-chi hover:underline" href="/legal/disclaimer">Research Disclaimer</Link>
          </p>
          <div className="mt-12 border-t border-space-border pt-8 space-y-8">
            <div>
              <h2 className="text-xl font-semibold text-text-primary">Contact</h2>
              <p>
                Questions about these Terms or requests for permissions: {' '}
                <a href="mailto:research@emergentphysicslab.com" className="text-accent-chi hover:underline">research@emergentphysicslab.com</a>
              </p>
            </div>
            <div className="text-center">
              <Link href="/" className="inline-flex items-center gap-2 px-5 py-2.5 rounded bg-accent-chi text-space-dark font-semibold hover:bg-accent-particle transition-colors">
                <span>←</span>
                <span>Back to Home</span>
              </Link>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

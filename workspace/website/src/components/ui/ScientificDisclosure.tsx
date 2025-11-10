/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import Link from 'next/link';

interface ScientificDisclosureProps {
  experimentName: string;
}

/**
 * Reusable scientific disclosure banner and About button for experiment pages.
 * Shows warning that this is exploratory research and links to full About page.
 */
export default function ScientificDisclosure({ experimentName }: ScientificDisclosureProps) {
  return (
    <div className="flex items-center justify-between gap-4 bg-yellow-500/10 border border-yellow-500/30 p-4 rounded-lg">
      <div className="flex-1">
        <p className="text-sm text-text-secondary">
          <strong className="text-yellow-400">Scientific Disclosure:</strong> This is an exploratory simulation. 
          We are NOT claiming this is proven physics.{' '}
          <Link href={`/about?from=${encodeURIComponent(experimentName)}`} className="text-accent-chi hover:underline">
            Learn more about our approach and limitations →
          </Link>
        </p>
      </div>
      <Link 
        href={`/about?from=${encodeURIComponent(experimentName)}`}
        className="px-4 py-2 bg-yellow-500/20 border-2 border-yellow-500/50 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors whitespace-nowrap text-sm font-semibold"
      >
        ⚠️ Read About This Project
      </Link>
    </div>
  );
}

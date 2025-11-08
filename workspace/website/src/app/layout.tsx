/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' });

export const metadata: Metadata = {
  title: 'Emergent Physics Lab - Interactive LFM Demonstrations',
  description: 'Interactive physics experiments showing how fundamental forces emerge from the Lattice Field Medium framework. Explore gravity, relativity, and quantum phenomena emerging from a single equation.',
  keywords: ['physics', 'lattice field medium', 'emergent gravity', 'quantum mechanics', 'klein-gordon', 'physics simulation'],
  authors: [{ name: 'Greg Partin' }],
  openGraph: {
    title: 'Emergent Physics Lab',
    description: 'Interactive demonstrations of force emergence from lattice physics',
    type: 'website',
    url: 'https://emergentphysicslab.com',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body>
        {children}
      </body>
    </html>
  );
}

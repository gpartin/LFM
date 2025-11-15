/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
 * SPDX-License-Identifier: CC-BY-NC-ND-4.0
 */

import type { Metadata } from 'next';
import { headers } from 'next/headers';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' });

export async function generateMetadata(): Promise<Metadata> {
  // Derive absolute origin dynamically so Open Graph/Twitter images work on previews and production
  const h = headers();
  const host = h.get('x-forwarded-host') ?? h.get('host') ?? 'emergentphysicslab.com';
  const proto = h.get('x-forwarded-proto') ?? 'https';
  const origin = `${proto}://${host}`;

  return {
    metadataBase: new URL(origin),
    title: {
      default: 'Emergent Physics Lab - Interactive Physics Simulations',
      template: '%s | Emergent Physics Lab',
    },
    description:
      'Interactive physics experiments demonstrating emergent gravity, quantum mechanics, and electromagnetic phenomena from Klein-Gordon lattice field dynamics. 105+ validated tests exploring the Lattice Field Medium framework.',
    keywords: [
      'physics simulation',
      'emergent gravity',
      'lattice field medium',
      'Klein-Gordon equation',
      'quantum mechanics',
      'computational physics',
      'emergent phenomena',
      'field theory',
      'physics experiments',
      'interactive physics',
      'gravitational effects',
      'relativistic physics',
      'LFM framework',
    ],
    authors: [{ name: 'Greg Partin', url: `${origin}/about` }],
    creator: 'Emergent Physics Lab',
    publisher: 'Emergent Physics Lab',
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        'max-video-preview': -1,
        'max-image-preview': 'large',
        'max-snippet': -1,
      },
    },
    openGraph: {
      type: 'website',
      locale: 'en_US',
      url: origin,
      siteName: 'Emergent Physics Lab',
      title: 'Emergent Physics Lab - Interactive Physics Simulations',
      description:
        'Interactive demonstrations of emergent physics: gravity, quantum mechanics, and electromagnetism from Klein-Gordon lattice field dynamics.',
      images: [
        {
          url: '/opengraph-image',
          width: 1200,
          height: 630,
          alt: 'Emergent Physics Lab - Klein-Gordon Field Simulations',
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title: 'Emergent Physics Lab',
      description:
        'Interactive physics simulations showing emergent phenomena from lattice field dynamics',
      images: ['/twitter-image'],
      creator: '@EmergentPhysics', // TODO: Create Twitter account
    },
    alternates: {
      canonical: origin,
    },
    manifest: '/manifest.json',
    verification: {
      google: '', // TODO: Add after Google Search Console setup
    },
  };
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Structured data for rich search results
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Organization',
    name: 'Emergent Physics Lab',
    url: 'https://emergentphysicslab.com',
    logo: 'https://emergentphysicslab.com/logo.png',
    description: 'Interactive physics research demonstrating emergent phenomena from Klein-Gordon lattice field dynamics',
    sameAs: [
      // TODO: Add social media profiles when created
      // 'https://twitter.com/EmergentPhysics',
      // 'https://github.com/gpartin',
    ],
    contactPoint: {
      '@type': 'ContactPoint',
      contactType: 'Research Inquiries',
      email: 'research@emergentphysicslab.com',
    },
  };

  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        {/* Google Tag Manager */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-TT8KBLZV');`,
          }}
        />
        {/* End Google Tag Manager */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body>
        {/* Google Tag Manager (noscript) */}
        <noscript>
          <iframe
            src="https://www.googletagmanager.com/ns.html?id=GTM-TT8KBLZV"
            height="0"
            width="0"
            style={{ display: 'none', visibility: 'hidden' }}
          />
        </noscript>
        {/* End Google Tag Manager (noscript) */}
        {children}
      </body>
    </html>
  );
}

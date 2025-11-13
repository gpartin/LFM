// -*- coding: utf-8 -*-
/**
 * Dynamic Open Graph image generator (1200x630 PNG)
 * Next.js file-based metadata convention
 */
import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const alt = 'Emergent Physics Lab - Klein-Gordon Field Simulations';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default async function Image() {
  const { width, height } = size;
  const title = 'Emergent Physics Lab';
  const subtitle = 'Interactive Physics Simulations';
  const tagline = 'Lattice Field Medium • Klein-Gordon ∂²E/∂t² = c²∇²E − χ²E';

  try {
    return new ImageResponse(
      (
        <div
          style={{
            width: 1200,
            height: 630,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-start',
            alignItems: 'flex-start',
            padding: 64,
            position: 'relative',
            background: '#0a0e27', // space.dark
            backgroundColor: '#0a0e27',
            color: '#e0e6ed', // text.primary
            fontFamily:
              'Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
          }}
        >
          {/* Accent strip */}
          <div style={{ position: 'absolute', top: 0, left: 0, width: 1200, height: 14, backgroundColor: '#00d9ff' }} />

          {/* Title block (mirrors homepage hero) */}
          <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 24 }}>
            <div style={{ fontSize: 80, fontWeight: 800, color: '#00d9ff' /* accent.chi */ }}>Fundamental Forces</div>
            <div style={{ fontSize: 64, fontWeight: 800, color: '#e0e6ed' }}>Emerging from</div>
            <div style={{ fontSize: 64, fontWeight: 800, color: '#ff6b35' /* accent.particle */ }}>a Single Equation</div>
          </div>

          {/* Subtitle */}
          <div style={{ fontSize: 28, color: '#8892a6', marginBottom: 28 }}>
            Watch gravity, relativity, and quantum phenomena emerge from the Lattice Field Medium.
          </div>

          {/* Equation panel */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              backgroundColor: '#141b3d', // space.panel
              border: '2px solid #1e2847', // space.border
              borderRadius: 12,
              padding: 20,
              marginBottom: 24,
            }}
          >
            <div style={{ fontFamily: 'ui-monospace, Menlo, monospace', fontSize: 34, color: '#00d9ff' }}>
              d²E/dt² = c²∇²E - χ²(x,t)E
            </div>
            <div style={{ fontSize: 18, color: '#555d6e' }}>One equation. All fundamental forces.</div>
          </div>

          {/* URL footer */}
          <div style={{ fontSize: 22, color: '#8892a6' }}>emergentphysicslab.com</div>
        </div>
      ),
      { width, height }
    );
  } catch (err) {
    // Log detailed error to console for local debugging
    console.error('[opengraph-image] generation failed:', err);
    // Fallback: ultra-simple card that should never fail
    return new ImageResponse(
      (
        <div
          style={{
            width: 1200,
            height: 630,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#111827',
            color: '#ffffff',
            fontSize: 54,
            fontFamily: 'system-ui, Arial',
          }}
        >
          Emergent Physics Lab
        </div>
      ),
      { width, height }
    );
  }
}

// -*- coding: utf-8 -*-
/**
 * Dynamic Twitter card image generator (1200x630 PNG)
 * Next.js file-based metadata convention
 */
import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const alt = 'Emergent Physics Lab - Interactive Physics Simulations';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default async function Image() {
  const { width, height } = size;
  const title = 'Emergent Physics Lab';
  const subtitle = 'Emergent Gravity • Quantum • Electromagnetism';

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
            padding: 56,
            position: 'relative',
            background: '#0a0e27',
            backgroundColor: '#0a0e27',
            color: '#e0e6ed',
            fontFamily:
              'Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
          }}
        >
          {/* Accent strip */}
          <div style={{ position: 'absolute', top: 0, left: 0, width: 1200, height: 12, backgroundColor: '#00d9ff' }} />

          {/* Title (condensed for Twitter) */}
          <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 18 }}>
            <div style={{ fontSize: 62, fontWeight: 800, color: '#00d9ff' }}>Fundamental Forces</div>
            <div style={{ fontSize: 46, fontWeight: 800, color: '#e0e6ed' }}>Emerging from</div>
            <div style={{ fontSize: 46, fontWeight: 800, color: '#ff6b35' }}>a Single Equation</div>
          </div>

          {/* Equation panel */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              backgroundColor: '#141b3d',
              border: '2px solid #1e2847',
              borderRadius: 12,
              padding: 18,
              marginBottom: 18,
            }}
          >
            <div style={{ fontFamily: 'ui-monospace, Menlo, monospace', fontSize: 28, color: '#00d9ff' }}>
              d²E/dt² = c²∇²E - χ²(x,t)E
            </div>
          </div>

          {/* URL */}
          <div style={{ fontSize: 18, color: '#8892a6' }}>emergentphysicslab.com</div>
        </div>
      ),
      { width, height }
    );
  } catch (err) {
    console.error('[twitter-image] generation failed:', err);
    return new ImageResponse(
      (
        <div
          style={{
            width: 1200,
            height: 630,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#0b1326',
            color: '#ffffff',
            fontSize: 48,
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

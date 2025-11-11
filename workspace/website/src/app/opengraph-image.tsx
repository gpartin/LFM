// -*- coding: utf-8 -*-
/**
 * Dynamic Open Graph image generator (1200x630 PNG)
 * URL: /og-image.png
 * Uses Next.js ImageResponse (Edge runtime).
 */
import { ImageResponse } from 'next/og';

export const runtime = 'edge';

export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export async function GET() {
  const { width, height } = size;
  const title = 'Emergent Physics Lab';
  const subtitle = 'Interactive Physics Simulations';
  const tagline = 'Lattice Field Medium • Klein-Gordon ∂²E/∂t² = c²∇²E − χ²E';

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'flex-start',
          padding: 64,
          background:
            'radial-gradient(1200px 630px at 0% 0%, #0b1220 0%, #0a0f1a 60%, #070c14 100%)',
          color: '#e5f0ff',
          fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 24,
            marginBottom: 40,
          }}
        >
          <div
            style={{
              width: 96,
              height: 96,
              borderRadius: 24,
              background:
                'conic-gradient(from 45deg, #99f6e4, #60a5fa, #a78bfa, #f472b6, #99f6e4)',
              boxShadow: '0 10px 40px rgba(96,165,250,0.35)',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 72, fontWeight: 800, letterSpacing: -1 }}>{title}</div>
            <div style={{ fontSize: 36, opacity: 0.9 }}>{subtitle}</div>
          </div>
        </div>
        <div style={{ fontSize: 28, color: '#99a7c2', marginBottom: 24 }}>emergentphysicslab.com</div>
        <div style={{ fontSize: 24, color: '#94a3b8' }}>{tagline}</div>
      </div>
    ),
    {
      width,
      height,
    }
  );
}

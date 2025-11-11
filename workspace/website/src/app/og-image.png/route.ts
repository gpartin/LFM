// -*- coding: utf-8 -*-
/**
 * Dynamic Open Graph image generator (1200x630 PNG)
 * Path: /og-image.png
 * Uses Next.js @vercel/og ImageResponse to render a branded card.
 */
import { ImageResponse } from 'next/og';

export const runtime = 'edge';

export async function GET() {
  const width = 1200;
  const height = 630;

  const title = 'Emergent Physics Lab';
  const subtitle = 'Interactive Physics Simulations';
  const url = 'emergentphysicslab.com';

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
            gap: 16,
            marginBottom: 24,
          }}
        >
          {/* Logo mark */}
          <div
            style={{
              width: 72,
              height: 72,
              borderRadius: 16,
              background:
                'conic-gradient(from 45deg, #99f6e4, #60a5fa, #a78bfa, #f472b6, #99f6e4)',
              boxShadow: '0 10px 40px rgba(96,165,250,0.35)',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 64, fontWeight: 800, letterSpacing: -1 }}>{title}</div>
            <div style={{ fontSize: 32, opacity: 0.9 }}>{subtitle}</div>
          </div>
        </div>

        <div
          style={{
            marginTop: 8,
            fontSize: 28,
            color: '#99a7c2',
          }}
        >
          {url}
        </div>

        <div
          style={{
            position: 'absolute',
            right: 48,
            bottom: 36,
            fontSize: 22,
            color: '#94a3b8',
          }}
        >
          Lattice Field Medium • Klein-Gordon ∂²E/∂t² = c²∇²E − χ²E
        </div>
      </div>
    ),
    {
      width,
      height,
    }
  );
}

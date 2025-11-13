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

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: 56,
          // Use solid color to avoid unsupported CSS functions in @vercel/og renderer
          background: '#0b1326',
          color: '#eaf1ff',
          fontFamily:
            'Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <div
            style={{
              width: 64,
              height: 64,
              borderRadius: 16,
              // Replace gradients with solid color for compatibility
              background: '#1e3a8a',
              boxShadow: '0 8px 30px rgba(96,165,250,0.45)',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 48, fontWeight: 800, letterSpacing: -0.5 }}>{title}</div>
            <div style={{ fontSize: 22, opacity: 0.9 }}>{subtitle}</div>
          </div>
        </div>

        <div
          style={{
            alignSelf: 'flex-end',
            fontSize: 18,
            color: '#c7d7ff',
            background: 'rgba(9,14,28,0.45)',
            border: '1px solid rgba(255,255,255,0.1)',
            padding: '10px 14px',
            borderRadius: 10,
            backdropFilter: 'blur(4px)',
          }}
        >
          emergentphysicslab.com
        </div>
      </div>
    ),
    { width, height }
  );
}

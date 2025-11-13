// -*- coding: utf-8 -*-
/**
 * Dynamic Apple touch icon (180x180 PNG)
 * Next.js app/apple-icon.tsx convention
 */
import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const dynamic = 'force-dynamic';
export const size = { width: 180, height: 180 };
export const contentType = 'image/png';

export default function AppleIcon() {
  const { width, height } = size;
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          // Solid background for compatibility
          backgroundColor: '#14213d',
          borderRadius: 36,
        }}
      >
        <div
          style={{
            width: 92,
            height: 92,
            borderRadius: 24,
            // Replace conic gradient with solid color
            backgroundColor: '#60a5fa',
          }}
        />
      </div>
    ),
    { width, height }
  );
}

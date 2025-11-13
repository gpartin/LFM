// -*- coding: utf-8 -*-
/**
 * Dynamic favicon generator (32x32 PNG)
 * Next.js app/icon.tsx convention
 */
import { ImageResponse } from 'next/og';

export const size = { width: 32, height: 32 };
export const contentType = 'image/png';

export default function Icon() {
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
          background: '#1e3a8a',
          borderRadius: 8,
        }}
      >
        <div
          style={{
            width: 18,
            height: 18,
            borderRadius: 6,
            // Avoid conic gradients
            background: '#60a5fa',
            boxShadow: '0 0 6px rgba(74,144,226,0.65) inset',
          }}
        />
      </div>
    ),
    { width, height }
  );
}

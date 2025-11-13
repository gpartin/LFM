// -*- coding: utf-8 -*-
/**
 * Dynamic Apple touch icon (180x180 PNG)
 * Next.js app/apple-icon.tsx convention
 */
import { ImageResponse } from 'next/og';

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
          background:
            'linear-gradient(135deg, #0b1326 0%, #14213d 60%, #1b2a52 100%)',
          borderRadius: 36,
        }}
      >
        <div
          style={{
            width: 92,
            height: 92,
            borderRadius: 24,
            background:
              'conic-gradient(from 45deg, #99f6e4, #60a5fa, #a78bfa, #f472b6, #99f6e4)',
            boxShadow: '0 12px 40px rgba(96,165,250,0.35)'
          }}
        />
      </div>
    ),
    { width, height }
  );
}

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Scientific dark theme
        'space': {
          'dark': '#0a0e27',
          'panel': '#141b3d',
          'border': '#1e2847',
        },
        'accent': {
          'chi': '#00d9ff',      // Cyan for chi field
          'particle': '#ff6b35', // Orange for particles
          'glow': '#00ff88',     // Green for success states
        },
        'text': {
          'primary': '#e0e6ed',
          'secondary': '#8892a6',
          'muted': '#555d6e',
        },
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        }
      }
    },
  },
  plugins: [],
}

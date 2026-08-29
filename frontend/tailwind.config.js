/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Geist Mono', 'Menlo', 'monospace'],
        sans: ['Inter', 'Geist Sans', 'system-ui', 'sans-serif'],
        display: ['Geist Sans', 'Inter', 'sans-serif'],
      },
      colors: {
        ink: {
          bg: '#080a0a',
          surface: '#0f1111',
          panel: '#141717',
          border: '#1e2323',
          border2: '#252a2a',
          muted: '#6b7280',
          dim: '#9ca3af',
        },
        aegis: {
          green: '#7cf000',
          lime: '#a3ff12',
          cyan: '#00e5cc',
          red: '#ff3b3b',
          orange: '#ff8c42',
          blue: '#58a6ff',
        }
      },
      keyframes: {
        pulseDot: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.35' } },
        blink: { '0%,50%': { opacity: '1' }, '51%,100%': { opacity: '0.35' } },
        drift: { '0%': { opacity: '0' }, '10%': { opacity: '1' }, '90%': { opacity: '1' }, '100%': { opacity: '0' } },
      },
      animation: {
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
        'blink': 'blink 1s step-end infinite',
      }
    },
  },
  plugins: [],
}

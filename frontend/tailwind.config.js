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
      // Motion an operator benefits from, and nothing else. `pulseDot` and
      // `blink` drew the eye continuously to indicators that were not changing;
      // on a screen watched for hours that is fatigue, not emphasis. `drift`
      // was declared with no matching animation and never ran at all.
      // `toast-in` is the one addition: a notification has to announce itself.
      keyframes: {
        toastIn: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'toast-in': 'toastIn 140ms ease-out',
      }
    },
  },
  plugins: [],
}

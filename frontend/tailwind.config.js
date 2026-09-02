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
        // The console skin carries a slight green cast, so the surfaces sit
        // under the signal green rather than beside it. Pure neutrals made the
        // accent read as a sticker on the chrome; these do not.
        ink: {
          bg: '#080a0a',
          surface: '#0f1111',
          panel: '#141717',
          border: '#1e2323',
          border2: '#252a2a',
        },
        aegis: {
          // Signal green: primary actions, active nav, healthy/live state.
          // Red and amber stay reserved for risk, so the three never collide.
          accent: '#7cf000',
          accentDim: '#5bb000',
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
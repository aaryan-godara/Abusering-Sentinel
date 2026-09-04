/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Forensic console palette: white / black / gray / red only.
        // Scale runs light (950 = surface white) to dark (50 = near black) so
        // surfaces stay white and typography stays black.
        ink: {
          950: '#ffffff',
          900: '#fbfbfc',
          850: '#f3f3f5',
          800: '#e3e3e7',
          700: '#d4d4da',
          600: '#c3c3cb',
          500: '#8e8e97',
          400: '#666666',
          350: '#555555',
          300: '#444444',
          200: '#333333',
          100: '#222222',
          50: '#111111',
        },
        signal: {
          DEFAULT: '#dc2626',
          bright: '#bf0d1e',
          dim: '#f0b8b8',
          deep: '#fdeaea',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'sans-serif',
        ],
      },
    },
  },
  plugins: [],
}

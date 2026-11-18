/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Fira Code', 'JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        thermal: {
          low: '#FACC15',
          med: '#EA580C',
          high: '#991B1B',
        }
      }
    },
  },
  plugins: [],
}

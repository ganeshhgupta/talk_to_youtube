import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        yt: {
          bg: '#0f0f0f',
          sidebar: '#212121',
          card: '#1e1e1e',
          border: '#303030',
          red: '#ff0000',
          'red-dark': '#cc0000',
          muted: '#aaaaaa',
          dim: '#666666',
        },
      },
      animation: {
        'spin-slow': 'spin 1.5s linear infinite',
      },
    },
  },
  plugins: [],
};

export default config;

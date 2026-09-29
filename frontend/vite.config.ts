import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  publicDir: false,
  worker: { format: 'es' },
  server: { strictPort: true },
  preview: { strictPort: true },
});

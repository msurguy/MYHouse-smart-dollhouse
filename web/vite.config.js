import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  base: '/MYHouse-smart-dollhouse/',
  build: {
    outDir: 'dist',
    sourcemap: true
  },
  server: {
    port: 3000,
    open: true
  }
});

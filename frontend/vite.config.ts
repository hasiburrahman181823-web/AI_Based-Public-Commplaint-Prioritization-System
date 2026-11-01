import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173, strictPort: true,
      proxy: { '/api': { target: env.DEV_API_TARGET || 'http://127.0.0.1:8000', changeOrigin: false, rewrite: path => path.replace(/^\/api(?=\/|$)/, '') } },
    },
    build: { target: 'es2022', sourcemap: false },
  }
})

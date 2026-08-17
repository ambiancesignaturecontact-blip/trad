import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI backend serves the classic dashboard at "/" on the app port
// (uvicorn --port $PORT, default 8080). The Vite dev server proxies API/WS
// calls to it so the React dashboard works in development too.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true
      }
    }
  }
})

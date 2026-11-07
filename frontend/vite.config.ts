import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const isDev = process.env.NODE_ENV !== 'production'
const envContent = isDev
  ? readFileSync(resolve(import.meta.dirname, '../.env'), 'utf8')
  : ''
const apiKeyLine = envContent.split(/\r?\n/).find(line =>
  line.trim().startsWith('CYBER_GUARDIAN_API_KEY=')
)
const apiKey = apiKeyLine?.split('=').slice(1).join('=').trim()

if (isDev && !apiKey) {
  throw new Error('CYBER_GUARDIAN_API_KEY is missing')
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            if (apiKey) {
              proxyReq.setHeader('X-API-Key', apiKey)
            }
          })
        },
      },
    },
  },
})

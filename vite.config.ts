import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import rollupNodePolyFill from 'rollup-plugin-node-polyfills'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps:{
    disabled: false 
  },
  resolve: {
    alias: {
      path: 'rollup-plugin-node-polyfills/polyfills/path',
      process: 'rollup-plugin-node-polyfills/polyfills/process-es6',
      // buffer: 'rollup-plugin-node-polyfills/polyfills/buffer-es6',
      events: 'rollup-plugin-node-polyfills/polyfills/events',
      util: 'rollup-plugin-node-polyfills/polyfills/util',
      sys: 'util',
      stream: 'rollup-plugin-node-polyfills/polyfills/stream',
    }
  },
  build:{
    commonjsOptions:{
      include: []
    },
    rollupOptions: {
      plugins: [
          // Enable rollup polyfills plugin
          // used during production bundling
          rollupNodePolyFill()
      ]
    }
  }
})

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || "/",

  optimizeDeps: {
    // web-ifc ships a WASM runtime that Vite's dependency pre-bundler cannot
    // rewrite correctly; excluding it is what keeps `vite dev` from crashing
    // on the first IFC load. @thatopen/fragments spawns a worker that must
    // stay a real module for the same reason.
    exclude: ["web-ifc", "@thatopen/fragments"],
  },

  worker: {
    format: "es",
  },

  build: {
    target: "es2022",
    chunkSizeWarningLimit: 2500,
    rollupOptions: {
      output: {
        // Three and the That Open engine are large and change on a different
        // cadence to app code; splitting them keeps app rebuilds off the
        // user's download path.
        manualChunks: {
          three: ["three"],
          thatopen: ["@thatopen/components", "@thatopen/fragments"],
        },
      },
    },
  },

  server: {
    // SharedArrayBuffer headers let the multi-threaded web-ifc build engage
    // where the browser supports it.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});

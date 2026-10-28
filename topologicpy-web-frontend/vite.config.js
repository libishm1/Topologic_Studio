import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** The installed version of a package, not the range package.json asks for. */
const installed = (name) =>
  JSON.parse(readFileSync(new URL(`./node_modules/${name}/package.json`, import.meta.url), "utf8")).version;

export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || "/",

  // The fragment cache keys converted models on the converter's versions
  // (src/lib/fragmentCache.js); only the two strings go into the bundle.
  define: {
    "import.meta.env.FRAGMENTS_VERSION": JSON.stringify(installed("@thatopen/fragments")),
    "import.meta.env.WEBIFC_VERSION": JSON.stringify(installed("web-ifc")),
  },

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

  // No Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy here, on
  // purpose. Setting them makes the page cross-origin isolated, which gives it
  // SharedArrayBuffer, which makes web-ifc select its multi-threaded pthread
  // build. That build then spawns workers with an undefined script URL
  // (allocateUnusedWorker -> initMainThread -> IfcAPI2.Init), every worker
  // fetches "/undefined", receives index.html, and dies with
  // "Uncaught SyntaxError: Unexpected token '<'". IFC loading never completes.
  //
  // The single-threaded build is what production runs anyway, since static
  // hosts do not send these headers. Keeping dev and production on the same
  // code path is worth more here than speculative threading.
});

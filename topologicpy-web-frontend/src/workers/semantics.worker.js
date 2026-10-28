/**
 * Semantics worker: reads what the IFC says about egress (exit doors, roof
 * slabs; src/core/ifcSemantics.js) with its own single-threaded web-ifc, so
 * it runs beside the geometry conversion instead of after it. No geometry.
 *
 * The page starts one of these per file and ends it after the answer:
 * web-ifc's heap never shrinks, and a worker kept for the session held
 * 133 MiB after the Duplex.
 */
import * as WEBIFC from "web-ifc";

import { ifcSemantics } from "../core/ifcSemantics.js";

// The loader's tape is reserved up front, 64 MiB by default. Reading the
// relations needs far less: with 4 MiB the heap is 16 MiB after the Duplex
// and 33 MiB after the 12 MiB dental clinic, against 133 MiB, with the same
// result; a larger file takes more of it as it goes.
const TAPE_SIZE = 4 * 1024 * 1024;

let ready = null;

function start(wasmPath) {
  const api = new WEBIFC.IfcAPI();
  api.SetWasmPath(wasmPath, /^https?:/.test(wasmPath));
  return api.Init(undefined, true).then(() => {
    api.SetLogLevel(WEBIFC.LogLevel.LOG_LEVEL_OFF);
    return api;
  });
}

self.onmessage = async (event) => {
  const { id, bytes, wasmPath } = event.data || {};
  try {
    // Kept only once Init has resolved, so a failed start is tried again
    // rather than reused half made.
    ready ||= start(wasmPath).catch((error) => {
      ready = null;
      throw error;
    });
    const api = await ready;
    const t0 = performance.now();
    const model = api.OpenModel(new Uint8Array(bytes), { TAPE_SIZE });
    try {
      const result = ifcSemantics(api, model, WEBIFC);
      result.ms = performance.now() - t0;
      self.postMessage({ id, ok: true, result });
    } finally {
      api.CloseModel(model);
    }
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};

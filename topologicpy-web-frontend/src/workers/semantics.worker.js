/**
 * Semantics worker: reads what the IFC says about egress (exit doors, roof
 * slabs; src/core/ifcSemantics.js) with its own single-threaded web-ifc, so
 * it runs beside the geometry conversion instead of after it. No geometry.
 */
import * as WEBIFC from "web-ifc";

import { ifcSemantics } from "../core/ifcSemantics.js";

let api = null;

self.onmessage = async (event) => {
  const { id, bytes, wasmPath } = event.data || {};
  try {
    if (!api) {
      api = new WEBIFC.IfcAPI();
      api.SetWasmPath(wasmPath, /^https?:/.test(wasmPath));
      await api.Init(undefined, true);
      api.SetLogLevel(WEBIFC.LogLevel.LOG_LEVEL_OFF);
    }
    const t0 = performance.now();
    const model = api.OpenModel(new Uint8Array(bytes));
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

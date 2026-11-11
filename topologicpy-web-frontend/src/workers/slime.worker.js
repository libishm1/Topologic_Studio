/**
 * The slime mould's way out (src/core/slimeEgress.js) off the main thread:
 * two thousand agents for four hundred steps take most of a second on the
 * NIBS Clinic's graph. Loaded only when the Route step's slime mould is run.
 */
import { slimeEgress } from "../core/slimeEgress.js";

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const result = slimeEgress(payload.graph, payload.opts);
    self.postMessage({ id, ok: true, result }, [result.weights.buffer, result.visits.buffer]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};

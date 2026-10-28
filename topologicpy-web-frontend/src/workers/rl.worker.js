/**
 * RL worker: trains the Q-learning policy and works out the exact optimum it
 * is compared with (src/core/rl.js) off the main thread. The optimum's
 * backward induction visits every open move at every step: on the 34,000-node
 * dental clinic about 0.3 s at 200 steps and 1.6 s at 1,000, all of it with
 * the page frozen when it ran there.
 *
 * The graph arrives as copies of its typed arrays (the page keeps its own).
 */
import { trainRl } from "../core/rl.js";

self.onmessage = (event) => {
  const { id, graph, options } = event.data || {};
  try {
    self.postMessage({ id, ok: true, result: trainRl(graph, options) });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};

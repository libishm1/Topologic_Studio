/**
 * RL worker: trains the Q-learning policy and works out the exact optimum it
 * is compared with (src/core/rl.js) off the main thread. The optimum's
 * backward induction visits every open move at every step: on the 34,000-node
 * dental clinic about 0.3 s at 200 steps and 1.6 s at 1,000, all of it with
 * the page frozen when it ran there.
 *
 * With `options.method === "shielded"` it runs the shielded learned escape
 * instead (src/core/rlSafe.js trainSafeRl), which takes the same options plus
 * `exitPoints` (escape by any of several exits) and returns the same shape
 * with its own fields (moves, certified, deadline_start, exit_id).
 *
 * The graph arrives as copies of its typed arrays (the page keeps its own).
 */
import { trainRl } from "../core/rl.js";
import { trainSafeRl } from "../core/rlSafe.js";

self.onmessage = (event) => {
  const { id, graph, options } = event.data || {};
  try {
    const train = options?.method === "shielded" ? trainSafeRl : trainRl;
    self.postMessage({ id, ok: true, result: train(graph, options) });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};

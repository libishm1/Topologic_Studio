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
 * The fire is the graph's hop front from the fire origin (fire_source "hop"),
 * unless `options.frames` carries a finished Flamescapes run's frames: then
 * both learners avoid that run's forecast (src/core/rlForecast.js
 * trainForecastRl), its closing times read as the forecast plan reads them
 * (forecast.js forecastFromFrames, with `options.tenability`).
 *
 * The graph arrives as copies of its typed arrays (the page keeps its own).
 */
import { forecastFromFrames } from "../core/forecast.js";
import { trainRl } from "../core/rl.js";
import { trainForecastRl } from "../core/rlForecast.js";
import { trainSafeRl } from "../core/rlSafe.js";

self.onmessage = (event) => {
  const { id, graph, options } = event.data || {};
  try {
    let result;
    if (options?.frames) {
      const { frames, tenability, ...rest } = options;
      const forecast = forecastFromFrames(frames, tenability);
      result = trainForecastRl(graph, { ...rest, forecast });
    } else {
      const train = options?.method === "shielded" ? trainSafeRl : trainRl;
      const hop = train(graph, options);
      result = { ...hop, fire_source: hop.fire_start_id != null ? "hop" : "none", fire_speed: options?.fireSpeed ?? 1 };
    }
    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};

/**
 * Routing-ready surrogates with error bounds (gap LG7 of the 2026-10-27 gap
 * analysis: wiki/research/gap-analysis-2026-10-27/README.md).
 *
 * A surrogate replaces a forecast's closing times c and FED rates r by cheaper
 * ones. The routers are monotone in both (Surrogate.lean): closing a point
 * earlier or raising a rate can only remove safe routes and raise doses. So
 *   - a conservative surrogate (c' <= c and r' >= r everywhere) gives only
 *     routes that are safe under the full forecast, doses no lower and
 *     deadlines no later;
 *   - a surrogate whose closing times are certified within e of the full
 *     model's, |c' - c| <= e, becomes conservative by closing every point e
 *     earlier (withClosingMargin), and the hard deadline then moves by at most
 *     2e: it is 1-Lipschitz in the closing times in the sup norm.
 *
 * One certified conservative surrogate comes with Studio. In the heat-
 * diffusion model a node changes only towards the mean of its open
 * neighbours, so no node leaves ambient before its hop distance from the fire
 * along open links: the spread-along-links (flood) forecast on the same graph,
 * walls and clock closes every node no later than diffusion does, for any
 * threshold above ambient, and its rates (120 C once reached, ambient before)
 * are never below diffusion's. It costs one breadth-first search instead of K
 * diffusion sweeps.
 */
import { buildForecast } from "./forecast.js";

/**
 * The forecast with every closing and untenable time moved `margin` seconds
 * earlier (not below 0): a surrogate whose closing times are within `margin`
 * of the true ones then closes no point later than the true forecast does.
 */
export function withClosingMargin(forecast, margin) {
  if (!(margin >= 0)) throw new Error("The margin must be a non-negative number of seconds.");
  const closes = Float64Array.from(forecast.closes, (c) => Math.max(0, c - margin));
  const untenable = Float64Array.from(forecast.untenable, (u) => (u === Infinity ? Infinity : Math.max(0, u - margin)));
  return { ...forecast, closes, untenable, margin: (forecast.margin || 0) + margin };
}

/**
 * The hop-front surrogate of the heat-diffusion forecast: the flood forecast
 * on the same graph, walls, clock and threshold. Conservative for any
 * threshold above ambient (see the header).
 */
export function hopSurrogate(graph, opts) {
  return { ...buildForecast(graph, { ...opts, model: "flood" }), surrogate: "hop front" };
}

/**
 * How a surrogate compares with a full forecast on the same graph: whether it
 * is conservative (never closes later, never a lower rate), and how much
 * earlier it closes points (the price of the guarantee).
 */
export function compareForecasts(full, surrogate) {
  const n = full.n;
  let lateNodes = 0, lowRates = 0, finite = 0, sum = 0, max = 0;
  for (let v = 0; v < n; v += 1) {
    const gap = full.closes[v] - surrogate.closes[v];
    if (gap < -1e-9) lateNodes += 1;
    if (Number.isFinite(gap)) {
      finite += 1;
      sum += gap;
      if (gap > max) max = gap;
    }
  }
  // Rates on the full forecast's frames: the surrogate's at the same time.
  for (let k = 0; k < full.K; k += 1) {
    const t = k * full.frameSeconds;
    const ks = Math.min(surrogate.K - 1, Math.floor(t / surrogate.frameSeconds));
    for (let v = 0; v < n; v += 1) {
      if (surrogate.rate[ks * n + v] < full.rate[k * n + v] * (1 - 1e-6)) lowRates += 1;
    }
  }
  return {
    conservative: lateNodes === 0 && lowRates === 0,
    lateNodes,
    lowRates,
    meanEarlier: finite ? sum / finite : 0,
    maxEarlier: max,
  };
}

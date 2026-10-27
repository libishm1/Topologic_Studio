/**
 * Forecast worker: runs the fire model to its horizon, plans against it and
 * maps the escape deadlines (src/core/forecast.js, src/core/deadline.js) off
 * the main thread. The deadline map is the slow part, a few seconds on a
 * 30,000-node graph.
 *
 * The graph arrives as copies of its typed arrays (the page keeps its own);
 * the deadline map goes back as a transferable buffer.
 */
import { buildForecast, planWithForecast } from "../core/forecast.js";
import { doseDeadline, doseDeadlineMap, escapeDeadlines } from "../core/deadline.js";

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const t0 = performance.now();
    const { graph: g, start, exits, seed, model, steps, secondsPerStep, threshold, speed, depart, limit, useWalls, withMap } = payload;
    const graph = { nodes: g.nodes, edges: g.edges, blocked: g.blocked };
    const blocked = useWalls ? graph.blocked : null;
    const forecast = buildForecast(graph, { model, seed, steps, dt: secondsPerStep, threshold, blocked });
    const tForecast = performance.now();
    const plan = planWithForecast(graph, { start, exits, forecast, depart, speed, blocked });
    const hard = escapeDeadlines(graph, { exits, untenable: forecast.closes, speed, blocked })[start];
    const leaveBy = doseDeadline(graph, start, { exits, forecast, limit, speed, blocked, tolerance: 0.1 });
    const tPlan = performance.now();
    const map = withMap ? doseDeadlineMap(graph, { exits, forecast, limit, speed, blocked }) : null;
    const tMap = performance.now();
    const result = {
      horizon: forecast.horizon,
      coarsened: forecast.stride > 1 ? forecast.frameSeconds : null,
      startUntenable: forecast.untenable[start],
      hardDeadline: hard,
      leaveBy,
      earliest: plan.earliest,
      fastest: plan.fastest,
      leastDose: plan.leastDose,
      front: plan.front,
      truncated: plan.truncated,
      labels: plan.labels,
      snapshot: plan.snapshot,
      deadlines: map?.deadline ?? null,
      mapStep: map?.step ?? null,
      ms: { forecast: tForecast - t0, plan: tPlan - tForecast, map: tMap - tPlan, total: tMap - t0 },
    };
    self.postMessage({ id, ok: true, result }, map ? [map.deadline.buffer] : []);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};

/**
 * Fire spread and hazard-aware rerouting in the browser, emitting the same
 * messages as the server's SSE stream (app/routers/simulation.py) so the UI
 * handles either.
 *
 * Models, as on the server:
 *   radial       ignition bands by straight-line distance from the seed
 *   flood        breadth-first along the graph, one hop per step (Classic)
 *   temperature  Classic's diffusion (eCAADe 2019): each node moves toward its
 *                neighbours' mean when that is hotter, clipped to 20-120 C
 *
 * One correction from the literature review (wiki/research/fire-egress-
 * methods-2026-10-27.md, finding W1): the gain coefficient is
 * min(1, heatTransfer / degree), not heatTransfer / degree. For degree 1 the
 * old 1.2 pushed a dead end past its own neighbour - a discrete maximum
 * principle violation. For degree 2 and up the two are identical, so results
 * stay comparable with Classic.
 */
import { astar, csr, edgeLengths, snap } from "./route.js";

export const AMBIENT = 20;
export const FIRE = 120;

function medianEdge(graph) {
  const l = Array.from(edgeLengths(graph)).sort((a, b) => a - b);
  return l.length ? l[Math.floor(l.length / 2)] || 1 : 1;
}

export function radialTimeline(graph, start, maxSteps = 60) {
  const P = graph.nodes;
  const step = medianEdge(graph);
  const n = P.length / 3;
  const bands = [];
  for (let i = 0; i < n; i += 1) {
    const d = Math.hypot(P[i * 3] - P[start * 3], P[i * 3 + 1] - P[start * 3 + 1], P[i * 3 + 2] - P[start * 3 + 2]);
    const b = Math.floor(d / step);
    if (b >= maxSteps) continue;
    (bands[b] ||= []).push(i);
  }
  return Array.from({ length: bands.length }, (_, b) => bands[b] || []);
}

export function floodTimeline(graph, start, maxSteps = 60, blocked = null) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const seen = new Uint8Array(n);
  seen[start] = 1;
  let frontier = [start];
  const timeline = [[start]];
  for (let s = 1; s < maxSteps; s += 1) {
    const next = [];
    for (const u of frontier) {
      for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
        if (blocked?.[edgeIds[k]]) continue;
        const v = neighbours[k];
        if (!seen[v]) {
          seen[v] = 1;
          next.push(v);
        }
      }
    }
    if (!next.length) break;
    timeline.push(next);
    frontier = next;
  }
  return timeline;
}

/** Temperature fields, one Float64Array per step (the first is the seed). */
export function* temperatureSteps(graph, start, { maxSteps = 60, blocked = null, heatTransfer = 1.2, ambient = AMBIENT, fire = FIRE } = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  let temps = new Float64Array(n).fill(ambient);
  temps[start] = fire;
  for (let s = 0; s < Math.max(1, maxSteps); s += 1) {
    yield Float64Array.from(temps);
    const next = Float64Array.from(temps);
    for (let i = 0; i < n; i += 1) {
      const degree = indptr[i + 1] - indptr[i];
      if (!degree) continue;
      let sum = 0, count = 0;
      for (let k = indptr[i]; k < indptr[i + 1]; k += 1) {
        if (blocked?.[edgeIds[k]]) continue;
        sum += temps[neighbours[k]];
        count += 1;
      }
      const delta = sum / Math.max(count, 1) - temps[i];
      // Blocked links still count toward the degree, so a wall slows heating
      // on its far side (as on the server).
      if (count && delta > 0) next[i] = temps[i] + delta * Math.min(1, heatTransfer / degree);
    }
    next[start] = fire;
    for (let i = 0; i < n; i += 1) next[i] = Math.min(fire, Math.max(ambient, next[i]));
    temps = next;
  }
}

/** length * (1 + alpha * normalised hazard), as the server's hazard_weights. */
export function hazardWeights(graph, temps, alpha, ambient = AMBIENT, span = 100) {
  const base = edgeLengths(graph);
  if (!temps || alpha <= 0) return base;
  const m = graph.edges.length / 2;
  const out = new Float64Array(m);
  for (let e = 0; e < m; e += 1) {
    const avg = (temps[graph.edges[e * 2]] + temps[graph.edges[e * 2 + 1]]) / 2;
    out[e] = base[e] * (1 + alpha * Math.max(0, (avg - ambient) / span));
  }
  return out;
}

/**
 * Run a fire simulation, calling onMessage with server-shaped messages and
 * onDone at the end. Returns a handle with close().
 */
export function runFire(graph, opts, { onMessage, onDone } = {}) {
  const {
    model = "temperature",
    firePoint,
    startPoint = null,
    exitPoint = null,
    maxSteps = 60,
    delayMs = 120,
    useWalls = true,
    reroute = false,
    interval = 5,
    alpha = 0.5,
    lethality = null,
  } = opts;
  const walls = useWalls ? graph.blocked : null;
  const seed = snap(graph, firePoint);
  const walker = startPoint ? snap(graph, startPoint) : null;
  const exit = exitPoint ? snap(graph, exitPoint) : null;
  const routing = reroute && walker !== null && exit !== null;
  const m = graph.edges.length / 2;
  let lastPath = "";
  let lastRecompute = -1e9;
  let closed = false;
  let timer = null;

  const pointsOf = (nodes) => nodes.map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]);
  const rerouteNow = (step, weights, blocked) => {
    let result = astar(graph, walker, exit, { weights, blocked });
    let found = Boolean(result);
    let note = null;
    if (!result && blocked) {
      // Keep the walls; a route that needs the fire is drawn but is not a
      // safe route, so it is not reported as found (review finding W4).
      result = astar(graph, walker, exit, { weights, blocked: walls });
      if (result) note = "The fire has cut every route; the least-hazard one is drawn, but it is not a safe route.";
    }
    const key = result ? result.nodes.join(",") : "";
    const changed = key !== lastPath;
    lastPath = key;
    onMessage?.({
      type: "path_update",
      step,
      found,
      path: result ? pointsOf(result.nodes) : [],
      cost: result ? Math.round(result.cost * 1000) / 1000 : null,
      changed,
      engine: "browser",
      note: note || (found ? null : "No route to the exit."),
    });
  };

  let steps;
  const burning = new Uint8Array(graph.nodes.length / 3);
  if (model === "temperature") {
    steps = temperatureSteps(graph, seed, { maxSteps, blocked: walls });
  } else {
    const timeline = model === "flood" ? floodTimeline(graph, seed, maxSteps, walls) : radialTimeline(graph, seed, maxSteps);
    steps = timeline.values();
  }

  let step = 0;
  const tick = () => {
    if (closed) return;
    const { value, done } = steps.next();
    if (done) {
      closed = true;
      onDone?.();
      return;
    }
    if (model === "temperature") {
      const temperatures = {};
      for (let i = 0; i < value.length; i += 1) if (value[i] > 21) temperatures[i] = Math.round(value[i] * 100) / 100;
      onMessage?.({ type: "temperature_step", step, temperatures });
      if (routing && step - lastRecompute >= interval) {
        lastRecompute = step;
        let blocked = walls ? Uint8Array.from(walls) : null;
        if (lethality !== null && lethality !== undefined) {
          blocked = blocked || new Uint8Array(m);
          for (let e = 0; e < m; e += 1) {
            if ((value[graph.edges[e * 2]] + value[graph.edges[e * 2 + 1]]) / 2 > lethality) blocked[e] = 1;
          }
        }
        rerouteNow(step, hazardWeights(graph, value, alpha), blocked);
      }
    } else {
      for (const node of value) burning[node] = 1;
      onMessage?.({ type: "step", step, nodes: value });
      if (routing && step - lastRecompute >= interval) {
        lastRecompute = step;
        const blocked = walls ? Uint8Array.from(walls) : new Uint8Array(m);
        for (let e = 0; e < m; e += 1) {
          if (burning[graph.edges[e * 2]] || burning[graph.edges[e * 2 + 1]]) blocked[e] = 1;
        }
        rerouteNow(step, null, blocked);
      }
    }
    step += 1;
    timer = setTimeout(tick, delayMs);
  };
  timer = setTimeout(tick, 0);

  return {
    close() {
      closed = true;
      if (timer) clearTimeout(timer);
    },
    get closed() {
      return closed;
    },
  };
}

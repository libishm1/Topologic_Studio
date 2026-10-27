/**
 * Planning against the fire forecast (gap analysis 2026-10-27, gaps LG1 and
 * LG2: wiki/research/gap-analysis-2026-10-27/gap-matrix.json).
 *
 * Rerouting during the simulation re-solves on the fire as it is now. Here the
 * fire model is run to its last step first, and the routers read each node's
 * state at the moment the evacuee would actually be there.
 *
 * Clock. Fire step k covers [k dt, (k + 1) dt) seconds; dt is a modelling
 * choice of the caller (the models count steps, not seconds). Walking a link
 * of length l takes l / speed seconds. The forecast ends at H = K dt: nothing
 * is known after it, so every node counts as closed from H on.
 *
 * Forecast. The model's heat at every node and step, stored as the running
 * maximum over time. Studio's three models never cool a node (diffusion only
 * heats; a burning node stays burning), so the maximum changes nothing for
 * them; for any other forecast it is the conservative choice, and it is what
 * makes both routers exact: heat that never falls gives FIFO links and doses
 * that never fall with the departure time.
 *
 * Untenability. A node closes at c(v) = min(u(v), H), where u(v) is the start
 * of the first step at which its heat reaches the threshold. A walk is safe if
 * it reaches every node strictly before that node closes.
 *
 * 1. earliestArrival: time-dependent Dijkstra (Dreyfus 1969). Each link's
 *    arrival function A(t) = t + l / speed, or closed if that is not before
 *    c(head), is FIFO and never earlier than t, so the labels are exact
 *    earliest arrivals (TimeDependent.td_dijkstra_correct).
 * 2. doseFront: fractional effective dose of convective heat (ISO 13571,
 *    Purser): rate 1 / t_I with t_I = 5e7 T^-3.4 min, T in degrees C,
 *    accumulated over the walk; a link takes the hotter of its two ends. The
 *    search keeps, per node, every label (arrival, dose) that no other label
 *    beats on both; because heat never falls, a label that arrives no later
 *    with no more dose stays at least as good along every continuation, and
 *    waiting never helps (MinDose.no_holdover), so the labels that reach an
 *    exit are the exact time-dose trade-off: its first point is the earliest
 *    safe arrival, its last the least-dose route.
 *
 * The heat is the model's index, not a measured gas temperature (Methods), so
 * the dose ranks routes under the model; it is not a measured FED.
 */
import { Heap, csr, edgeLengths } from "./route.js";
import { AMBIENT, FIRE, floodTimeline, radialTimeline, temperatureSteps } from "./fire.js";

/** Convective-heat FED per second at T degrees C: 1 / (5e7 T^-3.4 min). */
export function convectiveDoseRate(T) {
  return T > 0 ? Math.pow(T, 3.4) / (5e7 * 60) : 0;
}

/** Above this many node-steps the forecast is coarsened in time (conservatively). */
const MAX_CELLS = 16_000_000;

/**
 * Run the fire model to its horizon and keep, per step and node, the FED rate
 * of the running maximum of its heat.
 *
 * @param {object} opts  model: "temperature" | "flood" | "radial"; seed: the
 *                       fire's node; steps: fire steps; dt: seconds per step;
 *                       threshold: heat at which a node is untenable (model
 *                       degrees C; flood and radial nodes are at FIRE once
 *                       burning); blocked: per-edge flags (walls).
 */
export function buildForecast(graph, { model = "temperature", seed, steps = 60, dt = 5, threshold = FIRE, blocked = null }) {
  const n = graph.nodes.length / 3;
  const K0 = Math.max(1, Math.floor(steps));
  // Coarsening keeps one frame per `stride` steps, holding the hottest state
  // of its window (its last, since heat never falls): never optimistic.
  const stride = Math.max(1, Math.ceil((K0 * n) / MAX_CELLS));
  const K = Math.ceil(K0 / stride);
  const rate = new Float32Array(K * n);
  const untenable = new Float64Array(n).fill(Infinity);
  const peak = new Float64Array(n).fill(-Infinity);

  const take = (k0, field) => {
    // field(v) -> heat at fire step k0.
    for (let v = 0; v < n; v += 1) {
      const T = field(v);
      if (T > peak[v]) peak[v] = T;
      if (untenable[v] === Infinity && peak[v] >= threshold) untenable[v] = k0 * dt;
    }
    if ((k0 + 1) % stride === 0 || k0 === K0 - 1) {
      const row = Math.floor(k0 / stride) * n;
      for (let v = 0; v < n; v += 1) rate[row + v] = convectiveDoseRate(peak[v]);
    }
  };

  if (model === "temperature") {
    let k0 = 0;
    for (const frame of temperatureSteps(graph, seed, { maxSteps: K0, blocked })) {
      take(k0, (v) => frame[v]);
      k0 += 1;
    }
  } else {
    const timeline = model === "flood" ? floodTimeline(graph, seed, K0, blocked) : radialTimeline(graph, seed, K0);
    const ignite = new Int32Array(n).fill(-1);
    timeline.forEach((nodes, k) => nodes.forEach((v) => (ignite[v] = k)));
    for (let k0 = 0; k0 < K0; k0 += 1) take(k0, (v) => (ignite[v] >= 0 && ignite[v] <= k0 ? FIRE : AMBIENT));
  }

  const horizon = K0 * dt;
  const closes = new Float64Array(n);
  for (let v = 0; v < n; v += 1) closes[v] = Math.min(untenable[v], horizon);
  return { n, K, stride, dt, frameSeconds: stride * dt, horizon, rate, untenable, closes, threshold };
}

/**
 * FED accrued walking from v to w, leaving at t0 and arriving at t1 (t1 is
 * before the horizon): the integral of the hotter end's rate.
 */
export function linkDose(forecast, v, w, t0, t1) {
  const { rate, n, K, frameSeconds: f } = forecast;
  const k0 = Math.floor(t0 / f);
  const k1 = Math.min(K - 1, Math.floor(t1 / f));
  let dose = 0;
  for (let k = k0; k <= k1; k += 1) {
    const a = Math.max(t0, k * f);
    const b = Math.min(t1, (k + 1) * f);
    if (b > a) dose += (b - a) * Math.max(rate[k * n + v], rate[k * n + w]);
  }
  return dose;
}

function exitFlags(n, exits) {
  const isExit = new Uint8Array(n);
  for (const x of exits) if (x >= 0 && x < n) isExit[x] = 1;
  return isExit;
}

/**
 * Earliest safe arrival from `start` leaving at `depart`: time-dependent
 * Dijkstra, stopping at the first exit settled.
 *
 * @returns {{exit: number, arrive: Float64Array, parent: Int32Array}}  exit is
 *          -1 when no exit can be reached safely.
 */
export function earliestArrival(graph, start, depart, { exits, closes, speed = 1.2, blocked = null }) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const len = edgeLengths(graph);
  const arrive = new Float64Array(n).fill(Infinity);
  const parent = new Int32Array(n).fill(-1);
  if (!(depart < closes[start])) return { exit: -1, arrive, parent };
  const isExit = exitFlags(n, exits);
  const done = new Uint8Array(n);
  arrive[start] = depart;
  const heap = new Heap();
  heap.push(depart, start);
  while (heap.size) {
    const v = heap.pop();
    if (done[v]) continue;
    done[v] = 1;
    if (isExit[v]) return { exit: v, arrive, parent };
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const e = edgeIds[k];
      if (blocked?.[e]) continue;
      const w = neighbours[k];
      const t = arrive[v] + len[e] / speed;
      if (t < closes[w] && t < arrive[w]) {
        arrive[w] = t;
        parent[w] = v;
        heap.push(t, w);
      }
    }
  }
  return { exit: -1, arrive, parent };
}

/** Node path from a parent array, ending at `end`. */
function unwind(parent, end) {
  const nodes = [end];
  for (let c = parent[end]; c >= 0; c = parent[c]) nodes.push(c);
  return nodes.reverse();
}

/**
 * The time-dose trade-off of safe routes from `start` to any exit.
 *
 * @returns {{front: Array<{nodes: number[], times: number[], doses: number[], arrival: number, dose: number}>,
 *            labels: number, truncated: boolean}}
 *          front is ordered by arrival (dose strictly falling); empty when no
 *          exit can be reached safely. truncated: the label budget ran out, so
 *          the front may be incomplete.
 */
export function doseFront(graph, start, { exits, forecast, depart = 0, speed = 1.2, blocked = null, maxLabels = 3_000_000 }) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const len = edgeLengths(graph);
  const closes = forecast.closes;
  const isExit = exitFlags(n, exits);
  let cap = 1024;
  let node = new Int32Array(cap), parent = new Int32Array(cap);
  let time = new Float64Array(cap), dose = new Float64Array(cap);
  let count = 0;
  const add = (v, t, d, p) => {
    if (count === cap) {
      cap *= 2;
      const grow = (a, T) => {
        const b = new T(cap);
        b.set(a);
        return b;
      };
      node = grow(node, Int32Array);
      parent = grow(parent, Int32Array);
      time = grow(time, Float64Array);
      dose = grow(dose, Float64Array);
    }
    node[count] = v;
    time[count] = t;
    dose[count] = d;
    parent[count] = p;
    return count++;
  };

  const front = [];
  let truncated = false;
  if (depart < closes[start]) {
    // Least dose among labels settled so far at each node: labels pop in time
    // order, so a new label is beaten on both counts exactly when its dose is
    // not below this.
    const best = new Float64Array(n).fill(Infinity);
    let bestExit = Infinity;
    const heap = new Heap();
    heap.push(depart, add(start, depart, 0, -1));
    while (heap.size) {
      const L = heap.pop();
      const v = node[L], D = dose[L];
      if (!(D < best[v]) || !(D < bestExit)) continue;
      best[v] = D;
      if (isExit[v]) {
        front.push(L);
        bestExit = D;
        continue;
      }
      const t = time[L];
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const e = edgeIds[k];
        if (blocked?.[e]) continue;
        const w = neighbours[k];
        const t2 = t + len[e] / speed;
        if (!(t2 < closes[w])) continue;
        const D2 = D + linkDose(forecast, v, w, t, t2);
        if (!(D2 < best[w]) || !(D2 < bestExit)) continue;
        if (count >= maxLabels) {
          truncated = true;
          break;
        }
        heap.push(t2, add(w, t2, D2, L));
      }
      if (truncated) break;
    }
  }

  // Labels tied in time can settle in either dose order; keep the strict front.
  const kept = [];
  for (const L of front) if (!kept.length || dose[L] < dose[kept[kept.length - 1]]) kept.push(L);
  return {
    front: kept.map((L) => {
      const chain = [];
      for (let c = L; c >= 0; c = parent[c]) chain.push(c);
      chain.reverse();
      return {
        nodes: chain.map((c) => node[c]),
        times: chain.map((c) => time[c]),
        doses: chain.map((c) => dose[c]),
        arrival: time[L],
        dose: dose[L],
      };
    }),
    labels: count,
    truncated,
  };
}

/**
 * Walk a fixed node route from `depart` and report what the forecast says
 * about it: arrival times, the dose, and the first node it reaches at or after
 * that node closes (-1 if none).
 */
export function walkRoute(graph, nodes, { forecast, depart = 0, speed = 1.2 }) {
  const P = graph.nodes;
  const times = [depart];
  let dose = 0;
  let firstUnsafe = depart < forecast.closes[nodes[0]] ? -1 : 0;
  for (let i = 1; i < nodes.length; i += 1) {
    const a = nodes[i - 1] * 3, b = nodes[i] * 3;
    const t0 = times[i - 1];
    const t1 = t0 + Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]) / speed;
    times.push(t1);
    if (firstUnsafe < 0 && !(t1 < forecast.closes[nodes[i]])) firstUnsafe = i;
    if (t1 < forecast.horizon) dose += linkDose(forecast, nodes[i - 1], nodes[i], t0, t1);
  }
  return { times, dose, firstUnsafe };
}

/**
 * Everything the "Plan with the forecast" card shows, for one start, one or
 * more exits and a departure time.
 */
export function planWithForecast(graph, { start, exits, forecast, depart = 0, speed = 1.2, blocked = null }) {
  const t0 = performance.now();
  const ea = earliestArrival(graph, start, depart, { exits, closes: forecast.closes, speed, blocked });
  const earliest = ea.exit >= 0 ? (() => {
    const nodes = unwind(ea.parent, ea.exit);
    return { nodes, times: nodes.map((v) => ea.arrive[v]), arrival: ea.arrive[ea.exit], ...margin(nodes, ea.arrive, forecast) };
  })() : null;
  const tEarliest = performance.now();
  const { front, labels, truncated } = doseFront(graph, start, { exits, forecast, depart, speed, blocked });
  const tDose = performance.now();

  // A route planned once on the fire as it stands at departure: the nodes
  // already untenable are closed, the rest are open whatever comes later.
  let snapshot = null;
  if (exits.length) {
    const m = graph.edges.length / 2;
    const closedNow = new Uint8Array(m);
    for (let e = 0; e < m; e += 1) {
      if (blocked?.[e]) closedNow[e] = 1;
      else if (!(depart < forecast.untenable[graph.edges[e * 2]]) || !(depart < forecast.untenable[graph.edges[e * 2 + 1]])) closedNow[e] = 1;
    }
    // One search to the nearest exit by walking distance, whatever the number
    // of exits (one A* per exit took 3 s on a model with 1,125 of them).
    const open = new Float64Array(graph.nodes.length / 3).fill(Infinity);
    const near = earliestArrival(graph, start, depart, { exits, closes: open, speed, blocked: closedNow });
    const bestPlan = near.exit >= 0 ? { nodes: unwind(near.parent, near.exit) } : null;
    if (bestPlan) {
      const walked = walkRoute(graph, bestPlan.nodes, { forecast, depart, speed });
      const i = walked.firstUnsafe;
      snapshot = {
        nodes: bestPlan.nodes,
        arrival: walked.times[walked.times.length - 1],
        safe: i < 0,
        // How late it reaches the first node that has closed by then.
        unsafeAt: i < 0 ? null : walked.times[i],
        lateBy: i < 0 ? null : walked.times[i] - forecast.closes[bestPlan.nodes[i]],
        beyondForecast: i >= 0 && forecast.untenable[bestPlan.nodes[i]] === Infinity,
      };
    }
  }
  return {
    earliest,
    leastDose: front.length ? front[front.length - 1] : null,
    fastest: front.length ? front[0] : null,
    front: front.map(({ arrival, dose }) => ({ arrival, dose })),
    labels,
    truncated,
    snapshot,
    ms: { earliest: tEarliest - t0, dose: tDose - tEarliest },
  };
}

/** The tightest gap between reaching a node of the route and that node closing. */
function margin(nodes, arrive, forecast) {
  let slack = Infinity, at = -1;
  for (const v of nodes) {
    const s = forecast.untenable[v] - arrive[v];
    if (s < slack) {
      slack = s;
      at = v;
    }
  }
  return { slack, slackNode: at };
}

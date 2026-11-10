/**
 * Guided runs: presets on the welcome screen that step through the app as a
 * user would (open an example, build the graph, find a route, start a fire)
 * and stop at the result. App.jsx advances them; this file holds the presets
 * and the choices they make on the graph.
 */

export const GUIDED_RUNS = [
  {
    id: "way-out",
    title: "The way out",
    description: "The Duplex: its walkable graph, then the shortest route from the room furthest from an exit.",
    example: "duplex-apartment",
    fire: null,
    // The field note it ends on (core/fieldNotes.js).
    note: "walk-out",
  },
  {
    id: "fire-on-route",
    title: "A fire on the route",
    description: "The same route, then a fire part-way along it: heat spreads through the graph and the route bends round it.",
    example: "duplex-apartment",
    fire: { model: "temperature", along: 0.45 },
    note: "reroute",
  },
  {
    id: "smoke-doors",
    title: "Smoke behind closed doors",
    description: "Flamescapes, from the most likely ignition point, with every door closed: heat and smoke over 60 s (a minute or more to compute).",
    example: "duplex-apartment",
    fire: { model: "flamescapes", doors: "closed", horizon: 60 },
    note: "closed-doors",
  },
  {
    id: "vienna-gpu",
    title: "Vienna on the GPU",
    description:
      "The TU Wien building end to end: its graph and way out, a Flamescapes fire from the likeliest origin near the route (on your GPU where the browser offers WebGPU), everyone leaving with the fire, the forecast's escape, then a tour as it all plays again. A few minutes.",
    example: "vienna-escape-route-model",
    fire: { model: "flamescapes", doors: "open", horizon: 120, phoneHorizon: 60 },
    // The extra stages: people leaving with the fire, the forecast's plan, a tour.
    people: true,
    plan: true,
    tour: true,
    note: "ignition",
  },
];

/** The stages a guided run passes through, in order, with what each shows. */
export const GUIDED_STAGES = [
  { id: "open", caption: "Opening the example" },
  { id: "graph", caption: "Building the walkable graph" },
  { id: "start", caption: "Starting in the room furthest from an exit" },
  { id: "exit", caption: "Taking the nearest exit door" },
  { id: "route", caption: "Finding the shortest route" },
  { id: "origin", caption: "Placing the fire" },
  { id: "people", caption: "Everyone on the floors, ready to leave with the fire" },
  { id: "fire", caption: "The fire spreads; the route reacts" },
  { id: "plan", caption: "The forecast's earliest safe way out" },
  { id: "tour", caption: "A tour round the building as it all plays again" },
  { id: "done", caption: "Done: the result is in the panel" },
];

export const stagesFor = (run) =>
  GUIDED_STAGES.filter(
    (s) =>
      (run.fire || (s.id !== "origin" && s.id !== "fire")) &&
      (run.people || s.id !== "people") &&
      (run.plan || s.id !== "plan") &&
      (run.tour || s.id !== "tour"),
  );

/**
 * What the fire stage does next: "wait" until the fire has been seen
 * running and has ended, then "advance" if it ended by itself. A fire the
 * user stopped ends the run ("stopped"): taken as ended, the plan stage
 * computed the stopped run again under its own time limit
 * (wiki/architecture/state-machines.md, SM-3).
 */
export function fireStageNext({ sawRunning, running, step, stopped }) {
  if (!sawRunning || running || !(step > 0)) return "wait";
  return stopped ? "stopped" : "advance";
}

/** Links from the nearest exit to every node (breadth first, skipping links blocked by walls); -1 for none. */
export function hopsFromExits(graph, exitNodes) {
  const n = graph.nodes.length / 3;
  const E = graph.edges.length / 2;
  const head = new Int32Array(n).fill(-1);
  const next = new Int32Array(2 * E);
  const to = new Int32Array(2 * E);
  let m = 0;
  for (let e = 0; e < E; e += 1) {
    if (graph.blocked?.[e]) continue;
    const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
    to[m] = b; next[m] = head[a]; head[a] = m++;
    to[m] = a; next[m] = head[b]; head[b] = m++;
  }
  const hops = new Int32Array(n).fill(-1);
  const queue = new Int32Array(n);
  let qh = 0, qt = 0;
  for (const x of exitNodes || []) {
    if (x >= 0 && x < n && hops[x] < 0) {
      hops[x] = 0;
      queue[qt++] = x;
    }
  }
  while (qh < qt) {
    const v = queue[qh++];
    for (let k = head[v]; k >= 0; k = next[k]) {
      if (hops[to[k]] < 0) {
        hops[to[k]] = hops[v] + 1;
        queue[qt++] = to[k];
      }
    }
  }
  return hops;
}

/** The floor node with the most links between it and the nearest exit (hopsFromExits), or -1. */
export function farthestFromExits(graph, exitNodes) {
  const n = graph.nodes.length / 3;
  const hops = hopsFromExits(graph, exitNodes);
  let best = -1;
  for (let v = 0; v < n; v += 1) {
    if ((graph.kinds?.[v] ?? 0) !== 0 || hops[v] < 0) continue;
    if (best < 0 || hops[v] > hops[best]) best = v;
  }
  return best;
}

/**
 * Where a guided run lights its fire, so that the fire matters to the way
 * out and the solver has a room to burn: of the first `top` likely ignition
 * points (core/ignition.js, ranked), each taken to its nearest node as
 * useIgnition takes it (height counting four times) and dropped if that
 * node reaches no exit, or moved to a floor node within 3 m if it is a
 * door or stair, the one nearest the route (and its node), at least `clear` m from the
 * start and from every exit. With none, the floor node `along` the route.
 *
 * @returns {{node: number, candidate: object | null} | null}
 */
export function guidedOrigin(graph, { candidates = [], route = [], exits = [], start = -1, clear = 3, along = 0.45, top = 8 } = {}) {
  const n = graph.nodes.length / 3;
  const P = graph.nodes;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const hops = hopsFromExits(graph, exits);
  const floor = (v) => (graph.kinds?.[v] ?? 0) === 0 && hops[v] >= 0;
  const dist = (v, w) => Math.hypot(P[v * 3] - P[w * 3], P[v * 3 + 1] - P[w * 3 + 1], P[v * 3 + 2] - P[w * 3 + 2]);
  const awayFrom = [start, ...exits].filter((v) => v >= 0);
  const toRoute = (v) => (route.length ? Math.min(...route.map((w) => dist(v, w))) : 0);
  let best = null;
  for (const candidate of candidates.slice(0, top)) {
    const [x, y, z] = candidate.position || [];
    if (![x, y, z].every(Number.isFinite)) continue;
    const nearestTo = (ok) => {
      let node = -1, nearest = Infinity;
      for (let v = 0; v < n; v += 1) {
        if (!ok(v)) continue;
        const d = [P[v * 3] - x, P[v * 3 + 1] - y, P[v * 3 + 2] - z];
        d[up] *= 4;
        const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (dd < nearest) {
          nearest = dd;
          node = v;
        }
      }
      return node;
    };
    const snapped = nearestTo(() => true);
    if (snapped < 0 || hops[snapped] < 0) continue;
    let node = snapped;
    if (!floor(node)) {
      node = nearestTo(floor);
      if (node < 0 || dist(node, snapped) > 3) continue;
    }
    if (awayFrom.some((w) => dist(node, w) < clear)) continue;
    // Near the route, and near the graph itself.
    const score = toRoute(node) + Math.hypot(P[node * 3] - x, P[node * 3 + 1] - y, P[node * 3 + 2] - z);
    if (!best || score < best.score) best = { node, candidate, score };
  }
  if (best) return { node: best.node, candidate: best.candidate };
  // None qualifies: a floor node part-way along the route.
  const fraction = Math.min(1, Math.max(0, along));
  const ordered = route.map((v, k) => [v, Math.abs(k / Math.max(1, route.length - 1) - fraction)]).sort((a, b) => a[1] - b[1]);
  const fallback = ordered.find(([v]) => floor(v) && !awayFrom.some((w) => w !== v && dist(v, w) < clear))?.[0] ?? nodeAlong(route, fraction);
  return fallback >= 0 ? { node: fallback, candidate: null } : null;
}

/** The node `fraction` of the way along a route's nodes (0 the start, 1 the exit). */
export function nodeAlong(route, fraction) {
  if (!route?.length) return -1;
  return route[Math.min(route.length - 1, Math.max(0, Math.round(fraction * (route.length - 1))))];
}

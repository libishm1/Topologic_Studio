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
];

/** The stages a guided run passes through, in order, with what each shows. */
export const GUIDED_STAGES = [
  { id: "open", caption: "Opening the example" },
  { id: "graph", caption: "Building the walkable graph" },
  { id: "start", caption: "Starting in the room furthest from an exit" },
  { id: "exit", caption: "Taking the nearest exit door" },
  { id: "route", caption: "Finding the shortest route" },
  { id: "origin", caption: "Placing the fire" },
  { id: "fire", caption: "The fire spreads; the route reacts" },
  { id: "done", caption: "Done: the result is in the panel" },
];

export const stagesFor = (run) => GUIDED_STAGES.filter((s) => run.fire || (s.id !== "origin" && s.id !== "fire"));

/**
 * The floor node with the most links between it and the nearest exit
 * (breadth first from every exit, skipping links blocked by walls), or -1.
 */
export function farthestFromExits(graph, exitNodes) {
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
  let best = -1;
  for (let v = 0; v < n; v += 1) {
    if ((graph.kinds?.[v] ?? 0) !== 0 || hops[v] < 0) continue;
    if (best < 0 || hops[v] > hops[best]) best = v;
  }
  return best;
}

/** The node `fraction` of the way along a route's nodes (0 the start, 1 the exit). */
export function nodeAlong(route, fraction) {
  if (!route?.length) return -1;
  return route[Math.min(route.length - 1, Math.max(0, Math.round(fraction * (route.length - 1))))];
}

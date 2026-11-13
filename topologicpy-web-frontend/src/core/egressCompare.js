/**
 * The shortest walks against the slime mould's desire lines, on one graph
 * (wiki/research/egress-vs-slime-2026-11-12.md). The egress map
 * (egressMap.js) sends every room's worst place along its shortest walk to
 * its nearest exit, each room alone; the slime (slimeEgress.js) is a flux of
 * agents that follow the walk to the exits plus each other's trail, one
 * agent a node, starting again anywhere on the floor after leaving. Four
 * readings say where and how far they part:
 *
 * - `onShortest`: the share of the slime's trail within `near` metres of a
 *   shortest walk (1 when the slime only uses the shortest walks);
 * - `covered`: the share of the shortest walks' nodes within `near` metres
 *   of a desire line (a node of weight `lineMin` or more);
 * - `exits`: each exit's share of the floor it is nearest for, against its
 *   share of the slime's traffic, and the total variation between the two
 *   (0 when the slime leaves by each exit as the floor's nearest-exit split
 *   would; 1 when wholly apart);
 * - `atWalls`: the share of the shortest walks' nodes next to a wall (a node
 *   with fewer links than the lattice's most common count), against the
 *   share of the slime's trail on such nodes.
 */

/** Within this many metres a trail or a node counts as on the other's line. */
export const COMPARE_NEAR = 1;
/** A desire line: a node whose slime weight is at least this (weights run 0 to 1). */
export const LINE_MIN = 0.25;

/** Each node's neighbours from the graph's edge pairs. */
function neighbours(graph) {
  const n = graph.nodes.length / 3;
  const adj = Array.from({ length: n }, () => []);
  const E = graph.edges;
  for (let k = 0; k + 1 < E.length; k += 2) {
    adj[E[k]].push(E[k + 1]);
    adj[E[k + 1]].push(E[k]);
  }
  return adj;
}

/** The nodes within `hops` links of any node in `seed`. */
function grow(adj, seed, hops) {
  const mark = new Uint8Array(adj.length);
  let front = [];
  for (const v of seed) if (!mark[v]) {
    mark[v] = 1;
    front.push(v);
  }
  for (let h = 0; h < hops && front.length; h += 1) {
    const next = [];
    for (const u of front) for (const w of adj[u]) if (!mark[w]) {
      mark[w] = 1;
      next.push(w);
    }
    front = next;
  }
  return mark;
}

/**
 * The comparison. `egress` is egressMap's result (rooms with `route`, exits
 * with `catchment.area`, `field.exit`); `slime` is slimeEgress's (`weights`,
 * `exits` with `node` and `flux`). Null when either is missing.
 */
export function compareEgressSlime(graph, egress, slime, { near = COMPARE_NEAR, lineMin = LINE_MIN } = {}) {
  if (!graph || !egress?.rooms || !slime?.weights) return null;
  const n = graph.nodes.length / 3;
  const adj = neighbours(graph);
  const hops = Math.max(1, Math.round(near / (graph.spacing || 0.5)));
  const W = slime.weights;

  // Each room's walk: its route, or (as egressRun.js runMap sends it, routes
  // dropped) the descent through the walk field from its worst place, each
  // step to the neighbour that keeps dist[w] + |vw| least.
  const P = graph.nodes;
  const dist = egress.field?.dist;
  const gap = (a, b) => Math.hypot(P[a * 3] - P[b * 3], P[a * 3 + 1] - P[b * 3 + 1], P[a * 3 + 2] - P[b * 3 + 2]);
  const descend = (v) => {
    const path = [v];
    for (let k = 0; k < n && dist[v] > 1e-6; k += 1) {
      let best = -1, bestCost = Infinity;
      for (const w of adj[v]) {
        const c = dist[w] + gap(v, w);
        if (dist[w] < dist[v] && c < bestCost) [best, bestCost] = [w, c];
      }
      if (best < 0) break;
      v = best;
      path.push(v);
    }
    return path;
  };
  const shortest = new Set();
  for (const r of egress.rooms) {
    const route = r.route?.length ? r.route : dist && r.node >= 0 && Number.isFinite(r.dist) ? descend(r.node) : [];
    for (const v of route) shortest.add(v);
  }
  const lines = [];
  for (let v = 0; v < n; v += 1) if (W[v] >= lineMin) lines.push(v);

  const nearShortest = grow(adj, shortest, hops);
  const nearLines = grow(adj, lines, hops);
  let total = 0, onShortest = 0;
  for (let v = 0; v < n; v += 1) {
    total += W[v];
    if (nearShortest[v]) onShortest += W[v];
  }
  let covered = 0;
  for (const v of shortest) if (nearLines[v]) covered += 1;

  // Next to a wall: fewer links than the lattice's most common count.
  const counts = new Map();
  for (let v = 0; v < n; v += 1) counts.set(adj[v].length, (counts.get(adj[v].length) || 0) + 1);
  const interior = [...counts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 8;
  const wall = (v) => adj[v].length < interior;
  let shortestAtWall = 0, trailAtWall = 0;
  for (const v of shortest) if (wall(v)) shortestAtWall += 1;
  for (let v = 0; v < n; v += 1) if (wall(v)) trailAtWall += W[v];

  // Exits: the floor's nearest-exit split against the slime's traffic.
  const area = egress.exits.map((e) => e.catchment?.area || 0);
  const areaSum = area.reduce((a, b) => a + b, 0);
  const flux = egress.exits.map(() => 0);
  for (const e of slime.exits || []) {
    const g = egress.field?.exit?.[e.node];
    if (g >= 0 && g < flux.length) flux[g] += e.flux;
  }
  const fluxSum = flux.reduce((a, b) => a + b, 0);
  const exits = egress.exits.map((e, i) => ({ label: e.label, floorShare: areaSum ? area[i] / areaSum : 0, slimeShare: fluxSum ? flux[i] / fluxSum : 0 }));
  const apart = 0.5 * exits.reduce((s, e) => s + Math.abs(e.floorShare - e.slimeShare), 0);

  return {
    onShortest: total ? onShortest / total : 0,
    covered: shortest.size ? covered / shortest.size : 0,
    exits,
    exitsApart: fluxSum && areaSum ? apart : null,
    atWalls: { shortest: shortest.size ? shortestAtWall / shortest.size : 0, slime: total ? trailAtWall / total : 0 },
    route: slime.route ? { slime: slime.route.length, shortest: slime.route.shortest } : null,
    near,
    lineMin,
  };
}

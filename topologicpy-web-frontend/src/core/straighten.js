/**
 * Straight routes, as TopologicPy's Wire.Straighten draws them
 * (ForNeha.ipynb: `Wire.Straighten(sp, host=host, obstacles=[])`). A lattice
 * route turns only at 45 and 90 degrees, so across an open room it zigzags.
 * Each run of it is replaced by the longest straight segments that stay over
 * walkable floor (the host) and clear of the walls (the obstacles).
 *
 * Doors and stairs stay put, as Wire.Straighten keeps the crossing of each
 * portal: the straight route still goes through the doorway the lattice route
 * used and climbs the same flight. From each kept vertex the next one is the
 * farthest later vertex in clear sight, as in Wire.Straighten, looking a few
 * vertices past a blocked one rather than to the end of the route. A vertex
 * with no clear shortcut keeps its original edge, so the result is never
 * longer than the lattice route.
 *
 * "Over walkable floor" is tested on the graph itself: every point along the
 * segment must lie within a quarter step of an open link at the same level.
 * The links of a fully walkable lattice square (four sides, two diagonals)
 * leave no point further than 0.21 steps from one, so open floor always
 * passes, while a shortcut over a void or outside the slab edge fails once it
 * strays more than about 12 cm (at 0.5 m spacing) past the last row of nodes.
 * "Clear of the walls" uses the wall sections the lattice was blocked with,
 * and keeps a shoulder's width (0.25 m) off them, so a route does not graze
 * a corner. Wire.Straighten has no clearance; without one, routes on the
 * example models passed corners at 3 mm. Where a route starts or ends nearer
 * a wall than that (the lattice's wall-face nodes sit 2 cm off), the required
 * clearance tapers to the endpoint's own.
 */
import { KIND_FLOOR, SegmentIndex, axisIndex, horizontalAxes } from "./lattice.js";

const KEY_OFFSET = 1 << 20;
const cellKey = (i, j) => (i + KEY_OFFSET) * 2 ** 21 + (j + KEY_OFFSET);

/** Per-graph lookups, built once and cached on the graph. */
function context(graph) {
  if (graph._straight) return graph._straight;
  const up = axisIndex(graph.upAxis ?? "y");
  const [h0, h1] = horizontalAxes(graph.upAxis ?? "y");
  const lift = graph.agentHeight ?? 0.75;
  const s = graph.spacing ?? graph.lattice?.spacing ?? 0.5;
  const P = graph.nodes;
  const E = graph.edges;
  const add = (map, k, v) => {
    const list = map.get(k);
    if (list) list.push(v);
    else map.set(k, [v]);
  };
  const nodeCells = new Map();
  for (let v = 0; v < P.length / 3; v += 1) {
    add(nodeCells, cellKey(Math.floor(P[v * 3 + h0] / s), Math.floor(P[v * 3 + h1] / s)), v);
  }
  const edgeCells = new Map();
  for (let e = 0; e < E.length / 2; e += 1) {
    const a = E[e * 2], b = E[e * 2 + 1];
    const x0 = Math.floor(Math.min(P[a * 3 + h0], P[b * 3 + h0]) / s);
    const x1 = Math.floor(Math.max(P[a * 3 + h0], P[b * 3 + h0]) / s);
    const y0 = Math.floor(Math.min(P[a * 3 + h1], P[b * 3 + h1]) / s);
    const y1 = Math.floor(Math.max(P[a * 3 + h1], P[b * 3 + h1]) / s);
    for (let i = x0; i <= x1; i += 1) for (let j = y0; j <= y1; j += 1) add(edgeCells, cellKey(i, j), e);
  }
  const sections = new Map();
  for (const { key, segs } of graph.sections || []) sections.set(key, { segs, index: null });
  graph._straight = { up, h0, h1, lift, s, nodeCells, edgeCells, sections, hasSections: sections.size > 0 };
  return graph._straight;
}

function sectionIndex(ctx, floor, sectionHeight) {
  const k = Math.round((floor + sectionHeight) * 10);
  // Sections are cached per 10 cm of floor height; a ramp's ends can fall
  // either side of a step, so take the nearest one built.
  for (const key of [k, k - 1, k + 1]) {
    const entry = ctx.sections.get(key);
    if (!entry) continue;
    if (!entry.index) entry.index = new SegmentIndex(entry.segs);
    return entry.index;
  }
  return null;
}

/** Distance from (x, y) to the nearest outline segment, capped at `cap`. */
function wallDistance(index, x, y, cap) {
  let best = cap;
  const g = index.segs;
  index.near(x - cap, y - cap, x + cap, y + cap, (s) => {
    const d = distanceToSegment(x, y, g[s * 4], g[s * 4 + 1], g[s * 4 + 2], g[s * 4 + 3]);
    if (d < best) best = d;
  });
  return best;
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}

/** Whether (px, py) at floor height pf lies within reach of an open link. */
function onFloor(graph, ctx, px, py, pf, opts) {
  const { up, h0, h1, lift, s, edgeCells } = ctx;
  const P = graph.nodes, E = graph.edges;
  const r = 0.25 * s;
  for (let i = Math.floor((px - r) / s); i <= Math.floor((px + r) / s); i += 1) {
    for (let j = Math.floor((py - r) / s); j <= Math.floor((py + r) / s); j += 1) {
      for (const e of edgeCells.get(cellKey(i, j)) || []) {
        if (opts.useWalls && graph.blocked?.[e]) continue;
        const a = E[e * 2], b = E[e * 2 + 1];
        if (opts.avoid && (opts.avoid[a] || opts.avoid[b])) continue;
        const fa = P[a * 3 + up] - lift, fb = P[b * 3 + up] - lift;
        if (pf < Math.min(fa, fb) - opts.maxRise || pf > Math.max(fa, fb) + opts.maxRise) continue;
        if (distanceToSegment(px, py, P[a * 3 + h0], P[a * 3 + h1], P[b * 3 + h0], P[b * 3 + h1]) <= r) return true;
      }
    }
  }
  return false;
}

/** Whether a burning node stands within half a step of (px, py). */
function nearFire(graph, ctx, px, py, pf, opts) {
  const { up, h0, h1, lift, s, nodeCells } = ctx;
  const P = graph.nodes;
  const ci = Math.floor(px / s), cj = Math.floor(py / s);
  for (let di = -1; di <= 1; di += 1) {
    for (let dj = -1; dj <= 1; dj += 1) {
      for (const v of nodeCells.get(cellKey(ci + di, cj + dj)) || []) {
        if (!opts.avoid[v] || Math.abs(P[v * 3 + up] - lift - pf) > opts.maxRise) continue;
        if (Math.hypot(P[v * 3 + h0] - px, P[v * 3 + h1] - py) <= 0.5 * s) return true;
      }
    }
  }
  return false;
}

/**
 * Whether the straight segment from node a to node b may replace the lattice
 * route between them.
 */
function clear(graph, ctx, a, b, opts) {
  const { up, h0, h1, lift, s } = ctx;
  const P = graph.nodes;
  const ax = P[a * 3 + h0], ay = P[a * 3 + h1], fa = P[a * 3 + up] - lift;
  const bx = P[b * 3 + h0], by = P[b * 3 + h1], fb = P[b * 3 + up] - lift;
  if (Math.abs(fa - fb) > opts.maxRise) return false;
  const length = Math.hypot(bx - ax, by - ay);
  if (length < 1e-9) return true;

  let index = null;
  if (opts.useWalls) {
    index = sectionIndex(ctx, Math.min(fa, fb), opts.sectionHeight);
    if (!index || index.crosses(ax, ay, bx, by)) return false;
  }
  const c = opts.clearance;
  const ca = index && c > 0 ? wallDistance(index, ax, ay, c) : 0;
  const cb = index && c > 0 ? wallDistance(index, bx, by, c) : 0;

  const samples = Math.ceil(length / (s / 4));
  for (let k = 1; k < samples; k += 1) {
    const t = k / samples;
    const px = ax + (bx - ax) * t, py = ay + (by - ay) * t, pf = fa + (fb - fa) * t;
    if (!onFloor(graph, ctx, px, py, pf, opts)) return false;
    if (opts.avoid && nearFire(graph, ctx, px, py, pf, opts)) return false;
    // Through solid without crossing an outline: only possible exactly
    // through an outline vertex, so once a grid step is enough.
    if (index && k % 4 === 0 && index.inside(px, py)) return false;
    if (index && c > 0) {
      const need = Math.min(ca + (cb - ca) * t, c) - 1e-6;
      if (need > 0 && wallDistance(index, px, py, need) < need) return false;
    }
  }
  return true;
}

/**
 * Straighten a route given as node ids.
 *
 * @param {object} opts  useWalls: test against the wall sections (default
 *                       true); avoid: per-node flags (burning nodes) the
 *                       route must keep half a step away from; clearance:
 *                       metres to keep off the walls (default 0.25);
 *                       lookahead: how many blocked vertices to look past for
 *                       a farther clear one.
 * @returns {{node_ids: number[], points: number[][], length: number}}
 */
export function straightenRoute(
  graph,
  nodeIds,
  { useWalls = true, avoid = null, clearance = 0.25, maxRise = 0.35, sectionHeight = graph.sectionHeight ?? 0.75, lookahead = 8 } = {},
) {
  const P = graph.nodes;
  const pointOf = (i) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
  const finish = (ids) => {
    const points = ids.map(pointOf);
    let length = 0;
    for (let i = 1; i < points.length; i += 1) length += Math.hypot(...points[i].map((v, k) => v - points[i - 1][k]));
    return { node_ids: ids, points, length };
  };
  if (nodeIds.length < 3) return finish([...nodeIds]);
  const ctx = context(graph);
  // Without sections there is nothing to tell a wall from a gap between two
  // lattice rows, so leave the route as it is.
  if (useWalls && !ctx.hasSections) return finish([...nodeIds]);
  const opts = { useWalls, avoid, clearance, maxRise, sectionHeight };

  const kinds = graph.kinds;
  const last = nodeIds.length - 1;
  const pinned = (i) => i === 0 || i === last || (kinds && kinds[nodeIds[i]] !== KIND_FLOOR);
  const out = [nodeIds[0]];
  let i = 0;
  while (i < last) {
    // The next pinned vertex bounds the search: the route keeps it.
    let stop = i + 1;
    while (!pinned(stop)) stop += 1;
    let best = i + 1;
    let misses = 0;
    for (let j = i + 2; j <= stop; j += 1) {
      if (clear(graph, ctx, nodeIds[i], nodeIds[j], opts)) {
        best = j;
        misses = 0;
      } else if (++misses > lookahead) {
        break;
      }
    }
    out.push(nodeIds[best]);
    i = best;
  }
  return finish(out);
}

/**
 * A route with its straightened form added as `straight_points` and
 * `straight_length`; node ids come from `node_ids` (routes) or `path` (RL).
 */
export function withStraight(graph, route, opts = {}) {
  const ids = route?.node_ids?.length ? route.node_ids : route?.path;
  if (!ids?.length || typeof ids[0] !== "number") return route;
  const straight = straightenRoute(graph, ids, opts);
  return { ...route, straight_points: straight.points, straight_length: straight.length };
}

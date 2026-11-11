/**
 * Skeletons of the walkable floor (wiki/roadmap/egress-desire-lines-plan-2026-11-11.md, D1):
 * the medial axis of the floor and the slime mould's desire lines.
 *
 * Clearance. Each walkable node's distance from the floor's boundary: a
 * floor node missing one of its eight lattice links (a wall, a doorway's
 * jamb, the floor's edge) is on the boundary, half a link from it; the rest
 * are measured from there over the open links by a multi-source Dijkstra,
 * the robot planner's "brushfire", which also carries each node's nearest
 * boundary node (its feature). Doors and stairs are passages, never
 * boundary: a doorway's clearance is that of the jambs beside it.
 *
 * The medial axis. Where the nearest walls part: a node whose nearest
 * boundary node and a neighbour's lie at least `lambda` apart in plan, the
 * clearer of the two (the lambda-medial axis, Chazal and Lieutier 2005,
 * Graphical Models 67:304-331, doi:10.1016/j.gmod.2005.01.002: lambda
 * leaves out the axis's twigs, from
 * corners and the walls' small steps, that a strict ridge keeps). It is the
 * generalized Voronoi graph of the free space, the roadmap of maximum
 * clearance (Choset and Burdick 2000, doi:10.1177/02783640022066770), on
 * the lattice: one line through each room and corridor, furthest from both
 * sides, through the middle of each doorway. A stair's walking line is its
 * axis.
 *
 * Desire lines. The slime's ways out from every part of the floor: the
 * walk to the exits where a link costs its length times 1 + tube (1 - its
 * weight) (and, with the soft fire, times 1 + fireCost x its smoke), worked
 * out once from the exits; from one node in each `cell` metres square of
 * each floor the way out is traced along it and every link it passes counts
 * one more. Ways that share the slime's tubes bundle into trunks, as people's
 * trails across a park bundle (Helbing, Keltsch and Molnár 1997,
 * doi:10.1038/40353); a link's count is its flow, and each points towards
 * the exit it leads to. Where they leave the medial axis, the slime, drawn
 * by the exits and kept from the fire, takes another line than the one the
 * building's geometry centres.
 *
 * The medial way out. The cheapest walk from a node to an exit where a
 * link costs its length times 1 + beta / clearance: a route that keeps to
 * the medial axis, as a robot planner keeps clear of obstacles.
 */
import { Heap, csr, edgeLengths } from "./route.js";
import { KIND_DOOR, KIND_FLOOR, KIND_LIFT, KIND_STAIR } from "./lattice.js";

/** Lattice links a floor node away from any boundary has. */
export const FULL_DEGREE = 8;

/** Cosine below which two links from a node point opposite ways (along one lattice line). */
const OPPOSITE = -0.9;

/**
 * The open links of a graph as compressed rows, with lengths and plan
 * directions: no link across a wall (`blocked`), none to or from a closed
 * node (`open[v]` 0) or a lift.
 */
export function openAdjacency(graph, { blocked = null, open = null } = {}) {
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const [h0, h1] = [0, 1, 2].filter((c) => c !== up);
  const kinds = graph.kinds || new Uint8Array(n);
  const P = graph.nodes;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const length = edgeLengths(graph);
  const ok = (v) => (!open || open[v]) && kinds[v] !== KIND_LIFT;
  const start = new Int32Array(n + 1);
  const to = [], len = [], dx = [], dy = [];
  for (let a = 0; a < n; a += 1) {
    start[a] = to.length;
    if (!ok(a)) continue;
    for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const b = neighbours[k];
      if (!ok(b)) continue;
      const ex = P[b * 3 + h0] - P[a * 3 + h0], ey = P[b * 3 + h1] - P[a * 3 + h1];
      const plan = Math.hypot(ex, ey);
      to.push(b);
      len.push(length[edgeIds[k]]);
      dx.push(plan > 1e-6 ? ex / plan : 0);
      dy.push(plan > 1e-6 ? ey / plan : 0);
    }
  }
  start[n] = to.length;
  return { n, up, h0, h1, start, to: Int32Array.from(to), len: Float64Array.from(len), dx: Float32Array.from(dx), dy: Float32Array.from(dy) };
}

/**
 * Each node's distance from the floor's boundary, m (Infinity where none is
 * reached), and its nearest boundary node (-1 for none).
 */
export function clearanceOf(graph, adj, { spacing = graph.spacing || 0.5 } = {}) {
  const { n, start, to, len } = adj;
  const kinds = graph.kinds || new Uint8Array(n);
  const clearance = new Float64Array(n).fill(Infinity);
  const feature = new Int32Array(n).fill(-1);
  const heap = new Heap();
  for (let v = 0; v < n; v += 1) {
    if (start[v + 1] === start[v] && kinds[v] !== KIND_FLOOR) continue;
    const passage = kinds[v] === KIND_DOOR || kinds[v] === KIND_STAIR;
    if (!passage && start[v + 1] - start[v] < FULL_DEGREE) {
      clearance[v] = spacing / 2;
      feature[v] = v;
      heap.push(clearance[v], v);
    }
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const a = heap.pop();
    if (done[a]) continue;
    done[a] = 1;
    for (let k = start[a]; k < start[a + 1]; k += 1) {
      const b = to[k], d = clearance[a] + len[k];
      if (d < clearance[b]) {
        clearance[b] = d;
        feature[b] = feature[a];
        heap.push(d, b);
      }
    }
  }
  return { clearance, feature };
}

/**
 * The ridge of a field over the graph: nodes at least as high as both
 * neighbours along some lattice line and higher than one (a missing
 * neighbour counts as `missing`), among those at or above `min`.
 */
export function ridgeOf(adj, field, { min = -Infinity, missing = 0 } = {}) {
  const { n, start, to, dx, dy } = adj;
  const ridge = new Uint8Array(n);
  for (let v = 0; v < n; v += 1) {
    const f = field[v];
    if (!(f >= min) || !Number.isFinite(f)) continue;
    for (let k = start[v]; k < start[v + 1] && !ridge[v]; k += 1) {
      if (dx[k] === 0 && dy[k] === 0) continue;
      const a = field[to[k]];
      // The neighbour the other way along the same line, or none.
      let b = missing;
      for (let j = start[v]; j < start[v + 1]; j += 1) {
        if (dx[j] * dx[k] + dy[j] * dy[k] < OPPOSITE) {
          b = field[to[j]];
          break;
        }
      }
      if (f >= a && f >= b && (f > a || f > b)) ridge[v] = 1;
    }
  }
  return ridge;
}

/** The links whose both ends are marked, as pairs [a, b, ...] (each once). */
export function markedLinks(adj, marked) {
  const out = [];
  for (let a = 0; a < adj.n; a += 1) {
    if (!marked[a]) continue;
    for (let k = adj.start[a]; k < adj.start[a + 1]; k += 1) {
      const b = adj.to[k];
      if (b > a && marked[b]) out.push(a, b);
    }
  }
  return Int32Array.from(out);
}

/**
 * The medial axis of the walkable floor (the lambda-medial axis, lambda in m).
 *
 * @returns {{clearance: Float64Array, axis: Uint8Array, links: Int32Array}}
 */
export function medialAxis(graph, adj, { lambda = 1.0 } = {}) {
  const { clearance, feature } = clearanceOf(graph, adj);
  const { n, start, to, h0, h1 } = adj;
  const P = graph.nodes;
  const kinds = graph.kinds || new Uint8Array(n);
  const axis = new Uint8Array(n);
  const apart = (f, g) => Math.hypot(P[f * 3 + h0] - P[g * 3 + h0], P[f * 3 + h1] - P[g * 3 + h1]);
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] === KIND_STAIR) {
      if (start[v + 1] > start[v]) axis[v] = 1;
      continue;
    }
    const fv = feature[v];
    if (fv < 0) continue;
    for (let k = start[v]; k < start[v + 1]; k += 1) {
      const u = to[k], fu = feature[u];
      if (fu < 0 || fu === fv) continue;
      if (clearance[v] < clearance[u] || (clearance[v] === clearance[u] && v > u)) continue;
      if (apart(fv, fu) >= lambda) {
        axis[v] = 1;
        break;
      }
    }
  }
  return { clearance, axis, links: markedLinks(adj, axis) };
}

/**
 * The slime mould's desire lines: the ways out from every `cell` metres of
 * floor along its tubes, bundled.
 *
 * @param adj      openAdjacency's
 * @param weights  the slime's weights (0 to 1 a node)
 * @param exits    the exit nodes
 * @param opts     tube (2), fireCost and smoke (the soft fire's, or 0 and null), cell (2 m), minFlow (2): links fewer ways pass are left out;
 *                 people: persons on each node (to weigh each way by the people of its cell; null: one a way)
 * @returns {{links: Int32Array, flow: Int32Array, people: Float32Array, on: Uint8Array, maxFlow: number}}
 *   links as directed pairs [from, to, ...], towards the exit; flow a link (ways), people a link (persons); on: nodes on a kept link
 */
export function desireLines(graph, adj, weights, exits, { tube = 2, fireCost = 0, smoke = null, cell = 2, minFlow = 2, people = null } = {}) {
  const { n, start, to, len, up, h0, h1 } = adj;
  const P = graph.nodes;
  const kinds = graph.kinds || new Uint8Array(n);
  // The walk to the exits along the tubes, and each node's next step on it.
  const cost = new Float64Array(n).fill(Infinity);
  const parent = new Int32Array(n).fill(-1);
  const heap = new Heap();
  for (const e of exits) {
    if (e < 0 || cost[e] === 0) continue;
    cost[e] = 0;
    heap.push(0, e);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const a = heap.pop();
    if (done[a]) continue;
    done[a] = 1;
    for (let k = start[a]; k < start[a + 1]; k += 1) {
      const b = to[k];
      const w = (weights[a] + weights[b]) / 2;
      const fire = smoke ? 1 + fireCost * Math.max(smoke[a], smoke[b]) : 1;
      const c = cost[a] + len[k] * (1 + tube * (1 - w)) * fire;
      if (c < cost[b]) {
        cost[b] = c;
        parent[b] = a;
        heap.push(c, b);
      }
    }
  }
  // One floor node a cell (by plan and storey), traced out; a link's flow is the ways through it.
  const seeds = new Map(), folk = new Map();
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] !== KIND_FLOOR || !(cost[v] < Infinity)) continue;
    const key = `${Math.floor(P[v * 3 + h0] / cell)},${Math.floor(P[v * 3 + h1] / cell)},${Math.round(P[v * 3 + up] / 1.5)}`;
    if (!seeds.has(key)) seeds.set(key, v);
    if (people) folk.set(key, (folk.get(key) || 0) + people[v]);
  }
  const up1 = new Int32Array(n);
  const carried = new Float32Array(n);
  for (const [key, s] of seeds) {
    const w = people ? folk.get(key) : 1;
    for (let v = s; parent[v] >= 0; v = parent[v]) {
      up1[v] += 1;
      carried[v] += w;
    }
  }
  const links = [], flow = [], persons = [];
  const on = new Uint8Array(n);
  let maxFlow = 0;
  for (let v = 0; v < n; v += 1) {
    if (parent[v] < 0 || up1[v] < minFlow) continue;
    links.push(v, parent[v]);
    flow.push(up1[v]);
    persons.push(carried[v]);
    on[v] = 1;
    on[parent[v]] = 1;
    if (up1[v] > maxFlow) maxFlow = up1[v];
  }
  return { links: Int32Array.from(links), flow: Int32Array.from(flow), people: Float32Array.from(persons), on, maxFlow };
}

/** Persons a second a metre of effective width passes at the densest flow (SFPE, as performanceCheck.js DOOR_FLOW). */
export const SPECIFIC_FLOW = (1 - 0.266 * 1.9) * 1.4 * 1.9;

/** Metres lost at each side of a passage (performanceCheck.js DOOR_BOUNDARY). */
export const EDGE_LOSS = 0.15;

/**
 * Bottlenecks on the desire lines: each link's people over what its clear
 * width passes, in seconds (the time it would take everyone the slime sends
 * that way to get through): a door's width where a door node is (doorWidth,
 * m a node, 0 elsewhere), else twice the largest clearance within two links
 * (a passage's width at its axis, whichever side of it the line runs), less
 * EDGE_LOSS a side, at SPECIFIC_FLOW. Stairs are left out
 * (their width is not on the graph yet). The `top` worst, at least `apart`
 * metres from each other, from `minSeconds` up.
 *
 * @returns {Array<{node: number, people: number, width: number, seconds: number, door: boolean}>}
 */
export function bottlenecks(graph, links, people, { clearance, doorWidth = null, top = 8, apart = 4, minSeconds = 10, adj = null } = {}) {
  const P = graph.nodes;
  const kinds = graph.kinds || new Uint8Array(P.length / 3);
  // The largest clearance within two links of a node.
  const widest = (v) => {
    let best = clearance[v];
    if (!adj) return best;
    for (let k = adj.start[v]; k < adj.start[v + 1]; k += 1) {
      const u = adj.to[k];
      if (clearance[u] > best && clearance[u] < Infinity) best = clearance[u];
      for (let j = adj.start[u]; j < adj.start[u + 1]; j += 1) if (clearance[adj.to[j]] > best && clearance[adj.to[j]] < Infinity) best = clearance[adj.to[j]];
    }
    return best;
  };
  const found = [];
  for (let k = 0; k < links.length; k += 2) {
    const a = links[k], b = links[k + 1];
    if (kinds[a] === KIND_STAIR || kinds[b] === KIND_STAIR) continue;
    const at = kinds[b] === KIND_DOOR ? b : kinds[a] === KIND_DOOR ? a : b;
    const door = kinds[at] === KIND_DOOR;
    const width = door && doorWidth?.[at] > 0 ? doorWidth[at] : Math.max(0.5, 2 * widest(at));
    const passes = SPECIFIC_FLOW * Math.max(0.1, width - 2 * EDGE_LOSS);
    const seconds = people[k / 2] / passes;
    if (seconds >= minSeconds) found.push({ node: at, people: people[k / 2], width, seconds, door });
  }
  found.sort((x, y) => y.seconds - x.seconds);
  const kept = [];
  for (const f of found) {
    if (kept.length >= top) break;
    const near = kept.some((g) => Math.hypot(P[g.node * 3] - P[f.node * 3], P[g.node * 3 + 1] - P[f.node * 3 + 1], P[g.node * 3 + 2] - P[f.node * 3 + 2]) < apart);
    if (!near) kept.push(f);
  }
  return kept;
}

/**
 * One way out: each node's walk before a second, separate way out opens,
 * the part its shortest walks to its two nearest exits share (the common
 * path of egress travel; a dead end's depth): (dA + dB - dAB) / 2, with dA
 * and dB its walks to the two nearest exits and dAB the walk between them
 * (the Gromov product; on the lattice the two walks often run side by side
 * through different nodes, so comparing their steps would find them apart
 * at once). A node that reaches one exit only is all common path. Worked
 * from a walk tree to each exit, so with more than `maxExits` exits it is
 * not worked out (null).
 *
 * @returns {Float32Array|null}  m a node (Infinity where no exit is reached)
 */
export function commonPath(adj, exits, { maxExits = 32 } = {}) {
  const list = [...new Set(exits.filter((e) => e >= 0))];
  if (!list.length || list.length > maxExits) return null;
  const { n, start, to, len } = adj;
  const trees = list.map((e) => {
    const dist = new Float64Array(n).fill(Infinity);
    const parent = new Int32Array(n).fill(-1);
    const heap = new Heap();
    dist[e] = 0;
    heap.push(0, e);
    const done = new Uint8Array(n);
    while (heap.size) {
      const a = heap.pop();
      if (done[a]) continue;
      done[a] = 1;
      for (let k = start[a]; k < start[a + 1]; k += 1) {
        const b = to[k], d = dist[a] + len[k];
        if (d < dist[b]) {
          dist[b] = d;
          parent[b] = a;
          heap.push(d, b);
        }
      }
    }
    return { dist, parent };
  });
  const out = new Float32Array(n).fill(Infinity);
  for (let v = 0; v < n; v += 1) {
    let first = -1, second = -1;
    for (let g = 0; g < trees.length; g += 1) {
      const d = trees[g].dist[v];
      if (!(d < Infinity)) continue;
      if (first < 0 || d < trees[first].dist[v]) {
        second = first;
        first = g;
      } else if (second < 0 || d < trees[second].dist[v]) second = g;
    }
    if (first < 0) continue;
    const A = trees[first];
    if (second < 0) {
      out[v] = A.dist[v];
      continue;
    }
    const between = A.dist[list[second]];
    out[v] = between < Infinity ? Math.max(0, (A.dist[v] + trees[second].dist[v] - between) / 2) : A.dist[v];
  }
  return out;
}

/**
 * The medial way out: from `from` to the nearest exit by cost length x
 * (1 + beta / clearance). Null when no exit is reached.
 */
export function medialRoute(adj, clearance, from, exits, { beta = 1 } = {}) {
  const { n, start, to, len } = adj;
  const isExit = new Uint8Array(n);
  for (const e of exits) if (e >= 0) isExit[e] = 1;
  const cost = new Float64Array(n).fill(Infinity);
  const via = new Int32Array(n).fill(-1);
  const heap = new Heap();
  cost[from] = 0;
  heap.push(0, from);
  const done = new Uint8Array(n);
  while (heap.size) {
    const a = heap.pop();
    if (done[a]) continue;
    done[a] = 1;
    if (isExit[a]) {
      const nodes = [];
      for (let v = a; v >= 0; v = via[v]) nodes.push(v);
      nodes.reverse();
      let walked = 0;
      for (let i = 1; i < nodes.length; i += 1) {
        for (let k = start[nodes[i - 1]]; k < start[nodes[i - 1] + 1]; k += 1) {
          if (to[k] === nodes[i]) {
            walked += len[k];
            break;
          }
        }
      }
      return { nodes, length: walked };
    }
    for (let k = start[a]; k < start[a + 1]; k += 1) {
      const b = to[k];
      const c = Math.max(1e-3, Math.min(clearance[a], clearance[b]));
      const d = cost[a] + len[k] * (1 + beta / c);
      if (d < cost[b]) {
        cost[b] = d;
        via[b] = a;
        heap.push(d, b);
      }
    }
  }
  return null;
}

/** The share of `marked` nodes within `reach` links of an `axis` node: how far the desire lines keep to the medial axis. */
export function agreement(adj, marked, axis, { reach = 1 } = {}) {
  const { n, start, to } = adj;
  const near = new Uint8Array(n);
  let frontier = [];
  for (let v = 0; v < n; v += 1) if (axis[v]) {
    near[v] = 1;
    frontier.push(v);
  }
  for (let r = 0; r < reach; r += 1) {
    const next = [];
    for (const a of frontier) for (let k = start[a]; k < start[a + 1]; k += 1) if (!near[to[k]]) {
      near[to[k]] = 1;
      next.push(to[k]);
    }
    frontier = next;
  }
  let on = 0, all = 0;
  for (let v = 0; v < n; v += 1) if (marked[v]) {
    all += 1;
    if (near[v]) on += 1;
  }
  return all ? on / all : null;
}

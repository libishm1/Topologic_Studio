/**
 * Routing in the browser: A* over the lattice graph, the same rules as the
 * server's fast engine (topologicpy-web-backend/app/graphs/pathfinding.py),
 * so a route needs no network round trip. Measured on the Duplex graph in
 * Chrome, the server's own code took 7 ms under Pyodide against 3.5 ms
 * native; the round trip to Render was 310 ms on top of either.
 */

/** Adjacency in CSR form, both directions of every edge. Cached on the graph. */
export function csr(graph) {
  if (graph._csr) return graph._csr;
  const n = graph.nodes.length / 3;
  const m = graph.edges.length / 2;
  const degree = new Int32Array(n + 1);
  for (let e = 0; e < m; e += 1) {
    degree[graph.edges[e * 2] + 1] += 1;
    degree[graph.edges[e * 2 + 1] + 1] += 1;
  }
  for (let i = 0; i < n; i += 1) degree[i + 1] += degree[i];
  const indptr = degree;
  const fill = indptr.slice(0, n);
  const neighbours = new Int32Array(m * 2);
  const edgeIds = new Int32Array(m * 2);
  for (let e = 0; e < m; e += 1) {
    const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
    neighbours[fill[a]] = b;
    edgeIds[fill[a]++] = e;
    neighbours[fill[b]] = a;
    edgeIds[fill[b]++] = e;
  }
  graph._csr = { indptr, neighbours, edgeIds, n, m };
  return graph._csr;
}

/** Component label per node (walls on), cached. */
export function componentLabels(graph) {
  if (graph._labels) return graph._labels;
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const labels = new Int32Array(n).fill(-1);
  const sizes = [];
  const stack = [];
  for (let s = 0; s < n; s += 1) {
    if (labels[s] >= 0) continue;
    const id = sizes.length;
    let size = 0;
    labels[s] = id;
    stack.push(s);
    while (stack.length) {
      const u = stack.pop();
      size += 1;
      for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
        if (graph.blocked?.[edgeIds[k]]) continue;
        const v = neighbours[k];
        if (labels[v] < 0) {
          labels[v] = id;
          stack.push(v);
        }
      }
    }
    sizes.push(size);
  }
  graph._labels = { labels, sizes };
  return graph._labels;
}

/**
 * Nearest node to a picked point, among parts big enough to be real (at least
 * 20 nodes or 1% of the graph). Snapping only into the largest part dragged a
 * pick in the Duplex's second dwelling into the first.
 */
export function snap(graph, point) {
  const { labels, sizes } = componentLabels(graph);
  const n = graph.nodes.length / 3;
  const min = Math.max(20, Math.floor(0.01 * n));
  let best = -1, bestD = Infinity, fallback = -1, fallbackD = Infinity;
  for (let i = 0; i < n; i += 1) {
    const dx = graph.nodes[i * 3] - point[0];
    const dy = graph.nodes[i * 3 + 1] - point[1];
    const dz = graph.nodes[i * 3 + 2] - point[2];
    const d = dx * dx + dy * dy + dz * dz;
    if (d < fallbackD) {
      fallbackD = d;
      fallback = i;
    }
    if (sizes[labels[i]] >= min && d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best >= 0 ? best : fallback;
}

/** A binary min-heap of [priority, node]. */
class Heap {
  constructor() {
    this.p = [];
    this.v = [];
  }
  get size() {
    return this.p.length;
  }
  push(priority, value) {
    const p = this.p, v = this.v;
    let i = p.length;
    p.push(priority);
    v.push(value);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (p[parent] <= p[i]) break;
      [p[parent], p[i]] = [p[i], p[parent]];
      [v[parent], v[i]] = [v[i], v[parent]];
      i = parent;
    }
  }
  pop() {
    const p = this.p, v = this.v;
    const top = v[0];
    const lastP = p.pop(), lastV = v.pop();
    if (p.length) {
      p[0] = lastP;
      v[0] = lastV;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < p.length && p[l] < p[m]) m = l;
        if (r < p.length && p[r] < p[m]) m = r;
        if (m === i) break;
        [p[m], p[i]] = [p[i], p[m]];
        [v[m], v[i]] = [v[i], v[m]];
        i = m;
      }
    }
    return top;
  }
}

export function edgeLengths(graph) {
  if (graph._lengths) return graph._lengths;
  const m = graph.edges.length / 2;
  const out = new Float64Array(m);
  for (let e = 0; e < m; e += 1) {
    const a = graph.edges[e * 2] * 3, b = graph.edges[e * 2 + 1] * 3;
    out[e] = Math.hypot(
      graph.nodes[b] - graph.nodes[a],
      graph.nodes[b + 1] - graph.nodes[a + 1],
      graph.nodes[b + 2] - graph.nodes[a + 2],
    );
  }
  graph._lengths = out;
  return out;
}

/**
 * A* with a straight-line heuristic. Admissible and consistent because every
 * weight is at least the edge's length (hazard weighting only scales up), so
 * the route is optimal, not approximate.
 *
 * @param {object} opts  weights: Float64Array per edge (default lengths);
 *                       blocked: per-edge flags to avoid (default graph.blocked
 *                       when useWalls).
 */
export function astar(graph, start, end, { weights, useWalls = true, blocked } = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const w = weights || edgeLengths(graph);
  const block = blocked || (useWalls ? graph.blocked : null);
  const P = graph.nodes;
  const gx = P[end * 3], gy = P[end * 3 + 1], gz = P[end * 3 + 2];
  const h = (i) => Math.hypot(P[i * 3] - gx, P[i * 3 + 1] - gy, P[i * 3 + 2] - gz);

  const dist = new Float64Array(n).fill(Infinity);
  const prev = new Int32Array(n).fill(-1);
  const closed = new Uint8Array(n);
  dist[start] = 0;
  const heap = new Heap();
  heap.push(h(start), start);
  while (heap.size) {
    const u = heap.pop();
    if (closed[u]) continue;
    if (u === end) break;
    closed[u] = 1;
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
      const e = edgeIds[k];
      if (block?.[e]) continue;
      const v = neighbours[k];
      if (closed[v]) continue;
      const nd = dist[u] + w[e];
      if (nd < dist[v]) {
        dist[v] = nd;
        prev[v] = u;
        heap.push(nd + h(v), v);
      }
    }
  }
  if (!Number.isFinite(dist[end])) return null;
  const nodes = [end];
  for (let c = end; c !== start; ) {
    c = prev[c];
    nodes.push(c);
  }
  nodes.reverse();
  return { nodes, cost: dist[end] };
}

/** Shortest distance from `source` to every node over open links (Dijkstra). */
export function distancesFrom(graph, source, { blocked = null } = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const w = edgeLengths(graph);
  const dist = new Float64Array(n).fill(Infinity);
  dist[source] = 0;
  const heap = new Heap();
  heap.push(0, source);
  const done = new Uint8Array(n);
  while (heap.size) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const v = neighbours[k];
      const nd = dist[u] + w[edgeIds[k]];
      if (nd < dist[v]) {
        dist[v] = nd;
        heap.push(nd, v);
      }
    }
  }
  return dist;
}

/**
 * A route between two picked points, in the server's PathResponse shape so
 * the UI does not care which side computed it.
 */
export function findRoute(graph, startPoint, endPoint, { useWalls = true } = {}) {
  const t0 = performance.now();
  const start = snap(graph, startPoint);
  const end = snap(graph, endPoint);
  const base = { graph_id: graph.graphId ?? null, engine: "browser", fallback_from: null, node_ids: [], points: [], cost: 0, length: 0 };
  if (start < 0 || end < 0) return { ...base, found: false, note: "The graph is empty." };
  const { labels } = componentLabels(graph);
  if (useWalls && labels[start] !== labels[end]) {
    return {
      ...base,
      found: false,
      note:
        "The start and exit are in separate parts of the building (for example two dwellings, " +
        "or a room whose door is not in the model), so no route joins them.",
    };
  }
  const result = astar(graph, start, end, { useWalls });
  if (!result) return { ...base, found: false, note: "No route between the two points." };
  const points = result.nodes.map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]);
  let length = 0;
  for (let i = 1; i < points.length; i += 1) length += Math.hypot(...points[i].map((v, k) => v - points[i - 1][k]));
  return {
    ...base,
    found: true,
    node_ids: result.nodes,
    points,
    cost: result.cost,
    length,
    ms: performance.now() - t0,
  };
}

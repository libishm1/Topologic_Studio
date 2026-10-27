/**
 * Alternative routes in the browser: the same two methods as the server
 * (topologicpy-web-backend/app/graphs/alternatives.py).
 *
 * distinct  Routes through different space, by a penalty method (Akgun,
 *           Erkut & Batta 2000): links within `separation` of each route found
 *           cost 1 + penalty times more and the search reruns; a route is kept
 *           if at most `maxOverlap` of it runs within `separation` of a better
 *           one and it costs at most `maxStretch` times the shortest.
 * yen       The k shortest loopless paths (Yen 1971), TGraph.KShortestPaths'
 *           semantics. On a lattice its runners-up tie with the best route.
 */
import { astar, componentLabels, csr, edgeLengths, snap } from "./route.js";

export const SEPARATION = 1.5;

/** Points along a polyline, consecutive ones at most `step` apart. */
function samples(points, step) {
  const out = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1], b = points[i];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const n = Math.max(1, Math.ceil(len / step));
    for (let k = 1; k <= n; k += 1) {
      const t = k / n;
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
    }
  }
  return out;
}

/** A coarse 3D hash of points, for "is anything within r of p" queries. */
function pointHash(points, cell) {
  const map = new Map();
  const k = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  for (const p of points) {
    const key = k(p[0], p[1], p[2]);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(p);
  }
  return {
    within(p, r) {
      const [cx, cy, cz] = [Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell)];
      for (let dx = -1; dx <= 1; dx += 1) {
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dz = -1; dz <= 1; dz += 1) {
            for (const q of map.get(`${cx + dx},${cy + dy},${cz + dz}`) || []) {
              if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) <= r) return true;
            }
          }
        }
      }
      return false;
    },
  };
}

/** Share of the polyline's length lying within `separation` of `other`. */
export function sharedShare(points, otherHash, separation) {
  let total = 0, shared = 0;
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1], b = points[i];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    total += len;
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    if (otherHash.within(mid, separation)) shared += len;
  }
  return total > 0 ? shared / total : 1;
}

function pointsOf(graph, nodes) {
  return nodes.map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]);
}

function lengthOf(points) {
  let len = 0;
  for (let i = 1; i < points.length; i += 1) {
    len += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1], points[i][2] - points[i - 1][2]);
  }
  return len;
}

/** The cheapest edge between consecutive nodes, and the path's cost. */
function pathCost(graph, nodes, weights) {
  const { indptr, neighbours, edgeIds } = csr(graph);
  let cost = 0;
  const edges = [];
  for (let i = 1; i < nodes.length; i += 1) {
    let best = -1;
    for (let k = indptr[nodes[i - 1]]; k < indptr[nodes[i - 1] + 1]; k += 1) {
      if (neighbours[k] === nodes[i] && (best < 0 || weights[edgeIds[k]] < weights[best])) best = edgeIds[k];
    }
    edges.push(best);
    cost += weights[best];
  }
  return { cost, edges };
}

function distinct(graph, start, end, k, base, blocked, { maxOverlap, maxStretch, separation, penalty }) {
  const first = astar(graph, start, end, { weights: base, blocked });
  if (!first) return { routes: [], notes: ["No route between the two points."] };
  const best = pathCost(graph, first.nodes, base).cost;
  const routes = [{ nodes: first.nodes, cost: best }];
  const kept = [pointHash(samples(pointsOf(graph, first.nodes), separation / 4), separation)];

  // Edge midpoints, hashed once, to find the links near a route.
  const m = graph.edges.length / 2;
  const mids = [];
  for (let e = 0; e < m; e += 1) {
    const a = graph.edges[e * 2] * 3, b = graph.edges[e * 2 + 1] * 3;
    mids.push([(graph.nodes[a] + graph.nodes[b]) / 2, (graph.nodes[a + 1] + graph.nodes[b + 1]) / 2, (graph.nodes[a + 2] + graph.nodes[b + 2]) / 2, e]);
  }
  const cell = separation;
  const midMap = new Map();
  for (const p of mids) {
    const key = `${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)},${Math.floor(p[2] / cell)}`;
    if (!midMap.has(key)) midMap.set(key, []);
    midMap.get(key).push(p);
  }

  const penalised = Float64Array.from(base);
  let last = first.nodes;
  for (let attempt = 0; attempt < 4 * k && routes.length < k; attempt += 1) {
    // Penalise every link within `separation` of the last route found, kept
    // or not, so a rejected near-duplicate still pushes the search away.
    const hit = new Set();
    for (const p of samples(pointsOf(graph, last), separation / 2)) {
      const [cx, cy, cz] = [Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell)];
      for (let dx = -1; dx <= 1; dx += 1) {
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dz = -1; dz <= 1; dz += 1) {
            for (const q of midMap.get(`${cx + dx},${cy + dy},${cz + dz}`) || []) {
              if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) <= separation) hit.add(q[3]);
            }
          }
        }
      }
    }
    if (!hit.size) break;
    for (const e of hit) penalised[e] *= 1 + penalty;
    const next = astar(graph, start, end, { weights: penalised, blocked });
    if (!next) break;
    const cost = pathCost(graph, next.nodes, base).cost;
    // Penalties only rise, so past the stretch limit nothing more is found.
    if (cost > maxStretch * best + 1e-9) break;
    last = next.nodes;
    const pts = pointsOf(graph, next.nodes);
    if (Math.max(...kept.map((h) => sharedShare(pts, h, separation))) <= maxOverlap) {
      routes.push({ nodes: next.nodes, cost });
      kept.push(pointHash(samples(pts, separation / 4), separation));
    }
  }
  // Found in the order the penalties uncovered them, not by cost.
  routes.splice(1, routes.length - 1, ...routes.slice(1).sort((a, b) => a.cost - b.cost));
  return { routes, notes: [] };
}

function yen(graph, start, end, k, base, blocked, deadline) {
  const { indptr, edgeIds } = csr(graph);
  const mask = blocked ? Uint8Array.from(blocked) : new Uint8Array(graph.edges.length / 2);
  const first = astar(graph, start, end, { weights: base, blocked: mask });
  if (!first) return { routes: [], notes: ["No route between the two points."] };
  const accepted = [{ nodes: first.nodes, cost: first.cost }];
  const seen = new Set([first.nodes.join(",")]);
  const candidates = [];
  const notes = [];
  while (accepted.length < k) {
    const prev = accepted[accepted.length - 1].nodes;
    const { edges: prevEdges } = pathCost(graph, prev, base);
    let rootCost = 0;
    for (let i = 0; i < prev.length - 1; i += 1) {
      if (performance.now() > deadline) {
        notes.push(`Stopped after the time budget with ${accepted.length} routes.`);
        return { routes: accepted, notes };
      }
      const root = prev.slice(0, i + 1);
      const spurMask = Uint8Array.from(mask);
      for (const { nodes } of accepted) {
        if (nodes.length > i + 1 && root.every((n, x) => nodes[x] === n)) {
          for (let q = indptr[nodes[i]]; q < indptr[nodes[i] + 1]; q += 1) {
            const e = edgeIds[q];
            const other = graph.edges[e * 2] === nodes[i] ? graph.edges[e * 2 + 1] : graph.edges[e * 2];
            if (other === nodes[i + 1]) spurMask[e] = 1;
          }
        }
      }
      for (const n of root.slice(0, -1)) {
        for (let q = indptr[n]; q < indptr[n + 1]; q += 1) spurMask[edgeIds[q]] = 1;
      }
      const spur = astar(graph, prev[i], end, { weights: base, blocked: spurMask });
      if (spur) {
        const total = [...root.slice(0, -1), ...spur.nodes];
        const key = total.join(",");
        if (!seen.has(key)) {
          seen.add(key);
          candidates.push({ nodes: total, cost: rootCost + spur.cost });
        }
      }
      rootCost += base[prevEdges[i]];
    }
    if (!candidates.length) break;
    candidates.sort((a, b) => a.cost - b.cost);
    accepted.push(candidates.shift());
  }
  return { routes: accepted, notes };
}

/**
 * Up to k routes between two picked points, in the server's
 * AlternativesResponse shape.
 */
export function alternativeRoutes(graph, startPoint, endPoint, {
  k = 3,
  method = "distinct",
  useWalls = true,
  maxOverlap = 0.5,
  maxStretch = 1.5,
  separation = SEPARATION,
  penalty = 1.0,
  timeBudgetMs = 5000,
} = {}) {
  const t0 = performance.now();
  const start = snap(graph, startPoint);
  const end = snap(graph, endPoint);
  const base = edgeLengths(graph);
  const blocked = useWalls ? graph.blocked : null;
  const { labels } = componentLabels(graph);
  const response = { graph_id: graph.graphId ?? null, method, engine: "browser", requested: k, routes: [], ms: 0, note: null };
  if (useWalls && labels[start] !== labels[end]) {
    response.note = "The start and exit are in separate parts of the building, so no route joins them.";
    return response;
  }
  const { routes, notes } =
    method === "yen"
      ? yen(graph, start, end, k, base, blocked, t0 + timeBudgetMs)
      : distinct(graph, start, end, k, base, blocked, { maxOverlap, maxStretch, separation, penalty });

  const hashes = [];
  routes.forEach((route, i) => {
    const points = pointsOf(graph, route.nodes);
    const overlap = hashes.length ? Math.max(...hashes.map((h) => sharedShare(points, h, separation))) : 0;
    hashes.push(pointHash(samples(points, separation / 4), separation));
    response.routes.push({
      graph_id: response.graph_id,
      found: true,
      points,
      node_ids: route.nodes,
      cost: route.cost,
      length: lengthOf(points),
      engine: "browser",
      fallback_from: null,
      note: null,
      rank: i + 1,
      stretch: route.cost / routes[0].cost,
      overlap,
    });
  });
  if (method === "distinct" && routes.length && routes.length < k) {
    notes.push(
      `Only ${routes.length} of ${k} routes use substantially different space (at most ` +
        `${Math.round(maxOverlap * 100)}% within ${separation} m of another) within ${maxStretch}x the shortest cost.`,
    );
  }
  response.note = notes.join(" ") || null;
  response.ms = performance.now() - t0;
  return response;
}

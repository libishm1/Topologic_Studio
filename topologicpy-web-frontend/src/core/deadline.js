/**
 * Escape deadlines (prototype): for every node of the walkable graph, the
 * latest time an occupant standing there can set off and still reach an exit
 * without entering any node at or after the moment it becomes untenable.
 *
 * Model. Walking a link takes len / speed seconds (constant, so FIFO holds).
 * Node v is tenable before time u(v) and untenable from then on (u = Infinity
 * if it never is); here u comes from a fire front, but any monotone
 * untenability time works, including a dose threshold reached. A walk is safe
 * if it arrives at every node strictly before that node's u, the start
 * included.
 *
 * The deadline L(v) is the supremum of safe departure times (exclusive):
 *   L(exit) = u(exit),  L(v) = max over open links (v, w) of min(u(v), L(w) - tau(v, w)).
 * It is computed by label setting from the exits outward, always settling the
 * unsettled node with the largest label: the dual of Dijkstra, since a label
 * can only fall along a link. L = -Infinity means no exit is reachable at all.
 *
 * `departureSafe` is an independent forward check (earliest-arrival search
 * from one node at one departure time), used to verify the labels: leaving
 * just before L(v) must reach an exit, leaving at or after it must not.
 */
import { Heap, csr, distancesFrom, edgeLengths } from "./route.js";

/** Untenability times from a fire front spreading along the graph at `speed` m/s. */
export function frontArrival(graph, origin, { speed = 0.1, blocked = null } = {}) {
  const dist = distancesFrom(graph, origin, { blocked });
  const out = new Float64Array(dist.length);
  for (let i = 0; i < dist.length; i += 1) out[i] = Number.isFinite(dist[i]) ? dist[i] / speed : Infinity;
  return out;
}

/**
 * @param {object} opts  exits: node ids; untenable: Float64Array of times
 *                       (default all Infinity); speed: walking speed, m/s;
 *                       blocked: per-edge flags to avoid (walls).
 * @returns {Float64Array} L per node, seconds (Infinity: never cut off;
 *                         -Infinity: no exit reachable).
 */
export function escapeDeadlines(graph, { exits, untenable = null, speed = 1.2, blocked = null }) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const len = edgeLengths(graph);
  const u = untenable || new Float64Array(n).fill(Infinity);
  const L = new Float64Array(n).fill(-Infinity);
  const done = new Uint8Array(n);
  // Heap is a min-heap; store negated labels to settle the largest first.
  const heap = new Heap();
  for (const x of exits) {
    if (x < 0 || x >= n) continue;
    if (u[x] > L[x]) {
      L[x] = u[x];
      heap.push(-L[x], x);
    }
  }
  while (heap.size) {
    const w = heap.pop();
    if (done[w]) continue;
    done[w] = 1;
    for (let k = indptr[w]; k < indptr[w + 1]; k += 1) {
      const e = edgeIds[k];
      if (blocked?.[e]) continue;
      const v = neighbours[k];
      if (done[v]) continue;
      const candidate = Math.min(u[v], L[w] - len[e] / speed);
      if (candidate > L[v]) {
        L[v] = candidate;
        heap.push(-candidate, v);
      }
    }
  }
  return L;
}

/** Whether leaving `start` at `depart` reaches an exit safely (forward check). */
export function departureSafe(graph, start, depart, { exits, untenable, speed = 1.2, blocked = null }) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const len = edgeLengths(graph);
  if (!(depart < untenable[start])) return false;
  const isExit = new Uint8Array(n);
  for (const x of exits) isExit[x] = 1;
  const arrive = new Float64Array(n).fill(Infinity);
  arrive[start] = depart;
  const heap = new Heap();
  heap.push(depart, start);
  const done = new Uint8Array(n);
  while (heap.size) {
    const v = heap.pop();
    if (done[v]) continue;
    done[v] = 1;
    if (isExit[v]) return true;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const e = edgeIds[k];
      if (blocked?.[e]) continue;
      const w = neighbours[k];
      const t = arrive[v] + len[e] / speed;
      if (t < untenable[w] && t < arrive[w]) {
        arrive[w] = t;
        heap.push(t, w);
      }
    }
  }
  return false;
}

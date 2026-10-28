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
 * just before L(v) must reach an exit, leaving after it must not
 * (Deadline.mem_safeSet_iff, departureSafe_correct). Leaving at exactly L can
 * pass in doubles, where (L - a) + a may round below the closing time.
 *
 * With a dose limit (the method the scoop check recommends, C1:
 * wiki/research/gap-analysis-2026-10-27/scoop-check.json) a departure is good
 * if some safe route reaches an exit with a fractional effective dose at or
 * below the limit. Under a forecast whose heat never falls, leaving earlier is
 * never worse (the same route, walked earlier, arrives earlier everywhere and
 * accrues no more dose), so the good departures form an interval from 0.
 *   doseDeadline     the start's deadline, bracketed by bisection with the
 *                    exact least-dose search (forecast.doseFront) as the test.
 *   doseDeadlineMap  every node's deadline from one backward pass over the
 *                    time-expanded graph (MinDose's recursion without waiting):
 *                    V_j(v) = min over links of d_e(j) + V_{j + tau_e}(w), and
 *                    the deadline is the last grid time j with V_j(v) <= limit.
 *                    Link times are rounded up to the grid, so it is never
 *                    later than the exact deadline: the real walker takes the
 *                    same links without waiting, entering each no later and
 *                    walking it no longer, so it accrues no more dose
 *                    (Deadline.grid_conservative, studio_map_le_exact). The
 *                    rounding is checked in floating point, and the closing
 *                    and dose tests keep a margin of a part in 10^9, so that
 *                    rounding in the sums cannot make the map later either.
 */
import { Heap, csr, distancesFrom, edgeLengths } from "./route.js";
import { doseFront, linkDose } from "./forecast.js";

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

/**
 * The latest departure from `start` with a safe route to an exit whose dose is
 * at most `limit`, bracketed to within `tolerance` seconds.
 *
 * @returns {{good: number, bad: number, searches: number, truncated: boolean, horizon: boolean} | null}
 *          leaving at `good` works and at `bad` does not (Deadline.
 *          doseDeadline_bisection); null when even leaving at once does not.
 *          truncated: a search hit its label budget, so a "does not" may be
 *          wrong and the bracket is not certain. horizon: what stops leaving
 *          at `bad` is the end of the forecast, not the fire (leaving then
 *          works if points stay open until the fire reaches them), so the
 *          real deadline is at least `good`, perhaps later.
 */
export function doseDeadline(graph, start, { exits, forecast, limit = 1, speed = 1.2, blocked = null, tolerance = 0.05 }) {
  let searches = 0;
  let truncated = false;
  const works = (t) => {
    searches += 1;
    const result = doseFront(graph, start, { exits, forecast, depart: t, speed, blocked });
    truncated ||= result.truncated;
    const { front } = result;
    return front.length > 0 && front[front.length - 1].dose <= limit;
  };
  if (!works(0)) return null;
  // Leaving after the deadline without a dose limit is never safe; leaving at
  // exactly it can pass in doubles, so the end is checked, and moved past any
  // such rounding. A start that never closes is bounded by the forecast.
  let good = 0;
  let bad = escapeDeadlines(graph, { exits, untenable: forecast.closes, speed, blocked })[start];
  if (!Number.isFinite(bad)) bad = forecast.horizon;
  for (let nudge = 0; nudge < 4 && works(bad); nudge += 1) {
    good = bad;
    bad += 1e-9 * Math.max(1, Math.abs(bad));
  }
  while (bad - good > tolerance) {
    const mid = (good + bad) / 2;
    if (works(mid)) good = mid;
    else bad = mid;
  }
  // Every point closes at the forecast's end. If leaving at `bad` works once
  // they close only when the fire reaches them, the end of the forecast is
  // the limit, not the fire (review M11).
  const open = { ...forecast, closes: forecast.untenable };
  const past = doseFront(graph, start, { exits, forecast: open, depart: bad, speed, blocked }).front;
  const horizon = past.length > 0 && past[past.length - 1].dose <= limit;
  return { good, bad, searches, truncated, horizon };
}

/**
 * Every node's dose-limited deadline on a time grid (conservative; see the
 * header). The grid step defaults to a quarter of the median link time, or a
 * half or a whole one where that would be too much work, and is widened
 * further if needed so that the work, layers x (open link ends + nodes), is
 * at most `maxWork` whatever the horizon. A wider step only rounds walking
 * times up more, which is still conservative.
 *
 * @returns {{deadline: Float64Array, step: number, layers: number, work: number}}
 *          deadline per node in seconds, -Infinity when no grid departure
 *          works; work: the link updates the map took.
 */
export function doseDeadlineMap(graph, { exits, forecast, limit = 1, speed = 1.2, blocked = null, step = null, maxWork = 1.5e8 }) {
  const { indptr, neighbours, edgeIds, n, m } = csr(graph);
  const len = edgeLengths(graph);
  const closes = forecast.closes;
  const times = [];
  let slots = 0;
  for (let k = 0; k < neighbours.length; k += 1) {
    if (blocked?.[edgeIds[k]]) continue;
    slots += 1;
    if (len[edgeIds[k]] > 0) times.push(len[edgeIds[k]] / speed);
  }
  times.sort((a, b) => a - b);
  // A whole fraction of the typical (median) link time, so a typical (axis)
  // link takes a whole number of steps. A quarter or a half: either rounds a
  // diagonal (sqrt 2 times as long) up by 6%, where a whole link time would
  // round it up by 41%. Shorter links (door bridges) take one step, which is
  // still rounding up. A whole link time only on very large graphs.
  const typical = times.length ? times[times.length >> 1] : forecast.horizon;
  const perLayer = slots + n;
  const work = (d) => (Math.floor(forecast.horizon / d) + 1) * perLayer;
  let delta = step;
  if (!delta) {
    // The half step used to be allowed 8 times the budget and the whole one
    // any amount: 2,000 fire steps on the dental clinic took 38 times it
    // (81 s). The step is now widened until the layers fit (review K2).
    const q = work(typical / 4) <= maxWork ? 4 : work(typical / 2) <= maxWork ? 2 : 1;
    const layers = Math.max(2, Math.floor(maxWork / perLayer));
    delta = Math.max(typical / q, forecast.horizon / (layers - 1));
  }
  const J = Math.floor(forecast.horizon / delta);
  const tau = new Int32Array(m);
  let longest = 1;
  for (let e = 0; e < m; e += 1) {
    // Up, never down (Deadline.ceil_steps_cover). The epsilon keeps a link
    // whose time is an exact multiple of the step from gaining one through
    // rounding in the division; the check puts back any link it rounded
    // down (Deadline.epsilon_rounds_down).
    tau[e] = Math.max(1, Math.ceil(len[e] / speed / delta - 1e-9));
    if (tau[e] * delta < len[e] / speed) tau[e] += 1;
    if (!blocked?.[e] && tau[e] > longest) longest = tau[e];
  }
  const isExit = new Uint8Array(n);
  for (const x of exits) if (x >= 0 && x < n) isExit[x] = 1;
  const R = longest + 1;
  const ring = new Float64Array(R * n).fill(Infinity);
  const deadline = new Float64Array(n).fill(-Infinity);
  const { rate, frameSeconds, K } = forecast;
  // Each node's open links as two flat arrays, the far end and the link's
  // steps, in CSR order: the loop below then reads two arrays per link
  // instead of four and tests no wall flag. The same links in the same order
  // give the same minimum (tools/output-hash.mjs: byte-identical maps).
  const first = new Int32Array(n + 1);
  for (let v = 0; v < n; v += 1) {
    let count = 0;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) if (!blocked?.[edgeIds[k]]) count += 1;
    first[v + 1] = first[v] + count;
  }
  const to = new Int32Array(first[n]), steps = new Int32Array(first[n]);
  for (let v = 0, s = 0; v < n; v += 1) {
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const e = edgeIds[k];
      if (blocked?.[e]) continue;
      to[s] = neighbours[k];
      steps[s] = tau[e];
      s += 1;
    }
  }
  // What depends only on a link's number of steps d, computed once per layer
  // with the arithmetic the loop used to repeat per link: the ring row it
  // lands in, the arrival time, that time with its closing margin, and the
  // time on the link.
  const rowAt = new Int32Array(R), arriveAt = new Float64Array(R), clearAt = new Float64Array(R), spanAt = new Float64Array(R);
  // Values spread back from the exits along open links, so only exits and the
  // neighbours of nodes that have had a value need the link loop; the others
  // (nodes the exits do not reach yet, or at all) keep Infinity.
  const live = Uint8Array.from(isExit), reached = new Uint8Array(n);
  for (let j = J; j >= 0; j -= 1) {
    const t = j * delta;
    const row = (j % R) * n;
    // Most links start and end inside one forecast frame: then the dose is
    // the hotter end's rate times the time (linkDose's single-frame case).
    const frame = Math.min(K - 1, Math.floor(t / frameSeconds));
    const frameEnd = (frame + 1) * frameSeconds;
    const base = frame * n;
    // Links of more steps than this end past the grid (j + d > J).
    const reach = Math.min(longest, J - j);
    for (let d = 1; d <= reach; d += 1) {
      const jj = j + d;
      const t2 = jj * delta;
      rowAt[d] = (jj % R) * n;
      arriveAt[d] = t2;
      clearAt[d] = t2 + 1e-9 * Math.max(1, t2);
      spanAt[d] = t2 - t;
    }
    // A part in 10^9 in hand on every closing time and on the limit, so
    // that rounding in the walker's sums cannot make the map later.
    const leave = t + 1e-9 * Math.max(1, t);
    for (let v = 0; v < n; v += 1) {
      // A node none of whose neighbours has had a value yet has none now
      // either: its ring slots are still Infinity, and so is the answer.
      if (!live[v]) continue;
      let value = Infinity;
      if (leave < closes[v]) {
        if (isExit[v]) value = 0;
        else {
          const rv = rate[base + v];
          for (let s = first[v]; s < first[v + 1]; s += 1) {
            const d = steps[s];
            if (d > reach) continue;
            const w = to[s];
            const ahead = ring[rowAt[d] + w];
            if (ahead === Infinity) continue;
            if (!(clearAt[d] < closes[w])) continue;
            const t2 = arriveAt[d];
            const dose = t2 <= frameEnd ? spanAt[d] * Math.max(rv, rate[base + w]) : linkDose(forecast, v, w, t, t2);
            const total = ahead + dose;
            if (total < value) value = total;
          }
        }
      }
      ring[row + v] = value;
      if (value < Infinity && !reached[v]) {
        // From the next (earlier) layer on, v's neighbours can have values.
        reached[v] = 1;
        for (let s = first[v]; s < first[v + 1]; s += 1) live[to[s]] = 1;
      }
      // Finite first: with no limit (Infinity), "no route" must not count as within it.
      if (deadline[v] === -Infinity && value < Infinity && value * (1 + 1e-9) <= limit) deadline[v] = t;
    }
  }
  return { deadline, step: delta, layers: J + 1, work: (J + 1) * perLayer };
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

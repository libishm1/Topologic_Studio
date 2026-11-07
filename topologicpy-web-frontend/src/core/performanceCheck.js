/**
 * A performance-based ("simulated fire") egress check: at every place, the
 * time the fire leaves against the time an occupant needs to start moving
 * (ASET against RSET; wiki/research/fire-egress-methods-2026-10-27.md 4.2,
 * the user's request of 2026-11-07).
 *
 * The forecast's escape-deadline map (deadline.js doseDeadlineMap) gives
 * L(v): the latest time after ignition at which someone at v can set off and
 * still reach an exit with a fractional effective dose at or below the limit.
 * The map is never later than the exact deadline (Deadline.lean,
 * studio_map_le_exact). Walking is inside L, so the required escape time
 * reduces to the time to start:
 *   t_start = t_detection + t_alarm + t_pre.
 * Pre-movement is log-normal, fitted to its 1st and 99th percentiles (the
 * form PD 7974-6 tabulates). A place passes when L(v) >= t_detection +
 * t_alarm + t_pre99, with a margin and a factor on RSET when the criterion
 * asks for them (rsetPresets.js RSET_CRITERIA); the share of its occupants
 * who start in time is F_pre(L(v) - t_detection - t_alarm). A pass at the
 * 99th percentile therefore means at least 99 % start in time there
 * (Performance.lean).
 *
 * The map counts a departure safe only if the walk ends within the
 * forecast, so L(v) is at most the forecast's length less the walk. Where
 * even a walk with no fire, set off at the required start, would end after
 * the forecast (start + walk > horizon), the check cannot tell, and says
 * how long a forecast it needs; that is when the deadline sits at the
 * forecast's end less the walk. A deadline before the required start and
 * well before that end is the fire's doing, and fails (so does a detour the
 * fire forces whose end the forecast does not reach: the pessimistic side). A place with no
 * safe departure at all fails; one that cannot reach an exit on the graph
 * at all is a question about the model, as in the prescriptive check, and
 * is reported apart.
 *
 * Beside the design values:
 * - noResponse: the share who never respond (ABCB FSVM 1.4). They stay out
 *   of the flows and are never counted as starting in time, so the share
 *   that does is at most 1 - q (PreMovement.lean, responding_le); the
 *   percentiles are the responders'.
 * - measured: a Weibull fitted to measured pre-movement (rsetPresets.js
 *   MEASURED_PRE), judged the same way and reported beside the design
 *   log-normal, not in the verdict.
 * - queues: occupants on every place, at the area per occupant, walk to the
 *   nearest exit with no fire and queue at each door on the way
 *   (doorQueues). Time spent queueing counts in RSET like walking and as a
 *   later start, so a place needs L >= required(tau + W) + W, with W the
 *   queueing of its 99th-percentile starter (DoorQueue.lean,
 *   requiredQ_iff); the dose taken while queueing is not reckoned apart, as
 *   in a hand calculation (C/VM2 Eq. 3.1).
 *
 * It is evidence for an alternative method (SBC 801-CR-2018 104.9), not a
 * substitute for the prescriptive check: one fire scenario a call (the
 * report in App.jsx gathers several), one walking speed, and the room of
 * fire origin judged like any other place.
 */

import { Heap, csr, edgeLengths } from "./route.js";
import { storeysOf } from "./codeCheck.js";

const KIND_DOOR = 2;
/** The 99th percentile of the standard normal distribution. */
export const Z99 = 2.3263478740408408;
/** Its 99.99th percentile: a log-normal's support, for the queues. */
const Z9999 = 3.719016485455709;

/** erf by Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7). */
export function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

/** The log-normal whose 1st and 99th percentiles are `p1` and `p99` (seconds, 0 < p1 < p99). */
export function lognormalFromPercentiles(p1, p99) {
  const mu = (Math.log(p1) + Math.log(p99)) / 2;
  const sigma = (Math.log(p99) - Math.log(p1)) / (2 * Z99);
  return { mu, sigma };
}

/** Its cumulative distribution at `t` seconds. */
export function lognormalCdf(t, { mu, sigma }) {
  if (!(t > 0)) return 0;
  if (t === Infinity) return 1;
  return 0.5 * (1 + erf((Math.log(t) - mu) / (sigma * Math.SQRT2)));
}

/** The Weibull distribution function at `t` seconds, with its scale (s) and shape. */
export function weibullCdf(t, { scale, shape }) {
  if (!(t > 0)) return 0;
  if (t === Infinity) return 1;
  return 1 - Math.exp(-((t / scale) ** shape));
}

/** Its quantile: the time by which a share p (0 < p < 1) has started (PreMovement.lean, weibull_cdf_quantile). */
export function weibullQuantile(p, { scale, shape }) {
  return scale * (-Math.log(1 - p)) ** (1 / shape);
}

/**
 * Every node's walking time to the nearest exit with no fire, over open
 * links (Infinity where no exit is reached): the least time any route out
 * can take, fire or not; and `next`, the neighbour on that walk (-1 at an
 * exit and where no exit is reached).
 */
export function walkTree(graph, exits, { speed = 1.2, blocked = null } = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const w = edgeLengths(graph);
  const time = new Float64Array(n).fill(Infinity);
  const next = new Int32Array(n).fill(-1);
  const heap = new Heap();
  for (const e of exits) {
    if (e < 0 || time[e] === 0) continue;
    time[e] = 0;
    heap.push(0, e);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const v = neighbours[k];
      const t = time[u] + w[edgeIds[k]] / speed;
      if (t < time[v]) {
        time[v] = t;
        next[v] = u;
        heap.push(t, v);
      }
    }
  }
  return { time, next };
}

/** The walking times alone (walkTree). */
export function walkToExit(graph, exits, opts = {}) {
  return walkTree(graph, exits, opts).time;
}

/**
 * Persons a second per metre of effective width through a door at the
 * densest flow: (1 - a D) k D at D = 1.9 persons/m², k = 1.4 m/s and
 * a = 0.266, about 1.32 (NZ C/VM2 3.5.3, after the SFPE hydraulic model).
 */
export const DOOR_FLOW = (1 - 0.266 * 1.9) * 1.4 * 1.9;
/** The boundary layer at each side of a door, m (C/VM2 3.5.3). */
export const DOOR_BOUNDARY = 0.15;
/** The most time bins a queue takes; past it the bins widen. */
const MAX_BINS = 6000;

/** Persons a second through a door `width` m wide, less its boundary layers (at least 0.1 m left). */
export const doorCapacity = (width) => DOOR_FLOW * Math.max(0.1, width - 2 * DOOR_BOUNDARY);

/**
 * Queues at the doors: a hydraulic model with Lindley's recursion at each
 * door, on time bins of `dt` seconds. Everyone walks the tree to the
 * nearest exit with no fire; the occupants of a place set off over the
 * pre-movement distribution after the alarm, reach the first door on their
 * way after their walk, and pass each door at its capacity, first come
 * first served, walking on to the next. Nodes with no capacity hold no one
 * up.
 *
 * Times are rounded up to the bins, so every wait is on the long side: what
 * reaches a door in bin b has left it by the end of bin
 * b + ceil(Q(b + 1) / (c dt)) (DoorQueue.lean, exit_clears), and a later
 * arrival never leaves earlier (exit_mono).
 *
 * @param {{time: Float64Array, next: Int32Array}} tree  walkTree's
 * @param {object} opts
 *   occupants  persons at each node who respond (Float64Array)
 *   capacity   persons a second through each node (Float64Array; 0: no door)
 *   cue        detection plus alarm (s)
 *   cdf        the pre-movement distribution function, seconds after the alarm
 *   support    [lo, hi]: seconds after the alarm within which nearly everyone starts
 * @returns {{dt, exitTime: (v: number, s: number) => number, doors: Array}}
 *   exitTime: when someone who sets off from v at s seconds after ignition is out
 */
export function doorQueues({ time, next }, { occupants, capacity, cue, cdf, support }) {
  const n = time.length;
  // The first door strictly on from every node, -1 for none before the exit.
  const down = new Int32Array(n).fill(-2);
  const stack = [];
  for (let v = 0; v < n; v += 1) {
    let u = v;
    while (down[u] === -2) {
      const w = next[u];
      if (w < 0) down[u] = -1;
      else if (capacity[w] > 0) down[u] = w;
      else {
        stack.push(u);
        u = w;
      }
    }
    while (stack.length) down[stack.pop()] = down[u];
  }
  // The doors anyone walks through, and a bin width that covers the latest
  // start, the longest walk and everyone through the slowest door.
  const active = new Uint8Array(n);
  let total = 0, far = 0, slow = Infinity;
  for (let v = 0; v < n; v += 1) {
    if (!(occupants[v] > 0) || time[v] === Infinity) continue;
    total += occupants[v];
    far = Math.max(far, time[v]);
    for (let d = down[v]; d >= 0 && !active[d]; d = down[d]) {
      active[d] = 1;
      slow = Math.min(slow, capacity[d]);
    }
  }
  const end = cue + support[1] + far + (slow < Infinity ? total / slow : 0) + 60;
  const dt = Math.max(1, Math.ceil(end / MAX_BINS));
  const bins = (t) => Math.ceil(t / dt - 1e-9);
  // The start over bins b0 to b1, the mass before and after lumped at the ends.
  const b0 = Math.max(0, Math.floor((cue + support[0]) / dt));
  const b1 = Math.max(b0, Math.ceil((cue + support[1]) / dt));
  const pmf = new Float64Array(b1 - b0 + 1);
  let before = 0;
  for (let b = b0; b <= b1; b += 1) {
    const F = b === b1 ? 1 : cdf((b + 1) * dt - cue);
    pmf[b - b0] = Math.max(0, F - before);
    before = Math.max(before, F);
  }
  // Each door's arrivals straight from the places: mass by walk, in bins.
  const direct = new Map();
  for (let v = 0; v < n; v += 1) {
    const d = down[v];
    if (d < 0 || !(occupants[v] > 0) || time[v] === Infinity) continue;
    const delay = bins(time[v] - time[d]);
    let byDelay = direct.get(d);
    if (!byDelay) direct.set(d, (byDelay = new Map()));
    byDelay.set(delay, (byDelay.get(delay) || 0) + occupants[v]);
  }
  // Doors upstream first: each waits for the doors that feed it.
  const doors = [];
  for (let d = 0; d < n; d += 1) if (active[d]) doors.push(d);
  const waiting = new Int32Array(n);
  const upstream = new Map();
  for (const u of doors) {
    const d = down[u];
    if (d < 0) continue;
    waiting[d] += 1;
    if (!upstream.has(d)) upstream.set(d, []);
    upstream.get(d).push(u);
  }
  const lo = new Int32Array(n);
  const queue = new Array(n);
  const outs = new Array(n);
  const ready = doors.filter((d) => waiting[d] === 0);
  const summary = [];
  while (ready.length) {
    const d = ready.pop();
    const c = capacity[d] * dt;
    const byDelay = direct.get(d);
    const ups = upstream.get(d) || [];
    let first = Infinity, last = -Infinity;
    if (byDelay) {
      for (const delay of byDelay.keys()) {
        first = Math.min(first, b0 + delay);
        last = Math.max(last, b1 + delay);
      }
    }
    for (const u of ups) {
      const shift = bins(time[u] - time[d]);
      first = Math.min(first, lo[u] + shift);
      last = Math.max(last, lo[u] + outs[u].length - 1 + shift);
    }
    const len = last - first + 1;
    const a = new Float64Array(len);
    if (byDelay) {
      for (const [delay, m] of byDelay) {
        const at = b0 + delay - first;
        for (let i = 0; i < pmf.length; i += 1) a[at + i] += m * pmf[i];
      }
    }
    for (const u of ups) {
      const at = lo[u] + bins(time[u] - time[d]) - first;
      const o = outs[u];
      for (let i = 0; i < o.length; i += 1) a[at + i] += o[i];
      outs[u] = null;
    }
    // Lindley's recursion, run on until the queue clears.
    let q = 0;
    for (let i = 0; i < len; i += 1) q = Math.max(0, q + a[i] - c);
    const extra = Math.ceil(q / c - 1e-9);
    const Q = new Float64Array(len + extra + 1);
    const out = new Float64Array(len + extra);
    let persons = 0, peak = 0, wait = 0;
    for (let i = 0; i < len + extra; i += 1) {
      const ai = i < len ? a[i] : 0;
      Q[i + 1] = Math.max(0, Q[i] + ai - c);
      out[i] = Q[i] + ai - Q[i + 1];
      persons += ai;
      peak = Math.max(peak, Q[i + 1]);
      if (ai > 1e-9) wait = Math.max(wait, Math.ceil(Q[i + 1] / c - 1e-9) * dt);
    }
    lo[d] = first;
    queue[d] = Q;
    outs[d] = out;
    summary.push({ node: d, persons, peak, wait, capacity: capacity[d], clears: (first + len + extra) * dt });
    const nd = down[d];
    if (nd >= 0 && (waiting[nd] -= 1) === 0) ready.push(nd);
  }
  // The bin by whose end what reached door d in bin b has left it.
  const exitBin = (d, b) => {
    const Q = queue[d];
    if (!Q) return b + 1;
    const i = b + 1 - lo[d];
    const q = i > 0 && i < Q.length ? Q[i] : 0;
    return b + 1 + Math.ceil(q / (capacity[d] * dt) - 1e-9);
  };
  const exitTime = (v, s) => {
    let d = down[v];
    if (d < 0) return s + time[v];
    let b = Math.floor(s / dt) + bins(time[v] - time[d]);
    for (;;) {
      const e = exitBin(d, b);
      const d2 = down[d];
      if (d2 < 0) return e * dt + time[d];
      b = e - 1 + bins(time[d] - time[d2]);
      d = d2;
    }
  };
  return { dt, exitTime, doors: summary };
}

/**
 * The latest start from a place at which the walk, queues included, is out
 * by L + tau: the exit time never falls as the start grows (DoorQueue.lean,
 * safe_starts_downward), so bisection finds it to a tenth of a bin;
 * -Infinity when even a start at `from` is too late.
 */
function latestStart(queues, v, L, walk, from) {
  const late = (s) => queues.exitTime(v, s) - walk > L;
  if (!late(L)) return L;
  if (late(from)) return -Infinity;
  let lo = from, hi = L;
  for (let i = 0; i < 40 && hi - lo > 0.1 * queues.dt; i += 1) {
    const mid = (lo + hi) / 2;
    if (late(mid)) hi = mid;
    else lo = mid;
  }
  return lo;
}

/**
 * @param {object} graph      nodes, edges, kinds, blocked, upAxis, agentHeight
 * @param {Float64Array} deadlines  L(v) in seconds from ignition (-Infinity: no safe departure)
 * @param {object} opts
 *   levels, storeyNames  as the code check
 *   exits                the exits the forecast routed to
 *   horizon, step        the forecast's length and the map's grid step (s)
 *   speed                the walking speed the forecast used (m/s)
 *   detection, alarm     seconds from ignition to detection, and from detection to the alarm
 *   pre1, pre99          pre-movement time, 1st and 99th percentiles (s)
 *   factor               the least ASET / RSET (default 1)
 *   margin               the least ASET - RSET, seconds (default 0)
 *   noResponse           the share who never respond, 0 to 1 (default 0)
 *   measured             {scale, shape, label}: a measured Weibull judged beside (optional)
 *   queues               {occupantArea (m²), cellArea (m²), widths (m a node, 0 where
 *                        unknown), defaultWidth (m)}: queues at doors (optional)
 *   useWalls             respect links blocked by walls (default true)
 * @returns {{status, start, detection, cue, base, factor, margin, storeys, distribution, needHorizon, noResponse, measured, queues, curves}}
 *
 * With a factor, RSET takes the walk with no fire, tau(v), as its travel
 * term (as a hand calculation does) and ASET the deadline plus the same
 * walk, so ASET - RSET is the margin at the 99th percentile, L - cue - q99,
 * and ASET / RSET >= f is L >= f (cue + q99) + (f - 1) tau (Performance.lean,
 * ratio_iff): the margin must be at least f - 1 times RSET. A queue adds its
 * wait W to RSET and to the start (DoorQueue.lean, requiredQ_iff).
 */
export function performanceCheck(graph, deadlines, opts = {}) {
  const {
    levels = [],
    storeyNames = [],
    exits = [],
    horizon = Infinity,
    step = 0,
    speed = 1.2,
    detection = 0,
    alarm = 0,
    pre1,
    pre99,
    factor = 1,
    margin = 0,
    noResponse = 0,
    measured = null,
    queues = null,
    useWalls = true,
  } = opts;
  const n = graph.nodes.length / 3;
  const kinds = graph.kinds || new Uint8Array(n);
  const storeyOf = storeysOf(graph, levels);
  const tree = walkTree(graph, exits, { speed, blocked: useWalls ? graph.blocked : null });
  const walk = tree.time;
  const distribution = lognormalFromPercentiles(pre1, pre99);
  const cue = detection + alarm;
  const respond = 1 - Math.min(1, Math.max(0, noResponse));
  const design = {
    cdf: (t) => lognormalCdf(t, distribution),
    p99: pre99,
    support: [Math.exp(distribution.mu - Z9999 * distribution.sigma), Math.exp(distribution.mu + Z9999 * distribution.sigma)],
  };
  const data = measured
    ? { cdf: (t) => weibullCdf(t, measured), p99: weibullQuantile(0.99, measured), support: [weibullQuantile(1e-4, measured), weibullQuantile(0.9999, measured)] }
    : null;

  // The occupants who respond, and the doors' capacities, for the queues.
  let occupants = null, capacity = null;
  if (queues) {
    const { occupantArea, cellArea, widths = null, defaultWidth = 0.813 } = queues;
    occupants = new Float64Array(n);
    capacity = new Float64Array(n);
    for (let v = 0; v < n; v += 1) {
      if (kinds[v] === KIND_DOOR) capacity[v] = doorCapacity(widths?.[v] > 0 ? widths[v] : defaultWidth);
      else if (storeyOf[v] >= 0 && walk[v] < Infinity) occupants[v] = (respond * cellArea) / occupantArea;
    }
  }
  const queuesFor = (dist) => (queues ? doorQueues(tree, { occupants, capacity, cue, cdf: dist.cdf, support: dist.support }) : null);

  const floors = levels.length ? levels.length : 1;
  // The map's grid can end up to a step before the horizon.
  const covered = horizon - step;
  // Every place under one distribution: its 99th-percentile starter sets off
  // at `base`, and with queues waits W on the way.
  const judge = (dist, q) => {
    const base = cue + dist.p99;
    const required = (tau) => Math.max(base + margin, factor * base + (factor - 1) * tau);
    const per = Array.from({ length: floors }, () => ({ places: 0, pass: 0, fail: 0, open: 0, unreached: 0, worst: Infinity, ratio: Infinity, share: 0, need: 0, wait: 0, occupants: 0 }));
    for (let v = 0; v < n; v += 1) {
      const s = storeyOf[v];
      if (s < 0 || kinds[v] === KIND_DOOR) continue;
      const p = per[s];
      p.places += 1;
      if (walk[v] === Infinity) {
        p.unreached += 1;
        continue;
      }
      if (occupants) p.occupants += respond > 0 ? occupants[v] / respond : 0;
      const d = deadlines[v];
      const W = q ? Math.max(0, q.exitTime(v, base) - base - walk[v]) : 0;
      p.wait = Math.max(p.wait, W);
      const need = required(walk[v] + W) + W;
      if (d !== -Infinity) p.share += respond * dist.cdf((q ? latestStart(q, v, d, walk[v], cue) : d) - cue);
      // Bound by the forecast's end, not the fire: the deadline is the end less
      // the walk (within two grid steps), and the required start would end the
      // walk after it.
      const atEnd = d !== -Infinity && d + walk[v] >= covered - 2 * step - 1e-6 * Math.max(1, horizon);
      if (d >= need) p.pass += 1;
      else if (atEnd && need + walk[v] > covered) {
        p.open += 1;
        p.need = Math.max(p.need, need + walk[v] + step);
      } else p.fail += 1;
      p.worst = Math.min(p.worst, d - base - W);
      if (base + walk[v] + W > 0) p.ratio = Math.min(p.ratio, (d + walk[v]) / (base + walk[v] + W));
    }
    return { base, start: required(0), per };
  };
  const queued = queuesFor(design);
  const { base, start, per } = judge(design, queued);
  const queuedData = data ? queuesFor(data) : null;
  const beside = data ? judge(data, queuedData) : null;

  const fmt = (x) => (x === -Infinity ? "no safe time" : `${Math.round(x)} s`);
  const worstText = (p) => `worst margin ${fmt(p.worst)}${Number.isFinite(p.ratio) ? `, least ASET / RSET ${p.ratio.toFixed(2)}` : ""}`;
  // What every place must allow, in words.
  const over = margin ? `${Math.round(margin)} s over it` : "";
  const bar = factor === 1 ? `can set off ${Math.round(start)} s after ignition and still get out` : `has ASET at least ${factor} times RSET${over && ` and at least ${over}`}`;
  const short = factor === 1 ? `cannot wait ${Math.round(start)} s` : `fall short of ${factor} times RSET${over && ` or of ${over}`}`;
  const judgedOf = (p) => p.places - p.unreached;
  const statusOf = (p) => (p.fail ? "fail" : p.open ? "cannot tell" : judgedOf(p) ? "pass" : "cannot tell");
  const shareOf = (p) => (judgedOf(p) ? p.share / judgedOf(p) : 0);
  const pctText = (p) => {
    const share = shareOf(p);
    const never = respond < 1 ? ` (${+(100 * (1 - respond)).toFixed(1)} % never respond)` : "";
    return `${p.open ? "at least " : ""}${(100 * share).toFixed(share > 0.999 && share < 1 ? 2 : 1)} % of occupants start in time${never}`;
  };
  const storeys = per
    .map((p, index) => {
      if (!p.places) return null;
      const judged = judgedOf(p);
      const status = statusOf(p);
      const longer = p.open ? `; ${p.open} more places stay safe as far as the forecast goes (${Math.round(horizon)} s): plan one of at least ${Math.ceil(p.need)} s` : "";
      const text =
        status === "pass"
          ? `every place ${bar}; ${worstText(p)}; ${pctText(p)}`
          : status === "fail"
            ? `${p.fail} of ${judged} places ${short} (${worstText(p)}); ${pctText(p)}${longer}`
            : judged
              ? `${p.pass} of ${judged} places pass; ${p.open} stay safe as far as the forecast goes (${Math.round(horizon)} s), which ends before they would be out: plan one of at least ${Math.ceil(p.need)} s`
              : "no place on this storey reaches an exit on the graph";
      const m = beside?.per[index];
      return {
        index,
        name: storeyNames[index] || `Storey ${index + 1}`,
        places: p.places,
        judged,
        unreached: p.unreached,
        pass: p.pass,
        fail: p.fail,
        open: p.open,
        needHorizon: p.open ? Math.ceil(p.need) : null,
        worstMargin: p.worst,
        worstRatio: p.ratio,
        share: shareOf(p),
        status,
        occupants: queues ? p.occupants : null,
        worstWait: queues ? p.wait : null,
        measured: m ? { status: statusOf(m), pass: m.pass, fail: m.fail, open: m.open, share: shareOf(m), worstMargin: m.worst, worstRatio: m.ratio, worstWait: queues ? m.wait : null } : null,
        text: p.unreached && judged ? `${text}; ${p.unreached} places reach no exit on the graph and are not judged` : text,
      };
    })
    .filter(Boolean);
  const worstOf = (list) => (list.includes("fail") ? "fail" : list.includes("cannot tell") ? "cannot tell" : "pass");
  const status = worstOf(storeys.map((s) => s.status));
  const needHorizon = storeys.reduce((m, st) => Math.max(m, st.needHorizon || 0), 0) || null;
  // The door that holds people longest (most people on a tie).
  const busiest = (q) => q?.doors.reduce((b, d) => (!b || d.wait > b.wait || (d.wait === b.wait && d.persons > b.persons) ? d : b), null) ?? null;
  // The share started by t seconds after the alarm under each distribution,
  // the non-responders kept out, for the card.
  const tMax = Math.max(pre99, data?.p99 ?? 0) * 1.25;
  const curve = (cdf) => Array.from({ length: 65 }, (_, i) => [(tMax * i) / 64, respond * cdf((tMax * i) / 64)]);
  return {
    status,
    start,
    detection,
    cue,
    base,
    factor,
    margin,
    storeys,
    distribution,
    needHorizon,
    noResponse: 1 - respond,
    measured: data
      ? {
          ...measured,
          median: weibullQuantile(0.5, measured),
          p99: data.p99,
          base: beside.base,
          start: beside.start,
          status: worstOf(storeys.map((s) => s.measured.status)),
          busiest: busiest(queuedData),
        }
      : null,
    queues: queued
      ? {
          dt: queued.dt,
          occupants: storeys.reduce((a, s) => a + s.occupants, 0),
          occupantArea: queues.occupantArea,
          doors: queued.doors.length,
          busiest: busiest(queued),
          defaultWidth: queues.defaultWidth ?? 0.813,
        }
      : null,
    curves: { tMax, design: curve(design.cdf), measured: data ? curve(data.cdf) : null, p99: pre99, measuredP99: data?.p99 ?? null },
  };
}

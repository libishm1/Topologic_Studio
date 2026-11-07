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
 * It is evidence for an alternative method (SBC 801-CR-2018 104.9), not a
 * substitute for the prescriptive check: one fire scenario, one walking
 * speed, no queues, and the room of fire origin judged like any other place.
 */

import { Heap, csr, edgeLengths } from "./route.js";
import { storeysOf } from "./codeCheck.js";

const KIND_DOOR = 2;
/** The 99th percentile of the standard normal distribution. */
export const Z99 = 2.3263478740408408;

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

/**
 * Every node's walking time to the nearest exit with no fire, over open
 * links (Infinity where no exit is reached): the least time any route out
 * can take, fire or not.
 */
export function walkToExit(graph, exits, { speed = 1.2, blocked = null } = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const w = edgeLengths(graph);
  const time = new Float64Array(n).fill(Infinity);
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
        heap.push(t, v);
      }
    }
  }
  return time;
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
 *   useWalls             respect links blocked by walls (default true)
 * @returns {{status, start, detection, cue, base, factor, margin, storeys, distribution, needHorizon}}
 *
 * With a factor, RSET takes the walk with no fire, tau(v), as its travel
 * term (as a hand calculation does) and ASET the deadline plus the same
 * walk, so ASET - RSET is the margin at the 99th percentile, L - cue - q99,
 * and ASET / RSET >= f is L >= f (cue + q99) + (f - 1) tau (Performance.lean,
 * ratio_iff): the margin must be at least f - 1 times RSET.
 */
export function performanceCheck(graph, deadlines, opts = {}) {
  const { levels = [], storeyNames = [], exits = [], horizon = Infinity, step = 0, speed = 1.2, detection = 0, alarm = 0, pre1, pre99, factor = 1, margin = 0, useWalls = true } = opts;
  const n = graph.nodes.length / 3;
  const kinds = graph.kinds || new Uint8Array(n);
  const storeyOf = storeysOf(graph, levels);
  const walk = walkToExit(graph, exits, { speed, blocked: useWalls ? graph.blocked : null });
  const distribution = lognormalFromPercentiles(pre1, pre99);
  const cue = detection + alarm;
  // RSET less the walk, and the latest start a place must allow: `start`
  // beside an exit, later with distance when the factor is above 1.
  const base = cue + pre99;
  const required = (tau) => Math.max(base + margin, factor * base + (factor - 1) * tau);
  const start = required(0);
  // The map's grid can end up to a step before the horizon.
  const covered = horizon - step;
  const floors = levels.length ? levels.length : 1;
  const per = Array.from({ length: floors }, () => ({ places: 0, pass: 0, fail: 0, open: 0, unreached: 0, worst: Infinity, ratio: Infinity, share: 0, need: 0 }));
  for (let v = 0; v < n; v += 1) {
    const s = storeyOf[v];
    if (s < 0 || kinds[v] === KIND_DOOR) continue;
    const p = per[s];
    p.places += 1;
    if (walk[v] === Infinity) {
      p.unreached += 1;
      continue;
    }
    const d = deadlines[v];
    const need = required(walk[v]);
    p.share += d === -Infinity ? 0 : lognormalCdf(d - cue, distribution);
    // Bound by the forecast's end, not the fire: the deadline is the end less
    // the walk (within two grid steps), and the required start would end the
    // walk after it.
    const atEnd = d !== -Infinity && d + walk[v] >= covered - 2 * step - 1e-6 * Math.max(1, horizon);
    if (d >= need) p.pass += 1;
    else if (atEnd && need + walk[v] > covered) {
      p.open += 1;
      p.need = Math.max(p.need, need + walk[v] + step);
    } else p.fail += 1;
    p.worst = Math.min(p.worst, d - base);
    if (base + walk[v] > 0) p.ratio = Math.min(p.ratio, (d + walk[v]) / (base + walk[v]));
  }
  const fmt = (x) => (x === -Infinity ? "no safe time" : `${Math.round(x)} s`);
  const worstText = (p) => `worst margin ${fmt(p.worst)}${Number.isFinite(p.ratio) ? `, least ASET / RSET ${p.ratio.toFixed(2)}` : ""}`;
  // What every place must allow, in words.
  const over = margin ? `${Math.round(margin)} s over it` : "";
  const bar = factor === 1 ? `can set off ${Math.round(start)} s after ignition and still get out` : `has ASET at least ${factor} times RSET${over && ` and at least ${over}`}`;
  const short = factor === 1 ? `cannot wait ${Math.round(start)} s` : `fall short of ${factor} times RSET${over && ` or of ${over}`}`;
  const storeys = per
    .map((p, index) => {
      if (!p.places) return null;
      const judged = p.places - p.unreached;
      const share = judged ? p.share / judged : 0;
      const status = p.fail ? "fail" : p.open ? "cannot tell" : judged ? "pass" : "cannot tell";
      const pct = `${p.open ? "at least " : ""}${(100 * share).toFixed(share > 0.999 && share < 1 ? 2 : 1)} % of occupants start in time`;
      const longer = p.open ? `; ${p.open} more places stay safe as far as the forecast goes (${Math.round(horizon)} s): plan one of at least ${Math.ceil(p.need)} s` : "";
      const text =
        status === "pass"
          ? `every place ${bar}; ${worstText(p)}; ${pct}`
          : status === "fail"
            ? `${p.fail} of ${judged} places ${short} (${worstText(p)}); ${pct}${longer}`
            : judged
              ? `${p.pass} of ${judged} places pass; ${p.open} stay safe as far as the forecast goes (${Math.round(horizon)} s), which ends before they would be out: plan one of at least ${Math.ceil(p.need)} s`
              : "no place on this storey reaches an exit on the graph";
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
        share,
        status,
        text: p.unreached && judged ? `${text}; ${p.unreached} places reach no exit on the graph and are not judged` : text,
      };
    })
    .filter(Boolean);
  const status = storeys.some((s) => s.status === "fail") ? "fail" : storeys.some((s) => s.status === "cannot tell") ? "cannot tell" : "pass";
  const needHorizon = storeys.reduce((m, st) => Math.max(m, st.needHorizon || 0), 0) || null;
  return { status, start, detection, cue, base, factor, margin, storeys, distribution, needHorizon };
}
